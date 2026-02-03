import os
import argparse
import logging
from typing import Optional

import torch
import numpy as np
import tqdm
import yaml
import pandas as pd

from parallel import MultiprocessMixin
from simulator import PtychographySimulator
import grid_generator
import dataset
from helpers.misc import get_config_without_classname

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BatchSimulationTask(MultiprocessMixin):
    
    def __init__(
        self,
        config: dict,
        skip_existing: bool = False,
    ) -> None:
        self.config = config
        self.skip_existing = skip_existing
        
        self.object_dataset = None
        self.probe_dataset = None
        self._probe_sampler_state = None
        
        self.build()
        
    def build(self):
        self.build_output_dir()
        self.build_parallelism()
        self.build_object_dataset()
        self.build_probe_dataset()

    @property
    def sample_probe_randomly(self) -> bool:
        return self.config.get("simulator", {}).get("sample_probe_randomly", True)
        
    def build_output_dir(self):
        os.makedirs(self.config["task"]["output_root"], exist_ok=True)
        
    def build_object_dataset(self):
        object_dataset_class = getattr(dataset, self.config["object_dataset"]["class_name"])
        self.object_dataset = object_dataset_class(
            **get_config_without_classname(self.config["object_dataset"])
        )
        
    def build_probe_dataset(self):
        probe_dataset_class = getattr(dataset, self.config["probe_dataset"]["class_name"])
        kwargs = get_config_without_classname(self.config["probe_dataset"])
        if "probe_defocus_list" in kwargs and kwargs["probe_defocus_list"] is not None:
            with open(kwargs["probe_defocus_list"], "r") as f:
                values = [line.strip() for line in f.readlines()]
            values = [line for line in values if line]
            kwargs["probe_defocus_list"] = [float(v) for v in values]
        if (
            "probe_file_list" in kwargs
            and kwargs["probe_file_list"] is not None
            and "name_probability_map_file" in kwargs
            and kwargs["name_probability_map_file"] is not None
        ):
            raise ValueError("probe_file_list and name_probability_map_file cannot both be set")
        if "name_probability_map_file" in kwargs and kwargs["name_probability_map_file"] is not None:
            m = pd.read_csv(kwargs["name_probability_map_file"], header=None, index_col=None)
            m = {m[0][i]: float(m[1][i]) for i in range(len(m[0]))}
            del kwargs["name_probability_map_file"]
            kwargs["name_probability_map"] = m
        self.probe_dataset = probe_dataset_class(
            **kwargs
        )
        self.build_probe_sampler()

    def build_probe_sampler(self):
        if self.sample_probe_randomly:
            self._probe_sampler_state = None
            return
        n_probes = len(self.probe_dataset)
        if n_probes == 0:
            raise ValueError("probe_dataset is empty")
        if self.probe_dataset.probe_file_list is not None:
            sequence = list(range(n_probes))
            self._probe_sampler_state = {
                "sequence": sequence,
                "next_index": 0,
            }
            return
        if self.probe_dataset.name_probability_map is not None:
            weights = []
            for item in self.probe_dataset.index:
                name = os.path.basename(item)
                weights.append(float(self.probe_dataset.name_probability_map.get(name, 0.0)))
            weights = np.array(weights, dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(n_probes, dtype=float)
            repeats = n_probes * weights
            clipped = (repeats > 0) & (repeats < 1)
            repeats[clipped] = 1.0
            repeats = np.rint(repeats).astype(int)
            sequence = []
            for ind, count in enumerate(repeats):
                if count > 0:
                    sequence.extend([ind] * int(count))
        else:
            sequence = list(range(n_probes))
        if len(sequence) == 0:
            sequence = list(range(n_probes))
        self._probe_sampler_state = {
            "sequence": sequence,
            "next_index": 0,
        }

    def get_deterministic_probe_index(self) -> int:
        state = self._probe_sampler_state
        if state is None:
            return 0
        sequence = state["sequence"]
        ind = sequence[state["next_index"]]
        state["next_index"] = (state["next_index"] + 1) % len(sequence)
        return int(ind)
        
    def build_parallelism(self):
        launcher = self.detect_launcher()
        if launcher is not None:
            self.init_process_group()
            torch.set_default_device(f"cuda:{self.rank % torch.cuda.device_count()}")
            logger.info(f"Initiated rank ID {self.rank} on device {torch.get_default_device()}")
        
    def create_position_generator(self):
        position_generator_class = getattr(grid_generator, self.config["position_generator"]["class_name"])
        position_generator = position_generator_class(
            **get_config_without_classname(self.config["position_generator"])
        )
        return position_generator
    
    def get_probe(self, master_index: Optional[int] = None):
        if master_index is not None and self.probe_dataset.probe_file_list is not None:
            if len(self.probe_dataset) == 0:
                raise ValueError("probe_dataset is empty")
            ind = int(master_index) % len(self.probe_dataset)
            probe_item = self.probe_dataset[ind]
            if isinstance(probe_item, tuple):
                if len(probe_item) == 3:
                    probe, probe_file, probe_defocus_m = probe_item
                else:
                    probe, probe_file = probe_item
                    probe_defocus_m = None
            else:
                probe = probe_item
                probe_file = None
                probe_defocus_m = None
            return probe, probe_file, probe_defocus_m
        if self.probe_dataset.probe_file_list is not None:
            if self._probe_sampler_state is None:
                self.build_probe_sampler()
            ind = self.get_deterministic_probe_index()
            probe_item = self.probe_dataset[ind]
            if isinstance(probe_item, tuple):
                if len(probe_item) == 3:
                    probe, probe_file, probe_defocus_m = probe_item
                else:
                    probe, probe_file = probe_item
                    probe_defocus_m = None
            else:
                probe = probe_item
                probe_file = None
                probe_defocus_m = None
            return probe, probe_file, probe_defocus_m
        if self.sample_probe_randomly:
            p = None
            if self.probe_dataset.name_probability_map is not None:
                p = np.array(
                    [
                        float(self.probe_dataset.name_probability_map.get(os.path.basename(item), 0.0))
                        for item in self.probe_dataset.index
                    ],
                    dtype=float,
                )
                if p.sum() <= 0:
                    p = None
            ind = np.random.choice(
                a=len(self.probe_dataset),
                size=1,
                p=p
            )[0]
            ind = int(ind)
        else:
            ind = self.get_deterministic_probe_index()
        probe_item = self.probe_dataset[ind]
        if isinstance(probe_item, tuple):
            if len(probe_item) == 3:
                probe, probe_file, probe_defocus_m = probe_item
            else:
                probe, probe_file = probe_item
                probe_defocus_m = None
        else:
            probe = probe_item
            probe_file = None
            probe_defocus_m = None
        return probe, probe_file, probe_defocus_m
    
    def output_exists(self, name: str) -> bool:
        return (
            os.path.exists(self.get_output_dp_path(name)) and 
            os.path.exists(self.get_output_para_path(name))
        )
    
    def get_output_dp_path(self, name: str) -> str:
        return os.path.join(self.config["task"]["output_root"], name + "_dp.hdf5")
    
    def get_output_para_path(self, name: str) -> str:
        return os.path.join(self.config["task"]["output_root"], name + "_para.hdf5")
    
    def run(self):
        indices = np.arange(len(self.object_dataset))[self.rank::self.n_ranks]
        pbar = tqdm.tqdm(indices, desc=f"Rank {self.rank}")
        
        for object_ind in pbar:
            object_item = self.object_dataset[object_ind]
            if isinstance(object_item, tuple) and len(object_item) >= 3:
                object, name, object_file = object_item
            else:
                object, name = object_item
                object_file = None
            if self.skip_existing and self.output_exists(name):
                continue
            
            probe, probe_file, probe_defocus_m = self.get_probe(master_index=object_ind)
            
            position_generator = self.create_position_generator()
            positions = position_generator.generate_positions(
                object.shape[-2:], probe.shape[-2:]
            )
            if len(positions) < 4:
                logger.warning(f"Skipping {name} because only {len(positions)} positions generated")
                continue
            
            if probe.shape[0] == 1:
                opr_weights = np.ones([len(positions), 1])
            else:
                opr_weights = np.concatenate(
                    [
                        np.ones([len(positions), 1]),
                        np.zeros([len(positions), probe.shape[0] - 1]),
                    ],
                    axis=1
                )
            
            # Get compression config if specified
            compression = self.config["simulator"].get("compression")
            
            sim = PtychographySimulator(
                object_=object,
                probe=probe,
                positions=positions,
                opr_weights=opr_weights,
                pixel_size=self.config["simulator"]["pixel_size_m"],
                wavelength_m=1.24e-9 / self.config["simulator"]["energy_kev"],
                output_dir=os.path.join(self.config["task"]["output_root"]),
                output_file_prefix=name + "_",
                probe_file=probe_file,
                object_file=object_file,
                probe_defocus_m=probe_defocus_m,
                add_poisson_noise=self.config["simulator"]["add_poisson_noise"],
                total_photon_count=self.config["simulator"]["total_photon_count"],
                verbose=False,
                compression=compression,
            )
            sim.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument(
        "--skip-existing", 
        action="store_true", 
        help="Skip simulation if output files already exist."
    )
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    task = BatchSimulationTask(config, skip_existing=args.skip_existing)
    task.run()
