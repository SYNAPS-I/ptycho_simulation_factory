import os
import argparse
import logging

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
        
        self.build()
        
    def build(self):
        self.build_output_dir()
        self.build_parallelism()
        self.build_object_dataset()
        self.build_probe_dataset()
        
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
        if "name_probability_map_file" in kwargs and kwargs["name_probability_map_file"] is not None:
            m = pd.read_csv(kwargs["name_probability_map_file"], header=None, index_col=None)
            m = {m[0][i]: float(m[1][i]) for i in range(len(m[0]))}
            del kwargs["name_probability_map_file"]
            kwargs["name_probability_map"] = m
        self.probe_dataset = probe_dataset_class(
            **kwargs
        )
        
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
    
    def get_random_probe(self):
        p = None
        if self.probe_dataset.name_probability_map is not None:
            p = np.array(list(self.probe_dataset.name_probability_map.values()))
        ind = np.random.choice(
            a=len(self.probe_dataset),
            size=1,
            p=p
        )[0]
        ind = int(ind)
        probe = self.probe_dataset[ind]
        return probe
    
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
            object, name = self.object_dataset[object_ind]
            if self.skip_existing and self.output_exists(name):
                continue
            
            probe = self.get_random_probe()
            
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
