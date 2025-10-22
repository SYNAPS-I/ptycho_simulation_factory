import os
import argparse

import torch
import numpy as np
import tqdm
import yaml

from parallel import MultiprocessMixin
from simulator import PtychographySimulator
import grid_generator
import dataset
from helpers.misc import get_config_without_classname


class BatchSimulationTask(MultiprocessMixin):
    
    def __init__(
        self,
        config: dict
    ) -> None:
        self.config = config
        
        self.object_dataset = None
        self.probe_dataset = None
        
        self.build()
        
    def build(self):
        self.build_output_dir()
        self.build_parallelism()
        self.build_object_dataset()
        self.build_probe_dataset()
        self.build_position_generator()
        
    def build_output_dir(self):
        os.makedirs(self.config["task"]["output_root"], exist_ok=True)
        
    def build_object_dataset(self):
        object_dataset_class = getattr(dataset, self.config["object_dataset"]["class_name"])
        self.object_dataset = object_dataset_class(
            **get_config_without_classname(self.config["object_dataset"])
        )
        
    def build_probe_dataset(self):
        probe_dataset_class = getattr(dataset, self.config["probe_dataset"]["class_name"])
        self.probe_dataset = probe_dataset_class(
            **get_config_without_classname(self.config["probe_dataset"])
        )
        
    def build_parallelism(self):
        launcher = self.detect_launcher()
        if launcher is not None:
            self.init_process_group()
            torch.set_default_device(f"cuda:{self.rank % torch.cuda.device_count()}")
        
    def create_position_generator(self):
        position_generator_class = getattr(grid_generator, self.config["position_generator"]["class_name"])
        position_generator = position_generator_class(
            **get_config_without_classname(self.config["position_generator"])
        )
        return position_generator
    
    def get_random_probe(self):
        ind = np.random.randint(0, len(self.probe_dataset))
        probe = self.probe_dataset[ind]
        return probe
    
    def run(self):
        indices = np.arange(len(self.object_dataset))[self.rank::self.n_ranks]
        pbar = tqdm.tqdm(indices, disable=self.n_ranks > 1)
        
        for object_ind in pbar:
            object, name = self.object_dataset[object_ind]
            probe = self.get_random_probe()
            
            position_generator = self.create_position_generator()
            positions = position_generator.generate_positions(
                object.shape[-2:], probe.shape[-2:]
            )
            
            if probe.shape[0] == 1:
                opr_weights = np.ones(len(positions), 1)
            else:
                opr_weights = np.concatenate(
                    [
                        np.ones(len(positions), 1),
                        np.zeros(len(positions), probe.shape[0] - 1),
                    ],
                    axis=1
                )
            
            sim = PtychographySimulator(
                object=object,
                probe=probe,
                positions=positions,
                opr_weights=opr_weights,
                pixel_size_m=self.config["simulator"]["pixel_size_m"],
                wavelength_m=1.24e-9 / self.config["simulator"]["energy_kev"],
                output_dir=os.path.join(self.config["task"]["output_root"]),
                output_file_prefix=name + "_",
            )
            sim.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    task = BatchSimulationTask(config)
    task.run()
