from typing import Optional
import os
import glob
import logging

import torch
import numpy as np
from PIL import Image
from ptychi.propagate import AngularSpectrumPropagator, WavefieldPropagatorParameters

from helpers.image_helpers import create_complex_object, central_crop_or_pad

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, *args, **kwargs):
        pass
    
    def __len__(self):
        return 0
    

class ObjectDataset(Dataset):
    def __init__(
        self, 
        random_min_mag_range: tuple[float, float] = (0.5, 1.0),
        random_phase_range: tuple[float, float] = (0.0, 2.0),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.random_min_mag_range = random_min_mag_range
        self.random_phase_range = random_phase_range
    
    def __getitem__(self, index) -> tuple[np.ndarray, str]:
        """Get the object and its name at the given index.
        
        Parameters
        ----------
        index : int
            The index of the object to get.

        Returns
        -------
        tuple[np.ndarray, str]
            The object and its name.
        """
        pass


class ImageNetObjectDataset(ObjectDataset):
    def __init__(
        self, 
        root_dir: str, 
        random_min_mag_range: tuple[float, float] = (0.5, 1.0),
        random_phase_range: tuple[float, float] = (0.0, 2.0),
        *args, **kwargs
    ):
        super().__init__(
            random_min_mag_range=random_min_mag_range,
            random_phase_range=random_phase_range,
            *args, **kwargs
        )
        self.root_dir = root_dir
        self.index = self.create_index()
        
    def create_index(self, recursive: bool = True) -> list[str]:
        """Create a list of all object files in the root directory.
        """
        if os.path.exists(os.path.join(self.root_dir, "index.txt")):
            logger.info(f"Loading index for ImageNetObjectDataset from {os.path.join(self.root_dir, 'index.txt')}")
            with open(os.path.join(self.root_dir, "index.txt"), "r") as f:
                return [line.strip() for line in f.readlines()]
        else:
            logger.info(f"Creating index for ImageNetObjectDataset in {self.root_dir}")
            index = glob.glob(os.path.join(self.root_dir, "**", "*.JPEG"), recursive=recursive)
            with open(os.path.join(self.root_dir, "index.txt"), "w") as f:
                for item in index:
                    f.write(item + "\n")
            return index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index) -> tuple[np.ndarray, str]:
        """Get the object and its name at the given index.
        """
        obj = Image.open(self.index[index])
        obj = np.array(obj)
        if obj.ndim == 3:
            obj = obj.mean(axis=-1)
        obj = obj[None, ...]
        
        min_mag = np.random.uniform(*self.random_min_mag_range)
        phase_range = np.random.uniform(*self.random_phase_range)
        obj = create_complex_object(obj, min_mag, phase_range)
        return obj, os.path.splitext(os.path.basename(self.index[index]))[0]


class ProbeDataset(Dataset):
    def __init__(
        self, 
        name_probability_map: Optional[dict[str, float]] = None,
        output_shape: Optional[tuple[int, int]] = None, 
        remove_opr_modes: bool = False,
        random_defocus_range_m: Optional[tuple[float, float]] = None,
        energy_for_defocusing_kev: Optional[float] = None,
        pixel_size_for_defocusing_m: Optional[float] = None,
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        name_probability_map : Optional[dict[str, float]]
            A map from probe name to its probability weight. If None, all probes are
            equally weighted.
        output_shape : Optional[tuple[int, int]]
            The shape of the output probe. If the probe size is different from this value,
            it is cropped or padded.
        remove_opr_modes : bool
            If True, OPR modes are removed so that the size of the first dimension of the
            returned probe is 1.
        random_defocus_range_m : Optional[tuple[float, float]]
            The range of defocus values to sample from. If None, no defocus is added.
        energy_for_defocusing_kev : float
            The energy for defocusing in keV.
        pixel_size_for_defocusing_m : float
            The pixel size for defocusing in meters.
        """
        super().__init__(*args, **kwargs)
        if output_shape is not None:
            assert len(output_shape) == 2, "output_shape must be a tuple of length 2"
        self.output_shape = output_shape
        self.remove_opr_modes = remove_opr_modes
        self.random_defocus_range = random_defocus_range_m
        self.energy_for_defocusing_kev = energy_for_defocusing_kev
        self.pixel_size_for_defocusing_m = pixel_size_for_defocusing_m
        
        self.name_probability_map = name_probability_map
        if self.name_probability_map is not None:
            self.name_probability_map = self.normalize_name_probability_map(
                self.name_probability_map
            )

    def __getitem__(self, index) -> tuple[np.ndarray, str]:
        """Get the probe and its name at the given index.
        """
        raise NotImplementedError("Not implemented in base class.")
    
    def normalize_name_probability_map(self, m: dict[str, float]) -> dict[str, float]:
        """Normalize the name probability weight map so that the sum of the weights is 1.
        """
        s = sum(m.values())
        return {
            name: weight / s
            for name, weight in m.items()
        }
    
    def add_random_defocus(self, probe: np.ndarray) -> np.ndarray:
        """Add random defocus to the probe.
        """
        if self.random_defocus_range is None:
            return probe
        if self.energy_for_defocusing_kev is None:
            raise ValueError("energy_for_defocusing_kev must be set")
        if self.pixel_size_for_defocusing_m is None:
            raise ValueError("pixel_size_for_defocusing_m must be set")
        
        defocus_m = np.random.uniform(*self.random_defocus_range)
        propagator = AngularSpectrumPropagator(
            parameters=WavefieldPropagatorParameters.create_simple(
                wavelength_m=1.239e-9 / self.energy_for_defocusing_kev,
                width_px=probe.shape[2],
                height_px=probe.shape[3],
                pixel_width_m=self.pixel_size_for_defocusing_m,
                pixel_height_m=self.pixel_size_for_defocusing_m,
                propagation_distance_m=defocus_m,
            )
        )
        probe_tensor = torch.tensor(probe, device=propagator._transfer_function_real.device)
        probe_defocused = propagator.propagate_forward(probe_tensor).cpu().numpy()
        return probe_defocused
        

class NpyProbeDataset(ProbeDataset):
    def __init__(
        self, 
        root_dir: str, 
        output_shape: Optional[tuple[int, int]] = None, 
        remove_opr_modes: bool = False,
        recursive: bool = True, 
        *args, 
        **kwargs
    ):
        super().__init__(
            output_shape=output_shape,
            remove_opr_modes=remove_opr_modes,
            *args, **kwargs
        )
        self.root_dir = root_dir
        self.index = self.create_index(recursive)
        
    def create_index(self, recursive: bool = True) -> list[str]:
        """Create a list of all probe files in the root directory.
        """
        if self.name_probability_map is not None:
            logger.info("Using name-probability map for indexing in NpyProbeDataset.")
            flist = list(self.name_probability_map.keys())
        else:
            flist = glob.glob(os.path.join(self.root_dir, "*.npy"), recursive=recursive)
            flist = [os.path.basename(f) for f in flist]
        return flist
    
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, index) -> tuple[np.ndarray, str]:
        """Get the probe and its name at the given index.
        
        Parameters
        ----------
        index : int
            The index of the probe to get.

        Returns
        -------
        tuple[np.ndarray, str]
            A (n_opr_modes, n_modes, h, w) array of complex-valued probe.
        """
        p = np.load(os.path.join(self.root_dir, self.index[index]))
        if p.ndim == 3:
            p = p[None, ...]
        elif p.ndim == 2:
            p = p[None, None, ...]
        if self.remove_opr_modes:
            p = p[0:1, ...]
        if self.output_shape is not None:
            p = central_crop_or_pad(p, self.output_shape)
        p = self.add_random_defocus(p)
        return p
