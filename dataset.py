import os
import glob

import numpy as np
from PIL import Image

from helpers.image_helpers import create_complex_object


class Dataset:
    def __init__(self, *args, **kwargs):
        pass
    
    def __len__(self):
        return 0
    

class ObjectDataset(Dataset):
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
        self, root_dir: str, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root_dir = root_dir


class ProbeDataset(Dataset):
    pass


class NpyProbeDataset(ProbeDataset):
    def __init__(
        self, root_dir: str, recursive: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root_dir = root_dir
        self.index = self.create_index(recursive)
        
    def create_index(self, recursive: bool = True) -> list[str]:
        """Create a list of all probe files in the root directory.
        """
        return glob.glob(os.path.join(self.root_dir, "*.npy"), recursive=recursive)
    
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
        p = np.load(self.index[index])
        if p.ndim == 3:
            p = p[None, ...]
        elif p.ndim == 2:
            p = p[None, None, ...]
        return p
