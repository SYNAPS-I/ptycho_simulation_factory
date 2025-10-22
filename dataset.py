from helpers.image_helpers import create_complex_object

import numpy as np
from PIL import Image


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

