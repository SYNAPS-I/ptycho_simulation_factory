from typing import Optional, Tuple
import os

import torch
import h5py
import scipy.ndimage as ndi
import numpy as np
import ptychi.image_proc as ip
from ptychi.utils import rescale_probe, to_tensor


class RegexPatterns:
    float = r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"


def resize_probe(probe: np.ndarray, zoom_factor: float) -> np.ndarray:
    orig_shape = probe.shape
    probe = ndi.zoom(probe.reshape(-1, *orig_shape[-2:]), (1, zoom_factor, zoom_factor), order=2)
    probe = probe.reshape(*orig_shape[:-2], *probe.shape[-2:])
    return probe


def load_ptychodus_data(
    data_file: str,
    para_file: str,
    downsample_ratio: int = 1,
    subtract_position_mean: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patterns = h5py.File(data_file, "r")["dp"][...]
    f = h5py.File(para_file, "r")
    probe = f["probe"][...]
    if probe.ndim == 3:
        probe = probe[None]
    pixel_size_m = f["object"].attrs["pixel_width_m"]
    pos_y = f["probe_position_y_m"][...]
    pos_x = f["probe_position_x_m"][...]
    
    if downsample_ratio > 1:
        pixel_size_m *= downsample_ratio
        probe = resize_probe(probe, 1. / downsample_ratio)
        patterns = ip.central_crop(patterns, probe.shape[-2:])
    probe = to_tensor(probe)
    probe = rescale_probe(probe, patterns)
    pos_y = pos_y / pixel_size_m
    pos_x = pos_x / pixel_size_m
    if subtract_position_mean:
        pos_y -= pos_y.mean()
        pos_x -= pos_x.mean()
    return patterns, probe, pixel_size_m, pos_y, pos_x


def create_output_dir(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def prepare_postfix(postfix: Optional[str] = None) -> str:
    if postfix is None:
        postfix = ""
    else:
        if not postfix.startswith("_"):
            postfix = "_" + postfix
    return postfix

