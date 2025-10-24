from typing import Literal

import numpy as np


def create_complex_object(img, mag_min=0.95, phase_range=0.5):
    """Create a complex-valued field from a real-valued image.
    The differently scaled image is used as the magnitude and phase
    pf the complex object.

    Parameters
    ----------
    img : np.ndarray
        The real-valued image.
    mag_min : float, optional
        The minimum magnitude of the complex object. Generated
        magnitude values will be in the range [mag_min, 1].
    phase_range : float, optional
        The range of the phase of the complex object. This is
        the full range of the phase. Generated phase values
        will be in the range [-phase_range / 2, phase_range / 2].

    Returns
    -------
    np.ndarray
        The complex-valued object.
    """
    mag = img.max() - img
    mag = mag / mag.max() * (1 - mag_min) + mag_min
    phase = img / img.max() * phase_range - phase_range / 2
    return mag * np.exp(1j * phase)


def gaussian_2d(shape, sigma):
    probe = np.zeros(shape)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x = x - (shape[1] - 1) / 2
    y = y - (shape[0] - 1) / 2
    probe = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    probe = probe / probe.max()
    return probe


def central_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """
    Crop the center of an image.
    
    Parameters
    ----------
    img : np.ndarray
        A (..., H, W) array giving the input image(s).
    crop_size : tuple[int, int]
        crop size.
    
    Returns
    -------
    np.ndarray
        The image cropped to the target size.
    """
    return img[
        ..., 
        img.shape[-2] // 2 - crop_size[0] // 2 : img.shape[-2] // 2 - crop_size[0] // 2 + crop_size[0], 
        img.shape[-1] // 2 - crop_size[1] // 2 : img.shape[-1] // 2 - crop_size[1] // 2 + crop_size[1]
    ]


def central_pad(
    img: np.ndarray, 
    target_size: tuple[int, int], 
    mode: Literal["constant", "reflect", "replicate", "circular"] = "constant", 
    value: float = 0.0
) -> np.ndarray:
    """
    Pad the center of an image.
    
    Parameters
    ----------
    img : np.ndarray
        A (..., H, W) array giving the input image(s).
    target_size : tuple[int, int]
        target size.
    
    Returns
    -------
    np.ndarray
        The image padded to the target size.
    """
    pad_size = [(0, 0)] * (img.ndim - 2) + [
        ((target_size[0] - img.shape[-2]) // 2, target_size[-2] - (target_size[0] - img.shape[-2]) // 2 - img.shape[-2]),
        ((target_size[1] - img.shape[-1]) // 2, target_size[-1] - (target_size[1] - img.shape[-1]) // 2 - img.shape[-1])
    ]
    return np.pad(img, pad_size, mode=mode, constant_values=value)


def central_crop_or_pad(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Crop or pad the center of an image to the target size.
    
    Parameters
    ----------
    img : np.ndarray
        A (..., H, W) array giving the input image(s).
    target_size : tuple[int, int]
        target size.
    
    Returns
    -------
    np.ndarray
        The image cropped or padded to the target size.
    """
    for i in range(2):
        target_size_current_dim = [img.shape[-2]] * (i == 1) + [target_size[i]] + [img.shape[-1]] * (i == 0)
        if img.shape[-2 + i] > target_size[-2 + i]:
            img = central_crop(img, target_size_current_dim)
        elif img.shape[-2 + i] < target_size[-2 + i]:
            img = central_pad(img, target_size_current_dim)
    return img


def add_poisson_noise(intensity_images: np.ndarray, total_photon_count: int) -> np.ndarray:
    """
    Add Poisson noise to intensity images. The original scaling of the intensity
    is preserved in the output.
    
    Parameters
    ----------
    intensity_images : np.ndarray
        The (..., h, w) intensity images to which Poisson noise is added.
    total_photon_count : int
        The total photon count of the image.
    
    Returns
    -------
    np.ndarray
        The image with Poisson noise added.
    """
    powers = np.sum(np.abs(intensity_images), axis=(-2, -1))
    scaling_factor = total_photon_count / powers
    intensity_images = intensity_images * scaling_factor[..., None, None]
    intensity_images = np.random.poisson(intensity_images)
    intensity_images = intensity_images / scaling_factor[..., None, None]
    return intensity_images
