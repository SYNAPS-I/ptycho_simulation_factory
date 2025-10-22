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
