import os

import numpy as np
import skimage
import h5py
import scipy.ndimage as ndi
import tqdm
import tifffile
import torch

import ptychi.api as api
from ptychi.utils import generate_initial_opr_mode_weights
import ptychi.data_structures.object
import ptychi.data_structures.probe
import ptychi.data_structures.opr_mode_weights
import ptychi.data_structures.probe_positions
import ptychi.data_structures.parameter_group
import ptychi.forward_models as fm
import ptychi.image_proc as ip
import helpers.io_helpers as io_helpers


class PtychographySimulator:
    
    def __init__(
        self, 
        object_, 
        probe, 
        positions, 
        opr_weights, 
        pixel_size, 
        wavelength_m, 
        slice_spacings_m=None, 
        batch_size=100, 
        output_dir="data",
        probe_to_be_in_data_file=None
    ):
        self.object_ = object_
        self.probe = probe
        self.positions = positions
        self.opr_weights = opr_weights
        self.pixel_size = pixel_size
        self.wavelength_m = wavelength_m
        self.slice_spacings_m = slice_spacings_m
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.probe_to_be_in_data_file = probe_to_be_in_data_file

    def build_forward_model(self):
        options = api.base.ObjectOptions(
            slice_spacings_m=self.slice_spacings_m,
            pixel_size_m=self.pixel_size,
        )
        self.object_ = ptychi.data_structures.object.PlanarObject(data=self.object_, options=options)
        
        options = api.base.ProbeOptions()
        self.probe = ptychi.data_structures.probe.Probe(data=self.probe, options=options)
        
        options = api.base.ProbePositionOptions()
        self.positions = ptychi.data_structures.probe_positions.ProbePositions(data=self.positions, options=options)
        
        options = api.base.OPRModeWeightsOptions()
        self.opr_weights = ptychi.data_structures.opr_mode_weights.OPRModeWeights(data=self.opr_weights, options=options)
        
        params_group = ptychi.data_structures.parameter_group.PtychographyParameterGroup(
            object=self.object_,
            probe=self.probe,
            probe_positions=self.positions,
            opr_mode_weights=self.opr_weights,
        )
        
        self.forward_model = fm.PlanarPtychographyForwardModel(
            parameter_group=params_group,
            retain_intermediates=False,
            wavelength_m=self.wavelength_m,
        )
        
    def build_output_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.f_dp = h5py.File(os.path.join(self.output_dir, "ptychodus_dp.hdf5"), "w")
        self.f_para = h5py.File(os.path.join(self.output_dir, "ptychodus_para.hdf5"), "w")
        
        self.f_dp.create_dataset("dp", shape=(self.positions.shape[0], *self.probe.shape[-2:]), dtype="float32")
        
        if self.probe_to_be_in_data_file is None:
            self.f_para.create_dataset("probe", data=self.probe.data.detach().cpu().numpy())
        else:
            self.f_para.create_dataset("probe", data=self.probe_to_be_in_data_file)
        self.f_para.create_dataset("probe_position_indexes", data=np.arange(self.positions.shape[0]).astype(int))
        self.f_para.create_dataset("probe_position_x_m", data=self.positions.data[:, 1].detach().cpu().numpy() * self.pixel_size)
        self.f_para.create_dataset("probe_position_y_m", data=self.positions.data[:, 0].detach().cpu().numpy() * self.pixel_size)
        self.f_para.create_dataset("object", data=self.object_.data.detach().cpu().numpy())
        self.f_para["object"].attrs["pixel_height_m"] = self.pixel_size
        self.f_para["object"].attrs["pixel_width_m"] = self.pixel_size
        self.f_para["object"].attrs["center_x_m"] = 0
        self.f_para["object"].attrs["center_y_m"] = 0
        
        tifffile.imwrite(os.path.join(self.output_dir, "true_object_phase.tiff"), np.angle(self.object_.data.detach().cpu().numpy()))
        tifffile.imwrite(os.path.join(self.output_dir, "true_object_mag.tiff"), np.abs(self.object_.data.detach().cpu().numpy()))
        
    def wrap_up(self):
        self.f_dp.close()
        self.f_para.close()
        
    def run(self):
        self.build_forward_model()
        self.build_output_files()

        i_pos = 0
        n_pos = self.positions.shape[0]
        pbar = tqdm.tqdm(total=n_pos, desc="Simulating")
        while i_pos < n_pos:
            i_end = min(i_pos + self.batch_size, n_pos)
            indices = torch.arange(i_pos, i_end).long()
            intensities = self.forward_model(indices=indices).detach().cpu().numpy()
            intensities = np.fft.fftshift(intensities, axes=(-2, -1))
            self.f_dp["dp"][i_pos:i_end] = intensities
            pbar.update(i_end - i_pos)
            i_pos = i_end
        self.wrap_up()
        

def create_complex_object(img, mag_min=0.95, phase_range=0.5):
    mag = img.max() - img
    mag = mag / mag.max() * (1 - mag_min) + mag_min
    phase = img / img.max() * (phase_range * 2) - phase_range
    return mag * np.exp(1j * phase)


def create_positions(object_shape, probe_lateral_shape, ny=None, nx=None, spacing_y=None, spacing_x=None):
    """Create probe positions.
    
    Parameters
    ----------
    object_shape : tuple of int
        Lateral shape of the object.
    probe_lateral_shape : tuple of int
        Lateral shape of the probe. This is used to determine the safety margin,
        so that the probe does not reach outside the object.
    ny : int
        Number of positions in the y direction.
    nx : int
        Number of positions in the x direction.
    spacing_y : float
        Spacing in the y direction.
    spacing_x : float
        Spacing in the x direction.
    """
    margin = [probe_lateral_shape[i] // 2 for i in range(len(probe_lateral_shape))]
    if ny is not None:
        assert spacing_y is None
        y = np.linspace(margin[0], object_shape[0] - margin[0] - 2, ny)
    else:
        assert spacing_y is not None
        y = np.arange(margin[0], object_shape[0] - margin[0] - 1, spacing_y)
    if nx is not None:
        assert spacing_x is None
        x = np.linspace(margin[1], object_shape[1] - margin[1] - 2, nx)
    else:
        assert spacing_x is not None
        x = np.arange(margin[1], object_shape[1] - margin[1] - 1, spacing_x)
    y, x = np.meshgrid(y, x)
    positions = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)
    positions = positions - positions.mean(0)
    return positions


def gaussian_2d(shape, sigma):
    probe = np.zeros(shape)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x = x - (shape[1] - 1) / 2
    y = y - (shape[0] - 1) / 2
    probe = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    probe = probe / probe.max()
    return probe


if __name__ == "__main__":
    if False:
        # object_ = skimage.data.cells3d().sum(1)[20:21]
        # object_ = ndi.zoom(object_, [1, 2, 2])
        object_mag = tifffile.imread("data/charcoal_10x10/true_object_mag.tiff")[None]
        object_phase = tifffile.imread("data/charcoal_10x10/true_object_phase.tiff")[None]
        object_ = object_mag * np.exp(1j * object_phase)
        probe = gaussian_2d([128, 128], 20)[None, None].astype(np.complex64)
        # probe = h5py.File("data/tungsten/metadata_250_truePos.hdf5", "r")["probe"][()][0:1][None, ...]
        positions = create_positions(object_.shape[1:], probe.shape[-2:], 10, 10)
        opr_weights = generate_initial_opr_mode_weights(len(positions), 1)
        pixel_size_m = 1e-8
        wavelength_m = 1.24e-10
        
        simulator = PtychographySimulator(
            object_, probe, positions, opr_weights, pixel_size_m, wavelength_m, batch_size=100,
            output_dir="data/charcoal_10x10_guessedProbe",
        )
        
    if True:
        slice_1 = skimage.data.cells3d().sum(1)[20:21]
        slice_1 = ndi.zoom(slice_1, [1, 2, 2])
        slice_2 = np.zeros_like(slice_1)
        rod_length = 60
        rod_width = 8
        xx, yy = np.meshgrid(np.arange(0, slice_2.shape[-1], rod_length + 60), np.arange(0, slice_2.shape[-2], rod_width + 60))
        for y, x in zip(yy.reshape(-1), xx.reshape(-1)):
            slice_2[0, max(0, y - rod_width // 2):min(slice_2.shape[-2], y + rod_width // 2), max(0, x - rod_length // 2):min(slice_2.shape[-1], x + rod_length // 2)] += 1
        xx, yy = np.meshgrid(np.arange((rod_length + 60) // 2, slice_2.shape[-1], rod_length + 60), np.arange((rod_width + 60) // 2, slice_2.shape[-2], rod_width + 60))
        for y, x in zip(yy.reshape(-1), xx.reshape(-1)):
            slice_2[0, max(0, y - rod_width // 2):min(slice_2.shape[-2], y + rod_width // 2), max(0, x - rod_length // 2):min(slice_2.shape[-1], x + rod_length // 2)] += 1
        slice_2 = ndi.rotate(slice_2, 30, axes=(1, 2), reshape=False, order=3)
        slice_2 = ndi.zoom(slice_2, [1, 1.3, 1.3], order=1)
        margin = ((slice_2.shape[-2] - slice_1.shape[-2]) // 2, (slice_2.shape[-1] - slice_1.shape[-1]) // 2)
        slice_2 = slice_2[:, margin[0]:margin[0] + slice_1.shape[-2], margin[1]:margin[1] + slice_1.shape[-1]]
        slice_1 = create_complex_object(slice_1, mag_min=0.98, phase_range=0.05)
        slice_2 = create_complex_object(slice_2, mag_min=0.98, phase_range=0.05)
        object_ = np.concatenate([slice_1, slice_2], axis=0)
        
        _, probe, _, _, _ = io_helpers.load_ptychodus_data(
            "data/bnp_23c1_Volker1_fly048/ptychodus_dp.hdf5",
            "data/bnp_23c1_Volker1_fly048/ptychodus_para.hdf5",
            downsample_ratio=2
        )
        
        positions = create_positions(object_.shape[1:], probe.shape[-2:], 40, 40)
        
        opr_weights = generate_initial_opr_mode_weights(len(positions), 1)
        pixel_size_m = 1e-8
        wavelength_m = 1.24e-10
        
        simulator = PtychographySimulator(
            object_, probe, positions, opr_weights, pixel_size_m, wavelength_m, batch_size=100,
            slice_spacings_m=[1e-6],
            output_dir="data/multislice_sim",
        )
    
    if False:
        object_ = tifffile.imread("traditional/outputs/bnp_dpsize_128_lsqml/recon_phase_unwrapped_20250326_095346.tiff")[None]
        object_ = create_complex_object(object_)
        
        patterns, guessed_probe, pixel_size_m, pos_y, pos_x = io_helpers.load_ptychodus_data(
            "data/bnp_23c1_Volker1_fly048/ptychodus_dp.hdf5",
            "data/bnp_23c1_Volker1_fly048/ptychodus_para.hdf5",
            downsample_ratio=2
        )
        probe = np.load("data/bnp_23c1_Volker1_fly048/reconstructed_probe.npy")
        probe = ndi.zoom(probe, [1, 1, 0.5, 0.5])
        
        mask = (pos_x > -512 + probe.shape[-1] // 2) \
            & (pos_x < 512 - probe.shape[-1] // 2) \
            & (pos_y > -512 + probe.shape[-2] // 2) \
            & (pos_y < 512 - probe.shape[-2] // 2)
        pos_x = pos_x[mask]
        pos_y = pos_y[mask]
        positions = np.stack([pos_y, pos_x], axis=1)
        print(positions.shape)
        opr_weights = generate_initial_opr_mode_weights(len(positions), 1)
        wavelength_m = 1.24e-10
        
        guessed_probe = ndi.zoom(guessed_probe, [1, 1, 1.05, 1.05])
        guessed_probe = ip.central_crop(guessed_probe, [128, 128])
            
        simulator = PtychographySimulator(
            object_, probe, positions, opr_weights, pixel_size_m, wavelength_m, batch_size=100,
            output_dir="data/bnpSim_guessedProbe",
            probe_to_be_in_data_file=guessed_probe,
        )
    simulator.run()
        
