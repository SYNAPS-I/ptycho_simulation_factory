import os
from typing import Optional

import numpy as np
import h5py
import hdf5plugin  # Blosc compression support (EPICS area detector style)
import tqdm
import tifffile
import torch

import ptychi.api as api
import ptychi.data_structures.object
import ptychi.data_structures.probe
import ptychi.data_structures.opr_mode_weights
import ptychi.data_structures.probe_positions
import ptychi.data_structures.parameter_group
import ptychi.forward_models as fm

from helpers.image_helpers import add_poisson_noise


class PtychographySimulator:
    
    def __init__(
        self, 
        object_: np.ndarray | torch.Tensor,
        probe: np.ndarray | torch.Tensor, 
        positions: np.ndarray | torch.Tensor, 
        opr_weights: np.ndarray | torch.Tensor, 
        pixel_size: float, 
        wavelength_m: float, 
        slice_spacings_m: Optional[list[float]] = None, 
        batch_size: int = 100, 
        output_dir: str = "data",
        output_file_prefix: str = "ptychodus_",
        probe_to_be_in_data_file: Optional[np.ndarray | torch.Tensor] = None,
        probe_file: Optional[str] = None,
        object_file: Optional[str] = None,
        probe_defocus_m: Optional[float] = None,
        object_min_mag: Optional[float] = None,
        object_phase_range: Optional[float] = None,
        add_poisson_noise: bool = False,
        total_photon_count: Optional[int] = None,
        verbose: bool = True,
        compression: Optional[str | dict] = None,
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
        self.output_file_prefix = output_file_prefix
        self.probe_to_be_in_data_file = probe_to_be_in_data_file
        self.probe_file = probe_file
        self.object_file = object_file
        self.probe_defocus_m = probe_defocus_m
        self.object_min_mag = object_min_mag
        self.object_phase_range = object_phase_range
        self.add_poisson_noise = add_poisson_noise
        self.total_photon_count = total_photon_count
        self.verbose = verbose
        self.compression = compression
        
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
        
    def build_output_files(self, save_object_tiffs: bool = True):
        os.makedirs(self.output_dir, exist_ok=True)
        self.f_dp = h5py.File(os.path.join(self.output_dir, f"{self.output_file_prefix}dp.hdf5"), "w")
        self.f_para = h5py.File(os.path.join(self.output_dir, f"{self.output_file_prefix}para.hdf5"), "w")
        
        # Determine compression settings from config
        # Optimal chunking for diffraction patterns: chunk by position (one pattern per chunk)
        # This allows better compression as each pattern is compressed independently
        n_positions = self.positions.shape[0]
        probe_shape = self.probe.shape[-2:]
        optimal_chunks = (1, *probe_shape)  # One diffraction pattern per chunk
        
        dp_dataset_kwargs = {
            "shape": (n_positions, *probe_shape),
            "dtype": "float32",
            "chunks": optimal_chunks,  # Optimal chunking for diffraction patterns
        }
        
        # Apply compression if specified in config
        if self.compression is not None:
            if isinstance(self.compression, str):
                comp_name = self.compression.lower()
                if comp_name == "blosc":
                    # Default Blosc settings (EPICS area detector style)
                    blosc_kwargs = hdf5plugin.Blosc(
                        cname='zstd',
                        clevel=9,
                        shuffle=hdf5plugin.Blosc.SHUFFLE,
                    )
                    dp_dataset_kwargs.update(**blosc_kwargs)
                elif comp_name == "bitshuffle":
                    # Bitshuffle is optimal for scientific float data
                    bitshuffle_kwargs = hdf5plugin.Bitshuffle(nelems=0, cname='lz4')
                    dp_dataset_kwargs.update(**bitshuffle_kwargs)
                elif comp_name == "gzip":
                    # GZIP with shuffle for maximum compression
                    dp_dataset_kwargs.update({
                        "compression": "gzip",
                        "compression_opts": 9,
                        "shuffle": True,
                    })
            elif isinstance(self.compression, dict):
                # Custom compression settings
                comp_name = self.compression.get("name", "blosc").lower()
                
                if comp_name == "bitshuffle":
                    # Bitshuffle (optimal for float32 scientific data)
                    nelems = self.compression.get("nelems", 0)  # 0 = auto
                    cname = self.compression.get("cname", "lz4")  # lz4, none, or other
                    bitshuffle_kwargs = hdf5plugin.Bitshuffle(nelems=nelems, cname=cname)
                    dp_dataset_kwargs.update(**bitshuffle_kwargs)
                elif comp_name == "blosc":
                    # Handle shuffle: convert bool to Blosc constant if needed
                    shuffle_setting = self.compression.get("shuffle", True)
                    if isinstance(shuffle_setting, bool):
                        shuffle = hdf5plugin.Blosc.SHUFFLE if shuffle_setting else hdf5plugin.Blosc.NOSHUFFLE
                    else:
                        shuffle = shuffle_setting
                    
                    blosc_kwargs = hdf5plugin.Blosc(
                        cname=self.compression.get("cname", "zstd"),
                        clevel=self.compression.get("clevel", 9),
                        shuffle=shuffle,
                    )
                    dp_dataset_kwargs.update(**blosc_kwargs)
                elif comp_name == "gzip":
                    # GZIP compression
                    dp_dataset_kwargs.update({
                        "compression": "gzip",
                        "compression_opts": self.compression.get("clevel", 9),
                        "shuffle": self.compression.get("shuffle", True),
                    })
                
                # Override chunks if specified
                if "chunks" in self.compression:
                    dp_dataset_kwargs["chunks"] = self.compression["chunks"]
        
        # Create diffraction pattern dataset (with or without compression)
        self.f_dp.create_dataset("dp", **dp_dataset_kwargs)
        
        if self.probe_to_be_in_data_file is None:
            self.f_para.create_dataset("probe", data=self.probe.data.detach().cpu().numpy())
        else:
            self.f_para.create_dataset("probe", data=self.probe_to_be_in_data_file)
        probe_file_value = "" if self.probe_file is None else self.probe_file
        self.f_para.create_dataset(
            "probe_file",
            data=np.array(probe_file_value, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        object_file_value = "" if self.object_file is None else self.object_file
        self.f_para.create_dataset(
            "object_file",
            data=np.array(object_file_value, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        defocus_value = np.nan if self.probe_defocus_m is None else float(self.probe_defocus_m)
        self.f_para.create_dataset("probe_defocus_m", data=np.array(defocus_value, dtype=np.float64))
        object_min_mag_value = np.nan if self.object_min_mag is None else float(self.object_min_mag)
        self.f_para.create_dataset("object_min_mag", data=np.array(object_min_mag_value, dtype=np.float64))
        object_phase_range_value = np.nan if self.object_phase_range is None else float(self.object_phase_range)
        self.f_para.create_dataset("object_phase_range", data=np.array(object_phase_range_value, dtype=np.float64))
        self.f_para.create_dataset("probe_position_indexes", data=np.arange(self.positions.shape[0]).astype(int))
        self.f_para.create_dataset("probe_position_x_m", data=self.positions.data[:, 1].detach().cpu().numpy() * self.pixel_size)
        self.f_para.create_dataset("probe_position_y_m", data=self.positions.data[:, 0].detach().cpu().numpy() * self.pixel_size)
        self.f_para.create_dataset("object", data=self.object_.data.detach().cpu().numpy())
        self.f_para["object"].attrs["pixel_height_m"] = self.pixel_size
        self.f_para["object"].attrs["pixel_width_m"] = self.pixel_size
        self.f_para["object"].attrs["center_x_m"] = 0
        self.f_para["object"].attrs["center_y_m"] = 0
        
        if save_object_tiffs:
            tifffile.imwrite(os.path.join(self.output_dir, "true_object_phase.tiff"), np.angle(self.object_.data.detach().cpu().numpy()))
            tifffile.imwrite(os.path.join(self.output_dir, "true_object_mag.tiff"), np.abs(self.object_.data.detach().cpu().numpy()))
        
    def wrap_up(self):
        self.f_dp.close()
        self.f_para.close()
        
    def run(self, save_object_tiffs: bool = True):
        self.build_forward_model()
        self.build_output_files(save_object_tiffs=save_object_tiffs)

        i_pos = 0
        n_pos = self.positions.shape[0]
        pbar = tqdm.tqdm(total=n_pos, desc="Simulating", disable=not self.verbose)
        while i_pos < n_pos:
            i_end = min(i_pos + self.batch_size, n_pos)
            indices = torch.arange(i_pos, i_end).long()
            intensities = self.forward_model(indices=indices).detach().cpu().numpy()
            intensities = np.fft.fftshift(intensities, axes=(-2, -1))
            
            if self.add_poisson_noise:
                intensities = add_poisson_noise(intensities, self.total_photon_count)
            
            self.f_dp["dp"][i_pos:i_end] = intensities
            pbar.update(i_end - i_pos)
            i_pos = i_end
        self.wrap_up()
