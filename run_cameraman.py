import json

import numpy as np
import torch
import skimage.data

import simulator as sim


if __name__ == "__main__":
    torch.set_default_device("cuda")

    object = skimage.data.camera()
    object = sim.create_complex_object(object, mag_min=1, phase_range=0.5)
    object = object[None, ...]
    
    # FWHM = 2.355 sigma
    sigma = 20
    probe = sim.gaussian_2d([128, 128], sigma=sigma)
    probe = probe[None, None, ...].astype(np.complex64)
    
    target_overlap = 0.8
    spacing = (1 - target_overlap) * sigma * 2.355
    positions = sim.create_positions(object.shape[1:], probe.shape[-2:], spacing_y=spacing, spacing_x=spacing)
    
    opr_weights = sim.generate_initial_opr_mode_weights(len(positions), 1)
    
    pixel_size_m = 1e-8
    
    simulator = sim.PtychographySimulator(
        object, probe, positions, opr_weights, pixel_size_m, batch_size=100, wavelength_m=1e-10,
        output_dir="outputs/cameraman",
    )
    simulator.run()
    
    json.dump(
        {
            "pixel_size_m": pixel_size_m,
        },
        open("outputs/cameraman/simulator_params.json", "w"),
    )
