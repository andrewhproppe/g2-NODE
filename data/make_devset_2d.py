import h5py

from src.pipeline.make_dataset import (
    param_permutations_random,
    make_training_g2s_2d,
    augment_training_g2s_2d,
)
from src.modules.SimulatedPCFS import generate_pcfs
from matplotlib import pyplot as plt

# bounds_path = (f"../data/data_grid_params.xlsx")
# bounds_path = ("../../data/data_grid_params.xlsx")
bounds_path = "data_grid_params.xlsx"

# number of g2s to generate and number to augment
total = 50000
# total = 20
frac_aug = 0.3
n = int(total * (1 - frac_aug))
naug = int(total * frac_aug)

save = True

# Randomly samples parameters for generating simulated PCFS objects. Each object contains 30 g2s
param_permutations = param_permutations_random(bounds_path, n)

randomize_linshape = 0.5
randomize_diffusion = 0.5
lag_precision = 6
time_bounds = (1e5, 2.6e11)  # ps
nstage = 100
max_delay = 100  # ps

# Step size for Weiwei dots = 0.3335641 for 175 stage positions

header = {
    "total": total,
    "randomize_linshape": randomize_linshape,
    "randomize_diffusion": randomize_diffusion,
    "time_bounds": time_bounds,
    "nstage": nstage,
    "max_delay": max_delay,
}

# Make the g2s
g2s, params, spectra, t, df, _, δ, I_last, _ = make_training_g2s_2d(
    param_permutations,
    generate_pcfs,
    randomize_lineshape=randomize_linshape,
    randomize_diffusion=randomize_diffusion,
    lag_precision=lag_precision,
    time_bounds=time_bounds,
    nstage=nstage,
    max_delay=max_delay,
)

g2s, params = augment_training_g2s_2d(
    g2s,
    params,
    naug,
)

# raise RuntimeError

# # Testing dither
# import numpy as np
# plt.plot(np.exp(t[0]), g2s[0][0, :])
# plt.xscale('log')
# plt.xlim(1e10, 1e13)
# # plt.ylim(0.6, 1.4)


if save:
    # #  Save the data to .h5 file
    basepath = "raw/"
    filepath = "pcfs_g2_2d_n%i_20240820_nstage100_maxdelay120.h5" % (n + naug)

    with h5py.File(basepath + filepath, "a") as h5_data:
        h5_data["g2s"] = g2s
        h5_data["params"] = params
        h5_data["t"] = t[0]
        h5_data["df"] = df[0]
        h5_data["nstage"] = nstage
        h5_data["δ"] = δ[0]
        h5_data["lag_precision"] = lag_precision
        h5_data["time_bounds"] = time_bounds
        # h5_data["spectra"] = spectra
        # h5_data["I_last"] = I_last
        h5_data["header"] = str(header)
