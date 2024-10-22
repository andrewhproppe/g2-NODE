import h5py

from src.pipeline.make_dataset import (
    param_permutations_random,
    make_training_g2s_2d,
    augment_training_g2s_2d,
)
from src.modules.SimulatedPCFS import generate_pcfs

# bounds_path = (f"../data/data_grid_params.xlsx")
# bounds_path = ("../../data/data_grid_params.xlsx")
bounds_path = "data_grid_params.xlsx"

# number of g2s to generate and number to augment
total = 100
frac_aug = 0.5
n = int(total * (1 - frac_aug))
naug = int(total * frac_aug)

# Randomly samples parameters for generating simulated PCFS objects. Each object contains 30 g2s

param_permutations = param_permutations_random(bounds_path, n)

randomize_linshape = 0.5
randomize_diffusion = 0.5
time_bounds = (1e5, 1e11)  # ps
nstage_range = (100, 150)
max_delay_range = (75, 150)  # ps

header = {
    "total": total,
    "augmented_fraction": frac_aug,
    "randomize_linshape": randomize_linshape,
    "randomize_diffusion": randomize_diffusion,
    "time_bounds": time_bounds,
    "nstage_range": nstage_range,
    "max_delay_range": max_delay_range,
}

# Make the g2s
g2s, params, spectra, t, df, nstage, δ, I_last, simPCFS = make_training_g2s_2d(
    param_permutations,
    generate_pcfs,
    randomize_lineshape=randomize_linshape,
    randomize_diffusion=randomize_diffusion,
    time_bounds=time_bounds,
    nstage_range=nstage_range,
    max_delay_range=max_delay_range,
)

# from matplotlib import pyplot as plt
# plt.plot(δ, I_last[0])
# plt.imshow(g2s[0])

# TODO: make this work with I_last etc. For now, just dumped using prior at all in image_data.py
# Create linear combinations of the g2s to augment training data
g2s, params = augment_training_g2s_2d(
    g2s,
    params,
    naug,
)

# # Plot to verify the variance of the g2s
# from matplotlib import pyplot as plt
# for g2 in g2s:
#     plt.plot(t, g2, color='k', alpha=0.1)
#
#
#

# #  Save the data to .h5 file
basepath = "raw/"
filepath = "pcfs_g2_2d_n%i_20240622.h5" % (n + naug)

with h5py.File(basepath + filepath, "a") as h5_data:
    h5_data["g2s"] = g2s
    h5_data["params"] = params
    h5_data["t"] = t
    h5_data["df"] = df
    h5_data["nstage"] = nstage
    h5_data["δ"] = δ
    h5_data["spectra"] = spectra
    h5_data["I_last"] = I_last
    h5_data["header"] = str(header)
