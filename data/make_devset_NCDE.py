import h5py
import torch
import time

from src.pipeline.make_dataset import param_permutations_random, make_training_g2s_2d, augment_training_g2s_2d
from src.modules.SimulatedPCFS import generate_pcfs
from utils import get_cubic_spline_coeffs

# bounds_path = (f"../data/data_grid_params.xlsx")
# bounds_path = ("../../data/data_grid_params.xlsx")
bounds_path = ("data_grid_params.xlsx")

# number of g2s to generate and number to augment
total    = 1
frac_aug = 0
# total = 1
# frac_aug = 0
n = int(total*(1-frac_aug))
naug = int(total*frac_aug)

# Randomly samples parameters for generating simulated PCFS objects. Each object contains 30 g2s

param_permutations = param_permutations_random(bounds_path, n)

# Make the g2s
g2s, params, spectra, t, df, nstage, δ, I_last, simPCFS = make_training_g2s_2d(
    param_permutations,
    generate_pcfs,
    0.5,
    0.5,
)

# from matplotlib import pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
# ax[0].plot(δ, I_last[0])
# ax[1].imshow(g2s[0])
# raise Exception("Stop here")

# Create linear combinations of the g2s to augment training data
g2s, params = augment_training_g2s_2d(
    g2s,
    params,
    naug,
)


# Make cubic splines for NCDE
obs_inds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 9, 12, 20, 28, 44, 60, 99], dtype=torch.long)
spline_coeffs = []
for g2 in g2s:
    g2 = torch.tensor(g2, dtype=torch.float32)
    spline_coeffs.append(get_cubic_spline_coeffs(g2[obs_inds, :], torch.tensor(t, dtype=torch.float32), obs_inds).numpy())

# raise Exception("Stop here")

# Save the data to .h5 file
# basepath = "../../data/raw/"
basepath = "raw/"
filepath = 'ncde_g2_n%i.h5' % (n+naug)

with h5py.File(basepath+filepath, "a") as h5_data:
    h5_data["g2s"] = g2s
    h5_data["params"] = params
    h5_data["t"] = t
    h5_data["df"] = df
    h5_data["nstage"] = nstage
    h5_data["δ"] = δ
    h5_data["spectra"] = spectra
    h5_data["I_last"] = I_last
    h5_data["spline_coeffs"] = spline_coeffs
