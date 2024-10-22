import pickle
import sys
import matplotlib.pyplot as plt
import torch

from src.modules.PCFS import load_dot
from src.visualization.fig_utils import *
sys.path.append("../src/modules")

""" 
Use this script to grab dots from the NanoLett2023_dots or PRL2023_dots folder, and process them to ensure the white 
fringe  starts at the first stage index, and set the first and last tau entries to 1e5 and 1e11, respectively.

These experimental dot objects in NanoLett2023_dots are the same used in Sun (Bawendi) et al., Nano Lett. 2023, 23, 7, 2615–2622
(https://doi.org/10.1021/acs.nanolett.2c04895), and those from PRL2023_dots are the same used in Proppe (Bawendi) et al., PRL, 2023, 131, 053603 
"""

# This dot was correalted with time bounds of [1e5, 1e11] and lag precision of 7
# dot = load_dot("nanolett_dot_better.pickle")  # GOOD even with g2
# dot = load_dot("NanoLett2023_dots/DotY_PCFS_runfour_full.pickle")  # GOOD even with g2
# dot = load_dot("NanoLett2023_dots/DotAb_PCFS_runnineteen_full.pickle") # decays immediately
# dot = load_dot("NanoLett2023_dots/DotC_PCFS_runfive_full.pickle") # optical delay step siez of 0.533 ps
# dot = load_dot("NanoLett2023_dots/DotY_PCFS_runnine_full.pickle") # keep
# dot = load_dot("NanoLett2023_dots/DotY_PCFS_runeight_full.pickle") # keep
# dot = load_dot("NanoLett2023_dots/DotY_PCFS_runten_full.pickle") # keep
dot = load_dot("NanoLett2023_dots/DotY_PCFS_runseven_full.pickle") # keep
# dot = load_dot("NanoLett2023_dots/DotFe_PCFS_run_full.pickle") # keep
# dot = load_dot("CK174_dot6_full")

# plt.plot(dot.tau, dot.g2[0, :])
# plt.xscale('log')
# plt.xlim(1e10, 1e13)
# plt.ylim(0.6, 1.4)
#
# raise RuntimeError

# # Try some after-pulsing correction
# def afterpulse(x, function, theta):
#     w = np.array(theta)
#     f = eval(function)
#     return f
#
# afterpulse_func = "w[0]*np.exp(-(x-w[2])/w[1])+1"
# w = [0.3, 0.7, 11.413]  # dot2_overnight
# afterpulse_fit = afterpulse(np.log(dot.tau), afterpulse_func, w)
# afterpulse_fit[np.where(np.log(dot.tau) < w[2])] = 0.25
#
# #
# # plt.plot(dot.tau, dot.g2_a[5, :])
# # # plt.plot(dot.tau, dot.g2_x[3, :])
# # plt.xlabel('log$_{10}$(τ, ps)')
# # plt.ylabel('$g^2(τ)$')
# # # plt.xlim([5e4, 1e9])
# # # plt.ylim([0.9, 3])
# # plt.plot(dot.tau, afterpulse_fit)
# # plt.xscale('log')
#
# dot.afterpulse_corrected = False
# dot.afterpulse_correction(afterpulse_fit)
# # dot.afterpulse_decorrection()

# raise RuntimeError

# Limit the tau range
# tau_start = 0
# tau_end = -1
tau_start = find_nearest(dot.tau, 1e5)
tau_end = find_nearest(dot.tau, 1e11) + 1
new_tau = dot.tau[tau_start:tau_end]
# new_tau = dot.tau

new_g2s = dot.g2[:, tau_start:tau_end]

# Find the white fringe index
wf_idx = np.where(new_g2s[:, -1]==new_g2s[:, -1].min())[0][0]
end_idx = -2

# Start from wf and end with 175 stage positions
new_g2s = new_g2s[wf_idx:end_idx, :]

# Get the new stage positions
new_stage_pos = dot.stage_positions[wf_idx:end_idx]
new_stage_pos = new_stage_pos - new_stage_pos[0]

def stage_pos_to_optical_delay(x):
    return 2 * x / (299792458) / 1000 * 1e12

optical_delay = stage_pos_to_optical_delay(new_stage_pos)

# raise RuntimeError

# raise RuntimeError
dot.optical_delay = optical_delay
dot.stage_positions = new_stage_pos
dot.g2 = new_g2s
dot.tau = new_tau
# dot.g2 = np.array(dot.cross_correlations) # to avoid after-pulsing
#
# # raise RuntimeError

# pickle_out = open("../dots/dotY_run7.pickle", "wb")
# pickle.dump(dot, pickle_out)
# pickle_out.close()

#
# delta_start = 3
# delta_end = -2
#
# # new_g2s = dot.g2[delta_start:delta_end, tau_start:tau_end]
# # new_tau = dot.tau[tau_start:tau_end]
#
# new_g2s = dot.g2[delta_start:delta_end, :]
# new_tau = dot.tau
# new_delta = dot.optical_delay[delta_start:delta_end]
# new_delta -= new_delta.min()
#
# raise RuntimeError
#
# dot.g2 = new_g2s
# dot.tau = new_tau
# dot.optical_delay = new_delta
#
# # step_size = 0.3335641


#
# plt.contourf(
#     new_tau,
#     new_delta,
#     new_g2s,
#     cmap="viridis",
# )
# plt.xscale("log")
#
#
# # len(dot.tau[tau_start : tau_end])
#
# # Flip y-axis
# plt.gca().invert_yaxis()
# plt.colorbar()
# # Set c-axis limits to 0 and 1
# # plt.clim(0, 1)
# plt.show()
#
# # Plot some g2s in the new tau range to emulate in simulated data
# plt.figure()
# plt.plot(dot.tau[tau_start:tau_end], dot.g2[0, tau_start:tau_end])
# plt.xscale("log")
