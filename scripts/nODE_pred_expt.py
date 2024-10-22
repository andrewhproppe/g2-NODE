import pickle
import sys
import matplotlib.pyplot as plt
import torch

from src.utils import paths
from src.models.ode.ode import g2LSTMODE
from src.modules.PCFS import load_dot, correct_g2_oscillations
from src.visualization.fig_utils import *
from src.pipeline.image_data import ODEDataModule, PCFSDataset
sys.path.append("../src/modules")

# dot = load_dot("experimental_dot2.pickle")
# dot = load_dot("CK_dot_modded.pickle")
# dot = load_dot("dotY_run7.pickle") # YES
# dot = load_dot("dotY_run8.pickle") # YES
dot = load_dot("dotY_run9.pickle") # YES
# dot = load_dot("dotY_run10.pickle") # YES

# raise RuntimeError

model = g2LSTMODE.load_from_checkpoint(
    # checkpoint_path=paths.get("trained_models").joinpath("fine-aardvark-2.ckpt"),
    # checkpoint_path=paths.get("trained_models").joinpath("pious-aardvark-4.ckpt"),
    # checkpoint_path=paths.get("trained_models").joinpath("sweet-dawn-6-ic.ckpt"),
    # checkpoint_path=paths.get("trained_models").joinpath("sleek-jazz-15.ckpt"),
    # checkpoint_path=paths.get("trained_models").joinpath("laced-disco-27.ckpt"),
    checkpoint_path=paths.get("trained_models").joinpath("logical-puddle-28.ckpt"),
    # map_location=torch.device("cpu")
)#.eval()

# X = correct_g2_oscillations(dot, correction_scalar=1, tau_for_correction_factor=1e11)

X = dot.g2

t_0, t = PCFSDataset.prepare_t0(torch.tensor(X, dtype=torch.float32), nobs=15, add_time_dim=True, nsteps=175)

# set_font_size(10)
# for i, g2 in enumerate(t_0):
#     # Label with optical delay at index i
#     plt.plot(g2[:-1].detach().numpy(), linewidth=1.5,)

# raise RuntimeError

with torch.no_grad():
    pred_Y = model(t_0.unsqueeze(0), t).squeeze(0)  # solve the initial value problem

fig, ax = plt.subplots(1, 2)
ax[0].contourf(X, levels=100)
ax[1].contourf(pred_Y.detach().numpy(), levels=100)

fig2 = plt.figure()
idx = -1
plt.plot(pred_Y[:, idx])
plt.plot(X[:, idx])



#
# # Testing with simulated data
# dm = ODEDataModule(
#     "pcfs_g2_2d_n50000_20240623.h5",
#     batch_size=16,
#     add_noise=True,
#     scale_range=(1e1, 1e3),
#     add_time_dim=True,
#     nobs=10,
#     obs_type="fixed",
#     num_workers=0,
#     pin_memory=True,
#     split_type="random",
# )
#
# dm.setup()
# batch = next(iter(dm.val_dataloader()))
# X_sim, Y_sim, t_0_sim, t_sim = batch
#
# idx = 1
# t_0_sim, t_sim = PCFSDataset.prepare_t0(X_sim[idx])
#
# with torch.no_grad():
#     pred_Y_sim = model(t_0_sim.unsqueeze(0), t_sim).squeeze(0)  # solve the initial value problem
#
# # Plot Y_sim and pred_Y_sim to compare
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(Y_sim[idx])
# ax[1].imshow(pred_Y_sim.detach().numpy())