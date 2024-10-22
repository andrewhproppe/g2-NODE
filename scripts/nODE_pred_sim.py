import sys
import matplotlib.pyplot as plt
import torch

from src.utils import paths
from src.models.ode.ode import g2LSTMODE
from src.visualization.fig_utils import *
from src.pipeline.image_data import ODEDataModule, PCFSDataset

model = g2LSTMODE.load_from_checkpoint(
    # checkpoint_path=paths.get("trained_models").joinpath("fine-aardvark-2.ckpt"),
    # checkpoint_path=paths.get("trained_models").joinpath("pious-aardvark-4.ckpt"),
    # checkpoint_path=paths.get("trained_models").joinpath("sweet-dawn-6-ic.ckpt"),
    checkpoint_path=paths.get("trained_models").joinpath("sleek-jazz-15.ckpt"),
    # map_location=torch.device("cpu")
) #.eval()

# Testing with simulated data
dm = ODEDataModule(
    # "pcfs_g2_2d_n50000_20240623.h5",
    "pcfs_g2_2d_n110_20240714.h5",
    batch_size=10,
    add_noise=True,
    scale_range=(1e1, 1e3),
    add_time_dim=True,
    nobs=10,
    obs_type="fixed",
    num_workers=0,
    pin_memory=True,
    split_type="random",
)

dm.setup()
batch = next(iter(dm.val_dataloader()))
X, Y, t_0, t = batch

optical_delays = dm.val_set.dataset.optical_delays

with torch.no_grad():
    pred_Y_sim = model(t_0, t[0]).squeeze(0)  # solve the initial value problem


obs_inds = [0, 1, 2, 3, 6, 8, 14, 18, 22, 30, 38, 46, 54, 62, 70, 80, 90]

idx = 2
# Plot Y_sim and pred_Y_sim to compare
fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
set_font_size(10)
for i, g2 in enumerate(t_0[0]):
    # Label with optical delay at index i
    ax[0].plot(g2[:-1].detach().numpy(), linewidth=1.5, label=f"{optical_delays[obs_inds[i]]:.2f} ps")
ax[0].set_xlabel("$\it{τ}$ (ps)")
ax[0].set_ylabel("$g^{(2)}(τ)$")
ax[0].set_ylim(0.5, 1.05)
ax[0].legend(loc="best", fontsize=10, ncol=3, frameon=False)
ax[1].imshow(pred_Y_sim[idx].detach().numpy())
ax[1].set_xlabel("$\it{τ}$ (ps)")
ax[1].set_ylabel("$\it{t}$ (ps)")
add_colorbar(ax[1].images[0], labelsize=10)
ax[2].imshow(Y[idx])
ax[2].set_xlabel("$\it{τ}$ (ps)")
ax[2].set_ylabel("$\it{t}$ (ps)")
add_colorbar(ax[2].images[0], labelsize=10)
plt.tight_layout()