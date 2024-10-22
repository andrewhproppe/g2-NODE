import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from src.pipeline.image_data import ODEDataModule
from src.models.ode.ode_models import AttentionBlock, MLP
from src.models.ode.ode import g2ConvODE

if __name__ == "__main__":
    seed_everything(42, workers=True)

    dm = ODEDataModule(
        "pcfs_g2_2d_n50000_20240820_nstage100_maxdelay120.h5", # 128 input size, 100 stage positions
        batch_size=256,
        add_noise=True,
        scale_range=(1e1, 1e3),
        add_time_dim=True,
        nobs=10,
        nsol=100,
        obs_type="fixed",
        num_workers=0,
        pin_memory=True,
        split_type="random",
    )

    # dm.setup()
    # test = next(iter(dm.val_dataloader()))

    model = g2ConvODE(
        input_size=128,
        enc_depth=3,
        enc_channels=[1, 64, 64, 64, 64, 64],
        enc_kernels=[3, 3, 3, 3, 3],
        z_size=2**6,
        vf_depth=4,
        attn_depth=2,
        attn_heads=4,
        vector_field=MLP,
        attention=AttentionBlock,
        time_dim=1,
        augment=False,
        augment_size=1,
        nobs=10,
        obs_type="fixed",
        atol=1e-4,
        rtol=1e-4,
        lr=5e-4,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        fourier_weight=1,
        plot_interval=10,
    )

    logger = WandbLogger(
        entity="aproppe",
        project="g2-nODE",
        # mode="offline",
        mode="online",
        # log_model=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=300,
        # max_steps=50000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[0],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
