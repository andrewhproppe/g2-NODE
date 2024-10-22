import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from src.pipeline.image_data import ODEDataModule
from src.models.ode.ode import g2Conv1d

if __name__ == "__main__":
    seed_everything(42, workers=True)

    dm = ODEDataModule(
        "pcfs_g2_2d_n50000_20240820_nstage100_maxdelay120.h5", # 128 input size, 100 stage positions
        batch_size=256,
        add_noise=True,
        scale_range=(1e1, 1e3),
        add_time_dim=False,
        nobs=10,
        nsol=100,
        obs_type="fixed",
        num_workers=0,
        pin_memory=True,
        split_type="random",
    )

    # dm.setup()
    # test = next(iter(dm.val_dataloader()))

    model = g2Conv1d(
        depth=6,
        channels_in=10,
        channels_out=100,
        channels_hidden=256,
        lr=5e-4,
        lr_schedule="RLROP",
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
