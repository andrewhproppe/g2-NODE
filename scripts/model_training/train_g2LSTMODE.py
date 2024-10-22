import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from src.pipeline.image_data import ODEDataModule
from src.models.ode.ode_models import (
    AttentionBlock,
    LSTMEncoder,
    LSTMDecoder,
    MLPStack,
)
from src.models.ode.ode import g2LSTMODE

if __name__ == "__main__":
    seed_everything(42, workers=True)

    dm = ODEDataModule(
        "pcfs_g2_2d_n50000_20240820_nstage100_maxdelay120.h5",  # 128 input size, 100 stage positions
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

    model = g2LSTMODE(
        input_size=128,
        # input_size=140,
        enc_hidden_size=256,
        enc_depth=3,
        z_size=2**7,
        vf_depth=4,
        vf_hidden_size=256,
        attn_depth=2,
        attn_heads=4,
        norm=False,
        encoder=LSTMEncoder,
        vector_field=MLPStack,
        attention=AttentionBlock,
        # attention=None,
        decoder=LSTMDecoder,
        time_dim=1,
        nobs=10,
        augment=True,
        augment_size=1,
        atol=1e-2,
        rtol=1e-2,
        dropout=0.0,
        lr=5e-4,
        lr_schedule="RLROP",
        weight_decay=1e-5,
        plot_interval=10,
        fourier_weight=1.0,
        data_info=dm.header,
    )

    logger = WandbLogger(
        entity="aproppe",
        project="g2-nODE",
        mode="offline",
        # mode="online",
        # log_model=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=300,
        # max_steps=50000,
        logger=logger,
        # enable_checkpointing=False,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[3],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)