import os

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint,
)
from aim.pytorch_lightning import AimLogger
from torch.utils.data import DataLoader

from src.models import autoencoder
from src.utils.aim_callback import (
    AimParamGradientCallback,
    AimLayerOutputDisctributionCallback,
    get_run_folder_aim_logger,
)
from src.utils.torchviz import ModelGraphCallback


# track experimental data by using Aim
aim_logger = AimLogger(
    experiment="aim_on_pt_lightning",
    train_metric_prefix="train_",
    val_metric_prefix="val_",
)


dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = DataLoader(dataset, batch_size=32)

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    limit_train_batches=100,
    max_epochs=4,
    logger=aim_logger,
    log_every_n_steps=1,
    # default_root_dir="some/path/",
    # enable_checkpointing=False
    callbacks=[
        ModelCheckpoint(
            dirpath=get_run_folder_aim_logger(aim_logger),
            save_top_k=5,
            monitor="train_loss",
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        AimParamGradientCallback(),
        AimLayerOutputDisctributionCallback(),
        ModelGraphCallback(dirpath=get_run_folder_aim_logger(aim_logger)),
    ],
    #  Debug
    # fast_dev_run=5 # disable callbacks
    # limit_train_batches=0.1
    # limit_val_batches=5
    # num_sanity_val_steps=2 # Run at the start of training
    #  Performance
    # profiler="simple" / "advanced"
)

trainer.fit(
    model=autoencoder,
    train_dataloaders=train_loader,
    # val_dataloaders=,
)
