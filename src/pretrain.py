from datetime import datetime
import logging
import logging.config
from pathlib import Path
import json
from argparse import ArgumentParser

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.data.datasets import similarity_collator
from src.moma_config import MoMAConfig
from src.config import Config


def main(config: MoMAConfig, logging_dir: str):
    # Build lazy config object
    config.build()

    config.dataset.pre_compute_fingerprints()
    config.dataset.masking_strategy.pre_compute_motifs(
        config.dataset.smiles, Path(config.dataset.dataset_path).parent
    )

    train_dataloader = DataLoader(
        config.dataset,
        collate_fn=similarity_collator,
        persistent_workers=True,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=16,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=config.devices,
        # max_steps=2000,
        max_epochs=100,
        logger=TensorBoardLogger(f"{logging_dir}/tensorboard", name="MoAMa"),
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"{logging_dir}/checkpoints", save_top_k=5, monitor="train_loss"
            ),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
        ],
        #  Debug
        # fast_dev_run=5 # disable callbacks
        # limit_train_batches=0.1
        # limit_val_batches=5
        # num_sanity_val_steps=2 # Run at the start of training
        #  Performance
        # profiler="simple",  # / "advanced"
    )

    trainer.fit(
        model=config.model,
        train_dataloaders=train_dataloader,
        ckpt_path=config.ckpt_path,
    )


def setup_logging(experiment_dir: str):
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    with open("data/config/logging.json", "r") as file:
        config = json.load(file)
    config["handlers"]["file"]["filename"] = f"{experiment_dir}/finetuning.log"
    logging.config.dictConfig(config=config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    config: MoMAConfig = Config.fromfile(args.config)

    experiment_dir = Path(
        f"{config.experiment_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config.dump(experiment_dir / "config.py")
    setup_logging(experiment_dir)
    main(config, experiment_dir)
