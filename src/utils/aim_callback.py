from typing import Any, Optional
from typing_extensions import override
from collections import defaultdict
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback
from aim.pytorch_lightning import AimLogger
from aim import Distribution, Run


def get_experiment_folder_aim(aim_run: Run):
    return str(
        Path(aim_run.repo.path)
        / aim_run.experiment
        / aim_run.created_at.strftime("%Y-%m-%d_%H-%M")
    )


def get_run_folder_aim_logger(logger: AimLogger):
    return str(
        Path(logger.save_dir)
        / logger.experiment.experiment
        / logger.experiment.created_at.strftime("%Y-%m-%d_%H-%M")
    )


class AimParamGradientCallback(Callback):

    def __init__(self, logging_interval: Optional[int] = 25):
        super().__init__()
        self.logging_interval = logging_interval

    def get_aim_logger(self, trainer: L.Trainer):
        for logger in trainer.loggers:
            if isinstance(logger, AimLogger):
                return logger

    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.logging_interval == 0:
            logger = self.get_aim_logger(trainer)
            layers = get_model_layers(trainer.model)
            self.track_weight_bias_property_layers(
                logger.experiment, layers, props=["data", "grad"]
            )

    def track_weight_bias_property_layers(self, run, layers, props):
        for name, module in layers.items():
            for wb in ["weight", "biais"]:
                if hasattr(module, wb) and getattr(module, wb) is not None:
                    for prop in props:
                        weight_prop = getattr(module.weight, prop, None)
                        run.track(
                            Distribution(weight_prop.cpu().numpy()),
                            name=name,
                            context={
                                "type": prop,
                                "params": wb,
                            },
                        )


class AimLayerOutputDisctributionCallback(Callback):
    def __init__(self, logging_interval: Optional[int] = 1):
        super().__init__()
        self.logging_interval = logging_interval

        self.output = defaultdict(dict)
        self.hooks = []
        self.is_enable = 1

    def get_aim_logger(self, trainer: L.Trainer):
        for logger in trainer.loggers:
            if isinstance(logger, AimLogger):
                return logger

    @override
    def on_fit_start(self, trainer, pl_module):
        layers = get_model_layers(trainer.model)
        for name, module in layers.items():
            hook = module.register_forward_hook(self.get_output(name))
            self.hooks.append(hook)

    @override
    def on_fit_end(self, trainer, pl_module):
        for hooks in self.hooks:
            hooks.remove()

    def get_output(self, name):
        def hook(model, input, output):
            if self.is_enable:
                self.output[name] = output.detach().abs().histc(40, 0, 10)

        return hook

    @override
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.is_enable = batch_idx % self.logging_interval == 0

    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.is_enable:
            logger = self.get_aim_logger(trainer)
            for name, output in self.output.items():
                logger.experiment.track(
                    Distribution(hist=output.cpu().numpy(), bin_range=(0, 10)),
                    name=name,
                    context={
                        "type": "data",
                        "params": "output",
                    },
                    step=batch_idx,
                )


def get_model_layers(model, parent_name=None):
    layers = {}
    for name, m in model.named_children():
        layer_name = "{}__{}".format(parent_name, name) if parent_name else name
        layer_name += ".{}".format(type(m).__name__)

        if len(list(m.named_children())):
            layers.update(get_model_layers(m, layer_name))
        else:
            layers[layer_name] = m
    return layers
