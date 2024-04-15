from typing import Optional
from typing_extensions import override
from collections import defaultdict
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback
from aim.pytorch import track_gradients_dists, track_params_dists
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

    def __init__(self, logging_interval: Optional[int] = 1):
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
            track_params_dists(trainer.model, logger.experiment)
            track_gradients_dists(trainer.model, logger.experiment)


class AimLayerOutputDisctributionCallback(Callback):
    def __init__(self, logging_interval: Optional[int] = 1):
        super().__init__()
        self.logging_interval = logging_interval

        self.output = defaultdict(dict)
        self.hooks = []

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
            self.output[name] = output.detach().cpu().abs().histc(40, 0, 10).numpy()
            # self.output[name]["mean"] = output.mean()
            # self.output[name]["std"] = output.std()
            # self.output[name] = np.abs(output)

        return hook

    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.logging_interval == 0:
            logger = self.get_aim_logger(trainer)
            for name, output in self.output.items():
                logger.experiment.track(
                    Distribution(hist=output, bin_range=(0, 10)),
                    name=name,
                    context={
                        "type": "data",
                        "params": "output",
                    },
                )


# def track_property_distribution(run, layers, prop, attr):
#     data = get_property_layers(layers, prop, attr)
#     for name, params in data.items():
#         run.track(
#             Distribution(params[prop]),
#             name=name,
#             context={
#                 'type': 'data',
#                 'params': 'biases',
#             }
#         )


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


# def get_property_layers(layers: Dict[str, torch.nn.Module], prop, attr):
#     layer_prop = {}
#     for name, module in layers.items():
#         prop_value = None
#         if hasattr(module, prop) and getattr(module, prop) is not None:
#             prop_value = getattr(getattr(module, prop), attr)
#             if prop_value is not None:
#                 layer_prop[name][f"{prop}_{attr}"] = get_pt_tensor(prop_value).numpy()
#     return layer_prop


# Move tensor from GPU to CPU
# def get_pt_tensor(t):
#     return t.cpu() if hasattr(t, 'is_cuda') and t.is_cuda else t
