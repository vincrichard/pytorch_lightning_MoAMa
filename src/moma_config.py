from typing import Protocol, List, Union, Optional
from pathlib import Path

from src.featurizer.motif_extractor import AbstractMotifExtractor
from src.featurizer import MotifMaskAtom
from src.data.datasets import Zinc, Tox21
from src.models.moama import LitMoAMa, MoAMa
from src.models.finetuning_model import LitFinetune


class ConfigProtocol(Protocol):
    @staticmethod
    def fromfile(
        filename: Union[str, Path], format_python_code: bool = True
    ) -> "Config": ...  # noqa:F821

    def build(self) -> None: ...

    def dump(self, file: Optional[Union[str, Path]] = None): ...


class MoMAConfig(ConfigProtocol):
    experiment_dir: str
    devices: List[int]
    batch_size: int
    ckpt_path: Optional[str]

    dataset_path: str
    dataset: Zinc
    motif_extractor: AbstractMotifExtractor
    masking_strategy: MotifMaskAtom

    model: LitMoAMa
    moama: MoAMa


class FinetuningConfig(ConfigProtocol):
    experiment_dir: str
    devices: List[int]
    batch_size: int
    pretrained_model_ckpt_path: str

    dataset: Tox21
    model: LitFinetune
    pretrained_model: LitMoAMa
