from src.featurizer.motif_extractor import BricsRingMotifExtractor
from src.config import read_base

with read_base():
    from .pretraining_moama import *


experiment_dir = "logs/MoAMa_other_extractor/pretrain"
devices = [1]

dataset.update(
    masking_strategy=dict(
        motif_extractor=dict(type=BricsRingMotifExtractor, motif_depth=5),
    ),
)
