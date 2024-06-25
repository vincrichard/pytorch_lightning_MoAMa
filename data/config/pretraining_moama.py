from src.models.moama import LitMoAMa
from src.criterion import sce_loss
from src.config import read_base

with read_base():
    from .datasets.zinc_dataset import zinc_dataset, dataset_path
    from .models.moama import moama

dataset = zinc_dataset

# Misc

experiment_dir = "logs/MoAMa/pretrain"
devices = [0]

batch_size = 256


model = dict(type=LitMoAMa, moama=moama, criterion=sce_loss, beta=0.5)
