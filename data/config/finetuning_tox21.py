import torch
from torch_geometric.nn.aggr import MeanAggregation


from src.models.finetuning_model import LitFinetune
from src.config import read_base


with read_base():
    from .models.moama import encoder
    from .pretraining_moama import model as pretrained_model
    from .datasets.tox21 import tox21_dataset


experiment_dir = "logs/MoAMa_shared_weight/finetune/tox21"
# experiment_dir = "logs/MoAMa/finetune/tox21"
# experiment_dir = "logs/MoAMa_other_extractor/finetune/tox21"
devices = [0]
batch_size = 32

encoder.update(drop_ratio=0.5)
model = dict(
    type=LitFinetune,
    encoder=encoder,
    pooling=dict(type=MeanAggregation),
    prediction_head=dict(
        type=torch.nn.Linear, in_features=encoder.emb_dim, out_features=12
    ),
    criterion=dict(type=torch.nn.BCEWithLogitsLoss),
)

dataset = tox21_dataset


pretrained_model_encoder_path = "MoAMa/saved_model/encoder.pth"
# pretrained_model_ckpt_path = (
#     "logs/MoAMa/pretrain/2024-06-16_15-16/checkpoints/epoch=94-step=742235.ckpt"
# )
# pretrained_model_ckpt_path = "logs/MoAMa_other_extractor/pretrain/2024-06-24_09-42/checkpoints/epoch=98-step=773487.ckpt"
