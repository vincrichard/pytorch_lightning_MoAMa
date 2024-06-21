import torch

from src.models.moama import LitMoAMa, MoAMa
from src.models.gnn import GNN, GINConv

emb_dim = 300
num_atom_type = 119

encoder = dict(type=GNN, num_layer=5, emb_dim=emb_dim, drop_ratio=0.2)

moama = dict(
    type=MoAMa,
    encoder=encoder,
    encoder_to_decoder=dict(
        type=torch.nn.Sequential,
        args=[
            dict(type=torch.nn.PReLU),
            dict(type=torch.nn.Linear, in_features=emb_dim, out_features=emb_dim, bias=False),
        ],
    ),
    decoder=dict(type=GINConv, emb_dim=emb_dim, out_dim=num_atom_type, aggr="add"),
)
