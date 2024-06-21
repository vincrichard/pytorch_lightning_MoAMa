"""
Code use by multiple paper doing preptraining on graph.
The original implementation comes from: https://github.com/snap-stanford/pretrain-gnns

But it has been used in:
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from src.featurizer import SimpleGraph2dFeaturizer


num_atom_type = (
    max(SimpleGraph2dFeaturizer.ALLOW_ATOM_TYPES) + 2
)  # 120 #including the extra mask tokens, we also start at 0
num_chirality_tag = 3

num_bond_type = (
    len(SimpleGraph2dFeaturizer.BOND_TYPES) + 2
)  # 4 #including aromatic and self-loop edge
num_bond_direction = len(SimpleGraph2dFeaturizer.BOND_DIR)


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        out_dim


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault("aggr", aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(
            device=edge_attr.device, dtype=edge_attr.dtype
        )
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer < self.num_layer - 1:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            else:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)

        return h_list[-1]


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, gnn_type="gin"):
        super().__init__()
        self._dec_type = gnn_type
        self.conv = GINConv(hidden_dim, out_dim, aggr="add")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = torch.nn.PReLU()
        self.temp = 0.2

    def forward(self, x, batch):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            masked_node_indices = batch.masked_atom_indices_atom

            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[masked_node_indices] = 0

            out = self.conv(x, edge_index, edge_attr)

        return out
