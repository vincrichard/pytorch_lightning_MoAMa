from typing import Callable
import torch
from torch import optim
import lightning as L
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from jaxtyping import Float

from src.models.gnn import GNN, GINConv
from src.data.datasets.zinc import MoAMaBatch


class MoAMa(torch.nn.Module):
    def __init__(self, encoder: GNN, encoder_to_decoder: torch.nn.Module, decoder: GINConv) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_to_decoder = encoder_to_decoder
        self.decoder = decoder

    def forward(self, x, edge_index, edge_attr, masked_atom_mask):
        node_rep = self.encoder(x, edge_index, edge_attr)
        decoder_input = self.encoder_to_decoder(node_rep)
        decoder_input[masked_atom_mask] = 0

        return node_rep, self.decoder(decoder_input, edge_index, edge_attr)


class LitMoAMa(L.LightningModule):
    def __init__(self, moama: MoAMa, criterion: Callable, beta: int):
        super().__init__()
        self.moama = moama
        self.criterion = criterion
        self.beta = beta

    def forward(self, batch: MoAMaBatch):
        raise self.moama.forward(batch.x, batch.edge_index, batch.edge_attr, batch.node_mask)

    def training_step(self, batch: MoAMaBatch, batch_idx: int):
        node_mask = batch.node_mask
        node_representation, pred_node = self.moama(batch.x, batch.edge_index, batch.edge_attr, node_mask)

        node_loss = self.criterion(batch.node_target, pred_node[node_mask])
        sim_loss = self.knowledge_enhanced_auxiliary_loss(batch, node_representation)

        loss = self.beta * sim_loss + (1 - self.beta) * node_loss
        self.log("train_node_loss", node_loss.cpu().item())
        self.log("train_sim_loss", sim_loss.cpu().item())
        self.log("train_loss", loss.cpu().item(), prog_bar=True)
        return loss

    def knowledge_enhanced_auxiliary_loss(self, batch: MoAMaBatch, node_representation: torch.Tensor) -> float:
        embedding: Float[torch.Tensor, "B, emb_dim"] = global_mean_pool(node_representation, batch.batch)

        fingerprint_loss = 0
        for i in range(1, embedding.size(0)):
            emb_sim: Float[torch.Tensor, "i"] = F.cosine_similarity(embedding[None, i], embedding[:i])
            fingerprint_loss += ((batch.tanimoto_sim[i, :i] - emb_sim) ** 2).sum()

        return torch.sqrt(fingerprint_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0)
        return optimizer
