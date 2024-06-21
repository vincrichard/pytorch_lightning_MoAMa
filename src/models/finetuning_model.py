from typing import Union

import torch
from torch import optim
from torch_geometric.data import Data
import lightning as L
from sklearn.metrics import roc_auc_score


class LitFinetune(L.LightningModule):

    def __init__(self, encoder, pooling, prediction_head, criterion):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.prediction_head = prediction_head
        self.criterion = criterion

    def forward(self, batch: Data) -> torch.Any:
        pred, _ = self.shared_step(batch, batch.y)
        return pred

    def shared_step(self, data: Data, target: torch.Tensor):
        node_rep = self.encoder(data.x, data.edge_index, data.edge_attr)
        pred = self.prediction_head(self.pooling(node_rep, data.batch))

        is_not_nan = target == target
        loss = self.criterion(pred[is_not_nan], target[is_not_nan])

        return pred, loss

    def training_step(self, batch: Union[Data, torch.Tensor], batch_idx: int):
        pred, loss = self.shared_step(batch, batch.y)
        self.log("train_loss", loss.cpu().item(), prog_bar=True, batch_size=pred.size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.preds, self.target = [], []

    def validation_step(self, batch: Data, batch_idx):
        data, target = batch, batch.y
        pred, loss = self.shared_step(data, target)
        self.log("val_loss", loss.cpu().item(), prog_bar=True, batch_size=target.size(0))
        self.preds.append(pred)
        self.target.append(target)

        return loss

    def on_validation_epoch_end(self):
        roc_auc = self.compute_mask_roc_auc()
        self.log("val_roc_auc", roc_auc, prog_bar=True, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.preds, self.target = [], []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        roc_auc = self.compute_mask_roc_auc()
        self.log("test_roc_auc", roc_auc, prog_bar=True, on_epoch=True)
        self.test_roc_auc = roc_auc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0)
        return optimizer

    def compute_mask_roc_auc(self):
        preds, target = (
            torch.concat(self.preds).cpu().numpy(),
            torch.concat(self.target).cpu().numpy(),
        )
        is_nan = target == target
        roc_auc = 0
        for i in range(target.shape[1]):
            mask = is_nan[:, i]
            roc_auc += roc_auc_score(target[mask, i], preds[mask, i])
        return roc_auc / target.shape[1]
