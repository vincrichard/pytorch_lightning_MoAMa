import torch
from torch import optim, nn
import lightning as L

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(
    nn.Linear(28 * 28, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 3)
)
decoder = nn.Sequential(
    nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 28 * 28)
)


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        gain = torch.nn.init.calculate_gain("relu")
        for p in self.encoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
