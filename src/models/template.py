import torch
from torch import optim, nn
import lightning as L


class MyModule(torch.nn.Module):
    """For inference outside Pytorch Lightning"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.module = MyModule()

    def forward(self, x):
        # copy forward of module
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        # trainer.fit(model, train_loader, valid_loader)
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        # this is the test loop
        # trainer.test(model, dataloaders=DataLoader(test_set))
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
