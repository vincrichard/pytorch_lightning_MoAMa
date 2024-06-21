import torch.nn.functional as F


def sce_loss(x, y, alpha=1):
    """
    While I understand the use of the loss for a simple example in 2 dimension
    and for matching embeddings. I am not sure of it's use for atom reconstruction.
    See notebook for detail.
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()
