import numpy as np
import torch

from model.model_base import ModelBase
from .nll_loss import nll_loss


def perplexity(model: ModelBase, dataset: list, batch_size: int, device: str or int):
    model.eval()
    torch.no_grad()

    states = model.state_init(batch_size)

    model.eval()

    losses = []
    for x, y in dataset:
        x = x.to(device)
        y = y.to(device)
        scores, states = model(x, states)
        loss = nll_loss(scores, y)

        losses.append(loss.data.item()/batch_size)

    perplexity_value = np.exp(np.mean(losses))
    return perplexity_value
