import torch
from torch import Tensor


def nll_loss(scores: Tensor, expected: Tensor):
    batch_size: int = expected.size()[1]
    exp_scores: Tensor = scores.exp()

    probabilities = exp_scores / exp_scores.sum(1, keepdim = True)

    index = range(len(expected.reshape(-1))), expected.reshape(-1)
    answer_probabilities = probabilities[index]

    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-torch.log(answer_probabilities) * batch_size)
