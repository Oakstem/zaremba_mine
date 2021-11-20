import copy
import timeit

import torch
from torch import nn, Tensor

from data.data import Data
from model.model_base import ModelBase
from .nll_loss import nll_loss
from .perplexity import perplexity


def train(model: ModelBase, data: Data, total_epochs_num: int, first_epoch_modify_lr: int, learning_rate: float,
          lr_decrease_factor: float, max_gradients_norm: float, device: int or str) -> [ModelBase, [float], [float]]:
    validation_perplexities: [float] = []
    test_perplexities: [float] = []
    best_model: ModelBase = None
    best_test_perplexity: float = -1.0

    for epoch_num in range(total_epochs_num):

        if epoch_num >= first_epoch_modify_lr and lr_decrease_factor != 0:
            learning_rate = learning_rate / lr_decrease_factor

        print("Epoch no = {:d} / {:d}, ".format(epoch_num + 1, total_epochs_num) +
              "learning rate = {:.3f}".format(learning_rate))

        train_one_epoch(model, data, learning_rate, max_gradients_norm, device)

        validation_perplexity = perplexity(model, data.validation_dataset, data.batch_size, device)
        validation_perplexities.append(validation_perplexity)
        print("Validation perplexity : {:.3f}".format(validation_perplexity))

        test_perplexity = perplexity(model, data.test_dataset, data.batch_size, device)
        test_perplexities.append(test_perplexity)
        print("Test perplexity : {:.3f}".format(test_perplexity))

        if best_test_perplexity == -1 or test_perplexity < best_test_perplexity:
            best_test_perplexity = test_perplexity
            best_model = copy.deepcopy(model)

    return best_model, validation_perplexities, test_perplexities


def train_one_epoch(model: ModelBase, data: Data, learning_rate: float, max_gradients_norm: float, device: str or int):
    train_dataset_length = len(data.train_dataset)

    states = model.state_init(data.batch_size)
    model.train()

    for current_data_index, (x, y) in enumerate(data.train_dataset):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        states = model.detach(states)
        scores, states = model(x, states)
        loss = nll_loss(scores, y)
        loss.backward()

        with torch.no_grad():
            norm: Tensor = nn.utils.clip_grad_norm_(model.parameters(), max_gradients_norm)
            for param in model.parameters():
                param -= learning_rate * param.grad

        if current_data_index % (train_dataset_length // 100) == 0:
            print("Batch no = {:d} / {:d}, ".format(current_data_index + 1, train_dataset_length) + \
                  "Norm = {:.3f}, ".format(norm) + \
                  "Train loss = {:.3f}, ".format(loss / data.batch_size))
