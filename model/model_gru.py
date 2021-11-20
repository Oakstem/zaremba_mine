import torch
from torch import nn

from model.model_base import ModelBase


class ModelGRU(ModelBase):
    def __init__(self, vocab_size: int, num_of_layers: int, hidden_layer_units: int,
                 dropout: float, weights_uniforming: float, batch_sz: int):
        super(ModelGRU, self).__init__(vocab_size, num_of_layers, hidden_layer_units,
                                       dropout, weights_uniforming, batch_sz)

    def get_rnn_type(self) -> type:
        return nn.GRU

    def create_single_state(self, batch_size: int, hidden_size: int, device: str or int):
        return (torch.zeros(1, batch_size, hidden_size, device = device))

    @staticmethod
    def detach(states):
        states_detached: [] = []
        for h in states:
            states_detached.append((h.detach()))
        return states_detached
