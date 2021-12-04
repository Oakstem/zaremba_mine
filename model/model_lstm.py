import torch
from torch import nn

from model.model_base import ModelBase


class ModelLSTM(ModelBase):
    def __init__(self, vocab_size: int, num_of_layers: int, hidden_layer_units: int,
                 dropout: float, weights_uniforming: float, seq_sz: int):
        super(ModelLSTM, self).__init__(vocab_size, num_of_layers, hidden_layer_units,
                                        dropout, weights_uniforming, seq_sz)

    def get_rnn_type(self) -> type:
        return nn.LSTM

    def create_single_state(self, seq_size, hidden_size, device: str or int):
        return (torch.zeros(1, seq_size, hidden_size, device = device),
                torch.zeros(1, seq_size, hidden_size, device = device))

    @staticmethod
    def detach(states):
        states_detached: [] = []
        for (h, c) in states:
            states_detached.append((h.detach(), c.detach()))
        return states_detached
