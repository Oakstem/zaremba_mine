import abc

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import RNNBase


class ModelBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_size: int, num_of_layers: int, hidden_layer_units: int,
                 dropout: float, weights_uniforming: float, batch_sz: int):
        super().__init__()
        self.vocabsz = vocab_size
        self.hiddenu = hidden_layer_units
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_layer_units, sparse=False)
        self.batchnorm: nn.BatchNorm1d = nn.BatchNorm1d(batch_sz, affine=False)

        self.rnns: nn.ModuleList = nn.ModuleList()
        rnns_type: type = self.get_rnn_type()
        for i in range(num_of_layers):
            rnn: RNNBase = rnns_type(hidden_layer_units, hidden_layer_units)
            self.rnns.append(rnn)

        self.fc: nn.Linear = nn.Linear(hidden_layer_units, vocab_size)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.batch_sz = batch_sz

        for param in self.parameters():
            nn.init.uniform_(param, -weights_uniforming, weights_uniforming)

    def get_rnn_type(self) -> type:
        return None
            
    def state_init(self, device: str or int):
        states: [] = []
        for rnn in self.rnns:
            state = self.create_single_state(self.batch_sz, rnn.hidden_size, device)
            states.append(state)
        return states

    def create_single_state(self, seq_size, hidden_size, device: str or int):
        return None

    @staticmethod
    def detach(states):
        pass

    def forward(self, x: Tensor, states):
        x: Tensor = self.embedding(x)
        x: Tensor = self.batchnorm(x)
        x: Tensor = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x, states[i])
            x: Tensor = self.batchnorm(x)
            x: Tensor = self.dropout(x)
        x = x.view(x.size()[0] * x.size()[1], -1)
        scores: Tensor = self.fc(x)

        return scores, states