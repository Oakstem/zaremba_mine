from model.model_base import ModelBase
from model.model_gru import ModelGRU
from model.model_lstm import ModelLSTM
from model.type import ModelType


def get_model(type: str, vocabulary_size: int, dropout: float,
              num_of_layers: int, hidden_layer_units: int,
              weights_uniforming: float, batch_sz: int) -> ModelBase:

    model: ModelBase = None
    if type == 'LSTM':
        model: ModelBase = ModelLSTM(vocabulary_size, num_of_layers, hidden_layer_units,
                                     dropout, weights_uniforming, batch_sz)
    else:
        model: ModelBase = ModelGRU(vocabulary_size, num_of_layers, hidden_layer_units,
                                    dropout, weights_uniforming, batch_sz)

    return model
