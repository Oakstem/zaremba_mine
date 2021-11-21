from argparse import Namespace
from args_parser import parse_args
from data.data import Data
from data.data_getter import DataGetter
from data.penndataset import PennDataset
from model.model_base import ModelBase
from model.model_getter import get_model
from model.type import ModelType
from training.train import train
from training.nll_loss import nll_loss
# from torch.nn import NLLLoss
# import common as cm
from board_wrapper import train_w_RunManager
from collections  import OrderedDict
from collections  import namedtuple

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False



def main():
    params = OrderedDict(
        model_type=['ModelType.GRU', 'ModelType.LSTM'],
        lr=[.001],
        batch_size=[20],
        shuffle=[False],
        dropout=[0, 0.5],
    )

    if IN_COLAB:
        nt2 = namedtuple('temp', "num_of_layers, hidden_layer_units, dropout weights_uniforming, batch_size,"
                                 "sequence_length, learning_rate, total_epochs_num, first_epoch_modify_lr,"
                                 " lr_decrease_factor, max_gradients_norm")
        args = nt2(2, 200, 0.5, 0.05, 20, 35, 1, 39, 6, 1.2, 5)
    else:
        args: Namespace = parse_args()

    data: Data = DataGetter.get_data(args.batch_size, args.sequence_length)
    traindata = PennDataset(data.train_dataset)
    testdata = PennDataset(data.test_dataset)
    # model_gru_no_dropout: ModelBase = get_model(ModelType.GRU, data.vocabulary_size, 0,
    #                                             args.num_of_layers, args.hidden_layer_units,
    #                                             args.weights_uniforming, args.batch_size)
    # train_model("GRU No Dropout", model_gru_no_dropout, data, args)
    train_w_RunManager(data, traindata, testdata, nll_loss, args, params=params, epochs=10, IN_COLAB)
    # model_gru_dropout: ModelBase = get_model(ModelType.GRU, data.vocabulary_size, args.dropout,
    #                                          args.num_of_layers, args.hidden_layer_units,
    #                                          args.weights_uniforming)
    # train_model("GRU With Dropout", model_gru_dropout, data, args)
    #
    # model_lstm_no_dropout: ModelBase = get_model(ModelType.LSTM, data.vocabulary_size, 0,
    #                                              args.num_of_layers, args.hidden_layer_units,
    #                                              args.weights_uniforming)
    # train_model("LSTM No Dropout", model_lstm_no_dropout, data, args)
    #
    # model_lstm_dropout: ModelBase = get_model(ModelType.LSTM, data.vocabulary_size, args.dropout,
    #                                           args.num_of_layers, args.hidden_layer_units,
    #                                           args.weights_uniforming)
    # train_model("LSTM With Dropout", model_lstm_dropout, data, args)


def train_model(title: str, model: ModelBase, data: Data, args: Namespace):
    print("Model: " + title)
    print(model)

    model.to(args.device)

    total_epochs_num = args.total_epochs_num
    learning_rate = args.learning_rate
    first_epoch_modify_lr = args.first_epoch_modify_lr
    lr_decrease_factor = args.lr_decrease_factor
    device: str or int = args.device

    best_model: ModelBase = None
    validation_perplexities: [float] = None
    test_perplexities: [float] = None
    best_model, validation_perplexities, test_perplexities = \
        train(model, data, total_epochs_num, first_epoch_modify_lr, learning_rate, lr_decrease_factor,
              args.max_gradients_norm, device)

    print("Train Summary:")
    print("Validation :" + str(validation_perplexities))
    print("Test: " + str(test_perplexities))



main()
