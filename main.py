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
import common as cm


def main():
    params = OrderedDict(
        model_type=['ModelType.GRU'],
        lr=[0.001],
        batch_size=[20],
        dropout=[0],
        layers_num=[2],
        seq_sz=[20]
    )

    # if cm.IN_COLAB:
    nt2 = namedtuple('temp', "num_of_layers, hidden_layer_units, dropout weights_uniforming, batch_size,"
                             "sequence_length, learning_rate, total_epochs_num, first_epoch_modify_lr,"
                             " lr_decrease_factor, max_gradients_norm")
    args = nt2(2, 200, 0.5, 0.05, 20, params['seq_sz'][0], 1, 39, 6, 1.2, 5)
    # else:
    #     args: Namespace = parse_args()

    data: Data = DataGetter.get_data(args.batch_size, params['seq_sz'][0])
    traindata = PennDataset(data.train_dataset)
    testdata = PennDataset(data.test_dataset)

    train_w_RunManager(data, traindata, testdata, nll_loss, args, params=params, epochs=20)


main()
