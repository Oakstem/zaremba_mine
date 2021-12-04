
from data.data import Data
from data.data_getter import DataGetter
from data.penndataset import PennDataset
from training.nll_loss import nll_loss
from board_wrapper import train_w_RunManager
from collections  import OrderedDict
from collections  import namedtuple


def main():
    params = OrderedDict(
        model_type=['ModelType.GRU'],
        lr=[0.001],
        batch_size=[20],
        dropout=[0],
        layers_num=[2],
        seq_sz=[20],
        w_decay=[1e-6],
        grad_clip=[1, 5, 10]
    )

    args = OrderedDict(num_of_layers=2,
            hidden_layer_units=200,
            weights_uniforming=0.05,
            batch_sz=20,
            max_gradients_norm=5)

    data: Data = DataGetter.get_data(args['batch_sz'], params['seq_sz'][0])
    traindata = PennDataset(data.train_dataset)
    testdata = PennDataset(data.test_dataset)

    train_w_RunManager(data, traindata, testdata, nll_loss, args, params=params, epochs=20)


main()
