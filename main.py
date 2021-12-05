from training.nll_loss import nll_loss
from board_wrapper import train_w_RunManager
from collections import OrderedDict
from model.model_getter import load_model
from common import net_device
from board_wrapper import test_one_epoch
from data.data_getter import DataGetter

def main():
    params = OrderedDict(
        model_type=['GRU'],
        lr=[0.001],
        batch_size=[20],
        dropout=[0],
        seq_sz=[35],
        w_decay=[1e-4],
        grad_clip=[1]
    )

    const_args = OrderedDict(num_of_layers=2,
                             hidden_layer_units=200,
                             weights_uniforming=0.05,
                             vocab_sz=10000,
                             layers_num=2,
                             shuffle=False)

    train_w_RunManager(nll_loss, const_args, params=params, epochs=20)
    network_select = {1:'LSTM without dropout', 2: 'LSTM with dropout',
                      3: 'GRU without dropout', 4: 'GRU with dropout'}
    network = load_model(network_select[3], net_device)

    data = DataGetter.get_data\
        (params['batch_size'][0], params['seq_sz'][0], data=DataGetter.data_init(), device=net_device)
    loss, perplexity = test_one_epoch(network, net_device, data, nll_loss)
    print(f"Resulted loss:{loss:.2f}, Perplexity:{perplexity:.2f}")



main()
