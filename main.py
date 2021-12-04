from training.nll_loss import nll_loss
from board_wrapper import train_w_RunManager
from collections import OrderedDict


def main():
    params = OrderedDict(
        model_type=['ModelType.GRU'],
        lr=[0.001],
        batch_size=[10],
        dropout=[0],
        seq_sz=[30],
        w_decay=[1e-4],
        grad_clip=[1],
        shuffle=[True, False]
    )

    const_args = OrderedDict(num_of_layers=2,
                             hidden_layer_units=200,
                             weights_uniforming=0.05,
                             max_gradients_norm=5,
                             vocab_sz=10000,
                             layers_num=2)

    train_w_RunManager(nll_loss, const_args, params=params, epochs=20)


main()
