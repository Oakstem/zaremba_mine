"""# RunBuilder & RunManager"""
# A Slightly modified version of RunManager for TB logging from TWDs's article:
# https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582

# import modules to build RunBuilder and RunManager helper classes
import common as cm
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
import time
import pandas as pd
import json
from IPython.display import clear_output
import torch
import torch.nn as nn
from model.model_base import ModelBase
from model.model_getter import get_model
import numpy as np
import common as cm
from data.data import Data
from data.data_getter import DataGetter


# Read in the hyper-parameters and return a Run namedtuple containing all the
# combinations of hyper-parameters
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


# Helper class, help track loss, accuracy, epoch time, run time,
# hyper-parameters etc. Also record to TensorBoard and write into csv, json
class RunManager():
    def __init__(self, image=False, epoch_count=0, run_no=0):

        # tracking every epoch count, loss, accuracy, time
        self.epoch_count = epoch_count
        self.epoch_loss = 0
        self.test_epoch_loss = 0
        self.min_loss = 100
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        self.train_perplexity = 0
        self.test_perplexity = 0

        # tracking every run count, run data, hyper-params used, time
        self.run_params = None
        self.run_count = run_no
        self.run_data = []
        self.run_start_time = None

        # record model, loader and TensorBoard
        self.network = None
        self.loader = None
        self.test_loader = None
        self.tb = None

        # set True if the module gets image input
        self.image_data = image

    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard
    def begin_run(self, run, network, loader, test_loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.test_loader = test_loader
        self.tb = SummaryWriter(log_dir=f'{cm.LOG_DIR}-{run}')
        self.min_loss = 100

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self, i: int):
        self.tb.close()
        self.epoch_count = 0
        print(f"Run No.{i} ended")

    # zero epoch count, loss, accuracy,
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.test_epoch_loss = 0

    #
    def end_epoch(self, network, device):
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        self.epoch_loss = self.epoch_loss / len(self.loader)
        self.test_epoch_loss = self.test_epoch_loss / len(self.test_loader)
        self.train_perplexity = np.exp(self.epoch_loss)
        self.test_perplexity = np.exp(self.test_epoch_loss)

        # Record epoch loss and accuracy to TensorBoard
        self.tb.add_scalar('Loss/train', self.epoch_loss, self.epoch_count)
        self.tb.add_scalar('Loss/test', self.test_epoch_loss, self.epoch_count)
        self.tb.add_scalar('Perplexity/train', self.train_perplexity, self.epoch_count)
        self.tb.add_scalar('Perplexity/test', self.test_perplexity, self.epoch_count)

        # Record params to TensorBoard
        for name, param in self.network.named_parameters():
            if param.grad.is_sparse:
                grad_for_hist = param.grad.to_dense()
            else:
                grad_for_hist = param.grad
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', grad_for_hist, self.epoch_count)

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = int(self.epoch_count)
        results["train/loss"] = self.epoch_loss
        results["test/loss"] = self.test_epoch_loss
        results["train/perplexity"] = int(self.train_perplexity)
        results["test/perplexity"] = int(self.test_perplexity)
        results["epoch duration"] = int(epoch_duration)
        results["run duration"] = int(run_duration)

        # Record hyper-params into 'results'
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        if cm.IN_COLAB:
            # display epoch information and show progress
            clear_output(wait=True)
            display(df)
        if cm.DEBUG:
            print(f'Epoch No:{self.epoch_count}, TrainLoss:{self.epoch_loss:.2f}, TestLoss{self.test_epoch_loss:.2f}')

    def track_loss(self, loss, train):

        if train == 1:
            self.epoch_loss += loss.item()
        else:
            self.test_epoch_loss += loss.item()

    @torch.no_grad()
    def _get_num_correct(self, preds, y):
        amax = preds.argmax(dim=1)
        return amax.eq(y.view(amax.shape)).sum().item()

    # save end results of all runs into csv, json for further analysis
    def save(self, fileName, df, network, run):
        if not df.empty:
            res_df = df.append(pd.DataFrame.from_dict(self.run_data, orient='columns'))
        else:
            res_df = pd.DataFrame.from_dict(
                self.run_data,
                orient='columns')
        res_df.to_csv(f'results/{fileName}.csv')

        if self.test_epoch_loss < self.min_loss:
            torch.save(network, f'results/{run}.model')
            self.min_loss = self.test_epoch_loss


def load_prev(run: namedtuple, i: int, args: dict):
    # Try loading a previous save of the model
    try:
        network = torch.load(f'results/{run}.model')
        df = pd.read_csv(f'results/{run}.csv', index_col=0)
        epoch_start = df['epoch'].iloc[-1]
        print(f'Run:{i} loaded a previous model and continue its training from epoch:{epoch_start}')
    except:
        df = pd.DataFrame()
        network: ModelBase = get_model(run.model_type, args['vocab_sz'], run.dropout,
                                       args['layers_num'], args['hidden_layer_units'],
                                       args['weights_uniforming'], run.batch_size)
        epoch_start = 0
    return network, df, epoch_start


def embedding_weight_check(network, i: int):
    if network.embedding.weight.isnan().count_nonzero() > 0 or \
            network.embedding.weight.grad.isnan().count_nonzero() > 0:
        print(f'Run:{i}, embedding layer exploded once again :(')


def get_xy_from_batch(batch, device):
    x = batch[0].squeeze()
    y = batch[1].squeeze()
    x = x.to(device)
    y = y.to(device)
    return x, y


def status_print(btch_cnt: int, i: int, data: Data):
    if cm.DEBUG and btch_cnt % 100 == 0:
        print(f"Run: {i}, Batch No.{btch_cnt}/{len(data.train_loader)}")


def train_one_epoch(network: nn.Module, device: int or str,
                    data: Data, criterion, i: int,
                    run: namedtuple=None, m: RunManager=None,
                    optimizer: torch.optim.Optimizer=None, no_grad=False):
    if not m:
        m = RunManager(epoch_count=0, run_no=0)
    m.begin_epoch()
    if no_grad:
        torch.no_grad()
        network.eval()
    else:
        network.train()
    network.to(device)
    states = network.state_init(device)
    btch_cnt = 0

    for batch in data.train_loader:
        btch_cnt += 1
        status_print(btch_cnt, i, data)
        x, y = get_xy_from_batch(batch, device)
        if not no_grad:
            optimizer.zero_grad()
        states = network.detach(states)
        scores, states = network(x, states)
        loss = criterion(scores, y)
        if not no_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), run.grad_clip)
            optimizer.step()
            # Embedding layer weights check for possible explode
            embedding_weight_check(network, i)
        # Track losses for TB
        m.track_loss(loss/x.shape[1], train=1)
    epoch_loss = m.epoch_loss / len(data.train_loader)
    return epoch_loss, np.exp(epoch_loss)

def test_one_epoch(network: nn.Module, device: int or str,
                    data: Data, criterion, m: RunManager=None, valid=True):
    # Same run for Test only [without backprop]
    ###########################################################################
    if not m:
        m = RunManager(epoch_count=0, run_no=0)
    torch.no_grad()
    network.eval()
    states = network.state_init(device)
    loader = data.validation_loader if valid else data.test_loader
    for batch in loader:
        x, y = get_xy_from_batch(batch, device)
        scores, states = network(x, states)
        loss = criterion(scores, y)
        # Track losses for TB
        m.track_loss(loss/x.shape[1], train=0)
    epoch_loss = m.test_epoch_loss/len(loader)
    return epoch_loss, np.exp(epoch_loss)


def background_train(i: int, run: namedtuple, criterion, args: dict, epochs: int, rawdata: list):
    device = cm.net_device
    data = DataGetter.get_data(run.batch_size, run.seq_sz, data=rawdata, device=device, shuffle=args['shuffle'])
    network, df, epoch_start = load_prev(run, i, args)

    # RunManager responsible for logging results to file / TB dashboard
    m = RunManager(epoch_count=epoch_start, run_no=i)
    optimizer = torch.optim.Adam(network.parameters(), lr=run.lr, weight_decay=run.w_decay)  # other model params
    m.begin_run(run, network, data.train_loader, data.validation_loader)

    for epoch in range(epoch_start, epochs):
        # Run a batch in train mode
        ###########################################################################
        train_one_epoch(network, device, data, criterion, i, run, m, optimizer)
        test_one_epoch(network, device, data, criterion, m)
        m.end_epoch(network, device)
        # Save results to csv & json files + Model
        m.save(f'{run}', df, network, run)
    m.end_run(i)


def train_w_RunManager(criterion, args: dict, epochs: int, params: dict):
    # Get raw data
    rawdata: list = DataGetter.data_init()
    # Loop of several different parameters train check
    for i, run in enumerate(RunBuilder.get_runs(params)):
        background_train(i, run, criterion, args, epochs, rawdata)
