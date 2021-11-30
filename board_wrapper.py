
"""# RunBuilder & RunManager"""
# A Slightly modified version of Twds's article:
# https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582

# import modules to build RunBuilder and RunManager helper classes
import common as cm
from collections  import OrderedDict
from collections import namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter # TensorBoard 
import time
import pandas as pd
import json
from IPython.display import clear_output
import torch
import torchvision as tv
from argparse import Namespace
from model.model_base import ModelBase
from model.model_getter import get_model
from training.perplexity import perplexity
import numpy as np


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
    self.epoch_num_correct = 0
    self.epoch_start_time = None

    # tracking every run count, run data, hyper-params used, time
    self.run_params = None
    self.run_count = run_no
    self.run_data = []
    self.run_start_time = None

    # record model, loader and TensorBoard 
    self.network = None
    self.loader = None
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
    # self.tb = SummaryWriter(log_dir=LOG_DIR, comment=f'-{run}')
    self.tb = SummaryWriter(log_dir=f'{cm.LOG_DIR}-{run}')

    x, y = next(iter(self.loader))
    init_func = getattr(network, "states_init", None)
    # if callable(init_func):
    #   states = self.network.state_init(self.loader.batch_size)

    if self.image_data:
      grid = tv.utils.make_grid(x)
      self.tb.add_image('images', grid)
    # self.tb.add_graph(self.network, (x,states))

  # when run ends, close TensorBoard, zero epoch count
  def end_run(self):
    self.tb.close()
    self.epoch_count = 0

  # zero epoch count, loss, accuracy, 
  def begin_epoch(self):
    self.epoch_start_time = time.time()

    self.epoch_count += 1
    self.epoch_loss = 0
    self.epoch_num_correct = 0
    self.test_epoch_loss = 0
    self.test_epoch_num_correct = 0

  # 
  def end_epoch(self, network, device):
    # calculate epoch duration and run duration(accumulate)
    epoch_duration = time.time() - self.epoch_start_time
    run_duration = time.time() - self.run_start_time

    # record epoch loss and accuracy
    no_samples_in_batch = self.loader.dataset[0][0].shape[0]*self.loader.dataset[0][0].shape[1]
    self.epoch_loss = self.epoch_loss/len(self.loader.dataset)
    self.test_epoch_loss = self.test_epoch_loss/len(self.test_loader.dataset)
    accuracy = 100*self.epoch_num_correct / (len(self.loader.dataset)*no_samples_in_batch)
    test_accuracy = 100*self.test_epoch_num_correct / (len(self.test_loader.dataset)*no_samples_in_batch)

    train_perplexity = np.exp(self.epoch_loss)
    test_perplexity = np.exp(self.test_epoch_loss)

    # Record epoch loss and accuracy to TensorBoard
    self.tb.add_scalar('Loss/train', self.epoch_loss, self.epoch_count)
    self.tb.add_scalar('Loss/test', self.test_epoch_loss, self.epoch_count)
    # self.tb.add_scalar('Accuracy/train', accuracy, self.epoch_count)
    # self.tb.add_scalar('Accuracy/test', test_accuracy, self.epoch_count)
    self.tb.add_scalar('Perplexity/train', train_perplexity, self.epoch_count)
    self.tb.add_scalar('Perplexity/test', test_perplexity, self.epoch_count)

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
    results["epoch"] = self.epoch_count
    results["train/loss"] = self.epoch_loss
    results["test/loss"] = self.test_epoch_loss
    results["train/perplexity"] = train_perplexity
    results["test/perplexity"] = test_perplexity
    results["epoch duration"] = epoch_duration
    results["run duration"] = run_duration

    # Record hyper-params into 'results'
    for k,v in self.run_params._asdict().items(): results[k] = v
    self.run_data.append(results)
    df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')

    if cm.IN_COLAB:
      # display epoch information and show progress
      clear_output(wait=True)
      display(df)

  # accumulate loss of batch into entire epoch loss
  def track_loss(self, loss, train):

    if train==1:
      self.epoch_loss += loss.item()
    else:
      self.test_epoch_loss += loss.item()

  # accumulate number of corrects of batch into entire epoch num_correct
  def track_num_correct(self, preds, y, train):
    if train==1:
      self.epoch_num_correct += self._get_num_correct(preds, y)
    else:
      self.test_epoch_num_correct += self._get_num_correct(preds, y)

  @torch.no_grad()
  def _get_num_correct(self, preds, y):
    amax = preds.argmax(dim=1)
    return amax.eq(y.view(amax.shape)).sum().item()
  
  # save end results of all runs into csv, json for further analysis
  def save(self, fileName, df):
    if not df.empty:
      res_df = df.append(pd.DataFrame.from_dict(self.run_data, orient = 'columns'))
    else:
      res_df = pd.DataFrame.from_dict(
          self.run_data,
          orient = 'columns')
    res_df.to_csv(f'results/{fileName}.csv')

    with open(f'results/{fileName}.json', 'w', encoding='utf-8') as f:
      json.dump(self.run_data, f, ensure_ascii=False, indent=4)

import asyncio
import time


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def background_train(i: int, run, data, train_data, test_data, criterion, args: Namespace,
                       params=cm.params, epochs=5):
  m = RunManager(image=False)
  # if params changes, following line of code should reflect the changes too
  try:
    network = torch.load(f'results/{run}.model')
    df = pd.read_csv(f'results/{run}.csv', index_col=0)
    epoch_start = df['epoch'].iloc[-1]
    print(f'Run:{i} loaded a previous model and continue its training from epoch:{epoch_start}')
  except:
    df = pd.DataFrame()
    network: ModelBase = get_model(run.model_type, data.vocabulary_size, run.dropout,
                                   run.layers_num, args.hidden_layer_units,
                                   args.weights_uniforming, args.batch_size)
    epoch_start = 0
  m = RunManager(epoch_count=epoch_start, run_no=i)
  loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
  # Setting a different optimizer for the Embedding & all other model params

  optimizer = torch.optim.Adam(list(network.parameters())[1:], lr=run.lr)  # other model params
  optimizerE = torch.optim.SparseAdam([list(network.parameters())[0]], lr=0.1)  # embedding param

  m.begin_run(run, network, loader, testloader)
  for epoch in range(epoch_start, epochs):
    m.begin_epoch()
    network.train()
    states = network.state_init()
    device: str or int = next(network.parameters()).device
    btch_cnt = 0
    # Run a batch in train mode
    cnt = 0
    for batch in loader:
      btch_cnt += 1
      if cnt % 50 == 0:
        print(f"Run: {i}, Batch No.{cnt}/{len(loader)}")
      cnt += 1
      x = batch[0].squeeze()
      y = batch[1].squeeze()

      x = x.to(device)
      y = y.to(device)
      # network.zero_grad()
      optimizer.zero_grad()
      optimizerE.zero_grad()

      states = network.detach(states)
      scores, states = network(x, states)
      loss = criterion(scores, y)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_gradients_norm)
      optimizer.step()
      optimizerE.step()

      m.track_loss(loss / network.batch_sz, train=1)
      # m.track_num_correct(scores, y, train=1)   Using Perplexity instead of Accuracy measurement

    print(f'Epoch No:{epoch}')
    print("Loss:{0:.2f}".format(m.epoch_loss / len(loader.dataset)))

    # Same run for Test only without backprop
    torch.no_grad()
    network.eval()
    states = network.state_init()
    for batch in testloader:
      x = batch[0].squeeze()
      y = batch[1].squeeze()
      preds, states = network(x, states)
      # loss = F.cross_entropy(preds, y)
      loss = criterion(preds, y)
      m.track_loss(loss / network.batch_sz, train=0)
      m.track_num_correct(preds, y, train=0)

    m.end_epoch(network, device)
    if epoch % 2 == 0:
      torch.save(network, f'results/{run}.model')
  m.end_run()
  torch.save(network, f'results/{run}.model')
    # when run is done, save results to files
  m.save(f'{run}', df)

def train_w_RunManager(data, train_data, test_data, criterion, args: Namespace,
                       params=cm.params, epochs=5):

    # create array of dataframes for every run
    # dfs = [pd.DataFrame() for i in range(len(RunBuilder.get_runs(params)))]

    for i, run in enumerate(RunBuilder.get_runs(params)):
      background_train(i, run, data, train_data, test_data, criterion, args, params, epochs)



