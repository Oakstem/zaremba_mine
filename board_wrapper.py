
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
  def __init__(self, image=False):

    # tracking every epoch count, loss, accuracy, time
    self.epoch_count = 0
    self.epoch_loss = 0
    self.epoch_num_correct = 0
    self.epoch_start_time = None

    # tracking every run count, run data, hyper-params used, time
    self.run_params = None
    self.run_count = 0
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
  def end_epoch(self):
    # calculate epoch duration and run duration(accumulate)
    epoch_duration = time.time() - self.epoch_start_time
    run_duration = time.time() - self.run_start_time

    # record epoch loss and accuracy
    loss = self.epoch_loss / len(self.loader.dataset)
    test_loss = self.test_epoch_loss / len(self.test_loader.dataset)
    accuracy = 100*self.epoch_num_correct / len(self.loader.dataset)
    test_accuracy = 100*self.test_epoch_num_correct / len(self.test_loader.dataset)

    # Record epoch loss and accuracy to TensorBoard
    self.tb.add_scalars('Loss', {'Train':loss, 'Test':test_loss}, self.epoch_count)
    self.tb.add_scalars('Accuracy', {'Train':accuracy, 'Test':test_accuracy}, self.epoch_count)

    # Record params to TensorBoard
    for name, param in self.network.named_parameters():
      self.tb.add_histogram(name, param, self.epoch_count)
      self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
    
    # Write into 'results' (OrderedDict) for all run related data
    results = OrderedDict()
    results["run"] = self.run_count
    results["epoch"] = self.epoch_count
    results["loss"] = loss
    results["accuracy"] = accuracy
    results["epoch duration"] = epoch_duration
    results["run duration"] = run_duration

    # Record hyper-params into 'results'
    for k,v in self.run_params._asdict().items(): results[k] = v
    self.run_data.append(results)
    df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')

    # display epoch information and show progress
    clear_output(wait=True)
    # display(df)

  # accumulate loss of batch into entire epoch loss
  def track_loss(self, loss, train):
    # multiply batch size so variety of batch sizes can be compared
    if train==1:
      self.epoch_loss += loss.item() * self.loader.batch_size
    else:
      self.test_epoch_loss += loss.item() * self.test_loader.batch_size

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
  def save(self, fileName):

    pd.DataFrame.from_dict(
        self.run_data, 
        orient = 'columns',
    ).to_csv(f'{fileName}.csv')

    with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
      json.dump(self.run_data, f, ensure_ascii=False, indent=4)


def train_w_RunManager(network, train_data, test_data, criterion, args: Namespace,  params=cm.params, epochs=5):
    # put all hyper params into a OrderedDict, easily expandable

    m = RunManager(image=False)
    # get all runs from params using RunBuilder class
    for run in RunBuilder.get_runs(params):

        # if params changes, following line of code should reflect the changes too
        loader = torch.utils.data.DataLoader(train_data, batch_size = 1, shuffle=run.shuffle)
        testloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle=run.shuffle)
        optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)

        states = network.state_init()
        m.begin_run(run, network, loader, testloader)
        for epoch in range(epochs):

          m.begin_epoch()
          network.train()
          network.state_init()
          device: str or int = next(network.parameters()).device
          # Run a batch in train mode
          for batch in loader:
            if len(batch[0].shape) > 3:
              stop =1
            x = batch[0].squeeze()
            y = batch[1].squeeze()

            # preds, states = network(x, states)
            # loss = criterion(preds, y)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            x = x.to(device)
            y = y.to(device)
            network.zero_grad()
            states = network.detach(states)
            scores, states = network(x, states)
            loss = criterion(scores, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_gradients_norm)
            optimizer.step()

            m.track_loss(loss, train=1)
            m.track_num_correct(scores, y, train=1)

          print(f'Epoch No:{epoch} '
                f'Loss:{m.epoch_loss}')
          # Same run for Test only without backprop
          network.eval()
          states = network.state_init()
          for batch in testloader:
            x = batch[0].squeeze()
            y = batch[1].squeeze()
            preds, states = network(x,states)
            # loss = F.cross_entropy(preds, y)
            loss = criterion(preds, y)
            m.track_loss(loss, train=0)
            m.track_num_correct(preds, y, train=0)

          m.end_epoch()
        m.end_run()
        torch.save(network, f'saved_models/{network.name}-{run}.model')


    # when all runs are done, save results to files
    m.save(cm.LOG_DIR + '/results')


"""# Train with RunManager"""

