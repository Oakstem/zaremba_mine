from collections  import OrderedDict
from model.type import ModelType
# Params for RunManager
# params = OrderedDict(
#         lr=[.001, 0.01],
#         batch_size=[20],
#         shuffle=[False],
#         dropout=[0, 0.5],
#         model_name=[ModelType.GRU, ModelType.LSTM]
#     )
import torch
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if torch.cuda.is_available():
  net_device = torch.device('cuda')
else:
  print("Model will be training on the CPU.\n")
  net_device = torch.device('cpu')

# Paths:
LOG_DIR = 'runs/'
Model_DIR = 'saved_models/'