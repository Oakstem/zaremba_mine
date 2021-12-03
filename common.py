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