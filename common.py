from collections  import OrderedDict
from model.type import ModelType
# Params for RunManager
params = OrderedDict(
        lr=[.001, 0.01],
        batch_size=[20],
        shuffle=[False],
        dropout=[0, 0.5],
        model_name=[ModelType.GRU, ModelType.LSTM]
    )

# Paths:
LOG_DIR = '.runs/'
Model_DIR = '.saved_models/'