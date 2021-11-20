from collections  import OrderedDict

# Params for RunManager
params = OrderedDict(
        lr=[.001, 0.01],
        batch_size=[20],
        shuffle=[False],
    )

# Paths:
LOG_DIR = '.runs/'
Model_DIR = '.saved_models/'