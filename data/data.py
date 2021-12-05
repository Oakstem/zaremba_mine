from .penndataset import PennDataset
import torch

class Data:
    def __init__(self, train_dataset: [], validation_dataset: [], test_dataset: [],
                 vocabulary_size: int, batch_size: int, shuffle=False):

        self.batch_size: int = batch_size
        self.vocabulary_size: int = vocabulary_size

        self.train_loader: [] = torch.utils.data.DataLoader\
            (PennDataset(train_dataset), batch_size=1, shuffle=shuffle)
        self.validation_loader: [] = torch.utils.data.DataLoader\
            (PennDataset(validation_dataset), batch_size=1, shuffle=shuffle)
        self.test_loader: [] = torch.utils.data.DataLoader\
            (PennDataset(test_dataset), batch_size=1, shuffle=shuffle)
