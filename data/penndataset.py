import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class PennDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx][0]
        label = self.data[idx][1]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(label)
        return seq, label