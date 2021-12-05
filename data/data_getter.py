import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from .penndataset import PennDataset
from .data import Data
from .type import DatasetType

class DataGetter:

    @staticmethod
    def get_data(batch_size: int, sequence_length: int, device: str or int,
                 data: list, shuffle=False) -> Data:

        vocabulary: dict = DataGetter.get_vocabulary()
        vocabulary_size: int = len(vocabulary)
        train = data[0][:,:2000]
        valid = data[1]
        test = data[2]

        train_dataset = DataGetter.get_dataset(train, sequence_length, batch_size, device)
        validation_dataset = DataGetter.get_dataset(valid, sequence_length, batch_size, device)
        test_dataset = DataGetter.get_dataset(test, sequence_length, batch_size, device)

        data: Data = Data(train_dataset, validation_dataset, test_dataset, vocabulary_size, batch_size, shuffle)
        return data

    @staticmethod
    def get_vocabulary():
        import os
        cwd = os.getcwd()
        with open("data/text/ptb.train.txt") as f:
            file = f.read()
            train_words = file[1:].split(' ')

        words = sorted(set(train_words))
        vocabulary: {} = {}
        for i, c in enumerate(words):
            vocabulary[c] = i

        return vocabulary

    # Batches the data with [T, B] dimensionality.
    @staticmethod
    def get_dataset(data, sequence_length, batch_size, device):
        data: Tensor = torch.tensor(data, dtype=torch.int64).to(device).squeeze()
        num_batches: int = data.size()[0] // batch_size

        data: Tensor = data[:num_batches * batch_size]
        data: Tensor = data.view(batch_size, -1)

        dataset = []
        for i in range(0, data.size()[1] - 1, sequence_length):
            seqlen: int = int(np.min([sequence_length, data.size()[1] - 1 - i]))

            if seqlen < data.size()[1] - 1 - i:
                x: Tensor = data[:, i:i + seqlen]
                x: Tensor = x.transpose(1, 0)

                y: Tensor = data[:, i + 1:i + seqlen + 1]
                y: Tensor = y.transpose(1, 0)

                dataset.append((x, y))

        return dataset

    @staticmethod
    def data_init():
        all_data_arrays = []
        vocabulary: dict = DataGetter.get_vocabulary()
        train_file_name = "data/text/ptb.train.txt"
        valid_file_name = "data/text/ptb.valid.txt"
        test_file_name = "data/text/ptb.test.txt"
        data_names = (train_file_name, valid_file_name, test_file_name)
        for filename in data_names:
            with open(filename) as f:
                file = f.read()
                words = file[1:].split(' ')

            data: [] = []
            for word in words:
                data.append(vocabulary[word])
            all_data_arrays.append(np.array(data).reshape(1, -1))

        return all_data_arrays
