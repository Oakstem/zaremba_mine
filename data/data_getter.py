import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from .data import Data
from .type import DatasetType


class DataGetter:
    @staticmethod
    def get_data(batch_size: int, sequence_length: int) -> Data:

        vocabulary: dict = DataGetter.get_vocabulary()
        vocabulary_size: int = len(vocabulary)

        train_dataset = DataGetter.get_dataset(DatasetType.TRAIN, vocabulary, batch_size, sequence_length)
        validation_dataset = DataGetter.get_dataset(DatasetType.VALIDATION, vocabulary, batch_size, sequence_length)
        test_dataset = DataGetter.get_dataset(DatasetType.TEST, vocabulary, batch_size, sequence_length)

        data: Data = Data(train_dataset, validation_dataset, test_dataset, vocabulary_size, batch_size)
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
    def get_dataset(dataset_type: DatasetType, vocabulary, batch_size, sequence_length):
        data: ndarray = DataGetter.data_init(dataset_type, vocabulary)

        data: Tensor = torch.tensor(data, dtype=torch.int64)
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
    def data_init(dataset_type: DatasetType, vocabulary):
        data_file_name: str = ""
        if dataset_type == DatasetType.TRAIN:
            data_file_name = "data/text/ptb.train.txt"
        elif dataset_type == DatasetType.VALIDATION:
            data_file_name = "data/text/ptb.valid.txt"
        else:
            data_file_name = "data/text/ptb.test.txt"

        with open(data_file_name) as f:
            file = f.read()
            words = file[1:].split(' ')

        data: [] = []
        for word in words:
            data.append(vocabulary[word])

        data_array = np.array(data).reshape(-1, 1)
        return data_array
