class Data:
    def __init__(self, train_dataset: [], validation_dataset: [], test_dataset: [],
                 vocabulary_size: int, batch_size: int):

        self.batch_size: int = batch_size

        self.vocabulary_size: int = vocabulary_size

        self.train_dataset: [] = train_dataset
        self.validation_dataset: [] = validation_dataset
        self.test_dataset: [] = test_dataset

