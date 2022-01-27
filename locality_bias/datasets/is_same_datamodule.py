from typing import Union

import torch.utils.data

from common.data.loaders.fast_tensor_dataloader import FastTensorDataLoader
from common.data.modules import DataModule
from locality_bias.datasets.is_same_dataset import IsSameDataset


class IsSameDataModule(DataModule):

    def __init__(self, train_dataset_path: str, test_dataset_path: str, batch_size: int, num_train_samples: int = -1, num_test_samples: int = -1,
                 distance: int = 0, input_width: int = 224, input_height: int = 224, preload_data_to_memory: bool = False,
                 fit_train_and_test: bool = False):
        """
        :param train_dataset_path: path to the train IsSameClassDataset file.
        :param train_dataset_path: path to the test IsSameClassDataset file.
        :param batch_size: batch size, if <= 0 will use the size of the whole dataset.
        :param num_train_samples: number of train samples to use (if < 0 will use the whole dataset).
        :param num_test_samples: number of test samples to use (if < 0 will use the whole dataset).
        :param distance: Horizontal space in pixels between the two images.
        :param input_width: Total width of input image to use.
        :param input_height: Total height of input image to use.
        :param preload_data_to_memory: Preprocess and load all data to memory.
        :param fit_train_and_test:If True will use both training and test data as the training set.
        """
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.distance = distance
        self.input_width = input_width
        self.input_height = input_height
        self.preload_data_to_memory = preload_data_to_memory
        self.fit_train_and_test = fit_train_and_test

        self.train_dataset = IsSameDataset.load(self.train_dataset_path, num_samples=num_train_samples, distance=self.distance,
                                                input_width=self.input_width, input_height=self.input_height,
                                                preload_data_to_memory=self.preload_data_to_memory)
        self.test_dataset = IsSameDataset.load(self.test_dataset_path, num_samples=num_test_samples, distance=self.distance,
                                               input_width=self.input_width, input_height=self.input_height,
                                               preload_data_to_memory=self.preload_data_to_memory)

    def __get_preloaded_train_inputs_and_labels(self):
        train_inputs = self.train_dataset.inputs_cache
        train_labels = self.train_dataset.labels
        if not self.fit_train_and_test:
            return train_inputs, train_labels

        train_and_test_inputs = torch.cat([train_inputs, self.test_dataset.inputs_cache])
        train_and_test_labels = torch.cat([train_labels, self.test_dataset.labels])
        return train_and_test_inputs, train_and_test_labels

    def setup(self):
        # No setup required
        pass

    def train_dataloader(self) -> Union[FastTensorDataLoader, torch.utils.data.DataLoader]:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_dataset)

        if self.preload_data_to_memory:
            train_inputs, train_labels = self.__get_preloaded_train_inputs_and_labels()
            return FastTensorDataLoader(train_inputs, train_labels, batch_size=batch_size, shuffle=True)

        if self.fit_train_and_test:
            return torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset]),
                                               batch_size=batch_size, shuffle=True)

        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def val_dataloader(self) -> Union[FastTensorDataLoader, torch.utils.data.DataLoader]:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.test_dataset)

        if self.preload_data_to_memory:
            return FastTensorDataLoader(self.test_dataset.inputs_cache, self.test_dataset.labels, batch_size=batch_size, shuffle=True)

        return torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def test_dataloader(self) -> FastTensorDataLoader:
        return self.val_dataloader()
