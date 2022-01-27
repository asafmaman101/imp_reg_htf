import torch

from common.data.loaders.fast_tensor_dataloader import FastTensorDataLoader
from common.data.modules import DataModule
from tensor_factorization.datasets.tensor_sensing_dataset import TensorSensingDataset


class TensorSensingDataModule(DataModule):

    def __init__(self, dataset_path: str, num_train_samples: int = -1, num_test_samples: int = -1, batch_size: int = -1, shuffle_train: bool = False,
                 random_train_test_split: bool = False, load_dataset_to_device=None):
        """
        :param dataset_path: path to TensorSensingDataset file
        :param num_train_samples: number of samples to use for training, if < 0 will use the whole dataset
        :param num_test_samples: number of samples to use for testing, by default takes all samples except those used for trainingz
        :param batch_size: batch size, if <= 0 will use the size of the whole dataset
        :param shuffle_train: shuffle train samples each epoch
        :param random_train_test_split: randomize train test split
        :param load_dataset_to_device: device to load dataset to (default is CPU)
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.random_train_test_split = random_train_test_split
        self.load_dataset_to_device = load_dataset_to_device
        self.dataset = TensorSensingDataset.load(dataset_path)
        self.num_train_samples = num_train_samples if num_train_samples > 0 else len(self.dataset)

        num_samples_left = len(self.dataset) - self.num_train_samples
        self.num_test_samples = min(num_samples_left, num_test_samples) if num_test_samples > 0 else num_samples_left

        if self.load_dataset_to_device is not None:
            self.dataset.to_device(self.load_dataset_to_device)

        train_then_test_indices = torch.arange(len(self.dataset)) if not self.random_train_test_split else torch.randperm(len(self.dataset))
        self.train_indices = train_then_test_indices[:self.num_train_samples]
        self.test_indices = train_then_test_indices[max(self.num_train_samples, len(self.dataset) - self.num_test_samples):]

        self.train_inputs = self.dataset.inputs[self.train_indices]
        self.train_targets = self.dataset.targets[self.train_indices]

        self.test_inputs = self.dataset.inputs[self.test_indices]
        self.test_targets = self.dataset.targets[self.test_indices]

    def setup(self):
        # No setup required
        pass

    def train_dataloader(self) -> FastTensorDataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_indices)
        return FastTensorDataLoader(self.train_inputs, self.train_targets, batch_size=batch_size, shuffle=self.shuffle_train)

    def val_dataloader(self) -> FastTensorDataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.test_indices)
        return FastTensorDataLoader(self.test_inputs, self.test_targets, batch_size=batch_size, shuffle=False)

    def test_dataloader(self) -> FastTensorDataLoader:
        return self.val_dataloader()

    def train_and_test_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_indices) + len(self.test_indices)

        inputs = torch.cat([self.train_inputs, self.test_inputs]) if len(self.test_inputs) > 0 \
            else self.train_inputs
        targets = torch.cat([self.train_targets, self.test_targets]) if len(self.test_targets) > 0 else self.train_targets

        return FastTensorDataLoader(inputs, targets, batch_size=batch_size, shuffle=False)
