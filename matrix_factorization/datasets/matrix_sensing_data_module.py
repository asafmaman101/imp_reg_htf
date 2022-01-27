import torch.utils.data

from common.data.modules import DataModule
from matrix_factorization.datasets.matrix_sensing_dataset import MatrixSensingDataset


class MatrixSensingDataModule(DataModule):

    def __init__(self, dataset_path: str, num_train_samples: int, batch_size: int = -1, train_shuffle: bool = False):
        self.dataset_path = dataset_path
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.dataset = MatrixSensingDataset.load(dataset_path)
        self.train_dataset = MatrixSensingDataset(self.dataset.A[:num_train_samples], self.dataset.y[:num_train_samples], self.dataset.target_matrix)

    def setup(self):
        # No setup required
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.train_dataset)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=self.train_shuffle)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.dataset)
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        batch_size = self.batch_size if self.batch_size > 0 else len(self.dataset)
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
