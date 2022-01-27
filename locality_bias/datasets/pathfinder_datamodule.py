import torch.utils.data
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms

from common.data.modules import DataModule
from locality_bias.datasets.pathfinder_dataset import PathfinderDataset


class PathfinderDataModule(DataModule):
    TRAIN_PIXEL_MEAN = {
        6: 0.004644281230866909,
        9: 0.00523123936727643,
        14: 0.006013816688209772
    }
    TRAIN_PIXEL_STD = {
        6: 0.042085420340299606,
        9: 0.0443103164434433,
        14: 0.04708821699023247
    }

    def __init__(self, dataset_path: str, num_train_samples: int = -1, num_val_samples: int = -1,
                 num_test_samples: int = -1,
                 batch_size: int = 32, fit_train_and_test: bool = False, split_random_state: int = -1):
        super().__init__()
        self.fit_train_and_test = fit_train_and_test
        self.split_random_state = split_random_state

        train_dataset = PathfinderDataset(dataset_path=dataset_path, split='train')
        test_dataset = PathfinderDataset(dataset_path=dataset_path, split='test')

        try:
            val_dataset = PathfinderDataset(dataset_path=dataset_path, split='test')
        except KeyError:
            val_dataset = None

        transform = torchvision.transforms.Compose([
            transforms.Normalize(mean=(train_dataset.meta_attributes['train_mean'].item(),),
                                 std=(train_dataset.meta_attributes['train_std'].item(),)),
        ])

        self.input_dims = tuple(train_dataset[0][0].shape)
        self.num_classes = len(torch.unique(train_dataset.labels))

        if val_dataset is None:
            train_dataset.transform = test_dataset.transform = transform

            self.num_val_samples = max(num_val_samples, 0)
            self.num_train_samples = len(train_dataset) - self.num_val_samples if num_train_samples < 0 else num_train_samples

            self.train_dataset, self.val_dataset = self.__get_train_val_datasets(train_dataset, self.num_train_samples,
                                                                                 self.num_val_samples)

        else:
            train_dataset.transform = val_dataset.transform = test_dataset.transform = transform

            self.num_train_samples = min(num_train_samples, len(train_dataset)) if num_train_samples > 0 else len(train_dataset)
            self.num_val_samples = min(num_val_samples, len(val_dataset)) if num_val_samples > 0 else len(val_dataset)

            self.train_dataset = train_dataset
            if self.num_train_samples < len(train_dataset):
                self.train_dataset = self.__subsample_train_dataset(train_dataset, self.num_train_samples)

            self.val_dataset = val_dataset
            if self.num_val_samples < len(val_dataset):
                self.val_dataset = self.__subsample_train_dataset(val_dataset, self.num_val_samples)

        self.batch_size = batch_size if batch_size > 0 else len(self.train_dataset)

        self.num_test_samples = min(num_test_samples, len(test_dataset)) if num_test_samples > 0 else len(test_dataset)

        self.test_dataset = test_dataset
        if self.num_test_samples < len(test_dataset):
            self.test_dataset = self.__subsample_test_dataset(test_dataset, self.num_test_samples)


    def __get_train_val_datasets(self, train_dataset: PathfinderDataset, num_train_samples: int, num_val_samples: int):
        if num_train_samples == len(train_dataset) and num_val_samples == 0:
            return train_dataset, torch.utils.data.Subset(train_dataset, indices=[])

        train_indices, val_indices = train_test_split(torch.arange(len(train_dataset)),
                                                      train_size=num_train_samples,
                                                      test_size=num_val_samples if num_val_samples > 0 else None,
                                                      stratify=train_dataset.labels,
                                                      random_state=self.split_random_state if self.split_random_state > 0 else None)

        train_split_dataset = torch.utils.data.Subset(train_dataset, indices=train_indices)
        val_split_dataset = torch.utils.data.Subset(train_dataset, indices=val_indices) if num_val_samples > 0 \
            else torch.utils.data.Subset(train_dataset, indices=[])
        return train_split_dataset, val_split_dataset

    def __subsample_test_dataset(self, test_dataset: PathfinderDataset, num_samples: int):
        _, test_indices = train_test_split(torch.arange(len(test_dataset)), test_size=num_samples,
                                           stratify=test_dataset.labels,
                                           random_state=self.split_random_state if self.split_random_state > 0 else None)
        return torch.utils.data.Subset(test_dataset, indices=test_indices)

    def __subsample_train_dataset(self, train_dataset: PathfinderDataset, num_samples: int):
        train_indices, _ = train_test_split(torch.arange(len(train_dataset)), train_size=num_samples,
                                           stratify=train_dataset.labels,
                                           random_state=self.split_random_state if self.split_random_state > 0 else None)
        return torch.utils.data.Subset(train_dataset, indices=train_indices)

    def setup(self):
        pass

    def train_dataloader(self):
        if self.fit_train_and_test:
            return torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset]),
                                               batch_size=self.batch_size, shuffle=True)

        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
