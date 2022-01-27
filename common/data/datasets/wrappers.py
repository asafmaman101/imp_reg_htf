import torch.utils.data


class InMemoryDataset(torch.utils.data.Dataset):
    """
    An in memory wrapper for an existing dataset. Loads all of its items into memory for quick access.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, transform=None):
        self.__dict__.update(dataset.__dict__)
        self.wrapped_dataset = dataset
        self.in_memory_data = self.__load_dataset()
        self.transform = transform if transform is not None else lambda x: x

    def __load_dataset(self):
        data = [None] * len(self.wrapped_dataset)
        for i in range(len(self.wrapped_dataset)):
            data[i] = self.wrapped_dataset[i]
        return data

    def __getitem__(self, index: int):
        return self.transform(self.in_memory_data[index])

    def __len__(self):
        return len(self.in_memory_data)


class TransformDatasetWrapper(torch.utils.data.Dataset):
    """
    Transform wrapper for a dataset. Adds a transformation for the output of the given dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, transform=None):
        self.__dict__.update(dataset.__dict__)
        self.wrapped_dataset = dataset
        self.wrapping_transform = transform if transform is not None else lambda x: x

    def __getitem__(self, index: int):
        return self.wrapping_transform(self.wrapped_dataset[index])

    def __len__(self):
        return len(self.wrapped_dataset)


class ReturnSampleIndicesDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for dataset to add to each returned sample its index in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.__dict__.update(dataset.__dict__)
        self.wrapped_dataset = dataset

    def __getitem__(self, index: int):
        return self.wrapped_dataset[index], index

    def __len__(self):
        return len(self.wrapped_dataset)
