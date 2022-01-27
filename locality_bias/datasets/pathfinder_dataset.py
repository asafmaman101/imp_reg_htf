from __future__ import print_function

import os

import torch
import torch.utils.data as data
from torchvision import transforms


class PathfinderDataset(data.Dataset):

    def __init__(self, dataset_path:str, split: str = 'train', transform=None):
        loaded_file = torch.load(dataset_path)
        if split in loaded_file:
            data = loaded_file[split]
        else:
            raise KeyError

        self.meta_attributes = loaded_file['meta_attributes']

        self.transform = transform

        self.images = data['images']
        self.labels = data['labels']

    def __getitem__(self, index):
        img = self.images[index].to(torch.float32)
        target = self.labels[index].to(torch.long).squeeze()

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.images)
