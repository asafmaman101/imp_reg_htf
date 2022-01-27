import torch
import torch.utils.data

from common.data.modules.torchvision_datamodule import TorchvisionDataModule


class IsSameDataset(torch.utils.data.Dataset):

    def __init__(self, train: bool, image_pairs_indices: torch.tensor, labels: torch.tensor, num_samples: int = -1,
                 distance: int = 0, input_width: int = 224, input_height: int = 224, preload_data_to_memory: bool = False):
        """
        :param train: If True the dataset is created based on the original train dataset, otherwise it is based on the original test dataset.
        :param image_pairs_indices: Tensor of dimension (*, 2) containing pairs of indices of images in the dataset.
        :param num_samples: Number of samples to use (if < 0 will use all samples).
        :param distance: Horizontal space in pixels between the two images.
        :param input_width: Total width of input image to use.
        :param input_height: Total height of input image to use.
        :param preload_data_to_memory: Preprocess and load all data to memory.
        """
        self.train = train
        self.num_samples = min(num_samples, image_pairs_indices.shape[0]) if num_samples > 0 else image_pairs_indices.shape[0]
        self.image_pairs_indices = image_pairs_indices[:self.num_samples]
        self.labels = labels[:self.num_samples]
        self.distance = distance
        self.input_width = input_width
        self.input_height = input_height
        self.preload_data_to_memory = preload_data_to_memory

        self.original_datamodule = TorchvisionDataModule("cifar10")
        self.original_datamodule.setup()
        self.original_dataset = self.original_datamodule.train_dataset if self.train else self.original_datamodule.test_dataset

        if self.preload_data_to_memory:
            self.inputs_cache = self.__create_all_inputs()

    def __create_is_same_class_input(self, index: int):
        index_pair = self.image_pairs_indices[index]
        first_image, _ = self.original_dataset[index_pair[0]]
        second_image, _ = self.original_dataset[index_pair[1]]

        f_h_offset = self.input_height // 2 - first_image.shape[1] // 2
        f_w_offset = self.input_width // 2 - first_image.shape[2] - self.distance // 2
        s_h_offset = f_h_offset
        s_w_offset = f_w_offset + first_image.shape[1] + self.distance

        input_image = torch.zeros(first_image.shape[0], self.input_height, self.input_width, device=first_image.device, dtype=first_image.dtype)
        input_image[:, f_h_offset: f_h_offset + first_image.shape[1], f_w_offset: f_w_offset + first_image.shape[2]] = first_image
        input_image[:, s_h_offset: s_h_offset + second_image.shape[1], s_w_offset: s_w_offset + second_image.shape[2]] = second_image

        return input_image

    def __create_all_inputs(self):
        inputs = []
        for i in range(self.image_pairs_indices.shape[0]):
            input_image = self.__create_is_same_class_input(i)
            inputs.append(input_image)

        return torch.stack(inputs)

    def __getitem__(self, index: int):
        if self.preload_data_to_memory:
            return self.inputs_cache[index], self.labels[index]

        return self.__create_is_same_class_input(index), self.labels[index]

    def __len__(self) -> int:
        return self.num_samples

    def save(self, path: str):
        state_dict = {
            "image_pairs_indices": self.image_pairs_indices.cpu().detach(),
            "labels": self.labels,
            "train": self.train
        }
        torch.save(state_dict, path)

    @staticmethod
    def load(path: str, num_samples: int = -1, distance: int = 0, input_width: int = 224, input_height: int = 224,
             preload_data_to_memory: bool = False, device=torch.device("cpu")):
        state_dict = torch.load(path, map_location=device)

        return IsSameDataset(train=state_dict["train"],
                             image_pairs_indices=state_dict["image_pairs_indices"],
                             labels=state_dict["labels"],
                             num_samples=num_samples,
                             distance=distance,
                             input_width=input_width,
                             input_height=input_height,
                             preload_data_to_memory=preload_data_to_memory)
