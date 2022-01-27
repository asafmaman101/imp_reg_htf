import os
import re
from typing import Sequence

import torch.utils.data


class TorchDataset(torch.utils.data.Dataset):
    """
    Saves items using torch serializing and deserialization.
    """

    FILE_SUFFIX_FORMAT = "{}.pt"
    FILE_SUFFIX_REGEX_STR = r"(?P<id>\d+)\.pt"

    def __init__(self, dir: str, files_prefix: str = "item"):
        self.dir = dir
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self.files_prefix = files_prefix
        self.file_name_format = self.files_prefix + self.FILE_SUFFIX_FORMAT
        self.file_name_regex = re.compile(self.files_prefix + self.FILE_SUFFIX_REGEX_STR)

        self.file_paths = self.__get_sorted_relevant_file_names_in_dir()
        self.file_counter = len(self.file_paths)

    def __getitem__(self, index: int):
        file_path = self.file_paths[index]
        return torch.load(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __get_sorted_relevant_file_names_in_dir(self) -> Sequence[str]:
        relevant_file_paths = []
        for file_name in os.listdir(self.dir):
            file_path = os.path.join(self.dir, file_name)
            if os.path.isfile(file_path) and self.file_name_regex.match(file_name):
                relevant_file_paths.append(file_path)

        relevant_file_paths.sort(key=self.__extract_id_from_file_path)
        return relevant_file_paths

    def __extract_id_from_file_path(self, path: str):
        file_name = os.path.basename(path)
        match = self.file_name_regex.match(file_name)
        return int(match.group("id"))

    def get_all(self):
        return [self[i] for i in range(len(self))]

    def add(self, item):
        file_name = self.file_name_format.format(self.file_counter)
        file_path = os.path.join(self.dir, file_name)
        torch.save(item, file_path)
        self.file_paths.append(file_path)
        self.file_counter += 1

    def add_all(self, item_seq):
        for item in item_seq:
            self.add(item)
