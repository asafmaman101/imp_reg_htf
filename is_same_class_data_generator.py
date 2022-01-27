import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from locality_bias.datasets.is_same_dataset import IsSameDataset

import common.utils.logging as logging_utils
from common.data.modules.torchvision_datamodule import TorchvisionDataModule


def __set_initial_random_seed(random_seed: int):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def __create_dataset_filename(args, dataset):
    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")

    if args.custom_file_name:
        return f"{args.custom_file_name}_{now_utc_str}.pt"

    train_str = "train" if args.train else "test"
    filename = f"cifar10_is_same_{train_str}_s_{len(dataset)}_{now_utc_str}"
    filename = filename.replace(".", "-") + ".pt"
    return filename


def __plot_sample_of_images(dataset, num_images_to_plot: int = 2):
    figure = plt.figure()

    for i in range(num_images_to_plot):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        sample_img, label = dataset[sample_idx]

        sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min())
        sample_img = sample_img.permute(1, 2, 0)

        figure.add_subplot(num_images_to_plot, 1, i + 1)
        plt.title(str(label.item()))
        plt.axis("off")
        plt.imshow(sample_img, cmap="gray")

    figure.tight_layout()
    plt.show()


def __sample_negative_is_image_index(sampled_index, all_relevant_image_indices):
    neg_index_candidate = all_relevant_image_indices[np.random.randint(0, len(all_relevant_image_indices))]
    while neg_index_candidate == sampled_index:
        neg_index_candidate = all_relevant_image_indices[np.random.randint(0, len(all_relevant_image_indices))]

    return neg_index_candidate


def __sample_positive_image_pair_indices(per_relevant_label_image_indices: Dict[int, torch.Tensor]):
    label = np.random.randint(0, len(per_relevant_label_image_indices))
    current_label_image_indices = per_relevant_label_image_indices[label]

    first_image_sample = np.random.randint(0, len(current_label_image_indices))
    second_image_sample = np.random.randint(0, len(current_label_image_indices))
    return torch.stack([current_label_image_indices[first_image_sample],
                        current_label_image_indices[second_image_sample]], dim=0)


def __sample_negative_image_pair_indices(per_relevant_label_image_indices: Dict[int, torch.Tensor]):
    label = np.random.randint(0, len(per_relevant_label_image_indices))

    current_label_image_indices = per_relevant_label_image_indices[label]
    other_labels_image_indices = torch.cat([label_indices for l, label_indices in per_relevant_label_image_indices.items() if l != label])

    first_image_sample = np.random.randint(0, len(current_label_image_indices))
    first_image_index = current_label_image_indices[first_image_sample]

    second_image_sample = np.random.randint(0, len(other_labels_image_indices))
    second_image_index = other_labels_image_indices[second_image_sample]

    return torch.stack([first_image_index, second_image_index], dim=0)


def __create_image_pairs_indices_and_labels(args, original_dataset):
    original_labels = torch.tensor(original_dataset.targets)
    relevant_labels = original_labels.unique().tolist()
    per_relevant_label_image_indices = {label: torch.nonzero(original_labels == label).squeeze() for label in relevant_labels}
    num_samples = args.num_samples if args.num_samples > 0 else len(original_dataset)

    pos_image_pairs_indices = []
    neg_image_pairs_indices = []
    for _ in range(num_samples):
        if np.random.rand() >= 0.5:
            pos_image_pair_indices = __sample_positive_image_pair_indices(per_relevant_label_image_indices)
            pos_image_pairs_indices.append(pos_image_pair_indices)
        else:
            neg_image_pair_indices = __sample_negative_image_pair_indices(per_relevant_label_image_indices)
            neg_image_pairs_indices.append(neg_image_pair_indices)

    pos_image_pairs_indices = torch.stack(pos_image_pairs_indices)
    neg_image_pairs_indices = torch.stack(neg_image_pairs_indices)
    image_pairs_indices = torch.cat([pos_image_pairs_indices, neg_image_pairs_indices])
    is_same_labels = torch.cat([torch.ones(pos_image_pairs_indices.shape[0], dtype=torch.long),
                                torch.zeros(neg_image_pairs_indices.shape[0], dtype=torch.long)])

    shuffle_perm = torch.randperm(len(image_pairs_indices))
    return image_pairs_indices[shuffle_perm], is_same_labels[shuffle_perm]


def create_and_save_dataset(args):
    torchvision_datamodule = TorchvisionDataModule(dataset_name="cifar10")
    torchvision_datamodule.setup()

    original_dataset = torchvision_datamodule.train_dataset if args.train else torchvision_datamodule.test_dataset
    image_pairs_indices, labels = __create_image_pairs_indices_and_labels(args, original_dataset)

    dataset = IsSameDataset(train=args.train, image_pairs_indices=image_pairs_indices, labels=labels)

    __plot_sample_of_images(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    filename = __create_dataset_filename(args, dataset)
    output_path = os.path.join(args.output_dir, filename)
    dataset.save(output_path)

    logging_utils.info(f"Created dataset at: '{output_path}'\n"
                       f"Args: {json.dumps(args.__dict__, indent=2)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("--output_dir", type=str, default="data/is_same", help="Path to the directory to save the target matrix and dataset at")
    p.add_argument("--custom_file_name", type=str, default="", help="Custom file name prefix for the dataset")

    p.add_argument("--train", action="store_true", default=False, help="If True the dataset is created based on the original train dataset,"
                                                                       " otherwise it is based on the original test dataset")
    p.add_argument("--num_samples", type=int, default=-1,
                   help="Number of training samples. If < 0 (default) will create a dataset set the size of the original one.")

    args = p.parse_args()

    logging_utils.init_console_logging()
    __set_initial_random_seed(args.random_seed)

    create_and_save_dataset(args)


if __name__ == "__main__":
    main()
