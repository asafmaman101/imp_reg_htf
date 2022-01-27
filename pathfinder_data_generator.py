import argparse
import glob
import os
import os.path

import cv2
import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split


def load_images_from_folder(path: str) -> torch.Tensor:
    image_file_paths = glob.glob(os.path.join(path, '*.png'))
    image_file_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images = np.stack([cv2.imread(file, 0) for file in image_file_paths])
    return torch.tensor(images).unsqueeze(dim=1)


def load_labels_from_file(file_path: str):
    loaded_file = np.load(file_path)
    labels = np.array(list(map(int, loaded_file[:, 3])))
    return labels


def gen_dataset_from_self_generated_folder(path: str, output_path, num_train_to_sample: int = -1,
                                           num_test_to_sample: int = -1, resize=150, random_state: int = 42):
    path_length = os.path.basename(path)[-1]
    all_images = load_images_from_folder(os.path.join(path, 'imgs'))
    all_labels = load_labels_from_file(os.path.join(path, 'metadata.npy'))

    if num_train_to_sample + num_test_to_sample > len(all_images):
        raise ValueError(f"Not enough images to sample. train + test > total images. I.e. {num_train_to_sample} "
                         f" + {num_test_to_sample} > {len(all_images)}")

    all_indices = np.arange(len(all_labels))

    sampled_train_indices, _ = train_test_split(all_indices,
                                                train_size=num_train_to_sample,
                                                random_state=random_state if random_state > 0 else None)

    remaining_indices = np.array(list(set(all_indices) - set(sampled_train_indices)))

    if len(remaining_indices == num_test_to_sample):
        sampled_test_indices = remaining_indices
    else:
        sampled_test_indices, _ = train_test_split(remaining_indices,
                                                   train_size=num_test_to_sample,
                                                   random_state=random_state if random_state > 0 else None)

    train_indices = np.sort(sampled_train_indices).tolist()
    test_indices = np.sort(sampled_test_indices).tolist()

    train_images, train_labels = all_images[train_indices], all_labels[train_indices]
    test_images, test_labels = all_images[test_indices], all_labels[test_indices]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage('L'),
        torchvision.transforms.Resize(resize),
        torchvision.transforms.ToTensor()
    ])

    train_images = torch.stack([transform(image) for image in train_images])
    test_images = torch.stack([transform(image) for image in test_images])

    data_dict = {
        'meta_attributes': {
            'path_length': int(path_length),
            'num_train_examples': num_train_to_sample,
            'num_test_examples': num_test_to_sample,
            'train_mean': train_images.mean(),
            'train_std': train_images.std()
        },
        'train': {
            'images': train_images,
            'labels': torch.tensor(train_labels, dtype=torch.uint8).unsqueeze(dim=1),
            'original_indices': train_indices
        },
        'test': {
            'images': test_images,
            'labels': torch.tensor(test_labels, dtype=torch.uint8).unsqueeze(dim=1),
            'original_indices': test_indices
        }
    }
    file_name = f'self_gen_path_length_{path_length}_n_train_examples_{num_train_to_sample}_' \
                f'n_test_examples_{num_test_to_sample}_size_{resize}.pt'

    os.makedirs(output_path, exist_ok=True)
    torch.save(data_dict, os.path.join(output_path, file_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path where the raw dataset file is located.")
    parser.add_argument("--output_dir", default="data/pathfinder", type=str, help="Where to put the generated file.")
    parser.add_argument("--num_train_samples", required=True, type=int, default=-1,
                        help="Number of train samples for the train split.")
    parser.add_argument("--num_test_samples", required=True, type=int, default=-1,
                        help="Number of test samples for the test split.")

    args = parser.parse_args()

    for path_length_dir in glob.glob(os.path.join(args.dataset_path, '*')):
        gen_dataset_from_self_generated_folder(path=path_length_dir, output_path=args.output_dir,
                                               num_train_to_sample=args.num_train_samples,
                                               num_test_to_sample=args.num_test_samples)


if __name__ == '__main__':
    main()
