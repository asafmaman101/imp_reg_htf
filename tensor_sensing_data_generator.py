import argparse
import itertools
import os
import random
from datetime import datetime

import numpy as np
import torch

import common.utils.logging as logging_utils
import common.utils.tensor as tensor_utils
from tensor_factorization.datasets.tensor_sensing_dataset import TensorSensingDataset
from tensor_factorization.models import CPTensorFactorization, HierarchicalTensorFactorization


def __set_initial_random_seed(random_seed: int):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def create_cp_target(args):
    with torch.no_grad():
        target_tensor_factorization = CPTensorFactorization(num_dim_per_mode=[args.mode_dim_size] * args.order,
                                                            rank=args.target_tensor_cp_rank,
                                                            init_mean=args.target_init_mean,
                                                            init_std=args.target_init_std)

        normalization_const = __get_target_normalization_const(args)
        target_tensor_factorization.normalize(new_norm=normalization_const, p="fro")

        return target_tensor_factorization


def create_ht_target(args):
    num_dim_per_mode = [args.mode_dim_size] * args.order
    with torch.no_grad():
        target_tensor_factorization = HierarchicalTensorFactorization(num_dim_per_mode=num_dim_per_mode,
                                                                      pool_size=args.target_tensor_ht_pool_size,
                                                                      pool_pattern=args.target_tensor_ht_pool_pattern,
                                                                      layer_widths=args.target_tensor_ht_rank,
                                                                      init_mean=args.target_init_mean,
                                                                      init_std=args.target_init_std)

        normalization_const = __get_target_normalization_const(args)
        target_tensor_factorization.normalize(new_norm=normalization_const, p="fro")

        return target_tensor_factorization


def __get_target_normalization_const(args):
    num_dim_per_mode = [args.mode_dim_size] * args.order

    if args.target_calibrate_fro_norm:
        return np.sqrt(np.prod(num_dim_per_mode)).item()
    elif args.target_fro_norm > 0:
        return args.target_fro_norm

    return 1


def create_inputs(args):
    num_dim_per_mode = [args.mode_dim_size] * args.order
    num_entries = np.prod(num_dim_per_mode).item()

    if args.task_type == "sensing":
        X = torch.randn(args.num_samples, args.order, args.mode_dim_size)
        X = X / (np.power(num_entries, 1 / (2 * args.order)))

    elif args.task_type == "completion":
        num_samples = min(args.num_samples, num_entries)
        all_indices_tensors = __create_all_indices_tensor(num_dim_per_mode)

        chosen_observed_indices = torch.multinomial(torch.ones(num_entries), num_samples, replacement=False)
        shuffled_observed_indices_tensor = all_indices_tensors[chosen_observed_indices]

        X = tensor_utils.convert_tensor_to_one_hot(shuffled_observed_indices_tensor, num_options=args.mode_dim_size)
    else:
        raise ValueError(f"Unsupported task type: {args.task_type}.")

    return X


def __create_all_indices_tensor(mode_dims):
    indices = []
    per_mode_options = [range(dim) for dim in mode_dims]
    for tensor_index in itertools.product(*per_mode_options):
        indices.append(torch.tensor(tensor_index, dtype=torch.long))

    return torch.stack(indices)


def create_tensor_sensing_dataset(args, target_tensor_factorization):
    with torch.no_grad():
        inputs = create_inputs(args)
        targets = target_tensor_factorization(inputs)
        dataset = TensorSensingDataset(inputs, targets, additional_metadata=args.__dict__)
        return dataset


def __create_dataset_file_name(args, target):
    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")

    if args.custom_file_name:
        file_name = f"{args.custom_file_name}_{now_utc_str}.pt"
    else:
        tensor_fact_str = __create_dataset_file_name_factorization_str(args, target)
        file_name = f"t{args.task_type[0]}_{tensor_fact_str}_{now_utc_str}.pt"

    return file_name


def __convert_float_to_str(f):
    if isinstance(f, int):
        return str(f)

    return str(int(f)) if f.is_integer() else "{:.2f}".format(f)


def __create_dataset_file_name_factorization_str(args, target):
    order_dim_str = f"ord_{args.order}_d_{args.mode_dim_size}"

    if args.target_fro_norm > 0 or args.target_calibrate_fro_norm:
        target_fro_norm_str = __convert_float_to_str(__get_target_normalization_const(args))
        norm_str = f"fro_{target_fro_norm_str}".replace(".", "_")
    else:
        init_mean_str = __convert_float_to_str(args.target_init_mean)
        init_std_str = __convert_float_to_str(args.target_init_std)
        norm_str = f"init_m_{init_mean_str}_std_{init_std_str}".replace(".", "_")

    if args.target_tensor_factorization_type == "cp":
        return f"cpr_{args.target_tensor_cp_rank}_{order_dim_str}_{norm_str}"
    elif args.target_tensor_factorization_type == "ht":
        ht_rank_str = '_'.join([str(width) for width in target.hidden_layers_widths])
        return f"htr_{ht_rank_str}_" \
               f"pl_{args.target_tensor_ht_pool_size}{args.target_tensor_ht_pool_pattern[0]}_{order_dim_str}_{norm_str}"
    else:
        return f"rnd_{order_dim_str}_{norm_str}"


def __create_target_tensor_log_str(args, target):
    if args.target_tensor_factorization_type == "cp":
        return f"Target CP Rank: {args.target_tensor_cp_rank}"

    if args.target_tensor_factorization_type == "ht":
        return f"Target HT Rank: {'_'.join([str(width) for width in target.hidden_layers_widths])}, " \
               f"Target HT Pool Size: {args.target_tensor_ht_pool_size}, " \
               f"Target HT Pool Pattern: {args.target_tensor_ht_pool_pattern}"

    return f"Random Target (entries sampled directly)"


def save_dataset(args, dataset, target):
    file_name = __create_dataset_file_name(args, target)
    output_path = os.path.join(args.output_dir, file_name)
    dataset.save(output_path)

    target_norm_str = __convert_float_to_str(__get_target_normalization_const(args))
    target_tensor_str = __create_target_tensor_log_str(args, target)
    logging_utils.info(f"Created tensor {args.task_type} dataset at: {output_path}\nOrder: {args.order}, Modes dimension: {args.mode_dim_size}, "
                       f"{target_tensor_str}, Target Fro Norm: {target_norm_str},"
                       f" # Samples: {args.num_samples}")


def create_and_save_dataset(args):
    if args.target_tensor_factorization_type == "cp":
        target = create_cp_target(args)
    elif args.target_tensor_factorization_type == "ht":
        target = create_ht_target(args)
    else:
        raise ValueError(f"Unsupported target tensor factorization type: {args.target_tensor_factorization_type}")

    dataset = create_tensor_sensing_dataset(args, target)
    save_dataset(args, dataset, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("--output_dir", type=str, default="data/tc/sens", help="Path to the directory to save the target matrix and dataset at.")
    p.add_argument("--custom_file_name", type=str, default="", help="Custom file name prefix for the dataset.")

    p.add_argument("--task_type", type=str, default="completion", help="Task type. Can be either 'sensing' or 'completion'")
    p.add_argument("--num_samples", type=int, default=4096, help="Number of sensing samples to create.")
    p.add_argument("--order", type=int, default=3, help="Order of the tensor (number of modes).")
    p.add_argument("--mode_dim_size", type=int, default=16, help="Number of dimensions per each mode.")

    p.add_argument("--target_tensor_factorization_type", type=str, default="cp",
                   help="Create the target tensor using the given factorization type. Supports 'cp' and 'ht'.")
    p.add_argument("--target_tensor_cp_rank", type=int, default=5, help="CP rank of the target tensor.")
    p.add_argument("--target_tensor_ht_rank", nargs="+", type=int, default=[5, 5],
                   help="HT rank of target tensor. Will be used only if 'target_tensor_factorization_type' is set to 'ht'. The HT rank is a "
                        "tuple of ranks, one for each layer of the factorization")
    p.add_argument("--target_tensor_ht_pool_size", type=int, default=2,
                   help="Pool size of target HT factorization. Not relevant for CP factorization.")
    p.add_argument("--target_tensor_ht_pool_pattern", type=str, default="checkerboard",
                   help="HT factorization pooling pattern to use. Currently supports 'checkerboard' (pools close modes) and 'half' (pools opposite modes)")

    p.add_argument("--target_init_mean", type=float, default=0., help="Mean for target parameters Gaussian init.")
    p.add_argument("--target_init_std", type=float, default=1., help="Standard deviation for target parameters Gaussian init.")
    p.add_argument("--target_fro_norm", type=float, default=64, help="Fro norm of the target tensor. If <=0 will not normalize.")
    p.add_argument("--target_calibrate_fro_norm", action="store_true", help="Normalizes target to be of Frobenius norm equal to sqrt num entries "
                                                                            "(overrides the given 'target_fro_norm').")

    args = p.parse_args()

    logging_utils.init_console_logging()
    __set_initial_random_seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    create_and_save_dataset(args)


if __name__ == "__main__":
    main()
