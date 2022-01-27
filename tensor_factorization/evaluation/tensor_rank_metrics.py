import itertools
import string
from functools import reduce

import numpy as np
import torch

import common.utils.tensor as tensor_utils
import matrix_factorization.evaluation.matrix_metrics as matrix_metrics
from common.evaluation.metrics import ScalarMetric
from tensor_factorization.models import CPTensorFactorization


class TensorCPRank(ScalarMetric):

    def __init__(self, tol=1e-6, method="als"):
        """
        :param tol: Tolerance threshold used to find the rank of the tensor. The rank will be the minimal Parafac decomposition that has a
        reconstruction mse of less than tol.
        :param method: Method to compute the CP decomposition of rank r approximation. Currently supports: 'als' (Alternating Least Squares) and 'adam'
        which is gradient descent with Adam optimizer.
        """
        self.tol = tol
        self.method = method
        self.__current_value = None

    def __call__(self, tensor):
        cp_rank = find_cp_rank(tensor, tol=self.tol, method=self.method)
        self.__current_value = cp_rank
        return cp_rank

    def current_value(self):
        return self.__current_value

    def has_epoch_metric_to_update(self) -> bool:
        return self.__current_value is not None

    def reset_current_epoch_values(self):
        self.__current_value = None


class TensorMatricizationsEffectiveRanksMean(ScalarMetric):

    def __init__(self, num_row_modes=1):
        """
        :param num_row_modes: Number of row modes in each matricization.
        """
        self.num_row_modes = num_row_modes
        self.__current_value = None

    def __call__(self, tensor):
        effective_ranks = compute_matricizations_effective_ranks_with_num_row_modes(tensor, num_row_modes=self.num_row_modes)
        effective_ranks_mean = np.mean(effective_ranks).item()
        self.__current_value = effective_ranks_mean
        return effective_ranks_mean

    def current_value(self):
        return self.__current_value

    def has_epoch_metric_to_update(self) -> bool:
        return self.__current_value is not None

    def reset_current_epoch_values(self):
        self.__current_value = None


class TensorMatricizationsEffectiveRanksSTD(ScalarMetric):

    def __init__(self, num_row_modes=1):
        """
        :param num_row_modes: Number of row modes in each matricization.
        """
        self.num_row_modes = num_row_modes
        self.__current_value = None

    def __call__(self, tensor):
        effective_ranks = compute_matricizations_effective_ranks_with_num_row_modes(tensor, num_row_modes=self.num_row_modes)
        effective_ranks_std = np.std(effective_ranks).item()
        self.__current_value = effective_ranks_std
        return effective_ranks_std

    def current_value(self):
        return self.__current_value

    def has_epoch_metric_to_update(self) -> bool:
        return self.__current_value is not None

    def reset_current_epoch_values(self):
        self.__current_value = None


def compute_input_output_tensor_for_model(X, model, num_input_features=None):
    """
    :param X: Inputs, should include all possible options.
    :param model: Model to create outputs with.
    :param num_input_features: Number of input features. If none is given, assumes X.size(1) is the number of features.
    """
    num_options_per_mode = get_num_options_per_mode(X, num_input_features=num_input_features)
    y = model(X)
    return y.view(*num_options_per_mode)


def get_num_options_per_mode(X, num_input_features=None):
    """
    Computes the number of options per tensor order from the data X that contains all possible options.
    :param X: Data matrix of size (num_samples, num_features) that contains all possible inputs.
    :param num_input_features: Number of input features. If none, will assume it is simply X.size(1).
    :return: List of size for each order. Assumes each input feature has the same number of optional values.
    """
    num_features = num_input_features if num_input_features is not None else X.size(1)
    num_options = np.power(X.size(0), 1 / num_features).item()
    num_options_per_mode = [int(round(num_options))] * num_features
    return num_options_per_mode


def compute_multilinear_effective_ranks(tensor):
    """
    Computes the effective ranks for all matricizations where the rows is one mode and the columns are the rest. This is a continuous approximation
    for the tensor multilinear rank.
    :return: sequence of effective ranks, one for each matricization with a single mode in its rows.
    """
    return compute_matricizations_effective_ranks_with_num_row_modes(tensor, num_row_modes=1)


def compute_matricizations_effective_ranks_with_num_row_modes(tensor, num_row_modes=1):
    """
    Computes the effective ranks for all matricizations where the rows contain the number of modes given.
    :return: sequence of effective ranks, one for each matricization with a single mode in its rows.
    """
    effective_ranks = []
    tensor_order = len(tensor.size())

    mode_subsets = __get_subsets_of_size_exclude_symmetric(range(tensor_order), num_row_modes)
    for subset in mode_subsets:
        indices_not_in_subset = [j for j in range(tensor_order) if j not in subset]
        permute_indices = list(subset) + indices_not_in_subset

        row_dim = __compute_matricization_dim(tensor.size(), subset)
        col_dim = __compute_matricization_dim(tensor.size(), indices_not_in_subset)

        matricized_tensor = tensor.permute(*permute_indices).reshape(row_dim, col_dim)
        effective_ranks.append(matrix_metrics.matrix_effective_rank(matricized_tensor))

    return effective_ranks


def compute_matricization_effective_rank(tensor: torch.Tensor, matricization_row_modes):
    mat = tensor_utils.matricize(tensor, matricization_row_modes)
    return matrix_metrics.matrix_effective_rank(mat)


def __get_subsets_of_size_exclude_symmetric(indices, k):
    subsets_of_size_k = list(itertools.combinations(indices, k))
    if k != len(indices) // 2:
        return subsets_of_size_k

    # If i is exactly len(arr) // 2, need to remove subsets that perform the same partition.
    subsets_to_return = []
    existing_partitions = set()
    for subset in subsets_of_size_k:
        frozen_subset = frozenset(subset)
        frozen_subset_comp = frozenset([j for j in indices if j not in frozen_subset])

        if frozen_subset not in existing_partitions:
            subsets_to_return.append(subset)

        existing_partitions.add(frozen_subset)
        existing_partitions.add(frozen_subset_comp)

    return subsets_to_return


def __compute_matricization_dim(tensor_size, indices):
    return np.prod([tensor_size[i] for i in indices]).item()


def find_cp_rank(tensor, max_rank=-1, tol=1e-6, method="als"):
    """
    Finds the CP rank of the tensor by searching for the minimal r for which the reconstruction of the parafac decomposition is less than tol.
    :param tensor: PyTorch Tensor to compute CP rank for.
    :param max_rank: max rank to check. By default, will check for all ranks up to the product of dimensions except the maximal.
    :param tol: tolerance threshold used to determine the cp rank. The rank is determined as the minimal rank of CP decomposition for which
    the mse of the reconstruction is below tol.
    :param method: Method to compute the CP decomposition of rank r approximation. Currently supports: 'als' (Alternating Least Squares) and 'adam'
    which is gradient descent with Adam optimizer.
    :return: estimated cp_rank of tensor. Returns -1 on failure.
    """
    if torch.allclose(tensor, torch.zeros_like(tensor)):
        return 0

    max_rank = max_rank if max_rank != -1 else __compute_max_possible_tensor_rank(tensor)

    first = 0
    last = max_rank - 1
    curr_min = max_rank
    try:
        while first <= last:
            mid = (first + last) // 2
            mse = compute_reconstruction_mse(tensor, r=mid + 1, tol=tol, method=method)
            if mse < tol:
                curr_min = mid + 1
                last = mid - 1
            else:
                first = mid + 1
    except np.linalg.LinAlgError:
        return -1

    return curr_min.item()


def compute_reconstruction_mse(tensor, r, tol=1e-6, method="als"):
    if method == "adam":
        reconstructed_tensor = cp_adam(tensor, r, tol=tol, return_factors=False)
    elif method == "als":
        coeff, factors = cp_als(tensor, r, tol=tol)
        factors[0] = factors[0] * coeff
        reconstructed_tensor = tensor_utils.reconstruct_parafac(factors).to(tensor.device)
    else:
        raise ValueError("Unsupported method. Currently supports: 'als', 'adam'.")

    mse = ((tensor - reconstructed_tensor) ** 2).sum() / tensor.numel()
    return mse.item()


def cp_adam(tensor, r, tol=1e-6, init_std=0.01, lr=1e-3, max_iter=50000, validate_convergence_every=100, early_stop_patience=10,
            num_validations_early_stop_retention=100, verbose=False, return_factors=True):
    cp_factorization = CPTensorFactorization(num_dim_per_mode=list(tensor.size()), rank=r, init_std=init_std)
    cp_factorization.to(tensor.device)
    opt = torch.optim.Adam(cp_factorization.parameters(), lr=lr)

    best_mse = float('inf')
    num_not_improved = 0

    for i in range(max_iter):
        opt.zero_grad()

        reconstructed_tensor = cp_factorization.compute_tensor()
        mse_loss = ((tensor - reconstructed_tensor) ** 2).sum() / tensor.numel()
        mse_loss.backward()

        opt.step()

        # Validates whether the mse is small enough
        if i % validate_convergence_every == 0:
            with torch.no_grad():
                reconstructed_tensor = cp_factorization.compute_tensor()
                mse = ((tensor - reconstructed_tensor) ** 2).sum() / tensor.numel()
                if mse <= tol:
                    return reconstructed_tensor.detach()

                if mse < best_mse:
                    best_mse = mse
                else:
                    num_not_improved += 1

                if num_not_improved > early_stop_patience:
                    if verbose:
                        print(f"Early stopping CP rank computation for rank {r} with Adam since error has not improved in {early_stop_patience}"
                              f" validations.\nBest Error: {best_mse.item()}.")
                        break

                if i % (validate_convergence_every * num_validations_early_stop_retention) == 0:
                    num_not_improved = 0

    if verbose:
        print(f"Returned mse for rank {r} was higher than threshold after iter {i} iterations.\n"
              f"Threshold: {tol}, MSE: {mse.item()}")

    if return_factors:
        return reconstructed_tensor, [factor.data.detach() for factor in cp_factorization.factors]

    return reconstructed_tensor


def cp_als(tensor, r, max_iter=1000, validate_convergence_every=1, tol=1e-6):
    np_tensor = tensor.cpu().detach().numpy()
    factors = [np.random.randn(dim, r) for dim in np_tensor.shape]
    num_modes = len(np_tensor.shape)

    for i in range(max_iter):
        for n in range(num_modes):
            other_modes = [m for m in range(num_modes) if m != n]
            other_factors = [factors[m] for m in other_modes]

            V_elements = [np.matmul(factor.T, factor) for factor in other_factors]
            V = reduce(lambda a, b: a * b, V_elements)
            W = reduce(__khatri_rao_product, other_factors)
            Xn = np.rollaxis(np_tensor, n).reshape(np_tensor.shape[n], -1)

            update = np.matmul(Xn, np.matmul(W, np.linalg.pinv(V)))

            coeff = np.linalg.norm(factors[n], axis=0, keepdims=True)
            factors[n] = update / coeff

        if i % validate_convergence_every == 0:
            curr_factors = [factors[0] * coeff] + factors[1:]
            reconstructed_tensor = np_reconstruct_parafac(curr_factors)
            mse = ((np_tensor - reconstructed_tensor) ** 2).sum() / np_tensor.size
            if mse < tol:
                break

    factors = [torch.from_numpy(factor) for factor in factors]
    coeff = torch.from_numpy(coeff)
    return coeff, factors


def __khatri_rao_product(A, B):
    return np.einsum("ij, kj -> ikj", A, B).reshape(-1, A.shape[1])


# Adaptation of https://stackoverflow.com/a/13772838/160466
def np_reconstruct_parafac(factors):
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.ascii_lowercase[temp_dim] + 'z,'
    request = request[:-1] + '->' + string.ascii_lowercase[:ndims]
    return np.einsum(request, *factors)


def __compute_max_possible_tensor_rank(tensor):
    tensor_dims = list(tensor.size())
    max_index = tensor_dims.index(max(tensor_dims))
    tensor_dims.pop(max_index)

    return np.prod(tensor_dims)
