import itertools
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

import common.utils.tensor as tensor_utils
from common.utils.tensor import convert_tensor_to_one_hot
from tensor_factorization.models.tensor_factorization import TensorFactorization


class CPTensorFactorization(TensorFactorization):
    """
    Tensor CP factorization model.
    """

    def __init__(self, num_dim_per_mode: Sequence[int], rank: int = -1, init_mean: float = 0., init_std: float = 0.01,
                 linear_output_layer: bool = False, orthogonal_init: bool = False, device=torch.device("cpu")):
        """
        :param num_dim_per_mode: Number of dimensions per tensor mode.
        :param rank: Number of components. If negative the default is the max possible CP rank, which is the product of all dimensions except the max.
        :param init_mean: mean of vectors gaussian init.
        :param init_std: std of vectors gaussian init.
        :param linear_output_layer: If True, will use a linear output layer instead of a sum layer.
        :param orthogonal_init: if True, will initialize factorization such that each component is orthogonal. For standard CP factorization this
        means d_1 *... * d_N components containing all combinations of one-hot vectors.
        :param device: device to transfer model to after initializing.
        """
        super().__init__()
        self.num_dim_per_mode = num_dim_per_mode
        self.all_mode_dims_equal = len(set(self.num_dim_per_mode)) == 1
        self.max_mode_dim = max(self.num_dim_per_mode)
        self.order = len(self.num_dim_per_mode)
        self.init_mean = init_mean

        self.init_std = init_std
        self.linear_output_layer = linear_output_layer
        self.orthogonal_init = orthogonal_init
        self.rank = rank if rank != -1 else self.__compute_max_possible_tensor_rank()

        self.factors = nn.ParameterList(self.__create_factors())

        if self.linear_output_layer:
            self.output_layer_parameters = self.__create_output_layer_parameters(self.rank)

        if device is not None:
            self.to(device)

    def __compute_max_possible_tensor_rank(self):
        tensor_dims = list(self.num_dim_per_mode)
        if self.orthogonal_init:
            return np.prod(tensor_dims)

        max_index = tensor_dims.index(max(tensor_dims))
        tensor_dims.pop(max_index)

        return int(np.prod(tensor_dims).item())

    def __create_output_layer_parameters(self, dim: int):
        output_params = torch.randn(dim, 1) * self.init_std + self.init_mean
        return nn.Parameter(output_params, requires_grad=True)

    def __create_factors(self):
        if self.orthogonal_init:
            return self.__create_orthogonal_init_factors()

        factors = []
        for dim in self.num_dim_per_mode:
            factor = torch.randn(dim, self.rank, dtype=torch.float) * self.init_std + self.init_mean
            factors.append(factor)

        factors = [nn.Parameter(factor, requires_grad=True) for factor in factors]
        return factors

    def __create_factor_parameter_tensor(self, num_rows: int, num_cols: int, identity_init: bool = False) -> torch.Tensor:
        if identity_init:
            return torch.eye(num_rows, num_cols)

        return torch.randn(num_rows, num_cols) * self.init_std + self.init_mean

    def __create_orthogonal_init_factors(self):
        options_per_dim = [range(dim) for dim in self.num_dim_per_mode]
        all_options = torch.tensor(list(itertools.product(*options_per_dim)))

        factors = []
        for i, dim in enumerate(self.num_dim_per_mode):
            factor = torch.zeros(dim, self.rank, dtype=torch.float)
            factor[all_options[:, i], torch.arange(self.rank)] = self.init_std
            factors.append(nn.Parameter(factor, requires_grad=True))

        return factors

    def __create_zero_padded_factors(self):
        zero_padded_factors = []

        for factor in self.factors:
            if factor.size(0) < self.max_mode_dim:
                zero_padding = torch.zeros(self.max_mode_dim - factor.size(0), factor.size(1), dtype=factor.dtype, device=factor.device)
                factor = torch.cat([factor, zero_padding])

            zero_padded_factors.append(factor)

        return zero_padded_factors

    def __compute_tensor_factorization_values(self, inputs: torch.Tensor, is_input_encoded: bool = True):
        if not is_input_encoded:
            inputs = convert_tensor_to_one_hot(inputs, num_options=self.max_mode_dim)

        if not self.all_mode_dims_equal:
            factors = self.__create_zero_padded_factors()
        else:
            factors = [factor for factor in self.factors]

        factors = torch.stack(factors)

        # b --- batch size, o --- order, d --- mode dimension, r --- number of components
        outputs = torch.einsum('bod, odr -> bor', inputs, factors).prod(dim=1)
        outputs = outputs.sum(dim=1) if not self.linear_output_layer else torch.matmul(outputs, self.output_layer_parameters).squeeze(dim=-1)
        return outputs

    def normalize(self, new_norm: float, p: str = "fro"):
        """
        Multiplies the parameters by a constant (the same constant for all parameters), such that the end tensor norm is of the value given.
        Computes the whole tensor in the process (do not use for large scale tensors).
        :param new_norm: new value to normalize the end tensor norm to.
        :param p: name of the norm (see PyTorch torch.norm docs), defaults to Frobenius norm.
        """
        target_tensor = self.compute_tensor()
        homogeneity_coeff = self.order if not self.linear_output_layer else self.order + 1
        params_mul_factor = torch.pow(new_norm / target_tensor.norm(p=p), 1 / homogeneity_coeff)

        for param in self.parameters():
            param.data.mul_(params_mul_factor)

    def compute_tensor(self):
        if not self.linear_output_layer:
            output_tensor = tensor_utils.reconstruct_parafac(self.factors)
        else:
            output_tensor = tensor_utils.reconstruct_parafac(list(self.factors.parameters()),
                                                             coefficients=self.output_layer_parameters.squeeze(dim=1))

        return output_tensor

    def compute_tensor_at_indices(self, indices_tensor: torch.Tensor):
        """
        :param indices_tensor: Tensor of indices in shape (B, order), where B is the batch size and each column
        corresponds to a tensor mode.
        """
        return self.__compute_tensor_factorization_values(indices_tensor, is_input_encoded=False)

    def forward(self, inputs: torch.Tensor, is_input_encoded: bool = True):
        """
        :param inputs: If is_input_encoded is False, Tensor of indices in shape (B, order), where B is the batch size and each column
        corresponds to a tensor mode. Otherwise (default), either a one hot representation of the indices tensor of shape (B, order, max_mode_dim), or
        any other encoding of the same shape (corresponds to rank 1 tensor sensing/usage of templates to encode inputs).
        :param is_input_encoded: Flag that determines correct format for indices_tensor.
        """
        return self.__compute_tensor_factorization_values(inputs, is_input_encoded=is_input_encoded)
