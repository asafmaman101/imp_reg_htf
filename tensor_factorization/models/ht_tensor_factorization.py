import itertools
import math
import string
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn

from common.data.loaders.fast_tensor_dataloader import FastTensorDataLoader
from common.utils.tensor import convert_tensor_to_one_hot
from tensor_factorization.models.tensor_factorization import TensorFactorization


class HierarchicalTensorFactorization(TensorFactorization):
    """
    Tensor Hierarchical Tucker (HT) factorization model.
    """

    def __init__(self, num_dim_per_mode: Sequence[int], pool_size: int = 2, per_layer_pool_size: List[int] = None, pool_pattern: str = "checkerboard",
                 layer_widths: List[int] = None, init_mean: float = 0, init_std: float = 0.01, linear_output_layer: bool = True,
                 device=torch.device("cpu")):
        """
        :param num_dim_per_mode: number of dimensions per tensor mode. Currently only non-degenerate order > 1 factorizations are supported.
        :param pool_size: size of pooling window.
        :param per_layer_pool_size: custom per layer pool size. If None (default) then will use the the given 'pool_size' for all layers.
        Length must match depth of the factorization.
        :param pool_pattern: Pooling pattern to use. Currently supports 'checkerboard' (pools adjacent modes first), 'checkerboard_sq' (pools square
        patches, similar to standard pooling on 2D images, pool size must be square when using this pattern) and half' (pools adjacent modes last).
        :param layer_widths: Width in each layer of the factorization. If None (default) the max possible CP rank which
        is the product of all dimensions except the max is used for all layers. Only one of 'layer_widths' and 'layer_per_pool_window_widths' can
        be given.
        :param init_mean: mean of vectors gaussian init.
        :param linear_output_layer: If True, will use a linear output layer instead of a sum layer.
        :param device: device to transfer model to after initializing.
        """
        super().__init__()

        # Setup variables
        self.num_dim_per_mode = num_dim_per_mode
        self.all_mode_dims_equal = len(set(self.num_dim_per_mode)) == 1
        self.max_mode_dim = max(self.num_dim_per_mode)
        self.order = len(self.num_dim_per_mode)
        if self.order <= 1:
            raise ValueError("HT factorization of order 1 is currently not supported. Order must be >= 2.")

        self.pool_size = pool_size
        self.per_layer_pool_size = per_layer_pool_size
        self.pool_pattern = pool_pattern
        self.pool_fn = lambda x: x.prod(dim=-1)
        self.pool_perm_cache = {}

        self.init_mean = init_mean
        self.init_std = init_std
        self.linear_output_layer = linear_output_layer
        self.num_hidden_layers = self.__compute_num_hidden_layers(self.order)
        self.depth = self.num_hidden_layers + 1

        self.__set_hidden_layer_width_vars(layer_widths)

        # Verify constructor arguments are valid
        self.__verify_args()

        # Initialize module parameters
        self.__init_parameters()

        # Misc
        self.__normalize_params_mul_factor_inv_pow = self.__create_normalize_params_mul_factor_inv_pow()

        if device is not None:
            self.to(device)

    def __set_hidden_layer_width_vars(self, layer_widths):
        self.hidden_layers_widths = layer_widths
        if self.hidden_layers_widths is None:
            self.hidden_layers_widths = [self.__compute_max_possible_tensor_rank(self.num_dim_per_mode)] * self.num_hidden_layers

        per_layer_order = [len(level_nodes) for level_nodes in self.get_per_level_mode_tree_nodes()]
        self.hidden_layers_per_pool_window_widths = [[layer_width] * per_layer_order[i + 1] for i, layer_width in
                                                     enumerate(self.hidden_layers_widths)]

    def __verify_args(self):
        if len(self.hidden_layers_widths) != self.num_hidden_layers:
            raise ValueError(f"Mismatch between factorization # hidden layers to # layer widths. Computed # hidden layers: "
                             f"{self.num_hidden_layers}, # layer widths: {len(self.hidden_layers_widths)}.")

    @staticmethod
    def __compute_max_possible_tensor_rank(num_dim_per_mode):
        tensor_dims = list(num_dim_per_mode)
        max_index = tensor_dims.index(max(tensor_dims))
        tensor_dims.pop(max_index)

        return int(np.prod(tensor_dims).item())

    def __compute_num_hidden_layers(self, order):
        num_hidden_layers = 0
        curr_order = order
        while curr_order > 1:
            curr_order = int(math.ceil(curr_order / self.get_layer_pool_size(num_hidden_layers)))
            num_hidden_layers += 1

        return num_hidden_layers

    def __get_layer_width_for_pool_window(self, layer_index: int):
        return self.hidden_layers_widths[layer_index]

    def __init_parameters(self):
        self.per_hidden_layer_parameter_lists = nn.ModuleList(self.__create_per_hidden_layer_parameter_lists())

        if self.linear_output_layer:
            output_layer_input_width = self.__get_layer_width_for_pool_window(self.num_hidden_layers - 1)
            self.output_layer_parameters = self.__create_output_layer_parameters(output_layer_input_width)

    def __create_per_hidden_layer_parameter_lists(self):
        per_layer_parameter_lists = []

        curr_order = self.order
        curr_num_dim_per_mode = self.num_dim_per_mode
        for layer_index, layer_per_pool_window_widths in enumerate(self.hidden_layers_per_pool_window_widths):
            per_layer_parameter_lists.append(self.__create_layer_factors_params_list(curr_num_dim_per_mode, layer_index))

            curr_order = int(math.ceil(curr_order / self.get_layer_pool_size(layer_index)))
            curr_num_dim_per_mode = layer_per_pool_window_widths

        return per_layer_parameter_lists

    def __create_layer_factors_params_list(self, curr_num_dim_per_mode: Sequence[int], layer_index: int):
        layer_per_pool_window_widths = self.hidden_layers_per_pool_window_widths[layer_index]
        layer_factors = nn.ParameterList()

        for i, dim in enumerate(curr_num_dim_per_mode):
            layer_width = layer_per_pool_window_widths[i // self.get_layer_pool_size(layer_index)]
            factor = self.__create_factor_parameter_tensor(dim, layer_width, layer_index=layer_index,
                                                           input_index=i)
            layer_factors.append(nn.Parameter(factor, requires_grad=True))

        return layer_factors

    def __create_factor_parameter_tensor(self, num_rows: int, num_cols: int, layer_index: int, input_index: int,
                                         identity_init: bool = False) -> torch.Tensor:
        if identity_init:
            return torch.eye(num_rows, num_cols)

        return torch.randn(num_rows, num_cols) * self.init_std + self.init_mean

    def __create_output_layer_parameters(self, dim: int):
        output_params = torch.randn(dim, 1) * self.init_std + self.init_mean
        return nn.Parameter(output_params, requires_grad=True)

    def __get_hidden_layers_input_orders(self):
        return [len(layer_params) for layer_params in self.per_hidden_layer_parameter_lists]

    def __create_normalize_params_mul_factor_inv_pow(self):
        layers_input_orders = self.__get_hidden_layers_input_orders() + [1]
        homogeneity_coeffs = np.array([0] * self.order, dtype=np.int)

        for i in range(self.num_hidden_layers):
            homogeneity_coeffs += 1
            next_layer_homogeneity_coeffs = np.zeros(layers_input_orders[i + 1], dtype=np.int)

            pool_size = self.get_layer_pool_size(i)
            num_pool_outputs = len(homogeneity_coeffs) // pool_size
            num_pooled_orders = num_pool_outputs * pool_size
            if num_pool_outputs > 0:
                next_layer_homogeneity_coeffs[:num_pool_outputs] = homogeneity_coeffs[:num_pooled_orders].reshape([-1, pool_size]).sum(axis=1)

            if num_pooled_orders < len(homogeneity_coeffs):
                next_layer_homogeneity_coeffs[-1] = homogeneity_coeffs[num_pooled_orders:].sum()

            homogeneity_coeffs = next_layer_homogeneity_coeffs

        if len(homogeneity_coeffs.shape) > 1:
            raise ValueError(f"Implementation error: homogeneity_coeffs should have only 1 "
                             f"entry at this point, and it has {len(homogeneity_coeffs.shape)}.")

        mul_factor_inv_pow = homogeneity_coeffs[0].item()

        if self.linear_output_layer:
            mul_factor_inv_pow += 1

        return mul_factor_inv_pow

    def __checkerboard_pool_permutation(self, num_modes: int):
        return np.arange(num_modes, dtype=np.int)

    def __checkerboard_sq_pool_permutation(self, num_modes: int, pool_size: int):
        if num_modes <= pool_size:
            return np.arange(num_modes, dtype=np.int)

        sqrt_modes = int(math.sqrt(num_modes))
        square_sqrt_modes = sqrt_modes ** 2
        sqrt_pool_size = int(math.sqrt(pool_size))

        indices = torch.arange(square_sqrt_modes, dtype=torch.long).view(sqrt_modes, sqrt_modes)
        square_pooling_indices = indices.unfold(dimension=0, size=sqrt_pool_size, step=sqrt_pool_size)
        square_pooling_indices = square_pooling_indices.unfold(dimension=1, size=sqrt_pool_size, step=sqrt_pool_size)

        last_pool_index = (sqrt_modes // sqrt_pool_size) * sqrt_pool_size

        perm = torch.empty(num_modes, dtype=torch.long)
        perm[:last_pool_index ** 2] = square_pooling_indices.flatten()

        col_remainder = indices[:, last_pool_index:].flatten()
        row_remainder = indices[last_pool_index:, :last_pool_index].flatten()
        remainder = torch.cat([col_remainder, row_remainder])
        perm[last_pool_index ** 2: square_sqrt_modes] = remainder

        if num_modes != square_sqrt_modes:
            perm[square_sqrt_modes:] = torch.arange(start=square_sqrt_modes, end=num_modes)

        return perm

    def __half_pool_permutation(self, num_modes: int):
        perm = np.empty(num_modes, dtype=np.int)

        even_modes = num_modes % 2 == 0
        first_half_end = num_modes // 2

        first_half = np.arange(0, first_half_end)
        second_half = np.arange(first_half_end if even_modes else first_half_end + 1, num_modes)

        perm[0:first_half_end * 2:2] = first_half
        perm[1:first_half_end * 2:2] = second_half

        if not even_modes:
            perm[-1] = first_half_end

        return perm

    def __pool_activations(self, activations, pool_size: int):
        if activations.shape[1] <= 1:
            return activations

        activations = activations.permute(0, 2, 1)

        pool_size = min(pool_size, activations.shape[2])
        unfolded_activations = activations.reshape(activations.shape[0], activations.shape[1], -1, pool_size)
        pooled_unfolded_activations = self.pool_fn(unfolded_activations)

        return pooled_unfolded_activations.permute(0, 2, 1)

    def __compute_tensor_factorization_values(self, inputs: torch.Tensor, is_input_encoded: bool = True):
        if not is_input_encoded:
            inputs = convert_tensor_to_one_hot(inputs, num_options=self.max_mode_dim)

        stacked_per_layer_params = self.create_stacked_per_hidden_layer_params()
        curr_activations = inputs

        for curr_layer_index in range(0, self.num_hidden_layers):
            layer_params = stacked_per_layer_params[curr_layer_index]

            # b --- batch size, o --- order, d --- prev layer width, r --- layer width
            curr_activations = torch.einsum('bod, odr -> bor', curr_activations, layer_params)

            # pooling
            pool_size = self.get_layer_pool_size(curr_layer_index)
            pooling_permutation = self.get_pooling_mode_permutation(curr_activations.size(1), pool_size)
            curr_activations = curr_activations[:, pooling_permutation, :]

            num_orders_to_pool = (curr_activations.size(1) // pool_size) * pool_size
            activations_to_pool = curr_activations[:, :num_orders_to_pool, :]
            pooled_activations = self.__pool_activations(activations_to_pool, pool_size)

            if num_orders_to_pool < curr_activations.size(1):
                remainder_activations_to_pool = curr_activations[:, num_orders_to_pool:, :]
                remainder_pooled_activations = self.__pool_activations(remainder_activations_to_pool, pool_size)
                pooled_activations = torch.cat([pooled_activations, remainder_pooled_activations], dim=1)

            curr_activations = pooled_activations

        # Squeeze order dimension (should be 1)
        curr_activations = curr_activations.squeeze(dim=1)

        output = curr_activations.sum(dim=-1) if not self.linear_output_layer else torch.matmul(curr_activations,
                                                                                                self.output_layer_parameters).squeeze(dim=-1)
        return output

    def __compute_next_layer_tensor(self, tensors_list, filter_params):
        """
        Computes the tensor of next layer.
        :param tensors_list: list of tensors to pool together.
        :param filter_params: tensor of size (prev_layer_rank, next_layer_rank).
        """
        ndims = 0
        request = ""

        for tensor in tensors_list:
            curr_dims = len(tensor.size()) - 1
            request += string.ascii_lowercase[ndims: ndims + curr_dims] + "z,"
            ndims += curr_dims

        request += "z" + string.ascii_lowercase[ndims]
        request += "->" + string.ascii_lowercase[:ndims] + string.ascii_lowercase[ndims]
        return torch.einsum(request, *tensors_list, filter_params)

    def get_layer_pool_size(self, layer_index: int) -> int:
        """
        Returns the pooling size used for the given layer.
        """
        if self.per_layer_pool_size is None:
            return self.pool_size

        return self.per_layer_pool_size[layer_index]

    def get_layer_input_order(self, layer_index: int) -> int:
        """
        Returns the input order for the given layer.
        """
        if layer_index == len(self.hidden_layers_widths):
            return 1

        return len(self.per_hidden_layer_parameter_lists[layer_index])

    def get_pooling_mode_permutation(self, num_modes: int, pool_size: int):
        """
        Returns a list (or numpy array) representing the permutation which orders factors according to the pooling type.
        """
        cached_perm = self.pool_perm_cache.get(num_modes)
        if cached_perm is not None:
            return cached_perm

        if self.pool_pattern == "checkerboard":
            perm = self.__checkerboard_pool_permutation(num_modes)
        elif self.pool_pattern == "checkerboard_sq":
            perm = self.__checkerboard_sq_pool_permutation(num_modes, pool_size)
        elif self.pool_pattern == "half":
            perm = self.__half_pool_permutation(num_modes)
        else:
            raise ValueError(f"Unsupported pool pattern {self.pool_pattern}.")

        self.pool_perm_cache[num_modes] = perm
        return perm

    def get_per_level_mode_tree_nodes(self):
        """
        Returns a list containing one list per level in the factorization mode tree. Each list corresponding to a mode tree level contains tuples
        with the mode indices for each node in the level.
        """
        per_level_mode_tree_nodes = [[(i,) for i in range(self.order)]]
        curr_nodes = [(i,) for i in range(self.order)]

        while len(curr_nodes) != 1:
            next_level_nodes = []
            curr_layer_index = len(per_level_mode_tree_nodes) - 1
            pool_size = self.get_layer_pool_size(curr_layer_index)

            pooling_permutation = self.get_pooling_mode_permutation(len(curr_nodes), pool_size)
            curr_nodes = [curr_nodes[i] for i in pooling_permutation]

            while len(curr_nodes) != 0:
                merged_node = curr_nodes[0]
                for i in range(1, min(pool_size, len(curr_nodes))):
                    merged_node = merged_node + curr_nodes[i]

                next_level_nodes.append(merged_node)
                curr_nodes = curr_nodes[min(pool_size, len(curr_nodes)):]

            per_level_mode_tree_nodes.append(next_level_nodes)
            curr_nodes = next_level_nodes

        return per_level_mode_tree_nodes

    def create_stacked_per_hidden_layer_params(self):
        first_layer_params = self.per_hidden_layer_parameter_lists[0]

        if not self.all_mode_dims_equal:
            first_layer_params = self.__create_zero_padded_params(first_layer_params, self.max_mode_dim, self.hidden_layers_widths[0])
        else:
            first_layer_params = [param for param in first_layer_params]

        per_hidden_layer_params = [torch.stack(first_layer_params)]
        for curr_layer_index in range(1, self.num_hidden_layers):
            hidden_layer_parameter_list = self.per_hidden_layer_parameter_lists[curr_layer_index]
            layer_params = [param for param in hidden_layer_parameter_list]
            per_hidden_layer_params.append(torch.stack(layer_params))

        return per_hidden_layer_params

    def __create_zero_padded_params(self, params_list, target_row_size, target_col_size):
        zero_padded_params = []

        for param in params_list:
            if param.size(0) < target_row_size or param.size(1) < target_col_size:
                zero_padded = torch.zeros(target_row_size, target_col_size, dtype=param.dtype, device=param.device)
                zero_padded[:param.shape[0], :param.shape[1]] = param
                zero_padded_params.append(zero_padded)
            else:
                zero_padded_params.append(param)

        return zero_padded_params

    def normalize(self, new_norm: float, p: str = "fro"):
        """
        Multiplies the parameters by a constant (the same constant for all parameters), such that the end tensor norm is of the value given.
        Computes the whole tensor in the process (do not use for large scale tensors).
        :param new_norm: new value to normalize the end tensor norm to.
        :param p: name of the norm (see PyTorch torch.norm docs), defaults to Frobenius norm.
        """
        tensor = self.compute_tensor()
        params_mul_factor = torch.pow(new_norm / tensor.norm(p=p), 1 / self.__normalize_params_mul_factor_inv_pow)

        for param in self.parameters():
            param.data.mul_(params_mul_factor)

    def __compute_standard_ht_factorization_tensor(self):
        curr_tensors = self.per_hidden_layer_parameter_lists[0]
        for curr_layer_index in range(1, self.num_hidden_layers):
            filter_params_list = self.per_hidden_layer_parameter_lists[curr_layer_index]

            pool_size = self.get_layer_pool_size(curr_layer_index - 1)
            pooling_permutation = self.get_pooling_mode_permutation(len(curr_tensors), pool_size)
            curr_tensors = [curr_tensors[i] for i in pooling_permutation]

            new_tensors = []
            for filter_params in filter_params_list:
                tensors_list = curr_tensors[:pool_size]
                next_layer_tensor = self.__compute_next_layer_tensor(tensors_list, filter_params)
                new_tensors.append(next_layer_tensor)

                curr_tensors = curr_tensors[pool_size:]

            curr_tensors = new_tensors

        pooling_permutation = self.get_pooling_mode_permutation(len(curr_tensors), self.get_layer_pool_size(self.num_hidden_layers - 1))
        curr_tensors = [curr_tensors[i] for i in pooling_permutation]

        output_layer_params = self.output_layer_parameters if self.linear_output_layer else torch.ones(self.hidden_layers_widths[-1], 1,
                                                                                                       dtype=curr_tensors[0].dtype,
                                                                                                       device=curr_tensors[0].device)
        output_tensor = self.__compute_next_layer_tensor(curr_tensors, output_layer_params)
        output_tensor = output_tensor.squeeze(dim=-1)
        return output_tensor

    def __compute_conv_ac_tensor(self, batch_size: int = -1):
        options_per_dim = [range(dim) for dim in self.num_dim_per_mode]
        all_options = torch.tensor(list(itertools.product(*options_per_dim)), device=self.per_hidden_layer_parameter_lists[0][0].device)

        batch_size = batch_size if batch_size > 0 else all_options.shape[0]
        dataloader = FastTensorDataLoader(all_options, batch_size=batch_size, shuffle=False)

        outputs = []
        for batch in dataloader:
            indices_tensor = batch[0]
            outputs.append(self.compute_tensor_at_indices(indices_tensor))

        outputs = torch.cat(outputs)
        return outputs.view(*self.num_dim_per_mode)

    def compute_tensor(self):
        """
        Computes the tensor corresponding to the factorization. If pool type is 'prod' and 'activation' is not given then this is the standard HT
        factorization tensor. Otherwise it computes the function represented by the ConvAC corresponding to the given pool type and activation (i.e.
        outputs over all one-hot inputs arranged as a tensor).
        """
        return self.__compute_standard_ht_factorization_tensor()

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
