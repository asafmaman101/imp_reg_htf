from typing import Sequence

import torch
import torch.nn as nn


class DeepLinearNet(nn.Module):
    """
    Deep Linear Network model.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = None, weight_init_type: str = "normal", init_std: float = 1e-2):
        """
        :param input_dim: Number of input dimensions.
        :param output_dim: Number of output dimensions.
        :param hidden_dims: Sequence of hidden dimensions.
        :param weight_init_type: Str code for type of initialization. Supports: 'normal', 'identity'.
        :param init_std: Standard deviation/multiplicative factor of the initialization.
        """

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.depth = len(hidden_dims) + 1
        self.weight_init_type = weight_init_type
        self.init_std = init_std
        self.layers = nn.ParameterList(self.__create_layers())

    def __initialize_new_layer(self, input_dim: int, output_dim: int):
        if self.weight_init_type == "normal":
            weights = torch.randn(input_dim, output_dim) * self.init_std
            return nn.Parameter(weights, requires_grad=True)

        if self.weight_init_type == "identity":
            weights = self.init_std * torch.eye(input_dim, output_dim, dtype=torch.float)
            return nn.Parameter(weights, requires_grad=True)

        raise ValueError(f"Unsupported weight initialization type: {self.weight_init_type}")

    def __create_layers(self):
        layers = []
        curr_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(self.__initialize_new_layer(curr_dim, hidden_dim))
            curr_dim = hidden_dim

        layers.append(self.__initialize_new_layer(curr_dim, self.output_dim))
        return layers

    def __init_diag_matrix(self, rows, cols, diag_elements):
        mat = torch.zeros(rows, cols)
        mat[range(len(diag_elements)), range(len(diag_elements))] = diag_elements
        return mat

    def compute_prod_matrix(self):
        return self.__compute_prod_mat_matrix_from_layers(self.layers)

    def __compute_prod_mat_matrix_from_layers(self, layers):
        curr = layers[0]
        for i in range(1, len(layers)):
            curr = torch.matmul(curr, layers[i])

        return curr

    def forward(self, x):
        prod_mat = self.compute_prod_matrix()
        return torch.matmul(x, prod_mat)

    def inner_product(self, matrix):
        prod_mat = self.compute_prod_matrix()
        return (prod_mat * matrix).sum()
