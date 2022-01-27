from matrix_factorization.models.deep_linear_net import DeepLinearNet


class DLNModelFactory:

    @staticmethod
    def create_same_dim_deep_linear_network(input_dim: int, output_dim: int, depth: int, weight_init_type: str = "normal", init_std: float = 1e-3):
        hidden_dims = [min(input_dim, output_dim)] * (depth - 1)
        return DeepLinearNet(input_dim, output_dim, hidden_dims=hidden_dims, weight_init_type=weight_init_type, init_std=init_std)
