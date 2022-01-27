import torch

from common.evaluation.metrics.metric import AveragedMetric
from matrix_factorization.models.deep_linear_net import DeepLinearNet


class BalancednessMetric(AveragedMetric):

    def __init__(self, first_layer_index, norm="fro", normalize=True):
        super().__init__()
        self.first_layer_index = first_layer_index
        self.norm = norm
        self.normalize = normalize

    def _calc_metric(self, dln: DeepLinearNet):
        first_layer = dln.layers[self.first_layer_index].detach()
        second_layer = dln.layers[self.first_layer_index + 1].detach()

        first_layer_mul = torch.matmul(first_layer.T, first_layer)
        second_layer_mul = torch.matmul(second_layer, second_layer.T)
        balancedness_fro_dist = torch.norm(first_layer_mul - second_layer_mul, p=self.norm)
        normalization = (torch.norm(first_layer_mul, p=self.norm) + torch.norm(second_layer_mul, p=self.norm)) / 2

        if self.normalize:
            balancedness_fro_dist = balancedness_fro_dist / normalization

        return balancedness_fro_dist.item(), 1


class LayerMatrixEntryMetric(AveragedMetric):

    def __init__(self, layer_index, i, j):
        super().__init__()
        self.layer_index = layer_index
        self.i = i
        self.j = j

    def _calc_metric(self, dln: DeepLinearNet):
        layer = dln.layers[self.layer_index].detach()
        return layer[self.i, self.j].item(), 1


class LayerSingularValueMetric(AveragedMetric):

    def __init__(self, layer_index, sing_value_index):
        super().__init__()
        self.layer_index = layer_index
        self.sing_value_index = sing_value_index

    def _calc_metric(self, dln: DeepLinearNet):
        layer = dln.layers[self.layer_index].detach()
        svd_result = torch.svd(layer, compute_uv=False)
        singular_values = svd_result.S

        return singular_values[self.sing_value_index].item(), 1


class DLNLayerWeightMatrixNorm(AveragedMetric):

    def __init__(self, layer_index: int, norm: str = "fro"):
        super().__init__()
        self.layer_index = layer_index
        self.norm = norm

    def _calc_metric(self, dln: DeepLinearNet):
        layers = dln.layers
        layer_weight_mat = layers[self.layer_index]
        return torch.norm(layer_weight_mat, p=self.norm).item(), 1


class DLNLayerWeightMatrixColNorm(AveragedMetric):

    def __init__(self, layer_index: int, col_index: int, norm: str = "fro"):
        super().__init__()
        self.layer_index = layer_index
        self.norm = norm
        self.col_index = col_index

    def _calc_metric(self, dln: DeepLinearNet):
        layers = dln.layers
        layer_weight_mat = layers[self.layer_index]
        return torch.norm(layer_weight_mat[:, self.col_index], p=self.norm).item(), 1


class DLNLayerWeightMatrixRowNorm(AveragedMetric):

    def __init__(self, layer_index: int, row_index: int, norm: str = "fro"):
        super().__init__()
        self.layer_index = layer_index
        self.norm = norm
        self.row_index = row_index

    def _calc_metric(self, dln: DeepLinearNet):
        layers = dln.layers
        layer_weight_mat = layers[self.layer_index]
        return torch.norm(layer_weight_mat[self.row_index, :], p=self.norm).item(), 1
