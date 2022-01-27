import torch

from common.evaluation.metrics.metric import AveragedMetric


class MatrixNormMetric(AveragedMetric):

    def __init__(self, norm: str = "fro"):
        super().__init__()
        self.norm = norm

    def _calc_metric(self, matrix: torch.Tensor):
        return torch.norm(matrix, p=self.norm).item(), 1


class MatrixRankMetric(AveragedMetric):

    def _calc_metric(self, matrix: torch.Tensor):
        return torch.linalg.matrix_rank(matrix).item(), 1


class MatrixEffectiveRankMetric(AveragedMetric):

    def _calc_metric(self, matrix: torch.Tensor):
        effective_rank = matrix_effective_rank(matrix)
        return effective_rank, 1


def matrix_effective_rank(matrix):
    """
    Calculates the effective rank of the matrix.
    :param matrix: torch matrix of size (N, M)
    :return: Effective rank of the matrix.
    """
    svd_result = torch.svd(matrix, compute_uv=False)
    singular_values = svd_result.S
    non_zero_singular_values = singular_values[singular_values != 0]
    normalized_non_zero_singular_values = non_zero_singular_values / non_zero_singular_values.sum()

    singular_values_entropy = -(normalized_non_zero_singular_values * torch.log(normalized_non_zero_singular_values)).sum()
    return torch.exp(singular_values_entropy).item()


class MatrixEntryMetric(AveragedMetric):

    def __init__(self, i, j):
        super().__init__()
        self.i = i
        self.j = j

    def _calc_metric(self, matrix: torch.Tensor):
        return matrix[self.i, self.j].item(), 1


class MatrixDeterminantMetric(AveragedMetric):

    def _calc_metric(self, matrix: torch.Tensor):
        min_dim = min(matrix.size(0), matrix.size(1))
        matrix = matrix[: min_dim, : min_dim]
        return torch.det(matrix).item(), 1
