import torch

from common.evaluation.metrics.metric import AveragedMetric


class TensorNormMetric(AveragedMetric):

    def __init__(self, norm: str = "fro"):
        super().__init__()
        self.norm = norm

    def _calc_metric(self, tensor: torch.Tensor):
        return torch.norm(tensor, p=self.norm).item(), 1
