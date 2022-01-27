from typing import Dict

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators import Evaluator
from common.evaluation.evaluators import MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo
from common.evaluation.metrics import ReconstructionError
from common.train.tracked_value import TrackedValue
from tensor_factorization.evaluation.tensor_metrics import TensorNormMetric
from tensor_factorization.models import TensorFactorization


class TensorEvaluator(Evaluator):

    def __init__(self, model: TensorFactorization, target_tensor: torch.Tensor, device=torch.device("cpu")):
        self.model = model
        self.target_tensor = target_tensor
        self.device = device

        self.tensor_norms_metric_infos = self.__create_tensor_norms_metric_infos()
        self.reconstruction_metric_infos = self.__create_tensor_reconstruction_metric_infos()

        self.metric_infos = {}
        self.metric_infos.update(self.tensor_norms_metric_infos)
        self.metric_infos.update(self.reconstruction_metric_infos)

        self.metrics = {metric_info.name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_tensor_norms_metric_infos(self):
        metric_infos = {
            "tensor_fro_norm": MetricInfo("tensor_fro_norm", TensorNormMetric(norm="fro"))
        }

        return metric_infos

    def __create_tensor_reconstruction_metric_infos(self):
        metric_infos = {
            "normalized_reconstruction_error": MetricInfo("normalized_reconstruction_error", ReconstructionError(normalized=True))
        }

        return metric_infos

    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        return self.metric_infos

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return self.tracked_values

    def evaluate(self) -> dict:
        with torch.no_grad():
            self.model.to(self.device)
            tensor = self.model.compute_tensor()
            target_tensor = self.target_tensor.to(self.device)

            metric_values = {}
            for name, metric_info in self.tensor_norms_metric_infos.items():
                metric = metric_info.metric

                value = metric(tensor)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            for name, metric_info in self.reconstruction_metric_infos.items():
                metric = metric_info.metric

                value = metric(tensor, target_tensor)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            return metric_values
