from typing import Dict

import torch

import common.utils.module as module_utils
import tensor_factorization.evaluation.tensor_rank_metrics as trm
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo
from common.train.tracked_value import TrackedValue
from tensor_factorization.models import TensorFactorization


class TensorFactorizationRankEvaluator(Evaluator):

    def __init__(self, model: TensorFactorization, cp_tol=1e-6, cp_rank_method="als", device=torch.device("cpu")):
        self.model = model

        self.model = model
        self.cp_tol = cp_tol
        self.cp_rank_method = cp_rank_method
        self.device = device
        self.metric_infos = self.__create_tensor_rank_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_tensor_rank_metric_infos(self):
        metric_infos = {
            "model_tensor_rank": MetricInfo("model_tensor_rank", trm.TensorCPRank(tol=self.cp_tol, method=self.cp_rank_method))
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
            model_tensor = self.model.compute_tensor()

            metric_values = {}
            for name, metric in self.metrics.items():
                value = metric(model_tensor)

                if value != -1:
                    self.tracked_values[name].add_batch_value(value)

                metric_values[name] = value

            return metric_values
