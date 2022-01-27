from typing import Dict

import torch

import common.utils.module as module_utils
import tensor_factorization.evaluation.tensor_rank_metrics as trm
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo
from common.train.tracked_value import TrackedValue
from tensor_factorization.models import TensorFactorization


class TensorFactorizationMatricizationsEffectiveRankEvaluator(Evaluator):

    def __init__(self, model: TensorFactorization, num_row_modes_options=(1,), device=torch.device("cpu")):
        self.model = model

        self.num_row_modes_options = num_row_modes_options
        self.device = device
        self.metric_infos = self.__create_tensor_rank_metric_infos()
        self.metrics = {name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_tensor_rank_metric_infos(self):
        metric_infos = {}

        for num_row_modes in self.num_row_modes_options:
            num_col_modes = self.model.order - num_row_modes
            metric_name = f"model_{num_row_modes}_{num_col_modes}_matricization_effective_ranks"

            metric_infos[f"{metric_name}_mean"] = MetricInfo(f"{metric_name}_mean",
                                                             trm.TensorMatricizationsEffectiveRanksMean(num_row_modes=num_row_modes))
            metric_infos[f"{metric_name}_std"] = MetricInfo(f"{metric_name}_std",
                                                            trm.TensorMatricizationsEffectiveRanksSTD(num_row_modes=num_row_modes))

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
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            return metric_values
