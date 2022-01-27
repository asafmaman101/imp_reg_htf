from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data

import common.utils.module as module_utils
from common.evaluation.evaluators import Evaluator
from common.evaluation.evaluators import MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo
from common.evaluation.metrics import ReconstructionError
from common.train.tracked_value import TrackedValue


class ReconstructionEvaluator(Evaluator):

    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, device=torch.device("cpu")):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.reconstruction_metric_infos = self.__create_tensor_reconstruction_metric_infos()

        self.metric_infos = self.reconstruction_metric_infos
        self.metrics = {metric_info.name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

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

            metric_values = {}
            predictions = []
            targets = []

            for x, y in self.dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                predictions.append(self.model(x))
                targets.append(y)

            predictions = torch.cat(predictions)
            targets = torch.cat(targets)

            for name, metric_info in self.reconstruction_metric_infos.items():
                metric = metric_info.metric

                value = metric(predictions, targets)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            return metric_values
