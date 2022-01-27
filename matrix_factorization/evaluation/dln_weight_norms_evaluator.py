from typing import Dict

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators import Evaluator
from common.evaluation.evaluators import MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo
from common.train.tracked_value import TrackedValue
from matrix_factorization.evaluation.dln_metrics import DLNLayerWeightMatrixColNorm, DLNLayerWeightMatrixNorm, DLNLayerWeightMatrixRowNorm
from matrix_factorization.models.deep_linear_net import DeepLinearNet


class DLNWeightNormsEvaluator(Evaluator):

    def __init__(self, model: DeepLinearNet, track_col_norms: bool = True, track_row_norms: bool = True, track_matrix_norms: bool = True,
                 device=torch.device("cpu")):
        self.model = model
        self.track_col_norms = track_col_norms
        self.track_row_norms = track_row_norms
        self.track_matrix_norms = track_matrix_norms
        self.device = device

        self.metric_infos = {}

        if self.track_col_norms:
            self.metric_infos.update(self.__create_col_norms_metric_infos())

        if self.track_row_norms:
            self.metric_infos.update(self.__create_row_norms_metric_infos())

        if self.track_matrix_norms:
            self.metric_infos.update(self.__create_matrix_norms_metric_infos())

        self.metrics = {metric_info.name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_col_norms_metric_infos(self):
        metric_infos = {}
        for layer_index in range(self.model.depth):
            num_rows_in_layer = self.model.layers[layer_index].size(1)
            for col_index in range(num_rows_in_layer):
                layer_fro_norm_metric = DLNLayerWeightMatrixColNorm(layer_index=layer_index, col_index=col_index, norm="fro")
                metric_name = f"layer_{layer_index}_col_{col_index}_fro_norm"
                metric_tag = f"layer {layer_index} cols Fro norm"
                metric_infos[metric_name] = MetricInfo(metric_name, layer_fro_norm_metric, tag=metric_tag)

        return metric_infos

    def __create_row_norms_metric_infos(self):
        metric_infos = {}
        for layer_index in range(self.model.depth):
            num_rows_in_layer = self.model.layers[layer_index].size(0)
            for row_index in range(num_rows_in_layer):
                layer_fro_norm_metric = DLNLayerWeightMatrixRowNorm(layer_index=layer_index, row_index=row_index, norm="fro")
                metric_name = f"layer_{layer_index}_row_{row_index}_fro_norm"
                metric_tag = f"layer {layer_index} rows Fro norm"
                metric_infos[metric_name] = MetricInfo(metric_name, layer_fro_norm_metric, tag=metric_tag)

        return metric_infos

    def __create_matrix_norms_metric_infos(self):
        metric_infos = {}
        for layer_index in range(self.model.depth):
            layer_fro_norm_metric = DLNLayerWeightMatrixNorm(layer_index=layer_index, norm="fro")
            metric_name = f"layer_{layer_index}_fro_norm"
            metric_tag = "layers Fro norm"
            metric_infos[metric_name] = MetricInfo(metric_name, layer_fro_norm_metric, tag=metric_tag)

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
            for name, metric in self.metrics.items():
                value = metric(self.model)
                metric_values[name] = value

            return metric_values
