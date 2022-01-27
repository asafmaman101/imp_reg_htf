from typing import Dict
from typing import Sequence

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators import Evaluator, MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo, DummyAveragedMetric
from common.train.tracked_value import TrackedValue
from matrix_factorization.evaluation.dln_metrics import BalancednessMetric, LayerMatrixEntryMetric, LayerSingularValueMetric
from matrix_factorization.evaluation.matrix_metrics import MatrixNormMetric, MatrixEffectiveRankMetric, \
    MatrixRankMetric, MatrixEntryMetric, MatrixDeterminantMetric
from common.evaluation.metrics import ReconstructionError
from matrix_factorization.models.deep_linear_net import DeepLinearNet


class DLNMatrixValidationEvaluator(Evaluator):
    SINGULAR_VALUE_METRIC_NAME_TEMPLATE = "singular_value_{0}"

    def __init__(self, model: DeepLinearNet, target_matrix, norms: Sequence[str] = ("fro", "nuc"),
                 track_rank: bool = True,
                 track_reconstruction_metrics: bool = True, tracked_e2e_value_indices=None,
                 track_singular_values: bool = True,
                 max_num_tracked_sing_vals: int = 10,
                 track_determinant: bool = False,
                 track_per_layer_metrics: bool = False,
                 device=torch.device("cpu")):
        self.model = model
        self.target_matrix = target_matrix
        self.norms = norms
        self.track_rank = track_rank
        self.track_reconstruction_metrics = track_reconstruction_metrics
        self.tracked_e2e_value_indices = tracked_e2e_value_indices if tracked_e2e_value_indices is not None else []
        self.track_singular_values = track_singular_values
        self.num_tracked_sing_vals = min(self.model.input_dim, self.model.output_dim)
        if max_num_tracked_sing_vals >= 0:
            self.num_tracked_sing_vals = min(self.num_tracked_sing_vals, max_num_tracked_sing_vals)

        self.device = device

        self.matrix_metric_infos = self.__create_matrix_metric_infos(norms, track_rank, track_determinant)
        self.matrix_metric_infos.update(self.__create_matrix_values_metric_infos(self.tracked_e2e_value_indices))
        self.matrix_metrics = {name: metric_info.metric for name, metric_info in self.matrix_metric_infos.items()}

        self.__per_layer_metric_infos = self.__create_per_layer_metric_infos(track_per_layer_metrics)
        self.per_layer_metrics = {name: metric_info.metric for name, metric_info in
                                  self.__per_layer_metric_infos.items()}

        self.combined_metric_infos = {}
        self.combined_metric_infos.update(self.matrix_metric_infos)
        self.combined_metric_infos.update(self.__per_layer_metric_infos)

        if track_reconstruction_metrics:
            self.reconstruction_metric_infos = self.__create_reconstruction_metric_infos()
            self.reconstruction_metrics = {name: metric_info.metric for name, metric_info in
                                           self.reconstruction_metric_infos.items()}
            self.combined_metric_infos.update(self.reconstruction_metric_infos)

        if self.track_singular_values:
            self.singular_values_metric_infos = self.__create_singular_values_metric_infos()
            self.singular_values_metrics = {name: metric_info.metric for name, metric_info in
                                            self.singular_values_metric_infos.items()}
            self.combined_metric_infos.update(self.singular_values_metric_infos)

        self.combined_metrics = {metric_info.name: metric_info.metric for name, metric_info in self.combined_metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.combined_metric_infos)

    def __create_matrix_metric_infos(self, norms, track_rank, track_determinant):
        metric_infos = {}
        for norm in norms:
            metric_name = f"{norm}_e2e_norm"
            metric = MatrixNormMetric(norm)
            metric_info = MetricInfo(metric_name, metric)

            metric_infos[metric_name] = metric_info

        if track_rank:
            metric_name = "e2e_effective_rank"
            metric = MatrixEffectiveRankMetric()
            metric_info = MetricInfo(metric_name, metric)

            metric_infos[metric_name] = metric_info

            metric_name = "e2e_rank"
            metric = MatrixRankMetric()
            metric_info = MetricInfo(metric_name, metric)

            metric_infos[metric_name] = metric_info

        if track_determinant:
            metric_infos["e2e_determinant"] = MetricInfo("e2e_determinant", MatrixDeterminantMetric())

        return metric_infos

    def __create_matrix_values_metric_infos(self, tracked_e2e_value_indices, use_same_plot=True):
        metric_infos = {}
        for i, j in tracked_e2e_value_indices:
            metric_name = f"e2e_{i}_{j}_value"
            metric = MatrixEntryMetric(i, j)
            metric_tag = "E2E Matrix Values" if use_same_plot else ""
            metric_infos[metric_name] = MetricInfo(metric_name, metric, tag=metric_tag)

        return metric_infos

    def __create_per_layer_metric_infos(self, track_per_layer_metrics, use_same_plot=True):
        metric_infos = {}
        if not track_per_layer_metrics:
            return metric_infos

        for l in range(self.model.depth - 1):
            balancedness_metric = BalancednessMetric(first_layer_index=l)
            metric_name = f"balancedness_fro_dist_layers_{l}_{l + 1}"
            metric_tag = "Balancedness Fro Dist" if use_same_plot else ""
            metric_infos[metric_name] = MetricInfo(metric_name, balancedness_metric, tag=metric_tag)

        for l in range(self.model.depth):
            for i, j in self.tracked_e2e_value_indices:
                metric_name = f"layer_{l}_entry_{i}_{j}_value"
                metric = LayerMatrixEntryMetric(layer_index=l, i=i, j=j)
                metric_tag = f"Layer {l} Values" if use_same_plot else ""
                metric_infos[metric_name] = MetricInfo(metric_name, metric, tag=metric_tag)

            for j in range(self.num_tracked_sing_vals):
                layer_sing_val_metric = LayerSingularValueMetric(layer_index=l, sing_value_index=j)
                metric_name = f"layer_{l}_singular_value_{j}"
                metric_tag = "Layers Singular Values" if use_same_plot else ""
                metric_infos[metric_name] = MetricInfo(metric_name, layer_sing_val_metric, tag=metric_tag)

        return metric_infos

    def __create_reconstruction_metric_infos(self):
        return {
            "reconstruction_error": MetricInfo("reconstruction_error", ReconstructionError()),
            "normalized_reconstruction_error": MetricInfo("normalized_reconstruction_error",
                                                          ReconstructionError(normalized=True))
        }

    def __create_singular_values_metric_infos(self, use_same_plot=True):
        metric_infos = {}

        for i in range(self.num_tracked_sing_vals):
            singular_value_metric_name = self.SINGULAR_VALUE_METRIC_NAME_TEMPLATE.format(i)
            metric_tag = "e2e singular values" if use_same_plot else ""
            metric_infos[singular_value_metric_name] = MetricInfo(singular_value_metric_name, DummyAveragedMetric(),
                                                                  tag=metric_tag)

        return metric_infos

    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        return self.combined_metric_infos

    def get_metrics(self) -> Dict[str, Metric]:
        return self.combined_metrics

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return self.tracked_values

    def evaluate(self) -> dict:
        with torch.no_grad():
            self.model.to(self.device)

            end_to_end_matrix = self.model.compute_prod_matrix()
            end_to_end_matrix = end_to_end_matrix.to(self.device)
            target_matrix = self.target_matrix.to(self.device)

            metric_values = {}

            for name, metric in self.per_layer_metrics.items():
                value = metric(self.model)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            for name, metric in self.matrix_metrics.items():
                value = metric(end_to_end_matrix)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            if self.track_reconstruction_metrics:
                for name, metric in self.reconstruction_metrics.items():
                    value = metric(end_to_end_matrix, target_matrix)
                    self.tracked_values[name].add_batch_value(value)
                    metric_values[name] = value

            if self.track_singular_values:
                svd_result = torch.svd(end_to_end_matrix, compute_uv=False)
                singular_values = svd_result.S
                for i in range(self.num_tracked_sing_vals):
                    singular_value_metric_name = self.SINGULAR_VALUE_METRIC_NAME_TEMPLATE.format(i)
                    value = singular_values[i].item()
                    self.singular_values_metrics[singular_value_metric_name](value)
                    self.tracked_values[singular_value_metric_name].add_batch_value(value)
                    metric_values[singular_value_metric_name] = value

            return metric_values
