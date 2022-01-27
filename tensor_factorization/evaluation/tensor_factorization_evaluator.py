from typing import Dict

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators import Evaluator
from common.evaluation.evaluators import MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo, DummyAveragedMetric
from common.train.tracked_value import TrackedValue
from tensor_factorization.evaluation.cp_factorization_metrics import CPFactorColNorm, CPComponentFroNorm, CPFactorNorm
from tensor_factorization.models import CPTensorFactorization


class TensorFactorizationEvaluator(Evaluator):
    TOP_FACTOR_COL_NORM_METRIC_NAME_TEMPLATE = "factor_{0}_top_col_{1}_fro_norm"
    TOP_COMPONENT_NORM_METRIC_NAME_TEMPLATE = "top_component_{0}_fro_norm"

    def __init__(self, cp_model: CPTensorFactorization, track_all_col_norms: bool = False, track_all_component_norms: bool = False,
                 track_factor_norms: bool = True, track_top_col_norms: bool = True, track_top_component_norms: bool = True,
                 num_top_to_track: int = 10, device=torch.device("cpu")):
        self.cp_model = cp_model
        self.track_all_factor_col_norms = track_all_col_norms
        self.track_all_component_norms = track_all_component_norms
        self.track_factor_norms = track_factor_norms
        self.num_top_to_track = num_top_to_track
        self.track_top_factor_col_norms = track_top_col_norms
        self.track_top_component_norms = track_top_component_norms
        self.device = device

        self.metric_infos = {}
        self.manual_metric_infos = {}

        if self.track_all_factor_col_norms:
            self.metric_infos.update(self.__create_factor_col_norms_metric_infos())
        elif self.track_top_factor_col_norms:
            self.manual_metric_infos.update(self.__create_top_factor_col_norms_metric_infos())

        if self.track_all_component_norms:
            self.metric_infos.update(self.__create_component_norms_metric_infos())
        elif self.track_top_component_norms:
            self.manual_metric_infos.update(self.__create_top_component_norms_metric_infos())

        if self.track_factor_norms:
            self.metric_infos.update(self.__create_factor_norms_metric_infos())

        self.metric_infos.update(self.manual_metric_infos)
        self.metrics = {metric_info.name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __create_factor_col_norms_metric_infos(self):
        metric_infos = {}
        for factor_index in range(self.cp_model.order):
            for col_index in range(self.cp_model.rank):
                factor_col_norm_metric = CPFactorColNorm(factor_index=factor_index, col_index=col_index, norm="fro")
                metric_name = f"factor_{factor_index}_col_{col_index}_fro_norm"
                metric_tag = f"factor {factor_index} cols Fro norm"
                metric_infos[metric_name] = MetricInfo(metric_name, factor_col_norm_metric, tag=metric_tag)

        return metric_infos

    def __create_top_factor_col_norms_metric_infos(self):
        metric_infos = {}
        for factor_index in range(self.cp_model.order):
            for top_index in range(min(self.num_top_to_track, self.cp_model.rank)):
                top_factor_col_norm_metric = DummyAveragedMetric()
                metric_name = self.TOP_FACTOR_COL_NORM_METRIC_NAME_TEMPLATE.format(factor_index, top_index)
                metric_tag = f"factor {factor_index} top cols Fro norm"
                metric_infos[metric_name] = MetricInfo(metric_name, top_factor_col_norm_metric, tag=metric_tag)

        return metric_infos

    def __create_component_norms_metric_infos(self):
        metric_infos = {}
        for component_index in range(self.cp_model.rank):
            component_fro_norm_metric = CPComponentFroNorm(component_index=component_index)
            metric_name = f"component_{component_index}_fro_norm"
            metric_tag = f"components Fro norm"
            metric_infos[metric_name] = MetricInfo(metric_name, component_fro_norm_metric, tag=metric_tag)

        return metric_infos

    def __create_top_component_norms_metric_infos(self):
        metric_infos = {}
        for top_index in range(min(self.num_top_to_track, self.cp_model.rank)):
            component_fro_norm_metric = DummyAveragedMetric()
            metric_name = self.TOP_COMPONENT_NORM_METRIC_NAME_TEMPLATE.format(top_index)
            metric_tag = f"top components Fro norm"
            metric_infos[metric_name] = MetricInfo(metric_name, component_fro_norm_metric, tag=metric_tag)

        return metric_infos

    def __create_factor_norms_metric_infos(self):
        metric_infos = {}
        for factor_index in range(self.cp_model.order):
            factor_norm_metric = CPFactorNorm(factor_index=factor_index, norm="fro")
            metric_name = f"factor_{factor_index}_fro_norm"
            metric_tag = "factors Fro norm"
            metric_infos[metric_name] = MetricInfo(metric_name, factor_norm_metric, tag=metric_tag)

        return metric_infos

    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        return self.metric_infos

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return self.tracked_values

    def __compute_top_factor_col_norms_metric_infos(self, metric_values: dict):
        for factor_index in range(self.cp_model.order):
            factor = self.cp_model.factors[factor_index]
            factor_col_norms = torch.norm(factor, p="fro", dim=0)

            num_top_cols = min(self.num_top_to_track, self.cp_model.rank)
            top_norm_values = torch.topk(factor_col_norms, k=num_top_cols)[0]

            for top_index in range(num_top_cols):
                metric_name = self.TOP_FACTOR_COL_NORM_METRIC_NAME_TEMPLATE.format(factor_index, top_index)
                metric = self.manual_metric_infos[metric_name].metric
                tracked_value = self.tracked_values[metric_name]

                value = top_norm_values[top_index].item()
                metric(value)
                tracked_value.add_batch_value(value)
                metric_values[metric_name] = value

    def __compute_top_component_norms_metric_infos(self, metric_values: dict):
        factors = self.cp_model.factors
        factor_col_norms = [torch.norm(factor, p="fro", dim=0) for factor in factors]
        stacked_factor_col_norms = torch.stack(factor_col_norms, dim=0)
        component_fro_norms = torch.prod(stacked_factor_col_norms, dim=0)

        num_top_components = min(self.num_top_to_track, self.cp_model.rank)
        top_norm_values = torch.topk(component_fro_norms, k=num_top_components)[0]

        for top_index in range(num_top_components):
            metric_name = self.TOP_COMPONENT_NORM_METRIC_NAME_TEMPLATE.format(top_index)
            metric = self.manual_metric_infos[metric_name].metric
            tracked_value = self.tracked_values[metric_name]

            value = top_norm_values[top_index].item()
            metric(value)
            tracked_value.add_batch_value(value)
            metric_values[metric_name] = value

    def evaluate(self) -> dict:
        with torch.no_grad():
            self.cp_model.to(self.device)

            metric_values = {}
            for name, metric in self.metrics.items():
                if name in self.manual_metric_infos:
                    continue

                value = metric(self.cp_model)
                self.tracked_values[name].add_batch_value(value)
                metric_values[name] = value

            if self.track_top_factor_col_norms:
                self.__compute_top_factor_col_norms_metric_infos(metric_values)

            if self.track_top_component_norms:
                self.__compute_top_component_norms_metric_infos(metric_values)

            return metric_values
