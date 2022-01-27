from typing import Dict
from typing import List, Tuple

import numpy as np
import torch

import common.utils.tensor as tensor_utils
from common.evaluation.evaluators import Evaluator
from common.evaluation.evaluators import MetricsEvaluator
from common.evaluation.metrics import Metric, MetricInfo, DummyAveragedMetric
from common.train.tracked_value import TrackedValue
from tensor_factorization.models import HierarchicalTensorFactorization


class HTFactorizationEvaluator(Evaluator):
    TOP_WEIGHT_VECTOR_NORM_NAME_TEMPLATE = "l_{0}_f_{1}_top_{2}_wv_norm"
    MATRICIZATION_EFFECTIVE_RANK_NAME_TEMPLATE = "mat_{0}_eff_rank"
    MATRICIZATION_SING_VAL_NAME_TEMPLATE = "mat_{0}_sing_val_{1}"
    TOP_LOCAL_COMPONENT_NORM_NAME_TEMPLATE = "l_{0}_f_{1}_top_{2}_local_comp_norm"
    LOCAL_COMPONENT_NORM_NAME_TEMPLATE = "l_{0}_f_{1}_r_{2}_local_comp_norm"
    NEG_END_GRAD_NORMALIZED_LOCAL_COMPONENT_INPROD_NAME_TEMPLATE = "l_{0}_f_{1}_r_{2}_neg_end_grad_cosine_sim"

    def __init__(self, ht_model: HierarchicalTensorFactorization,
                 track_top_weight_vector_norms_per_layer_and_factor: bool = True,
                 track_top_local_components_norms: bool = True,
                 num_top_weight_vector_norms_to_track_per_layer_and_factor: int = 10,
                 track_mode_tree_mat_effective_ranks: bool = True,
                 track_mode_tree_mat_top_sing_vals: bool = True,
                 num_top_sing_vals_to_track: int = 10,
                 additional_matricizations_to_track: List[Tuple[int, ...]] = None,
                 compute_svd_device=torch.device("cpu"),
                 device=torch.device("cpu")):
        self.ht_model = ht_model
        self.track_top_weight_vector_norms = track_top_weight_vector_norms_per_layer_and_factor
        self.track_top_local_components_norms = track_top_local_components_norms
        self.num_top_weight_vector_norms_to_track_per_layer_and_factor = num_top_weight_vector_norms_to_track_per_layer_and_factor
        self.track_mode_tree_mat_effective_ranks = track_mode_tree_mat_effective_ranks
        self.track_mode_tree_mat_top_sing_vals = track_mode_tree_mat_top_sing_vals
        self.num_top_sing_vals_to_track = num_top_sing_vals_to_track
        self.additional_matricizations_to_track = additional_matricizations_to_track
        self.matricizations_to_track = self.__get_matricizations_to_track(additional_matricizations_to_track)
        self.compute_svd_device = compute_svd_device
        self.device = device

        self.metric_infos = {}
        self.manual_metric_infos = {}

        if self.track_top_weight_vector_norms:
            self.manual_metric_infos.update(self.__create_top_weight_vector_norms_metric_infos())

        if self.track_top_local_components_norms:
            self.manual_metric_infos.update(self.__create_top_local_components_norms_metric_infos())

        if self.track_mode_tree_mat_effective_ranks:
            self.manual_metric_infos.update(self.__create_mode_tree_mat_effective_ranks_metric_infos())

        if self.track_mode_tree_mat_top_sing_vals:
            self.manual_metric_infos.update(self.__create_mode_tree_mat_sing_vals_metric_infos())

        self.metric_infos.update(self.manual_metric_infos)
        self.metrics = {metric_info.name: metric_info.metric for name, metric_info in self.metric_infos.items()}
        self.tracked_values = MetricsEvaluator.create_tracked_values_for_metrics(self.metric_infos)

    def __get_matricizations_to_track(self, additional_matricizations_to_track: List[Tuple[int, ...]] = None):
        per_layer_modes = self.ht_model.get_per_level_mode_tree_nodes()
        per_layer_modes = per_layer_modes[:-1]  # Remove last layer which contains all modes as its corresponding matricization is just a vector

        matricizations_to_track = []
        for layer_modes in per_layer_modes:
            for matricization_row_modes in layer_modes:
                matricizations_to_track.append(tuple(matricization_row_modes))

        if additional_matricizations_to_track:
            for matricization_row_modes in additional_matricizations_to_track:
                matricization_row_modes_tuple = tuple(matricization_row_modes)
                if matricization_row_modes_tuple not in matricizations_to_track:
                    matricizations_to_track.append(matricization_row_modes_tuple)

        return matricizations_to_track

    def __create_top_weight_vector_norms_metric_infos(self):
        metric_infos = {}
        for layer_index in range(self.ht_model.num_hidden_layers):
            layer_width = self.ht_model.hidden_layers_widths[layer_index]
            layer_num_factors = len(self.ht_model.per_hidden_layer_parameter_lists[layer_index])

            for factor_index in range(layer_num_factors):
                for top_index in range(min(self.num_top_weight_vector_norms_to_track_per_layer_and_factor, layer_width)):
                    top_layer_factor_wv_norm_metric = DummyAveragedMetric()
                    metric_name = self.TOP_WEIGHT_VECTOR_NORM_NAME_TEMPLATE.format(layer_index, factor_index, top_index)
                    metric_tag = f"l {layer_index} f {factor_index} top wv norm"
                    metric_infos[metric_name] = MetricInfo(metric_name, top_layer_factor_wv_norm_metric, tag=metric_tag)

        return metric_infos

    def __create_top_local_components_norms_metric_infos(self):
        metric_infos = {}

        for layer_index in range(self.ht_model.num_hidden_layers):
            layer_width = self.ht_model.hidden_layers_widths[layer_index]
            layer_num_factors = len(self.ht_model.per_hidden_layer_parameter_lists[layer_index])
            pool_size = self.ht_model.get_layer_pool_size(layer_index)

            pooled_factors_indices_seq = self.__get_pooled_factors_indices_seq(layer_num_factors, pool_size)
            for pooled_factors_indices in pooled_factors_indices_seq:
                for top_index in range(min(self.num_top_weight_vector_norms_to_track_per_layer_and_factor, layer_width)):
                    top_local_comp_norms_metric = DummyAveragedMetric()
                    factor_indices_str = "_".join([str(i) for i in pooled_factors_indices])
                    metric_name = self.TOP_LOCAL_COMPONENT_NORM_NAME_TEMPLATE.format(layer_index, factor_indices_str, top_index)
                    metric_tag = f"l {layer_index} f {factor_indices_str} top local components norm"

                    metric_infos[metric_name] = MetricInfo(metric_name, top_local_comp_norms_metric, tag=metric_tag)

        return metric_infos

    def __get_pooled_factors_indices_seq(self, layer_num_factors: int, pool_size: int):
        pooling_permutation = self.ht_model.get_pooling_mode_permutation(num_modes=layer_num_factors, pool_size=pool_size)
        pooled_factor_indices_seq = []
        for factor_index in range(0, layer_num_factors, pool_size):
            pooled_factors_indices = list(pooling_permutation[factor_index: min(layer_num_factors, factor_index + pool_size)])
            pooled_factor_indices_seq.append(pooled_factors_indices)

        return pooled_factor_indices_seq

    def __create_matricization_modes_str(self, matricization_row_modes):
        return "_".join([str(row) for row in matricization_row_modes])

    def __compute_matricization_min_dim(self, matricization_row_modes):
        num_dim_per_mode = self.ht_model.num_dim_per_mode
        col_modes = [j for j in range(len(num_dim_per_mode)) if j not in matricization_row_modes]

        row_dim = np.prod([num_dim_per_mode[r] for r in matricization_row_modes]).item()
        col_dim = np.prod([num_dim_per_mode[c] for c in col_modes]).item()
        return min(row_dim, col_dim)

    def __create_mode_tree_mat_effective_ranks_metric_infos(self):
        metric_infos = {}
        for matricization_row_modes in self.matricizations_to_track:
            matricization_eff_rank_metric = DummyAveragedMetric()
            metric_name = self.MATRICIZATION_EFFECTIVE_RANK_NAME_TEMPLATE.format(self.__create_matricization_modes_str(matricization_row_modes))
            metric_tag = f"mats effective rank"
            metric_infos[metric_name] = MetricInfo(metric_name, matricization_eff_rank_metric, tag=metric_tag)

        return metric_infos

    def __create_mode_tree_mat_sing_vals_metric_infos(self):
        metric_infos = {}
        for matricization_row_modes in self.matricizations_to_track:

            num_top = min(self.num_top_sing_vals_to_track, self.__compute_matricization_min_dim(matricization_row_modes))
            for top_index in range(num_top):
                mat_top_sing_val_metric = DummyAveragedMetric()
                mat_str = self.__create_matricization_modes_str(matricization_row_modes)
                metric_name = self.MATRICIZATION_SING_VAL_NAME_TEMPLATE.format(mat_str, top_index)
                metric_tag = f"mat {mat_str} top sing vals"
                metric_infos[metric_name] = MetricInfo(metric_name, mat_top_sing_val_metric, tag=metric_tag)

        return metric_infos

    def get_metric_infos(self) -> Dict[str, MetricInfo]:
        return self.metric_infos

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def get_tracked_values(self) -> Dict[str, TrackedValue]:
        return self.tracked_values

    def __compute_top_weight_vector_norms_metrics(self, metric_values: dict):
        per_layer_params = self.ht_model.create_stacked_per_hidden_layer_params()
        for layer_index in range(len(per_layer_params)):
            layer_params = per_layer_params[layer_index]
            weight_vector_norms = layer_params.norm(dim=1)

            num_top = min(self.num_top_weight_vector_norms_to_track_per_layer_and_factor, weight_vector_norms.shape[1])
            top_weight_vector_norms = torch.topk(weight_vector_norms, k=num_top)[0]

            for factor_index in range(top_weight_vector_norms.shape[0]):
                for top_index in range(num_top):
                    metric_name = self.TOP_WEIGHT_VECTOR_NORM_NAME_TEMPLATE.format(layer_index, factor_index, top_index)
                    metric = self.manual_metric_infos[metric_name].metric
                    tracked_value = self.tracked_values[metric_name]

                    value = top_weight_vector_norms[factor_index][top_index].item()
                    metric(value)
                    tracked_value.add_batch_value(value)
                    metric_values[metric_name] = value

    def __compute_top_local_components_norms_metrics(self, metric_values: dict):
        per_layer_params = self.ht_model.create_stacked_per_hidden_layer_params()
        for layer_index in range(len(per_layer_params)):
            local_components_norms = self.__compute_local_components_norms(per_layer_params, layer_index)

            num_top = min(self.num_top_weight_vector_norms_to_track_per_layer_and_factor, local_components_norms.shape[1])
            top_local_components_norms = torch.topk(local_components_norms, k=num_top)[0]

            pool_size = self.ht_model.get_layer_pool_size(layer_index)
            pooled_factors_indices_seq = self.__get_pooled_factors_indices_seq(layer_num_factors=per_layer_params[layer_index].shape[0],
                                                                               pool_size=pool_size)

            for i, pooled_factors_indices in enumerate(pooled_factors_indices_seq):
                factor_indices_str = "_".join([str(j) for j in pooled_factors_indices])
                for top_index in range(num_top):
                    metric_name = self.TOP_LOCAL_COMPONENT_NORM_NAME_TEMPLATE.format(layer_index, factor_indices_str, top_index)
                    metric = self.manual_metric_infos[metric_name].metric
                    tracked_value = self.tracked_values[metric_name]

                    value = top_local_components_norms[i][top_index].item()
                    metric(value)
                    tracked_value.add_batch_value(value)
                    metric_values[metric_name] = value

    def __compute_local_components_norms(self, per_layer_params, layer_index):
        pool_size = self.ht_model.get_layer_pool_size(layer_index)
        pooling_permutation = self.ht_model.get_pooling_mode_permutation(num_modes=per_layer_params[layer_index].shape[0], pool_size=pool_size)

        layer_params = per_layer_params[layer_index]
        pool_ordered_layer_params = layer_params[pooling_permutation]
        weight_vector_norms = pool_ordered_layer_params.norm(dim=1)

        num_orders_to_pool = (weight_vector_norms.shape[0] // pool_size) * pool_size
        weight_vector_norms_to_pool = weight_vector_norms[:num_orders_to_pool]
        prod_pooled_weight_vectors_norms = self.__pool_weight_vector_norms(weight_vector_norms_to_pool, pool_size)

        if num_orders_to_pool < weight_vector_norms.shape[0]:
            remainder_to_pool = weight_vector_norms[num_orders_to_pool:]
            remainder_pooled_norms = self.__pool_weight_vector_norms(remainder_to_pool, pool_size)
            prod_pooled_weight_vectors_norms = torch.cat([prod_pooled_weight_vectors_norms, remainder_pooled_norms], dim=0)

        if layer_index < len(per_layer_params) - 1:
            next_layer_norms = per_layer_params[layer_index + 1].norm(dim=2)
        elif self.ht_model.linear_output_layer:
            next_layer_norms = self.ht_model.output_layer_parameters.unsqueeze(0).norm(dim=2)
        else:
            next_layer_norms = torch.ones(1, weight_vector_norms.shape[1], device=weight_vector_norms.device)

        return prod_pooled_weight_vectors_norms * next_layer_norms

    def __pool_weight_vector_norms(self, weight_vector_norms, pool_size: int):
        if weight_vector_norms.shape[0] <= 1:
            return weight_vector_norms

        weight_vector_norms = weight_vector_norms.permute(1, 0)

        pool_size = min(pool_size, weight_vector_norms.shape[1])
        unfolded_weight_vector_norms = weight_vector_norms.reshape(weight_vector_norms.shape[0], -1, pool_size)
        pooled_unfolded_activations = unfolded_weight_vector_norms.prod(dim=-1)

        return pooled_unfolded_activations.permute(1, 0)

    def __compute_mode_tree_mat_metrics(self, metric_values: dict):
        tensor = self.ht_model.compute_tensor()

        for matricization_row_modes in self.matricizations_to_track:
            tensor_mat = tensor_utils.matricize(tensor, matricization_row_modes)
            tensor_mat = tensor_mat.to(self.compute_svd_device)
            singular_values = torch.svd(tensor_mat, compute_uv=False).S

            if self.track_mode_tree_mat_effective_ranks:
                self.__compute_mat_effective_rank_metric(matricization_row_modes, singular_values, metric_values)

            if self.track_mode_tree_mat_top_sing_vals:
                self.__compute_top_sing_vals_metrics(matricization_row_modes, singular_values, metric_values)

    def __compute_mat_effective_rank_metric(self, matricization_row_modes, singular_values, metric_values):
        metric_name = self.MATRICIZATION_EFFECTIVE_RANK_NAME_TEMPLATE.format(self.__create_matricization_modes_str(matricization_row_modes))
        metric = self.manual_metric_infos[metric_name].metric
        tracked_value = self.tracked_values[metric_name]

        value = self.__compute_effective_rank(singular_values).item()
        metric(value)
        tracked_value.add_batch_value(value)
        metric_values[metric_name] = value

    def __compute_effective_rank(self, singular_values: torch.Tensor):
        non_zero_singular_values = singular_values[singular_values != 0]
        normalized_non_zero_singular_values = non_zero_singular_values / non_zero_singular_values.sum()
        singular_values_entropy = -(normalized_non_zero_singular_values * torch.log(normalized_non_zero_singular_values)).sum()
        return torch.exp(singular_values_entropy)

    def __compute_top_sing_vals_metrics(self, matricization_row_modes, singular_values, metric_values):
        num_top = min(self.num_top_sing_vals_to_track, self.__compute_matricization_min_dim(matricization_row_modes))
        for top_index in range(num_top):
            mat_str = self.__create_matricization_modes_str(matricization_row_modes)
            metric_name = self.MATRICIZATION_SING_VAL_NAME_TEMPLATE.format(mat_str, top_index)
            metric = self.manual_metric_infos[metric_name].metric
            tracked_value = self.tracked_values[metric_name]

            value = singular_values[top_index].item()
            metric(value)
            tracked_value.add_batch_value(value)
            metric_values[metric_name] = value

    def evaluate(self) -> dict:
        with torch.no_grad():
            self.ht_model.to(self.device)

            metric_values = {}

            if self.track_top_weight_vector_norms:
                self.__compute_top_weight_vector_norms_metrics(metric_values)

            if self.track_top_local_components_norms:
                self.__compute_top_local_components_norms_metrics(metric_values)

            if self.track_mode_tree_mat_effective_ranks or self.track_mode_tree_mat_top_sing_vals:
                self.__compute_mode_tree_mat_metrics(metric_values)

            return metric_values
