import torch
import torch.nn.functional as F

from common.evaluation.metrics.metric import AveragedMetric
from tensor_factorization.models import CPTensorFactorization


class CPFactorNorm(AveragedMetric):

    def __init__(self, factor_index: int, norm: str = "fro"):
        super().__init__()
        self.factor_index = factor_index
        self.norm = norm

    def _calc_metric(self, cp: CPTensorFactorization):
        factors = cp.factors
        factor = factors[self.factor_index]
        return torch.norm(factor, p=self.norm).item(), 1


class CPFactorColNorm(AveragedMetric):

    def __init__(self, factor_index: int, col_index: int, norm: str = "fro"):
        super().__init__()
        self.factor_index = factor_index
        self.col_index = col_index
        self.norm = norm

    def _calc_metric(self, cp: CPTensorFactorization):
        factors = cp.factors
        factor = factors[self.factor_index]
        return torch.norm(factor[:, self.col_index], p=self.norm).item(), 1


class CPComponentFroNorm(AveragedMetric):

    def __init__(self, component_index: int, norm: str = "fro"):
        super().__init__()
        self.component_index = component_index
        self.norm = norm

    def _calc_metric(self, cp: CPTensorFactorization):
        factors = cp.factors
        component_vector_norms = [factor[:, self.component_index].norm(p=self.norm) for factor in factors]
        component_vector_norms = torch.tensor(component_vector_norms, device=factors[0].device)
        return torch.prod(component_vector_norms).item(), 1


class CPComponentsCosineSimilarity(AveragedMetric):

    def __init__(self, first_component_index: int, second_component_index: int):
        super().__init__()
        self.first_component_index = first_component_index
        self.second_component_index = second_component_index

    def _calc_metric(self, cp: CPTensorFactorization):
        factors = cp.factors

        cosine_sims = []
        for factor in factors:
            first_component = factor[:, self.first_component_index]
            second_component = factor[:, self.second_component_index]
            cosine_sims.append(F.cosine_similarity(first_component, second_component, dim=0))

        cosine_sims = torch.tensor(cosine_sims, device=factors[0].device)
        return torch.prod(cosine_sims).item(), 1
