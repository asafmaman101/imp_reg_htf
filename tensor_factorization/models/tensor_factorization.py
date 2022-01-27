from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TensorFactorization(nn.Module, ABC):

    @abstractmethod
    def compute_tensor(self) -> torch.Tensor:
        """
        Computes the tensor.
        """
        raise NotImplementedError
