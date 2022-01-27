from typing import Callable
from typing import List

import numpy as np
import torch
import torch.nn as nn

from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer


class PatchCoordinates:

    def __init__(self, top_left_i: int, top_left_j: int, bottom_right_i: int, bottom_right_j: int):
        super().__init__()
        self.top_left_i = top_left_i
        self.top_left_j = top_left_j
        self.bottom_right_i = bottom_right_i
        self.bottom_right_j = bottom_right_j

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PatchCoordinates):
            return False

        return self.top_left_i == o.top_left_i and self.top_left_j == o.top_left_j \
               and self.bottom_right_i == o.bottom_right_i and self.bottom_right_j == o.bottom_right_j

    def __str__(self) -> str:
        return f"({self.top_left_i}, {self.top_left_j}, {self.bottom_right_i}, {self.bottom_right_j})"

    def __repr__(self) -> str:
        return f"({self.top_left_i}, {self.top_left_j}, {self.bottom_right_i}, {self.bottom_right_j})"


class InputGradChangeRegularizationTrainer(Trainer):
    """
    Trainer for supervised task of predicting y given x (classification or regression) with Input Gradient Change (IGC) regularization.
    """

    def __init__(self, model, optimizer, loss_fn,
                 sample_first_patch_func: Callable[[int, int], PatchCoordinates],
                 sample_second_patch_func: Callable[[int, int, PatchCoordinates], PatchCoordinates],
                 shuffle_patch_func: Callable[[torch.Tensor, PatchCoordinates], torch.Tensor],
                 reg_coeff: float = 0,
                 gradient_accumulation: int = -1,
                 train_evaluator=VoidEvaluator(),
                 val_evaluator=VoidEvaluator(),
                 callback=None,
                 device=torch.device("cpu")):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn
        self.sample_first_patch_func = sample_first_patch_func
        self.sample_second_patch_func = sample_second_patch_func
        self.shuffle_patch_func = shuffle_patch_func
        self.reg_coeff = reg_coeff
        self.reg_active = self.reg_coeff > 0
        self.gradient_accumulation = gradient_accumulation
        self.is_bce_loss = isinstance(self.loss_fn, nn.BCEWithLogitsLoss)

    def batch_update(self, batch_num, batch, total_num_batches):
        self.optimizer.zero_grad()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if self.reg_active:
            x.requires_grad = True

        y_pred = self.model(x)

        if self.is_bce_loss:
            y = y.to(torch.float)
            y_pred = y_pred.squeeze(dim=1)

        loss = self.loss_fn(y_pred, y)

        if self.reg_active:
            regularization = self.__compute_grad_change_reg(x, y, y_pred)
            loss += self.reg_coeff * regularization
        else:
            regularization = torch.zeros(1)

        x.requires_grad = False

        if self.gradient_accumulation > 0:
            loss = loss / self.gradient_accumulation

        loss.backward()

        do_accumulated_grad_update = (batch_num + 1) % self.gradient_accumulation == 0 or batch_num == total_num_batches - 1
        if self.gradient_accumulation <= 0 or do_accumulated_grad_update:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y,
            "regularization": regularization.item()
        }

    def __compute_grad_change_reg(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor):
        # Computes output grad w.r.t. inputs for each sample in batch
        relevant_outputs = y_pred[torch.arange(y_pred.shape[0]), y] if not self.is_bce_loss else y_pred
        x_grad = torch.autograd.grad(relevant_outputs.sum(), x, only_inputs=True, create_graph=True)[0]

        patch = self.sample_first_patch_func(x.shape[2], x.shape[3])

        patch_grad = x_grad[:, :, patch.top_left_i: patch.bottom_right_i, patch.top_left_j: patch.bottom_right_j]
        flattened_patch_grad = patch_grad.reshape(patch_grad.shape[0], -1)
        flattened_patch_grad = flattened_patch_grad / flattened_patch_grad.norm(dim=1, keepdim=True)

        other_patch = self.sample_second_patch_func(x.shape[2], x.shape[3], patch)
        x_shuffled = self.shuffle_patch_func(x.detach().clone(), other_patch)

        x_shuffled.requires_grad = True
        y_shuffled_pred = self.model(x_shuffled)
        relevant_shuffled_outputs = y_shuffled_pred[torch.arange(y_shuffled_pred.shape[0]), y] if not self.is_bce_loss else y_shuffled_pred

        # Computes output grad w.r.t. inputs for each sample in batch after changing values in other patch
        x_shuffled_grad = torch.autograd.grad(relevant_shuffled_outputs.sum(), x_shuffled, only_inputs=True, create_graph=True)[0]
        after_shuffle_grad = x_shuffled_grad[:, :, patch.top_left_i: patch.bottom_right_i, patch.top_left_j: patch.bottom_right_j]
        flattened_after_shuffle_grad = after_shuffle_grad.reshape(after_shuffle_grad.shape[0], -1)
        flattened_after_shuffle_grad = flattened_after_shuffle_grad / flattened_after_shuffle_grad.norm(dim=1, keepdim=True)

        grads_abs_cosine_sim = torch.abs((flattened_patch_grad * flattened_after_shuffle_grad).sum(dim=1))
        return grads_abs_cosine_sim.mean()


def get_sample_preset_first_patch_func(preset_patch_coords: List[PatchCoordinates]):
    def sample_first_patch_func(input_height: int, input_width: int) -> PatchCoordinates:
        chosen_patch = np.random.randint(0, len(preset_patch_coords))
        return preset_patch_coords[chosen_patch]

    return sample_first_patch_func


def get_sample_preset_second_patch_func(preset_patch_coords: List[PatchCoordinates]):
    def sample_second_patch_func(input_height: int, input_width: int, patch_coords: PatchCoordinates) -> PatchCoordinates:
        remaining_preset_patch_coords = [coords for coords in preset_patch_coords if coords != patch_coords]
        chosen_patch = np.random.randint(0, len(remaining_preset_patch_coords))
        return remaining_preset_patch_coords[chosen_patch]

    return sample_second_patch_func


def get_sample_rand_first_patch_func(patch_size: int):
    def sample_first_patch_func(input_height: int, input_width: int) -> PatchCoordinates:
        top_left_i = np.random.randint(0, input_height - patch_size + 1)
        top_left_j = np.random.randint(0, input_width - patch_size + 1)
        return PatchCoordinates(top_left_i, top_left_j, top_left_i + patch_size, top_left_j + patch_size)

    return sample_first_patch_func


def get_sample_rand_second_patch_func(patch_size: int, distance: int, adjecent: bool = False):
    def sample_second_patch_func(input_height: int, input_width: int, patch_coords: PatchCoordinates) -> PatchCoordinates:

        top_outer_border = 0
        left_outer_border = 0
        bottom_outer_border = input_height - patch_size + 1
        right_outer_border = input_width - patch_size + 1

        if adjecent:
            top_outer_border = max(top_outer_border, patch_coords.top_left_i - patch_size - distance)
            left_outer_border = max(left_outer_border, patch_coords.top_left_j - patch_size - distance)
            bottom_outer_border = min(bottom_outer_border, patch_coords.bottom_right_i + distance + 1)
            right_outer_border = min(right_outer_border, patch_coords.bottom_right_j + distance + 1)

        top_inner_border = max(top_outer_border, patch_coords.top_left_i - patch_size - distance + 1)
        left_inner_border = max(left_outer_border, patch_coords.top_left_j - patch_size - distance + 1)
        bottom_inner_border = min(bottom_outer_border, patch_coords.bottom_right_i + distance)
        right_inner_border = min(right_outer_border, patch_coords.bottom_right_j + distance)

        top_left_slice = torch.cartesian_prod(torch.arange(top_outer_border, top_inner_border),
                                              torch.arange(left_outer_border, right_inner_border))
        top_right_slice = torch.cartesian_prod(torch.arange(top_outer_border, bottom_inner_border),
                                               torch.arange(right_inner_border, right_outer_border))
        bottom_right_slice = torch.cartesian_prod(torch.arange(bottom_inner_border, bottom_outer_border),
                                                  torch.arange(left_inner_border, right_outer_border))
        bottom_left_slice = torch.cartesian_prod(torch.arange(top_inner_border, bottom_outer_border),
                                                 torch.arange(left_outer_border, left_inner_border))

        availiable_i_and_j = torch.cat([top_left_slice, top_right_slice, bottom_right_slice, bottom_left_slice])

        if len(availiable_i_and_j) > 0:
            top_left_i, top_left_j = availiable_i_and_j[np.random.randint(0, len(availiable_i_and_j))]
        else:
            raise ValueError('Image is not large enough to sample this patch size and distance.')

        return PatchCoordinates(top_left_i, top_left_j, top_left_i + patch_size, top_left_j + patch_size)

    return sample_second_patch_func


def __get_available_start_indices(dim: int, start_index: int, end_index: int, patch_size: int, distance: int):
    available_before = min(start_index - patch_size + 1, start_index - distance + 1)
    available_before = max(available_before, 0)
    available = torch.arange(0, available_before)

    if end_index + distance < dim - patch_size + 1:
        available = torch.cat([available, torch.arange(end_index + distance, dim - patch_size + 1)])

    if len(available) == 0:
        raise ValueError(f"Unable to sample second patch for InputGradChangeRegularization. "
                         f"Input dim (height or width): {dim}, Start and end indices: {start_index}--{end_index}")
    return available


def get_shuffle_patch_func():
    def shuffle_patch_func(x: torch.Tensor, coords: PatchCoordinates):
        shift_by_one_perm = torch.cat([torch.arange(1, x.shape[0]), torch.zeros(1, dtype=torch.long)])
        shuffled_other_patch = x[shift_by_one_perm, :, coords.top_left_i: coords.bottom_right_i, coords.top_left_j: coords.bottom_right_j]
        x[:, :, coords.top_left_i: coords.bottom_right_i, coords.top_left_j: coords.bottom_right_j] = shuffled_other_patch
        return x

    return shuffle_patch_func
