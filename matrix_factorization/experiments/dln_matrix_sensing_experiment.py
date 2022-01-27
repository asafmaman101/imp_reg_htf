import argparse
import logging
from collections import OrderedDict
from typing import Tuple

import torch
import torch.utils
import torch.utils.data
from torch import nn as nn

import common.evaluation.metrics as metrics
import common.train.callbacks as callbacks
import common.utils.module as module_utils
import matrix_factorization.evaluation.matrix_metrics as matrix_metrics
from common.data.modules import DataModule
from common.evaluation.evaluators import SupervisedTrainEvaluator
from common.evaluation.evaluators import TrainEvaluator, Evaluator
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.optim import GroupRMSprop
from common.train.trainer import Trainer
from matrix_factorization.datasets.matrix_sensing_data_module import MatrixSensingDataModule
from matrix_factorization.evaluation.dln_matrix_evaluator import DLNMatrixValidationEvaluator
from matrix_factorization.models.deep_linear_net import DeepLinearNet
from matrix_factorization.models.dln_model_factory import DLNModelFactory
from matrix_factorization.trainers.dln_matrix_sensing_trainer import DLNMatrixSensingTrainer


class DLNMatrixSensingExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser: argparse.ArgumentParser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, required=True, help="Path to the matrix sensing dataset")
        parser.add_argument("--num_train_samples", type=int, default=2048, help="Number of sensing samples to use")
        parser.add_argument("--lr", type=float, default=1e-2, help="Training learning rate")
        parser.add_argument("--optimizer", type=str, default="grouprmsprop", help="optimizer to use. Supports: 'grouprmsprop' and 'sgd'")
        parser.add_argument("--momentum", type=float, default=0, help="Momentum to use for SGD")
        parser.add_argument("--stop_on_zero_loss_tol", type=float, default=5e-5, help="Stops when train loss reaches below this threshold.")
        parser.add_argument("--stop_on_zero_loss_patience", type=int, default=100,
                            help="Number of validated epochs loss has to remain 0 before stopping.")

        parser.add_argument("--depth", type=int, default=3, help="Depth of the factorization used to define the weight matrix.")
        parser.add_argument("--weight_init_type", type=str, default="normal",
                            help="Type of initialization for weights. Currently supports: 'identity', 'normal'")
        parser.add_argument("--init_std", type=float, default=1e-3,
                            help="Weight initialization std or identity multiplicative scalar (depending on init "
                                 "type) for each layer. If 'use_balanced_init' is used the product matrix will be "
                                 "initialized using this value.")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        return MatrixSensingDataModule(config["dataset_path"], config["num_train_samples"])

    def create_model(self, datamodule: MatrixSensingDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        target_matrix = datamodule.dataset.target_matrix
        model = DLNModelFactory.create_same_dim_deep_linear_network(target_matrix.size(0), target_matrix.size(1), depth=config["depth"],
                                                                    weight_init_type=config["weight_init_type"], init_std=config["init_std"])

        return model

    def create_train_and_validation_evaluators(self, model: DeepLinearNet, datamodule: MatrixSensingDataModule,
                                               val_dataloader: torch.utils.data.DataLoader, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [metrics.MetricInfo("train_mse_loss", metrics.MSELoss())]
        train_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)

        target_matrix = datamodule.dataset.target_matrix
        return train_evaluator, DLNMatrixValidationEvaluator(model, target_matrix, device=device)

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="normalized_reconstruction_error", is_train_metric=False, largest=False, return_best_score=False)

    def create_additional_metadata_to_log(self, model: nn.Module, datamodule: MatrixSensingDataModule, config: dict, state: dict,
                                          logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)

        target_matrix = datamodule.dataset.target_matrix
        additional_metadata.update({
            "Number of model parameters": module_utils.get_number_of_parameters(model),
            "Target matrix rows": target_matrix.size(0),
            "Target matrix cols": target_matrix.size(1),
            "Target Frobenius norm": torch.norm(target_matrix, p="fro").item(),
            "Target matrix nuclear norm": torch.norm(target_matrix, p="nuc").item(),
            "Target matrix rank": torch.linalg.matrix_rank(target_matrix).item(),
            "Target matrix effective rank": matrix_metrics.matrix_effective_rank(target_matrix)
        })

        return additional_metadata

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: nn.Module, config: dict, state: dict,
                            logger: logging.Logger):
        train_loss_fn = lambda trainer: trainer.train_evaluator.get_tracked_values()["train_mse_loss"].current_value

        callbacks_dict["stop_on_zero_train_loss"] = callbacks.StopOnZeroTrainLoss(train_loss_fn=train_loss_fn,
                                                                                  tol=config["stop_on_zero_loss_tol"],
                                                                                  validate_every=config["validate_every"],
                                                                                  patience=config["stop_on_zero_loss_patience"])
        callbacks_dict["terminate_on_nan"] = callbacks.TerminateOnNaN(verify_batches=False)

    def create_trainer(self, model: DeepLinearNet, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        elif config["optimizer"] == "grouprmsprop":
            optimizer = GroupRMSprop(model.parameters(), lr=config["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        mse_loss = nn.MSELoss()
        return DLNMatrixSensingTrainer(model, optimizer, mse_loss, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                       callback=callback, device=device)
