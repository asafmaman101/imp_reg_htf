import argparse
import logging
from collections import OrderedDict
from typing import Tuple

import torch
import torch.utils
import torch.utils.data
from torch import nn as nn

from common.data.modules import DataModule
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import SupervisedTrainEvaluator, SupervisedValidationEvaluator, TrainEvaluator, \
    Evaluator, ComposeEvaluator
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.train import callbacks as callbacks
from common.train.callbacks import Callback
from common.train.optim import GroupRMSprop
from common.train.trainer import Trainer
from common.train.trainers import SupervisedTrainer
from tensor_factorization.datasets.tensor_sensing_datamodule import TensorSensingDataModule
from tensor_factorization.evaluation.ht_factorization_evaluator import HTFactorizationEvaluator
from tensor_factorization.evaluation.reconstruction_evaluator import ReconstructionEvaluator
from tensor_factorization.models import HierarchicalTensorFactorization, TensorFactorization


class HierarchicalTensorFactorizationSensingExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser: argparse.ArgumentParser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, required=True, help="Path to the the dataset file")
        parser.add_argument("--load_dataset_to_gpu", action="store_true", help="Stores all dataset on the main GPU (if GPU device is given)")
        parser.add_argument("--num_train_samples", type=int, default=2048, help="Number of train samples to use")
        parser.add_argument("--num_test_samples", type=int, default=-1, help="Number of test samples to use. If < 0 will use all remaining samples")
        parser.add_argument("--batch_size", type=int, default=-1, help="Train/test batch size. If <= 0 will use the whole train/test set each batch")
        parser.add_argument("--loss", type=str, default="l2", help="Loss to use. Currently supports: 'l2', 'l1', 'huber'.")
        parser.add_argument("--huber_loss_thresh", type=float, default=1., help="Threshold to use for the Huber loss.")
        parser.add_argument("--lr", type=float, default=1e-2, help="Training learning rate")
        parser.add_argument("--optimizer", type=str, default="grouprmsprop", help="optimizer to use. Supports: 'grouprmsprop' and 'sgd'")
        parser.add_argument("--momentum", type=float, default=0, help="Momentum for SGD")
        parser.add_argument("--stop_on_zero_loss_tol", type=float, default=5e-5, help="Stops when train loss reaches below this threshold.")
        parser.add_argument("--stop_on_zero_loss_patience", type=int, default=100,
                            help="Number of validated epochs loss has to remain 0 before stopping.")

        parser.add_argument("--init_mean", type=float, default=0, help="Init mean for gaussian init")
        parser.add_argument("--init_std", type=float, default=1e-2, help="Init std for gaussian init")

        parser.add_argument("--ht_pool_size", type=int, default=2, help="Pool size of HT decomposition.")
        parser.add_argument("--ht_layer_widths", nargs="+", type=int, default=[512, 512],
                            help="Width of each layer. If None (default) the max possible CP rank which is the product of all dimensions "
                                 "except the max is used for all layers.")

        parser.add_argument("--additional_matricizations_to_track", nargs="+", type=int, action="append",
                            help="Additional mode indices corresponding to matricizations to track")
        parser.add_argument("--compute_svd_on_cpu", action="store_true",
                            help="Compute metrics requiring SVD on CPU (prevents GPU out of memory for large tensors).")
        parser.add_argument("--num_top_to_track", type=int, default=15, help="Number of top weight vector norms per factor "
                                                                             "and matricization singular values to track")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None
        datamodule = TensorSensingDataModule(config["dataset_path"],
                                             num_train_samples=config["num_train_samples"],
                                             num_test_samples=config["num_test_samples"],
                                             batch_size=config["batch_size"],
                                             shuffle_train=config["batch_size"] > 0,
                                             load_dataset_to_device=load_dataset_to_device)

        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: TensorSensingDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        dataset = datamodule.dataset
        model = HierarchicalTensorFactorization(num_dim_per_mode=[dataset.inputs.shape[2]] * dataset.inputs.shape[1],
                                                pool_size=config["ht_pool_size"],
                                                layer_widths=config["ht_layer_widths"],
                                                init_mean=config["init_mean"],
                                                init_std=config["init_std"],
                                                linear_output_layer=True)

        return model

    def create_train_and_validation_evaluators(self, model: HierarchicalTensorFactorization, datamodule: TensorSensingDataModule,
                                               val_dataloader, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [metrics.MetricInfo("train_mse_loss", metrics.MSELoss(), tag="mse_loss"),
                                 self.__get_train_loss_metric_info(config, metric_name_prefix="train")]
        train_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)

        val_evaluators = []

        if len(datamodule.test_indices) > 0:
            val_metric_info_seq = [metrics.MetricInfo("test_mse_loss", metrics.MSELoss(), tag="mse_loss")]
            val_evaluators.append(SupervisedValidationEvaluator(model, val_dataloader, val_metric_info_seq, device=device))

        train_and_test_dataloader = datamodule.train_and_test_dataloader()
        val_evaluators.append(ReconstructionEvaluator(model, train_and_test_dataloader, device=device))

        ht_eval = HTFactorizationEvaluator(model,
                                           additional_matricizations_to_track=config["additional_matricizations_to_track"],
                                           num_top_sing_vals_to_track=config["num_top_to_track"],
                                           num_top_weight_vector_norms_to_track_per_layer_and_factor=config["num_top_to_track"],
                                           compute_svd_device=device if not config["compute_svd_on_cpu"] else torch.device("cpu"),
                                           device=device)
        val_evaluators.append(ht_eval)

        return train_evaluator, ComposeEvaluator(val_evaluators)

    def __get_train_loss_metric_info(self, config: dict, metric_name_prefix: str):
        if config["loss"] == "l2":
            return metrics.MetricInfo(f"{metric_name_prefix}_l2_loss", metrics.MSELoss(), tag="l2_loss")
        elif config["loss"] == "l1":
            return metrics.MetricInfo(f"{metric_name_prefix}_l1_loss", metrics.L1Loss(), tag="l1_loss")
        elif config["loss"] == "huber":
            return metrics.MetricInfo(f"{metric_name_prefix}_huber_loss", metrics.SmoothL1Loss(beta=config["huber_loss_thresh"]), tag="huber_loss")
        else:
            raise ValueError(f"Unsupported loss type: '{config['loss']}'")

    def __get_train_loss(self, config: dict):
        if config["loss"] == "l2":
            return nn.MSELoss()
        elif config["loss"] == "l1":
            return nn.L1Loss()
        elif config["loss"] == "huber":
            return nn.SmoothL1Loss(beta=config["huber_loss_thresh"])
        else:
            raise ValueError(f"Unsupported loss type: '{config['loss']}'")

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="train_mse_loss", is_train_metric=True, largest=False, return_best_score=False)

    def create_additional_metadata_to_log(self, model: HierarchicalTensorFactorization, datamodule: TensorSensingDataModule,
                                          config: dict, state: dict, logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata.update({
            "HT factorization hidden layers widths": model.hidden_layers_widths,
            "HT factorization num hidden layers": model.num_hidden_layers,
            "HT factorization depth": model.depth,
            "dataset size": len(datamodule.dataset),
            "# train samples": len(datamodule.train_indices),
            "# test samples": len(datamodule.test_indices),
            "train targets Fro norm": torch.norm(datamodule.train_targets, p="fro").item(),
            "test targets Fro norm": torch.norm(datamodule.test_targets, p="fro").item()
        })

        return additional_metadata

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: HierarchicalTensorFactorization, config: dict, state: dict,
                            logger: logging.Logger):
        train_loss_fn = lambda trainer: trainer.train_evaluator.get_tracked_values()["train_mse_loss"].current_value
        callbacks_dict["stop_on_zero_train_loss"] = callbacks.StopOnZeroTrainLoss(train_loss_fn=train_loss_fn,
                                                                                  tol=config["stop_on_zero_loss_tol"],
                                                                                  validate_every=config["validate_every"],
                                                                                  patience=config["stop_on_zero_loss_patience"])
        callbacks_dict["terminate_on_nan"] = callbacks.TerminateOnNaN(verify_batches=False)

    def create_trainer(self, model: TensorFactorization, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        if config["optimizer"] == "grouprmsprop":
            optimizer = GroupRMSprop(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        loss = self.__get_train_loss(config)
        return SupervisedTrainer(model, optimizer, loss, train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                 callback=callback, device=device)
