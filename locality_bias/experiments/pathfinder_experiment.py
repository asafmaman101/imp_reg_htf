import logging
from collections import OrderedDict
from typing import Tuple

import torch
import torch.utils.data
from torch import nn as nn

import common.utils.model as model_utils
from common.data.modules import DataModule
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import SupervisedTrainEvaluator, SupervisedValidationEvaluator, TrainEvaluator, \
    Evaluator, \
    TrainBatchOutputEvaluator, ComposeTrainEvaluator
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback, StopOnMetricValue
from common.train.trainer import Trainer
from locality_bias.datasets.pathfinder_datamodule import PathfinderDataModule
from locality_bias.train.input_grad_change_regularization_trainer import InputGradChangeRegularizationTrainer, \
    get_shuffle_patch_func, \
    get_sample_rand_first_patch_func, get_sample_rand_second_patch_func


class PathfinderExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--dataset_path", type=str, help="Path of dataset to use.")
        parser.add_argument("--num_train_samples", type=int, default=-1, help="Number of train samples to use. If < 0, will use all samples except "
                                                                              "those left for validation.")

        parser.add_argument("--model", type=str, default="resnet18",
                            help="Model to use. Currently supports 'resnetX', where in place of X is the code number of the network.")
        parser.add_argument("--batch_size", type=int, default=64, help="Train batch size.")
        parser.add_argument("--gradient_accumulation", type=int, default=-1,
                            help="If > 0, will accumulate gradients for this amount of batches before each optimizer step.")
        parser.add_argument("--lr", type=float, default=1e-2, help="Training learning rate.")
        parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer to use. Supports: 'sgd' and 'adam'.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
        parser.add_argument("--weight_decay", type=float, default=0, help="L2 regularization coefficient")
        parser.add_argument("--dropout", type=float, default=0, help="dropout regularization probability")
        parser.add_argument("--stop_on_perfect_train_acc", action="store_true",
                            help="Stops training 'stop_on_perfect_train_acc_patience' epochs after reaching perfect train accuracy.")
        parser.add_argument("--stop_on_perfect_train_acc_patience", type=int, default=20,
                            help="Number of epochs to wait after perfect train accuracy before stopping.")

        parser.add_argument("--grad_change_reg_coeff", type=float, default=0, help="Input GradChange regularization coefficient.")
        parser.add_argument("--reg_patch_size", type=float, default=2, help="Patch size for Input GradChange regularization.")
        parser.add_argument("--reg_distance", type=float, default=0, help="Distance between patches for Input GradChange regularization.")
        parser.add_argument("--reg_only_adjecent_patches", action="store_true", help="Sample only adjecent patches for the regularization.")

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        datamodule = PathfinderDataModule(dataset_path=config["dataset_path"],
                                          num_train_samples=config["num_train_samples"],
                                          batch_size=config["batch_size"],
                                          split_random_state=168)

        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        in_size = datamodule.input_dims
        num_classes = datamodule.num_classes

        return model_utils.create_modified_model(config["model"], input_size=in_size, output_size=1, pretrained=False, dropout=config["dropout"])

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: DataModule, val_dataloader: torch.utils.data.DataLoader, device,
                                               config: dict, state: dict, logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        train_metric_info_seq = [
            metrics.MetricInfo("train loss", metrics.BCEWithLogitsLoss(), tag="loss"),
            metrics.MetricInfo("train accuracy", metrics.BinaryClassificationAccuracyWithLogits(), tag="accuracy")
        ]

        train_supervised_evaluator = SupervisedTrainEvaluator(train_metric_info_seq)
        regularization_evaluator = TrainBatchOutputEvaluator(metric_names=["regularization"])
        train_evaluator = ComposeTrainEvaluator([train_supervised_evaluator, regularization_evaluator])

        val_metric_info_seq = [
            metrics.MetricInfo("val loss", metrics.BCEWithLogitsLoss(), tag="loss"),
            metrics.MetricInfo("val accuracy", metrics.BinaryClassificationAccuracyWithLogits(), tag="accuracy")
        ]

        val_evaluator = SupervisedValidationEvaluator(model, val_dataloader, metric_info_seq=val_metric_info_seq, device=device)
        return train_evaluator, val_evaluator

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="val accuracy", is_train_metric=False, largest=True, return_best_score=False)

    def create_additional_metadata_to_log(self, model: nn.Module, datamodule: DataModule, config: dict, state: dict,
                                          logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["train dataset size"] = len(datamodule.train_dataset)
        additional_metadata["test dataset size"] = len(datamodule.test_dataset)
        return additional_metadata

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: nn.Module, config: dict, state: dict, logger: logging.Logger):
        if config["stop_on_perfect_train_acc"]:
            callbacks_dict["stop_on_perfect_train_acc"] = StopOnMetricValue("train accuracy", is_train_metric=True, threshold_value=1, largest=True,
                                                                            patience=config["stop_on_perfect_train_acc_patience"],
                                                                            validate_every=config["validate_every"])

    def create_trainer(self, model: nn.Module, datamodule: DataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:

        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        sample_first_patch_func = get_sample_rand_first_patch_func(patch_size=config["reg_patch_size"])
        sample_second_patch_func = get_sample_rand_second_patch_func(patch_size=config["reg_patch_size"], distance=config["reg_distance"],
                                                                     adjecent=config["reg_only_adjecent_patches"])
        shuffle_patch_func = get_shuffle_patch_func()
        loss_fn = nn.BCEWithLogitsLoss()
        return InputGradChangeRegularizationTrainer(model, optimizer, loss_fn,
                                                    sample_first_patch_func=sample_first_patch_func,
                                                    sample_second_patch_func=sample_second_patch_func,
                                                    shuffle_patch_func=shuffle_patch_func,
                                                    reg_coeff=config["grad_change_reg_coeff"],
                                                    gradient_accumulation=config["gradient_accumulation"],
                                                    train_evaluator=train_evaluator,
                                                    val_evaluator=val_evaluator,
                                                    callback=callback,
                                                    device=device)
