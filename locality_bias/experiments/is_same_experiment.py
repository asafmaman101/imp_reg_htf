import logging
from collections import OrderedDict
from typing import Tuple, List

import torch
import torch.utils.data
import torchvision.models
from torch import nn as nn

import common.utils.model as model_utils
from common.evaluation import metrics as metrics
from common.evaluation.evaluators import SupervisedTrainEvaluator, SupervisedValidationEvaluator, TrainEvaluator, Evaluator, \
    TrainBatchOutputEvaluator, ComposeTrainEvaluator
from common.experiment import FitExperimentBase
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback, StopOnMetricValue
from common.train.trainer import Trainer
from locality_bias.datasets.is_same_datamodule import IsSameDataModule
from locality_bias.train.input_grad_change_regularization_trainer import InputGradChangeRegularizationTrainer, PatchCoordinates, \
    get_sample_preset_first_patch_func, get_sample_preset_second_patch_func, get_shuffle_patch_func


class IsSameExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(parser):
        FitExperimentBase.add_experiment_base_specific_args(parser)

        parser.add_argument("--train_dataset_path", type=str, help="Path to train dataset file.")
        parser.add_argument("--test_dataset_path", type=str, help="Path to test dataset file.")
        parser.add_argument("--preload_data_to_memory", action="store_true", help="Preprocess and load all data to memory.")
        parser.add_argument("--distance", type=int, default=0, help="Horizontal space in pixels between the two images.")
        parser.add_argument("--input_width", type=int, default=224, help="Total width of input image to use.")
        parser.add_argument("--input_height", type=int, default=224, help="Total height of input image to use.")
        parser.add_argument("--num_train_samples", type=int, default=-1, help="Number of train samples to use (if < 0 will use the whole train set).")
        parser.add_argument("--num_test_samples", type=int, default=-1, help="Number of test samples to use (if < 0 will use the whole test set).")
        parser.add_argument("--model", type=str, default="resnet18",
                            help="Model to use. Currently supports 'resnetX', where in place of X is the code number of"
                                 "the network.")

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

        parser.add_argument("--grad_change_reg_coeff", type=float, default=0, help="GradChange regularization coefficient.")

    def validate_config(self, config: dict):
        super().validate_config(config)

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> IsSameDataModule:
        datamodule = IsSameDataModule(train_dataset_path=config["train_dataset_path"],
                                      test_dataset_path=config["test_dataset_path"],
                                      batch_size=config["batch_size"],
                                      num_train_samples=config["num_train_samples"],
                                      num_test_samples=config["num_test_samples"],
                                      distance=config["distance"],
                                      input_width=config["input_width"],
                                      input_height=config["input_height"],
                                      preload_data_to_memory=config["preload_data_to_memory"])
        datamodule.setup()
        return datamodule

    def create_model(self, datamodule: IsSameDataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        in_size = datamodule.train_dataset[0][0].shape
        return model_utils.create_modified_model(config["model"], input_size=in_size, output_size=1, pretrained=False, dropout=config["dropout"])

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: IsSameDataModule, val_dataloader: torch.utils.data.DataLoader,
                                               device, config: dict, state: dict, logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
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

    def create_additional_metadata_to_log(self, model: nn.Module, datamodule: IsSameDataModule, config: dict, state: dict,
                                          logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)

        train_dataset_size = len(datamodule.train_dataset)
        test_dataset_size = len(datamodule.test_dataset)

        additional_metadata["train dataset size"] = train_dataset_size
        additional_metadata["test dataset size"] = test_dataset_size
        return additional_metadata

    def customize_callbacks(self, callbacks_dict: OrderedDict, model: nn.Module, config: dict, state: dict, logger: logging.Logger):
        if config["stop_on_perfect_train_acc"]:
            callbacks_dict["stop_on_perfect_train_acc"] = StopOnMetricValue("train accuracy", is_train_metric=True, threshold_value=1, largest=True,
                                                                            patience=config["stop_on_perfect_train_acc_patience"],
                                                                            validate_every=config["validate_every"])

    def __get_bottom_layers_of_model(self, model: nn.Module, num_layers: int) -> List[nn.Module]:
        if isinstance(model, torchvision.models.ResNet):
            all_layers = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
        elif isinstance(model, torchvision.models.VGG):
            all_layers = list(model.features.children()) + list(model.classifier.children())
        elif isinstance(model, torchvision.models.DenseNet):
            all_layers = list([model.features.children()]) + [model.classifier]
        elif isinstance(model, torchvision.models.GoogLeNet):
            all_layers = list(model.children())
        else:
            raise ValueError(f"Cannot get bottom layers for unsupported model type {model.__class__}")

        return all_layers[:num_layers]

    def create_trainer(self, model: nn.Module, datamodule: IsSameDataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:

        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

        loss_fn = nn.BCEWithLogitsLoss()

        preset_patches = self.__create_in_gradchange_reg_preset_patch_coords_list(config, datamodule)
        sample_first_patch_func = get_sample_preset_first_patch_func(preset_patches)
        sample_second_patch_func = get_sample_preset_second_patch_func(preset_patches)
        shuffle_patch_func = get_shuffle_patch_func()

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

    def __create_in_gradchange_reg_preset_patch_coords_list(self, config, datamodule):
        grad_change_patch_size = 32
        in_size = datamodule.train_dataset[0][0].shape
        top_left_i = in_size[1] // 2 - grad_change_patch_size // 2
        top_left_j = in_size[2] // 2 - grad_change_patch_size - config["distance"] // 2

        left_im_coords = PatchCoordinates(top_left_i, top_left_j,
                                          top_left_i + grad_change_patch_size, top_left_j + grad_change_patch_size)

        shifted_top_left_j = top_left_j + grad_change_patch_size + config["distance"]
        right_im_coords = PatchCoordinates(top_left_i, shifted_top_left_j,
                                           top_left_i + grad_change_patch_size, shifted_top_left_j + grad_change_patch_size)

        return [left_im_coords, right_im_coords]
