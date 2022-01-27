from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from common.models import resnet_with_dropout

from . import module as module_utils


def create_model(model_name: str, pretrained: bool = False, include_fc_top: bool = True, requires_grad: bool = True, dropout=0) -> nn.Module:
    """
    Creates a model based on one of the standard torchvision models.
    @param model_name: model name: e.g. 'resnet18', 'resnet34', 'resnet50'
    @param pretrained: if true, loads pretrained weights
    @param include_fc_top: if true, includes the fully connected classification top
    @param requires_grad: the requires_grad of the network parameters
    @return: Module.
    """
    supported_resnet_models = {"resnet18": resnet_with_dropout.resnet18, "resnet34": resnet_with_dropout.resnet34, "resnet50": resnet_with_dropout.resnet50, "resnet101": resnet_with_dropout.resnet101,
                               "resnet152": resnet_with_dropout.resnet152, "wide_resnet50": resnet_with_dropout.wide_resnet50_2}
    if model_name in supported_resnet_models:
        return __create_resnet_model(supported_resnet_models[model_name], pretrained=pretrained,
                                     include_fc_top=include_fc_top, requires_grad=requires_grad, dropout=dropout)
    else:
        raise ValueError(f"Unable to create model with unsupported model name: '{model_name}'")


def __create_resnet_model(create_model_func, pretrained: bool = False, include_fc_top: bool = True, requires_grad: bool = True, dropout=0) -> nn.Module:
    model = create_model_func(pretrained=pretrained, dropout=dropout)
    module_utils.set_requires_grad(model, requires_grad)

    if include_fc_top:
        return model

    model.fc = module_utils.PassthroughLayer()
    return model


def create_modified_model(model_name: str, input_size: Tuple[int, int, int], output_size: int,
                          pretrained: bool = False, requires_grad: bool = True, dropout=0) -> nn.Module:
    """
    Creates a model based on one of the standard torchvision models that fits the given input size and output size. The modifications include
    replacing the final linear layer
    @param model_name: currently supports only resnets, e.g. 'resnet18', 'resnet34', 'resnet50'
    inception models, e.g. 'googlenet' and 'inception_v3'.
    @param pretrained: if true, loads pretrained weights
    @param requires_grad: the requires_grad of the network parameters
    @return: Module.
    """
    supported_resnet_models = {"resnet18": resnet_with_dropout.resnet18, "resnet34": resnet_with_dropout.resnet34, "resnet50": resnet_with_dropout.resnet50, "resnet101": resnet_with_dropout.resnet101,
                               "resnet152": resnet_with_dropout.resnet152}

    if model_name in supported_resnet_models:
        return __create_modified_resnet_model(supported_resnet_models[model_name], input_size=input_size, output_size=output_size,
                                              pretrained=pretrained, requires_grad=requires_grad, dropout=dropout)
    else:
        raise ValueError(f"Unable to create model with unsupported model name: '{model_name}'")


def __create_modified_resnet_model(create_model_func, input_size: Tuple[int, int, int], output_size: int,
                                   pretrained: bool = False, requires_grad: bool = True, dropout=0) -> nn.Module:
    model = __create_resnet_model(create_model_func, pretrained=pretrained, include_fc_top=True, requires_grad=requires_grad, dropout=dropout)
    model.fc = nn.Linear(model.fc.in_features, output_size)

    if input_size[0] != 3:
        model.conv1 = nn.Conv2d(input_size[0], model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride, padding=model.conv1.padding, bias=False)

    module_utils.set_requires_grad(model, requires_grad)
    return model


def load_modified_model_from_trainer_checkpoint(checkpoint_path: str, model_name: str, input_size: Tuple[int, int, int], output_size: int,
                                                requires_grad: bool = True, device=torch.device("cpu"), dropout=0) -> nn.Module:
    trainer_state_dict = torch.load(checkpoint_path, map_location=device)
    model = create_modified_model(model_name, input_size, output_size, dropout=dropout)
    model.load_state_dict(trainer_state_dict["model"])

    module_utils.set_requires_grad(model, requires_grad)
    return model
