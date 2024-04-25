# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import comet_ml
from comet_ml import exceptions

import torch
import torch.nn

from . import parameter_after_backward_hook, tensor_helpers


def watch(model: torch.nn.Module, log_step_interval: int = 1000) -> None:
    """
    Enables automatic logging of each layer's parameters and gradients in the given PyTorch module.
    These will be logged as histograms. Note that an Experiment must be created before
    calling this function.

    Args:
        model: torch.nn.Module, an instance of `torch.nn.Module`.
        log_step_interval: int (optional), determines how often layers are logged (default is every
        1000 steps).
    """
    experiment = comet_ml.get_global_experiment()

    if experiment is None:
        raise exceptions.CometException(
            "An Experiment must be created before calling `comet_ml.integration.pytorch.watch`"
        )

    initial_step = 0 if experiment.curr_step is None else experiment.curr_step

    for name, parameter in model.named_parameters():
        after_backward_hook = parameter_after_backward_hook.ParameterAfterBackwardHook(
            experiment=experiment,
            name=name,
            parameter=parameter,
            log_step_interval=log_step_interval,
            initial_step=initial_step,
        )
        parameter.register_hook(after_backward_hook)

        experiment.log_histogram_3d(
            tensor_helpers.to_numpy(parameter).flatten(),
            name,
            step=initial_step,
        )
