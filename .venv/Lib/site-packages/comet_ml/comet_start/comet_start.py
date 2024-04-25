# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
from typing import Optional

from ..experiment import BaseExperiment
from ..logging_messages import COMET_START_FAILED_TO_CREATE_EXPERIMENT_ERROR
from .config_manager import ConfigurationManager
from .experiment_config import ExperimentConfig
from .init_parameters import InitParameters

ExperimentBase = BaseExperiment

LOGGER = logging.getLogger(__name__)


def start(
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project: Optional[str] = None,
    experiment_key: Optional[str] = None,
    mode: Optional[str] = None,
    online: Optional[bool] = None,
    experiment_config: Optional[ExperimentConfig] = None,
) -> ExperimentBase:
    """
    The factory method allowing to create the appropriate Comet experiment.
    Args:
        api_key: (str, optional, defaults to None) - the Comet API Key
        workspace: (str, optional, defaults to None) - the Workspace name
        project: (str, optional, defaults to None) - the Project name
        experiment_key: (str, optional, defaults to None) - the Experiment Key to use
        mode: (str, optional, defaults to `get_or_create`) - Creating mode, has the options
            `get_or_create`, `create` and `get`.
        online: (boolean, optional, defaults to True) - If `True`, data will be uploaded to the Comet platform.
            If `False`, data will be saved to disk.
        experiment_config: (ExperimentConfig, optional, defaults to None) - additional configuration options.
    Returns:
        The initialized Comet Experiment object.
    """
    init_parameters = InitParameters(
        api_key=api_key,
        workspace=workspace,
        project=project,
        experiment_key=experiment_key,
        mode=mode,
        online=online,
    )
    config_manager = ConfigurationManager(
        init_parameters=init_parameters, experiment_config=experiment_config
    )
    try:
        return config_manager.get_or_create_experiment()
    except Exception as e:
        LOGGER.error(COMET_START_FAILED_TO_CREATE_EXPERIMENT_ERROR, e, exc_info=True)
        raise e
