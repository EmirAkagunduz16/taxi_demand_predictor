# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from typing import Any, Optional, Union

from .._online import Experiment
from ..config import (
    get_api_key,
    get_config,
    get_previous_experiment,
    get_project_name,
    get_workspace,
)
from ..exceptions import InvalidExperimentModeUnsupported
from ..offline import OfflineExperiment
from .experiment_config import ExperimentConfig
from .start_modes import (
    RESUME_STRATEGY_CREATE,
    RESUME_STRATEGY_GET,
    RESUME_STRATEGY_GET_OR_CREATE,
    SUPPORTED_START_MODES,
)


class InitParameters:
    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[str] = None,
        online: Optional[bool] = None,
    ):
        config = get_config()
        self.api_key = get_api_key(api_key, config=config)
        self.workspace = get_workspace(workspace, config=config)
        self.project = get_project_name(project, config=config)
        self.experiment_key = get_previous_experiment(experiment_key, config=config)

        self.mode = mode
        if self.mode is None:
            self.mode = RESUME_STRATEGY_GET_OR_CREATE
        self._online = online

    @property
    def online(self) -> bool:
        if self._online is None:
            # the default is True if not set by user explicitly
            return True

        return self._online

    def validate(self):
        if self.mode is None or self.mode not in SUPPORTED_START_MODES:
            raise InvalidExperimentModeUnsupported(
                mode=str(self.mode), supported_modes=SUPPORTED_START_MODES
            )

    def is_create(self) -> bool:
        return self.mode == RESUME_STRATEGY_CREATE

    def is_get_or_create(self) -> bool:
        return self.mode == RESUME_STRATEGY_GET_OR_CREATE

    def is_get(self) -> bool:
        return self.mode == RESUME_STRATEGY_GET

    def __str__(self):
        return "InitParameters: %r" % self.__dict__


class KeyParameters:
    """
    Holds key parameters to be compared when deciding if existing experiment is the same as requested.
    """

    def __init__(
        self,
        api_key: Optional[str],
        workspace: Optional[str],
        project: Optional[str],
        experiment_key: Optional[str],
        online: Optional[bool],
        disabled: Optional[bool],
        offline_directory: Optional[str],
        distributed_node_identifier: Optional[str],
    ):
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.experiment_key = experiment_key
        self.online = online
        self.disabled = disabled
        self.distributed_node_identifier = distributed_node_identifier
        if online is False:
            self.offline_directory = offline_directory
        else:
            self.offline_directory = None

    def __eq__(self, other: "KeyParameters") -> bool:
        return (
            _none_or_equal(self.api_key, other=other.api_key)
            and _none_or_equal(self.workspace, other=other.workspace)
            and _none_or_equal(self.project, other=other.project)
            and _none_or_equal(self.experiment_key, other=other.experiment_key)
            and _none_or_equal(self.online, other=other.online)
            and _none_or_equal(self.disabled, other=other.disabled)
            and _none_or_equal(
                self.distributed_node_identifier,
                other=other.distributed_node_identifier,
            )
            and _none_or_equal(self.offline_directory, other=other.offline_directory)
        )

    def __str__(self) -> str:
        return "KeyParameters: %r" % self.__dict__

    @staticmethod
    def build(
        experiment_config: ExperimentConfig, init_params: InitParameters
    ) -> "KeyParameters":
        return KeyParameters(
            api_key=init_params.api_key,
            workspace=init_params.workspace,
            project=init_params.project,
            experiment_key=init_params.experiment_key,
            online=init_params._online,
            disabled=experiment_config._disabled,
            offline_directory=experiment_config.offline_directory,
            distributed_node_identifier=experiment_config.distributed_node_identifier,
        )


def key_parameters_matched(
    key_params: KeyParameters, experiment: Union[Experiment, OfflineExperiment]
) -> bool:
    offline_directory = None
    if isinstance(experiment, OfflineExperiment):
        offline_directory = experiment.offline_directory

    experiment_key_params = KeyParameters(
        api_key=experiment.api_key,
        workspace=experiment.workspace,
        project=experiment.project_name,
        experiment_key=experiment.get_key(),
        online=isinstance(experiment, Experiment),
        disabled=experiment.disabled,
        offline_directory=offline_directory,
        distributed_node_identifier=experiment.distributed_node_identifier,
    )

    return experiment_key_params == key_params


def _none_or_equal(this: Any, other: Any) -> bool:
    if other is None:
        return True

    return this == other
