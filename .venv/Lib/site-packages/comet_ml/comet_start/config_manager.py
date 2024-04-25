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
import logging
from typing import Optional, Union

from .._online import Experiment
from ..config import get_global_experiment
from ..experiment import BaseExperiment
from ..logging_messages import COMET_START_FAILED_TO_END_RUNNING_EXPERIMENT_WARNING
from ..offline import OfflineExperiment
from .experiment_config import ExperimentConfig
from .experiment_factory import (
    create_offline_experiment,
    create_online_experiment,
    resume_online_experiment,
)
from .init_parameters import InitParameters, KeyParameters, key_parameters_matched

LOGGER = logging.getLogger(__name__)


class ConfigurationManager:
    def __init__(
        self,
        init_parameters: InitParameters,
        experiment_config: Optional[ExperimentConfig] = None,
    ):
        self.init_parameters = init_parameters
        if experiment_config is None:
            self.experiment_config = ExperimentConfig()
            LOGGER.debug(
                "User hasn't provided an experiment config, using default: %r",
                self.experiment_config,
            )
        else:
            self.experiment_config = experiment_config

    def validate(self):
        LOGGER.debug(
            "Validating configuration options provided by user. Initialization parameters: %s, experiment config: %s",
            self.init_parameters,
            self.experiment_config,
        )
        self.init_parameters.validate()

    def get_or_create_experiment(self) -> BaseExperiment:
        self.validate()
        experiment = get_global_experiment()
        if experiment is not None and not experiment.ended:
            experiment = self.evaluate_running_experiment(experiment)
            if experiment is not None:
                return experiment

        if not self.init_parameters.online:
            return create_offline_experiment(
                experiment_config=self.experiment_config,
                init_parameters=self.init_parameters,
            )

        if self.init_parameters.experiment_key is None:
            return create_online_experiment(
                experiment_config=self.experiment_config,
                init_parameters=self.init_parameters,
            )
        else:
            return resume_online_experiment(
                experiment_config=self.experiment_config,
                init_parameters=self.init_parameters,
            )

    def evaluate_running_experiment(
        self, experiment: Union[Experiment, OfflineExperiment]
    ) -> Optional[BaseExperiment]:
        LOGGER.debug(
            "Running experiment found, evaluating whether to return it to the user"
        )
        if self.init_parameters.is_get_or_create() or self.init_parameters.is_get():
            key_params = KeyParameters.build(
                experiment_config=self.experiment_config,
                init_params=self.init_parameters,
            )
            if key_parameters_matched(key_params, experiment):
                LOGGER.debug("Key parameters match, existing experiment is returned")
                return experiment

        LOGGER.debug("Key parameters do not match, finishing running experiment")
        try:
            experiment.end()
        except Exception as e:
            LOGGER.warning(
                COMET_START_FAILED_TO_END_RUNNING_EXPERIMENT_WARNING, e, exc_info=True
            )
        return None
