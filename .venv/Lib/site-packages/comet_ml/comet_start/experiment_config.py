# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
from typing import Any, Dict, Optional

from ..config import get_config
from ..logging_messages import ENV_DETAILS_MISMATCHED_WARNING
from .config_helpers import update_with_default

PARAMS_WITH_EXPERIMENT_DEFAULTS = [
    "log_code",
    "log_graph",
    "parse_args",
    "log_git_metadata",
    "log_git_patch",
    "log_env_details",
]

ENVIRONMENT_DETAILS_PARAMS = [
    "log_env_gpu",
    "log_env_host",
    "log_env_cpu",
    "log_env_network",
    "log_env_disk",
]

LOGGER = logging.getLogger(__name__)


class ExperimentConfig:
    """
    The Comet experiment configuration.
    Args:
        disabled (boolean, optional, defaults to `False`) Allows you to disable all network
            communication with the Comet.com backend. It is useful when you want to test to make
            sure everything is working, without actually logging anything. Defaults to `False`.
        log_code: (boolean, optional, defaults to None) - Allows you to enable and disable automatic code logging.
            Defaults to `True` when a new experiment is created and `False` when resuming an experiment.
        log_graph: (boolean, optional, defaults to None) - Allows you to enable and disable automatic computation graph
            logging. Defaults to `True` when a new experiment is created and `False` when resuming an experiment.
        parse_args: (boolean, optional, defaults to None) - Allows you to enable and disable automatic parsing of CLI
            arguments. Defaults to `True` when a new experiment is created and `False` when resuming an experiment.
        display_summary_level: (int, optional, defaults to 1) - Allows you to control the summary detail that is displayed
            on the console at the end of an experiment. If 0, the summary will not be displayed. Valid values are 0 to 2.
        log_git_metadata: (boolean, optional, defaults to None) Allows you to enable and disable automatic git metadata
            logging. Defaults to `True` when a new experiment is created and `False` when resuming an experiment.
        log_git_patch: (boolean, optional, defaults to None) - Allows you to enable and disable automatic git patch
            logging. Defaults to `True` when a new experiment is created and `False` when resuming an experiment.
        log_env_details: (boolean, optional, defaults to None) - Allows you to enable and disable automatic logging of
            various environment information in order to identify where the script ran. If `log_env_details` is `False`,
            GPU, CPU, network, disk and host information will not be logged. Defaults to True when a new experiment
            is created and False when resuming an experiment.
        log_env_gpu: (boolean, optional, defaults to None) - Allows you to enable and disable automatic logging of
            GPU details and metrics (utilization, memory usage, etc...).
        log_env_host: (boolean, optional, defaults to None) Allows you to enable and disable automatic logging of
            host information (ip, hostname, python version, user, etc...)
        log_env_cpu: (boolean, optional, defaults to None) - Allows you to enable and disable automatic logging of CPU
            details and metrics (utilization, memory usage, etc...). Defaults to `True` when a new experiment is
            created and `False` when resuming an experiment.
        log_env_network: (boolean, optional, defaults to None) - Allows you to enable and disable automatic logging of
            network details and metrics (sent and received rates, etc...).
        log_env_disk: (boolean, optional, defaults to None) - Allows you to enable and disable automatic logging of
            disk utilization metrics (usage, IO rates, etc...).
        auto_output_logging: (["default", "native", "simple", False], optional, defaults to "default") - Allows you to
            select which output logging mode to use. You can pass "native" which will log all output even when it
            originated from a C native library. You can also pass "simple" which will work only for output made
            by Python code. If you want to disable automatic output logging, you can pass `False`. The default is
            "default" which will detect your environment and deactivate the output logging for `IPython` and `Jupyter`
            environment and sets "native" in the other cases.
        auto_param_logging: (boolean, optional, defaults to True) - Allows you to enable and disable automatic
            hyperparameter logging.
        auto_metric_logging: (boolean, optional, defaults to True) - Allows you to enable and disable automatic
            metric logging.
        auto_log_co2: (boolean, optional, defaults to True) - Allows you to enable and disable automatic tracking of
            the CO2 emission of this experiment if codecarbon package is installed in the environment.
        auto_metric_step_rate: (int, optional, defaults to 10) - Allows you to control how often in-batch metrics
            are logged.
        auto_histogram_epoch_rate: (int, optional, defaults to 1) - Allows you to control how often histograms are logged.
        auto_histogram_gradient_logging (boolean, optional, defaults to False) - Allows you to enable and disable
            automatic histogram logging of gradients.
        auto_histogram_activation_logging (boolean, optional, defaults to False) - Allows you to enable and disable
            automatic histogram logging of activations.
        auto_histogram_tensorboard_logging (boolean, optional, defaults to False) - Allows you to enable and disable
            automatic tensorboard histogram logging.
        offline_directory (string, optional, defaults to None) - Allows you to set the directory used to save the
            offline archive for the experiment.
        distributed_node_identifier (string, optional, defaults to None) - Allows you to set a prefix as node identifier.
    """

    def __init__(
        self,
        disabled: Optional[bool] = None,
        log_code: Optional[bool] = None,
        log_graph: Optional[bool] = None,
        parse_args: Optional[bool] = None,
        display_summary_level: int = 1,
        log_git_metadata: Optional[bool] = None,
        log_git_patch: Optional[bool] = None,
        log_env_details: Optional[bool] = None,
        log_env_gpu: Optional[bool] = None,
        log_env_host: Optional[bool] = None,
        log_env_cpu: Optional[bool] = None,
        log_env_network: Optional[bool] = None,
        log_env_disk: Optional[bool] = None,
        auto_output_logging: str = "default",
        auto_param_logging: bool = True,
        auto_metric_logging: bool = True,
        auto_log_co2: bool = True,
        auto_metric_step_rate: int = 10,
        auto_histogram_epoch_rate: int = 1,
        auto_histogram_gradient_logging: bool = False,
        auto_histogram_activation_logging: bool = False,
        auto_histogram_tensorboard_logging: bool = False,
        auto_histogram_weight_logging: bool = False,
        offline_directory: Optional[str] = None,
        distributed_node_identifier: Optional[str] = None,
    ):
        config = get_config()

        self._disabled = config.get_bool(
            disabled, "comet.auto_log.disable", default=None, not_set_value=None
        )

        self.log_code = config.get_bool(
            log_code, "comet.auto_log.code", default=None, not_set_value=None
        )
        self.log_graph = config.get_bool(
            log_graph, "comet.auto_log.graph", default=None, not_set_value=None
        )
        self.parse_args = config.get_bool(
            parse_args, "comet.auto_log.cli_arguments", default=None, not_set_value=None
        )
        self.display_summary_level = config.get_int(
            display_summary_level,
            "comet.display_summary_level",
            default=1,
            not_set_value=1,
        )
        self.log_git_metadata = config.get_bool(
            log_git_metadata,
            "comet.auto_log.git_metadata",
            default=None,
            not_set_value=None,
        )
        self.log_git_patch = config.get_bool(
            log_git_patch, "comet.auto_log.git_patch", default=None, not_set_value=None
        )
        self.log_env_details = config.get_bool(
            log_env_details,
            "comet.auto_log.env_details",
            default=None,
            not_set_value=None,
        )
        self.log_env_gpu = config.get_bool(
            log_env_gpu, "comet.auto_log.env_gpu", default=None, not_set_value=None
        )
        self.log_env_host = config.get_bool(
            log_env_host, "comet.auto_log.env_host", default=None, not_set_value=None
        )
        self.log_env_cpu = config.get_bool(
            log_env_cpu, "comet.auto_log.env_cpu", default=None, not_set_value=None
        )
        self.log_env_network = config.get_bool(
            log_env_network,
            "comet.auto_log.env_network",
            default=None,
            not_set_value=None,
        )
        self.log_env_disk = config.get_bool(
            log_env_disk, "comet.auto_log.env_disk", default=None, not_set_value=None
        )

        self.auto_output_logging = config.get_string(
            auto_output_logging,
            "comet.auto_log.output_logger",
            default="default",
            not_set_value="default",
        )

        self.auto_param_logging = config.get_bool(
            auto_param_logging,
            "comet.auto_log.parameters",
            default=True,
            not_set_value=True,
        )
        self.auto_metric_logging = config.get_bool(
            auto_metric_logging,
            "comet.auto_log.metrics",
            default=True,
            not_set_value=True,
        )
        self.auto_log_co2 = config.get_bool(
            auto_log_co2, "comet.auto_log.co2", default=True, not_set_value=True
        )

        self.auto_metric_step_rate = config.get_int(
            auto_metric_step_rate,
            "comet.auto_log.metric_step_rate",
            default=10,
            not_set_value=10,
        )
        self.auto_histogram_epoch_rate = config.get_int(
            auto_histogram_epoch_rate,
            "comet.auto_log.histogram_epoch_rate",
            default=1,
            not_set_value=1,
        )

        self.auto_histogram_gradient_logging = config.get_bool(
            auto_histogram_gradient_logging,
            "comet.auto_log.histogram_gradients",
            default=False,
            not_set_value=False,
        )
        self.auto_histogram_activation_logging = config.get_bool(
            auto_histogram_activation_logging,
            "comet.auto_log.histogram_activations",
            default=False,
            not_set_value=False,
        )
        self.auto_histogram_tensorboard_logging = config.get_bool(
            auto_histogram_tensorboard_logging,
            "comet.auto_log.histogram_tensorboard",
            default=False,
            not_set_value=False,
        )
        self.auto_histogram_weight_logging = config.get_bool(
            auto_histogram_weight_logging,
            "comet.auto_log.histogram_weights",
            default=False,
            not_set_value=False,
        )

        self.offline_directory = config.get_string(
            offline_directory,
            "comet.offline_directory",
            default=None,
            not_set_value=None,
        )

        self.distributed_node_identifier = config.get_string(
            distributed_node_identifier,
            "comet.distributed_node_identifier",
            default=None,
            not_set_value=None,
        )

    @property
    def disabled(self):
        if self._disabled is None:
            return False
        return self._disabled

    def __str__(self):
        return "ExperimentConfig: %r" % self.__dict__


def effective_experiment_config(
    user_config: ExperimentConfig, new_experiment: bool
) -> ExperimentConfig:
    """
    The factory to create effective configuration options from provided `user_config` options by applying default value
    if particular parameter is `None`. The default value is defined by `new_experiment` parameter,
    i.e., if the experiment is created or if using existing experiment.
    Args:
        user_config: The `ExperimentConfig` object provided by user.
        new_experiment: If `True` the configuration options will be for new experiment, otherwise for existing.
    Returns:
        The new `ExperimentConfig` object created from user config object by applying default values.
    """
    kwargs = user_config.__dict__.copy()

    # update defaults if appropriate
    for param in PARAMS_WITH_EXPERIMENT_DEFAULTS:
        update_with_default(key=param, values_dict=kwargs, default=new_experiment)

    # update with environment details
    update_experiment_config_with_env_details(kwargs)

    # restore disabled parameter
    kwargs["disabled"] = kwargs.get("_disabled", None)
    if "_disabled" in kwargs:
        del kwargs["_disabled"]

    return ExperimentConfig(**kwargs)


def update_experiment_config_with_env_details(
    user_config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    log_env_details = user_config_dict.get("log_env_details")
    for param in ENVIRONMENT_DETAILS_PARAMS:
        param_value = user_config_dict.get(param)
        if param_value is None:
            update_with_default(
                param, values_dict=user_config_dict, default=log_env_details
            )
        elif param_value != log_env_details:
            LOGGER.warning(
                ENV_DETAILS_MISMATCHED_WARNING, param, param_value, log_env_details
            )

    return user_config_dict
