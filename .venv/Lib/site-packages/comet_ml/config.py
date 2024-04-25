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

from __future__ import print_function

import logging
import os
import os.path
import shutil
from typing import TYPE_CHECKING

import comet_ml.secrets.interpreter

from . import connection, secrets
from ._logging import _get_comet_logging_config
from ._streamlit import _in_streamlit_environment
from ._typing import Any, Dict, Optional, Tuple, Union
from .api_key.comet_api_key import CometApiKey, parse_api_key
from .config_class import ConfigDictEnv  # noqa
from .config_class import ConfigEnvFileEnv  # noqa
from .config_class import ConfigIniEnv  # noqa
from .config_class import ConfigOSEnv  # noqa
from .config_class import (
    Config,
    _config_path_from_directory,
    _get_default_config_path,
    _input_user,
)
from .dataclasses import experiment_info
from .exceptions import (
    AnonymousModeException,
    CometException,
    CometRestApiException,
    InvalidAPIKey,
)
from .logging_messages import (
    BASE_URL_MISMATCH_CONFIG_API_KEY,
    CONFIG_ANONYMOUS_UNALLOWED_ON_PREM,
    CONFIG_API_KEY_CHECK_FAILED,
    CONFIG_API_KEY_IS_INVALID,
    CONFIG_API_KEY_IS_NOT_SET,
    CONFIG_API_KEY_SAVE_FAILED_WARNING,
    CONFIG_API_KEY_SAVED_INFO,
    CONFIG_CLAIMED_ANONYMOUS_USER,
    CONFIG_FAILED_BACKUP_CONFIG_KEY_WARNING,
    CONFIG_GET_DISPLAY_SUMMARY_LEVEL_INVALID_WARNING,
    CONFIG_INIT_ONPREM_FAILED_EXCEPTION,
    CONFIG_INIT_ONPREM_START_INFO,
    CONFIG_INIT_ONPREM_USER_INPUT_INFO,
    CONFIG_USING_ANONYMOUS_MODE,
)
from .utils import (
    get_api_key_from_user,
    get_root_url,
    is_interactive,
    sanitize_url,
    url_join,
)

if TYPE_CHECKING:
    from comet_ml import BaseExperiment


LOGGER = logging.getLogger(__name__)

DEBUG = False
UNUSED = 0

# Global experiment placeholder. Should be set by the latest call of Experiment.init()
experiment = None

# global placeholder for comet_url_override
COMET_URL_OVERRIDE_CONFIG_KEY = "comet.url_override"

DEFAULT_IMAGE_NAME_MAX_CHARACTER_LENGTH = 100

DEFAULT_UPLOAD_SIZE_LIMIT = 200 * 1024 * 1024  # 200 MebiBytes

DEFAULT_ASSET_UPLOAD_SIZE_LIMIT = 100 * 1024 * 1024 * 1024  # 100GiB

DEFAULT_3D_CLOUD_UPLOAD_LIMITS = {
    "maxPoints": 1000000,
    "maxPointsFileSizeInBytes": UNUSED,
    "maxBoxes": 250000,
    "maxBoxesFileSizeInBytes": UNUSED,
}

DEFAULT_STREAMER_MSG_TIMEOUT = 60 * 60  # 1 Hour

ADDITIONAL_STREAMER_UPLOAD_TIMEOUT = 3 * 60 * 60  # 3 hours

DEFAULT_FILE_UPLOAD_READ_TIMEOUT = 900

DEFAULT_ARTIFACT_DOWNLOAD_TIMEOUT = 3 * 60 * 60  # 3 hours

DEFAULT_REMOTE_MODEL_DOWNLOAD_TIMEOUT = 3 * 60 * 60  # 3 hours

DEFAULT_INITIAL_DATA_LOGGER_JOIN_TIMEOUT = 5 * 60

DEFAULT_WS_JOIN_TIMEOUT = 30

DEFAULT_WS_RECONNECT_INTERVAL = 0.5

DEFAULT_PARAMETERS_BATCH_INTERVAL_SECONDS = 60

DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL = 15

DEFAULT_OFFLINE_DATA_DIRECTORY = ".cometml-runs"

DEFAULT_POOL_RATIO = 4

DEFAULT_SYSTEM_MONITORING_INTERVAL = 30

MAX_POOL_SIZE = 64

MESSAGE_BATCH_USE_COMPRESSION_DEFAULT = True

ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS = 10

ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE = 10000

MESSAGE_BATCH_METRIC_INTERVAL_SECONDS = 2

MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE = 1000

MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS = 5

MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE = 500

OFFLINE_EXPERIMENT_MESSAGES_JSON_FILE_NAME = "messages.json"

OFFLINE_EXPERIMENT_JSON_FILE_NAME = "experiment.json"

FALLBACK_STREAMER_CONNECTION_CHECK_INTERVAL_SECONDS = 2

UPLOAD_FILE_MAX_RETRIES = 4
UPLOAD_FILE_RETRY_BACKOFF_FACTOR = 3

S3_MULTIPART_SIZE_THRESHOLD_DEFAULT = 50 * 1024 * 1024  # 50MiB
S3_MULTIPART_EXPIRES_IN = 3 * 60 * 60  # 3 hours

HTTP_SESSION_RETRY_TOTAL_DEFAULT = 3
HTTP_SESSION_RETRY_BACKOFF_FACTOR_DEFAULT = 2

COMET_ROOT_URL = "https://www.comet.com/"
DEFAULT_COMET_BASE_URL = "https://www.comet.com"
DEFAULT_COMET_URL_OVERRIDE_PATH = "/clientlib/"
DEFAULT_COMET_URL_OVERRIDE = DEFAULT_COMET_BASE_URL + DEFAULT_COMET_URL_OVERRIDE_PATH


def get_global_experiment() -> "BaseExperiment":
    global experiment
    return experiment


def set_global_experiment(new_experiment):
    global experiment
    experiment = new_experiment


def get_running_experiment() -> Optional["BaseExperiment"]:
    """Returns a currently running experiment or None if there is no such.

    Returns: None or "BaseExperiment"

    Examples:
        ```python linenums="1"
        import comet_ml

        experiment = comet_ml.get_running_experiment()
        ```
    """

    global experiment
    return experiment


CONFIG_MAP = {
    "comet.disable_auto_logging": {"type": int, "default": 0},
    "comet.api_key": {"type": str},
    "comet.offline_directory": {"type": str, "default": DEFAULT_OFFLINE_DATA_DIRECTORY},
    "comet.git_directory": {"type": str},
    "comet.offline_sampling_size": {"type": int, "default": 15000},
    COMET_URL_OVERRIDE_CONFIG_KEY: {
        "type": str,
        "default": None,
    },
    "comet.optimizer_url": {
        "type": str,
        "default": None,
    },
    "comet.ws_url_override": {"type": str, "default": None},
    "comet.experiment_key": {"type": str},
    "comet.project_name": {"type": str},
    "comet.workspace": {"type": str},
    "comet.experiment_name": {"type": str},
    "comet.display_summary_level": {"type": int, "default": 1},
    # Logging
    "comet.logging.file": {"type": str},
    "comet.logging.file_level": {"type": str, "default": "INFO"},
    "comet.logging.file_overwrite": {"type": bool, "default": False},
    "comet.logging.hide_api_key": {"type": bool, "default": True},
    "comet.logging.console": {"type": str},
    "comet.logging.metrics_ignore": {
        "type": list,
        "default": "keras:batch_size,keras:batch_batch",
    },
    "comet.logging.parameters_ignore": {
        "type": list,
        "default": "keras:verbose,keras:do_validation,keras:validation_steps",
    },
    "comet.logging.others_ignore": {"type": list, "default": ""},
    "comet.logging.env_blacklist": {
        "type": list,
        "default": "api_key,apikey,authorization,passwd,password,secret,token,comet",
    },
    # Timeout, unit is seconds
    "comet.timeout.cleaning": {"type": int, "default": DEFAULT_STREAMER_MSG_TIMEOUT},
    "comet.timeout.upload": {
        "type": int,
        "default": ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    },
    "comet.timeout.http": {"type": int, "default": 10},
    "comet.optimizer_timeout.http": {"type": int, "default": 30},
    "comet.timeout.api": {"type": int, "default": 10},
    "comet.timeout.file_upload": {
        "type": int,
        "default": DEFAULT_FILE_UPLOAD_READ_TIMEOUT,
    },
    "comet.timeout.file_download": {"type": int, "default": 600},
    "comet.timeout.artifact_download": {
        "type": int,
        "default": DEFAULT_ARTIFACT_DOWNLOAD_TIMEOUT,
    },
    "comet.timeout.remote_model_download": {
        "type": int,
        "default": DEFAULT_REMOTE_MODEL_DOWNLOAD_TIMEOUT,
    },
    # HTTP Allow header
    "comet.allow_header.name": {"type": str},
    "comet.allow_header.value": {"type": str},
    # Backend minimal rest V2 version
    "comet.rest_v2_minimal_backend_version": {"type": str, "default": "1.2.78"},
    # Feature flags
    "comet.override_feature.sdk_http_logging": {
        "type": bool
    },  # Leave feature toggle default to None
    "comet.override_feature.sdk_use_http_messages": {
        "type": bool
    },  # Leave feature toggle default to None
    "comet.override_feature.sdk_log_env_variables": {
        "type": bool
    },  # Leave feature toggle default to None
    "comet.override_feature.sdk_announcement": {
        "type": bool
    },  # Leave feature toggle default to None
    # Experiment log controls:
    "comet.system_cpu_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.system_gpu_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.system_network_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.system_disk_interval": {
        "type": int,
        "default": DEFAULT_SYSTEM_MONITORING_INTERVAL,
    },
    "comet.auto_log.cli_arguments": {"type": bool},
    "comet.auto_log.code": {"type": bool},
    "comet.auto_log.disable": {"type": bool},
    "comet.auto_log.env_cpu": {"type": bool},
    "comet.auto_log.env_cpu_per_core": {"type": bool, "default": False},
    "comet.auto_log.env_details": {"type": bool},
    "comet.auto_log.env_gpu": {"type": bool},
    "comet.auto_log.env_host": {"type": bool},
    "comet.auto_log.env_network": {"type": bool},
    "comet.auto_log.env_disk": {"type": bool},
    "comet.auto_log.git_metadata": {"type": bool},
    "comet.auto_log.git_patch": {"type": bool},
    "comet.auto_log.graph": {"type": bool},
    "comet.auto_log.metrics": {"type": bool},
    "comet.auto_log.figures": {"type": bool, "default": True},
    "comet.auto_log.output_logger": {"type": str},
    "comet.auto_log.parameters": {"type": bool},
    "comet.auto_log.histogram_tensorboard": {"type": bool, "default": False},
    "comet.auto_log.histogram_epoch_rate": {"type": int, "default": 1},
    "comet.auto_log.histogram_weights": {"type": bool, "default": False},
    "comet.auto_log.histogram_gradients": {"type": bool, "default": False},
    "comet.auto_log.histogram_activations": {"type": bool, "default": False},
    "comet.keras.histogram_name_prefix": {
        "type": str,
        "default": "{layer_num:0{max_digits}d}",
    },
    "comet.keras.histogram_activation_index_list": {"type": "int_list", "default": "0"},
    "comet.keras.histogram_activation_layer_list": {"type": list, "default": "-1"},
    "comet.keras.histogram_batch_size": {"type": int, "default": 1000},
    "comet.keras.histogram_gradient_index_list": {"type": "int_list", "default": "0"},
    "comet.keras.histogram_gradient_layer_list": {"type": list, "default": "-1"},
    "comet.auto_log.metric_step_rate": {"type": int, "default": 10},
    "comet.auto_log.co2": {"type": bool},
    "comet.auto_log.tfma": {"type": bool, "default": False},
    "comet.distributed_node_identifier": {"type": str},
    # Internals:
    "comet.internal.reporting": {"type": bool, "default": True},
    "comet.internal.file_upload_worker_ratio": {
        "type": int,
        "default": DEFAULT_POOL_RATIO,
    },
    "comet.internal.worker_count": {"type": int},
    "comet.internal.check_tls_certificate": {"type": bool, "default": True},
    "comet.internal.sentry_dsn": {
        "type": str,
        "default": "https://cf2d5b1c58bc9ffcc76092dd8ddb5edf@o168229.ingest.us.sentry.io/4505505645330432",
    },
    "comet.internal.sentry_debug": {"type": bool, "default": False},
    # Deprecated:
    "comet.display_summary": {"type": bool, "default": None},
    "comet.auto_log.weights": {"type": bool, "default": None},
    # Related to `comet_ml.start`
    "comet.resume_strategy": {"type": str, "default": None},
    "comet.offline": {"type": bool, "default": False},
    # Error tracking.
    "comet.error_tracking.enable": {"type": bool},
    # Related to message batch processing
    "comet.message_batch.use_compression": {
        "type": bool,
        "default": MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
    },
    "comet.message_batch.metric_interval": {
        "type": float,
        "default": MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
    },
    "comet.message_batch.metric_max_size": {
        "type": int,
        "default": MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
    },
    "comet.message_batch.parameters_interval": {
        "type": int,
        "default": DEFAULT_PARAMETERS_BATCH_INTERVAL_SECONDS,
    },
    "comet.message_batch.stdout_interval": {
        "type": int,
        "default": MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS,
    },
    "comet.message_batch.stdout_max_size": {
        "type": int,
        "default": MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE,
    },
    "comet.message_batch.artifact_remote_assets_interval": {
        "type": int,
        "default": ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS,
    },
    "comet.message_batch.artifact_remote_assets_max_size": {
        "type": int,
        "default": ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE,
    },
    # Fallback streamer
    "comet.fallback_streamer.connection_check_interval": {
        "type": int,
        "default": FALLBACK_STREAMER_CONNECTION_CHECK_INTERVAL_SECONDS,
    },
    "comet.fallback_streamer.keep_offline_zip": {
        "type": bool,
        "default": False,
    },
    "comet.fallback_streamer.fallback_to_offline_min_backend_version": {
        "type": str,
        "default": "3.3.11",
    },
    "comet.novel_model_registry_api.minimum_backend_version": {
        "type": str,
        "default": "3.5.42",
    },
    "comet.api_experiment.delete_tags_minimum_backend_version": {
        "type": str,
        "default": "3.12.26",
    },
    "comet.api_experiment.get_all_experiment_metrics_minimum_backend_version": {
        "type": str,
        "default": "3.20.60",
    },
    "comet.artifact.remote_assets_batch_minimum_backend_version": {
        "type": str,
        "default": "3.12.26",
    },
    "comet.disable_announcement": {"type": bool, "default": False},
    "comet.rich_output": {"type": bool, "default": True},
    "comet.s3_multipart.size_threshold": {
        "type": int,
        "default": S3_MULTIPART_SIZE_THRESHOLD_DEFAULT,
    },
    "comet.s3_multipart.expires_in": {
        "type": int,
        "default": S3_MULTIPART_EXPIRES_IN,
    },
    "comet.s3_direct_multipart.upload_enabled": {
        "type": bool,
        "default": False,
    },
    "comet.http_session.retry_total": {
        "type": int,
        "default": HTTP_SESSION_RETRY_TOTAL_DEFAULT,
    },
    "comet.http_session.retry_backoff_factor": {
        "type": int,
        "default": HTTP_SESSION_RETRY_BACKOFF_FACTOR_DEFAULT,
    },
}


def get_config(setting=None):
    # type: (Any) -> Union[Config, Any]
    """
    Get a config or setting from the current config
    (os.environment or .env file).

    Note: this is not cached, so every time we call it, it
    re-reads the file. This makes these values always up to date
    at the expense of re-getting the data.
    """
    if _in_streamlit_environment():
        from ._streamlit import StreamlitConfig

        cfg = StreamlitConfig(CONFIG_MAP)
    else:
        cfg = Config(CONFIG_MAP)

    if setting is None:
        return cfg
    else:
        return cfg[setting]


_last_api_key_returned = None


def get_last_returned_api_key() -> Optional[str]:
    return _last_api_key_returned


def get_api_key(api_key: Optional[str], config: Config) -> Optional[str]:
    if api_key is None:
        api_key = config["comet.api_key"]
    else:
        api_key = api_key

    final_api_key = secrets.interpreter.interpret(api_key)

    parsed_api_key = parse_api_key(final_api_key)
    if parsed_api_key is None:
        return None

    # update config parameters from API key
    _update_comet_url_override(parsed_api_key, config=config)

    final_api_key = parsed_api_key.api_key
    # Hide api keys from the log
    if final_api_key and config.get_bool(None, "comet.logging.hide_api_key") is True:
        _get_comet_logging_config().redact_string(final_api_key)

    global _last_api_key_returned
    _last_api_key_returned = final_api_key

    return final_api_key


def get_comet_url_override(config: Optional[Config]) -> str:
    """This MUST always be used to get the COMET_URL_OVERRIDE value"""
    if config is None:
        config = get_config()
    url_override = config[COMET_URL_OVERRIDE_CONFIG_KEY]
    if url_override is None or url_override == "":
        return DEFAULT_COMET_URL_OVERRIDE
    else:
        return url_override


def _update_comet_url_override(api_key: CometApiKey, config: Config):
    config_url_override = config[COMET_URL_OVERRIDE_CONFIG_KEY]
    if config_url_override is not None and config_url_override != "":
        config_base_url = get_root_url(config_url_override)
        if api_key.base_url is not None and api_key.base_url != config_base_url:
            LOGGER.warning(
                BASE_URL_MISMATCH_CONFIG_API_KEY, config_base_url, api_key.base_url
            )
        # do not change base url
        return

    # set the global variable with value from API key or with default one
    if api_key.base_url is not None:
        comet_url_override = url_join(api_key.base_url, DEFAULT_COMET_URL_OVERRIDE_PATH)
    else:
        comet_url_override = DEFAULT_COMET_URL_OVERRIDE

    CONFIG_MAP[COMET_URL_OVERRIDE_CONFIG_KEY]["default"] = comet_url_override


def get_project_name(project_name: str, config: Config) -> str:
    return project_name if project_name is not None else config["comet.project_name"]


def get_workspace(workspace: str, config: Config) -> str:
    return workspace if workspace is not None else config["comet.workspace"]


def get_check_tls_certificate(config: Config) -> bool:
    return config.get_bool(None, "comet.internal.check_tls_certificate")


def get_comet_timeout_http(config: Config) -> float:
    return config.get_int(None, "comet.timeout.http")


def discard_api_key(api_key):
    # type: (str) -> None
    """Discards the provided API key as invalid. After this method invocation the discarded key will not be masked in
    the logger output.
    """
    if api_key is not None:
        _get_comet_logging_config().discard_string_from_redact(api_key)


def get_display_summary_level(display_summary_level, config):
    if display_summary_level is None:
        return config["comet.display_summary_level"]
    else:
        try:
            return int(display_summary_level)
        except Exception:
            LOGGER.warning(
                CONFIG_GET_DISPLAY_SUMMARY_LEVEL_INVALID_WARNING, display_summary_level
            )
            return 1


def get_ws_url(ws_server_from_backend, config):
    """Allow users to override the WS url from the backend using the usual
    config mechanism
    """
    ws_server = config.get_string(
        None, "comet.ws_url_override", default=ws_server_from_backend
    )
    # Ws server can be None if the backend stop returning it or when used in `comet_check` for example
    if ws_server is None:
        return ws_server

    return sanitize_url(
        ws_server,
        ending_slash=False,
    )


def get_previous_experiment(previous_experiment, config):
    if previous_experiment is None:
        return config["comet.experiment_key"]
    else:
        return previous_experiment


def save(directory=None, save_all=False, force=False, **settings):
    """
    An easy way to create a config file.

    Args:
        directory: str (optional), location to save the
            .comet.config file. Typical values are "~/" (home)
            and "./" (current directory). Default is "~/" or
            COMET_CONFIG, if set
        save_all: bool (optional). If True, will create
            entries for all items that are configurable
            with their defaults. Default is False
        force: bool (optional). If True, overwrite pre-existing
            .comet.config file. If False, ask.
        settings: any valid setting and value

    Valid settings include:

    * api_key
    * disable_auto_logging
    * experiment_key
    * offline_directory
    * workspace
    * project_name
    * logging_console
    * logging_file
    * logging_file_level
    * logging_file_overwrite
    * timeout_cleaning
    * timeout_upload

    Examples:

    ```python
    >>> import comet_ml
    >>> comet_ml.config.save(api_key="...")
    >>> comet_ml.config.save(api_key="...", directory="./")
    ```
    """
    cfg = get_config()
    cfg._set_settings(settings)
    cfg.save(directory, save_all=save_all, force=force)


def _api_key_save(config_path, api_key):
    # type: (str, str) -> None
    """
    Low-level function to only change the api_key
    of a .comet.config file in the home directory.
    """
    from configobj import ConfigObj

    if os.path.exists(config_path):
        config = ConfigObj(config_path)
        try:
            shutil.copyfile(config_path, config_path + ".bak")
        except Exception:
            LOGGER.warning(CONFIG_FAILED_BACKUP_CONFIG_KEY_WARNING, exc_info=True)
    else:
        config = ConfigObj()
        config.filename = config_path

    if "comet" not in config:
        config["comet"] = {}

    config["comet"]["api_key"] = api_key
    config.write()
    LOGGER.info(
        CONFIG_API_KEY_SAVED_INFO,
        config_path,
    )


def init(
    directory: Optional[str] = None,
    force: Optional[bool] = False,
    anonymous: bool = False,
    **settings: Any
) -> None:
    """
    An easy, safe, interactive way to set and save your settings.

    Will ask for your api_key if not already set. Your
    api_key will not be shown.

    Args:
        directory: str (optional), location to save the
            .comet.config file. Typical values are "~/" (home)
            and "./" (current directory). Default is "~/" or
            COMET_CONFIG, if set.
        force: bool (optional), if True, force the user to
            enter required information, such as the api key
        anonymous: bool (optional), if True, use comet temporarily without an api key.
        settings: any valid setting and value

    Valid settings include:

    * api_key
    * disable_auto_logging
    * experiment_key
    * offline_directory
    * workspace
    * project_name
    * logging_console
    * logging_file
    * logging_file_level
    * logging_file_overwrite
    * timeout_cleaning
    * timeout_upload

    Examples:

    For use in a notebook, or other interactive code:

    ```python
    >>> import comet_ml
    >>> comet_ml.init()
    ```

    You can also provide values for other settings:

    ```python
    >>> import comet_ml
    >>> comet_ml.init(project_name='colab')
    ```
    """
    _init(directory=directory, force=force, anonymous=anonymous, settings=settings)


def _init_get_api_key(
    prompt_user: bool,
    settings: Optional[Dict[str, Any]],
    config: Config,
    config_path: str,
    anonymous: bool = False,
    force: Optional[bool] = False,
) -> Tuple[Optional[str], bool]:
    api_client = connection.get_comet_api_client()

    if settings is None:
        settings = {}

    need_to_save = False
    if "api_key" in settings:
        api_key = settings.pop("api_key", None)
        if api_key:
            need_to_save = True
    elif not force:
        api_key = config.get_string(None, "comet.api_key")
    else:
        api_key = None

    if api_key is not None and anonymous is True:
        is_active = _get_anonymous_api_key_state(api_client, api_key)
        if is_active is not None:
            if is_active:
                LOGGER.info(CONFIG_USING_ANONYMOUS_MODE)
            else:
                raise CometException(CONFIG_CLAIMED_ANONYMOUS_USER % config_path)

    elif api_key is None and anonymous is True:
        api_key = _get_anonymous_api_key(api_client)
        if api_key:
            LOGGER.info(CONFIG_USING_ANONYMOUS_MODE)
            need_to_save = True

    if api_key is None:
        if prompt_user:
            api_key = get_api_key_from_user()
            if api_key:
                need_to_save = True

    return api_key, need_to_save


def _check_api_key_validity(api_key: str) -> None:
    from .api import API

    API(api_key=api_key).get_workspaces()


def _get_anonymous_api_key(api_client: "connection.CometApiClient") -> str:
    try:
        response = api_client.post_from_endpoint("anonymous-user", payload={})
    except CometRestApiException as e:
        raise AnonymousModeException(e.safe_msg, e.sdk_error_code)

    return response.json()["apiKey"]


def _get_anonymous_api_key_state(
    api_client: "connection.CometApiClient", api_key: str
) -> Optional[bool]:
    try:
        response = api_client.get_from_endpoint(
            "anonymous/is-active", params={"apiKey": api_key}
        )
    except CometRestApiException as e:
        raise AnonymousModeException(e.safe_msg, e.sdk_error_code)

    if "sdk_error_code" in response:
        sdk_error_code = response.get("sdk_error_code")

        if sdk_error_code == 7000007:
            LOGGER.error(CONFIG_ANONYMOUS_UNALLOWED_ON_PREM)

    return response.get("active")


def _init(
    directory: Optional[str] = None,
    should_prompt_user: Optional[bool] = None,
    anonymous: bool = False,
    settings: Optional[Dict[str, Any]] = None,
    force: Optional[bool] = False,
) -> None:
    # We only save the api_key if given by parameter, or
    # by getpass. We don't save it if in the environment

    config = get_config()
    config_path = (
        _config_path_from_directory(directory)
        if directory is not None
        else _get_default_config_path()
    )

    if should_prompt_user is None:
        should_prompt_user = is_interactive()

    api_key, need_to_save = _init_get_api_key(
        prompt_user=should_prompt_user,
        settings=settings,
        config=config,
        config_path=config_path,
        anonymous=anonymous,
        force=force,
    )

    if api_key is None:
        LOGGER.info(CONFIG_API_KEY_IS_NOT_SET)
    else:
        try:
            _check_api_key_validity(api_key)
        except InvalidAPIKey:
            LOGGER.error(
                CONFIG_API_KEY_IS_INVALID,
                get_comet_url_override(config),
                exc_info=True,
            )
            return None
        except Exception:
            LOGGER.error(CONFIG_API_KEY_CHECK_FAILED, exc_info=True)
            return None

        # From here we know that the API Key is valid for the configured Comet installation

        if need_to_save:
            try:
                _api_key_save(config_path, api_key)
            except Exception:
                LOGGER.warning(CONFIG_API_KEY_SAVE_FAILED_WARNING, exc_info=True)
                return None

    if settings:
        # Set in environment to save:
        config._set_settings(settings, environ=True)


def init_onprem(force=False):
    # type: (bool) -> None

    if force:
        for var in ["COMET_OPTIMIZER_URL", "COMET_URL_OVERRIDE", "COMET_API_KEY"]:
            if var in os.environ:
                del os.environ[var]

    config = get_config()
    config_path = _get_default_config_path()

    client_url = get_comet_url_override(config)

    root_url = sanitize_url(get_root_url(client_url))

    if root_url == COMET_ROOT_URL or force:
        LOGGER.info(CONFIG_INIT_ONPREM_START_INFO)

        client_url = _input_user(CONFIG_INIT_ONPREM_USER_INPUT_INFO)

        # Set the environment variables, config.save will save them in the local config file
        os.environ["COMET_URL_OVERRIDE"] = client_url

    # We currently force the user confirmation but it might hang
    # if not in a TTY
    config.save(_prompt_user_confirmation=True, force=force)

    # Set api key if necessary, and save it too
    try:
        api_key, need_to_save = _init_get_api_key(
            prompt_user=True,
            settings={},
            config=config,
            config_path=config_path,
            force=force,
        )

        if api_key is None:
            LOGGER.info(CONFIG_API_KEY_IS_NOT_SET)
        else:
            _check_api_key_validity(api_key)

            # From here we know that the API Key is valid for the configured Comet installation

            if need_to_save:
                _api_key_save(config_path, api_key)
    except Exception:
        # TODO: Add more specific error message for individual part of the try block
        raise Exception(CONFIG_INIT_ONPREM_FAILED_EXCEPTION)


def collect_experiment_info(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    workspace: Optional[str] = None,
) -> experiment_info.ExperimentInfo:
    config = get_config()
    final_project_name = config.get_string(project_name, "comet.project_name")
    final_workspace = config.get_string(workspace, "comet.workspace")
    final_api_key = get_api_key(api_key, config)

    return experiment_info.ExperimentInfo(
        api_key=final_api_key,
        project_name=final_project_name,
        workspace=final_workspace,
    )
