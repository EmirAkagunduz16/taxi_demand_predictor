# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import platform
import threading
import warnings
from typing import Any, Dict, List, Optional

from comet_ml.batch_utils import MessageBatchItem
from comet_ml.config import get_config
from comet_ml.constants import (
    PAYLOAD_EXPERIMENT_KEY,
    PAYLOAD_LOCAL_TIMESTAMP,
    PAYLOAD_OFFSET,
    PAYLOAD_OUTPUT,
    PAYLOAD_OUTPUT_LINES,
    PAYLOAD_RUN_CONTEXT,
    PAYLOAD_STDERR,
)
from comet_ml.messages import StandardOutputMessage
from comet_ml.utils import get_comet_version

import urllib3
from requests import Session
from requests.adapters import HTTPAdapter
from requests_toolbelt.adapters.socket_options import TCPKeepAliveAdapter
from urllib3 import Retry

STATUS_FORCELIST_NO_AUTH_ERRORS = [500, 502, 503, 504]
STATUS_FORCELIST_FULL = [401, 403]
STATUS_FORCELIST_FULL.extend(STATUS_FORCELIST_NO_AUTH_ERRORS)

# Maximum backoff time.
BACKOFF_MAX = 120


def get_http_session(
    retry_strategy: Optional[Retry] = None,
    verify_tls: bool = True,
    tcp_keep_alive: bool = False,
) -> Session:
    session = Session()

    # Add default debug headers
    session.headers.update(
        {
            "X-COMET-DEBUG-SDK-VERSION": get_comet_version(),
            "X-COMET-DEBUG-PY-VERSION": platform.python_version(),
        }
    )

    # Setup retry strategy if asked
    http_adapter = None
    https_adapter = None
    if tcp_keep_alive is True:
        http_adapter = TCPKeepAliveAdapter(
            idle=60, count=5, interval=60, max_retries=retry_strategy
        )
        https_adapter = TCPKeepAliveAdapter(
            idle=60, count=5, interval=60, max_retries=retry_strategy
        )
    elif tcp_keep_alive is False and retry_strategy is not None:
        http_adapter = HTTPAdapter(max_retries=retry_strategy)
        https_adapter = HTTPAdapter(max_retries=retry_strategy)

    if http_adapter is not None:
        session.mount("http://", http_adapter)

    if https_adapter is not None:
        session.mount("https://", https_adapter)

    # Setup HTTP allow header if configured
    config = get_config()  # This can be slow if called for every new session
    allow_header_name = config["comet.allow_header.name"]
    allow_header_value = config["comet.allow_header.value"]

    if allow_header_name and allow_header_value:
        session.headers[allow_header_name] = allow_header_value

    if verify_tls is False:
        # Only the set the verify if it's disabled. The current default for the verify attribute is
        # True but this way we will survive any change of the default value
        session.verify = False
        # Also filter the warning that urllib3 emits to not overflow the output with them
        warnings.filterwarnings(
            "ignore", category=urllib3.exceptions.InsecureRequestWarning
        )

    return session


def get_retry_strategy(
    status_forcelist: Optional[List[int]] = None,
    total_retries: Optional[int] = None,
    backoff_factor: Optional[int] = None,
) -> Retry:

    # The total backoff sleeping time is computed like that:
    # backoff = 2
    # retries = 3
    # s = lambda b, i: b * (2 ** (i - 1))
    # sleep = sum(s(backoff, i) for i in range(1, retries + 1))
    # Will wait up to 14s

    if status_forcelist is None:
        status_forcelist = STATUS_FORCELIST_NO_AUTH_ERRORS

    settings = get_config()
    if total_retries is None:
        total_retries = settings.get_int(None, "comet.http_session.retry_total")
    if backoff_factor is None:
        backoff_factor = settings.get_int(
            None, "comet.http_session.retry_backoff_factor"
        )

    if urllib3.__version__.startswith("2."):
        kwargs = {"allowed_methods": None}
    else:
        kwargs = {"method_whitelist": False}

    return Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        **kwargs
    )


THREAD_SESSIONS = threading.local()


def get_thread_session(retry: bool, verify_tls: bool, tcp_keep_alive: bool) -> Session:
    # As long as the session is not part of a reference loop, the thread local dict will be cleaned
    # up when each thread ends, garbage-collecting the Session object and closing the
    # resources
    session_key = (retry, tcp_keep_alive, verify_tls)

    cached_session = THREAD_SESSIONS.__dict__.get(
        session_key, None
    )  # type: Optional[Session]

    if cached_session:
        return cached_session

    retry_strategy = False
    if retry is True:
        retry_strategy = get_retry_strategy()

    new_session = get_http_session(
        retry_strategy=retry_strategy,
        tcp_keep_alive=tcp_keep_alive,
        verify_tls=verify_tls,
    )
    THREAD_SESSIONS.__dict__[session_key] = new_session

    return new_session


def format_stdout_message_batch_items(
    batch_items: List[MessageBatchItem],
    timestamp: int,
    experiment_key: str,
    stderr: bool,
) -> Optional[Dict[str, Any]]:
    stdout_lines = []
    timestamp = int(timestamp * 1000)  # the Java format - milliseconds since epoch

    for item in batch_items:
        if not isinstance(item.message, StandardOutputMessage):
            continue

        if stderr != item.message.stderr:
            # different message type than requested
            continue

        stdout_lines.append(
            {
                PAYLOAD_STDERR: stderr,
                PAYLOAD_OUTPUT: item.message.output,
                PAYLOAD_LOCAL_TIMESTAMP: timestamp,
                PAYLOAD_OFFSET: item.message.message_id,
            }
        )

    if len(stdout_lines) == 0:
        return None

    payload = {
        PAYLOAD_EXPERIMENT_KEY: experiment_key,
        PAYLOAD_OUTPUT_LINES: stdout_lines,
        PAYLOAD_RUN_CONTEXT: None,
    }
    return payload


def format_message_batch_items(
    batch_items: List[MessageBatchItem], experiment_key: str
) -> Dict[str, Any]:
    """Encodes a list of messages into batch body dictionary to be used for batch endpoints"""
    messages_arr = []
    for item in batch_items:
        messages_arr.append(item.message.to_batch_message_dict())

    batch_body = {
        PAYLOAD_EXPERIMENT_KEY: experiment_key,
        "values": messages_arr,
    }
    return batch_body


def format_remote_assets_batch_items(
    batch_items: List[MessageBatchItem],
) -> Dict[str, Any]:
    messages_arr = []
    for item in batch_items:
        messages_arr.append(item.message.to_batch_message_dict())

    batch_body = {"remoteArtifactAssetRequests": messages_arr}
    return batch_body


def calculate_backoff_time(backoff_factor: float, retry_attempt: int) -> float:
    if retry_attempt <= 1:
        return 1

    backoff_value = backoff_factor * (2 ** (retry_attempt - 1))
    return min(BACKOFF_MAX, backoff_value)
