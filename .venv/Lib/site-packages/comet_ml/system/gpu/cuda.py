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

import logging
from typing import List, Optional

from comet_ml.vendor.nvidia_ml import pynvml

from ...logging_messages import (
    SYSTEM_GPU_CUDA_INVALID_DEVICE_ID_ERROR,
    SYSTEM_GPU_CUDA_VISIBLE_DEVICES_AMBIGUOUS_UUID_ERROR,
    SYSTEM_GPU_CUDA_VISIBLE_DEVICES_PARSE_FAILED_PYNVML_ERROR,
    SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_DEVICE_NUM_ERROR,
    SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_FORMAT_ERROR,
    SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_UUID_ERROR,
)
from . import identifiers

LOGGER = logging.getLogger(__name__)
_NO_DEVICE_VALUES = ["", "-1"]


def parse_visible_devices(raw_value: str) -> List[str]:
    result = []
    if raw_value in _NO_DEVICE_VALUES:
        return result

    device_identifiers = raw_value.split(",")

    for device_identifier in device_identifiers:
        uuid = _parse_device_identifier(device_identifier)
        if uuid is None:
            LOGGER.error(
                SYSTEM_GPU_CUDA_INVALID_DEVICE_ID_ERROR,
            )
            break

        result.append(uuid)

    return result


def _parse_device_identifier(device_identifier: str) -> Optional[str]:
    if device_identifier.startswith("GPU-"):
        return _parse_uuid(device_identifier)
    if device_identifier.isdigit():
        device_index = int(device_identifier)
        return _parse_index(device_index)

    LOGGER.error(
        SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_FORMAT_ERROR.format(device_identifier)
    )
    return None


def _parse_uuid(uuid: str) -> str:
    all_uuids = identifiers.all()
    if uuid not in all_uuids:
        uuid = _complete_uuid(uuid, all_uuids)

    return uuid


def _complete_uuid(short_uuid: str, full_uuids: List[str]) -> Optional[str]:
    suitable_uuids = [uuid for uuid in full_uuids if uuid.startswith(short_uuid)]
    if len(suitable_uuids) == 1:
        return suitable_uuids[0]

    if len(suitable_uuids) == 0:
        LOGGER.error(
            SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_UUID_ERROR.format(short_uuid)
        )
        return None

    if len(suitable_uuids) > 1:
        LOGGER.error(
            SYSTEM_GPU_CUDA_VISIBLE_DEVICES_AMBIGUOUS_UUID_ERROR.format(short_uuid)
        )
        return None


def _parse_index(index: int) -> Optional[str]:
    try:
        if index > pynvml.nvmlDeviceGetCount():
            LOGGER.error(
                SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_DEVICE_NUM_ERROR.format(index)
            )
            return None

        return identifiers.get_by_index(index)
    except pynvml.NVMLError:
        LOGGER.error(
            SYSTEM_GPU_CUDA_VISIBLE_DEVICES_PARSE_FAILED_PYNVML_ERROR,
            exc_info=True,
        )
        return None
