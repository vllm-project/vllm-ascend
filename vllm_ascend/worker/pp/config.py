# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_pp_send_buffer_size(config: VllmConfig) -> int:
    return _get_buffer_size(config, "pp_send_buffer_size")


def get_pp_recv_buffer_size(config: VllmConfig) -> int:
    return _get_buffer_size(config, "pp_recv_buffer_size")


def _get_buffer_size(config: VllmConfig, key: str) -> int:
    additional = config.additional_config
    if not isinstance(additional, Mapping):
        return 0
    size = additional.get(key, 0)
    if type(size) is not int or size < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    if size and not config.use_v2_model_runner:
        raise ValueError("PP buffering requires Model Runner V2")
    return size
