# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

from vllm.config import VllmConfig


def get_pp_buffer_size(config: VllmConfig) -> int:
    additional = config.additional_config
    if not isinstance(additional, Mapping):
        return 0
    size = additional.get("pp_send_buffer_size", 0)
    if type(size) is not int or size < 0:
        raise ValueError("pp_send_buffer_size must be a non-negative integer")
    if size and not config.use_v2_model_runner:
        raise ValueError("PP buffering requires Model Runner V2")
    return size
