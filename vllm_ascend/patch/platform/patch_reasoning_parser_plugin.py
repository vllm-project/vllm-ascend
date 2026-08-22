# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any

from vllm.config import VllmConfig
from vllm.reasoning import ReasoningParserManager

_EXPECTED_POST_INIT_PARAMETERS = ("self",)


def _validate_post_init_contract(post_init: Callable[..., Any]) -> None:
    parameters = tuple(signature(post_init).parameters)
    if parameters != _EXPECTED_POST_INIT_PARAMETERS:
        raise RuntimeError(
            "vLLM VllmConfig.__post_init__ signature changed: "
            f"expected {_EXPECTED_POST_INIT_PARAMETERS}, got {parameters}. "
            "Remove or update the vLLM Ascend reasoning parser plugin patch."
        )


_ORIGINAL_VLLM_CONFIG_POST_INIT = VllmConfig.__post_init__
_validate_post_init_contract(_ORIGINAL_VLLM_CONFIG_POST_INIT)


def _import_reasoning_parser_plugin(vllm_config: VllmConfig) -> None:
    reasoning_config = vllm_config.reasoning_config
    if reasoning_config is None or not reasoning_config.reasoning_parser:
        return

    plugin_path = vllm_config.structured_outputs_config.reasoning_parser_plugin
    if not plugin_path:
        return

    parser_name = reasoning_config.reasoning_parser
    if parser_name not in ReasoningParserManager.list_registered():
        ReasoningParserManager.import_reasoning_parser(plugin_path)


@wraps(_ORIGINAL_VLLM_CONFIG_POST_INIT)
def _vllm_config_post_init(self: VllmConfig) -> None:
    _import_reasoning_parser_plugin(self)
    _ORIGINAL_VLLM_CONFIG_POST_INIT(self)


VllmConfig.__post_init__ = _vllm_config_post_init
