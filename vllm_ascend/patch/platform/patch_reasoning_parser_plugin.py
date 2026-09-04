# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any

from vllm.entrypoints.cli import serve
from vllm.reasoning import ReasoningParserManager

_EXPECTED_RUN_HEADLESS_PARAMETERS = ("args",)
_ORIGINAL_RUN_HEADLESS = serve.run_headless


def _validate_run_headless_contract(run_headless: Callable[..., Any]) -> None:
    parameters = tuple(signature(run_headless).parameters)
    if parameters != _EXPECTED_RUN_HEADLESS_PARAMETERS:
        raise RuntimeError(
            "vLLM run_headless signature changed: "
            f"expected {_EXPECTED_RUN_HEADLESS_PARAMETERS}, got {parameters}. "
            "Remove or update the vLLM Ascend reasoning parser plugin patch."
        )


def _import_reasoning_parser_plugin(args: Any) -> None:
    plugin_path = getattr(args, "reasoning_parser_plugin", None)
    if not plugin_path or len(plugin_path) <= 3:
        return

    ReasoningParserManager.import_reasoning_parser(plugin_path)


@wraps(_ORIGINAL_RUN_HEADLESS)
def _patched_run_headless(args: Any) -> Any:
    _import_reasoning_parser_plugin(args)
    return _ORIGINAL_RUN_HEADLESS(args)


_validate_run_headless_contract(_ORIGINAL_RUN_HEADLESS)
serve.run_headless = _patched_run_headless
