#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Callable

from vllm_ascend.trace.prof_wrapper import marker_prof_wrapper, timer_prof_wrapper

logger = logging.getLogger(__name__)

_WRAPPED_ATTR = "__vllm_ascend_trace_wrapped__"
_APPLIED_SCOPES: set[str] = set()
_DEFAULT_NAMELIST = Path(__file__).with_name("omnilogger_namelist_vllm_023.yml")

_WRAPPERS: dict[str, Callable[..., Any]] = {
    "marker": marker_prof_wrapper,
    "timer": timer_prof_wrapper,
}


def trace_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TRACE", "0") == "1"


def iter_namelist_paths() -> list[Path]:
    paths: list[Path] = []
    custom_path = os.getenv("PROFILING_NAMELIST")
    if custom_path:
        path = Path(custom_path)
        if path.exists():
            paths.append(path)
        else:
            logger.warning(
                "VLLM_ASCEND_TRACE is enabled but PROFILING_NAMELIST does not "
                "exist: %s. Falling back to packaged vLLM 0.23 namelist.",
                custom_path,
            )

    if _DEFAULT_NAMELIST.exists() and _DEFAULT_NAMELIST not in paths:
        paths.append(_DEFAULT_NAMELIST)
    return paths


def _module_allowed_for_scope(module_name: str, scope: str) -> bool:
    if scope == "platform":
        return (
            module_name.startswith("vllm.entrypoints.")
            or module_name.startswith("vllm.v1.")
            or module_name.startswith("vllm_ascend.core.")
        )
    if scope == "worker":
        return (
            module_name.startswith("vllm_ascend.worker.")
            or module_name.startswith("vllm_ascend.distributed.kv_transfer.")
        )
    return True


def _target_allowed(target: dict[str, Any], module_name: str, scope: str) -> bool:
    target_scope = target.get("trace_scope") or target.get("scope")
    if target_scope:
        return target_scope == scope
    return _module_allowed_for_scope(module_name, scope)


def _split_module_and_class(module_value: str) -> tuple[str, str | None]:
    if ":" not in module_value:
        return module_value, None
    module_name, class_name = module_value.split(":", 1)
    return module_name, class_name


def _wrap_target(
    module_name: str,
    class_name: str | None,
    function_name: str,
    wrapper_method: Callable[..., Any],
    params: dict[str, Any],
) -> bool:
    original_module = importlib.import_module(module_name)
    target_obj: Any = original_module
    target_label = module_name
    if class_name:
        target_obj = getattr(original_module, class_name)
        target_label = f"{module_name}:{class_name}"

    original_function = getattr(target_obj, function_name)
    if getattr(original_function, _WRAPPED_ATTR, False):
        logger.info("Trace target already wrapped: %s.%s", target_label, function_name)
        return False

    wrapped_function = wrapper_method(original_function, params)
    setattr(wrapped_function, _WRAPPED_ATTR, True)
    setattr(target_obj, function_name, wrapped_function)
    logger.info("Trace wrapped %s.%s", target_label, function_name)
    return True


def apply_trace_patches(scope: str) -> None:
    if not trace_enabled():
        return
    if scope in _APPLIED_SCOPES:
        return

    paths = iter_namelist_paths()
    if not paths:
        logger.warning(
            "VLLM_ASCEND_TRACE is enabled but no trace namelist is available."
        )
        _APPLIED_SCOPES.add(scope)
        return

    for path in paths:
        _apply_trace_patches_from_path(path, scope)
    _APPLIED_SCOPES.add(scope)


def _apply_trace_patches_from_path(path: Path, scope: str) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        logger.warning(
            "Skipping trace namelist %s because PyYAML is not installed.",
            path,
        )
        return

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        logger.exception("Failed to parse trace namelist %s: %s", path, exc)
        return

    profiler_type = config.get("type", "marker")
    wrapper_method = _WRAPPERS.get(profiler_type)
    if wrapper_method is None:
        logger.warning(
            "Skipping trace namelist %s with unsupported type %r",
            path,
            profiler_type,
        )
        return

    base_params = config.get("base_params", {}) or {}
    for target in config.get("targets", []) or []:
        module_value = target.get("module")
        function_name = target.get("function_name")
        if not module_value or not function_name:
            logger.warning("Skipping invalid trace target in %s: %s", path, target)
            continue

        module_name, class_name = _split_module_and_class(module_value)
        if not _target_allowed(target, module_name, scope):
            continue

        params = dict(base_params)
        params.update(
            {
                "entry_operation": target.get("entry_operation"),
                "exit_operation": target.get("exit_operation"),
                "entry_message": target.get("entry_message"),
                "exit_message": target.get("exit_message"),
            }
        )
        try:
            _wrap_target(
                module_name,
                class_name,
                function_name,
                wrapper_method,
                params,
            )
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Skipping trace target %s.%s from %s: %s",
                module_value,
                function_name,
                path,
                exc,
            )
        except Exception as exc:
            logger.exception(
                "Unexpected error while wrapping trace target %s.%s from %s: %s",
                module_value,
                function_name,
                path,
                exc,
            )
