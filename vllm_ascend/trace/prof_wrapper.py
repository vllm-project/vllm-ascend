#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import functools
import inspect
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def execute_operation(
    operation_str: str | None, param_dict: dict[str, Any], func_name: str = ""
) -> None:
    if not operation_str:
        return
    try:
        exec(operation_str, param_dict)
    except Exception as exc:
        logger.exception("Error executing trace code for %s: %s", func_name, exc)


def marker_prof_wrapper(original_method: Callable[..., Any],
                        params: dict[str, Any]) -> Callable[..., Any]:
    entry_operation = params.get("entry_operation")
    exit_operation = params.get("exit_operation")

    if inspect.iscoroutinefunction(original_method):

        @functools.wraps(original_method)
        async def async_wrapper(self, *args, **kwargs):
            param_dict = {"self": self, "args": args, "kwargs": kwargs}
            execute_operation(entry_operation, param_dict, original_method.__name__)
            result = await original_method(self, *args, **kwargs)
            param_dict["result"] = result
            execute_operation(exit_operation, param_dict, original_method.__name__)
            return result

        return async_wrapper

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        param_dict = {"self": self, "args": args, "kwargs": kwargs}
        execute_operation(entry_operation, param_dict, original_method.__name__)
        result = original_method(self, *args, **kwargs)
        param_dict["result"] = result
        execute_operation(exit_operation, param_dict, original_method.__name__)
        return result

    return wrapper


def timer_prof_wrapper(original_method: Callable[..., Any],
                       params: dict[str, Any]) -> Callable[..., Any]:
    entry_operation = params.get("entry_operation")
    exit_operation = params.get("exit_operation")

    if inspect.iscoroutinefunction(original_method):

        @functools.wraps(original_method)
        async def async_wrapper(self, *args, **kwargs):
            param_dict = {"self": self, "args": args, "kwargs": kwargs}
            execute_operation(entry_operation, param_dict, original_method.__name__)
            start_time = time.time()
            result = await original_method(self, *args, **kwargs)
            end_time = time.time()
            param_dict.update(
                {"result": result, "start_time": start_time, "end_time": end_time}
            )
            execute_operation(exit_operation, param_dict, original_method.__name__)
            return result

        return async_wrapper

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        param_dict = {"self": self, "args": args, "kwargs": kwargs}
        execute_operation(entry_operation, param_dict, original_method.__name__)
        start_time = time.time()
        result = original_method(self, *args, **kwargs)
        end_time = time.time()
        param_dict.update(
            {"result": result, "start_time": start_time, "end_time": end_time}
        )
        execute_operation(exit_operation, param_dict, original_method.__name__)
        return result

    return wrapper
