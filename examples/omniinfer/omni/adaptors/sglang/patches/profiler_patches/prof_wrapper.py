# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _execute_operation(operation_str: Optional[str], param_dict: Dict[str, Any]) -> None:
    """execute_operation"""
    if operation_str:
        try:
            exec(operation_str, param_dict)
        except Exception as e:
            logging.error(f"Error executing operation: {e}")


def _get_caller_info(stack: List[inspect.FrameInfo]) -> Tuple[str, str, str]:
    """Obtain the caller's module name, class name, and function name"""
    caller_module: str = ""
    caller_class: str = ""
    caller_function: str = ""
    try:
        if len(stack) > 1:
            caller_frame: inspect.FrameType = stack[1].frame
            # module name
            caller_module = caller_frame.f_globals.get('__name__', '')
            # class name
            caller_self: Optional[Union[object, Type]] = caller_frame.f_locals.get('self') or caller_frame.f_locals.get(
                'cls')

            if caller_self:
                if inspect.isclass(caller_self):
                    caller_class = caller_self.__name__
                else:
                    caller_class = caller_self.__class__.__name__

            # function name
            caller_function = stack[1].function
    except Exception as e:
        logging.error(f"Error getting caller info: {e}")

    return caller_module, caller_class, caller_function


def _should_wrap(
        scope_name: Optional[str],
        scope_class_name: Optional[str],
        scope_function: Optional[str],
        caller_module: str,
        caller_class: str,
        caller_function: str
) -> bool:
    """Determine whether packaging logic needs to be executed """
    if not all([scope_name, scope_class_name, scope_function]):
        return True

    return (caller_module == scope_name and
            caller_class == scope_class_name and
            caller_function == scope_function)


def _sync_execute(original_method, entry_operation, exit_operation, args, first_arg, kwargs, param_dict):
    """Execute entry_operation, function, exit_operation"""
    # entry_operation
    _execute_operation(entry_operation, param_dict)
    # original_method
    result: Any = original_method(first_arg, *args, **kwargs)
    # Add result to parameter dictionary
    param_dict["result"] = result
    # exit_operation
    _execute_operation(exit_operation, param_dict)
    return result


async def _async_execute(original_method, entry_operation, exit_operation, args, first_arg, kwargs, param_dict):
    """Execute entry_operation, function, exit_operation"""
    # entry_operation
    _execute_operation(entry_operation, param_dict)
    # original_method
    result: Any = await original_method(first_arg, *args, **kwargs)
    # Add result to parameter dictionary
    param_dict["result"] = result
    # exit_operation
    _execute_operation(exit_operation, param_dict)
    return result


def _set_param_dict(args, first_arg, is_cls, kwargs):
    """Fill in cls or self """
    param_dict: Dict[str, Any] = {
        "args": args,
        "kwargs": kwargs
    }
    if is_cls:
        param_dict["cls"] = first_arg
    else:
        param_dict["self"] = first_arg
    return param_dict


def _is_cls_method(original_method):
    """Determine whether it is a class function. """
    is_cls = False
    if inspect.ismethod(original_method):
        is_cls = isinstance(original_method.__self__, type)
    return is_cls


# Define method wrapper type
SyncWrapper = Callable[..., Any]
AsyncWrapper = Callable[..., Any]


def _sync_func(
        original_method: Callable,
        entry_operation: Optional[str],
        exit_operation: Optional[str],
        scope_name: Optional[str],
        scope_class_name: Optional[str],
        scope_function: Optional[str]
) -> SyncWrapper:
    """Create a Synchronization Method Wrapper """

    @functools.wraps(original_method)
    def wrapper(first_arg: Any, *args: Any, **kwargs: Any) -> Any:
        # Obtain the current call stack
        stack: List[inspect.FrameInfo] = inspect.stack()

        # caller info
        caller_module: str
        caller_class: str
        caller_function: str
        caller_module, caller_class, caller_function = _get_caller_info(stack)

        should_wrap: bool = _should_wrap(
            scope_name, scope_class_name, scope_function,
            caller_module, caller_class, caller_function
        )

        if not should_wrap:
            logger.info(
                f"<<<INFO: Skipping wrapper for {original_method.__qualname__} "
                f"due to scope {caller_module}.{caller_class}.{caller_function} mismatch")
            return original_method(first_arg, *args, **kwargs)

        is_cls = _is_cls_method(original_method)

        param_dict = _set_param_dict(args, first_arg, is_cls, kwargs)

        result = _sync_execute(original_method, entry_operation, exit_operation, args, first_arg, kwargs, param_dict)
        return result

    return cast(SyncWrapper, wrapper)


def _async_func(
        original_method: Callable,
        entry_operation: Optional[str],
        exit_operation: Optional[str],
        scope_name: Optional[str],
        scope_class_name: Optional[str],
        scope_function: Optional[str]
) -> AsyncWrapper:
    """Create an asynchronous method wrapper """

    @functools.wraps(original_method)
    async def async_wrapper(first_arg: Any, *args: Any, **kwargs: Any) -> Any:
        # Obtain the current call stack
        stack: List[inspect.FrameInfo] = inspect.stack()

        # caller info
        caller_module: str
        caller_class: str
        caller_function: str
        caller_module, caller_class, caller_function = _get_caller_info(stack)

        should_wrap: bool = _should_wrap(
            scope_name, scope_class_name, scope_function,
            caller_module, caller_class, caller_function
        )

        if not should_wrap:
            logger.info(
                f"<<<INFO: Skipping wrapper for {original_method.__qualname__} "
                f"due to scope {caller_module}.{caller_class}.{caller_function} mismatch")
            return await original_method(first_arg, *args, **kwargs)

        is_cls = _is_cls_method(original_method)

        param_dict = _set_param_dict(args, first_arg, is_cls, kwargs)

        result = await _async_execute(original_method, entry_operation, exit_operation,
                                      args, first_arg, kwargs, param_dict)
        return result

    return cast(AsyncWrapper, async_wrapper)


def marker_prof_wrapper(
        original_method: Callable,
        params: Dict[str, Any]
) -> Union[SyncWrapper, AsyncWrapper]:
    """marker_prof_wrapper"""
    function_name: Optional[str] = params.get("function_name")
    if function_name is None:
        return original_method

    entry_operation: Optional[str] = params.get("entry_operation")
    exit_operation: Optional[str] = params.get("exit_operation")
    scope_name: Optional[str] = params.get("scope_name")
    scope_class_name: Optional[str] = params.get("scope_class_name")
    scope_function: Optional[str] = params.get("scope_function")

    is_async: bool = inspect.iscoroutinefunction(original_method)

    if is_async:
        logger.info(f"<<<INFO: {original_method.__qualname__} is async function, use async wrapper")
        return _async_func(original_method, entry_operation, exit_operation,
                           scope_name, scope_class_name, scope_function)
    else:
        logger.info(f"<<<INFO: {original_method.__qualname__} is sync function, use sync wrapper")
        return _sync_func(original_method, entry_operation, exit_operation,
                          scope_name, scope_class_name, scope_function)
