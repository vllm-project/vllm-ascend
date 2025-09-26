# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import os
from pathlib import Path
import logging
import yaml
from .utils import safe_print, ip_str, trace_output_directory
from .prof_wrapper import marker_prof_wrapper
import time
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wrapper_dict = {
    "marker": marker_prof_wrapper
}


# Parse config from namelist, apply profiler monkey patch
def apply_patches(namelist_path: str):
    try:
        namelist_file = Path(__file__).parent / namelist_path

        # Load namelist
        with namelist_file.open('r') as f:
            config = yaml.safe_load(f)

        profiler_type = config.get('type')
        if not (profiler_type == 'torchnpu' or
                profiler_type == 'timer' or
                profiler_type == 'viztracer' or
                profiler_type == 'marker'):
            logger.error(f"<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
            raise RuntimeError("<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
        logger.info(f"<<<Applying {profiler_type} profiler patches from {namelist_path}")
        wrapper_method = wrapper_dict[profiler_type]

        base_params = config.get("base_params", {})

        # Extract target modules and methods
        targets: List[Tuple[str, Optional[str]]] = []
        for target in config.get('targets', []):
            module_name = target.get('module')
            class_name = None
            if ":" in module_name:
                module_name, class_name = module_name.split(":")
            scope_name = target.get('scope')
            scope_class_name = None
            if scope_name and ":" in scope_name:
                scope_name, scope_class_name = scope_name.split(":")
            scope_function = target.get('scope_function', None)
            function_name = target.get('function_name')
            entry_operation = target.get('entry_operation', None)
            exit_operation = target.get('exit_operation', None)
            entry_message = target.get('entry_message', None)
            exit_message = target.get('exit_message', None)
            if module_name:
                targets.append(
                    (
                        module_name,
                        class_name,
                        function_name,
                        (entry_operation, exit_operation),
                        (entry_message, exit_message),
                        scope_name,
                        scope_class_name,
                        scope_function
                    )
                )
            else:
                logger.warning(f"<<<Skipping target with missing 'module': {target}")

        if not targets:
            logger.warning(f"<<<No valid targets found in {namelist_path}")
            return

        for module_name, class_name, function_name, \
                (entry_operation, exit_operation), \
                (entry_message, exit_message), scope_name, scope_class_name, scope_function in targets:
            logger.info(f"<<<Patching {module_name}.{function_name or 'all methods'}")
            try:
                original_module = importlib.import_module(module_name)

                base_params['entry_operation'] = entry_operation
                base_params['exit_operation'] = exit_operation
                base_params['entry_message'] = entry_message
                base_params['exit_message'] = exit_message
                base_params['function_name'] = function_name
                base_params['scope_name'] = scope_name
                base_params['scope_class_name'] = scope_class_name
                base_params['scope_function'] = scope_function
                if class_name:
                    try:
                        target_class = getattr(original_module, class_name)
                        try:
                            original_function = getattr(target_class, function_name)
                            wrapped_function = wrapper_method(original_function, base_params)
                            setattr(target_class, function_name, wrapped_function)
                            logger.info(f"<<<<{module_name}.{class_name}.{function_name} is wrapped")
                        except AttributeError:
                            logger.warning(
                                f"<<<Function '{function_name}' not found in class '{class_name}' "
                                f"of module '{module_name}'"
                            )
                            continue
                    except AttributeError:
                        logger.warning(f"<<<Class '{class_name}' not found in module '{module_name}'")
                        continue
                else:
                    try:
                        original_function = getattr(original_module, function_name)
                        wrapped_function = wrapper_method(original_function, base_params)
                        setattr(original_module, function_name, wrapped_function)
                        logger.info(f"<<<<{module_name}.{function_name} is wrapped")
                    except AttributeError:
                        logger.warning(f"<<<Function '{function_name}' not found in module '{module_name}'")
                        continue
            except ImportError as e:
                logger.warning(f"<<<Failed to import module '{module_name}': {str(e)}")
                continue
            except Exception as e:
                logger.warning(
                    f"<<<Unexpected error while wrapping {module_name}.{class_name or ''}."
                    f"{function_name}: {str(e)}"
                )
                continue

    except (FileNotFoundError, ImportError, AttributeError, RuntimeError, yaml.YAMLError) as e:
        logger.error(f"<<<Failed to apply model patches: {e}")
        raise


def monkey_patch_prefill_pop_bootstrapped_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue

    original_method = PrefillBootstrapQueue.pop_bootstrapped

    @wraps(original_method)
    def new_method(self, *args, **kwargs) -> AsyncGenerator:
        result = original_method(self, *args, **kwargs)

        if isinstance(result, tuple):
            bootstrapped_reqs, failed_reqs = result
        else:
            bootstrapped_reqs = result
        for req in bootstrapped_reqs:
            raw_request_id = req.bootstrap_room
            safe_print(trace_output_directory,
                       f"<<<Action: Prefill add waiting queue; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")

        return result

    PrefillBootstrapQueue.pop_bootstrapped = new_method
    print("<<< Monkey patch monkey_patch_prefill_pop_bootstrapped_logger is applied")


def monkey_patch_async_handle_request_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase

    original_method = OpenAIServingBase.handle_request

    @wraps(original_method)
    async def new_method(self, *args, **kwargs) -> AsyncGenerator:
        raw_request_id = args[0].bootstrap_room

        result = await original_method(self, *args, **kwargs)

        if os.getenv('ROLE') == "prefill":
            safe_print(trace_output_directory,
                       f"<<<Action: PD api server get request; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            safe_print(trace_output_directory,
                       f"<<<Action: Get prefill engine request and start pickle; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            safe_print(trace_output_directory,
                       f"<<<Action: Start process request in prefill engine; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
        else:
            safe_print(trace_output_directory,
                       f"<<<Action: Enter decode to generate; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            safe_print(trace_output_directory,
                       f"<<<Action: Start to dispatch decode request; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")

        return result

    OpenAIServingBase.handle_request = new_method
    print("<<< Monkey patch monkey_patch_async_handle_request_logger is applied")


def monkey_patch_async_wait_one_response_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    original_method = TokenizerManager._wait_one_response

    @wraps(original_method)
    async def new_method(self, *args, **kwargs) -> AsyncGenerator:
        yield_count = 0
        raw_request_id = args[0].bootstrap_room  # get request_id
        async for item in original_method(self, *args, **kwargs):
            if os.getenv('ROLE') == "prefill":
                safe_print(trace_output_directory,
                           f"<<<Action: Start to send output in prefill stage; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
                safe_print(trace_output_directory,
                           f"<<<Action: Client get prefill output; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
                safe_print(trace_output_directory,
                           f"<<<Action: Pop output queues; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
                safe_print(trace_output_directory,
                           f"<<<Action: Finish prefill pickle and start response; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            yield_count += 1
            if yield_count == 1 and os.getenv('ROLE') != "prefill":
                safe_print(trace_output_directory,
                           f"<<<Action: First decode output token; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            elif yield_count == 2 and os.getenv('ROLE') != "prefill":
                safe_print(trace_output_directory,
                           f"<<<Action: Second decode output token; Timestamp:{time.time()}; RequestID:{raw_request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            yield item

    TokenizerManager._wait_one_response = new_method
    print("<<< Monkey patch monkey_patch_async_wait_one_response_logger is applied")


profiling_namelist = os.getenv("PROFILING_NAMELIST", None)
if profiling_namelist is not None:
    if os.path.isfile(profiling_namelist):
        apply_patches(profiling_namelist)
        monkey_patch_async_wait_one_response_logger()
        monkey_patch_async_handle_request_logger()
        monkey_patch_prefill_pop_bootstrapped_logger()
    else:
        logger.error(f"'{profiling_namelist}' does not exist.")
        raise FileNotFoundError(f"'{profiling_namelist}' does not exist.")