# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .prof_wrapper import (marker_prof_wrapper, timer_prof_wrapper,
                           torchnpu_prof_wrapper, viztracer_prof_wrapper)
from .utils import ip_str, safe_print, trace_output_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wrapper_dict = {
    "torchnpu": torchnpu_prof_wrapper, 
    "timer": timer_prof_wrapper, 
    "viztracer": viztracer_prof_wrapper, 
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
        if not (profiler_type=='torchnpu' or 
                profiler_type=='timer' or 
                profiler_type=='viztracer' or
                profiler_type=='marker'):
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
                        (entry_message, exit_message)
                    )
                )            
            else:
                logger.warning(f"<<<Skipping target with missing 'module': {target}")

        if not targets:
            logger.warning(f"<<<No valid targets found in {namelist_path}")
            return

        for module_name, class_name, function_name, \
                (entry_operation, exit_operation), \
                (entry_message, exit_message) in targets:
            logger.info(f"<<<Patching {module_name}.{function_name or 'all methods'}")
            try:
                original_module = importlib.import_module(module_name)

                base_params['entry_operation'] = entry_operation
                base_params['exit_operation'] = exit_operation
                base_params['entry_message'] = entry_message
                base_params['exit_message'] = exit_message
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

# following monkey patch is for triggering a printing message 
#   when a request enter WAITING_FOR_REMOTE_KVS status
def monkey_patch_request_status():
    from vllm.v1.request import Request, RequestStatus
    original_status = Request.__dict__.get('status', None)
    def status_getter(self):
        return self._status

    def status_setter(self, value):
        self._status = value
        self.waiting_pull_len += 1
        if value == RequestStatus.WAITING_FOR_REMOTE_KVS:
            safe_print(
                trace_output_directory, 
                f"<<<Action: Add need pullling sequence;|waiting_pull_len={self.waiting_pull_len} "
                f"Timestamp:{time.time()}; "
                f"RequestID:{self.request_id}; "
                f"Role:{os.getenv('ROLE')}_{ip_str}"
            )

    Request.status = property(status_getter, status_setter)
    original_init = Request.__init__
    def new_init(self, *args, **kwargs):
        self.waiting_pull_len = 0
        original_init(self, *args, **kwargs)
        self._status = self.status

    Request.__init__ = new_init
    print("<<< Monkey patch request status is applied")

# following monkey patch is for marking the first token, second token and last token
def monkey_patch_async_generator_io_logger():
    from functools import wraps
    from typing import AsyncGenerator

    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    original_method = OpenAIServingChat.chat_completion_stream_generator
    @wraps(original_method)
    async def new_method(self, *args, **kwargs) -> AsyncGenerator:
        yield_count = 0
        request_id = args[2] # get request_id
        async for item in original_method(self, *args, **kwargs):
            yield_count += 1
            if yield_count == 1:
                # First chat_completion_stream_generator yield.
                pass
            elif yield_count == 2:
                # Second chat_completion_stream_generator yield.
                safe_print(trace_output_directory, f"<<<Action: First decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            elif yield_count == 3:
                # Third chat_completion_stream_generator yield.
                safe_print(trace_output_directory, f"<<<Action: Second decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            if item == "data: [DONE]\n\n":
                safe_print(trace_output_directory, f"<<<Action: Finish decode pickle and start response; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            yield item

    OpenAIServingChat.chat_completion_stream_generator = new_method
    print("<<< Monkey patch monkey_patch_async_generator_io_logger is applied")

# following monkey patch is for passing raw request_id inside _preprocess_chat function
def patch_chatcompletionrequest():
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    OriginalChatCompletionRequest = ChatCompletionRequest
    class PatchedChatCompletionRequest(OriginalChatCompletionRequest):
        raw_request_id: Optional[str] = None
    ChatCompletionRequest = PatchedChatCompletionRequest
    print("<<< Monkey patch patch_chatcompletionrequest is applied")

profiling_namelist = os.getenv("PROFILING_NAMELIST", None)
if os.path.isfile(profiling_namelist):
    apply_patches(profiling_namelist)
    monkey_patch_request_status()
    patch_chatcompletionrequest()
    monkey_patch_async_generator_io_logger()
else:
    logger.error(f"'{profiling_namelist}' does not exist.")
    raise FileNotFoundError(f"'{profiling_namelist}' does not exist.")