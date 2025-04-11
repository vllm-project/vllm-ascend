#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/worker.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import torch
import torch_npu
from contextlib import contextmanager

from vllm.logger import init_logger

logger = init_logger(__name__)
VLLM_ENABLE_GRAPH_MODE = os.environ.get('VLLM_ENABLE_GRAPH_MODE', '0')


def try_register_lib(lib_name: str, lib_info: str = ""):
    import importlib
    import importlib.util
    try:
        module_spec = importlib.util.find_spec(lib_name)
        if module_spec is not None:
            importlib.import_module(lib_name)
            if lib_info:
                logger.info(lib_info)
    except Exception:
        pass

@contextmanager
def profiling_this(encoder_count, is_prefill):
    profiling_path = os.environ.get("VLLM_ENABLE_PROFILING_PATH", "/tmp")
    torch.npu.synchronize()
    start_time = time.time()
    if int(os.environ.get("VLLM_ENABLE_PROFILING", "0")) == 0:
        yield
    else:
        if encoder_count == 50:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                data_simplification=False
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,
                record_shapes=True,
                profile_memory=True,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_path),
                experimental_config=experimental_config
            ):
            
                yield  # 在此处执行目标代码行
        else:
            yield
    torch.npu.synchronize()
    end_time = time.time()
    run_time = end_time - start_time
    if is_prefill:
        model_name = "perfill"
    else:
        model_name = "decoder"
    print(f"{model_name} inference time cost is: {run_time * 1000:.2f} ms")