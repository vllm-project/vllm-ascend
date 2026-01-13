#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""
Service profiling configuration generator module.

This module generates the service_profiling_symbols.yaml configuration file
to ~/.config/vllm_ascend/ directory.
"""

import tempfile
from pathlib import Path
from typing import Optional

import vllm
from vllm.logger import logger

VLLM_VERSION = vllm.__version__
# Configuration file name
CONFIG_FILENAME = f"service_profiling_symbols.{VLLM_VERSION}.yaml"

# Hard-coded YAML content, default symbols changed by user can be added here.
SERVICE_PROFILING_SYMBOLS_YAML = """
# ===== Batch / Scheduler =====
- symbol: vllm.v1.engine.processor:Processor.process_inputs
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.batch_hookers:process_inputs

- symbol: vllm.v1.core.sched.scheduler:Scheduler.schedule
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.batch_hookers:schedule
  name: batchFrameworkProcessing

- symbol: vllm.v1.core.sched.scheduler:Scheduler._free_request
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.batch_hookers:free_request

- symbol: vllm.v1.core.sched.scheduler:Scheduler.add_request
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.batch_hookers:add_request

# ===== KV Cache =====
- symbol: vllm.v1.core.kv_cache_manager:KVCacheManager.allocate_slots
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.kvcache_hookers:allocate_slots

- symbol: vllm.v1.core.kv_cache_manager:KVCacheManager.free
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.kvcache_hookers:free

- symbol: vllm.v1.core.kv_cache_manager:KVCacheManager.get_computed_blocks
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.kvcache_hookers:get_computed_blocks

# ===== Model Execute =====
- symbol: vllm.model_executor.layers.logits_processor:LogitsProcessor.forward
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.model_hookers:compute_logits
  name: computing_logits

- symbol: vllm.v1.sample.sampler:Sampler.forward
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.model_hookers:sampler_forward
  name: sample

- symbol: vllm.v1.executor.abstract:Executor.execute_model
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.model_hookers:execute_model
  name: modelExec

- symbol: vllm.v1.executor.multiproc_executor:MultiprocExecutor.execute_model
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.model_hookers:execute_model
  name: modelExec

- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model
  name: modelRunnerExec
  domain: ModelExecute

- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._update_states
  name: _update_states
  domain: ModelExecute

- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner._prepare_inputs
  name: _prepare_inputs
  domain: ModelExecute

- symbol: vllm_ascend.utils:ProfileExecuteDuration.capture_async
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.model_hookers:capture_async

# ===== Request Lifecycle =====
- symbol: vllm.v1.engine.async_llm:AsyncLLM.add_request
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.request_hookers:add_request_async

- symbol: vllm.engine.async_llm_engine:AsyncLLMEngine.add_request
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.request_hookers:add_request_async

- symbol: vllm.v1.engine.output_processor:OutputProcessor.process_outputs
  min_version: "0.9.1"
  handler: msserviceprofiler.vllm_profiler.vllm_v1.request_hookers:process_outputs
"""


def get_config_dir() -> Path:
    """
    Get the vllm_ascend configuration directory path.
    
    Returns:
        Path: The path to ~/.config/vllm_ascend/ directory.
    """
    home_dir = Path.home()
    config_dir = home_dir / ".config" / "vllm_ascend"
    return config_dir


def _cleanup_temp_file(tmp_path: Optional[Path]) -> None:
    """
    Clean up a temporary file if it exists.
    
    Args:
        tmp_path: Path to the temporary file to clean up.
    """
    if tmp_path is not None and tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass  # Ignore cleanup errors


def generate_service_profiling_config() -> Optional[Path]:
    """
    Generate the service_profiling_symbols.yaml configuration file
    to ~/.config/vllm_ascend/ directory.
    
    If the configuration file already exists, this function will skip
    creating it and return the existing file path.
    
    If any error occurs during file creation, it will be logged but
    will not interrupt the execution. The function will return None
    to indicate that the file could not be created.
    
    Returns:
        Optional[Path]: The path to the generated (or existing) configuration file.
                       Returns None if file creation failed.
    """
    config_dir = get_config_dir()
    config_file = config_dir / CONFIG_FILENAME

    # Check if the configuration file already exists
    if config_file.exists():
        return config_file

    # Create the configuration directory if it doesn't exist
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(
            f"Failed to create configuration directory {config_dir}: {e}",
            exc_info=True)
        return None

    # Write the configuration file atomically using a temporary file
    # This ensures the file is only written if the write succeeds completely
    tmp_path = None
    try:
        # Create a temporary file in the same directory for atomic write
        with tempfile.NamedTemporaryFile(mode='w',
                                         encoding='utf-8',
                                         dir=config_dir,
                                         delete=False,
                                         suffix='.tmp',
                                         prefix=CONFIG_FILENAME +
                                         '.') as tmp_file:
            tmp_file.write(SERVICE_PROFILING_SYMBOLS_YAML)
            tmp_path = Path(tmp_file.name)

        # Atomically replace the target file with the temporary file
        tmp_path.replace(config_file)
        return config_file
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to write configuration file {config_file}: {e}",
                     exc_info=True)
        return None
    finally:
        # Clean up the temporary file if it wasn't successfully replaced
        _cleanup_temp_file(tmp_path)
