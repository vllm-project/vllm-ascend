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

"""
Configuration management utilities for vLLM Ascend.

This module provides functionality for:
- Managing block sizes
- Validating KV transfer configurations
- Version checking
- Singleton pattern decorator
"""

import functools
from packaging.version import InvalidVersion, Version
from typing import TYPE_CHECKING

from vllm_ascend import envs_ascend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


@functools.cache
def vllm_version_is(target_vllm_version: str):
    if envs_ascend.VLLM_VERSION is not None:
        vllm_version = envs_ascend.VLLM_VERSION
    else:
        import vllm
        vllm_version = vllm.__version__
    try:
        return Version(vllm_version) == Version(target_vllm_version)
    except InvalidVersion:
        raise ValueError(
            f"Invalid vllm version {vllm_version} found. A dev version of vllm "
            "is installed probably. Set the environment variable VLLM_VERSION "
            "to control it by hand. And please make sure the value follows the "
            "format of x.y.z.")


def refresh_block_size(vllm_config):
    """
    Refresh the block size in cache config.
    """
    from vllm.logger import logger

    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    if cache_config.block_size is None:
        cache_config.block_size = 128

    if not scheduler_config or not model_config:
        return

    # TODO(MengqingCao): Remove the model_type check, after resolving the hidden error in get_kv_cache_groups.
    if not model_config.hf_text_config.model_type == "qwen3_next" and cache_config.block_size != 128:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info(
                "Block size is set to 128 if prefix cache or chunked prefill is enabled."
            )
            cache_config.block_size = 128


def check_kv_extra_config(vllm_config):

    def _check(name: str, config: dict):
        tp_key = "tp_size"
        dp_key = "dp_size"
        if tp_key in config:
            config_tp = config[tp_key]
            vllm_tp = vllm_config.parallel_config.tensor_parallel_size
            if config_tp != vllm_tp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting tensor parallel size. "
                    f"Expected {vllm_tp}, but got {config_tp}.")
        if dp_key in config:
            config_dp = config[dp_key]
            vllm_dp = vllm_config.parallel_config.data_parallel_size
            if config_dp != vllm_dp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting data parallel size. "
                    f"Expected {vllm_dp}, but got {config_dp}.")

    if vllm_config.kv_transfer_config.is_kv_producer:
        _check(
            "prefill",
            vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {}))
    if vllm_config.kv_transfer_config.is_kv_consumer:
        _check(
            "decode",
            vllm_config.kv_transfer_config.get_from_extra_config("decode", {}))


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
