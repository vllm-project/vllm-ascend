# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/cache_engine.py
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

import torch

from vllm.worker.cache_engine import CacheEngine
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, get_dtype_size)

logger = init_logger(__name__)


class NPUCacheEngine(CacheEngine):
    """Manages the KV cache.

    This class is responsible for initializing and managing the NPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        CacheEngine.__init__(self,
                             cache_config = cache_config,
                             model_config = model_config,
                             parallel_config = parallel_config,
                             device_config = device_config)
        self.dtype = _get_cache_dtype(cache_config, model_config)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        dtype = _get_cache_dtype(cache_config, model_config)

        key_cache_entry = num_heads * head_size

        # For MLA there is no value cache, since the latent vector
        # is joint keys and values.
        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_attention_layers * cache_config.block_size * \
            (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


def _get_cache_dtype(cache_config: CacheConfig,
                     model_config: ModelConfig,
    ) -> torch.dtype:
    if cache_config.cache_dtype == "auto":
        return model_config.dtype
    if cache_config.cache_dtype == "int8":
        return torch.int8
    else:
        return STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
