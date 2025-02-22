#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/vllm/attention/layer.py
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
# This file is used to monkey patch vLLM Attention.__init__ function
# and move the instantiation of num_heads, head_size, num_kv_heads
# ahead of the initialization of attention quant methods, which is
# required by ascend attention quant method to initialize.
# Remove this file when vllm support it. Cuda-related codes (although
# it's not related to ascend npu) are still maintained to be compatible
# with vLLM original codes.

from typing import Any, Dict, List, Optional

import torch

import vllm.envs as envs
from vllm.attention import Attention, AttentionType
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import current_platform


def attention_init(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: Optional[int] = None,
    alibi_slopes: Optional[List[float]] = None,
    cache_config: Optional[CacheConfig] = None,
    quant_config: Optional[QuantizationConfig] = None,
    blocksparse_params: Optional[Dict[str, Any]] = None,
    logits_soft_cap: Optional[float] = None,
    per_layer_sliding_window: Optional[int] = None,
    use_mla: bool = False,
    prefix: str = "",
    attn_type: str = AttentionType.DECODER,
    **extra_impl_args,
) -> None:
    super(Attention, self).__init__()
    if per_layer_sliding_window is not None:
        # per-layer sliding window
        sliding_window = per_layer_sliding_window
    elif cache_config is not None:
        # model-level sliding window
        sliding_window = cache_config.sliding_window
    else:
        sliding_window = None

    if cache_config is not None:
        kv_cache_dtype = cache_config.cache_dtype
        block_size = cache_config.block_size
        is_attention_free = cache_config.is_attention_free
        calculate_kv_scales = cache_config.calculate_kv_scales
    else:
        kv_cache_dtype = "auto"
        block_size = 16
        is_attention_free = False
        calculate_kv_scales = False
    if num_kv_heads is None:
        num_kv_heads = num_heads

    self.kv_cache_dtype = kv_cache_dtype
    self.calculate_kv_scales = calculate_kv_scales
    self._k_scale = torch.tensor(1.0, dtype=torch.float32)
    self._v_scale = torch.tensor(1.0, dtype=torch.float32)

    # We also keep the float32 versions of k/v_scale for attention
    # backends that don't support tensors (Flashinfer)
    self._k_scale_float = 1.0
    self._v_scale_float = 1.0

    # should move following three lines before quant method is instantiated.
    self.num_heads = num_heads
    self.head_size = head_size
    self.num_kv_heads = num_kv_heads

    quant_method = quant_config.get_quant_method(
        self, prefix=prefix) if quant_config else None
    if quant_method is not None:
        assert isinstance(quant_method, BaseKVCacheMethod)
        if self.kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with "
                             "fp8 checkpoints.")
        self.quant_method = quant_method
        self.quant_method.create_weights(self)

    # During model initialization, the default dtype is set as the model
    # weight and activation dtype.
    dtype = torch.get_default_dtype()
    attn_backend = get_attn_backend(head_size,
                                    dtype,
                                    kv_cache_dtype,
                                    block_size,
                                    is_attention_free,
                                    blocksparse_params is not None,
                                    use_mla=use_mla)
    impl_cls = attn_backend.get_impl_cls()
    self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **extra_impl_args)
    self.sliding_window = sliding_window
    self.backend = backend_name_to_enum(attn_backend.get_name())
    self.dtype = dtype

    self.use_direct_call = not current_platform.is_cuda_alike(
    ) and not current_platform.is_cpu()

    self.use_output = attn_backend.accept_output_buffer
    compilation_config = get_current_vllm_config().compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = self
    self.layer_name = prefix
    self.attn_type = attn_type
    self.kv_cache = [
        torch.tensor([]) for _ in range(
            get_current_vllm_config().parallel_config.pipeline_parallel_size)
    ]

    self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
    self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)


Attention.__init__ = attention_init