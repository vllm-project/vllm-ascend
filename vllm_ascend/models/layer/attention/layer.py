# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""

from typing import cast

import torch
import torch.nn as nn
import vllm.envs as envs
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention.attention import _init_kv_cache_quant
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

# from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
)


def get_dsv4_block_sizes():
    # cache_config.block_size: [mla, swa, c4 state, c128 state], [page_size_padded_t1, page_size_padded_t2]
    _DSV4_BLOCK_SIZES = {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64: [[64, 64, 4, 16], [8320, 65536]],
        32: [[32, 32, 2, 8], [4160, 32768]],
    }
    _DSV4_BLOCK_SIZES_A5 = {
        128: [[128, 128, 8, 16], [16896, 81920]],
        64: [[64, 64, 4, 8], [8448, 40960]],
        32: [[32, 32, 2, 4], [4224, 20480]],
    }
    if get_ascend_device_type() in {AscendDeviceType.A5}:
        return _DSV4_BLOCK_SIZES_A5
    else:
        return _DSV4_BLOCK_SIZES


DSV4_BLOCK_SIZES = get_dsv4_block_sizes()


class DSAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

    This class takes query, and compressed key/value tensors as input.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **extra_impl_args,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = scale
        self.n_local_heads = n_local_heads
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = nope_head_dim
        self.n_groups = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.layer_name = prefix
        self.head_size = self.head_dim
        self.swa_cache_layer: DeepseekV4SWACache = extra_impl_args.get("swa_cache_layer")

        assert self.swa_cache_layer is not None

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(self, quant_config, prefix)

        self.attn_backend = AscendDSABackend

        # NOTE(zxr): vllm_is_batch_invariant is delete during updating to v0.20.1
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and (self.attn_backend.get_name() == "TRITON_MLA" or self.attn_backend.get_name() == "FLASHINFER")
        ):
            cache_config.enable_prefix_caching = False

        impl_cls = cast(type[DSAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            **extra_impl_args,
        )

        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)
        ]
        self.kv_cache_dtype = kv_cache_dtype

        self.use_sparse = True

        # Initialize q/k/v range constants.
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        return q

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(act_dtype)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if self.compress_ratio <= 1:  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(self.kv_cache_dtype, vllm_config.model_config)
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            kv_cache_dtype = torch.float8_e4m3fn
            vllm_config.cache_config.cache_dtype = "float8_e4m3fn"

        cached_head_size = (
            (self.head_size + 128) if get_ascend_device_type() in {AscendDeviceType.A5} else self.head_size
        )
        return MLAAttentionSpec(
            block_size=DSV4_BLOCK_SIZES[vllm_config.cache_config.block_size][0][0],
            num_kv_heads=1,
            head_size=cached_head_size,
            dtype=kv_cache_dtype,
            model_version="deepseek_v4",
            compress_ratio=self.compress_ratio,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )
