# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import typing
from dataclasses import dataclass

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.compressor import CompressorStateCache
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class AscendCompressorStateCache(CompressorStateCache):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        block_size: int,
        prefix: str,
    ):
        super().__init__(state_dim, dtype, compress_ratio, prefix)
        self.compress_ratio = compress_ratio
        self.block_size = block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        from vllm_ascend.models.layer.attention.layer import dsv4_block_sizes

        pads = dsv4_block_sizes(vllm_config)[vllm_config.cache_config.block_size][1]
        page_size_padded = pads[0] if self.state_dim == 2 * 256 and self.compress_ratio == 4 else pads[1]

        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
            page_size_padded=page_size_padded,
        )

    def forward(self): ...

    def get_attn_backend(self):
        # Keep these imports lazy to avoid a model-inspection circular import.
        if self.compress_ratio == 4:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC4StateBackend

            return AscendDSAC4StateBackend
        if self.compress_ratio == 128:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC128StateBackend

            return AscendDSAC128StateBackend
        raise ValueError(f"Unsupported DeepSeek V4 state-cache compression ratio: {self.compress_ratio}")


@dataclass(frozen=True)
class AscendCompressorMetadata:
    """Request metadata for the compressed KV and compressor state caches."""

    cache: typing.Any
    state: typing.Any


class Compressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
        *,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        self.vllm_config = vllm_config
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = config.rms_norm_eps
        self.coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32))
        self.wkv = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None if get_ascend_device_type() in {AscendDeviceType.A5} else quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.wgate = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None if get_ascend_device_type() in {AscendDeviceType.A5} else quant_config,
            prefix=f"{prefix}.wgate",
            return_bias=False,
        )

        # Compressor kernel only accepts FP32 norm_weight.
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=torch.float32)

        state_dtype = torch.float32
        # TODO(zyj): change following codes if block_size is configurable & refactor the magic numbers
        if compress_ratio == 4:
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][2],
            )
        elif compress_ratio == 128:
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][3],
            )
        else:
            raise ValueError(
                f"Only support compress_ratio in [4, 128]. Got unsupported compress_ratio: {compress_ratio}"
            )

    def _compute_metadata(
        self,
        metadata: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan

        assert metadata.full_compress_cos is not None
        assert metadata.full_compress_sin is not None
        assert metadata.num_compressed_tokens is not None
        assert metadata.start_pos is not None
        assert metadata.num_actual_reqs is not None
        full_compress_cos = metadata.full_compress_cos.view(
            metadata.full_compress_cos.shape[0],
            metadata.full_compress_cos.shape[-1],
        )
        full_compress_sin = metadata.full_compress_sin.view(
            metadata.full_compress_sin.shape[0],
            metadata.full_compress_sin.shape[-1],
        )
        return torch.ops._C_ascend.compressor_metadata(
            full_compress_cos,
            full_compress_sin,
            metadata.query_start_loc,
            metadata.start_pos,
            metadata.block_table,
            metadata.storage_block_size,
            get_dsa_attn_kv_plan(self.vllm_config).get_dsa_compressor_slot_mapping_format(),
            self.compress_ratio,
            metadata.num_compressed_tokens,
            metadata.num_actual_reqs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_cache: torch.Tensor,
        metadata: AscendCompressorMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressor_metadata = metadata.cache.req_metadata
        state_metadata = metadata.state.req_metadata
        assert compressor_metadata is not None
        assert state_metadata is not None
        compress_cos, compress_sin, slot_mapping = self._compute_metadata(compressor_metadata)
        compressed_kv = torch.ops._C_ascend.compressor(
            hidden_states,
            self.wkv.weight,
            self.wgate.weight,
            state_cache.squeeze(-2),
            self.ape,
            self.norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table=state_metadata.block_table,
            cu_seqlens=compressor_metadata.query_start_loc,
            seqused=None,
            start_pos=compressor_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=2 if self.overlap else 1,
            norm_eps=self.norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        return compressed_kv, slot_mapping
