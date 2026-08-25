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
import math
import typing
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.attention.dsa_attn_kv_plan import is_a5_bf16_kv_enabled
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata, Compressor
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    npu_stream_switch,
    vllm_version_is,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata


def hadamard_linear(x: torch.Tensor, hadamard: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    dim_padded = 2 ** math.ceil(math.log2(dim))
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    return F.linear(x, hadamard), x_shape, dim


def hadamard_scale(out: torch.Tensor, x_shape: tuple[int, ...], dim: int, scale: float = 1.0) -> torch.Tensor:
    """Scale and reshape the output of hadamard_linear."""
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    out, x_shape, dim = hadamard_linear(x, hadamard)
    return (out * dim**-0.5)[..., :dim].reshape(*x_shape)


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8DynamicLinearMethod)


class AscendDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__(head_dim, dtype, prefix, cache_config, compress_ratio)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
            self.dtype = torch.float8_e4m3fn
            if not is_a5_bf16_kv_enabled(vllm_config):
                vllm_config.cache_config.cache_dtype = "float8_e4m3fn"

        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        storage_block_size = DSV4_BLOCK_SIZES[vllm_config.cache_config.block_size][0][0]
        # vLLM #51718 replaced MLAAttentionSpec.compress_ratio with
        # AttentionSpec.tokens_per_state on main.
        ratio_kwargs = (
            {"compress_ratio": self.compress_ratio}
            if vllm_version_is("0.27.1")
            else {"tokens_per_state": self.compress_ratio}
        )
        return AscendMLAAttentionSpec(
            block_size=storage_block_size * self.compress_ratio,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            model_version="deepseek_v4",
            cache_dtype_str=self.cache_config.cache_dtype,
            scale_dim=1 if self.head_dim == 128 else 0,
scale_dtype=torch.float
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else torch.float16,
            **ratio_kwargs,
        )

    def forward(self): ...

    def get_attn_backend(self):
        # Keep these imports lazy to avoid a model-inspection circular import.
        if self.compress_ratio == 4:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC4Backend

            return AscendDSAC4Backend
        if self.compress_ratio == 128:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC128Backend

            return AscendDSAC128Backend
        raise ValueError(f"Unsupported DeepSeek V4 indexer compression ratio: {self.compress_ratio}")


@dataclass(frozen=True)
class AscendIndexerMetadata:
    compressor: AscendCompressorMetadata


@dataclass(frozen=True)
class IndexerOverlapPlan:
    """Main-attention compressor work scheduled around Indexer selection."""

    compute_attention_compressed_kv: typing.Callable[[], tuple[torch.Tensor, torch.Tensor]]
    scatter_attention_compressed_kv: typing.Callable[[torch.Tensor, torch.Tensor], None]
    aux_stream: torch.npu.Stream | None = None


class AscendIndexerOps:
    def __init__(self, index_topk: int) -> None:
        from vllm_ascend.device.device_op import DeviceOperator

        self.device_operator = DeviceOperator
        self.index_topk = index_topk

    def unpack_dsa_indexer_kv_cache(self, kv_cache: tuple[torch.Tensor, ...]):
        return self.device_operator.unpack_dsa_indexer_kv_cache(kv_cache)

    def quantize_query(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.device_operator.indexer_quantize_query(query)

    def quantize_key_and_update_cache(
        self,
        key: torch.Tensor,
        key_cache: torch.Tensor,
        full_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
    ):
        return self.device_operator.indexer_quant_scatter_part1(
            key,
            key_cache,
            full_cache,
            slot_mapping,
        )

    def update_scale_cache(
        self,
        key_scale: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self.device_operator.dsa_indexer_scatter_scale_part3(
            key_scale,
            scale_cache,
            slot_mapping,
        )

    def select_topk(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        metadata: typing.Any,
    ) -> torch.Tensor:
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(metadata.qli_metadata))
        topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
            query=query,
            key=key_cache,
            weights=self.device_operator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=self.device_operator.prepare_dsa_indexer_query_scale(query_scale),
            key_dequant_scale=self.device_operator.prepare_dsa_indexer_key_scale(scale_cache),
            actual_seq_lengths_query=metadata.query_start_loc[1:],
            actual_seq_lengths_key=metadata.seq_lens,
            block_table=metadata.block_table,
            metadata=metadata.qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False,
        )
        return topk_idxs

    def quantize_update_cache_and_select_topk(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        weights: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        full_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        metadata: typing.Any,
    ) -> torch.Tensor:
        query, query_scale, _, _ = self.device_operator.indexer_quant_scatter(
            query,
            key,
            key_cache,
            scale_cache,
            full_cache,
            slot_mapping,
        )
        return self.select_topk(
            query,
            weights,
            query_scale,
            key_cache,
            scale_cache,
            metadata,
        )


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int,
        skip_topk: bool,
        use_index_cache: bool,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.skip_topk = skip_topk
        self.use_index_cache = use_index_cache

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )

        self.cv_wq_b = CVLinearWrapper(self.wq_b)
        self.topk_indices_buffer = topk_indices_buffer
        if self.skip_topk and self.topk_indices_buffer is None:
            raise ValueError("skip_topk requires topk_indices_buffer")
        self.ops = AscendIndexerOps(index_topk=self.index_topk)
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        k_dtype = (
            torch.float8_e4m3fn
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else torch.int8
        )

        if self.compress_ratio == 4:
            # TODO(cmq): change the dtype of cache
            self.k_cache = AscendDeepseekV4IndexerCache(
                head_dim=self.head_dim,
                dtype=k_dtype,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                head_dim=self.head_dim,
                rotate=True,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
            )  # Compressor(4, 128)

    @staticmethod
    def _get_indexer_cache_metadata(
        metadata: AscendIndexerMetadata,
    ) -> tuple[typing.Any, torch.Tensor]:
        cache_metadata = metadata.compressor.cache
        cache_req_metadata = cache_metadata.req_metadata
        hadamard = cache_metadata.hadamard
        assert cache_req_metadata is not None
        assert hadamard is not None
        return cache_req_metadata, hadamard

    def update_cache(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
    ) -> None:
        """Update Indexer caches without projecting queries or selecting TopK."""
        if hidden_states.shape[0] == 0:
            return

        state_cache, key_cache, scale_cache, full_cache = self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        _, hadamard = self._get_indexer_cache_metadata(metadata)
        compressor = self.compressor
        assert compressor is not None
        key, slot_mapping = compressor(
            hidden_states=hidden_states,
            state_cache=state_cache,
            metadata=metadata.compressor,
        )
        if key.shape[0] == 0:
            return
        if compressor.rotate:
            key = rotate_activation(key, hadamard)
        _, key_scale = self.ops.quantize_key_and_update_cache(
            key,
            key_cache,
            full_cache,
            slot_mapping,
        )
        if key_scale is not None:
            self.ops.update_scale_cache(
                key_scale,
                scale_cache,
                slot_mapping,
            )

    def _get_cached_topk_indices(self, num_tokens: int, offset: int = 0) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("topk_indices_buffer is required to read cached TopK indices")
        topk_indices = self.topk_indices_buffer[offset : offset + num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_cached_topk_indices(self, topk_indices: torch.Tensor, offset: int = 0) -> None:
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        topk_indices_buffer = self.topk_indices_buffer[offset : offset + num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            if topk_indices_to_cache.shape[1] != 1:
                raise ValueError("TopK indices must have a singleton head dimension")
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        topk_indices_buffer.copy_(topk_indices_to_cache)

    def forward(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        overlap_plan: IndexerOverlapPlan,
        *,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        cache_metadata, _ = self._get_indexer_cache_metadata(metadata)
        cos = cache_metadata.cos[layer_name][:num_tokens]
        sin = cache_metadata.sin[layer_name][:num_tokens]
        aux_stream = overlap_plan.aux_stream
        if self.skip_topk:
            topk_indices = self._get_cached_topk_indices(num_tokens)
        elif aux_stream is not None:
            indexer_q = self._cv_compute_query_and_update_cache_multistream(
                hidden_states,
                qr,
                kv_cache,
                metadata,
                cos,
                sin,
                aux_stream,
                qr_pertoken_scale,
            )
            compressed_kv, compress_slot_mapping = overlap_plan.compute_attention_compressed_kv()
            topk_indices = self._select_topk_multistream(
                hidden_states,
                indexer_q,
                kv_cache,
                metadata,
                aux_stream,
                lambda: overlap_plan.scatter_attention_compressed_kv(
                    compressed_kv,
                    compress_slot_mapping,
                ),
            )
        else:
            topk_indices = self._select_topk_serial(
                hidden_states,
                qr,
                kv_cache,
                metadata,
                cos,
                sin,
                qr_pertoken_scale,
                write_cache=write_cache,
            )

        if write_cache and (self.skip_topk or aux_stream is None):
            compressed_kv, compress_slot_mapping = overlap_plan.compute_attention_compressed_kv()
            overlap_plan.scatter_attention_compressed_kv(compressed_kv, compress_slot_mapping)

        if self.use_index_cache:
            self._update_cached_topk_indices(topk_indices)
        return topk_indices

    def _cv_compute_query_and_update_cache_multistream(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        aux_stream: torch.npu.Stream,
        qr_pertoken_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Indexer query and update its cache.

        The internal multistream strategy keeps the original four-part layout:
        - Part0: Main pre-compute qr_quant[V] + compressor[C/mixed] + kv_hadamard[V]
        - Part1: Main matmul[C] ∥ Aux kv_quant[V] + scatter_k_cache[AIV]
        - Part2: Main rope[V] (serial)
        - Part3: Main q_hadamard[C] ∥ Aux scatter_scale_cache[AIV]
        """
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        _, hadamard = self._get_indexer_cache_metadata(metadata)
        main_stream = torch.npu.current_stream()
        compressor = self.compressor
        assert compressor is not None

        # ===== Part0: Pre-compute on main =====
        if _is_w8a8_dynamic(self.wq_b) and qr_pertoken_scale is not None:
            qr_quant_ready = qr
            qr_scale_ready = qr_pertoken_scale
        else:
            qr_quant_ready, qr_scale_ready = self.cv_wq_b.quantize(qr)

        kv, slot_mapping_indexer = compressor(
            hidden_states=hidden_states,
            state_cache=indexer_state_cache,
            metadata=metadata.compressor,
        )
        if kv.numel() == 0:
            kv = None
        elif compressor.rotate:
            kv = rotate_activation(kv, hadamard)

        # ===== Part1: matmul[C] ∥ kv_quant[V] + scatter_k_cache[AIV] =====
        # Record event before main stream operations for aux_stream to wait
        e_kv_ready = main_stream.record_event()

        # Aux: kv_quant + scatter_k_cache (parallel with main matmul + rope)
        if kv is not None:
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_kv_ready)
                kv, kv_scale = self.ops.quantize_key_and_update_cache(
                    kv,
                    indexer_k_cache,
                    indexer_full_cache,
                    slot_mapping_indexer,
                )

        # Main: matmul q from qr (directly submit, V/C different engines dispatch naturally)
        if _is_w8a8_dynamic(self.wq_b) and qr_pertoken_scale is not None:
            q = torch_npu.npu_quant_matmul(
                qr_quant_ready,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_scale_ready,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            q = self.cv_wq_b.matmul(qr_quant_ready, qr_scale_ready)  # qr_matmul

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        q = q.view(-1, self.n_heads, self.head_dim)

        # ===== Part2: rope[V] (main only) =====
        torch.ops._C_ascend.inplace_partial_rotary_mul(  # rope
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        )

        # Wait for aux_stream kv_scatter to complete before proceeding
        if kv is not None:
            main_stream.wait_stream(aux_stream)

        e_rope_done = main_stream.record_event()

        # ===== Part3: q_hadamard[C] ∥ scatter_scale_cache[AIV] =====
        # Note: On A5, indexer_compress_epilog_v2 in Part1 handles both k_cache
        # and scale_cache in one fused operation, so Part3 is skipped
        # (kv_scale is None on A5 from indexer_quant_scatter_part1).
        if kv is not None and kv_scale is not None:
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_rope_done)
                self.ops.update_scale_cache(
                    kv_scale,
                    indexer_scale_cache,
                    slot_mapping_indexer,
                )

        # Main: q_hadamard[Part1 - linear] (directly submit, C/AIV different engines dispatch naturally)
        # Part1: F.linear - parallel with aux_stream kv_scatter
        hidden_size = q.size(-1)
        q_linear, q_shape, q_dim = hadamard_linear(q, hadamard)

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        # Main: q_hadamard[Part2 - scale] (after aux_stream completes)
        # Part2: scale * reshape - dot multiplication
        q = hadamard_scale(q_linear, q_shape, q_dim, scale=hidden_size**-0.5)

        return q

    def _select_topk_multistream(
        self,
        hidden_states: torch.Tensor,
        indexer_q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        aux_stream: torch.npu.Stream,
        scatter_attention_compressed_kv: typing.Callable[[], None],
    ) -> torch.Tensor:
        """Overlap Indexer selection inputs with caller-provided main-stream work."""
        main_stream = torch.npu.current_stream()
        weights_proj_start = main_stream.record_event()
        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(weights_proj_start)
            weights_proj_output = self.weights_proj(hidden_states)
            weights_proj_done = torch.npu.current_stream().record_event()

        q_quant, q_scale = self.ops.quantize_query(indexer_q)
        # Enqueue only independent Vector/AIV work on the current main stream;
        # do not switch streams or launch Cube work that would contend with the
        # auxiliary weights projection.
        scatter_attention_compressed_kv()
        main_stream.wait_event(weights_proj_done)

        (_, indexer_k_cache, indexer_scale_cache, _) = self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        cache_metadata, _ = self._get_indexer_cache_metadata(metadata)
        weights = weights_proj_output * (self.softmax_scale * self.n_heads**-0.5)
        return self.ops.select_topk(
            q_quant,
            weights,
            q_scale,
            indexer_k_cache,
            indexer_scale_cache,
            cache_metadata,
        )

    def _indexer_qkv_prepare(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ):
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        cache_metadata, hadamard = self._get_indexer_cache_metadata(metadata)
        compressor = self.compressor
        assert compressor is not None

        if (
            _is_w8a8_dynamic(self.wq_b)
            and qr_pertoken_scale is not None
            and not get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
        ):
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)  # [T, N, D]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        )

        q = rotate_activation(q, hadamard)
        kv = None
        indexer_slot_mapping = None
        if write_cache:
            kv, indexer_slot_mapping = compressor(
                hidden_states=x,
                state_cache=indexer_state_cache,
                metadata=metadata.compressor,
            )
            if kv.numel() == 0:
                kv = None
            elif compressor.rotate:
                kv = rotate_activation(kv, hadamard)

        return (
            q,
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            cache_metadata,
            indexer_slot_mapping,
        )

    def _select_topk_serial(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ):
        q, kv, ik, isc, ifc, cache_metadata, indexer_slot_mapping = self._indexer_qkv_prepare(
            x,
            qr,
            kv_cache,
            metadata,
            cos,
            sin,
            qr_pertoken_scale,
            write_cache=write_cache,
        )

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

        if write_cache:
            return self.ops.quantize_update_cache_and_select_topk(
                q,
                kv,
                weights,
                ik,
                isc,
                ifc,
                indexer_slot_mapping,
                cache_metadata,
            )

        q, q_scale = self.ops.quantize_query(q)
        return self.ops.select_topk(
            q,
            weights,
            q_scale,
            ik,
            isc,
            cache_metadata,
        )
