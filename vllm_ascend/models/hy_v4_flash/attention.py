# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend MLA and IndexPool with cache pages owned by the vLLM scheduler."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import torch
import torch_npu
from vllm_ascend.device.device_config import is_950
from torch import nn
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import make_attention_sink
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionBackend, CacheOnlyAttentionImpl
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION
from vllm.models.hy_v4_flash.reference import (
    INDEX_DIM,
    indexpool_fp8,
    mla_attention_probs,
    pooled_keys,
    sparse_candidates,
    window_candidates,
)
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec, MLAAttentionSpec, SlidingWindowMLASpec

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows, pack_c8_components
from vllm_ascend.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm_ascend.models.hy_v4_flash.cache_layout import PACKED_CACHE_FORMAT
from vllm_ascend.quantization.methods.base import PreparedLinearInput
from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockLinearMethod
from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg
from vllm_ascend.utils import attention_calculation_stream, npu_stream_switch

from . import triton_ops
from .c8_kvqsfa import attend_c8_v2, attend_c8_window
from .c8_read import supports_c8_qk_heads
from .index_pool_select import select_pool_candidates, write_pool_scores
from .mla_prolog import CANNBotBf16MlaProlog
from .pd_role import uses_prefill_ops
from .prefill_metadata import logical_positions, reuse_prefill_metadata

logger = init_logger(__name__)

ACL_FORMAT_ND = 2
C8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
C8_GPU_MIN_SCALE = 1e-4
BF16_QUERY_PROJECTION_LAYER = 1
QUERY_CHUNK = 16
C8_PREFILL_MAX_HEADS = 16
TRANSPOSE_BMM_MAX_TOKENS = 65536
# Larger gathers benefit from the bulk CANN copy/cast path.
MAX_FUSED_C8_READ_ROWS = 4096
# The bulk CANN writer has lower latency for large sparse prefill batches.
MAX_FUSED_C8_WRITE_TOKENS = 4096
# 512 captured decode tokens with 16 index heads fit in this bound.
MAX_FUSED_INDEXPOOL_ROWS = 8192
# Bound the vector kernel to the decode tensor sizes measured on 950DT.
MAX_FUSED_OUTPUT_GATE_ELEMENTS = 96 * 2048
# Equal-width CANN shapes verified on 950DT at TP8.
BATCH_VERIFY_MAX_QUERIES = 6
BATCH_VERIFY_SWA_SHAPE = (8, 512)
BATCH_VERIFY_SPARSE_SHAPE = (4, 2048)
# Queries, index heads, compression ratio, and complete groups recalled.
PAGED_POOL_DECODE_SHAPE = (6, 16, 4, 511)
# Store three int64 MRoPE coordinates in the BF16 sidecar by bit reinterpretation.
POSITION_LANES = 3 * torch.int64.itemsize // torch.bfloat16.itemsize


def _head_bmm(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # [tokens, heads, in] x [heads, in, out] -> [tokens, heads, out].
    if (
        tokens.device.type == "npu"
        and tokens.dtype == weight.dtype == torch.bfloat16
        and 0 < tokens.shape[0] < TRANSPOSE_BMM_MAX_TOKENS
        and hasattr(torch_npu, "npu_transpose_batchmatmul")
    ):
        return torch_npu.npu_transpose_batchmatmul(tokens, weight, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
    return torch.bmm(tokens.transpose(0, 1).contiguous(), weight).transpose(0, 1)


@dataclass
class FlashAttentionMetadata:
    query_start_loc: tuple[int, ...]
    seq_lens: tuple[int, ...]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    precise_decode_reqs: tuple[bool, ...] = ()
    decode_seq_lens: torch.Tensor | None = None
    prefill_reqs: tuple[bool, ...] = ()
    query_start_loc_device: torch.Tensor | None = None


def get_flash_attention_metadata(layer_name: str | None = None) -> FlashAttentionMetadata | None:
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    batch = metadata.get(layer_name) if layer_name is not None else next(iter(metadata.values()), None)
    return batch if isinstance(batch, FlashAttentionMetadata) else None


class FlashMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.precise_decode = vllm_config.additional_config.get(
            "hy4flash_precise_decode", vllm_config.speculative_config is not None
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        meta = common_attn_metadata
        query_starts = tuple(meta.query_start_loc_cpu[: meta.num_reqs + 1].tolist())
        is_prefilling = getattr(meta, "is_prefilling", None)
        prefill_reqs = tuple(is_prefilling[: meta.num_reqs].tolist()) if is_prefilling is not None else ()
        precise_decode_reqs = tuple(not prefill for prefill in prefill_reqs) if self.precise_decode else ()
        return FlashAttentionMetadata(
            query_starts,
            tuple(meta.seq_lens_cpu[: meta.num_reqs].tolist()),
            meta.block_table_tensor,
            meta.slot_mapping,
            precise_decode_reqs,
            prefill_reqs=prefill_reqs,
            query_start_loc_device=getattr(meta, "query_start_loc", None),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        metadata = self.build(0, common_attn_metadata)
        metadata.decode_seq_lens = common_attn_metadata.seq_lens
        return metadata


class FlashAttentionImpl(CacheOnlyAttentionImpl):
    @staticmethod
    def update_graph_params(*args, **kwargs):
        # The captured operators read the runner's device metadata directly.
        pass


class FlashAttentionBackend(CacheOnlyAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16", "fp8"]

    @staticmethod
    def get_name():
        return "HYV4_FLASH_ASCEND"

    @staticmethod
    def get_builder_cls():
        return FlashMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return FlashAttentionImpl


@lru_cache(maxsize=16)
def _indexpool_hadamard_basis(device, stream_handle):
    # The input FP8 bins are defined by the unnormalized Sylvester transform.
    basis = torch.ones((1, 1), dtype=torch.bfloat16, device="cpu")
    while basis.shape[0] < INDEX_DIM:
        upper = torch.cat((basis, basis), dim=1)
        lower = torch.cat((basis, -basis), dim=1)
        basis = torch.cat((upper, lower))
    return basis.to(device)


def _encode_prefill_indexpool_fp8(vectors, *, output_dtype=torch.float32):
    from .prefill_ops.index_pool_fp8 import round_prefill_fp8_rows

    stream = torch.npu.current_stream(vectors.device)
    basis = _indexpool_hadamard_basis(vectors.device, stream.npu_stream)
    transformed = torch.mm(vectors.view(-1, INDEX_DIM), basis, out_dtype=torch.float32)
    return round_prefill_fp8_rows(transformed, output_dtype=output_dtype).view_as(vectors)


class FlashIndexPool(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        rope_parameters,
        prefix,
        *,
        fuse_fp8_encoding=False,
        fp8_encoder=triton_ops.encode_indexpool_fp8,
        use_eager_fp8_encoding=False,
    ):
        super().__init__()
        self.fuse_fp8_encoding = fuse_fp8_encoding
        self.fp8_encoder = fp8_encoder
        self.use_eager_fp8_encoding = use_eager_fp8_encoding
        self.ratio = config.compressed_indexer_compress_ratio
        self.budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        self.rope_dim = config.qk_rope_head_dim
        self.theta = rope_parameters.get("rope_theta", 10000.0)
        self.sections = rope_parameters.get("mrope_section")
        self.interleaved = rope_parameters.get("mrope_interleaved", True)
        self.eps = config.rms_norm_eps
        if rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("HYV4 Flash IndexPool requires default RoPE")
        for name, input_size, output_size in (
            ("query_projection", config.q_lora_rank, self.num_heads * INDEX_DIM),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, INDEX_DIM),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size, output_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.{name}"
                ),
            )
        self.token_key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps)
        self.key_value_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
        self.key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(torch.empty(self.ratio, INDEX_DIM, dtype=torch.float32))

    def rotate(
        self,
        vectors,
        positions,
        *,
        shared_positions=None,
        position_start=0,
        position_step=1,
        eager_prefill=False,
        group_map=None,
        history_context_key=None,
        normalization=None,
    ):
        if (
            eager_prefill
            and self.use_eager_fp8_encoding
            and vectors.device.type == "npu"
            and (vectors.dtype == torch.bfloat16 or normalization is not None)
            and vectors.is_contiguous()
            and vectors.shape[-1] == INDEX_DIM == 128
            and self.rope_dim == 64
            and 0 < vectors.shape[0] <= 4096
            and positions.ndim in (1, 2)
            and (positions.ndim == 1 or (positions.shape[0] == 3 and self.sections))
        ):
            from .prefill_ops.index_pool_rope import apply_factors, make_factors

            stream = torch.npu.current_stream(vectors.device).npu_stream
            sections = tuple(self.sections) if self.sections else None
            if shared_positions is None:
                factors = make_factors(positions, self.theta, self.rope_dim, sections, self.interleaved, stream)
                return apply_factors(vectors, factors, normalization=normalization)
            cache = get_forward_context().additional_kwargs.setdefault("hy4_index_pool_rope", {})
            key = (id(shared_positions), self.theta, self.rope_dim, sections, self.interleaved, vectors.device, stream)
            if key not in cache:
                factors = make_factors(shared_positions, self.theta, self.rope_dim, sections, self.interleaved, stream)
                cache[key] = (shared_positions, factors)
            if group_map is not None:
                history_factors = None
                if positions.shape[-1]:
                    def make_history_factors():
                        return make_factors(positions, self.theta, self.rope_dim, sections, self.interleaved, stream)

                    if history_context_key is None:
                        history_factors = make_history_factors()
                    else:
                        history_factors = reuse_prefill_metadata(
                            vectors.device, ("index_pool_history_rope", key, history_context_key), make_history_factors
                        )
                return apply_factors(
                    vectors, cache[key][1], group_map=group_map, history_factors=history_factors,
                    normalization=normalization,
                )
            return apply_factors(vectors, cache[key][1], position_start, position_step, normalization=normalization)

        if normalization is not None:
            weight, multiplier = normalization
            vectors = ((vectors * multiplier[:, None]) * weight).bfloat16()
        inverse = self.theta ** (
            -torch.arange(0, self.rope_dim, 2, device=vectors.device, dtype=torch.float32) / self.rope_dim
        )
        phase = positions.float().unsqueeze(-1) * inverse
        if positions.ndim == 2:
            if self.interleaved:
                from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

                phase = apply_interleaved_rope(phase, self.sections)
            else:
                parts = phase.split(self.sections, dim=-1)
                phase = torch.cat([part[axis] for axis, part in enumerate(parts)], dim=-1)
        if vectors.ndim == 3:
            phase = phase.unsqueeze(1)
        nope, rotary = vectors.split((INDEX_DIM - self.rope_dim, self.rope_dim), dim=-1)
        even, odd = rotary.float()[..., ::2], rotary.float()[..., 1::2]
        cos, sin = phase.cos(), phase.sin()
        return torch.cat(
            (nope, (even * cos - odd * sin).to(vectors.dtype), (odd * cos + even * sin).to(vectors.dtype)), dim=-1
        )

    def _encode_fp8(self, vectors, *, full_graph, output_dtype=torch.float32):
        if (
            self.use_eager_fp8_encoding
            and not full_graph
            and vectors.device.type == "npu"
            and vectors.dtype == torch.bfloat16
            and vectors.is_contiguous()
            and vectors.shape[-1] == INDEX_DIM
            and 512 * INDEX_DIM <= vectors.numel() <= 65536 * INDEX_DIM
        ):
            return _encode_prefill_indexpool_fp8(vectors, output_dtype=output_dtype)
        if (
            full_graph
            and self.fuse_fp8_encoding
            and vectors.device.type == "npu"
            and vectors.dtype == torch.bfloat16
            and vectors.is_contiguous()
            and 0 < vectors.numel() <= MAX_FUSED_INDEXPOOL_ROWS * INDEX_DIM
        ):
            return self.fp8_encoder(vectors)
        return indexpool_fp8(vectors)

    def encode_queries(
        self, hidden, q_c, positions, *, full_graph=False, shared_positions=None, output_dtype=torch.float32
    ):
        query = self.query_projection(q_c)[0].reshape(-1, self.num_heads, INDEX_DIM)
        weight = self.head_weight_projection(hidden)[0].float() * (self.num_heads * INDEX_DIM) ** -0.5
        rotated = self.rotate(
            query, positions, shared_positions=shared_positions, eager_prefill=shared_positions is not None
        )
        return self._encode_fp8(rotated, full_graph=full_graph, output_dtype=output_dtype), weight

    def encode_keys(
        self,
        raw,
        positions,
        *,
        full_graph=False,
        eager_prefill=False,
        shared_positions=None,
        position_start=0,
        group_map=None,
        history_context_key=None,
    ):
        groups = raw.shape[0] // self.ratio
        raw = raw[: groups * self.ratio].reshape(groups, self.ratio, INDEX_DIM)
        if not groups:
            return raw.new_empty((0, INDEX_DIM), dtype=torch.float32)
        normalization = None
        if (
            HAS_TRITON
            and eager_prefill
            and self.use_eager_fp8_encoding
            and not full_graph
            and (group_map is not None or raw.dtype == torch.bfloat16)
            and self.ratio == 4
            and self.rope_dim == 64
            and raw.device.type == "npu"
            and raw.dtype in (torch.bfloat16, torch.float32)
            and self.key_norm.weight.dtype == torch.float32
            and self.key_norm.weight.is_contiguous()
            and 128 <= groups <= 1024
        ):
            from .prefill_ops.pool_weighted_stats import pool_with_inverse_norm

            raw_fp32 = raw.float()
            values = self.key_value_projection(raw_fp32)
            probabilities = (self.key_gate_projection(raw_fp32) + self.intra_group_ape).softmax(dim=1)
            if values.is_contiguous() and probabilities.is_contiguous():
                pooled, inverse_norm = pool_with_inverse_norm(values, probabilities, self.eps)
                normalization = (self.key_norm.weight, inverse_norm)
        if normalization is None:
            # Both projections consume the same FP32 input. Materialize it once in prefill.
            pooled = pooled_keys(
                raw.float() if eager_prefill else raw,
                self.key_value_projection.weight,
                self.key_gate_projection.weight,
                self.intra_group_ape,
                self.key_norm.weight,
                self.eps,
            ).to(torch.bfloat16 if group_map is not None else raw.dtype)
        if group_map is None:
            positions = positions[..., : groups * self.ratio : self.ratio]
        pooled = self.rotate(
            pooled,
            positions,
            eager_prefill=eager_prefill,
            group_map=group_map,
            shared_positions=shared_positions,
            position_start=position_start,
            position_step=self.ratio,
            history_context_key=history_context_key,
            normalization=normalization,
        )
        return self._encode_fp8(pooled, full_graph=full_graph)


class AscendHYV4FlashAttention(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, config, layer_idx, prefix, *, reuse_target=False):
        super().__init__()
        self.use_cann_prefill = uses_prefill_ops(vllm_config) and not reuse_target
        transfer = vllm_config.kv_transfer_config
        self.use_pd_decode = transfer is not None and transfer.kv_role == "kv_consumer"
        self.bulk_c8_decode_reads = vllm_config.additional_config.get("hy4flash_bulk_c8_decode_reads", False)
        if not isinstance(self.bulk_c8_decode_reads, bool):
            raise ValueError("hy4flash_bulk_c8_decode_reads must be a boolean")
        self.ops = triton_ops
        if self.use_cann_prefill:
            from . import li_prefill_p0, prefill_ops

            self.ops = prefill_ops
            self.li_prefill = li_prefill_p0
        self.layer_name = f"{prefix}.attn"
        self.is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.window = config.sliding_window
        self.max_model_len = vllm_config.model_config.max_model_len
        dim_prefix = "swa_" if self.is_window_layer else ""
        heads = getattr(config, f"{dim_prefix}num_attention_heads")
        self.local_heads = heads // get_tensor_model_parallel_world_size()
        self.nope_dim = getattr(config, f"{dim_prefix}qk_nope_head_dim")
        self.rope_dim = getattr(config, f"{dim_prefix}qk_rope_head_dim")
        self.v_dim = getattr(config, f"{dim_prefix}v_head_dim")
        self.latent_dim = config.kv_lora_rank
        self.scale = (self.nope_dim + self.rope_dim) ** -0.5
        self.kv_width = self.latent_dim + self.rope_dim
        self.use_fp8_cache = vllm_config.cache_config.cache_dtype == "fp8"
        self.use_gpu_c8_scale = vllm_config.additional_config.get("hy4flash_gpu_c8_scale", False)
        if not isinstance(self.use_gpu_c8_scale, bool):
            raise ValueError("hy4flash_gpu_c8_scale must be a boolean")
        if self.use_gpu_c8_scale and not self.use_fp8_cache:
            raise ValueError("hy4flash_gpu_c8_scale requires --kv-cache-dtype fp8")
        has_fused_ops = self.use_cann_prefill or HAS_TRITON
        use_a5_c8 = has_fused_ops and self.use_fp8_cache and get_ascend_device_type() is AscendDeviceType.A5
        self.shard_prefill_li = (
            self.use_cann_prefill
            and use_a5_c8
            and vllm_config.model_config.enforce_eager
            and vllm_config.parallel_config.prefill_context_parallel_size == 1
            and not vllm_config.parallel_config.use_sequence_parallel_moe
            and not vllm_config.additional_config.get("enable_flashcomm1", False)
            and os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0").strip().lower() not in ("1", "true")
        )
        self.fuse_c8_write = use_a5_c8
        self.fuse_c8_qk = use_a5_c8 and not reuse_target and (self.latent_dim, self.rope_dim) == (512, 64)
        self.fuse_c8_prefill_window = self.use_cann_prefill and self.fuse_c8_qk
        self.batch_prefill_window = (
            self.use_cann_prefill
            and vllm_config.model_config.enforce_eager
            and os.getenv("HY4_PREFILL_CAUSAL_PREFIX") == "1"
            and os.getenv("HY4_PREFILL_BATCH_WINDOW") == "1"
            and is_950()
        )
        self.fuse_indexpool_select = use_a5_c8 and not reuse_target
        self.share_swa_c8_reads = use_a5_c8
        self.fuse_swa_q_norm = (
            self.is_window_layer
            and not reuse_target
            and get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_FUSION)
            and hasattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant")
        )
        self.fuse_attn_output_gate = (
            not reuse_target and has_fused_ops and get_ascend_device_type() is AscendDeviceType.A5
        )
        self.overlap_attn_output_gate = not reuse_target and get_ascend_device_type() is AscendDeviceType.A5
        self.use_cannbot_prolog = not reuse_target and vllm_config.additional_config.get(
            "hy4flash_cannbot_mla_prolog", False
        )
        self._cannbot_prolog = None
        self.use_eager_prefill_rotary = (
            self.use_cann_prefill
            and use_a5_c8
            and vllm_config.model_config.enforce_eager
            and hasattr(torch_npu, "npu_apply_rotary_pos_emb")
        )
        self.use_triton_mla_logits = has_fused_ops and vllm_config.additional_config.get(
            "hy4flash_triton_mla_logits", False
        )
        self.batch_verify_attention = use_a5_c8 and not self.use_triton_mla_logits
        self.kv_storage_width = C8_ROW_BYTES if self.use_fp8_cache else self.kv_width
        # Indexer keys and position lanes remain BF16 in either layout.
        index_element_width = 2 if self.use_fp8_cache else 1
        index_width = INDEX_DIM + POSITION_LANES
        self.token_storage_width = self.kv_storage_width + index_width * index_element_width
        ratio = config.compressed_indexer_compress_ratio
        # Use the same page size for SWA and sparse layers in this cache group.
        self.incremental_index_pool = (
            has_fused_ops and ratio > 0 and INDEX_DIM % ratio == 0 and vllm_config.cache_config.block_size % ratio == 0
        )
        self.cache_width = self.token_storage_width
        if self.incremental_index_pool:
            self.cache_width += INDEX_DIM // ratio * index_element_width
        self.kv_cache = torch.empty(0)
        self.topk_snapshot = None
        if not reuse_target:
            cache_layers = vllm_config.compilation_config.static_forward_context
            if self.layer_name in cache_layers:
                raise ValueError(f"Duplicate HYV4 Flash cache layer {self.layer_name}")
            cache_layers[self.layer_name] = self

        quant = vllm_config.quant_config
        hidden = config.hidden_size
        self.q_a_proj = ReplicatedLinear(
            hidden, config.q_lora_rank, bias=False, quant_config=quant, prefix=f"{prefix}.q_a_proj"
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden, self.kv_width, bias=False, quant_config=quant, prefix=f"{prefix}.kv_a_proj_with_mqa"
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.latent_dim, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            heads * (self.nope_dim + self.rope_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        if not reuse_target and layer_idx == BF16_QUERY_PROJECTION_LAYER:
            # Keep the BF16 reconstruction from the checkpoint's block scales.
            self.q_b_proj.keep_block_fp8_weight_dense = True
        self.kv_b_proj = ColumnParallelLinear(
            self.latent_dim,
            heads * (self.nope_dim + self.v_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        # Absorbed MLA reads this projection's weight directly, as in DSV4.1 wo_a.
        self.kv_b_proj.skip_weight_nz_conversion = True
        self.register_buffer("_decode_key_projection", None, persistent=False)
        self.register_buffer("_decode_value_projection", None, persistent=False)
        self.linear_gate = ColumnParallelLinear(
            hidden, heads * self.v_dim, bias=False, quant_config=quant, prefix=f"{prefix}.linear_gate"
        )
        self.o_proj = RowParallelLinear(
            heads * self.v_dim, hidden, bias=False, quant_config=quant, prefix=f"{prefix}.o_proj"
        )
        self.sequence_parallel_output = False
        rope_parameters = config.rope_for_layer(layer_idx)
        self.rotary_emb = get_rope(
            self.rope_dim,
            max_position=vllm_config.model_config.max_model_len,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )
        self._mrope_channel_bits = None
        if (
            self.use_eager_prefill_rotary
            and HAS_TRITON
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        ):
            from .prefill_ops.small_aiv import mrope_channel_bits

            self._mrope_channel_bits = mrope_channel_bits(self.rotary_emb.mrope_section)
        self.indexer = (
            None
            if self.is_window_layer or reuse_target
            else FlashIndexPool(
                config,
                quant,
                rope_parameters,
                f"{prefix}.indexer",
                fuse_fp8_encoding=use_a5_c8,
                fp8_encoder=self.ops.encode_indexpool_fp8,
                use_eager_fp8_encoding=(self.use_cann_prefill and use_a5_c8 and vllm_config.model_config.enforce_eager),
            )
        )
        self.learnable_sink_param = make_attention_sink(self.local_heads, torch.float32)
        self.post_proj_norm = (
            RMSNorm(hidden, config.rms_norm_eps)
            if config.sandwich_norm and (self.is_window_layer or not config.dsa_skip_attn_post_norm)
            else None
        )

    def get_attn_backend(self):
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        geometry = dict(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.cache_width,
            dtype=torch.uint8 if self.use_fp8_cache else torch.bfloat16,
        )
        if (
            not envs.VLLM_USE_V2_MODEL_RUNNER
            or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            or not vllm_config.additional_config.get("hy4flash_swa_cache", True)
        ):
            return HiddenStateCacheSpec(**geometry)
        if self.is_window_layer:
            return SlidingWindowMLASpec(
                **geometry,
                cache_dtype_str=PACKED_CACHE_FORMAT,
                sliding_window=self.window,
                # A P/D consumer can replay the final prefill token.
                extra_retained_tokens=int(vllm_config.kv_transfer_config is not None),
            )
        return MLAAttentionSpec(**geometry, cache_dtype_str=PACKED_CACHE_FORMAT)

    def _project_query(self, hidden_states):
        q_a = self.q_a_proj(hidden_states)[0]
        if self.fuse_swa_q_norm:
            norm = self.q_a_layernorm
            scheme = getattr(self.q_b_proj.quant_method, "quant_method", None)
            if (
                isinstance(scheme, AscendFp8BlockLinearMethod)
                and scheme.use_mxfp8
                and scheme.mxfp8_method.group_size == 32
                and not any(getattr(self.q_b_proj, "mxfp8_tp_padding", (0, 0)))
                and q_a.device.type == "npu"
                and q_a.dtype == norm.weight.dtype == torch.bfloat16
                and q_a.ndim == 2
                and q_a.numel() > 0
                and q_a.is_contiguous()
                and q_a.shape[-1] % 64 == 0
                and (getattr(norm, "bias", None) is None or not getattr(norm, "bias_loaded", True))
            ):
                logger.info_once("HYV4 Flash uses fused SWA query RMSNorm and MXFP8 activation quantization.")
                quantized, scale, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                    q_a,
                    norm.weight,
                    epsilon=norm.variance_epsilon,
                    dst_type=torch.float8_e4m3fn,
                    scale_alg=scheme.mxfp8_method.dynamic_mx_quant_scale_alg,
                )
                return self.q_b_proj(PreparedLinearInput(quantized, scale, q_a.dtype))[0], None

        q_c = self.q_a_layernorm(q_a)
        return self.q_b_proj(q_c)[0], q_c

    def _can_fuse_output_gate(self, value: torch.Tensor) -> bool:
        scheme = getattr(self.o_proj.quant_method, "quant_method", None)
        return (
            isinstance(scheme, AscendFp8BlockLinearMethod)
            and scheme.use_mxfp8
            and scheme.mxfp8_method.group_size == 32
            and scheme.mxfp8_method.dynamic_mx_quant_scale_alg == DynamicMXQuantScaleAlg.MAX_EXPONENT
            and not any(getattr(self.o_proj, "mxfp8_tp_padding", (0, 0)))
            and self.o_proj.input_is_parallel
            and getattr(self.o_proj, "custom_op", None) is None
            and value.device.type == "npu"
            and value.dtype == torch.bfloat16
            and value.shape[-1] in (1024, 2048)
            and 0 < value.numel() <= MAX_FUSED_OUTPUT_GATE_ELEMENTS
        )

    def _project_output_gate(self, hidden_states: torch.Tensor, fuse_gate: bool) -> torch.Tensor:
        logits = self.linear_gate(hidden_states)[0]
        return logits if fuse_gate else logits.sigmoid()

    def _rotate_qk(self, positions, q_pe, k_pe):
        rope = self.rotary_emb
        if (
            self.use_eager_prefill_rotary
            and positions.ndim == 2
            and positions.shape[0] == 3
            and getattr(rope, "mrope_interleaved", False)
            and not rope.is_neox_style
            and rope.head_size == rope.rotary_dim == self.rope_dim == 64
            and q_pe.device.type == "npu"
            and q_pe.dtype == k_pe.dtype == torch.bfloat16
            and q_pe.shape[1] == self.local_heads * 64
            and k_pe.shape[1] == 64
            and 512 <= q_pe.shape[0] <= 4096
            and positions.shape[1] == q_pe.shape[0]
        ):
            context = get_forward_context()
            metadata = context.attn_metadata
            batch = metadata.get(self.layer_name) if isinstance(metadata, dict) else None
            if batch is not None and batch.prefill_reqs and all(batch.prefill_reqs):
                stream = torch.npu.current_stream(q_pe.device)
                cache = context.additional_kwargs.setdefault("hy4_prefill_rotary", {})
                key = (rope, id(positions), q_pe.device, q_pe.dtype, stream.npu_stream)
                factors = cache.get(key)
                rows = q_pe.shape[0]
                if factors is None:
                    rotary_cache = rope._match_cos_sin_cache_dtype(q_pe)
                    if (
                        self._mrope_channel_bits is not None
                        and rotary_cache.is_contiguous()
                        and rotary_cache.shape[-1] == 64
                        and positions.dtype == torch.int64
                    ):
                        from .prefill_ops.small_aiv import gather_mrope_factors

                        cos, sin = gather_mrope_factors(rotary_cache, positions, self._mrope_channel_bits)
                    else:
                        from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

                        cos, sin = rotary_cache[positions].chunk(2, dim=-1)
                        cos = apply_interleaved_rope(cos, rope.mrope_section)
                        sin = apply_interleaved_rope(sin, rope.mrope_section)
                        cos = cos.repeat_interleave(2, dim=-1).view(1, rows, 1, 64)
                        sin = sin.repeat_interleave(2, dim=-1).view(1, rows, 1, 64)
                    factors = (positions, cos, sin)
                    cache[key] = factors
                q, k = torch_npu.npu_apply_rotary_pos_emb(
                    q_pe.view(1, rows, self.local_heads, 64),
                    k_pe.view(1, rows, 1, 64),
                    factors[1],
                    factors[2],
                    layout="BSND",
                    rotary_mode="interleave",
                )
                return q.reshape_as(q_pe), k.reshape_as(k_pe)
        return rope(positions, q_pe, k_pe)

    def forward(self, positions, hidden_states):
        needs_meta = self.use_cann_prefill or self.fuse_attn_output_gate or self.overlap_attn_output_gate
        meta = get_flash_attention_metadata(self.layer_name) if needs_meta else None
        full_decode = meta is not None and meta.decode_seq_lens is not None
        full_prefill = (
            self.use_cann_prefill
            and meta is not None
            and not full_decode
            and bool(meta.prefill_reqs)
            and all(meta.prefill_reqs)
            and meta.query_start_loc[0] == 0
        )
        absorbed = (
            self.use_cannbot_prolog
            and hidden_states.shape[0] > 0
            and self.q_a_proj.weight.dtype
            == self.q_b_proj.weight.dtype
            == self.kv_a_proj_with_mqa.weight.dtype
            == torch.bfloat16
        )
        if absorbed:
            if self._cannbot_prolog is None:
                wk, _ = self._kv_b_weights()
                self._cannbot_prolog = CANNBotBf16MlaProlog(
                    self.layer_name,
                    self.q_a_proj.weight,
                    self.q_b_proj.weight,
                    self.kv_a_proj_with_mqa.weight,
                    wk,
                    self.q_a_layernorm.weight,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                )
            q_c, q, kv = self._cannbot_prolog(hidden_states)
            q_nope, q_pe = q.split((self.latent_dim, self.rope_dim), dim=-1)
            kv_c, k_pe = kv.split((self.latent_dim, self.rope_dim), dim=-1)
        else:
            q, q_c = self._project_query(hidden_states)
            q = q.reshape(-1, self.local_heads, self.nope_dim + self.rope_dim)
            kv = self.kv_a_proj_with_mqa(hidden_states)[0]
            if (
                full_prefill
                and meta.query_start_loc[-1] == q.shape[0] > 0
                and self.use_eager_prefill_rotary
                and HAS_TRITON
                and q.device.type == "npu"
                and q.dtype == kv.dtype == torch.bfloat16
                and q.is_contiguous()
                and kv.is_contiguous()
                and 512 <= q.shape[0] <= 4096
                and self.local_heads in (4, 8, 16)
                and (self.nope_dim, self.latent_dim, self.rope_dim) == (192, 512, 64)
            ):
                from .prefill_ops.small_aiv import pack_attention_projections

                q_nope, q_pe, kv_c, k_pe = pack_attention_projections(q, kv, self.rope_dim)
            else:
                kv_c, k_pe = kv.split((self.latent_dim, self.rope_dim), -1)
                q_nope, q_pe = q.split((self.nope_dim, self.rope_dim), dim=-1)
            kv_c = self.kv_a_layernorm(kv_c)
        q_pe, k_pe = self._rotate_qk(positions, q_pe.reshape(q.shape[0], -1).contiguous(), k_pe.contiguous())
        q_pe = q_pe.reshape(-1, self.local_heads, self.rope_dim)
        full_prefill = full_prefill and meta.query_start_loc[-1] == q.shape[0] > 0
        out_shape = (q.shape[0], self.local_heads * self.v_dim)
        out = hidden_states.new_empty(out_shape) if full_prefill else hidden_states.new_zeros(out_shape)
        fuse_gate = self.fuse_attn_output_gate and full_decode and self._can_fuse_output_gate(out)
        overlap_gate = self.overlap_attn_output_gate and full_decode and not self.sequence_parallel_output
        if overlap_gate:
            logger.info_once("HYV4 Flash overlaps the output gate projection with paged decode attention.")
            main_stream = torch.npu.current_stream()
            gate_stream = attention_calculation_stream()
            gate_stream.wait_stream(main_stream)
            with npu_stream_switch(gate_stream, enabled=True):
                hidden_states.record_stream(gate_stream)
                gate = self._project_output_gate(hidden_states, fuse_gate)

        if full_prefill:
            out = self._forward_paged(
                hidden_states, q_c, q_nope, q_pe, kv_c, k_pe, positions, out,
                absorbed=absorbed, reuse_prefill_output=True,
            )
        else:
            self._forward_paged(
                hidden_states, q_c, q_nope, q_pe, kv_c, k_pe, positions, out, absorbed=absorbed,
            )
        if overlap_gate:
            main_stream.wait_stream(gate_stream)
            gate.record_stream(main_stream)
        else:
            gate = self._project_output_gate(hidden_states, fuse_gate)

        if (
            fuse_gate
            and gate.dtype == out.dtype
            and gate.is_contiguous()
            and torch_npu.get_npu_format(gate) == ACL_FORMAT_ND
        ):
            if self.use_cann_prefill:
                logger.info_once("HYV4 Flash uses CANN DynamicMXQuant after the output gate.")
            else:
                logger.info_once("HYV4 Flash fuses the output gate and MXFP8 input quantization.")
            quantized, scale = self.ops.gate_dynamic_mx_quant(out, gate)
            output = self.o_proj(PreparedLinearInput(quantized, scale, out.dtype))[0]
        else:
            if fuse_gate:
                gate = gate.sigmoid()
            output = self.o_proj(out * gate)[0]
        if self.sequence_parallel_output:
            output = sp_reduce_scatter(output)
        return self.post_proj_norm(output) if self.post_proj_norm is not None else output

    def _forward_paged(
        self, hidden, q_c, q_nope, q_pe, kv_c, k_pe, positions, out, *,
        absorbed=False, reuse_prefill_output=False,
    ):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or self.layer_name not in metadata:
            return out
        meta = metadata[self.layer_name]
        cache = self.kv_cache.view(-1, self.cache_width)
        block_size = self.kv_cache.shape[2]
        num_tokens = meta.query_start_loc[-1]
        if not num_tokens:
            return out
        snapshot = self.topk_snapshot
        if snapshot is not None:
            snapshot[:num_tokens].fill_(-1)
        full_graph = meta.decode_seq_lens is not None
        keys = self._write_cache_rows(cache, meta, hidden, kv_c, k_pe, positions, num_tokens, full_graph=full_graph)
        query, wv = self._absorb_query(q_nope[:num_tokens], q_pe[:num_tokens], absorbed=absorbed)
        if full_graph:
            self._forward_decode_graph(meta, cache, block_size, hidden, q_c, positions, query, wv, out)
            return out

        if keys is not None and self.incremental_index_pool:
            self._update_index_pool(cache, block_size, meta, keys, positions)
        return self._attend_eager(
            meta, cache, block_size, hidden, q_c, positions, query, wv, out,
            reuse_output=reuse_prefill_output,
        )

    def _write_cache_rows(self, cache, meta, hidden, kv_c, k_pe, positions, num_tokens, *, full_graph):
        slots = meta.slot_mapping[:num_tokens]
        use_fused_write = (
            self.fuse_c8_write
            and cache.device.type == "npu"
            and num_tokens <= MAX_FUSED_C8_WRITE_TOKENS
            and self.cache_width % torch.int32.itemsize == 0
        )
        keys = index_norm = None
        if self.indexer is not None:
            keys = self.indexer.token_key_projection(hidden[:num_tokens])[0]
            norm = self.indexer.token_key_norm
            if (
                HAS_TRITON
                and use_fused_write
                and self.use_cann_prefill
                and self.indexer.use_eager_fp8_encoding
                and not full_graph
                and meta.prefill_reqs
                and all(meta.prefill_reqs)
                and cache.is_contiguous()
                and keys.dtype == norm.weight.dtype == torch.bfloat16
                and keys.is_contiguous()
                and norm.weight.is_contiguous()
                and keys.shape == (num_tokens, INDEX_DIM)
                and 512 <= num_tokens <= 4096
            ):
                from .prefill_ops.index_pool_norm import token_key_norm_inverse

                index_norm = (norm.weight, token_key_norm_inverse(keys, norm.eps))
            else:
                keys = norm(keys).bfloat16()
        if use_fused_write:
            payload, scales = self._quantize_c8(kv_c[:num_tokens], gpu_scale=self.use_gpu_c8_scale)
            if index_norm is not None:
                return self.ops.write_c8_rows(
                    cache, slots, payload, k_pe[:num_tokens], scales, keys, positions[..., :num_tokens],
                    index_norm=index_norm,
                )
            self.ops.write_c8_rows(cache, slots, payload, k_pe[:num_tokens], scales, keys, positions[..., :num_tokens])
        else:
            kv = self._pack_kv(kv_c[:num_tokens], k_pe[:num_tokens])
            if keys is not None:
                if positions.ndim == 1:
                    coordinates = positions[:num_tokens, None].expand(-1, 3)
                else:
                    coordinates = positions[:, :num_tokens].T
                packed_pos = coordinates.to(torch.int64).contiguous().view(torch.bfloat16)
                index_rows = torch.cat((keys, packed_pos), dim=-1)
                kv = torch.cat((kv, index_rows.view(torch.uint8) if self.use_fp8_cache else index_rows), dim=-1)
            rows = nn.functional.pad(kv, (0, self.cache_width - kv.shape[-1]))
            if full_graph:
                # Capture uses PAD_SLOT_ID. Page zero is reserved by vLLM.
                rows = torch.where(slots[:, None] >= 0, rows, cache[:1])
            cache.index_copy_(0, slots.long().clamp_min(0), rows)
        return keys

    def _attend_eager(self, meta, cache, block_size, hidden, q_c, positions, query, wv, out, *, reuse_output=False):
        parts = [] if reuse_output else None

        def write_output(start, stop, value):
            value = value.reshape_as(out[start:stop])
            if parts is None:
                out[start:stop].copy_(value)
            else:
                parts.append((start, stop, value))

        num_tokens = query.shape[0]
        snapshot = self.topk_snapshot
        uses_indexpool = self.indexer is not None and any(
            (length - 1) // self.indexer.ratio > self.indexer.budget for length in meta.seq_lens
        )
        all_prefill = uses_indexpool and self.use_cann_prefill and bool(meta.prefill_reqs and all(meta.prefill_reqs))
        if uses_indexpool:
            # Whole indexer chunks can store the FP8 bins in the dtype consumed by LI.
            can_store_bf16 = (
                all_prefill
                and self.fuse_c8_qk
                and query.dtype == torch.bfloat16
                and cache.dtype == torch.uint8
                and all(
                    (stop - start) % QUERY_CHUNK == 0
                    and seq_len - (stop - start) >= (self.indexer.budget + 1) * self.indexer.ratio
                    for seq_len, start, stop in zip(meta.seq_lens, meta.query_start_loc, meta.query_start_loc[1:])
                    if stop > start
                )
            )
            index_inputs = (hidden[:num_tokens], q_c[:num_tokens], positions[..., :num_tokens])
            if all_prefill:
                index_query, index_weight = self.indexer.encode_queries(
                    *index_inputs,
                    shared_positions=positions,
                    output_dtype=torch.bfloat16 if can_store_bf16 else torch.float32,
                )
            else:
                index_query, index_weight = self.indexer.encode_queries(*index_inputs)
        for i, seq_len in enumerate(meta.seq_lens):
            start, stop = meta.query_start_loc[i : i + 2]
            if start == stop:
                continue
            num_queries = stop - start
            verify = num_queries > 1 and bool(meta.precise_decode_reqs and meta.precise_decode_reqs[i])
            batch_verify = (
                self.batch_verify_attention
                and verify
                and num_queries <= BATCH_VERIFY_MAX_QUERIES
                and query.device.type == "npu"
            )
            is_prefill = not verify and bool(meta.prefill_reqs and meta.prefill_reqs[i])
            block_table = meta.block_table[i]
            may_score_groups = uses_indexpool and (seq_len - 1) // self.indexer.ratio > self.indexer.budget
            pooled_history = None
            fallback_pool = None
            if may_score_groups:
                if self.incremental_index_pool:
                    ratio = self.indexer.ratio
                    groups = seq_len // ratio
                    if (
                        is_prefill
                        and self.use_cann_prefill
                        and cache.device.type == "npu"
                        and cache.dtype == torch.uint8
                        and cache.is_contiguous()
                        and block_table.dtype in (torch.int32, torch.int64)
                        and block_table.is_contiguous()
                        and block_size == 64
                        and ratio == 4
                    ):
                        pooled_history = self.ops.read_prefill_pool(cache, block_table, groups, block_size, ratio)
                    else:
                        pooled_history = torch.empty((groups, INDEX_DIM), dtype=torch.float32, device=hidden.device)
                        self.ops.copy_pool_keys(cache, block_table, pooled_history, 0, block_size, ratio, write=False)
                else:
                    history = torch.arange(seq_len, device=hidden.device)
                    raw_keys, key_pos = self._read_index_rows(cache, block_table, block_size, history, positions.ndim)
            loop_stop = stop
            if (
                self.fuse_c8_prefill_window
                and is_prefill
                and self.indexer is None
                and self.window == 512
                and query.device.type == "npu"
                and query.dtype == torch.bfloat16
                and query.is_contiguous()
                and cache.dtype == torch.uint8
                and cache.is_contiguous()
                and block_table.is_contiguous()
                and 0 < query.shape[1] <= C8_PREFILL_MAX_HEADS
            ):
                early = max(0, self.window - 1 - (seq_len - num_queries))
                early = ((early + QUERY_CHUNK - 1) // QUERY_CHUNK) * QUERY_CHUNK
                if num_queries - early >= QUERY_CHUNK:
                    loop_stop = start + early
            reuse_window = is_prefill and self.indexer is None and num_queries > QUERY_CHUNK and loop_stop > start
            batch_early_window = (
                reuse_window
                and loop_stop - start > QUERY_CHUNK
                and snapshot is None
                and query.device.type == "npu"
                and query.dtype == torch.bfloat16
                and query.is_contiguous()
                and cache.dtype == torch.uint8
                and self.batch_prefill_window
            )
            fuse_sparse_prefill = (
                is_prefill
                and may_score_groups
                and query.device.type == "npu"
                and self.fuse_c8_qk
                and query.dtype == torch.bfloat16
                and query.is_contiguous()
                and cache.dtype == torch.uint8
                and cache.is_contiguous()
                and cache.shape[-1] % 4 == 0
                and block_table.is_contiguous()
                and (
                    0 < query.shape[1] <= C8_PREFILL_MAX_HEADS
                    if self.use_cann_prefill
                    else supports_c8_qk_heads(query.shape[1])
                )
            )
            use_cann_sparse = self.use_cann_prefill and fuse_sparse_prefill
            prefill_c8_v2_rows = []
            history_start = max(0, seq_len - num_queries + 1 - self.window) if self.indexer is None else 0
            shared_kv = None
            if (
                reuse_window
                or verify
                and (not may_score_groups or seq_len <= (stop - start) * self.indexer.ratio * (self.indexer.budget + 1))
            ):
                # The sliding-window union is bounded by this step's query
                # count plus the window width, independent of the prefix length.
                shared_end = seq_len - (stop - loop_stop)
                if batch_early_window:
                    shared_positions = logical_positions(history_start, shared_end, hidden.device)
                else:
                    shared_positions = torch.arange(history_start, shared_end, device=hidden.device)
                if (
                    batch_early_window
                    and self.use_cann_prefill
                    and self.use_fp8_cache
                    and HAS_TRITON
                    and get_ascend_device_type() is AscendDeviceType.A5
                    and cache.ndim == 2
                    and cache.is_contiguous()
                    and cache.shape[-1] >= C8_ROW_BYTES
                    and cache.shape[-1] % 4 == 0
                    and cache.storage_offset() % 4 == 0
                    and block_size == 64
                    and block_table.ndim == 1
                    and block_table.is_contiguous()
                    and block_table.device == cache.device
                    and block_table.dtype in (torch.int32, torch.int64)
                    and 0 < shared_positions.numel() <= MAX_FUSED_C8_READ_ROWS
                    and shared_end <= block_table.numel() * block_size
                ):
                    # Reuse the FP32 C8 reader without changing SWA reduction shapes.
                    shared_kv = triton_ops.gather_c8_rows(cache, block_table, block_size, shared_positions)
                else:
                    shared_kv = self._gather_attention_rows(cache, block_table, block_size, shared_positions)
            prefill_query = query[start:loop_stop].float() if reuse_window else None
            first_position = seq_len - num_queries
            end_position = seq_len - (stop - loop_stop)
            if use_cann_sparse:
                prefill_logical = logical_positions(first_position, end_position, hidden.device)
            elif reuse_window and not batch_early_window:
                prefill_logical = torch.arange(first_position, end_position, device=hidden.device)
            else:
                prefill_logical = None
            if (
                batch_verify
                and self.indexer is None
                and (self.local_heads, self.window) == BATCH_VERIFY_SWA_SHAPE
                and seq_len - num_queries + 1 >= self.window
            ):
                # Every query has a complete window, one row wider in history
                # than the preceding query. Use overlapping views of the gather.
                selected = shared_positions.unfold(0, self.window, 1)
                kv_selected = shared_kv.unfold(0, self.window, 1).transpose(1, 2)
                if snapshot is not None:
                    snapshot[start:stop, : self.window].copy_(selected)
                value = self._attend_decoded(query[start:stop], selected, kv_selected, wv)
                write_output(start, stop, value)
                continue
            batch_sparse = (
                batch_verify
                and may_score_groups
                and (seq_len - num_queries) // self.indexer.ratio > self.indexer.budget
                and (self.local_heads, (self.indexer.budget + 1) * self.indexer.ratio) == BATCH_VERIFY_SPARSE_SHAPE
                and cache.shape[-1] % 4 == 0
            )
            candidate_rows = []
            # Match the candidate and reduction widths of a single-token decode
            # for each verification query. Keep the batched projection and MoE.
            cached_groups = -1
            li_cache = None
            li_tail = None
            processed_until = start
            if batch_early_window:
                from .prefill_ops.swa_early import attend_window_chunks

                early_contexts = attend_window_chunks(
                    prefill_query, shared_kv, first_position, history_start, min(self.window, seq_len),
                    self.learnable_sink_param, self.scale, self.latent_dim, QUERY_CHUNK, query.dtype,
                )
                for offset, context in early_contexts:
                    lo = start + offset
                    write_output(lo, lo + context.size(0), _head_bmm(context, wv))
                processed_until = loop_stop
            chunk = 1 if verify else QUERY_CHUNK
            early_fusion = (
                os.getenv("HY4_PREFILL_CAUSAL_PREFIX") == "1"
                and is_prefill
                and snapshot is None
                and query.device.type == "npu"
                and is_950()
            )
            if early_fusion and use_cann_sparse:
                from .prefill_ops.early_recall import build_full_recall_rows

                history_budget = self.indexer.budget
                ratio = self.indexer.ratio
                first = seq_len - num_queries
                full_recall_end = (history_budget + 1) * ratio
                full_recall_rows = max(0, min(loop_stop - start, full_recall_end - first))
                if full_recall_rows != loop_stop - start:
                    # Preserve the original 16-row groups at the LI boundary.
                    full_recall_rows = full_recall_rows // chunk * chunk
                if full_recall_rows:
                    dtype = torch.int32 if first >= history_budget * ratio else torch.int64
                    selected = reuse_prefill_metadata(
                        query.device,
                        ("full_recall", first, full_recall_rows, history_budget, ratio, dtype),
                        lambda: build_full_recall_rows(
                            first, full_recall_rows, history_budget, ratio, query.device, dtype,
                        ),
                    )
                    prefill_c8_v2_rows.append(selected)
                    processed_until += full_recall_rows
            for lo in range(processed_until, loop_stop, chunk):
                if lo < processed_until:
                    continue
                hi = min(lo + chunk, loop_stop)
                visible_len = seq_len - (stop - hi)
                history_len = visible_len if verify else seq_len
                score_groups = may_score_groups and (history_len - 1) // self.indexer.ratio > self.indexer.budget
                scores = None
                if shared_kv is not None and not score_groups and not reuse_window:
                    first = max(0, visible_len - self.window) if self.indexer is None else 0
                    selected_slice = slice(first - history_start, visible_len - history_start)
                    selected = shared_positions[selected_slice].unsqueeze(0)
                    kv_selected = shared_kv[selected_slice].unsqueeze(0)
                else:
                    logical = (
                        prefill_logical[lo - start : hi - start]
                        if prefill_logical is not None
                        else torch.arange(visible_len - (hi - lo), visible_len, device=hidden.device)
                    )
                    if self.indexer is None:
                        selected = window_candidates(logical, min(self.window, history_len))
                    elif score_groups:
                        groups = history_len // self.indexer.ratio
                        if groups != cached_groups:
                            if pooled_history is not None:
                                pool = pooled_history[:groups]
                            else:
                                end = groups * self.indexer.ratio
                                pool = self.indexer.encode_keys(raw_keys[:end], key_pos[..., :end])
                            cached_groups = groups
                        q_index, w_index = index_query[lo:hi], index_weight[lo:hi]
                        if use_cann_sparse and self.li_prefill.eligible(
                            q_index,
                            w_index,
                            pool,
                            visible_len,
                            hi - lo,
                            self.indexer.ratio,
                            self.indexer.budget,
                        ):
                            if li_cache is None or li_cache.shape[0] != pool.shape[0]:
                                li_cache = pool.to(torch.bfloat16).contiguous()
                            if li_tail is None:
                                li_tail = reuse_prefill_metadata(
                                    hidden.device,
                                    ("li_tail", first_position, end_position),
                                    lambda: self.li_prefill.prepare_tail(prefill_logical),
                                )
                            # Full Prefill chunks share the same key pool. Only their causal lengths vary.
                            rows = min(self.li_prefill.BATCH_ROWS, (loop_stop - lo) // chunk * chunk)
                            if rows > chunk:
                                end = lo + rows
                                selected = self.li_prefill.select_many(
                                    index_query[lo:end],
                                    index_weight[lo:end],
                                    li_cache,
                                    visible_len,
                                    tail=li_tail[lo - start : end - start],
                                    tp_shard=self.shard_prefill_li and all_prefill,
                                )
                                prefill_c8_v2_rows.append(selected)
                                processed_until = end
                                continue
                            selected = self.li_prefill.select(
                                q_index,
                                w_index,
                                li_cache,
                                visible_len,
                                tail=li_tail[lo - start : hi - start],
                            )
                        else:
                            score_pool = pool
                            if pooled_history is not None and pooled_history.dtype == torch.bfloat16:
                                if fallback_pool is None:
                                    fallback_pool = pooled_history.float()
                                score_pool = fallback_pool[:groups]
                            select_candidates = sparse_candidates
                            if use_cann_sparse:
                                from .prefill_ops.index_pool_select import select_partial_prefill_pool

                                select_candidates = select_partial_prefill_pool
                                q_index = q_index.float()
                            selected = select_candidates(
                                q_index,
                                w_index,
                                score_pool,
                                logical,
                                self.indexer.ratio,
                                self.indexer.budget,
                            )
                            if (
                                use_cann_sparse
                                and seq_len - num_queries >= self.indexer.budget * self.indexer.ratio
                            ):
                                # LightningIndexer emits int32. Cast just the fallback rows so
                                # the mixed batch stays int32 through concatenation and CANN.
                                selected = selected.to(torch.int32)
                        if batch_sparse:
                            candidate_rows.append(selected)
                            continue
                    else:
                        # Every complete group fits the recall budget, so only causality matters.
                        visible = torch.arange(visible_len, device=hidden.device)
                        selected = visible.expand(hi - lo, -1).masked_fill(visible > logical[:, None], -1)
                    if use_cann_sparse:
                        prefill_c8_v2_rows.append(selected)
                        continue
                    if fuse_sparse_prefill:
                        scores, kv_selected = self.ops.gather_c8_qk(
                            query[lo:hi], cache, block_table, block_size, selected
                        )
                    elif shared_kv is None:
                        kv_selected = self._gather_attention_rows(cache, block_table, block_size, selected.clamp_min(0))
                    else:
                        valid_indices = (selected - history_start).clamp_min(0)
                        kv_selected = shared_kv.index_select(0, valid_indices.flatten()).view(*selected.shape, -1)
                if snapshot is not None:
                    snapshot[lo:hi, : selected.shape[-1]].copy_(selected)
                value = self._attend_decoded(
                    query[lo:hi],
                    selected,
                    kv_selected,
                    wv,
                    scores=scores,
                    query_fp32=prefill_query[lo - start : hi - start] if reuse_window else None,
                    fuse_prefill_logits=early_fusion and reuse_window,
                )
                write_output(lo, hi, value)
            if loop_stop < stop:
                context = attend_c8_window(
                    query[loop_stop:stop],
                    cache,
                    block_table,
                    block_size,
                    seq_len,
                    self.learnable_sink_param,
                    self.scale,
                    window=self.window,
                )
                if snapshot is not None:
                    logical = torch.arange(seq_len - (stop - loop_stop), seq_len, device=query.device)
                    snapshot[loop_stop:stop, : self.window].copy_(window_candidates(logical, self.window))
                value = _head_bmm(context, wv)
                write_output(loop_stop, stop, value)
                logger.info_once(
                    "HYV4 Prefill uses CANN KvQuantSparseFlashAttentionV2 for sliding windows. "
                    "seq_len=%d prefill_queries=%d fused_queries=%d",
                    seq_len,
                    num_queries,
                    stop - loop_stop,
                )
            if prefill_c8_v2_rows:
                selected = (
                    prefill_c8_v2_rows[0] if len(prefill_c8_v2_rows) == 1 else torch.cat(prefill_c8_v2_rows, dim=0)
                )
                context = attend_c8_v2(
                    query[start:stop],
                    cache,
                    block_table,
                    block_size,
                    selected,
                    seq_len,
                    self.learnable_sink_param,
                    self.scale,
                    compact_prefix=max(
                        0, min(num_queries, self.indexer.budget * self.indexer.ratio - (seq_len - num_queries))
                    ),
                )
                if snapshot is not None:
                    snapshot[start:stop, : selected.size(-1)].copy_(selected)
                value = _head_bmm(context, wv)
                write_output(start, stop, value)
                logger.info_once(
                    "HYV4 Prefill used aclnnKvQuantSparseFlashAttentionV2: queries=%d topk=%d seq_len=%d",
                    num_queries,
                    selected.size(-1),
                    seq_len,
                )
            if batch_sparse:
                selected = torch.cat(candidate_rows, dim=0)
                valid_indices = selected.clamp_min(0)
                if shared_kv is None:
                    kv_selected = self.ops.gather_c8_rows(cache, meta.block_table[i], block_size, valid_indices)
                else:
                    kv_selected = shared_kv.index_select(0, valid_indices.flatten()).view(*selected.shape, -1)
                if snapshot is not None:
                    snapshot[start:stop, : selected.shape[-1]].copy_(selected)
                value = self._attend_decoded(query[start:stop], selected, kv_selected, wv, split_pv=True)
                write_output(start, stop, value)

        if parts:
            # Keep the projection output directly when it covers the batch.
            # If several adjacent pieces cover it, materialize the final layout once.
            cursor = 0
            for start, stop, _ in parts:
                if start != cursor:
                    break
                cursor = stop
            else:
                if cursor == out.shape[0]:
                    if len(parts) == 1:
                        return parts[0][2]
                    if len(parts) <= 32:
                        return torch.cat([value for _, _, value in parts], dim=0)
            for start, stop, value in parts:
                out[start:stop].copy_(value)
        return out

    def _forward_decode_graph(self, meta, cache, block_size, hidden, q_c, positions, query, wv, out):
        seq_lens = meta.decode_seq_lens
        num_tokens = query.shape[0]
        num_queries = num_tokens // seq_lens.shape[0]
        geometry = {}
        if self.use_pd_decode and query.device.type == "npu":
            # Cache only read-only tensors within this forward and stream.
            context = get_forward_context()
            stream = torch.npu.current_stream(query.device).npu_stream
            key = (id(meta), num_tokens, query.device, stream)
            geometry = context.additional_kwargs.setdefault("hy4_decode_geometry", {}).setdefault(key, {})
        layout = geometry.get("logical")
        if layout is None:
            live = meta.slot_mapping[:num_tokens].view(-1, num_queries) >= 0
            seq_lens = seq_lens.long().masked_fill(~live[:, 0], 0)
            offsets = torch.arange(num_queries, device=query.device)
            logical = (seq_lens[:, None] - num_queries + offsets).masked_fill(~live, -1)
            layout = geometry["logical"] = (live, seq_lens, offsets, logical)
        else:
            logger.info_once("HYV4 Flash reuses Decode query geometry within a forward.")
        live, seq_lens, offsets, logical = layout
        snapshot = self.topk_snapshot
        window = min(self.window, self.max_model_len)
        shared_positions = None
        if (
            self.share_swa_c8_reads
            and self.indexer is None
            and num_queries == BATCH_VERIFY_MAX_QUERIES
            and window == BATCH_VERIFY_SWA_SHAPE[1]
            and (self.use_pd_decode or self.local_heads == BATCH_VERIFY_SWA_SHAPE[0])
            and cache.shape[-1] % torch.int32.itemsize == 0
        ):
            key = ("window_positions", window)
            shared_positions = geometry.get(key)
            if shared_positions is None:
                # The union of consecutive windows has window + num_queries - 1 rows.
                history_offsets = torch.arange(1 - window - num_queries, 0, device=query.device)
                shared_positions = seq_lens[:, None] + history_offsets
                shared_positions.masked_fill_(shared_positions > logical.amax(dim=1, keepdim=True), -1)
                geometry[key] = shared_positions

        batch_size = seq_lens.shape[0]
        if (
            shared_positions is not None
            and self.use_pd_decode
            and self.use_fp8_cache
            and query.device.type == "npu"
            and self.local_heads == 64
            and query.shape[-1] == 576
            and batch_size in (2, 3, 4, 5, 8)
        ):
            logger.info_once("HYV4 Flash batches sliding-window Decode attention across requests.")
            if batch_size == 8:
                logger.info_once("HYV4 Flash batches sliding-window Decode attention for eight requests.")
            key = ("window_selection", window)
            selection = geometry.get(key)
            if selection is None:
                selected = window_candidates(logical.flatten(), window)
                window_rows = (offsets[:, None] + torch.arange(window, device=query.device)).flatten()
                selection = geometry[key] = (selected, window_rows)
            selected, window_rows = selection
            if snapshot is not None:
                snapshot[:num_tokens, :window].copy_(selected)
            rows = torch.empty((num_tokens, window, query.shape[-1]), dtype=torch.float32, device=query.device)
            for request in range(batch_size):
                history = self._gather_attention_rows(
                    cache, meta.block_table[request], block_size, shared_positions[request], mask_padding=True
                )
                start = request * num_queries
                target = rows[start : start + num_queries].view(-1, rows.shape[-1])
                torch.index_select(history, 0, window_rows, out=target)
            self._attend_decoded(query, selected, rows, wv, out[:num_tokens])
            out[:num_tokens].masked_fill_(~live.reshape(-1, 1), 0)
            return

        score_groups = self.indexer is not None and self.max_model_len > (self.indexer.budget + 1) * self.indexer.ratio
        use_paged_pool = False
        if score_groups:
            index_query, index_weight = self.indexer.encode_queries(
                hidden[:num_tokens], q_c[:num_tokens], positions[..., :num_tokens], full_graph=True
            )
            use_paged_pool = (
                self.fuse_indexpool_select
                and self.incremental_index_pool
                and (self.use_pd_decode or self.local_heads == BATCH_VERIFY_SPARSE_SHAPE[0])
                and (num_queries, self.indexer.num_heads, self.indexer.ratio, self.indexer.budget)
                == PAGED_POOL_DECODE_SHAPE
                and cache.device.type == "npu"
                and meta.block_table.stride(-1) == 1
                and index_query.is_contiguous()
                and index_weight.is_contiguous()
            )
        batch_sparse = (
            use_paged_pool
            and self.use_pd_decode
            and self.use_fp8_cache
            and self.ops is triton_ops
            and cache.shape[-1] % torch.int32.itemsize == 0
            and self.local_heads == 32
            and query.shape[-1] == 576
            and (self.indexer.budget + 1) * self.indexer.ratio == 2048
            and batch_size in (2, 4, 8)
        )
        if batch_sparse:
            logger.info_once("HYV4 Flash batches sparse Decode attention across requests.")
            sparse_selected = []
        batched_selected = None
        if use_paged_pool and self.use_pd_decode and self.ops is triton_ops and batch_size in (2, 4, 8):
            logger.info_once("HYV4 Flash batches Decode IndexPool selection across requests.")
            if batch_size == 8:
                logger.info_once("HYV4 Flash batches Decode IndexPool selection for eight requests.")
            ratio = self.indexer.ratio
            budget = self.indexer.budget
            max_groups = self.max_model_len // ratio
            scores = torch.empty((batch_size, num_queries, max_groups), dtype=torch.float32, device=query.device)
            for request in range(batch_size):
                start = request * num_queries
                stop = start + num_queries
                table = meta.block_table[request]
                self._update_decode_pool(cache, table, block_size, seq_lens[request], num_queries, positions.ndim)
                write_pool_scores(
                    index_query[start:stop],
                    index_weight[start:stop],
                    cache,
                    table,
                    logical[request],
                    block_size,
                    max_groups,
                    ratio,
                    budget,
                    scores[request],
                )
            batched_selected = select_pool_candidates(
                scores.flatten(0, 1), logical.flatten(), max_groups, ratio, budget
            ).view(batch_size, num_queries, -1)
        for request in range(seq_lens.shape[0]):
            start = request * num_queries
            stop = start + num_queries
            table = meta.block_table[request]
            visible = logical[request]
            if self.indexer is None:
                selected = window_candidates(visible, window)
            else:
                if self.incremental_index_pool and batched_selected is None:
                    self._update_decode_pool(cache, table, block_size, seq_lens[request], num_queries, positions.ndim)
                if score_groups:
                    ratio = self.indexer.ratio
                    if batched_selected is not None:
                        selected = batched_selected[request]
                    elif use_paged_pool:
                        logger.info_once("HYV4 Flash uses paged IndexPool scoring for six verification queries.")
                        selected = self.ops.select_paged_pool(
                            index_query[start:stop],
                            index_weight[start:stop],
                            cache,
                            table,
                            visible,
                            block_size,
                            self.max_model_len // ratio,
                            ratio,
                            self.indexer.budget,
                        )
                    else:
                        pool = self._read_decode_pool(cache, table, block_size, seq_lens[request], positions.ndim)
                        selected = sparse_candidates(
                            index_query[start:stop], index_weight[start:stop], pool, visible, ratio, self.indexer.budget
                        )
                    dense = self._dense_decode_candidates(visible, selected.shape[-1])
                    selected = torch.where((visible // ratio > self.indexer.budget)[:, None], selected, dense)
                else:
                    selected = self._dense_decode_candidates(visible, self.max_model_len)
            if snapshot is not None:
                snapshot[start:stop, : selected.shape[-1]].copy_(selected)
            if batch_sparse:
                sparse_selected.append(selected)
                continue
            split_pv = (
                self.batch_verify_attention
                and num_queries > 1
                and (self.local_heads, selected.shape[-1]) == BATCH_VERIFY_SPARSE_SHAPE
            )
            q = query[start:stop]
            scores = None
            if (
                self.indexer is not None
                and self.fuse_c8_qk
                and split_pv
                and num_queries == BATCH_VERIFY_MAX_QUERIES
                and q.dtype == torch.bfloat16
                and q.is_contiguous()
                and cache.device.type == "npu"
                and cache.shape[-1] % 4 == 0
                and table.is_contiguous()
            ):
                if self.use_cann_prefill:
                    logger.info_once("HYV4 Flash uses CANN paged C8 reads and FP32 QK for sparse verification.")
                else:
                    logger.info_once("HYV4 Flash fuses paged C8 reads and FP32 QK for sparse verification.")
                scores, rows = self.ops.gather_c8_qk(q, cache, table, block_size, selected)
            elif shared_positions is None:
                rows = self._gather_attention_rows(cache, table, block_size, selected, mask_padding=True)
            else:
                history = self._gather_attention_rows(
                    cache, table, block_size, shared_positions[request], mask_padding=True
                )
                rows = history.unfold(0, window, 1).transpose(1, 2)
            self._attend_decoded(q, selected, rows, wv, out[start:stop], split_pv=split_pv, scores=scores)
        if batch_sparse:
            logger.info_once("HYV4 Flash gathers sparse Decode C8 cache rows for the whole rank with CANN.")
            selected = torch.stack(sparse_selected)
            logical = selected.clamp_min(0).flatten(1)
            packed = self._gather(
                cache, meta.block_table[:batch_size], block_size, logical, slice(self.kv_storage_width)
            )
            selected = selected.flatten(0, 1)
            rows = dequantize_c8_rows(packed).view(num_tokens, selected.shape[-1], query.shape[-1])
            rows = rows.masked_fill(selected[..., None] < 0, 0)
            self._attend_decoded(query, selected, rows, wv, out[:num_tokens])
        if shared_positions is not None:
            # Inactive queries may overlap live KV rows in the unfolded view.
            out[:num_tokens].masked_fill_(~live.reshape(-1, 1), 0)

    @staticmethod
    def _dense_decode_candidates(visible, width):
        history = torch.arange(width, device=visible.device).expand(visible.shape[0], -1)
        return history.masked_fill(history > visible[:, None], -1)

    def _update_decode_pool(self, cache, table, block_size, seq_len, num_queries, position_ndim):
        ratio = self.indexer.ratio
        first = (seq_len - num_queries).clamp_min(0) // ratio
        end = seq_len // ratio
        # A width-q decode can close at most ceil(q / ratio) groups.
        width = ((num_queries + ratio - 1) // ratio) * ratio
        history = first * ratio + torch.arange(width, device=seq_len.device)
        history = history.masked_fill(history >= end * ratio, 0)
        raw, key_pos = self._read_index_rows(cache, table, block_size, history, position_ndim)
        keys = self.indexer.encode_keys(raw, key_pos, full_graph=True)
        self.ops.copy_pool_keys(cache, table, keys, first, block_size, ratio, write=True, end_group=end)

    def _read_decode_pool(self, cache, table, block_size, seq_len, position_ndim):
        ratio = self.indexer.ratio
        groups = self.max_model_len // ratio
        end = seq_len // ratio
        if self.incremental_index_pool:
            keys = torch.empty((groups, INDEX_DIM), dtype=torch.float32, device=cache.device)
            self.ops.copy_pool_keys(cache, table, keys, 0, block_size, ratio, write=False, end_group=end)
            return keys
        history = torch.arange(groups * ratio, device=cache.device)
        history = history.masked_fill(history >= end * ratio, 0)
        raw, key_pos = self._read_index_rows(cache, table, block_size, history, position_ndim)
        return self.indexer.encode_keys(raw, key_pos, full_graph=True)

    def _gather_attention_rows(self, cache, block_table, block_size, positions, *, mask_padding=False):
        # Keep small reads fused, including shared sliding-window histories.
        # Large decode reads use the existing bulk path; negative indices still
        # go through its clamp-and-mask handling below.
        use_bulk_decode_read = (
            cache.device.type == "npu"
            and self.use_pd_decode
            and self.bulk_c8_decode_reads
            and self.ops is triton_ops
            and positions.numel() > MAX_FUSED_C8_READ_ROWS
        )
        if (
            self.use_fp8_cache
            and (self.use_cann_prefill or HAS_TRITON)
            and cache.device.type == "npu"
            and cache.shape[-1] % 4 == 0
            and (mask_padding or positions.numel() <= MAX_FUSED_C8_READ_ROWS)
            and not use_bulk_decode_read
        ):
            return self.ops.gather_c8_rows(cache, block_table, block_size, positions)
        logical = positions.clamp_min(0) if mask_padding else positions
        rows = self._gather(cache, block_table, block_size, logical, slice(self.kv_storage_width))
        rows = dequantize_c8_rows(rows) if self.use_fp8_cache else rows
        return rows.masked_fill(positions[..., None] < 0, 0) if mask_padding else rows

    def _read_index_rows(self, cache, block_table, block_size, logical, position_ndim):
        index_columns = slice(self.kv_storage_width, self.token_storage_width)
        rows = self._gather(cache, block_table, block_size, logical, index_columns)
        if self.use_fp8_cache:
            rows = rows.view(torch.bfloat16)
        coords = rows[:, INDEX_DIM:].contiguous().view(torch.int64)
        return rows[:, :INDEX_DIM], coords[:, 0] if position_ndim == 1 else coords.T

    def _read_partial_index_group(
        self,
        cache,
        table,
        block_size,
        first_token,
        end_token,
        position_ndim,
        *,
        prefill,
    ):
        if (
            prefill
            and self.use_cann_prefill
            and self.use_fp8_cache
            and 0 <= first_token < end_token <= table.numel() * block_size
        ):
            from .prefill_ops.index_pool_history import read_index_history
            from .prefill_ops.paged_cache_copy import supports_paged_cache_copy

            if supports_paged_cache_copy(cache, table, block_size):
                return read_index_history(
                    cache,
                    table,
                    block_size,
                    first_token,
                    end_token - first_token,
                    self.kv_storage_width,
                    position_ndim,
                )
        logical = torch.arange(first_token, end_token, device=cache.device)
        return self._read_index_rows(cache, table, block_size, logical, position_ndim)

    def _update_index_pool(self, cache, block_size, meta, keys, positions):
        ratio = self.indexer.ratio
        segments = []
        eager_prefill = self.use_cann_prefill and bool(meta.prefill_reqs and all(meta.prefill_reqs))
        for request, seq_len in enumerate(meta.seq_lens):
            start, stop = meta.query_start_loc[request : request + 2]
            if start == stop:
                continue
            context_len = seq_len - (stop - start)
            first_group, end_group = context_len // ratio, seq_len // ratio
            if first_group == end_group:
                continue
            # A partial group starts in the cache and ends in the current batch.
            first_token = first_group * ratio
            past = max(0, context_len - first_token)
            first_new = start + max(0, first_token - context_len)
            segments.append((request, first_token, past, first_new, (end_group - first_group) * ratio))
        if not segments:
            return

        _, _, first_past, first_new, _ = segments[0]
        shared = eager_prefill and len(segments) == 1 and first_past == 0
        if (
            # Limit the mapped path to partial groups. Computing a full shared
            # factor table on a cache miss can add latency to a no-history batch.
            eager_prefill
            and any(past for _, _, past, _, _ in segments)
            and self.indexer.use_eager_fp8_encoding
            and self.use_fp8_cache
            and ratio == 4
            and self.indexer.rope_dim == 64
            and keys.shape[1] == INDEX_DIM == 128
            and keys.dtype is torch.bfloat16
            and keys.is_contiguous()
            and keys.device == cache.device == positions.device
            and positions.dtype == torch.int64
            and positions.ndim in (1, 2)
            and (positions.ndim == 1 or (positions.shape[0] == 3 and self.indexer.sections))
            and sum(count // ratio for _, _, _, _, count in segments) <= 4096
            and self.kv_storage_width % 8 == 0
            and cache.storage_offset() % 8 == 0
            and all(
                new + count - past <= min(keys.shape[0], positions.shape[-1])
                and (not past or first + past <= meta.block_table.shape[-1] * block_size)
                for _, first, past, new, count in segments
            )
        ):
            from .prefill_ops.paged_cache_copy import supports_paged_cache_copy

            assemble = supports_paged_cache_copy(cache, meta.block_table[segments[0][0]], block_size)
        else:
            assemble = False

        if assemble:
            from .prefill_ops.index_pool_assembly import assemble_index_pool_groups

            raw, group_map, history_positions = assemble_index_pool_groups(
                cache,
                meta.block_table,
                block_size,
                self.kv_storage_width,
                keys,
                positions.ndim,
                segments,
            )
            pooled = self.indexer.encode_keys(
                raw,
                history_positions,
                eager_prefill=True,
                shared_positions=positions,
                group_map=group_map,
                # Layer sidecars record identical positions for these request
                # tokens within the same KV cache group and forward step.
                history_context_key=(id(meta.block_table), block_size, tuple(segments)),
            )
        else:
            raw_keys, key_positions = [], []
            for request, first_token, past, new, rows in segments:
                end = new + rows - past
                current_keys, current_positions = keys[new:end], positions[..., new:end]
                if past:
                    previous_keys, previous_positions = self._read_partial_index_group(
                        cache,
                        meta.block_table[request],
                        block_size,
                        first_token,
                        first_token + past,
                        positions.ndim,
                        prefill=eager_prefill,
                    )
                    current_keys = torch.cat((previous_keys, current_keys), dim=0)
                    current_positions = torch.cat((previous_positions, current_positions), dim=-1)
                raw_keys.append(current_keys)
                key_positions.append(current_positions)
            if eager_prefill:
                raw = raw_keys[0] if len(raw_keys) == 1 else torch.cat(raw_keys)
                key_pos = key_positions[0] if len(key_positions) == 1 else torch.cat(key_positions, dim=-1)
                pooled = self.indexer.encode_keys(
                    raw,
                    key_pos,
                    eager_prefill=True,
                    shared_positions=positions if shared else None,
                    position_start=first_new if shared else 0,
                )
            else:
                pooled = self.indexer.encode_keys(torch.cat(raw_keys), torch.cat(key_positions, dim=-1))

        offset = 0
        for request, first_token, _, _, rows in segments:
            count = rows // ratio
            self.ops.copy_pool_keys(
                cache,
                meta.block_table[request],
                pooled[offset : offset + count],
                first_token // ratio,
                block_size,
                ratio,
                write=True,
            )
            offset += count

    def _pack_kv(self, latent, rope):
        if not self.use_fp8_cache:
            return torch.cat((latent, rope), dim=-1)
        payload, scales = self._quantize_c8(latent, gpu_scale=self.use_gpu_c8_scale)
        return pack_c8_components(payload, rope.contiguous(), scales)

    @staticmethod
    def _quantize_c8(latent, *, gpu_scale=False):
        if gpu_scale:
            # Match the CUDA fp8_ds_mla writer's power-of-two tile scale.
            tiles = latent.float().unflatten(-1, (-1, SFA_QSFA_TILE_SIZE))
            amax = tiles.abs().amax(-1)
            scales = (amax / C8_E4M3_MAX).clamp_min(C8_GPU_MIN_SCALE).log2().ceil().exp2()
            payload = (tiles / scales.unsqueeze(-1)).clamp(-C8_E4M3_MAX, C8_E4M3_MAX).to(torch.float8_e4m3fn)
            return payload.flatten(-2).unsqueeze(1).contiguous(), scales.unsqueeze(1).contiguous()
        return torch_npu.npu_dynamic_block_quant(
            latent.contiguous().unsqueeze(1),
            dst_type=torch.float8_e4m3fn,
            row_block_size=1,
            col_block_size=SFA_QSFA_TILE_SIZE,
        )

    def _kv_b_weights(self):
        weight = self.kv_b_proj.weight.data
        if torch_npu.get_npu_format(weight) != ACL_FORMAT_ND:
            weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_ND)
        return weight.reshape(self.local_heads, self.nope_dim + self.v_dim, self.latent_dim).split(
            (self.nope_dim, self.v_dim), dim=1
        )

    def process_weights_after_loading(self):
        if not (self.use_pd_decode or self.use_cann_prefill) or self.kv_b_proj.weight.device.type != "npu":
            return
        wk, wv = self._kv_b_weights()
        wv = wv.transpose(1, 2)
        if self._decode_key_projection is None:
            self._decode_key_projection = wk.contiguous()
            self._decode_value_projection = wv.contiguous()
            logger.info_once("HYV4 Flash prepacked KV projection weights.")
        else:
            # Captured graphs retain these addresses across weight reloads.
            self._decode_key_projection.copy_(wk)
            self._decode_value_projection.copy_(wv)

    def _absorb_query(self, q_nope, q_pe, *, absorbed=False):
        wk = self._decode_key_projection
        wv = self._decode_value_projection
        if wk is None:
            wk, wv = self._kv_b_weights()
            wk = wk if absorbed else wk.contiguous()
            wv = wv.transpose(1, 2).contiguous()
        q_latent = q_nope if absorbed else _head_bmm(q_nope, wk)
        return torch.cat((q_latent, q_pe), dim=-1), wv

    def _attend(self, query, selected, kv_selected, wv, out):
        if self.use_fp8_cache:
            kv_selected = dequantize_c8_rows(kv_selected)
        self._attend_decoded(query, selected, kv_selected, wv, out)

    def _attend_decoded(
        self, query, selected, kv_selected, wv, out=None, *, split_pv=False, scores=None, query_fp32=None,
        fuse_prefill_logits=False,
    ):
        attention_query = query if query_fp32 is None else query_fp32
        if fuse_prefill_logits:
            from .attention_logits import build_mla_logits

            scores = torch.bmm(attention_query.float(), kv_selected.float().transpose(1, 2))
            if torch_npu.get_npu_format(scores) != ACL_FORMAT_ND:
                scores = torch_npu.npu_format_cast(scores, ACL_FORMAT_ND)
            logits = build_mla_logits(scores, selected, self.learnable_sink_param, self.scale)
            # Keep the same FP32 softmax and PV reductions as the eager path.
            probs = logits.softmax(dim=-1)[..., :-1]
        elif scores is not None or (
            self.use_triton_mla_logits and query.device.type == "npu" and query.shape[0] == 1 and selected.shape[-1] > 0
        ):
            if scores is None:
                scores = torch.bmm(attention_query.float(), kv_selected.float().transpose(1, 2))
                if torch_npu.get_npu_format(scores) != ACL_FORMAT_ND:
                    scores = torch_npu.npu_format_cast(scores, ACL_FORMAT_ND)
            logits = self.ops.build_mla_logits(scores, selected, self.learnable_sink_param, self.scale)
            # Use the reference FP32 softmax reduction before discarding the sink.
            probs = logits.softmax(dim=-1)[..., :-1]
        else:
            probs = mla_attention_probs(attention_query, kv_selected, selected, self.learnable_sink_param, self.scale)
        values = kv_selected[..., : self.latent_dim].float()
        if split_pv:
            # The batched FP32 PV kernel changes the 2048-wide reduction order.
            context = torch.cat([torch.bmm(p, v) for p, v in zip(probs.split(1), values.split(1))])
        else:
            context = torch.bmm(probs, values)
        context = context.to(query.dtype)
        value = _head_bmm(context, wv)
        if out is None:
            return value
        out.copy_(value.reshape_as(out))

    @staticmethod
    def _gather(cache, block_table, block_size, positions, columns: slice):
        # Gather from dense rows before slicing fields. A column slice of the
        # full cache can cause torch-npu to materialize every page for each read.
        pages = positions // block_size
        if block_table.ndim == 1:
            blocks = block_table.index_select(0, pages.flatten()).view_as(positions)
        else:
            blocks = block_table.gather(1, pages)
        slots = blocks.long() * block_size + positions % block_size
        rows = cache.index_select(0, slots.flatten())[:, columns].contiguous()
        return rows.view(*positions.shape, rows.shape[-1])
