# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend MLA and IndexPool with cache pages owned by the vLLM scheduler."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import torch
import torch_npu
from torch import nn
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import make_attention_sink
from vllm.model_executor.models.extract_hidden_states import (
    CacheOnlyAttentionBackend,
    CacheOnlyAttentionImpl,
)
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION
from vllm.models.hy_v4_flash.reference import (
    INDEX_DIM,
    indexpool_fp8,
    mla_attention_probs,
    pooled_keys,
)
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec, MLAAttentionSpec, SlidingWindowMLASpec

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE
from vllm_ascend.device.device_config import get_ascend_device_type, is_950
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows, pack_c8_components
from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm_ascend.quantization.methods.base import PreparedLinearInput
from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockLinearMethod
from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg
from vllm_ascend.utils import attention_calculation_stream, npu_stream_switch

from . import ops as triton_ops
from .attention_decode import DecodeAttentionMixin
from .attention_eager import EagerAttentionMixin
from .attention_ops import (
    ACL_FORMAT_ND,
    BF16_QUERY_PROJECTION_LAYER,
    MAX_FUSED_C8_READ_ROWS,
    MAX_FUSED_C8_WRITE_TOKENS,
    MAX_FUSED_INDEXPOOL_ROWS,
    MAX_FUSED_OUTPUT_GATE_ELEMENTS,
    POSITION_LANES,
    _head_bmm,
    fused_output_gate,
    is_decode_batch,
    native_indexpool_rope,
    native_mla_context,
    native_rms_norm,
    reuse_prefill_metadata,
    uses_decode_fusions,
)
from .cache import PACKED_CACHE_FORMAT
from .config import uses_prefill_ops

logger = init_logger_ascend(__name__)


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
    if not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    batch = metadata.get(layer_name) if layer_name is not None else next(iter(metadata.values()), None)
    return batch if isinstance(batch, FlashAttentionMetadata) else None


class FlashMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.precise_decode = vllm_config.speculative_config is not None

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
    supports_updatable_graph = True
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
    from .prefill_ops.indexer import round_prefill_fp8_rows

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
                    input_size,
                    output_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps)
        self.register_buffer("_decode_inverse_frequency", None, persistent=False)
        self.register_buffer("_decode_rope_axis", None, persistent=False)
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
            from .prefill_ops.indexer import apply_factors, make_factors

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
                    vectors,
                    cache[key][1],
                    group_map=group_map,
                    history_factors=history_factors,
                    normalization=normalization,
                )
            return apply_factors(vectors, cache[key][1], position_start, position_step, normalization=normalization)

        if normalization is not None:
            weight, multiplier = normalization
            vectors = ((vectors * multiplier[:, None]) * weight).bfloat16()
        decode = is_decode_batch(get_flash_attention_metadata())
        inverse = self._decode_inverse_frequency if decode else None
        if inverse is None:
            inverse = self.theta ** (
                -torch.arange(0, self.rope_dim, 2, device=vectors.device, dtype=torch.float32) / self.rope_dim
            )
            if decode:
                self._decode_inverse_frequency = inverse
        phase = positions.float().unsqueeze(-1) * inverse
        if positions.ndim == 2:
            if self.interleaved and decode:
                axis = self._decode_rope_axis
                if axis is None:
                    dims = torch.arange(self.rope_dim // 2, device=vectors.device)
                    axis = torch.zeros_like(dims)
                    axis = torch.where((dims % 3 == 1) & (dims < 3 * self.sections[1]), 1, axis)
                    axis = torch.where((dims % 3 == 2) & (dims < 3 * self.sections[2]), 2, axis)
                    self._decode_rope_axis = axis
                phase = torch.gather(phase, 0, axis.view(1, 1, -1).expand(1, phase.shape[1], -1)).squeeze(0)
            elif self.interleaved:
                from vllm.model_executor.layers.rotary_embedding.mrope import (
                    apply_interleaved_rope,
                )

                phase = apply_interleaved_rope(phase, self.sections)
            else:
                parts = phase.split(self.sections, dim=-1)
                phase = torch.cat([part[axis] for axis, part in enumerate(parts)], dim=-1)
        if vectors.ndim == 3:
            phase = phase.unsqueeze(1)
        nope, rotary = vectors.split((INDEX_DIM - self.rope_dim, self.rope_dim), dim=-1)
        rotary_fp32 = rotary.float()
        even, odd = rotary_fp32[..., ::2], rotary_fp32[..., 1::2]
        cos, sin = phase.cos(), phase.sin()
        if is_decode_batch(get_flash_attention_metadata()) and vectors.device.type == "npu" and is_950():
            return native_indexpool_rope(nope, even, odd, cos, sin, vectors.dtype)
        return torch.cat(
            (
                nope,
                (even * cos - odd * sin).to(vectors.dtype),
                (odd * cos + even * sin).to(vectors.dtype),
            ),
            dim=-1,
        )

    def _encode_fp8(self, vectors, *, full_graph, output_dtype=torch.float32):
        if is_decode_batch(get_flash_attention_metadata()):
            from .indexer import encode_indexpool_fp8_fused

            return encode_indexpool_fp8_fused(vectors)
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
        decode_pool = is_decode_batch(get_flash_attention_metadata())
        if decode_pool:
            from .indexer import pool_norm

            if not hasattr(self, "_decode_pool_projection"):
                self._decode_pool_projection = torch.cat(
                    (self.key_value_projection.weight, self.key_gate_projection.weight),
                    0,
                ).contiguous()
            pooled = pool_norm(
                raw,
                self._decode_pool_projection,
                self.intra_group_ape,
                self.key_norm.weight,
                self.eps,
            )
        normalization = None
        if (
            not decode_pool
            and HAS_TRITON
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
            from .prefill_ops.indexer import pool_with_inverse_norm

            raw_fp32 = raw.float()
            values = self.key_value_projection(raw_fp32)
            probabilities = (self.key_gate_projection(raw_fp32) + self.intra_group_ape).softmax(dim=1)
            if values.is_contiguous() and probabilities.is_contiguous():
                pooled, inverse_norm = pool_with_inverse_norm(values, probabilities, self.eps)
                normalization = (self.key_norm.weight, inverse_norm)
        if not decode_pool and normalization is None:
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


class AscendHYV4FlashAttention(DecodeAttentionMixin, EagerAttentionMixin, nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, config, layer_idx, prefix, *, reuse_target=False):
        super().__init__()
        self.use_cann_prefill = uses_prefill_ops(vllm_config) and not reuse_target
        transfer = vllm_config.kv_transfer_config
        self.use_pd_decode = transfer is not None and transfer.kv_role == "kv_consumer"
        self._short_indexpool_capture = False
        self.native_decode_attention = not reuse_target and uses_decode_fusions(vllm_config)
        self.decode_data_fusions = self.native_decode_attention and HAS_TRITON
        self.bulk_c8_decode_reads = True
        self.cann_c8_decode = vllm_config.additional_config.get("hy4flash_cann_c8_decode", False)
        self.cann_c8_swa_decode = vllm_config.additional_config.get("hy4flash_cann_c8_swa_decode", False)
        self.cann_indexer_decode = vllm_config.additional_config.get("hy4flash_cann_indexer_decode", False)
        if not all(
            isinstance(value, bool)
            for value in (self.cann_c8_decode, self.cann_c8_swa_decode, self.cann_indexer_decode)
        ):
            raise ValueError("HY4 experimental CANN Decode options must be boolean")
        self.decode_index_pages = None
        self.ops = triton_ops
        if self.use_cann_prefill:
            from . import indexer as li_prefill_p0
            from . import prefill_ops

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
        self.use_eager_prefill_rotary = (
            self.use_cann_prefill
            and use_a5_c8
            and vllm_config.model_config.enforce_eager
            and hasattr(torch_npu, "npu_apply_rotary_pos_emb")
        )
        self.batch_verify_attention = use_a5_c8
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
            hidden,
            config.q_lora_rank,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_a_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden,
            self.kv_width,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
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
            hidden,
            heads * self.v_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.linear_gate",
        )
        self.o_proj = RowParallelLinear(
            heads * self.v_dim,
            hidden,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.o_proj",
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
        if self.use_eager_prefill_rotary and HAS_TRITON and getattr(self.rotary_emb, "mrope_interleaved", False):
            from .prefill_ops.fusions import mrope_channel_bits

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
        if not envs.VLLM_USE_V2_MODEL_RUNNER or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
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
        scheme = getattr(self.q_b_proj.quant_method, "quant_method", None)
        if (
            is_decode_batch(get_flash_attention_metadata())
            and is_950()
            and q_a.device.type == "npu"
            and q_a.dtype == self.q_a_layernorm.weight.dtype == torch.bfloat16
            and hasattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant")
            and isinstance(scheme, AscendFp8BlockLinearMethod)
            and scheme.use_mxfp8
            and scheme.mxfp8_method.group_size == 32
            and not any(getattr(self.q_b_proj, "mxfp8_tp_padding", (0, 0)))
        ):
            norm = self.q_a_layernorm
            quantized, scale, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                q_a,
                norm.weight,
                epsilon=norm.variance_epsilon,
                dst_type=torch.float8_e4m3fn,
                scale_alg=scheme.mxfp8_method.dynamic_mx_quant_scale_alg,
            )
            q_c = (
                torch_npu.npu_rms_norm(q_a, norm.weight, norm.variance_epsilon)[0]
                if (
                    self.indexer is not None
                    and not self._short_indexpool_capture
                    and self.max_model_len > (self.indexer.budget + 1) * self.indexer.ratio
                )
                else None
            )
            return self.q_b_proj(PreparedLinearInput(quantized, scale, q_a.dtype))[0], q_c
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
                        from .prefill_ops.fusions import gather_mrope_factors

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
        if self.native_decode_attention and is_decode_batch(get_flash_attention_metadata()):
            return self._forward_native_decode(positions, hidden_states)
        return self._forward_existing(positions, hidden_states)

    def _forward_native_decode(self, positions, hidden_states):
        q, q_c = self._project_query(hidden_states)
        q = q.reshape(-1, self.local_heads, self.nope_dim + self.rope_dim)
        kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split((self.latent_dim, self.rope_dim), -1)
        kv_c = self.kv_a_layernorm(kv_c)
        q_nope, q_pe = q.split((self.nope_dim, self.rope_dim), dim=-1)
        q_pe, k_pe = self._rotate_qk(positions, q_pe.reshape(q.shape[0], -1).contiguous(), k_pe.contiguous())
        q_pe = q_pe.reshape(-1, self.local_heads, self.rope_dim)
        needs_meta = self.use_cann_prefill or self.fuse_attn_output_gate or self.overlap_attn_output_gate
        meta = get_flash_attention_metadata() if needs_meta else None
        full_decode = meta is not None and meta.decode_seq_lens is not None
        full_prefill = (
            self.use_cann_prefill
            and meta is not None
            and not full_decode
            and bool(meta.prefill_reqs)
            and all(meta.prefill_reqs)
            and meta.query_start_loc[0] == 0
            and meta.query_start_loc[-1] == q.shape[0] > 0
        )
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

        self._forward_paged(
            hidden_states,
            q_c,
            q_nope,
            q_pe,
            kv_c,
            k_pe,
            positions,
            out,
        )
        if overlap_gate:
            main_stream.wait_stream(gate_stream)
            gate.record_stream(main_stream)
        else:
            gate = self._project_output_gate(hidden_states, fuse_gate)

        if fuse_gate:
            gate = torch_npu.npu_format_cast(gate, ACL_FORMAT_ND).contiguous()
            scheme = getattr(self.o_proj.quant_method, "quant_method", None)
            if (
                isinstance(scheme, AscendFp8BlockLinearMethod)
                and scheme.use_mxfp8
                and scheme.mxfp8_method.group_size == 32
                and not any(getattr(self.o_proj, "mxfp8_tp_padding", (0, 0)))
            ):
                quantized, scale = triton_ops.gate_dynamic_mx_quant(out, gate)
                output = self.o_proj(PreparedLinearInput(quantized, scale, out.dtype))[0]
            else:
                output = self.o_proj(fused_output_gate(out, gate))[0]
        else:
            output = self.o_proj(out * gate)[0]
        if self.sequence_parallel_output:
            output = sp_reduce_scatter(output)
        return self.post_proj_norm(output) if self.post_proj_norm is not None else output

    def _forward_existing(self, positions, hidden_states):
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
            from .prefill_ops.fusions import pack_attention_projections

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
                hidden_states,
                q_c,
                q_nope,
                q_pe,
                kv_c,
                k_pe,
                positions,
                out,
                reuse_prefill_output=True,
            )
        else:
            self._forward_paged(
                hidden_states,
                q_c,
                q_nope,
                q_pe,
                kv_c,
                k_pe,
                positions,
                out,
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
        self,
        hidden,
        q_c,
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        positions,
        out,
        *,
        reuse_prefill_output=False,
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
        keys = self._write_cache_rows(
            cache,
            meta,
            hidden,
            kv_c,
            k_pe,
            positions,
            num_tokens,
            full_graph=full_graph,
        )
        query, wv = self._absorb_query(q_nope[:num_tokens], q_pe[:num_tokens])
        if (
            self.cann_indexer_decode
            and self.use_pd_decode
            and self.use_fp8_cache
            and self.indexer is not None
            and self.incremental_index_pool
            and self.fuse_indexpool_select
            and block_size == 64
            and (self.indexer.num_heads, self.indexer.ratio, self.indexer.budget) == (16, 4, 511)
            and cache.device.type == "npu"
            and self.decode_index_pages is None
        ):
            self.decode_index_pages = torch.zeros(
                (cache.shape[0] // block_size, block_size // self.indexer.ratio, 1, INDEX_DIM),
                dtype=torch.bfloat16,
                device=cache.device,
            )
        if full_graph:
            self._forward_decode_graph(meta, cache, block_size, hidden, q_c, positions, query, wv, out)
            return out

        if keys is not None and self.incremental_index_pool:
            self._update_index_pool(cache, block_size, meta, keys, positions)
        if self.decode_index_pages is not None:
            from .indexer import refresh_pages

            for request, length in enumerate(meta.seq_lens):
                if meta.query_start_loc[request + 1] > meta.query_start_loc[request]:
                    refresh_pages(cache, self.decode_index_pages, meta.block_table[request], block_size, length)
        return self._attend_eager(
            meta,
            cache,
            block_size,
            hidden,
            q_c,
            positions,
            query,
            wv,
            out,
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
                from .prefill_ops.indexer import token_key_norm_inverse

                index_norm = (norm.weight, token_key_norm_inverse(keys, norm.eps))
            else:
                keys = (native_rms_norm(keys, norm) if is_decode_batch(meta) else norm(keys)).bfloat16()
        if use_fused_write:
            payload, scales = self._quantize_c8(kv_c[:num_tokens])
            if index_norm is not None:
                return self.ops.write_c8_rows(
                    cache,
                    slots,
                    payload,
                    k_pe[:num_tokens],
                    scales,
                    keys,
                    positions[..., :num_tokens],
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
            from .prefill_ops.cache import supports_paged_cache_copy
            from .prefill_ops.indexer import read_index_history

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
            from .prefill_ops.cache import supports_paged_cache_copy

            assemble = supports_paged_cache_copy(cache, meta.block_table[segments[0][0]], block_size)
        else:
            assemble = False

        if assemble:
            from .prefill_ops.indexer import assemble_index_pool_groups

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
        payload, scales = self._quantize_c8(latent)
        return pack_c8_components(payload, rope.contiguous(), scales)

    @staticmethod
    def _quantize_c8(latent):
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

    def _absorb_query(self, q_nope, q_pe):
        wk = self._decode_key_projection
        wv = self._decode_value_projection
        if wk is None:
            wk, wv = self._kv_b_weights()
            wk = wk.contiguous()
            wv = wv.transpose(1, 2).contiguous()
        q_latent = _head_bmm(q_nope, wk)
        return torch.cat((q_latent, q_pe), dim=-1), wv

    def _attend(self, query, selected, kv_selected, wv, out):
        if self.use_fp8_cache:
            kv_selected = dequantize_c8_rows(kv_selected)
        self._attend_decoded(query, selected, kv_selected, wv, out)

    def _attend_decoded(
        self,
        query,
        selected,
        kv_selected,
        wv,
        out=None,
        *,
        split_pv=False,
        scores=None,
        query_fp32=None,
        fuse_prefill_logits=False,
    ):
        if getattr(self, "native_decode_attention", False) and is_decode_batch(get_flash_attention_metadata()):
            context = native_mla_context(
                query,
                selected,
                kv_selected,
                self.learnable_sink_param,
                self.scale,
                scores=scores,
                query_fp32=query_fp32,
            )
            value = _head_bmm(context, wv)
            if out is None:
                return value
            out.copy_(value.reshape_as(out))
            return
        attention_query = query if query_fp32 is None else query_fp32
        if fuse_prefill_logits:
            from .attention_ops import build_mla_logits

            scores = torch.bmm(attention_query.float(), kv_selected.float().transpose(1, 2))
            if torch_npu.get_npu_format(scores) != ACL_FORMAT_ND:
                scores = torch_npu.npu_format_cast(scores, ACL_FORMAT_ND)
            logits = build_mla_logits(scores, selected, self.learnable_sink_param, self.scale)
            # Keep the same FP32 softmax and PV reductions as the eager path.
            probs = logits.softmax(dim=-1)[..., :-1]
        elif scores is not None:
            logits = self.ops.build_mla_logits(scores, selected, self.learnable_sink_param, self.scale)
            # Use the reference FP32 softmax reduction before discarding the sink.
            probs = logits.softmax(dim=-1)[..., :-1]
        else:
            probs = mla_attention_probs(
                attention_query,
                kv_selected,
                selected,
                self.learnable_sink_param,
                self.scale,
            )
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
