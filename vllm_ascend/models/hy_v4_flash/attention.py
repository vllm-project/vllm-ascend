# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend MLA and IndexPool with cache pages owned by the vLLM scheduler."""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch_npu
from torch import nn
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
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows, pack_c8_components
from vllm_ascend.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm_ascend.quantization.methods.base import PreparedLinearInput
from vllm_ascend.quantization.methods.w8a8.fp8_block import AscendFp8BlockLinearMethod
from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg
from vllm_ascend.utils import attention_calculation_stream, npu_stream_switch

from .attention_logits import build_mla_logits
from .c8_read import gather_c8_qk, gather_c8_rows
from .c8_write import write_c8_rows
from .index_pool_cache import copy_pool_keys
from .index_pool_fp8 import encode_indexpool_fp8
from .index_pool_select import select_paged_pool
from .mla_prolog import CANNBotBf16MlaProlog
from .output_gate import gate_dynamic_mx_quant

logger = init_logger(__name__)

ACL_FORMAT_ND = 2
C8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
C8_GPU_MIN_SCALE = 1e-4
BF16_QUERY_PROJECTION_LAYER = 1
QUERY_CHUNK = 16
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


def get_flash_attention_metadata() -> FlashAttentionMetadata | None:
    metadata = get_forward_context().attn_metadata
    batch = next(iter(metadata.values()), None) if isinstance(metadata, dict) else None
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
        precise_decode_reqs = ()
        if self.precise_decode and meta.is_prefilling is not None:
            precise_decode_reqs = tuple(not prefill for prefill in meta.is_prefilling[: meta.num_reqs].tolist())
        return FlashAttentionMetadata(
            query_starts,
            tuple(meta.seq_lens_cpu[: meta.num_reqs].tolist()),
            meta.block_table_tensor,
            meta.slot_mapping,
            precise_decode_reqs,
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


class FlashIndexPool(nn.Module):
    def __init__(self, config, quant_config, rope_parameters, prefix, *, fuse_fp8_encoding=False):
        super().__init__()
        self.fuse_fp8_encoding = fuse_fp8_encoding
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

    def rotate(self, vectors, positions):
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

    def _encode_fp8(self, vectors, *, full_graph):
        if (
            full_graph
            and self.fuse_fp8_encoding
            and vectors.device.type == "npu"
            and vectors.dtype == torch.bfloat16
            and vectors.is_contiguous()
            and 0 < vectors.numel() <= MAX_FUSED_INDEXPOOL_ROWS * INDEX_DIM
        ):
            return encode_indexpool_fp8(vectors)
        return indexpool_fp8(vectors)

    def encode_queries(self, hidden, q_c, positions, *, full_graph=False):
        query = self.query_projection(q_c)[0].reshape(-1, self.num_heads, INDEX_DIM)
        weight = self.head_weight_projection(hidden)[0].float() * (self.num_heads * INDEX_DIM) ** -0.5
        return self._encode_fp8(self.rotate(query, positions), full_graph=full_graph), weight

    def encode_keys(self, raw, positions, *, full_graph=False):
        groups = raw.shape[0] // self.ratio
        raw = raw[: groups * self.ratio].reshape(groups, self.ratio, INDEX_DIM)
        if not groups:
            return raw.new_empty((0, INDEX_DIM), dtype=torch.float32)
        pooled = pooled_keys(
            raw,
            self.key_value_projection.weight,
            self.key_gate_projection.weight,
            self.intra_group_ape,
            self.key_norm.weight,
            self.eps,
        )
        pooled = self.rotate(pooled, positions[..., : groups * self.ratio : self.ratio])
        return self._encode_fp8(pooled, full_graph=full_graph)


class AscendHYV4FlashAttention(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, config, layer_idx, prefix, *, reuse_target=False):
        super().__init__()
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
        use_a5_c8 = HAS_TRITON and self.use_fp8_cache and get_ascend_device_type() is AscendDeviceType.A5
        self.fuse_c8_write = use_a5_c8 and vllm_config.additional_config.get("hy4flash_fuse_c8_write", True)
        self.fuse_c8_qk = (
            use_a5_c8
            and not reuse_target
            and (self.latent_dim, self.rope_dim) == (512, 64)
            and vllm_config.additional_config.get("hy4flash_fuse_c8_qk", True)
        )
        self.fuse_indexpool_select = (
            use_a5_c8 and not reuse_target and vllm_config.additional_config.get("hy4flash_fuse_indexpool_select", True)
        )
        self.share_swa_c8_reads = use_a5_c8 and vllm_config.additional_config.get("hy4flash_share_swa_c8_reads", True)
        self.fuse_swa_q_norm = (
            self.is_window_layer
            and not reuse_target
            and vllm_config.additional_config.get("hy4flash_fuse_swa_q_norm", True)
            and get_current_hardware_profile().supports(HardwareCapability.DYNAMIC_MX_QUANT_FUSION)
            and hasattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant")
        )
        self.fuse_attn_output_gate = (
            not reuse_target
            and HAS_TRITON
            and vllm_config.additional_config.get("hy4flash_fuse_attn_output_gate", True)
            and get_ascend_device_type() is AscendDeviceType.A5
        )
        self.overlap_attn_output_gate = (
            not reuse_target
            and vllm_config.additional_config.get("hy4flash_overlap_attn_output_gate", True)
            and get_ascend_device_type() is AscendDeviceType.A5
        )
        self.use_cannbot_prolog = not reuse_target and vllm_config.additional_config.get(
            "hy4flash_cannbot_mla_prolog", False
        )
        self._cannbot_prolog = None
        self.use_triton_mla_logits = HAS_TRITON and vllm_config.additional_config.get(
            "hy4flash_triton_mla_logits", False
        )
        self.batch_verify_attention = (
            use_a5_c8
            and not self.use_triton_mla_logits
            and vllm_config.additional_config.get("hy4flash_batch_verify_attention", True)
        )
        self.kv_storage_width = C8_ROW_BYTES if self.use_fp8_cache else self.kv_width
        # Indexer keys and position lanes remain BF16 in either layout.
        index_element_width = 2 if self.use_fp8_cache else 1
        index_width = INDEX_DIM + POSITION_LANES
        self.token_storage_width = self.kv_storage_width + index_width * index_element_width
        ratio = config.compressed_indexer_compress_ratio
        # Use the same page size for SWA and sparse layers in this cache group.
        self.incremental_index_pool = (
            HAS_TRITON and ratio > 0 and INDEX_DIM % ratio == 0 and vllm_config.cache_config.block_size % ratio == 0
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
        self.indexer = (
            None
            if self.is_window_layer or reuse_target
            else FlashIndexPool(
                config,
                quant,
                rope_parameters,
                f"{prefix}.indexer",
                fuse_fp8_encoding=use_a5_c8 and vllm_config.additional_config.get("hy4flash_fuse_indexpool_fp8", True),
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
        return HiddenStateCacheSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.cache_width,
            dtype=torch.uint8 if self.use_fp8_cache else torch.bfloat16,
        )

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

    def forward(self, positions, hidden_states):
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
            kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split((self.latent_dim, self.rope_dim), -1)
            kv_c = self.kv_a_layernorm(kv_c)
            q_nope, q_pe = q.split((self.nope_dim, self.rope_dim), dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe.reshape(q.shape[0], -1).contiguous(), k_pe.contiguous())
        q_pe = q_pe.reshape(-1, self.local_heads, self.rope_dim)
        out = hidden_states.new_zeros((q.shape[0], self.local_heads * self.v_dim))
        meta = get_flash_attention_metadata() if self.fuse_attn_output_gate or self.overlap_attn_output_gate else None
        full_decode = meta is not None and meta.decode_seq_lens is not None
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

        self._forward_paged(hidden_states, q_c, q_nope, q_pe, kv_c, k_pe, positions, out, absorbed=absorbed)
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
            logger.info_once("HYV4 Flash fuses the output gate and MXFP8 input quantization.")
            quantized, scale = gate_dynamic_mx_quant(out, gate)
            output = self.o_proj(PreparedLinearInput(quantized, scale, out.dtype))[0]
        else:
            if fuse_gate:
                gate = gate.sigmoid()
            output = self.o_proj(out * gate)[0]
        if self.sequence_parallel_output:
            output = sp_reduce_scatter(output)
        return self.post_proj_norm(output) if self.post_proj_norm is not None else output

    def _forward_paged(self, hidden, q_c, q_nope, q_pe, kv_c, k_pe, positions, out, *, absorbed=False):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or self.layer_name not in metadata:
            return
        meta = metadata[self.layer_name]
        cache = self.kv_cache.view(-1, self.cache_width)
        block_size = self.kv_cache.shape[2]
        num_tokens = meta.query_start_loc[-1]
        if not num_tokens:
            return
        snapshot = self.topk_snapshot
        if snapshot is not None:
            snapshot[:num_tokens].fill_(-1)
        full_graph = meta.decode_seq_lens is not None
        keys = self._write_cache_rows(cache, meta, hidden, kv_c, k_pe, positions, num_tokens, full_graph=full_graph)
        query, wv = self._absorb_query(q_nope[:num_tokens], q_pe[:num_tokens], absorbed=absorbed)
        if full_graph:
            self._forward_decode_graph(meta, cache, block_size, hidden, q_c, positions, query, wv, out)
            return

        if keys is not None and self.incremental_index_pool:
            self._update_index_pool(cache, block_size, meta, keys, positions)
        self._attend_eager(meta, cache, block_size, hidden, q_c, positions, query, wv, out)

    def _write_cache_rows(self, cache, meta, hidden, kv_c, k_pe, positions, num_tokens, *, full_graph):
        slots = meta.slot_mapping[:num_tokens]
        keys = (
            self.indexer.token_key_norm(self.indexer.token_key_projection(hidden[:num_tokens])[0]).bfloat16()
            if self.indexer is not None
            else None
        )
        if (
            self.fuse_c8_write
            and cache.device.type == "npu"
            and num_tokens <= MAX_FUSED_C8_WRITE_TOKENS
            and self.cache_width % torch.int32.itemsize == 0
        ):
            payload, scales = self._quantize_c8(kv_c[:num_tokens], gpu_scale=self.use_gpu_c8_scale)
            write_c8_rows(cache, slots, payload, k_pe[:num_tokens], scales, keys, positions[..., :num_tokens])
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

    def _attend_eager(self, meta, cache, block_size, hidden, q_c, positions, query, wv, out):
        num_tokens = query.shape[0]
        snapshot = self.topk_snapshot
        uses_indexpool = self.indexer is not None and any(
            (length - 1) // self.indexer.ratio > self.indexer.budget for length in meta.seq_lens
        )
        if uses_indexpool:
            index_query, index_weight = self.indexer.encode_queries(
                hidden[:num_tokens], q_c[:num_tokens], positions[..., :num_tokens]
            )
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
            may_score_groups = uses_indexpool and (seq_len - 1) // self.indexer.ratio > self.indexer.budget
            pooled_history = None
            if may_score_groups:
                if self.incremental_index_pool:
                    groups = seq_len // self.indexer.ratio
                    pooled_history = torch.empty((groups, INDEX_DIM), dtype=torch.float32, device=hidden.device)
                    copy_pool_keys(
                        cache, meta.block_table[i], pooled_history, 0, block_size, self.indexer.ratio, write=False
                    )
                else:
                    history = torch.arange(seq_len, device=hidden.device)
                    raw_keys, key_pos = self._read_index_rows(
                        cache, meta.block_table[i], block_size, history, positions.ndim
                    )
            history_start = max(0, seq_len - (stop - start) + 1 - self.window) if self.indexer is None else 0
            shared_kv = None
            if verify and (
                not may_score_groups or seq_len <= (stop - start) * self.indexer.ratio * (self.indexer.budget + 1)
            ):
                # Each query keeps its own candidate width. Gather and decode
                # the overlapping KV history once for the whole request.
                shared_positions = torch.arange(history_start, seq_len, device=hidden.device)
                shared_kv = self._gather_attention_rows(cache, meta.block_table[i], block_size, shared_positions)
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
                self._attend_decoded(query[start:stop], selected, kv_selected, wv, out[start:stop])
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
            chunk = 1 if verify else QUERY_CHUNK
            for lo in range(start, stop, chunk):
                hi = min(lo + chunk, stop)
                visible_len = seq_len - (stop - hi)
                history_len = visible_len if verify else seq_len
                score_groups = may_score_groups and (history_len - 1) // self.indexer.ratio > self.indexer.budget
                if shared_kv is not None and not score_groups:
                    first = max(0, visible_len - self.window) if self.indexer is None else 0
                    selected_slice = slice(first - history_start, visible_len - history_start)
                    selected = shared_positions[selected_slice].unsqueeze(0)
                    kv_selected = shared_kv[selected_slice].unsqueeze(0)
                else:
                    logical = torch.arange(visible_len - (hi - lo), visible_len, device=hidden.device)
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
                        selected = sparse_candidates(
                            index_query[lo:hi],
                            index_weight[lo:hi],
                            pool,
                            logical,
                            self.indexer.ratio,
                            self.indexer.budget,
                        )
                        if batch_sparse:
                            candidate_rows.append(selected)
                            continue
                    else:
                        # Every complete group fits the recall budget, so only causality matters.
                        visible = torch.arange(visible_len, device=hidden.device)
                        selected = visible.expand(hi - lo, -1).masked_fill(visible > logical[:, None], -1)
                    valid_indices = selected.clamp_min(0)
                    if shared_kv is None:
                        kv_selected = self._gather_attention_rows(cache, meta.block_table[i], block_size, valid_indices)
                    else:
                        kv_selected = shared_kv.index_select(0, valid_indices.flatten()).view(*selected.shape, -1)
                if snapshot is not None:
                    snapshot[lo:hi, : selected.shape[-1]].copy_(selected)
                self._attend_decoded(query[lo:hi], selected, kv_selected, wv, out[lo:hi])
            if batch_sparse:
                selected = torch.cat(candidate_rows, dim=0)
                valid_indices = selected.clamp_min(0)
                if shared_kv is None:
                    kv_selected = gather_c8_rows(cache, meta.block_table[i], block_size, valid_indices)
                else:
                    kv_selected = shared_kv.index_select(0, valid_indices.flatten()).view(*selected.shape, -1)
                if snapshot is not None:
                    snapshot[start:stop, : selected.shape[-1]].copy_(selected)
                self._attend_decoded(query[start:stop], selected, kv_selected, wv, out[start:stop], split_pv=True)

    def _forward_decode_graph(self, meta, cache, block_size, hidden, q_c, positions, query, wv, out):
        seq_lens = meta.decode_seq_lens
        num_tokens = query.shape[0]
        num_queries = num_tokens // seq_lens.shape[0]
        live = meta.slot_mapping[:num_tokens].view(-1, num_queries) >= 0
        seq_lens = seq_lens.long().masked_fill(~live[:, 0], 0)
        offsets = torch.arange(num_queries, device=query.device)
        logical = (seq_lens[:, None] - num_queries + offsets).masked_fill(~live, -1)
        snapshot = self.topk_snapshot
        window = min(self.window, self.max_model_len)
        shared_positions = None
        if (
            self.share_swa_c8_reads
            and self.indexer is None
            and num_queries == BATCH_VERIFY_MAX_QUERIES
            and (self.local_heads, window) == BATCH_VERIFY_SWA_SHAPE
            and cache.shape[-1] % torch.int32.itemsize == 0
        ):
            # The union of consecutive windows has window + num_queries - 1 rows.
            history_offsets = torch.arange(1 - window - num_queries, 0, device=query.device)
            shared_positions = seq_lens[:, None] + history_offsets
            # Slot padding can also occur at the tail of an otherwise live request.
            shared_positions.masked_fill_(shared_positions > logical.amax(dim=1, keepdim=True), -1)

        score_groups = self.indexer is not None and self.max_model_len > (self.indexer.budget + 1) * self.indexer.ratio
        use_paged_pool = False
        if score_groups:
            index_query, index_weight = self.indexer.encode_queries(
                hidden[:num_tokens], q_c[:num_tokens], positions[..., :num_tokens], full_graph=True
            )
            use_paged_pool = (
                self.fuse_indexpool_select
                and self.incremental_index_pool
                and self.local_heads == BATCH_VERIFY_SPARSE_SHAPE[0]
                and (num_queries, self.indexer.num_heads, self.indexer.ratio, self.indexer.budget)
                == PAGED_POOL_DECODE_SHAPE
                and cache.device.type == "npu"
                and meta.block_table.stride(-1) == 1
                and index_query.is_contiguous()
                and index_weight.is_contiguous()
            )
        for request in range(seq_lens.shape[0]):
            start = request * num_queries
            stop = start + num_queries
            table = meta.block_table[request]
            visible = logical[request]
            if self.indexer is None:
                selected = window_candidates(visible, window)
            else:
                if self.incremental_index_pool:
                    self._update_decode_pool(cache, table, block_size, seq_lens[request], num_queries, positions.ndim)
                if score_groups:
                    ratio = self.indexer.ratio
                    if use_paged_pool:
                        logger.info_once("HYV4 Flash uses paged IndexPool scoring for six verification queries.")
                        selected = select_paged_pool(
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
                logger.info_once("HYV4 Flash fuses paged C8 reads and FP32 QK for sparse verification.")
                scores, rows = gather_c8_qk(q, cache, table, block_size, selected)
            elif shared_positions is None:
                rows = self._gather_attention_rows(cache, table, block_size, selected, mask_padding=True)
            else:
                history = self._gather_attention_rows(
                    cache, table, block_size, shared_positions[request], mask_padding=True
                )
                rows = history.unfold(0, window, 1).transpose(1, 2)
            self._attend_decoded(q, selected, rows, wv, out[start:stop], split_pv=split_pv, scores=scores)
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
        copy_pool_keys(cache, table, keys, first, block_size, ratio, write=True, end_group=end)

    def _read_decode_pool(self, cache, table, block_size, seq_len, position_ndim):
        ratio = self.indexer.ratio
        groups = self.max_model_len // ratio
        end = seq_len // ratio
        if self.incremental_index_pool:
            keys = torch.empty((groups, INDEX_DIM), dtype=torch.float32, device=cache.device)
            copy_pool_keys(cache, table, keys, 0, block_size, ratio, write=False, end_group=end)
            return keys
        history = torch.arange(groups * ratio, device=cache.device)
        history = history.masked_fill(history >= end * ratio, 0)
        raw, key_pos = self._read_index_rows(cache, table, block_size, history, position_ndim)
        return self.indexer.encode_keys(raw, key_pos, full_graph=True)

    def _gather_attention_rows(self, cache, block_table, block_size, positions, *, mask_padding=False):
        if (
            self.use_fp8_cache
            and HAS_TRITON
            and cache.device.type == "npu"
            and cache.shape[-1] % 4 == 0
            and (mask_padding or positions.numel() <= MAX_FUSED_C8_READ_ROWS)
        ):
            return gather_c8_rows(cache, block_table, block_size, positions)
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

    def _update_index_pool(self, cache, block_size, meta, keys, positions):
        ratio = self.indexer.ratio
        updates = []
        raw_keys = []
        key_positions = []
        for request, seq_len in enumerate(meta.seq_lens):
            start, stop = meta.query_start_loc[request : request + 2]
            if start == stop:
                continue
            context_len = seq_len - (stop - start)
            first_group, end_group = context_len // ratio, seq_len // ratio
            if first_group == end_group:
                continue
            # Recompute the first group when it straddles a chunk or contains
            # rejected draft tokens. Complete groups in the prefix stay valid.
            first_token = first_group * ratio
            end = start + end_group * ratio - context_len
            first_new = start + max(0, first_token - context_len)
            current_keys = keys[first_new:end]
            current_positions = positions[..., first_new:end]
            if first_token < context_len:
                logical = torch.arange(first_token, context_len, device=keys.device)
                previous_keys, previous_positions = self._read_index_rows(
                    cache, meta.block_table[request], block_size, logical, positions.ndim
                )
                current_keys = torch.cat((previous_keys, current_keys), dim=0)
                current_positions = torch.cat((previous_positions, current_positions), dim=-1)
            raw_keys.append(current_keys)
            key_positions.append(current_positions)
            updates.append((request, first_group, end_group - first_group))
        if updates:
            pooled = self.indexer.encode_keys(torch.cat(raw_keys), torch.cat(key_positions, dim=-1))
            offset = 0
            for request, first_group, count in updates:
                copy_pool_keys(
                    cache,
                    meta.block_table[request],
                    pooled[offset : offset + count],
                    first_group,
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

    def _absorb_query(self, q_nope, q_pe, *, absorbed=False):
        wk, wv = self._kv_b_weights()
        q_latent = q_nope if absorbed else _head_bmm(q_nope, wk.contiguous())
        return torch.cat((q_latent, q_pe), dim=-1), wv.transpose(1, 2).contiguous()

    def _attend(self, query, selected, kv_selected, wv, out):
        if self.use_fp8_cache:
            kv_selected = dequantize_c8_rows(kv_selected)
        self._attend_decoded(query, selected, kv_selected, wv, out)

    def _attend_decoded(self, query, selected, kv_selected, wv, out, *, split_pv=False, scores=None):
        if scores is not None or (
            self.use_triton_mla_logits and query.device.type == "npu" and query.shape[0] == 1 and selected.shape[-1] > 0
        ):
            if scores is None:
                scores = torch.bmm(query.float(), kv_selected.float().transpose(1, 2))
                if torch_npu.get_npu_format(scores) != ACL_FORMAT_ND:
                    scores = torch_npu.npu_format_cast(scores, ACL_FORMAT_ND)
            logits = build_mla_logits(scores, selected, self.learnable_sink_param, self.scale)
            # Use the reference FP32 softmax reduction before discarding the sink.
            probs = logits.softmax(dim=-1)[..., :-1]
        else:
            probs = mla_attention_probs(query, kv_selected, selected, self.learnable_sink_param, self.scale)
        values = kv_selected[..., : self.latent_dim].float()
        if split_pv:
            # The batched FP32 PV kernel changes the 2048-wide reduction order.
            context = torch.cat([torch.bmm(p, v) for p, v in zip(probs.split(1), values.split(1))])
        else:
            context = torch.bmm(probs, values)
        context = context.to(query.dtype)
        value = _head_bmm(context, wv)
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
