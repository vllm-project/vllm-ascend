# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import typing
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.rope_dsv4 import RopeDataProxy, get_cos_and_sin_dsa
from vllm_ascend.utils import AscendDeviceType, enable_dsa_cp, get_ascend_device_type

from .deepseek_v4 import (
    DSV4_STACKED_PARAMS_MAPPING,
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4Attention,
    _hc_head_torch,
    _normalize_dsv4_layer_weight_name,
)

DSPARK_WO_A_DEQUANT_BLOCK_SIZE = 128
DSPARK_DEFAULT_BLOCK_SIZE = 5
DSPARK_SAS_OP_NAMESPACES = ("_ascend_dsv4", "_ascend_v4", "custom")
DSparkFusedAttentionMetadata = tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class DSparkDecodeMetadata:
    cos: RopeDataProxy
    sin: RopeDataProxy
    context_cache_indices: torch.Tensor | None
    context_cache_valid: torch.Tensor | None
    context_request_slots: torch.Tensor | None


def _get_dspark_sas_op(name: str):
    for namespace in DSPARK_SAS_OP_NAMESPACES:
        op_namespace = getattr(torch.ops, namespace, None)
        if op_namespace is not None and hasattr(op_namespace, name):
            return getattr(op_namespace, name)
    raise RuntimeError(f"DSpark fused attention operator {name!r} is unavailable.")


def _dequant_dspark_wo_a_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    block_size = DSPARK_WO_A_DEQUANT_BLOCK_SIZE
    return (
        weight.unflatten(0, (-1, block_size))
        .unflatten(-1, (-1, block_size))
        .float()
        .mul(scale[:, None, :, None].float())
        .flatten(2, 3)
        .flatten(0, 1)
        .bfloat16()
    )


def get_dspark_num_layers(config: typing.Any) -> int:
    """Read the number of DSpark draft layers from the model config."""
    speculative_config = getattr(config, "speculative_config", None)
    if speculative_config is not None:
        config = speculative_config
    draft_model_config = getattr(config, "draft_model_config", None)
    if draft_model_config is not None:
        config = draft_model_config
    hf_config = getattr(config, "hf_config", None)
    if hf_config is not None:
        config = hf_config

    target_layer_ids = getattr(config, "dspark_target_layer_ids", None)
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ValueError("DSpark requires a non-empty dspark_target_layer_ids list.")
    return len(target_layer_ids)


def _apply_dsv4_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x: torch.Tensor,
    *,
    inverse: bool = False,
    cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    partial_slice: list[int] | None = None,
) -> torch.Tensor:
    if cos_sin is None:
        cos, sin = get_cos_and_sin_dsa(positions)
        cos_t = cos[rotary_emb.layername]
        sin_t = sin[rotary_emb.layername]
    else:
        cos_t, sin_t = cos_sin
    if inverse:
        sin_t = -sin_t
    if partial_slice is None:
        return rotary_emb(x, cos_t, sin_t)

    x_rope = x
    if x_rope.dim() == 2:
        x_rope = x_rope.unsqueeze(-2)
    if x_rope.dim() == 3:
        x_rope = x_rope.unsqueeze(1)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        x_rope,
        cos_t,
        sin_t,
        rotary_mode="interleave",
        partial_slice=partial_slice,
    )
    return x


def _wo_a_weight_for_eager_projection(
    wo_a_weight: torch.Tensor,
    n_local_groups: int,
    o_lora_rank: int,
    group_dim: int,
) -> torch.Tensor:
    if wo_a_weight.ndim == 3:
        return wo_a_weight.transpose(1, 2).contiguous()
    return wo_a_weight.view(n_local_groups, o_lora_rank, group_dim)


def _grouped_wo_a_projection(attn_out: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    return torch.matmul(attn_out.transpose(0, 1), wo_a.transpose(1, 2)).transpose(0, 1)


def dspark_sparse_attn_sharedkv(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    sinks: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    ori_win_left: int,
) -> None:
    forward_context = get_forward_context()
    fused_attn_metadata = getattr(forward_context, "dspark_fused_attn_metadata", None)
    if fused_attn_metadata is None:
        raise RuntimeError("DSpark fused attention metadata was not prepared before model forward.")

    _, block_table, seqused_kv, metadata = fused_attn_metadata
    batch_size = q.shape[0]
    if block_table.shape[0] != batch_size:
        raise ValueError(f"DSpark block table batch mismatch: q={batch_size}, block_table={block_table.shape[0]}")
    if seqused_kv.shape[0] != batch_size:
        raise ValueError(f"DSpark seqused_kv batch mismatch: q={batch_size}, seqused_kv={seqused_kv.shape[0]}")

    result = _get_dspark_sas_op("npu_sparse_attn_sharedkv")(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=None,
        ori_sparse_indices=None,
        cmp_sparse_indices=None,
        ori_block_table=block_table,
        cmp_block_table=None,
        cu_seqlens_q=None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=3,
        ori_win_left=ori_win_left,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="PA_BNBD",
        return_softmax_lse=False,
    )[0]
    output.copy_(result.to(output.dtype))


def dspark_sparse_attn_sharedkv_fake(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    sinks: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    ori_win_left: int,
) -> None:
    return


direct_register_custom_op(
    op_name="dspark_sparse_attn_sharedkv",
    op_func=dspark_sparse_attn_sharedkv,
    mutates_args=["output"],
    fake_impl=dspark_sparse_attn_sharedkv_fake,
    dispatch_key="PrivateUse1",
)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        config = kwargs["config"]
        super().__init__(*args, **kwargs)
        importlib.import_module("ascend_ops")
        self.dspark_sparse_attn_op = _get_dspark_sas_op("npu_sparse_attn_sharedkv")
        self.dspark_sparse_attn_metadata_op = _get_dspark_sas_op("npu_sparse_attn_sharedkv_metadata")
        self._dspark_fused_attn_metadata_cache: dict[tuple[object, ...], DSparkFusedAttentionMetadata] = {}
        self.dsa_attn.compress_ratio = 0
        # DSpark uses the uncompressed SWA cache as its context cache. Disable
        # the unused compressed DSA cache so only the transferable cache is
        # included in the KV cache configuration.
        self.dsa_attn.dsa_attn.compress_ratio = 0
        self.dsa_attn.swa_cache_layer.is_dspark_cache = True
        self.dsa_attn.swa_cache_layer.dtype = vllm_config.model_config.dtype
        self.block_size = int(
            getattr(config, "dspark_block_size", getattr(config, "n_predict", DSPARK_DEFAULT_BLOCK_SIZE))
            or DSPARK_DEFAULT_BLOCK_SIZE
        )
        self.window_size = int(self.window_size)
        self.pa_block_size = int(self.dsa_attn.swa_cache_layer.block_size)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        max_request_slots = max(
            1,
            int(getattr(scheduler_config, "max_num_seqs", 1) or 1),
        )
        self.register_buffer(
            "_dspark_kv_cache",
            torch.zeros(
                (max_request_slots, self.window_size, self.head_dim),
                dtype=vllm_config.model_config.dtype,
                device=self.attn_sink.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_cache_positions",
            torch.full(
                (max_request_slots, self.window_size),
                -1,
                dtype=torch.int32,
                device=self.attn_sink.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_window_offsets",
            torch.arange(self.window_size, device=self.attn_sink.device, dtype=torch.int64).view(1, -1),
            persistent=False,
        )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        if request_slots is None or request_slots.numel() == 0:
            return
        slots = torch.unique(request_slots.to(device=self._dspark_cache_positions.device, dtype=torch.int64))
        self._dspark_kv_cache[slots] = 0
        self._dspark_cache_positions[slots] = -1

    def _get_dspark_kv_cache(self) -> torch.Tensor | None:
        kv_cache = self.dsa_attn.swa_cache_layer.kv_cache
        while isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 1:
            kv_cache = kv_cache[0]
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            return None
        return kv_cache

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        kv = self.kv_norm(self.wkv(hidden_states))
        return _apply_dsv4_rope(
            self.rotary_emb,
            positions,
            kv,
            cos_sin=rope_cos_sin,
            partial_slice=[self.nope_head_dim, self.head_dim],
        ).contiguous()

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        if positions.numel() == 0:
            return
        if slot_mapping is None:
            return
        valid = (positions >= 0) & (slot_mapping >= 0)
        safe_positions = torch.where(valid, positions, torch.zeros_like(positions))
        shared_kv = self._project_shared_kv(main_x, safe_positions, rope_cos_sin)
        kv_cache = self._get_dspark_kv_cache()
        if kv_cache is None:
            # KV caches are not bound during the memory profiling dummy run.
            return

        cache_tokens = kv_cache.flatten(start_dim=2)
        cache_block_size = kv_cache.shape[1]
        slots_int64 = slot_mapping.to(device=shared_kv.device, dtype=torch.int64)
        block_ids = torch.div(slots_int64, cache_block_size, rounding_mode="floor")
        block_offsets = slots_int64.remainder(cache_block_size)
        valid &= block_ids < kv_cache.shape[0]
        flat_valid = valid.reshape(-1)
        flat_block_ids = block_ids.reshape(-1)[flat_valid]
        flat_block_offsets = block_offsets.reshape(-1)[flat_valid]
        flat_shared_kv = shared_kv.reshape(-1, shared_kv.shape[-1])[flat_valid]
        cache_tokens[flat_block_ids, flat_block_offsets, : self.head_dim] = flat_shared_kv.to(cache_tokens.dtype)

    def sync_context_cache_from_paged(
        self,
        block_table: torch.Tensor | None,
        context_lens: torch.Tensor | None,
        request_slots: torch.Tensor | None,
    ) -> None:
        """Restore the position-checked proposal cache from transferable pages."""
        kv_cache = self._get_dspark_kv_cache()
        if kv_cache is None or block_table is None or context_lens is None or request_slots is None:
            return

        batch_size = min(block_table.shape[0], context_lens.numel(), request_slots.numel())
        if batch_size == 0:
            return
        context_lens = context_lens[:batch_size].to(device=kv_cache.device, dtype=torch.int64)
        request_slots = request_slots[:batch_size].to(device=kv_cache.device, dtype=torch.int64)
        context_end = context_lens.view(-1, 1) - 1
        context_positions = context_end + 1 - self.window_size + self._dspark_window_offsets
        position_valid = (context_positions >= 0) & (context_positions <= context_end)

        cache_block_size = kv_cache.shape[1]
        block_numbers = torch.div(
            context_positions.clamp_min(0),
            cache_block_size,
            rounding_mode="floor",
        )
        table_valid = block_numbers < block_table.shape[1]
        safe_block_numbers = block_numbers.clamp(max=block_table.shape[1] - 1)
        block_ids = block_table[:batch_size].gather(1, safe_block_numbers).to(torch.int64)
        cache_valid = position_valid & table_valid & (block_ids >= 0) & (block_ids < kv_cache.shape[0])
        safe_block_ids = block_ids.clamp(min=0, max=kv_cache.shape[0] - 1)
        block_offsets = context_positions.clamp_min(0).remainder(cache_block_size)
        paged_tokens = kv_cache.flatten(start_dim=2)
        context_kv = paged_tokens[safe_block_ids, block_offsets, : self.head_dim]

        self._dspark_kv_cache[request_slots] = 0
        self._dspark_cache_positions[request_slots] = -1
        slot_indices = request_slots.view(-1, 1).expand_as(context_positions)
        cache_indices = context_positions.clamp_min(0).remainder(self.window_size)
        flat_valid = cache_valid.reshape(-1)
        self._dspark_kv_cache[
            slot_indices.reshape(-1)[flat_valid],
            cache_indices.reshape(-1)[flat_valid],
        ] = context_kv.reshape(-1, self.head_dim)[flat_valid].to(self._dspark_kv_cache.dtype)
        self._dspark_cache_positions[
            slot_indices.reshape(-1)[flat_valid],
            cache_indices.reshape(-1)[flat_valid],
        ] = context_positions.to(torch.int32).reshape(-1)[flat_valid]

    def _get_dspark_fused_attention_metadata(
        self,
        device: torch.device,
        batch_size: int,
        draft_len: int,
    ) -> DSparkFusedAttentionMetadata:
        total_tokens = self.window_size + draft_len
        blocks_per_request = (total_tokens + self.pa_block_size - 1) // self.pa_block_size
        key = (
            device.type,
            device.index,
            batch_size,
            draft_len,
            total_tokens,
            self.pa_block_size,
            self.n_local_heads,
            self.head_dim,
            self.window_size,
        )
        cached = self._dspark_fused_attn_metadata_cache.get(key)
        if cached is not None:
            return cached
        block_table = torch.arange(
            batch_size * blocks_per_request,
            dtype=torch.int32,
            device=device,
        ).view(batch_size, blocks_per_request)
        seqused_kv = torch.full((batch_size,), total_tokens, dtype=torch.int32, device=device)
        metadata = self.dspark_sparse_attn_metadata_op(
            num_heads_q=self.n_local_heads,
            num_heads_kv=1,
            head_dim=self.head_dim,
            cu_seqlens_q=None,
            seqused_kv=seqused_kv,
            batch_size=batch_size,
            max_seqlen_q=draft_len,
            max_seqlen_kv=total_tokens,
            cmp_ratio=1,
            ori_mask_mode=0,
            cmp_mask_mode=3,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="BSND",
            layout_kv="PA_BNBD",
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        fused_attention_metadata = (blocks_per_request, block_table, seqused_kv, metadata)
        self._dspark_fused_attn_metadata_cache[key] = fused_attention_metadata
        return fused_attention_metadata

    def _dspark_attention_from_cache(
        self,
        q: torch.Tensor,
        draft_kv: torch.Tensor,
        draft_positions: torch.Tensor,
        request_slots: torch.Tensor,
        context_cache_indices: torch.Tensor | None = None,
        context_cache_valid: torch.Tensor | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, draft_len = q.shape[:2]

        if context_cache_indices is None or context_cache_valid is None or context_request_slots is None:
            block_start = draft_positions[:, :1].to(torch.int64)
            context_end = block_start - 1
            context_start = torch.clamp(context_end + 1 - self.window_size, min=0)
            ctx_positions = context_start + self._dspark_window_offsets
            context_cache_valid = ctx_positions <= context_end
            context_cache_indices = ctx_positions.remainder(self.window_size)
            context_request_slots = (
                request_slots[:, :1].to(device=draft_kv.device, dtype=torch.int64).expand(-1, self.window_size)
            )
        ctx_kv = self._dspark_kv_cache[context_request_slots, context_cache_indices]
        ctx_kv = torch.where(context_cache_valid.unsqueeze(-1), ctx_kv, torch.zeros_like(ctx_kv))

        total_tokens = self.window_size + draft_len
        blocks_per_request = (total_tokens + self.pa_block_size - 1) // self.pa_block_size
        padded_tokens = blocks_per_request * self.pa_block_size
        if padded_tokens > total_tokens:
            pad = draft_kv.new_zeros((batch_size, padded_tokens - total_tokens, self.head_dim))
            kv = torch.cat([ctx_kv, draft_kv, pad], dim=1)
        else:
            kv = torch.cat([ctx_kv, draft_kv], dim=1)
        kv = kv.view(batch_size * blocks_per_request, self.pa_block_size, 1, self.head_dim).contiguous()

        output = torch.empty_like(q)
        torch.ops.vllm.dspark_sparse_attn_sharedkv(
            q,
            kv,
            self.attn_sink,
            output,
            float(self.scale),
            self.window_size - 1,
        )
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_metadata: DSparkDecodeMetadata | None = None,
    ) -> torch.Tensor:
        if dspark_metadata is None:
            rope_cos_sin = None
            context_cache_indices = None
            context_cache_valid = None
            context_request_slots = None
        else:
            layer_name = self.rotary_emb.layername
            rope_cos_sin = (dspark_metadata.cos[layer_name], dspark_metadata.sin[layer_name])
            context_cache_indices = dspark_metadata.context_cache_indices
            context_cache_valid = dspark_metadata.context_cache_valid
            context_request_slots = dspark_metadata.context_request_slots
        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_norm_without_weight(q)
        q = _apply_dsv4_rope(
            self.rotary_emb,
            positions,
            q,
            cos_sin=rope_cos_sin,
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        shared_kv = self._project_shared_kv(hidden_states, positions, rope_cos_sin)
        if positions.numel() % self.block_size != 0:
            raise ValueError(
                f"DSpark decode requires a multiple of block_size tokens, got "
                f"{positions.numel()} tokens for block_size={self.block_size}"
            )
        batch_size = positions.numel() // self.block_size
        q = q.view(batch_size, self.block_size, self.n_local_heads, self.head_dim)
        draft_kv = shared_kv.view(batch_size, self.block_size, self.head_dim)
        draft_positions = positions.view(batch_size, self.block_size)
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        request_slots = request_slots.view(batch_size, self.block_size)
        out = self._dspark_attention_from_cache(
            q,
            draft_kv,
            draft_positions,
            request_slots,
            context_cache_indices,
            context_cache_valid,
            context_request_slots,
        ).flatten(0, 1)

        out = _apply_dsv4_rope(
            self.rotary_emb,
            positions,
            out,
            inverse=True,
            cos_sin=rope_cos_sin,
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        group_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        out = out.reshape(-1, self.n_local_groups, group_dim)
        if hasattr(self.wo_a, "weight_scale") and self.wo_a.weight.dtype == torch.float8_e4m3fn:
            out, out_scale = torch_npu.npu_dynamic_mx_quant(out, dst_type=torch.float8_e4m3fn)
            # Recipe A5 MXFP8 o-proj path: 32 is the MX scale group size on
            # the innermost dimension; the permutations compute grouped
            # [groups, tokens, dim] x [groups, dim, rank] -> [tokens, groups, rank].
            z = torch_npu.npu_transpose_quant_batchmatmul(
                out,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=out_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            ).flatten(1)
        else:
            wo_a = _wo_a_weight_for_eager_projection(self.wo_a.weight, self.n_local_groups, self.o_lora_rank, group_dim)
            z = _grouped_wo_a_projection(out, wo_a).flatten(1)
        return self.wo_b(z)


class DeepseekV4DSparkDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            topk_indices_buffer=None,
            is_draft_layer=True,
            attn_cls=DeepseekV4DSparkAttention,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_metadata: DSparkDecodeMetadata | None = None,
    ) -> torch.Tensor:
        hidden_states, _ = self._forward_hc_blocks(
            positions,
            hidden_states,
            input_ids=input_ids,
            extra_attn_kwargs={
                "request_slots": request_slots,
                "slot_mapping": slot_mapping,
                "block_table": block_table,
                "dspark_metadata": dspark_metadata,
            },
        )
        return hidden_states


class DSparkMarkovHead(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size, config.dspark_markov_rank, prefix=f"{prefix}.markov_w1"
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            config.dspark_markov_rank,
            params_dtype=torch.float32,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.markov_w2, markov_embed)


class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        ascend_device_type = get_ascend_device_type()
        if ascend_device_type != AscendDeviceType.A5:
            raise RuntimeError(
                f"DeepSeek V4 DSpark is only supported on Ascend A5, but detected Ascend {ascend_device_type.name}."
            )
        config = vllm_config.speculative_config.draft_model_config.hf_config
        logger.info_once("Initializing the DeepSeek V4 DSpark draft model.")
        self.config = config
        self.hc_mult = config.hc_mult
        self.target_layer_ids = list(getattr(config, "dspark_target_layer_ids", []) or [])
        self.num_dspark_layers = get_dspark_num_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.embed_tokens = None
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV4DSparkDecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx + idx}"),
                )
                for idx in range(self.num_dspark_layers)
            }
        )
        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.main_proj = ReplicatedLinear(
            config.hidden_size * max(1, len(self.target_layer_ids)),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(config, maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"))
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale
        shared_fused_attn_metadata_cache: dict[tuple[object, ...], DSparkFusedAttentionMetadata] = {}
        for layer in self.layers.values():
            layer.self_attn._dspark_fused_attn_metadata_cache = shared_fused_attn_metadata_cache

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise AttributeError("DSpark draft model requires shared target embed_tokens.")
        return self.embed_tokens(input_ids)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping=None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        if context_states.numel() == 0:
            return
        main_x = self.main_norm(self.main_proj(context_states))
        safe_context_positions = torch.where(
            context_positions >= 0,
            context_positions,
            torch.zeros_like(context_positions),
        )
        rope_cos, rope_sin = get_cos_and_sin_dsa(safe_context_positions)
        for layer in self.layers.values():
            rotary_emb = layer.self_attn.rotary_emb
            rope_cos_sin = (rope_cos[rotary_emb.layername], rope_sin[rotary_emb.layername])
            layer.self_attn.precompute_context_kv(
                main_x,
                context_positions,
                context_slot_mapping,
                rope_cos_sin,
            )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        for layer in self.layers.values():
            layer.self_attn.reset_request_slots(request_slots)

    def prepare_fused_attention_metadata(
        self,
        device: torch.device,
        batch_size: int,
        draft_len: int,
    ) -> DSparkFusedAttentionMetadata:
        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        return first_layer.self_attn._get_dspark_fused_attention_metadata(
            device,
            batch_size,
            draft_len,
        )

    def sync_context_cache_from_paged(
        self,
        block_table: torch.Tensor | None,
        context_lens: torch.Tensor | None,
        request_slots: torch.Tensor | None,
    ) -> None:
        for layer in self.layers.values():
            layer.self_attn.sync_context_cache_from_paged(block_table, context_lens, request_slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        context_cache_indices: torch.Tensor | None = None,
        context_cache_valid: torch.Tensor | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        rope_cos, rope_sin = get_cos_and_sin_dsa(positions)
        dspark_metadata = DSparkDecodeMetadata(
            cos=rope_cos,
            sin=rope_sin,
            context_cache_indices=context_cache_indices,
            context_cache_valid=context_cache_valid,
            context_request_slots=context_request_slots,
        )
        for layer in self.layers.values():
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                input_ids=input_ids,
                request_slots=request_slots,
                slot_mapping=slot_mapping,
                block_table=block_table,
                dspark_metadata=dspark_metadata,
            )
        return hidden_states

    def compute_head_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            return hidden_states
        return _hc_head_torch(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, lm_head: ParallelLMHead, logits_processor: LogitsProcessor
    ) -> torch.Tensor:
        return logits_processor(lm_head, self.norm(self.compute_head_hidden(hidden_states)))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            finalize = getattr(layer.mlp, "finalize_mega_moe_weights", None)
            if finalize is not None:
                finalize()


@support_torch_compile
class DeepSeekV4DSparkMTP(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.has_own_embed_tokens = False
        self.model = DeepseekV4DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.set_moe_parameters_from_layers(iter(self.model.layers.values()))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        context_cache_indices: torch.Tensor | None = None,
        context_cache_valid: torch.Tensor | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input_ids is not None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            request_slots=request_slots,
            slot_mapping=slot_mapping,
            block_table=block_table,
            context_cache_indices=context_cache_indices,
            context_cache_valid=context_cache_valid,
            context_request_slots=context_request_slots,
        )

    def prepare_fused_attention_metadata(
        self,
        device: torch.device,
        batch_size: int,
        draft_len: int,
    ) -> DSparkFusedAttentionMetadata:
        return self.model.prepare_fused_attention_metadata(device, batch_size, draft_len)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor | None:
        if self.lm_head is None:
            raise AttributeError("DSpark draft model requires shared target lm_head.")
        return self.model.compute_logits(hidden_states, self.lm_head, self.logits_processor)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed)

    def precompute_and_store_context_kv(
        self, context_states, context_positions, context_slot_mapping=None, context_request_slots=None
    ) -> None:
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping)

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        self.model.reset_request_slots(request_slots)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        skipped_params: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        expert_mapping = self.model.get_expert_mapping()
        start_layer_idx = self.config.num_hidden_layers
        last_layer_idx = start_layer_idx + self.model.num_dspark_layers - 1
        wo_a_dequant_cache: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            if name in ("embed.weight", "head.weight"):
                continue
            mapped_name = self._map_dspark_weight_name(name)
            if mapped_name is None:
                continue
            name = mapped_name
            if ".confidence_head." in name:
                continue

            if name.startswith(f"model.layers.{last_layer_idx}.hc_head_"):
                canonical_name = name.replace(f"model.layers.{last_layer_idx}.", "model.", 1)
                if canonical_name in params_dict:
                    name = canonical_name

            if (
                name.endswith(".self_attn.wo_a.weight")
                or name.endswith(".self_attn.wo_a.scale")
                or name.endswith(".self_attn.wo_a.weight_scale")
            ):
                base_name, attr = name.rsplit(".", 1)
                if attr == "weight_scale":
                    attr = "scale"
                wo_a_dequant_cache.setdefault(base_name, {})[attr] = loaded_weight
                cache_entry = wo_a_dequant_cache[base_name]
                if "weight" in cache_entry and "scale" in cache_entry:
                    del wo_a_dequant_cache[base_name]
                    mapped = f"{base_name}.weight"
                    if mapped not in params_dict:
                        skipped_params.add(mapped)
                        continue
                    param = params_dict[mapped]
                    dequant_weight = _dequant_dspark_wo_a_weight(cache_entry["weight"], cache_entry["scale"])
                    module_name = base_name.removeprefix("model.")
                    module = self.model.get_submodule(module_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    new_weight = nn.Parameter(
                        torch.empty(param.shape, dtype=dequant_weight.dtype, device=param.device),
                        requires_grad=False,
                    )
                    set_weight_attrs(
                        new_weight,
                        {
                            "input_dim": getattr(param, "input_dim", 1),
                            "output_dim": getattr(param, "output_dim", 0),
                            "weight_loader": weight_loader,
                        },
                    )
                    module.weight = new_weight
                    if "weight_scale" in module._parameters:
                        del module._parameters["weight_scale"]
                    params_dict.pop(f"{base_name}.weight_scale", None)
                    module.quant_method.process_weights_after_loading = lambda layer: None
                    params_dict[mapped] = module.weight
                    weight_loader(module.weight, dequant_weight)
                    loaded_params.add(mapped)
                continue

            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            for param_name, weight_name, stacked_shard_id in DSV4_STACKED_PARAMS_MAPPING:
                if ".experts." in name or f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(mapped)
                break
            else:
                if ".experts." in name:
                    for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
                        if weight_name not in name:
                            continue
                        mapped = name.replace(weight_name, param_name)
                        if mapped not in params_dict:
                            continue
                        param = params_dict[mapped]
                        weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(mapped)
                            break
                    continue
                if "attn_sink" in name:
                    if name not in params_dict:
                        skipped_params.add(name)
                        continue
                    narrow = loaded_weight[head_start:head_end] if not enable_dsa_cp() else loaded_weight
                    with torch.no_grad():
                        params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if name not in params_dict:
                    skipped_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        required_params = {
            f"model.layers.{start_layer_idx}.main_proj.weight",
            f"model.layers.{start_layer_idx}.main_norm.weight",
            f"model.layers.{last_layer_idx}.norm.weight",
            "model.hc_head_fn",
            "model.hc_head_base",
            "model.hc_head_scale",
            f"model.layers.{last_layer_idx}.markov_head.markov_w1.weight",
            f"model.layers.{last_layer_idx}.markov_head.markov_w2.weight",
        }
        missing_required = sorted(required_params - loaded_params)
        if missing_required:
            raise ValueError(
                f"DSpark speculative decoding required weights missing from checkpoint load: {missing_required}"
            )
        missing_params = set(params_dict) - loaded_params
        optional_keys = (
            "q_norm_without_weight.",
            "rotary_emb.inv_freq",
        )
        missing_params = {name for name in missing_params if not any(key in name for key in optional_keys)}
        if missing_params:
            logger.warning("DSpark weights not initialized from checkpoint: %s", sorted(missing_params))
        if skipped_params:
            logger.warning("DSpark checkpoint weights skipped by name mapping: %s", sorted(skipped_params))
        self.model.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _map_dspark_weight_name(self, name: str) -> str | None:
        if "rotary_emb.inv_freq" in name:
            return None
        if not name.startswith("mtp."):
            return None

        parts = name.split(".", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            return None
        stage_idx = int(parts[1])
        if stage_idx >= self.model.num_dspark_layers:
            return None
        layer_idx = self.config.num_hidden_layers + stage_idx
        suffix = parts[2]

        name = f"model.layers.{layer_idx}.{suffix}"
        return _normalize_dsv4_layer_weight_name(name, preserve_wo_a_scale=True)
