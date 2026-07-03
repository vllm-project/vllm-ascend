# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import json
import os
import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend import envs
from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4Attention,
    _apply_dsv4_rope,
    _apply_dsv4_rope_tail,
    _grouped_wo_a_projection,
    _hc_head_torch,
    _linear_output,
    _make_deepseek_v4_expert_params_mapping,
    _wo_a_weight_for_eager_projection,
)
from vllm_ascend.ops.dspark_attention import (
    _gather_context_kv,
    _gather_paged_swa_kv_positions,
    dspark_attention,
    dspark_attention_from_standard_cache,
    dspark_attention_from_standard_cache_sas,
)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
_LAYER_ID_RE = re.compile(r"model\.layers\.(\d+)\.")

_FP8_E4M3FN_SUBNORMAL_STEP = 2.0**-9
_FP8_E4M3FN_MIN_NORMAL = 2.0**-6
_FP8_E4M3FN_SUBNORMAL_NORMAL_MIDPOINT = (7 * _FP8_E4M3FN_SUBNORMAL_STEP + _FP8_E4M3FN_MIN_NORMAL) * 0.5


def _draft_quant_config(vllm_config: VllmConfig):
    assert vllm_config.speculative_config is not None
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(draft_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _should_apply_dspark_fp8_qdq(config: PretrainedConfig) -> bool:
    return not (
        getattr(config, "dspark_mtp_dequantized_to_bf16", False)
        or getattr(config, "dspark_full_dequantized_to_bf16", False)
    )


def _dspark_mhc_pre_torch(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_alpha: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = residual.dtype
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    residual_hc = residual_flat.reshape(-1, hc_mult * hidden_size).float()
    mixes = F.linear(residual_hc, hc_fn.float())
    mixes = mixes * torch.rsqrt(residual_hc.square().mean(dim=-1, keepdim=True) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_start = hc_mult
    post_end = 2 * hc_mult
    post_logits = mixes[:, post_start:post_end] * hc_scale[1] + hc_base[post_start:post_end]
    post_mix = torch.sigmoid(post_logits) * hc_post_alpha

    comb_logits = mixes[:, post_end:].reshape(-1, hc_mult, hc_mult) * hc_scale[2] + hc_base[post_end:].reshape(
        1, hc_mult, hc_mult
    )
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(max(int(sinkhorn_repeat) - 1, 0)):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.float(), dim=1).to(dtype)
    return (
        layer_input.reshape(*outer_shape, hidden_size),
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
    )


def _dspark_mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        res_mix.float(),
        residual.float(),
    )
    post_term = post_mix.float() * x.unsqueeze(-2).float()
    return (mixed_residual + post_term).to(residual.dtype)


def _dspark_mhc_fused_post_pre_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_alpha: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_cur = _dspark_mhc_post_torch(x, residual, post_mix, res_mix)
    layer_input, post_mix_cur, res_mix_cur = _dspark_mhc_pre_torch(
        residual_cur,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    return residual_cur, post_mix_cur, res_mix_cur, layer_input


def _dspark_cache_capacity(vllm_config: VllmConfig, block_size: int, window_size: int | None = None) -> int:
    if window_size is not None:
        return max(block_size, int(window_size) + block_size)
    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    return max(block_size, max_model_len + block_size)


def _dspark_max_request_slots(vllm_config: VllmConfig) -> int:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return max(1, int(getattr(scheduler_config, "max_num_seqs", 1) or 1))


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


def _dspark_standard_dsa_enabled() -> bool:
    return not envs.VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE


def _dspark_standard_dsa_sas_enabled() -> bool:
    return not envs.VLLM_ASCEND_DSPARK_USE_PTA_REF


def _dspark_attention_diff_path() -> str | None:
    return envs.VLLM_ASCEND_DSPARK_ATTENTION_DIFF_PATH


def _dspark_attention_diff_max_records() -> int:
    return envs.VLLM_ASCEND_DSPARK_ATTENTION_DIFF_MAX_RECORDS


def _dspark_kv_diff_path() -> str | None:
    return envs.VLLM_ASCEND_DSPARK_KV_DIFF_PATH


def _dspark_kv_diff_max_records() -> int:
    return envs.VLLM_ASCEND_DSPARK_KV_DIFF_MAX_RECORDS


def _dspark_kv_write_trace_path() -> str | None:
    return envs.VLLM_ASCEND_DSPARK_KV_WRITE_TRACE_PATH


def _dspark_kv_write_trace_max_records() -> int:
    return envs.VLLM_ASCEND_DSPARK_KV_WRITE_TRACE_MAX_RECORDS


def _sync_npu_device_for_standard_pta(tensor: torch.Tensor) -> None:
    if tensor.device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _select_layer_value(
    value: typing.Any,
    layer_idx: int,
    layer_key: str,
    layer_prefix: str,
):
    if isinstance(value, dict):
        if layer_prefix in value:
            return value[layer_prefix]
        if layer_key in value:
            return value[layer_key]
        if layer_idx in value:
            return value[layer_idx]
        return None
    if isinstance(value, (list, tuple)):
        return value[layer_idx]
    return value


def _get_layer_prefix(layer: nn.Module, layer_key: str) -> str:
    return getattr(
        getattr(getattr(getattr(layer, "self_attn", None), "dsa_attn", None), "swa_cache_layer", None),
        "prefix",
        layer_key,
    )


def _fp8_e4m3fn_quantized_abs(abs_scaled: torch.Tensor) -> torch.Tensor:
    subnormal = torch.floor(abs_scaled / _FP8_E4M3FN_SUBNORMAL_STEP + 0.5).clamp(0, 7) * _FP8_E4M3FN_SUBNORMAL_STEP

    normal_exp = torch.floor(torch.log2(abs_scaled.clamp_min(_FP8_E4M3FN_MIN_NORMAL))).clamp(-6, 8)
    normal_base = torch.exp2(normal_exp)
    mantissa = torch.floor((abs_scaled / normal_base - 1.0) * 8.0 + 0.5)
    carry = mantissa >= 8
    normal_exp = torch.where(carry, normal_exp + 1.0, normal_exp).clamp(-6, 8)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    mantissa = torch.where(
        normal_exp >= 8,
        mantissa.clamp(0, 6),
        mantissa.clamp(0, 7),
    )
    normal = (1.0 + mantissa / 8.0) * torch.exp2(normal_exp)

    return torch.where(
        abs_scaled < _FP8_E4M3FN_SUBNORMAL_NORMAL_MIDPOINT,
        subnormal,
        normal,
    )


def _fp8_e4m3fn_qdq(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.numel() == 0:
        return x

    orig_shape = x.shape
    last_dim = orig_shape[-1]
    assert last_dim % block_size == 0
    x_view = x.float().reshape(-1, last_dim)
    blocks = x_view.reshape(-1, last_dim // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(
        torch.full((), 2.0, dtype=torch.float32, device=x.device),
        torch.ceil(torch.log2(amax / 448.0)),
    )
    scaled = (blocks / scale).clamp(-448.0, 448.0)

    quantized_abs = _fp8_e4m3fn_quantized_abs(scaled.abs())
    qdq = torch.where(scaled < 0, -quantized_abs, quantized_abs) * scale
    return qdq.reshape(orig_shape).to(x.dtype)


def _fp8_qdq_nope_dims(
    kv: torch.Tensor,
    nope_head_dim: int,
    block_size: int = 64,
) -> torch.Tensor:
    if nope_head_dim <= 0:
        return kv
    kv_nope = _fp8_e4m3fn_qdq(kv[..., :nope_head_dim], block_size)
    return torch.cat([kv_nope, kv[..., nope_head_dim:]], dim=-1)


def _maybe_fp8_qdq_nope_dims(
    kv: torch.Tensor,
    nope_head_dim: int,
    apply_fp8_qdq: bool,
    block_size: int = 64,
) -> torch.Tensor:
    if not apply_fp8_qdq:
        return kv
    return _fp8_qdq_nope_dims(kv, nope_head_dim, block_size)


def _maybe_fp8_e4m3fn_qdq(
    x: torch.Tensor,
    apply_fp8_qdq: bool,
    block_size: int,
) -> torch.Tensor:
    if not apply_fp8_qdq:
        return x
    return _fp8_e4m3fn_qdq(x, block_size)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """DSpark sliding-window attention with an internal eager context cache."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        config = kwargs["config"]
        super().__init__(*args, **kwargs)
        self.compress_ratio = 1
        self.dsa_attn.compress_ratio = 1
        self.block_size = int(config.dspark_block_size)
        self._dspark_apply_fp8_qdq = _should_apply_dspark_fp8_qdq(config)
        cache_capacity = _dspark_cache_capacity(
            vllm_config,
            self.block_size,
            self.window_size if self.window_size is not None else None,
        )
        max_request_slots = _dspark_max_request_slots(vllm_config)
        cache_shape = (max_request_slots, cache_capacity, self.n_local_heads, self.head_dim)
        self.register_buffer(
            "_dspark_k_cache",
            torch.empty(
                cache_shape,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_v_cache",
            torch.empty(
                cache_shape,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_cache_valid",
            torch.zeros((max_request_slots, cache_capacity), dtype=torch.bool, device=current_platform.device_type),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_cache_positions",
            torch.full(
                (max_request_slots, cache_capacity),
                -1,
                dtype=torch.int32,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self._dspark_cache_capacity = cache_capacity
        self._dspark_max_request_slots = max_request_slots
        self._dspark_attention_diff_records = 0
        self._dspark_kv_diff_records = 0
        self._dspark_kv_write_trace_records = 0

    def _ensure_dspark_cache(self, length: int, like: torch.Tensor) -> None:
        del like
        if length > self._dspark_cache_capacity:
            raise ValueError(
                "DSpark attention cache position exceeds preallocated capacity: "
                f"length={length}, capacity={self._dspark_cache_capacity}"
            )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        if request_slots is None or request_slots.numel() == 0:
            return
        slots = torch.unique(request_slots.to(torch.long))
        if slots.numel() == 0:
            return
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        self._dspark_cache_valid[slots] = False
        self._dspark_cache_positions[slots] = -1

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._expand_private_kv(self._project_shared_kv(hidden_states, positions))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()

    def _expand_private_kv(self, shared_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = shared_kv.squeeze(1)
        kv = _maybe_fp8_qdq_nope_dims(kv, self.nope_head_dim, self._dspark_apply_fp8_qdq)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k = torch.cat(
            [
                k_nope.unsqueeze(1).expand(-1, self.n_local_heads, -1),
                k_pe.unsqueeze(1).expand(-1, self.n_local_heads, -1),
            ],
            dim=-1,
        ).contiguous()
        v = kv.unsqueeze(1).expand(-1, self.n_local_heads, -1).contiguous()
        return k, v

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if not _dspark_standard_dsa_enabled():
            return
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]

        from vllm_ascend.device.device_op import DeviceOperator

        slot_mapping = slot_mapping.to(device=shared_kv.device, dtype=torch.int32)
        valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)
        if not bool(torch.any(valid).item()):
            return
        if not bool(torch.all(valid).item()):
            shared_kv = shared_kv[valid]
            slot_mapping = slot_mapping[valid]
        if slot_mapping.ndim == 1:
            slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, swa_cache_layer.block_size)
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)
        # The PTA reference reads the raw SWA cache immediately after scatter,
        # outside the normal DSA attention op stream choreography.
        _sync_npu_device_for_standard_pta(shared_kv)

    _store_standard_swa_context_kv = _store_standard_swa_kv

    def _maybe_dump_standard_kv_write_trace(
        self,
        phase: str,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        request_slots: torch.Tensor | None,
    ) -> None:
        path = _dspark_kv_write_trace_path()
        if not path or slot_mapping is None:
            return
        if self._dspark_kv_write_trace_records >= _dspark_kv_write_trace_max_records():
            return
        valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)
        valid_positions = positions[valid] if valid.numel() == positions.numel() else positions
        valid_slots = slot_mapping[valid] if valid.numel() == slot_mapping.shape[0] else slot_mapping
        record = {
            "pid": os.getpid(),
            "rank": os.getenv("RANK"),
            "local_rank": os.getenv("LOCAL_RANK"),
            "prefix": getattr(self.dsa_attn, "prefix", None),
            "phase": phase,
            "num_tokens": int(positions.numel()),
            "num_valid_tokens": int(valid.sum().item()) if valid.numel() else 0,
            "positions_head": valid_positions.detach().cpu()[:16].tolist(),
            "positions_tail": valid_positions.detach().cpu()[-16:].tolist(),
            "slot_mapping_head": valid_slots.detach().cpu()[:16].tolist(),
            "slot_mapping_tail": valid_slots.detach().cpu()[-16:].tolist(),
            "request_slots_head": None if request_slots is None else request_slots.detach().cpu()[:16].tolist(),
            "request_slots_tail": None if request_slots is None else request_slots.detach().cpu()[-16:].tolist(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._dspark_kv_write_trace_records += 1

    def _standard_query_slot_mapping_from_block_table(
        self,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not _dspark_standard_dsa_enabled() or block_table is None:
            return slot_mapping

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        cache_block_size = int(swa_cache_layer.block_size)
        out = torch.full_like(positions, -1, dtype=torch.int32)
        valid = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        if slot_mapping is not None:
            slot_mapping = slot_mapping.to(device=positions.device)
            valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)

        pos_long = positions.to(torch.long)
        if token_to_req_indices is not None:
            if token_to_req_indices.numel() < positions.numel():
                raise ValueError(
                    "DSpark token_to_req_indices must cover query tokens: "
                    f"token_to_req_indices={token_to_req_indices.numel()}, positions={positions.numel()}"
                )
            token_to_req = token_to_req_indices[: positions.numel()].to(
                device=positions.device,
                dtype=torch.long,
            )
            for req_idx in range(block_table.shape[0]):
                row_indices = torch.nonzero(token_to_req == req_idx, as_tuple=False).flatten()
                row_indices = row_indices[row_indices < positions.numel()]
                if row_indices.numel() == 0:
                    continue
                block_pos = pos_long.index_select(0, row_indices)
                block_nums = block_pos // cache_block_size
                block_offsets = block_pos % cache_block_size
                block_ids = (
                    block_table[req_idx]
                    .to(device=positions.device, dtype=torch.long)
                    .index_select(
                        0,
                        block_nums,
                    )
                )
                out[row_indices] = (block_ids * cache_block_size + block_offsets).to(torch.int32)
        else:
            for block_offset in range(0, positions.numel(), self.block_size):
                block_end = min(block_offset + self.block_size, positions.numel())
                req_idx = block_offset // self.block_size
                if req_idx >= block_table.shape[0]:
                    continue
                block_pos = pos_long[block_offset:block_end]
                block_nums = block_pos // cache_block_size
                block_offsets = block_pos % cache_block_size
                block_ids = (
                    block_table[req_idx].to(device=positions.device, dtype=torch.long).index_select(0, block_nums)
                )
                out[block_offset:block_end] = (block_ids * cache_block_size + block_offsets).to(torch.int32)
        out.masked_fill_(~valid, -1)
        return out

    def _run_standard_dspark_attention(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        draft_kv: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not _dspark_standard_dsa_enabled():
            return None
        if block_table is None:
            logger.warning_once("DSpark standard SWA cache PTA path has no block_table; falling back to private cache")
            return None

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            logger.warning_once("DSpark standard SWA cache PTA path has no kv_cache; falling back to private cache")
            return None

        if _dspark_standard_dsa_sas_enabled():
            sas_out = dspark_attention_from_standard_cache_sas(
                q,
                swa_kv_cache,
                block_table,
                positions,
                slot_mapping,
                self.attn_sink[: self.n_local_heads],
                self.block_size,
                int(self.window_size),
                int(swa_cache_layer.block_size),
                float(self.scale),
                query_start_loc=dspark_query_start_loc,
                seq_lens=dspark_seq_lens,
                token_to_req_indices=dspark_token_to_req_indices,
            )
            if sas_out is not None:
                return sas_out

        return dspark_attention_from_standard_cache(
            q,
            swa_kv_cache,
            block_table,
            positions,
            slot_mapping,
            draft_kv,
            self.attn_sink[: self.n_local_heads],
            self.block_size,
            int(self.window_size),
            int(swa_cache_layer.block_size),
            float(self.scale),
            request_slots=request_slots,
            cache_positions=getattr(self, "_dspark_cache_positions", None),
            cache_valid=getattr(self, "_dspark_cache_valid", None),
            query_start_loc=dspark_query_start_loc,
            seq_lens=dspark_seq_lens,
            token_to_req_indices=dspark_token_to_req_indices,
        )

    def _maybe_dump_standard_attention_diff(
        self,
        standard_attn_out: torch.Tensor | None,
        private_attn_out: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> None:
        path = _dspark_attention_diff_path()
        if not path or standard_attn_out is None:
            return
        if self._dspark_attention_diff_records >= _dspark_attention_diff_max_records():
            return

        diff = (standard_attn_out.float() - private_attn_out.float()).abs()
        standard_flat = standard_attn_out.float().reshape(standard_attn_out.shape[0], -1)
        private_flat = private_attn_out.float().reshape(private_attn_out.shape[0], -1)
        denom = standard_flat.norm(dim=-1) * private_flat.norm(dim=-1)
        cosine = (standard_flat * private_flat).sum(dim=-1) / denom.clamp_min(1e-20)
        record = {
            "pid": os.getpid(),
            "rank": os.getenv("RANK"),
            "local_rank": os.getenv("LOCAL_RANK"),
            "prefix": getattr(self.dsa_attn, "prefix", None),
            "num_tokens": int(positions.numel()),
            "positions": positions.detach().cpu().tolist(),
            "slot_mapping": None if slot_mapping is None else slot_mapping.detach().cpu().tolist(),
            "block_table_head": None
            if block_table is None
            else block_table[: min(int(block_table.shape[0]), 4), : min(int(block_table.shape[1]), 8)]
            .detach()
            .cpu()
            .tolist(),
            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
            "per_token_max_abs": diff.reshape(diff.shape[0], -1).max(dim=-1).values.detach().cpu().tolist()
            if diff.numel()
            else [],
            "per_token_cosine": cosine.detach().cpu().tolist(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._dspark_attention_diff_records += 1

    def _maybe_dump_standard_kv_diff(
        self,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        request_slots: torch.Tensor | None,
    ) -> None:
        path = _dspark_kv_diff_path()
        if not path or block_table is None:
            return
        if self._dspark_kv_diff_records >= _dspark_kv_diff_max_records():
            return
        if request_slots is None or request_slots.numel() != positions.numel():
            return

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return

        pos_long = positions.to(device=positions.device, dtype=torch.long)
        for block_offset in range(0, positions.numel(), self.block_size):
            if self._dspark_kv_diff_records >= _dspark_kv_diff_max_records():
                break
            block_end = min(block_offset + self.block_size, positions.numel())
            valid_mask = torch.ones(block_end - block_offset, dtype=torch.bool, device=positions.device)
            if slot_mapping is not None:
                block_slots = slot_mapping[block_offset:block_end].to(device=positions.device)
                valid_mask = block_slots >= 0 if block_slots.ndim == 1 else torch.all(block_slots >= 0, dim=-1)
                if block_slots.numel() == 0 or torch.all(~valid_mask):
                    continue

            req_idx = block_offset // self.block_size
            if req_idx >= block_table.shape[0]:
                continue

            valid_pos = pos_long[block_offset:block_end][valid_mask]
            if valid_pos.numel() == 0:
                continue

            end_pos = int(valid_pos.min().item())
            start_pos = max(end_pos - int(self.window_size), 0)
            context_positions = torch.arange(start_pos, end_pos, dtype=torch.long, device=positions.device)
            request_slot = int(request_slots[block_offset].item())
            if context_positions.numel() == 0:
                continue

            cache_indices = context_positions % self._dspark_cache_positions.shape[1]
            cached_positions = self._dspark_cache_positions[request_slot, cache_indices].to(
                device=positions.device,
                dtype=torch.long,
            )
            valid_context = self._dspark_cache_valid[request_slot, cache_indices].to(device=positions.device) & (
                cached_positions == context_positions
            )
            context_positions = context_positions[valid_context]
            if context_positions.numel() == 0:
                continue

            standard_kv = _gather_paged_swa_kv_positions(
                swa_kv_cache,
                block_table,
                req_idx,
                context_positions,
                int(swa_cache_layer.block_size),
            )
            private_k, _ = _gather_context_kv(
                self._dspark_k_cache,
                self._dspark_v_cache,
                self._dspark_cache_positions,
                self._dspark_cache_valid,
                request_slot,
                int(context_positions.min().item()),
                int(context_positions.max().item()),
            )

            private_cmp = private_k[:, :1, :] if private_k.ndim == 3 and private_k.shape[1] != 1 else private_k
            count = min(int(standard_kv.shape[0]), int(private_cmp.shape[0]))
            if count == 0:
                continue
            standard_flat = standard_kv[:count].float().reshape(count, -1)
            private_flat = private_cmp[:count].float().reshape(count, -1)
            denom = standard_flat.norm(dim=-1) * private_flat.norm(dim=-1)
            cosine = (standard_flat * private_flat).sum(dim=-1) / denom.clamp_min(1e-20)
            l2 = (standard_flat - private_flat).norm(dim=-1)

            ctx_cpu = context_positions.detach().cpu()
            cache_block_size = int(swa_cache_layer.block_size)
            block_nums = context_positions // cache_block_size
            block_offsets = context_positions % cache_block_size
            req_block_table = block_table[req_idx].to(device=positions.device, dtype=torch.long)
            slot_ids = req_block_table.index_select(0, block_nums) * cache_block_size + block_offsets
            record = {
                "pid": os.getpid(),
                "rank": os.getenv("RANK"),
                "local_rank": os.getenv("LOCAL_RANK"),
                "prefix": getattr(self.dsa_attn, "prefix", None),
                "req_idx": req_idx,
                "request_slot": request_slot,
                "query_positions": positions[block_offset:block_end].detach().cpu().tolist(),
                "context_len": int(context_positions.numel()),
                "context_positions_head": ctx_cpu[:16].tolist(),
                "context_positions_tail": ctx_cpu[-16:].tolist(),
                "standard_slots_head": slot_ids.detach().cpu()[:16].tolist(),
                "standard_slots_tail": slot_ids.detach().cpu()[-16:].tolist(),
                "standard_len": int(standard_kv.shape[0]),
                "private_len": int(private_cmp.shape[0]),
                "kv_cosine_min": float(cosine.min().item()),
                "kv_cosine_mean": float(cosine.mean().item()),
                "kv_l2_max": float(l2.max().item()),
                "kv_l2_mean": float(l2.mean().item()),
                "first_bad_context_idx": int(torch.argmin(cosine).item()),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            self._dspark_kv_diff_records += 1

    def _run_dspark_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor | None,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.empty_like(q)
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        if request_slots.numel() != positions.numel():
            raise ValueError(
                "DSpark request_slots length must match query positions: "
                f"request_slots={request_slots.numel()}, positions={positions.numel()}"
            )

        assert self._dspark_k_cache is not None
        assert self._dspark_v_cache is not None
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        return dspark_attention(
            q,
            self._dspark_k_cache,
            self._dspark_v_cache,
            self._dspark_cache_positions,
            self._dspark_cache_valid,
            k,
            v,
            request_slots,
            positions,
            self.attn_sink[: self.n_local_heads],
            self.block_size,
            int(self.window_size),
            float(self.scale),
            shared_kv=True,
        )

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor | None = None,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        if positions.numel() == 0:
            return
        shared_kv = self._project_shared_kv(main_x, positions)
        k, v = self._expand_private_kv(shared_kv)
        max_pos = int(positions.max().item())
        self._ensure_dspark_cache(min(max_pos + 1, self._dspark_cache_capacity), k)
        assert self._dspark_k_cache is not None
        assert self._dspark_v_cache is not None
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        slots_long = request_slots.to(torch.long)
        if slots_long.numel() != positions.numel():
            raise ValueError(
                "DSpark request_slots length must match context positions: "
                f"request_slots={slots_long.numel()}, positions={positions.numel()}"
            )
        if int(slots_long.max().item()) >= self._dspark_max_request_slots:
            raise ValueError(
                "DSpark request slot exceeds preallocated cache slots: "
                f"slot={int(slots_long.max().item())}, capacity={self._dspark_max_request_slots}"
            )
        pos_long = positions.to(torch.long)
        cache_indices = pos_long % self._dspark_cache_capacity
        self._dspark_k_cache[slots_long, cache_indices] = k
        self._dspark_v_cache[slots_long, cache_indices] = v
        self._dspark_cache_positions[slots_long, cache_indices] = positions.to(torch.int32)
        self._dspark_cache_valid[slots_long, cache_indices] = True
        self._store_standard_swa_kv(shared_kv, context_slot_mapping)
        self._maybe_dump_standard_kv_write_trace(
            "context",
            positions,
            context_slot_mapping,
            request_slots,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        qr = self.q_norm(_linear_output(self.wq_a, hidden_states))
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))

        q = _linear_output(self.wq_b, qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_norm_without_weight(q)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_dsv4_rope(self.rotary_emb, positions, q_pe)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        shared_kv = torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()
        kv = _maybe_fp8_qdq_nope_dims(
            shared_kv.squeeze(1),
            self.nope_head_dim,
            self._dspark_apply_fp8_qdq,
        )
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat(
            [
                k_nope.unsqueeze(1).expand(-1, self.n_local_heads, -1),
                k_pe.unsqueeze(1).expand(-1, self.n_local_heads, -1),
            ],
            dim=-1,
        ).contiguous()
        v = kv.unsqueeze(1).expand(-1, self.n_local_heads, -1).contiguous()
        standard_slot_mapping = self._standard_query_slot_mapping_from_block_table(
            positions,
            slot_mapping,
            block_table,
            dspark_token_to_req_indices,
        )
        self._store_standard_swa_kv(shared_kv, standard_slot_mapping)
        self._maybe_dump_standard_kv_write_trace(
            "query",
            positions,
            standard_slot_mapping,
            request_slots,
        )
        self._maybe_dump_standard_kv_diff(
            positions,
            standard_slot_mapping,
            block_table,
            request_slots,
        )
        standard_attn_out = self._run_standard_dspark_attention(
            q,
            positions,
            standard_slot_mapping,
            block_table,
            shared_kv,
            request_slots,
            dspark_query_start_loc,
            dspark_seq_lens,
            dspark_token_to_req_indices,
        )
        private_attn_out = None
        if standard_attn_out is not None and _dspark_attention_diff_path():
            private_attn_out = self._run_dspark_attention(q, k, v, positions, request_slots)
        if _dspark_attention_diff_path() and standard_attn_out is not None and private_attn_out is not None:
            self._maybe_dump_standard_attention_diff(
                standard_attn_out,
                private_attn_out,
                positions,
                standard_slot_mapping,
                block_table,
            )
        attn_out = (
            standard_attn_out
            if standard_attn_out is not None
            else self._run_dspark_attention(q, k, v, positions, request_slots)
        )

        attn_out = _apply_dsv4_rope_tail(
            self.rotary_emb,
            positions,
            attn_out,
            inverse=True,
        )
        group_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        attn_out = attn_out.reshape(-1, self.n_local_groups, group_dim)
        attn_out = _maybe_fp8_e4m3fn_qdq(attn_out, self._dspark_apply_fp8_qdq, 128)
        wo_a = _wo_a_weight_for_eager_projection(
            self.wo_a.weight,
            self.n_local_groups,
            self.o_lora_rank,
            group_dim,
        )
        z = _grouped_wo_a_projection(attn_out, wo_a).flatten(1)
        projected = _linear_output(self.wo_b, z)
        return projected


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
            quant_config_override=_draft_quant_config(vllm_config),
            use_quant_config_override=True,
        )
        self.hc_post_alpha = 2.0

    def _mhc_pre(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _dspark_mhc_pre_torch(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            self.norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def _mhc_post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return _dspark_mhc_post_torch(hidden_states, residual, post_mix, res_mix)

    def _mhc_fused_post_pre(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _dspark_mhc_fused_post_pre_torch(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            hc_fn,
            hc_scale,
            hc_base,
            self.norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del llama_4_scaling
        if residual is None:
            residual = hidden_states
            hidden_states, post_mix, res_mix = self._mhc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
            )
        else:
            assert post_mix is not None and res_mix is not None
            residual, post_mix, res_mix, hidden_states = self._mhc_fused_post_pre(
                hidden_states,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
            )
        hidden_states = self.input_layernorm(hidden_states)
        attn_kwargs = {
            "request_slots": request_slots,
            "slot_mapping": slot_mapping,
            "block_table": block_table,
        }
        if dspark_query_start_loc is not None or dspark_seq_lens is not None or dspark_token_to_req_indices is not None:
            attn_kwargs.update(
                dspark_query_start_loc=dspark_query_start_loc,
                dspark_seq_lens=dspark_seq_lens,
                dspark_token_to_req_indices=dspark_token_to_req_indices,
            )
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            None,
            **attn_kwargs,
        )

        residual, post_mix, res_mix, hidden_states = self._mhc_fused_post_pre(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids)
        return hidden_states, residual, post_mix, res_mix


class DSparkMarkovHead(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            config.dspark_markov_rank,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.markov_w2, markov_embed)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        return self.bias(embed), embed


class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
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
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(
            config,
            maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"),
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor
        | list[torch.Tensor | None]
        | tuple[torch.Tensor | None, ...]
        | dict[str, torch.Tensor | None]
        | dict[int, torch.Tensor | None]
        | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        if context_states.numel() == 0:
            return
        main_x = self.main_norm(_linear_output(self.main_proj, context_states))
        for layer_idx, (layer_key, layer) in enumerate(self.layers.items()):
            layer_prefix = _get_layer_prefix(layer, layer_key)
            layer_context_slot_mapping = _select_layer_value(
                context_slot_mapping,
                layer_idx,
                layer_key,
                layer_prefix,
            )
            layer.self_attn.precompute_context_kv(
                main_x,
                context_positions,
                request_slots=context_request_slots,
                context_slot_mapping=layer_context_slot_mapping,
            )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        for layer in self.layers.values():
            layer.self_attn.reset_request_slots(request_slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | dict[str, torch.Tensor] | dict[int, torch.Tensor] | None = None,
        block_table: torch.Tensor | dict[str, torch.Tensor] | dict[int, torch.Tensor] | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = None
        for layer_idx, (layer_key, layer) in enumerate(self.layers.items()):
            layer_prefix = _get_layer_prefix(layer, layer_key)
            layer_kwargs = {
                "positions": positions,
                "hidden_states": hidden_states,
                "residual": residual,
                "post_mix": post_mix,
                "res_mix": res_mix,
                "input_ids": input_ids,
                "request_slots": request_slots,
                "slot_mapping": _select_layer_value(slot_mapping, layer_idx, layer_key, layer_prefix),
                "block_table": _select_layer_value(block_table, layer_idx, layer_key, layer_prefix),
            }
            if (
                dspark_query_start_loc is not None
                or dspark_seq_lens is not None
                or dspark_token_to_req_indices is not None
            ):
                layer_kwargs.update(
                    dspark_query_start_loc=dspark_query_start_loc,
                    dspark_seq_lens=dspark_seq_lens,
                    dspark_token_to_req_indices=dspark_token_to_req_indices,
                )
            layer_output = layer(**layer_kwargs)
            if isinstance(layer_output, tuple) and len(layer_output) == 4:
                hidden_states, residual, post_mix, res_mix = layer_output
            else:
                hidden_states = layer_output
        head_hidden = self.compute_head_hidden(hidden_states, residual, post_mix, res_mix)
        return head_hidden

    def compute_head_hidden(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None and post_mix is not None and res_mix is not None:
            hidden_states = _dspark_mhc_post_torch(hidden_states, residual, post_mix, res_mix)
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

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        head_hidden = self.compute_head_hidden(hidden_states)
        return logits_processor(lm_head, self.norm(head_hidden))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return _make_deepseek_v4_expert_params_mapping(
            self,
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            finalize = getattr(layer.mlp, "finalize_mega_moe_weights", None)
            if finalize is not None:
                finalize()


@support_torch_compile
class DeepSeekV4DSparkMTP(nn.Module, DeepseekV2MixtureOfExperts):
    # DSpark draft embed/head are aliases of the target model, matching
    # upstream vLLM's DSparkDeepseekV4ForCausalLM contract.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = _draft_quant_config(vllm_config)
        self.model = DeepseekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.set_moe_parameters_from_layers(self.config, self.model.layers.values())

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
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors, spec_step_idx
        assert input_ids is not None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            request_slots=request_slots,
            slot_mapping=slot_mapping,
            block_table=block_table,
            dspark_query_start_loc=dspark_query_start_loc,
            dspark_seq_lens=dspark_seq_lens,
            dspark_token_to_req_indices=dspark_token_to_req_indices,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        del spec_step_idx
        return self.model.compute_logits(
            hidden_states,
            self.lm_head,
            self.logits_processor,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor
        | list[torch.Tensor | None]
        | tuple[torch.Tensor | None, ...]
        | dict[str, torch.Tensor | None]
        | dict[int, torch.Tensor | None]
        | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
            context_request_slots,
        )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        self.model.reset_request_slots(request_slots)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        missing_mtp_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        expert_mapping = self.model.get_expert_mapping()
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
        )
        start_layer_idx = self.config.num_hidden_layers
        last_layer_idx = start_layer_idx + self.model.num_dspark_layers - 1

        for name, loaded_weight in weights:
            if name == "embed.weight":
                embed_name = "model.embed_tokens.weight"
                param = params_dict[embed_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(embed_name)
                continue
            if name == "head.weight":
                head_name = "lm_head.weight"
                param = params_dict[head_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(head_name)
                continue
            mapped_name = self._remap_dspark_name(name)
            if mapped_name is None:
                continue
            name = mapped_name
            if name.startswith(f"model.layers.{last_layer_idx}.hc_head_"):
                canonical_name = name.replace(f"model.layers.{last_layer_idx}.", "model.", 1)
                if canonical_name in params_dict:
                    name = canonical_name
            if name.endswith(".scale"):
                suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale"
                name = name.removesuffix(".scale") + suffix
                if name not in params_dict:
                    continue
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if ".experts." in name or f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    missing_mtp_params.add(mapped)
                    break
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(mapped)
                break
            else:
                if ".experts." in name:
                    matched_expert_mapping = False
                    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
                        if weight_name not in name:
                            continue
                        matched_expert_mapping = True
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
                    if not matched_expert_mapping:
                        missing_mtp_params.add(name)
                    continue
                if "attn_sink" in name:
                    if name not in params_dict:
                        missing_mtp_params.add(name)
                        continue
                    narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name not in params_dict:
                    missing_mtp_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if missing_mtp_params:
            raise ValueError(
                "DSpark speculative decoding checkpoint weights did not match model parameters: "
                f"{sorted(missing_mtp_params)}"
            )

        loaded_layer_ids: set[int] = set()
        for param_name in loaded_params:
            match = _LAYER_ID_RE.search(param_name)
            if match:
                loaded_layer_ids.add(int(match.group(1)))
        for layer_idx in range(start_layer_idx, start_layer_idx + self.model.num_dspark_layers):
            if layer_idx not in loaded_layer_ids:
                raise ValueError(f"DSpark speculative decoding layer {layer_idx} weights missing from checkpoint.")
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
        self.model.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_dspark_name(self, name: str) -> str | None:
        match = re.match(r"mtp\.(\d+)\.(.*)", name)
        if match is None:
            return None
        stage_idx = int(match.group(1))
        layer_idx = self.config.num_hidden_layers + stage_idx
        rest = match.group(2)
        if rest.startswith("confidence_head."):
            return None
        name = f"model.layers.{layer_idx}.{rest}"
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        name = name.replace(".mlp.gate.bias", ".mlp.gate.e_score_correction_bias")
        return name


DSparkDeepseekV4ForCausalLM = DeepSeekV4DSparkMTP
