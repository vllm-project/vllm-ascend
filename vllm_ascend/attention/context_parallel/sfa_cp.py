from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar

import torch
import torch_npu
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec

import vllm_ascend.ops.triton.sfa_cp  # noqa: F401
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import (
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    get_dcp_local_seq_lens,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    SFAForwardContext,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.quantization.tp_weight_switch import TPWeightSwitchMixin
from vllm_ascend.utils import (
    _round_up,
    enable_dsa_cp,
    enable_dsa_cp_with_o_proj_tp,
    enable_sfa_dcp_replicated_indexer,
    vllm_version_is,
)

if vllm_version_is("0.28.0"):
    from vllm.model_executor.layers.attention.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]
else:
    from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]

M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFAPCPImpl(AscendSFAImpl):
    def _get_sfa_kv_slot_mapping(
        self,
        attn_metadata: M,
    ) -> torch.Tensor:
        assert attn_metadata.pcp_slot_mapping is not None
        return attn_metadata.pcp_slot_mapping

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        num_decode_tokens = attn_metadata.num_decode_tokens or 0
        (kv_no_split, cos, sin), slots = _gather_prefill_cache_inputs((kv_no_split, cos, sin), slots, num_decode_tokens)
        assert slots.numel() == kv_no_split.shape[0], (
            "SFA PCP cache write requires one slot per gathered token: "
            f"tokens={kv_no_split.shape[0]}, slots={slots.numel()}."
        )

        return super().exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)

    def _write_indexer_cache(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: M,
    ) -> None:
        num_decode_tokens = attn_metadata.num_decode_tokens or 0
        tensors = (k_li,) if k_li_scale is None else (k_li, k_li_scale)
        gathered_tensors, gathered_slot_mapping = _gather_prefill_cache_inputs(tensors, slot_mapping, num_decode_tokens)
        k_li = gathered_tensors[0]
        assert gathered_slot_mapping.numel() == k_li.shape[0], (
            "SFA PCP indexer cache write requires one slot per gathered token: "
            f"tokens={k_li.shape[0]}, slots={gathered_slot_mapping.numel()}."
        )
        if k_li_scale is not None:
            k_li_scale = gathered_tensors[1]
        super()._write_indexer_cache(
            k_li,
            k_li_scale,
            gathered_slot_mapping,
            kv_cache,
            attn_metadata,
        )


@dataclass
class DSACPContext:
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    slot_mapping_cp: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


@dataclass
class AscendSFADSACPMetadata(AscendSFAMetadata):
    """SFA metadata fields used only by the DSA-CP execution path."""

    dsa_cp_context: DSACPContext | None = None


class DCPGatherContext(NamedTuple):
    """State needed to finish an async fused DCP KV all-gather."""

    gathered: torch.Tensor
    handle: torch.distributed.Work | None
    restore_perm: tuple[int, ...] | None
    split_sizes: tuple[int, ...]


@dataclass
class DCPContext:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_gather_block_ids: torch.Tensor | None = None
    kv_gather_block_table: torch.Tensor | None = None
    gather_context: DCPGatherContext | None = None


@dataclass
class AscendSFADCPMetadata(AscendSFAMetadata):
    """SFA metadata fields used only by the DCP execution path."""

    dcp_context: DCPContext | None = None


@dataclass
class AscendSFADSADCPMetadata(AscendSFADCPMetadata):
    """SFA metadata for the combined DSA-CP and DCP execution path."""

    dsa_cp_context: DSACPContext | None = None


class AscendSFADSACPMetadataBuilder(AscendSFAMetadataBuilder):
    """Adds TP-token-sharded DSA-CP metadata to the shared SFA builder."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls or AscendSFADSACPMetadata,
            supports_dcp_with_varlen,
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.dsa_cp_actual_seq_lengths_query = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.dsa_cp_actual_seq_lengths_key = torch.empty_like(self.dsa_cp_actual_seq_lengths_query)
        self.dsa_cp_spec_actual_seq_lengths_query: list[torch.Tensor] | None = None
        self.dsa_cp_spec_actual_seq_lengths_key: list[torch.Tensor] | None = None
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.dsa_cp_spec_actual_seq_lengths_query = [
                torch.zeros(max_num_reqs * (spec_token_num + 1) + 1, dtype=torch.int32, device=device)
                for _ in range(spec_token_num)
            ]
            self.dsa_cp_spec_actual_seq_lengths_key = [
                torch.zeros(max_num_reqs * (spec_token_num + 1) + 1, dtype=torch.int32, device=device)
                for _ in range(spec_token_num)
            ]

    def _prepare_parallel_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        cum_query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        draft_index: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        cos, sin, slot_mapping, extra = super()._prepare_parallel_metadata(
            common_attn_metadata,
            cos,
            sin,
            slot_mapping,
            cum_query_lens,
            seq_lens,
            draft_index,
        )
        global_tp_size = get_tp_group().world_size
        num_tokens = common_attn_metadata.num_input_tokens
        num_tokens_pad = _round_up(num_tokens, global_tp_size)
        num_tokens_per_device = num_tokens_pad // global_tp_size
        local_start = get_tp_group().rank_in_group * num_tokens_per_device
        local_end_with_pad = local_start + num_tokens_per_device
        local_end = min(local_end_with_pad, common_attn_metadata.num_actual_tokens)

        assert cos.shape == sin.shape, f"cos.shape must equal sin.shape, got {cos.shape} and {sin.shape}"
        pad_size = num_tokens_pad - cos.shape[0]
        if pad_size > 0:
            cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
            sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))
        pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
        if pad_size_slot > 0:
            slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size_slot), value=-1)
        else:
            slot_mapping = slot_mapping[:num_tokens_pad]

        slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]
        cos = cos[local_start:local_end_with_pad]
        sin = sin[local_start:local_end_with_pad]
        assert cos.shape[0] == num_tokens_per_device
        assert slot_mapping_cp.shape[0] == num_tokens_per_device
        assert slot_mapping.shape[0] == num_tokens_pad

        if draft_index is not None:
            assert self.dsa_cp_spec_actual_seq_lengths_query is not None
            assert self.dsa_cp_spec_actual_seq_lengths_key is not None
            actual_seq_lengths_query = self.dsa_cp_spec_actual_seq_lengths_query[draft_index - 1]
            actual_seq_lengths_key = self.dsa_cp_spec_actual_seq_lengths_key[draft_index - 1]
        else:
            actual_seq_lengths_query = self.dsa_cp_actual_seq_lengths_query
            actual_seq_lengths_key = self.dsa_cp_actual_seq_lengths_key

        num_segs = cum_query_lens.shape[0]
        global_start = common_attn_metadata.query_start_loc[:num_segs]
        global_end = cum_query_lens
        req_local_start = global_start.clamp(min=local_start)
        req_local_end = global_end.clamp(max=local_end_with_pad)
        num_local_tokens = req_local_end - req_local_start
        local_query_lens = torch.cumsum(num_local_tokens.clamp(min=0), dim=0)
        offset = global_end - req_local_end
        local_key_lens = torch.where(
            num_local_tokens > 0,
            torch.clamp_min(seq_lens - offset, 0),
            0,
        )
        actual_seq_lengths_query[:num_segs] = local_query_lens
        actual_seq_lengths_key[:num_segs] = local_key_lens

        extra["dsa_cp_context"] = DSACPContext(
            num_tokens=num_tokens,
            num_tokens_pad=num_tokens_pad,
            local_start=local_start,
            local_end=local_end,
            local_end_with_pad=local_end_with_pad,
            slot_mapping_cp=slot_mapping_cp,
            actual_seq_lengths_query=actual_seq_lengths_query[: common_attn_metadata.num_reqs],
            actual_seq_lengths_key=actual_seq_lengths_key[: common_attn_metadata.num_reqs],
        )
        return cos, sin, slot_mapping, extra

    def _update_parallel_slot_mapping(
        self,
        metadata: AscendSFAMetadata,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> None:
        super()._update_parallel_slot_mapping(metadata, slot_mapping, num_input_tokens)
        dsa_cp_context = getattr(metadata, "dsa_cp_context", None)
        if dsa_cp_context is None:
            return
        local_mapping = slot_mapping[:num_input_tokens]
        if dsa_cp_context.num_tokens_pad > local_mapping.shape[0]:
            local_mapping = nn.functional.pad(
                local_mapping,
                (0, dsa_cp_context.num_tokens_pad - local_mapping.shape[0]),
                value=-1,
            )
        else:
            local_mapping = local_mapping[: dsa_cp_context.num_tokens_pad]
        dsa_cp_context.slot_mapping_cp = local_mapping[dsa_cp_context.local_start : dsa_cp_context.local_end_with_pad]


class AscendSFADSACPImpl(AscendSFAImpl):
    """SFA implementation for DSA-CP token sharding in the TP group."""

    o_proj_full_pools: dict[Any, torch.Tensor] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_num_heads = self.num_heads * self.tp_size
        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        result = super().process_weights_after_loading(act_dtype)
        self._o_proj_tp_weight_switch_enabled = False
        if self.enable_dsa_cp_with_o_proj_tp:
            self._enable_o_proj_tp_full_weight_switch()
        return result

    def _get_fused_type_unsupported_reasons(self, pp_type):
        reasons = super()._get_fused_type_unsupported_reasons(pp_type)
        reasons.insert(0, "Fused preprocessing does not support DSA-CP.")
        return reasons

    def _parallel_query_gather_dim(self) -> int:
        return 0

    def _prepare_native_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: M,
    ) -> torch.Tensor:
        context = getattr(attn_metadata, "dsa_cp_context", None)
        assert context is not None, "DSA-CP requires attn_metadata.dsa_cp_context."
        actual_tokens = hidden_states.shape[0]
        if actual_tokens > context.num_tokens_pad:
            raise RuntimeError(
                "SFA DSA-CP input exceeds its TP-aligned metadata, "
                f"got {actual_tokens} tokens and num_tokens_pad={context.num_tokens_pad}."
            )
        if actual_tokens < context.num_tokens_pad:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, context.num_tokens_pad - actual_tokens))
        return hidden_states[context.local_start : context.local_end_with_pad]

    def _get_parallel_forward_context(
        self,
        attn_metadata: M,
        num_input_tokens: int,
        hidden_states: torch.Tensor,
    ) -> SFAForwardContext:
        context = getattr(attn_metadata, "dsa_cp_context", None)
        assert context is not None, "DSA-CP requires attn_metadata.dsa_cp_context."
        gather_full_o_proj = (
            self.tp_size > 1
            and self.enable_dsa_cp_with_o_proj_tp
            and attn_metadata.attn_state
            not in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            }
        )
        return SFAForwardContext(
            actual_seq_lengths_query=context.actual_seq_lengths_query,
            actual_seq_lengths_key=context.actual_seq_lengths_key,
            kv_slot_mapping=context.slot_mapping_cp,
            topk_num_tokens=context.local_end_with_pad - context.local_start,
            gather_full_o_proj=gather_full_o_proj,
        )

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        if self.enable_sparse_sfa_c8:
            return super().exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
        kv_a_layernorm = self.kv_a_layernorm
        assert kv_a_layernorm is not None, "kv_a_layernorm must be initialized for DSA-CP KV preprocessing"
        B = kv_no_split.shape[0]
        kv_no_split = kv_no_split.view(B, self.num_kv_heads, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            kv_a_layernorm.weight,
            cos,
            sin,
            slots.to(torch.int64),
            kv_cache[1],
            kv_cache[0],
            epsilon=kv_a_layernorm.variance_epsilon,
            cache_mode="PA",
            is_output_kv=True,
        )
        return k_pe, k_nope, None

    def _prepare_kv_for_parallel(
        self,
        k_pe,
        k_nope,
        knope_scale,
        k_li,
        k_li_scale,
        full_gather_o_proj_enabled,
    ):
        assert k_pe is not None and k_nope is not None
        async_op = full_gather_o_proj_enabled
        handles: list[torch.distributed.Work] = []
        if self.enable_sparse_sfa_c8:
            assert knope_scale is not None
            parts = [
                k_nope.view(-1, k_nope.shape[-1]),
                k_pe.view(-1, k_pe.shape[-1]),
                knope_scale.view(-1, knope_scale.shape[-1]),
            ]
        else:
            parts = [k_pe.view(-1, k_pe.shape[-1]), k_nope.view(-1, k_nope.shape[-1])]
            if self.has_indexer and not self.enable_sparse_li_c8:
                assert k_li is not None
                parts.append(k_li.view(-1, k_li.shape[-1]))
        fused_kv, handle = all_gather_async(torch.cat(parts, dim=1), get_tp_group(), async_op=async_op)
        if handle is not None:
            handles.append(handle)
        if self.has_indexer and (self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8):
            assert k_li is not None
            k_li, handle = all_gather_async(k_li, get_tp_group(), async_op=async_op)
            if handle is not None:
                handles.append(handle)
        if self.has_indexer and self.enable_sparse_li_c8:
            assert k_li_scale is not None
            k_li_scale, handle = all_gather_async(k_li_scale, get_tp_group(), async_op=async_op)
            if handle is not None:
                handles.append(handle)
        return k_li, k_li_scale, fused_kv, handles

    def _store_parallel_kv(
        self,
        k_pe,
        k_nope,
        knope_scale,
        k_li,
        fused_kv_no_split,
        kv_ag_handles,
        kv_cache,
        slot_mapping_sfa,
        attn_metadata,
        full_gather_o_proj_enabled,
    ):
        for handle in kv_ag_handles:
            handle.wait()
        if full_gather_o_proj_enabled:
            self._enable_o_proj_tp_full_weight_switch()
            linear_method = self._get_o_proj_linear_method()
            assert isinstance(linear_method, TPWeightSwitchMixin)
            assert self.o_proj_tp_weight_state is not None
            linear_method.all_gather_tp_weight(
                self.o_proj_tp_weight_state,
                get_tp_group(),
            )

        if kv_cache is not None:
            assert fused_kv_no_split is not None
            if self.enable_sparse_sfa_c8:
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[0].view(-1, fused_kv_no_split.shape[-1]),
                    slot_mapping_sfa[: attn_metadata.num_actual_tokens].view(-1, 1),
                    fused_kv_no_split[: attn_metadata.num_actual_tokens],
                )
                k_pe = k_nope = None
            elif not self.has_indexer:
                k_pe, k_nope = fused_kv_no_split.split([self.qk_rope_head_dim, self.kv_lora_rank], dim=-1)
            elif not self.enable_sparse_li_c8:
                k_pe, k_nope, k_li = fused_kv_no_split.split(
                    [self.qk_rope_head_dim, self.kv_lora_rank, self.head_dim], dim=-1
                )
            else:
                k_pe, k_nope = fused_kv_no_split.split([self.qk_rope_head_dim, self.kv_lora_rank], dim=-1)
            if not self.enable_sparse_sfa_c8:
                assert k_pe is not None and k_nope is not None
                k_nope = k_nope.view(k_nope.shape[0], 1, -1)
                k_pe = k_pe.view(k_pe.shape[0], 1, -1)
                DeviceOperator.reshape_and_cache(
                    key=k_nope[: attn_metadata.num_actual_tokens],
                    value=k_pe[: attn_metadata.num_actual_tokens],
                    key_cache=kv_cache[0],
                    value_cache=kv_cache[1],
                    slot_mapping=slot_mapping_sfa[: attn_metadata.num_actual_tokens],
                )
        return k_pe, k_nope, k_li

    def _enable_o_proj_tp_full_weight_switch(self) -> None:
        if self._o_proj_tp_weight_switch_enabled:
            return

        linear_method = self._get_o_proj_linear_method()
        if not isinstance(linear_method, TPWeightSwitchMixin) or not linear_method.supports_tp_weight_switch:
            raise RuntimeError(
                "SFA DSA-CP o_proj full-weight switching requires a TP weight-switch capable method, "
                f"got {type(linear_method).__name__}."
            )
        self.o_proj_tp_weight_state = linear_method.enable_tp_weight_switch(
            self.o_proj,
            self.tp_size,
            pool=AscendSFADSACPImpl.o_proj_full_pools,
            pool_key_prefix=(type(linear_method).__qualname__, "sfa_o_proj"),
        )
        self._o_proj_tp_weight_switch_enabled = True

    def _get_o_proj_linear_method(self):
        quant_method = self.o_proj.quant_method
        return getattr(quant_method, "quant_method", quant_method)

    def _apply_o_proj_full_weight(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self._get_o_proj_linear_method().apply(self.o_proj, attn_output)

    def _finalize_o_proj(
        self,
        attn_output,
        output,
        gather_full_o_proj,
    ):
        if not self.enable_dsa_cp_with_o_proj_tp:
            return super()._finalize_o_proj(
                attn_output,
                output,
                gather_full_o_proj,
            )
        if gather_full_o_proj:
            linear_method = self._get_o_proj_linear_method()
            assert isinstance(linear_method, TPWeightSwitchMixin)
            assert self.o_proj_tp_weight_state is not None
            linear_method.wait_tp_weight_all_gather(self.o_proj_tp_weight_state)
            linear_method.switch_tp_weight(
                self.o_proj,
                self.o_proj_tp_weight_state,
                use_full_weight=True,
            )
            try:
                local_output = self._apply_o_proj_full_weight(attn_output)
                full_output = get_tp_group().all_gather(local_output.contiguous(), dim=0)
                if full_output.shape[0] < output.shape[0] or full_output.shape[1:] != output.shape[1:]:
                    raise RuntimeError(
                        "SFA DSA-CP gathered output does not match the replicated "
                        f"model state, got {tuple(full_output.shape)} and expected "
                        f"{tuple(output.shape)}."
                    )
                output[...] = full_output[: output.shape[0]]
            finally:
                linear_method.switch_tp_weight(
                    self.o_proj,
                    self.o_proj_tp_weight_state,
                    use_full_weight=False,
                )
            return output

        send = (
            attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
            .permute(1, 0, 2)
            .reshape(-1, self.num_heads * self.v_head_dim)
        )
        sharded_output = torch.empty_like(send)
        torch.distributed.all_to_all_single(sharded_output, send, group=get_tp_group().device_group)
        projected_output = self.o_proj(sharded_output)[0]
        if projected_output.shape[0] < output.shape[0] or projected_output.shape[1:] != output.shape[1:]:
            raise RuntimeError(
                "SFA DSA-CP projected output does not match the replicated "
                f"model state, got {tuple(projected_output.shape)} and expected "
                f"{tuple(output.shape)}."
            )
        output[...] = projected_output[: output.shape[0]]
        return output


# SFA DCP replicated-indexer layout:
#
# - LightningIndexer cache is replicated on every DCP rank so index selection
#   can run against the full sequence and keep the same sparse topk semantics as
#   non-DCP SFA.
# - SFA KV cache remains DCP-local to preserve the KV memory saving. The sparse
#   topk indices produced from the replicated indexer view are remapped to local
#   KV indices before calling sparse flash attention.
# - BlockTable only owns the DCP-local physical layout. This builder derives the
#   replicated block table and slot mapping on demand, temporarily builds the
#   indexer-facing metadata with that replicated view, and then stores the
#   original DCP-local view in metadata.dcp_context for KV writes and SFA reads.
# - The replicated view uses the same logical/kernel block size as BlockTable,
#   including hybrid block splitting.
class AscendSFADCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendSFAMetadataBuilder,
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        metadata_cls = metadata_cls or AscendSFADCPMetadata
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls,
            supports_dcp_with_varlen,
        )
        self.cp_kv_cache_interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        assert self.dcp_size > 1, "AscendSFADCPMetadataBuilder requires DCP world size > 1."
        if self.cp_kv_cache_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self.cp_kv_cache_interleave_size}")

        # Full-graph FIA padding can append one dummy request.
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs + 1
        self.dcp_local_seq_lens_buf = torch.empty(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.replicated_view_block_size = self.kernel_block_size
        if kv_cache_spec.block_size % self.replicated_view_block_size != 0:
            raise RuntimeError(
                "SFA replicated view requires the KV cache block size "
                f"({kv_cache_spec.block_size}) to be divisible by "
                f"{self.replicated_view_block_size}."
            )
        self.blocks_per_phys_block = kv_cache_spec.block_size // self.replicated_view_block_size
        max_num_input_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_model_len = vllm_config.model_config.max_model_len
        total_cp_size = self.dcp_size
        # The generic vLLM BlockTable may expose global-width storage, while
        # the DCP physical KV layout only populates rank-local block columns.
        self.max_local_block_table_cols = (
            cdiv(max_model_len, kv_cache_spec.block_size * total_cp_size) * self.blocks_per_phys_block
        )
        max_replicated_block_table_cols = self.max_local_block_table_cols * total_cp_size
        self.block_table_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_reqs, max_replicated_block_table_cols),
            dtype=torch.int32,
            device=device,
        )
        self.arange_buffer: torch.Tensor = torch.arange(
            max_replicated_block_table_cols,
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_input_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def _get_dcp_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return get_dcp_local_seq_lens(
            seq_lens,
            self.dcp_size,
            self.cp_kv_cache_interleave_size,
        )[:, self.dcp_rank]

    def _get_dcp_local_block_table(self, block_table: torch.Tensor, num_reqs: int) -> torch.Tensor:
        local_cols = min(block_table.shape[1], self.max_local_block_table_cols)
        return block_table[:num_reqs, :local_cols]

    def _ensure_replicated_view_buffers(
        self,
        num_reqs: int,
        num_input_tokens: int,
        local_block_table_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_table_cols = local_block_table_cols * self.dcp_size
        if (
            self.block_table_replicated_view_buf.shape[0] < num_reqs
            or self.block_table_replicated_view_buf.shape[1] < block_table_cols
        ):
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"block_table_replicated_view_buf.shape={self.block_table_replicated_view_buf.shape}, "
                f"num_reqs={num_reqs}, block_table_cols={block_table_cols}"
            )
        if self.slot_mapping_replicated_view_buf.shape[0] < num_input_tokens:
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"slot_mapping_replicated_view_buf.shape={self.slot_mapping_replicated_view_buf.shape}, "
                f"num_input_tokens={num_input_tokens}"
            )
        return (
            self.block_table_replicated_view_buf[:num_reqs, :block_table_cols],
            self.arange_buffer[:block_table_cols],
            self.slot_mapping_replicated_view_buf[:num_input_tokens],
        )

    def _build_block_table_replicated_view(
        self,
        dcp_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = dcp_block_table.shape[0]
        local_block_table_cols = dcp_block_table.shape[1]
        block_table_replicated_view, replicated_col_idx, _ = self._ensure_replicated_view_buffers(
            num_reqs,
            0,
            local_block_table_cols,
        )

        total_cp_size = self.dcp_size
        blocks_per_phys_block = self.blocks_per_phys_block
        local_col_idx = (
            replicated_col_idx // (total_cp_size * blocks_per_phys_block) * blocks_per_phys_block
            + replicated_col_idx % blocks_per_phys_block
        )
        rank_in_replicated_view = (replicated_col_idx // blocks_per_phys_block) % total_cp_size

        local_logical_blocks = torch.index_select(dcp_block_table, 1, local_col_idx)
        if blocks_per_phys_block == 1:
            replicated_blocks = local_logical_blocks * total_cp_size + rank_in_replicated_view
        else:
            local_sub_blocks = local_logical_blocks % blocks_per_phys_block
            local_phys_blocks = local_logical_blocks // blocks_per_phys_block
            replicated_blocks = (
                local_phys_blocks * total_cp_size + rank_in_replicated_view
            ) * blocks_per_phys_block + local_sub_blocks

        valid_req_mask = (seq_lens[:num_reqs].to(device=self.device) > 0).to(replicated_blocks.dtype).view(-1, 1)
        replicated_blocks = replicated_blocks * valid_req_mask
        block_table_replicated_view.copy_(replicated_blocks)
        return block_table_replicated_view

    def _build_slot_mapping_replicated_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        block_table_replicated_view: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens)
        local_block_table_cols = block_table_replicated_view.shape[1] // self.dcp_size
        _, _, slot_mapping_replicated_view = self._ensure_replicated_view_buffers(
            num_reqs,
            num_input_tokens,
            local_block_table_cols,
        )
        slot_mapping_replicated_view.fill_(-1)
        if num_actual_tokens == 0:
            return slot_mapping_replicated_view

        query_lens = (
            common_attn_metadata.query_start_loc[1 : num_reqs + 1] - common_attn_metadata.query_start_loc[:num_reqs]
        )
        req_indices = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=self.device),
            query_lens.to(device=self.device),
            output_size=num_input_tokens,
        )[:num_actual_tokens]
        if req_indices.numel() == 0:
            return slot_mapping_replicated_view

        num_actual_tokens = min(num_actual_tokens, req_indices.shape[0])
        req_indices = req_indices[:num_actual_tokens]
        positions = common_attn_metadata.positions[:num_actual_tokens].to(
            device=self.device,
            dtype=torch.int32,
        )
        logical_block_idx = positions // self.replicated_view_block_size
        block_offsets = positions % self.replicated_view_block_size
        block_table_indices = req_indices * block_table_replicated_view.shape[1] + logical_block_idx
        block_numbers = block_table_replicated_view.flatten()[block_table_indices]
        slot_mapping_replicated_view[:num_actual_tokens] = (
            block_numbers * self.replicated_view_block_size + block_offsets
        )
        return slot_mapping_replicated_view

    def _build_compact_kv_gather_metadata(
        self,
        dcp_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the compact cross-DCP KV view used by prefill attention."""
        valid_block_ids, compact_block_table = dcp_block_table.flatten().unique(return_inverse=True)
        compact_block_table = compact_block_table.view_as(dcp_block_table)
        num_blocks = valid_block_ids.shape[0]
        dcp_rank_arange = self.arange_buffer[: self.dcp_size]
        remapped_block_table = (
            compact_block_table.unsqueeze(-1) + (dcp_rank_arange * num_blocks).view(1, 1, -1).to(dcp_block_table)
        ).reshape(dcp_block_table.shape[0], -1)
        return valid_block_ids, remapped_block_table.to(torch.int32)

    def _build_with_metadata_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        build_metadata: Callable[[], AscendSFAMetadata],
    ) -> AscendSFAMetadata:
        dcp_slot_mapping = common_attn_metadata.slot_mapping
        full_dcp_block_table = common_attn_metadata.block_table_tensor
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        dcp_block_table = self._get_dcp_local_block_table(full_dcp_block_table, num_reqs)
        block_table_replicated_view = self._build_block_table_replicated_view(
            dcp_block_table,
            common_attn_metadata.seq_lens,
        )
        slot_mapping_replicated_view = self._build_slot_mapping_replicated_view(
            common_attn_metadata,
            block_table_replicated_view,
        )

        common_attn_metadata.slot_mapping = slot_mapping_replicated_view
        common_attn_metadata.block_table_tensor = block_table_replicated_view
        try:
            metadata = build_metadata()
        finally:
            common_attn_metadata.slot_mapping = dcp_slot_mapping
            common_attn_metadata.block_table_tensor = full_dcp_block_table

        assert isinstance(metadata, AscendSFADCPMetadata)
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens
        if dcp_local_seq_lens is None:
            dcp_local_seq_lens = self._get_dcp_local_seq_lens(metadata.seq_lens)
        local_seq_lens_src = dcp_local_seq_lens[:num_reqs].to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=True,
        )
        self.dcp_local_seq_lens_buf[:num_reqs].copy_(local_seq_lens_src, non_blocking=True)
        local_seq_lens = self.dcp_local_seq_lens_buf[:num_reqs]

        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=False,
        )
        kv_gather_block_ids = None
        kv_gather_block_table = None
        if num_prefills > 0:
            kv_gather_block_ids, kv_gather_block_table = self._build_compact_kv_gather_metadata(dcp_block_table)
        metadata.dcp_context = DCPContext(
            slot_mapping=dcp_slot_mapping[:num_input_tokens],
            block_table=dcp_block_table,
            seq_lens=local_seq_lens,
            kv_gather_block_ids=kv_gather_block_ids,
            kv_gather_block_table=kv_gather_block_table,
        )
        metadata.num_decodes = num_decodes
        metadata.num_decode_tokens = num_decode_tokens
        metadata.num_prefills = num_prefills
        self._update_parallel_slot_mapping(metadata, dcp_slot_mapping, num_input_tokens)
        return metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ) -> AscendSFAMetadata:
        if attn_state not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendSFADCPImpl(DCPImplMixin, AscendSFAImpl):
    can_return_lse_for_decode: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        # DCP shards only the SFA KV cache. MLAPO writes the SFA KV cache
        # internally, so keep DCP on the native path where we pass the DCP
        # slot mapping explicitly.
        self.enable_mlapo = False
        self._dcp_interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        if self._dcp_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self._dcp_interleave_size}")
        self._dcp_index_topk = 0
        for config in (
            getattr(self.vllm_config.model_config, "hf_text_config", None),
            getattr(self.vllm_config.model_config, "hf_config", None),
        ):
            index_topk = getattr(config, "index_topk", None)
            if isinstance(index_topk, int) and index_topk > 0:
                self._dcp_index_topk = index_topk
                break
        if self._dcp_index_topk <= 0:
            raise RuntimeError("index_topk must be set in the model config for DCP SFA.")
        device = self.q_proj.weight.device
        self._remap_order = torch.arange(self._dcp_index_topk, dtype=torch.float32, device=device)
        self._remap_invalid_index = torch.tensor(-1.0, dtype=torch.float32, device=device)

    @staticmethod
    def _has_prefill(attn_metadata: M) -> bool:
        return attn_metadata.num_prefills > 0

    def _record_dcp_kv_gather_context(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
    ) -> None:
        """Start the compact KV all-gather used by prefill/mixed DCP batches."""
        if not self._has_prefill(attn_metadata):
            return
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."

        valid_block_ids = attn_metadata.dcp_context.kv_gather_block_ids
        block_table = attn_metadata.dcp_context.kv_gather_block_table
        assert valid_block_ids is not None and block_table is not None
        kv = torch.index_select(kv_cache[0], 0, valid_block_ids)
        split_sizes: tuple[int, ...]
        if self.enable_sparse_sfa_c8:
            # Sparse C8 stores nope, rope, and quantization data in one packed
            # SFA KV cache. The remaining cache entries belong to the indexer
            # and must not participate in the DCP SFA KV all-gather.
            gather_input = kv.contiguous()
            split_sizes = (kv.shape[-1],)
        else:
            if len(kv_cache) < 2:
                raise RuntimeError("DCP SFA KV all-gather requires nope and rope KV caches.")
            key_rope = torch.index_select(kv_cache[1], 0, valid_block_ids)
            if kv.shape[:-1] != key_rope.shape[:-1] or kv.dtype != key_rope.dtype:
                raise RuntimeError(
                    "Cannot fuse DCP KV gather for KV/nope and KV/rope caches with "
                    f"shapes {tuple(kv.shape)} / {tuple(key_rope.shape)} and dtypes {kv.dtype} / {key_rope.dtype}."
                )
            gather_input = torch.cat([kv, key_rope], dim=-1).contiguous()
            split_sizes = (kv.shape[-1], key_rope.shape[-1])
        attn_metadata.dcp_context.gather_context = self._start_dcp_gather(
            gather_input,
            dim=0,
            split_sizes=split_sizes,
        )

    def _start_dcp_gather(
        self,
        x: torch.Tensor,
        dim: int,
        split_sizes: tuple[int, ...],
    ) -> DCPGatherContext:
        gathered, handle, restore_perm = self._all_gather_dim_async(x, dim)
        return DCPGatherContext(
            gathered=gathered,
            handle=handle,
            restore_perm=restore_perm,
            split_sizes=split_sizes,
        )

    @staticmethod
    def _finish_dcp_gather(
        context: DCPGatherContext,
    ) -> tuple[torch.Tensor, ...]:
        if context.handle is not None:
            context.handle.wait()
        gathered = context.gathered
        if context.restore_perm is not None:
            gathered = gathered.permute(context.restore_perm).contiguous()
        return torch.split(gathered, context.split_sizes, dim=-1)

    def _all_gather_dim_async(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.distributed.Work | None, tuple[int, ...] | None]:
        assert self.dcp_group is not None
        if dim == 0:
            gathered, handle = all_gather_async(x.contiguous(), self.dcp_group)
            return gathered, handle, None

        perm = (dim, *[i for i in range(x.dim()) if i != dim])
        restore_perm = tuple(perm.index(i) for i in range(x.dim()))
        gathered, handle = all_gather_async(x.permute(perm).contiguous(), self.dcp_group)
        return gathered, handle, restore_perm

    def _remap_sparse_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.dcp_size <= 1:
            return topk_indices

        topk_count = topk_indices.shape[-1]
        if topk_count > self._dcp_index_topk:
            raise RuntimeError(
                f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({self._dcp_index_topk})."
            )
        if topk_indices.numel() == 0:
            return topk_indices

        if HAS_TRITON and topk_indices.is_npu:
            from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton

            return remap_sparse_indices_triton(
                topk_indices,
                self.dcp_size,
                self.dcp_rank,
                self._dcp_interleave_size,
            )

        # Fallback for environments without Triton: remap the topk indices from
        # the replicated view to the DCP-local KV cache view. We use float32 for
        # better performance on Ascend.
        topk_indices_fp32 = topk_indices.to(torch.float32)
        interleave_size = self._dcp_interleave_size
        local_block_indices = torch.floor(topk_indices_fp32 / interleave_size)
        local_owner_base = torch.floor(local_block_indices / self.dcp_size) * self.dcp_size
        local_owner = local_block_indices - local_owner_base
        local_owner_mask = (topk_indices_fp32 >= 0) & (local_owner == self.dcp_rank)
        if interleave_size == 1:
            remapped_indices_fp32 = torch.floor(topk_indices_fp32 / self.dcp_size)
        else:
            local_offsets = topk_indices_fp32 - local_block_indices * interleave_size
            remapped_indices_fp32 = torch.floor(topk_indices_fp32 / (self.dcp_size * interleave_size))
            remapped_indices_fp32 = remapped_indices_fp32 * interleave_size + local_offsets
        remapped_indices = torch.where(
            local_owner_mask,
            remapped_indices_fp32,
            self._remap_invalid_index,
        ).to(topk_indices.dtype)

        # Compact local indices to the front without changing their top-k order.
        original_order = self._remap_order[:topk_count].expand_as(topk_indices)
        pack_keys = original_order + (~local_owner_mask).to(torch.float32) * topk_count
        _, pack_order = torch.sort(pack_keys, dim=-1)
        return torch.gather(remapped_indices, dim=-1, index=pack_order.to(torch.int32))

    def _merge_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
        dsa_cp_context: DSACPContext | None = None,
    ) -> torch.Tensor:
        scatter_dim = 1
        if dsa_cp_context is not None:
            # DSA-CP keeps heads replicated and shards tokens. The All2All
            # destination must match the token range assigned to this rank.
            num_tokens = sfa_output.shape[0]
            if num_tokens != dsa_cp_context.num_tokens_pad:
                raise RuntimeError(
                    "DSA-CP DCP All2All expects the SFA token count to match "
                    f"num_tokens_pad, got {num_tokens} and {dsa_cp_context.num_tokens_pad}."
                )
            if num_tokens % self.dcp_size != 0:
                raise RuntimeError(
                    f"DSA-CP DCP All2All requires {num_tokens} tokens to be divisible by dcp_size={self.dcp_size}."
                )
            local_num_tokens = num_tokens // self.dcp_size
            expected_local_start = self.dcp_rank * local_num_tokens
            actual_local_num_tokens = dsa_cp_context.local_end_with_pad - dsa_cp_context.local_start
            if dsa_cp_context.local_start != expected_local_start or actual_local_num_tokens != local_num_tokens:
                raise RuntimeError(
                    "DSA-CP token shards must follow DCP rank order for the output All2All, "
                    f"but rank {self.dcp_rank} expects [{expected_local_start}, "
                    f"{expected_local_start + local_num_tokens}) and metadata provides "
                    f"[{dsa_cp_context.local_start}, {dsa_cp_context.local_end_with_pad})."
                )
            scatter_dim = 0

        assert self.dcp_group is not None, "DCP output All2All requires dcp_group when dcp_size > 1."
        return torch.ops.vllm.sfa_dcp_a2a_fused(
            sfa_output,
            softmax_lse,
            self.dcp_size,
            scatter_dim,
            self.dcp_group.unique_name,
        )

    def _start_dcp_query_gather(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> DCPGatherContext:
        query_gather_dim = self._parallel_query_gather_dim()
        assert self.dcp_group is not None, "DCP query gather requires dcp_group when dcp_size > 1."
        if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
            raise RuntimeError(
                "Cannot fuse DCP query gather for ql_nope/q_pe with "
                f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
                f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
            )

        # Avoid back-to-back DCP all_gather calls for the two SFA query
        # fragments. On Ascend the separate gathers can leave SFA with an
        # incomplete stream dependency on the first prefill. DSA-CP restores
        # token shards on dim 0; native DCP restores query shards on dim 1.
        fused_q = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
        return self._start_dcp_gather(
            fused_q,
            dim=query_gather_dim,
            split_sizes=(ql_nope.shape[-1], q_pe.shape[-1]),
        )

    def _record_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
    ) -> None:
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        # Prefill/mixed batches gather compact KV after its cache write instead.
        # Keeping Q local avoids a full query all-gather and the subsequent LSE
        # output merge in the all-KV attention path.
        if self._has_prefill(attn_metadata):
            return
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        attn_metadata.dcp_context.gather_context = self._start_dcp_query_gather(ql_nope, q_pe)

    def _get_sfa_kv_slot_mapping(
        self,
        attn_metadata: M,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None
        return attn_metadata.dcp_context.slot_mapping

    def _store_parallel_kv(
        self,
        k_pe: torch.Tensor | None,
        k_nope: torch.Tensor | None,
        knope_scale: torch.Tensor | None,
        k_li: torch.Tensor | None,
        fused_kv_no_split: torch.Tensor | None,
        kv_ag_handles: list[torch.distributed.Work],
        kv_cache: tuple[torch.Tensor, ...] | None,
        slot_mapping_sfa: torch.Tensor,
        attn_metadata: M,
        full_gather_o_proj_enabled: bool,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        result = super()._store_parallel_kv(
            k_pe,
            k_nope,
            knope_scale,
            k_li,
            fused_kv_no_split,
            kv_ag_handles,
            kv_cache,
            slot_mapping_sfa,
            attn_metadata,
            full_gather_o_proj_enabled,
        )
        # Prefill DCP gathers referenced blocks after the current layer writes
        # its SFA KV cache and before indexer/top-k work begins.
        if kv_cache is not None:
            self._record_dcp_kv_gather_context(kv_cache, attn_metadata)
        return result

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."
        dcp_context = attn_metadata.dcp_context
        if self._has_prefill(attn_metadata):
            gather_context = dcp_context.gather_context
            dcp_context.gather_context = None
            if gather_context is None:
                # The normal forward path starts this after KV writes so it can
                # overlap indexer selection. Keep a synchronous fallback for
                # callers that invoke this method outside that path.
                self._record_dcp_kv_gather_context(kv_cache, attn_metadata)
                gather_context = dcp_context.gather_context
                dcp_context.gather_context = None
            assert gather_context is not None
            gathered_kv_cache = self._finish_dcp_gather(gather_context)
            block_table = dcp_context.kv_gather_block_table
            assert block_table is not None
            # The gathered KV cache is complete, so each rank can attend with
            # its local Q heads/tokens directly. In particular, DSA-CP keeps
            # its token shard local; no Q all-gather, sparse-index remap, LSE,
            # or output all-to-all merge is required.
            attn_output = DeviceOperator.execute_sparse_flash_attention_process(
                self,
                ql_nope,
                q_pe,
                gathered_kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table=block_table,
                sparse_mode=3,
                return_lse=False,
            )
            return attn_output

        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            gather_context = self._start_dcp_query_gather(ql_nope, q_pe)
        dsa_cp_context = getattr(attn_metadata, "dsa_cp_context", None)
        if dsa_cp_context is not None:
            # DSA-CP shards the token sequence. Restore the flat token order for
            # SFA, and use the original full query lengths for varlen metadata.
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            # topk_indices are in per-request global token coordinates. Gather
            # the DSA token shards first, then remap for this receiver rank's
            # DCP-local KV shard.
            topk_indices = self.dcp_group.all_gather(topk_indices.contiguous(), dim=0)
        topk_indices = self._remap_sparse_indices(topk_indices)
        ql_nope, q_pe = self._finish_dcp_gather(gather_context)
        sfa_output, softmax_max, softmax_sum = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            dcp_context.seq_lens,
            block_table=dcp_context.block_table,
            # The replicated-view indexer already applies the causal visibility rule.
            # After DCP remaps topk indices to local KV positions, local KV
            # length no longer shares the same coordinate system as global
            # query length, so SFA must not apply its right-down causal crop.
            sparse_mode=0,
            return_lse=True,
        )
        softmax_lse = softmax_max + torch.log(softmax_sum)
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        output_dtype = sfa_output.dtype
        output = self._merge_dcp_outputs(
            sfa_output,
            softmax_lse,
            getattr(attn_metadata, "dsa_cp_context", None),
        )
        return output.to(output_dtype)


class AscendSFADSADCPMetadataBuilder(
    AscendSFADCPMetadataBuilder,
    AscendSFADSACPMetadataBuilder,
):
    """Composes DCP's outer KV view with DSA-CP token sharding."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls or AscendSFADSADCPMetadata,
            supports_dcp_with_varlen,
        )


class AscendSFADSADCPImpl(AscendSFADCPImpl, AscendSFADSACPImpl):
    """Composes DCP collectives around the DSA-CP SFA implementation."""


def resolve_sfa_metadata_builder() -> type[AscendSFAMetadataBuilder]:
    """Resolve one SFA metadata builder from the two independent CP switches."""
    dsa_cp_enabled = enable_dsa_cp()
    dcp_enabled = enable_sfa_dcp_replicated_indexer()
    if dsa_cp_enabled and dcp_enabled:
        return AscendSFADSADCPMetadataBuilder
    if dsa_cp_enabled:
        return AscendSFADSACPMetadataBuilder
    if dcp_enabled:
        return AscendSFADCPMetadataBuilder
    return AscendSFAMetadataBuilder


def resolve_sfa_impl(vllm_config: VllmConfig | None = None) -> type[AscendSFAImpl]:
    """Resolve one SFA implementation from the two independent CP switches."""
    dsa_cp_enabled = enable_dsa_cp()
    dcp_enabled = enable_sfa_dcp_replicated_indexer()
    if dsa_cp_enabled and dcp_enabled:
        return AscendSFADSADCPImpl
    if dsa_cp_enabled:
        return AscendSFADSACPImpl
    if dcp_enabled:
        return AscendSFADCPImpl
    if vllm_config is not None and vllm_config.parallel_config.prefill_context_parallel_size > 1:
        return AscendSFAPCPImpl
    return AscendSFAImpl
