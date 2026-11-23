# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend-local deferred RecoverSSM orchestration.

The Triton kernels live in ``vllm_ascend.ops.triton.kda.recoverssm``.  This
module vendors the Python orchestration that upstream keeps in
``vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm`` (PR #66030), so
vllm-ascend does not depend on a vllm version that ships it.  Only the pieces
Ascend needs are kept: the per-request ``pending`` buffer, record-slot staging,
the ``eager`` mask, and the metadata wiring.  Ascend's KDA layer forward does
its own verify and record pass, so the upstream ``install_layers`` /
``deferred_recoverssm_verify`` / ``allocate_records`` are not ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ops.triton.kda.recoverssm import (
    KDARecoverSSMCommitContext as AscendCommitContext,
)

FP32_DEFERRED = "fp32_deferred"


@dataclass
class KDARecoverSSMDeferred:
    """A step's deferred RecoverSSM (the KDA metadata's ``recoverssm_deferred``).

    After sampling, each request's accepted count becomes the pending replay of
    its record slot instead of a commit; the next verify replays it.
    """

    record_rows: torch.Tensor  # [>= spec rows] int32: record slot per row, -1: none
    pending: torch.Tensor  # [max_num_reqs] int32, per record slot
    eager: torch.Tensor | None  # [spec rows] mask, None: no row
    context: "KDADeferredCommitContext" = field(repr=False, compare=False)
    state_indices: torch.Tensor = field(repr=False, compare=False)
    layer_names: list[str] = field(default_factory=list, repr=False, compare=False)
    started: bool = False
    flushed: bool = False

    def flush(self) -> None:
        """Fold the step's pending replays into the checkpoints (every KDA
        layer, once per step) before a verify that does not replay them."""
        if self.flushed:
            return
        self.flushed = True
        rows = self.state_indices.shape[0]
        self.context.flush(self.state_indices, self.record_rows[:rows], self.pending)

    def _layer_index(self, prefix: str) -> int | None:
        try:
            return self.layer_names.index(prefix)
        except ValueError:
            return None

    def start_flush(self, prefix: str) -> None:
        """Prepare the replay plan and compact the conv history once, at the
        first KDA layer.

        The recurrent-state replay is fused into each layer's verify (it reuses
        the state load the verify already performs), so only the conv history
        has to be shifted before ``causal_conv1d``.
        """
        if self.started or self.flushed:
            return
        index = self._layer_index(prefix)
        if index is None:
            self.flush()
            return
        rows = self.state_indices.shape[0]
        if rows == 0:
            return
        self.started = True
        self.context.compact_conv(self.state_indices, self.record_rows[:rows], self.pending)

    def wait_layer(self, prefix: str) -> None:
        """No-op: the fused verify replays the pending state in-kernel."""
        if not self.started and not self.flushed:
            self.start_flush(prefix)

    def replay_plan(self):
        """The prepared plan for ``kda_recoverssm_verify``'s fused replay."""
        if not self.started:
            return None
        return (
            self.context.commit_lens,
            self.context.final_state_indices,
            self.context.boundary_state_indices,
            self.context.boundary_recovery_lens,
        )


class KDADeferredCommitContext(AscendCommitContext):
    """Ascend ``KDARecoverSSMCommitContext`` tagged as the deferred backend.

    ``create`` skips the upstream CuTe probe and always builds the Ascend
    Triton context; ``commit`` / ``flush`` / ``_fold`` are inherited.
    """

    @classmethod
    def create(
        cls,
        layers,
        *,
        spec_query_len: int,
        max_num_reqs: int,
        backend: str = FP32_DEFERRED,
    ) -> "KDADeferredCommitContext":
        context = AscendCommitContext.create(
            layers,
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            backend="triton",
        )
        context.__class__ = cls
        context.backend = FP32_DEFERRED  # type: ignore[attr-defined]
        return context


class DeferredRecoverSSMState(RecoverSSMState):
    """``pending`` [max_num_reqs]: per request-state slot, the accepted tokens of
    its last verify not yet folded into its checkpoint."""

    def __init__(self, max_num_reqs: int, device: torch.device) -> None:
        super().__init__()
        self.pending = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def add_request(self, req_index: int) -> None:
        # A new or resumed request starts from a committed (prefilled) state.
        self.pending[req_index].fill_(0)

    def bind(
        self,
        attn_groups: list[list[AttentionGroup]],
        idx_mapping: torch.Tensor | None,
    ) -> None:
        """Hand deferring builders the step's request slots (None: capture or
        dummy batches) and, once, the pending counts."""
        for group_list in attn_groups:
            for group in group_list:
                builder = group.get_metadata_builder(0)
                if not isinstance(builder, KimiK3KDAMetadataBuilder):
                    continue
                if not builder.use_recoverssm:
                    continue
                if getattr(builder, "recoverssm_pending", None) is not self.pending:
                    install_builder(builder, self.pending)
                builder.recoverssm_request_slots = idx_mapping


def install_model_state(state: Any) -> None:
    """Swap a model state's ``recoverssm`` for the deferred one and wrap
    ``add_request`` / ``prepare_attn`` to reset pending and bind builders."""
    if state.recoverssm is None:
        return
    deferred = DeferredRecoverSSMState(state.max_num_reqs, state.device)
    state.recoverssm = deferred
    base_add_request = state.add_request
    base_prepare_attn = state.prepare_attn

    def add_request(req_index, new_req_data) -> None:
        base_add_request(req_index, new_req_data)
        deferred.add_request(req_index)

    def prepare_attn(
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ) -> dict[str, Any]:
        deferred.bind(attn_groups, None if for_capture else input_batch.idx_mapping)
        return base_prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )

    state.add_request = add_request
    state.prepare_attn = prepare_attn


class _DeferredStepCommit:
    """A step's commit context: the deferred context's commit with the step's
    deferred operands."""

    def __init__(
        self, context: KDADeferredCommitContext, deferred: KDARecoverSSMDeferred
    ) -> None:
        self.context, self.deferred = context, deferred

    def commit(self, *args, **kwargs) -> None:
        deferred = self.deferred
        self.context.commit(
            *args,
            **kwargs,
            record_rows=deferred.record_rows,
            pending=deferred.pending,
            eager=deferred.eager,
        )


def install_builder(builder: KimiK3KDAMetadataBuilder, pending: torch.Tensor) -> None:
    """Set up a KDA metadata builder for the deferred RecoverSSM."""
    builder.recoverssm_pending = pending
    builder.recoverssm_request_slots = None
    if getattr(builder, "recoverssm_record_rows", None) is not None:
        return  # rebound to new pending counts
    stale = builder.recoverssm_context
    if (
        stale is not None
        and not isinstance(stale, KDADeferredCommitContext)
        and getattr(stale, "backend", "triton") == "triton"
    ):
        builder.recoverssm_context = None
    max_num_reqs = builder.vllm_config.scheduler_config.max_num_seqs
    # Each speculative row's record slot (-1 past the real rows).
    builder.recoverssm_record_rows = torch.full(
        (max_num_reqs,), -1, dtype=torch.int32, device=builder.device
    )
    base_get_context = builder._get_recoverssm_context
    base_build = builder.build

    def get_context():
        if builder.recoverssm_context is not None:
            return builder.recoverssm_context
        compilation_config = builder.vllm_config.compilation_config
        forward_context = compilation_config.static_forward_context
        layers = [forward_context[name] for name in builder.layer_names]
        # Only fp32_deferred layers defer; "triton" is the immediate path.
        if layers[0].recoverssm_backend != FP32_DEFERRED:
            return base_get_context()
        context = KDADeferredCommitContext.create(
            layers,
            spec_query_len=1 + builder.vllm_config.num_speculative_tokens,
            max_num_reqs=max_num_reqs,
        )
        builder.recoverssm_context = context
        return context

    def build(
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
        **kwargs,
    ):
        metadata = base_build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            **kwargs,
        )
        deferred = None
        commit = metadata.recoverssm_commit
        if commit is not None and builder.recoverssm_pending is not None:
            context = metadata.recoverssm_context
            assert context is not None
            if getattr(context, "backend", "triton") == FP32_DEFERRED:
                assert isinstance(context, KDADeferredCommitContext)
                deferred = _build_deferred(
                    builder,
                    common_attn_metadata,
                    context,
                    metadata.num_spec_decodes,
                    commit.request_indices,
                    num_decode_draft_tokens_cpu,
                    commit.state_indices[:, 0],
                )
                metadata.recoverssm_context = _DeferredStepCommit(context, deferred)
        metadata.recoverssm_deferred = deferred  # type: ignore[attr-defined]
        return metadata

    builder._get_recoverssm_context = get_context  # type: ignore[method-assign]
    builder.build = build  # type: ignore[method-assign]


def _spec_sequence_masks_cpu(
    builder: KimiK3KDAMetadataBuilder, m, num_decode_draft_tokens_cpu: torch.Tensor
) -> torch.Tensor:
    """The speculative rows of a RecoverSSM build."""
    query_lens_cpu = m.query_start_loc_cpu.diff()
    masks = num_decode_draft_tokens_cpu >= 0
    masks |= (~m.is_prefilling) & (query_lens_cpu > 0)
    masks &= query_lens_cpu <= builder.num_spec + 1
    return masks


def _build_deferred(
    builder: KimiK3KDAMetadataBuilder,
    m,
    context: KDADeferredCommitContext,
    num_spec_decodes: int,
    spec_request_indices: torch.Tensor | None,
    num_decode_draft_tokens_cpu: torch.Tensor,
    state_indices: torch.Tensor,
) -> KDARecoverSSMDeferred:
    """Stage the speculative rows' record slots and, with align mode, mark the
    rows that must commit after sampling (their window may reach a block
    boundary)."""
    pending = builder.recoverssm_pending
    assert pending is not None
    rows = builder.recoverssm_record_rows
    slots = builder.recoverssm_request_slots
    live = 0
    if slots is not None:
        if spec_request_indices is not None:
            slots = slots[spec_request_indices]
        live = min(num_spec_decodes, slots.shape[0])
        rows[:live].copy_(slots[:live], non_blocking=True)
    rows[live : max(m.num_reqs, num_spec_decodes)].fill_(-1)
    eager = None
    if builder.vllm_config.cache_config.mamba_cache_mode == "align":
        assert isinstance(builder.kv_cache_spec, MambaSpec)
        block = builder.kv_cache_spec.block_size
        margin = 2 * (builder.num_spec + 1)
        upper = m.seq_lens_cpu_upper_bound
        query_lens = m.query_start_loc_cpu.diff()
        masks = _spec_sequence_masks_cpu(builder, m, num_decode_draft_tokens_cpu)
        spec_rows = masks.nonzero(as_tuple=True)[0]
        if upper is None:
            near = torch.ones(num_spec_decodes, dtype=torch.bool)
        else:
            end = upper[spec_rows].to(torch.int64)
            low = end - query_lens[spec_rows].to(torch.int64) - margin
            high = end + margin
            near = (low < 0) | (low.clamp(min=0) // block != high // block)
        if bool(near.any()):
            eager = async_tensor_h2d(near.to(torch.int32), device=state_indices.device)
    return KDARecoverSSMDeferred(
        record_rows=rows,
        pending=pending,
        eager=eager,
        context=context,
        state_indices=state_indices[:num_spec_decodes],
        layer_names=list(builder.layer_names),
    )
