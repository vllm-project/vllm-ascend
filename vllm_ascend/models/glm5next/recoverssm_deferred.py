# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-side orchestration for the deferred KDA RecoverSSM commit path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ops.triton.kda.recoverssm import (
    KDARecoverSSMCommitContext as AscendCommitContext,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata

FP32_DEFERRED = "fp32_deferred"

# Deferral saves the fold launch but pays a per-token premium in the fused
# replay; past this mean acceptance rate the premium outgrows the saving.
# Bracketed by the measured points (~0.43 deferred wins, full acceptance it
# loses); recalibrate against the bench's uniform-rate sweep.
RECOVERSSM_DEFER_RATE = 0.75
# Per-step EMA decay of the rate estimate; acceptance is autocorrelated, and
# every gate flip is a free transition (eager rows zero their pending replay).
RECOVERSSM_RATE_DECAY = 0.75

# The fused replay plan handed to ``kda_recoverssm_verify``:
# (commit_lens, final_state_indices, boundary_state_indices, boundary_recovery_lens).
ReplayPlan = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class RecoverSSMAcceptanceGate:
    """Tracks the mean acceptance rate of the speculative rows.

    The rate is an EMA kept on the device (updated from the commit's
    acceptance tensor with the commit plan's row indexing) and mirrored to
    host memory with an asynchronous copy: the build-side read lags by the
    enqueue depth, which a slowly moving rate tolerates, and never
    synchronizes the pipeline.
    """

    def __init__(self, max_rows: int, device: torch.device) -> None:
        self.rate = torch.zeros((), dtype=torch.float32, device=device)
        self.rate_host = torch.zeros((), dtype=torch.float32, pin_memory=is_pin_memory_available())
        self.eager_ones = torch.ones(max_rows, dtype=torch.int32, device=device)

    def update(
        self,
        num_accepted: torch.Tensor,
        request_indices: torch.Tensor | None,
        spec_query_len: int,
        batch: int,
    ) -> None:
        """Fold this step's acceptance into the rate estimate."""
        if batch <= 0:
            return
        rows = num_accepted[:batch]
        if request_indices is not None:
            rows = num_accepted[request_indices[:batch].long()]
        rate = rows.float().mean() / spec_query_len
        self.rate.mul_(RECOVERSSM_RATE_DECAY).add_(rate, alpha=1.0 - RECOVERSSM_RATE_DECAY)
        self.rate_host.copy_(self.rate, non_blocking=True)

    def engaged(self) -> bool:
        """Whether the next commit should fold every row now."""
        return float(self.rate_host) >= RECOVERSSM_DEFER_RATE

    def eager_mask(self, rows: int) -> torch.Tensor:
        """The all-eager mask the engaged gate hands the commit."""
        return self.eager_ones[:rows]


@dataclass
class KDARecoverSSMDeferred:
    """A step's deferred RecoverSSM (the KDA metadata's ``recoverssm_deferred``).

    After sampling, each request's accepted count becomes the pending replay of
    its record slot instead of a commit; the next verify replays it.

    A step whose first KDA layer is unknown (or an explicit flush) folds the
    pending replays into the checkpoints first (``flush``, upstream's
    "flush + core" path) and the verify runs without a replay.
    """

    record_rows: torch.Tensor  # [>= spec rows] int32: record slot per row, -1: none
    pending: torch.Tensor  # [max_num_reqs] int32, per record slot
    eager: torch.Tensor | None  # [spec rows] mask, None: no row
    context: KDADeferredCommitContext = field(repr=False, compare=False)
    state_indices: torch.Tensor = field(repr=False, compare=False)
    layer_names: list[str] = field(default_factory=list, repr=False, compare=False)
    gate: RecoverSSMAcceptanceGate | None = field(default=None, repr=False, compare=False)
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
        self.context.compact_conv(
            self.state_indices, self.record_rows[:rows], self.pending
        )

    def replay_plan(self) -> ReplayPlan | None:
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
    """The deferred RecoverSSM commit context.

    ``create`` / ``commit`` / ``flush`` / ``_fold`` are inherited: the base
    factory constructs this subclass directly, and the subclass identity
    itself marks the deferred path (the stock context stays the immediate
    one).
    """


class DeferredRecoverSSMState(RecoverSSMState):
    """``pending`` [max_num_reqs]: per request-state slot, the accepted tokens of
    its last verify not yet folded into its checkpoint."""

    def __init__(self, max_num_reqs: int, device: torch.device) -> None:
        super().__init__()
        self.pending = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.gate = RecoverSSMAcceptanceGate(max_num_reqs, device)

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
                builder.recoverssm_gate = self.gate
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
        gate = deferred.gate
        if gate is not None and len(args) >= 2:
            gate.update(
                args[0],
                kwargs.get("request_indices"),
                self.context.spec_query_len,
                args[1].shape[0],
            )
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
        return  # already installed; only the pending tensor needed rebinding
    stale = builder.recoverssm_context
    # A stock Triton context built before the first bind would pin the stock
    # commit; rebuild it as the deferred one (no runner path does, today).
    if stale is not None and not isinstance(stale, KDADeferredCommitContext):
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
            if isinstance(context, KDADeferredCommitContext):
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
    builder: KimiK3KDAMetadataBuilder,
    m: CommonAttentionMetadata,
    num_decode_draft_tokens_cpu: torch.Tensor,
) -> torch.Tensor:
    """The speculative rows of a RecoverSSM build."""
    query_lens_cpu = m.query_start_loc_cpu.diff()
    masks = num_decode_draft_tokens_cpu >= 0
    masks |= (~m.is_prefilling) & (query_lens_cpu > 0)
    masks &= query_lens_cpu <= builder.num_spec + 1
    return masks


def _build_deferred(
    builder: KimiK3KDAMetadataBuilder,
    m: CommonAttentionMetadata,
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
    gate = getattr(builder, "recoverssm_gate", None)
    if gate is not None and gate.engaged() and num_spec_decodes > 0:
        # The gate outranks the boundary mask: every row folds now, and each
        # row's pending replay is zeroed so the next verify runs plain.
        eager = gate.eager_mask(num_spec_decodes)
    return KDARecoverSSMDeferred(
        record_rows=rows,
        pending=pending,
        eager=eager,
        context=context,
        state_indices=state_indices[:num_spec_decodes],
        layer_names=list(builder.layer_names),
        gate=gate,
    )
