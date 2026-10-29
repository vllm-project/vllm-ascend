# SPDX-License-Identifier: Apache-2.0
"""Capture the original QSA device producer at the target model's graph entry.

The existing builder still constructs Python metadata outside the graph. Only
its device work is deferred. No independent graph, staging copy, kernel fusion,
or upstream source change is involved. MTP builders never enter this scope.

The context variable is worker-thread-local call routing, not a transient patch.
The weak registry is diagnostic only; each model state owns its capture plans.
Capture-time input/output tensors stay alive through those plans. Runtime input
addresses and Python-specialized geometry must match before replay is allowed.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any
from weakref import WeakSet

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.models.qwen4_exp.common import qsa_cache

_ORIGINAL_PRODUCER = qsa_cache._build_qsa_metadata_torch
_SCOPE: ContextVar[Any] = ContextVar("qsa_target_metadata_scope", default=None)
_STATES: WeakSet = WeakSet()
_OUTSIDE_FALLBACKS: Counter = Counter()
_OUTSIDE_LOCATIONS: Counter = Counter()
_MAX_TARGET_TOKENS = 32
_TARGET_QUERY_LEN = 4


def _tensor_signature(value):
    if value is None:
        return None
    return (str(value.device), str(value.dtype), int(value.data_ptr()), tuple(value.shape), tuple(value.stride()))


def _same_view(left, right):
    return _tensor_signature(left) == _tensor_signature(right)


@dataclass
class _Plan:
    source_common: Any
    common: Any
    buffers: tuple
    kwargs: dict
    signature: tuple
    outputs: tuple

    def execute(self, state, capturing):
        # CommonAttentionMetadata caches the token mapping after one call. A
        # fresh clone is essential: warmup must not remove that producer from
        # the subsequent real capture. Device inputs remain the same views.
        common = replace(self.common, _token_to_req_indices_cache=None)
        location = "target_native_capture" if capturing else "target_capture_warmup"
        state.locations[location] += 1
        actual = _invoke_original(common, self.buffers, self.kwargs, state=state, eligible=True, at_model_entry=True)
        if not all(_same_view(a, b) for a, b in zip(actual, self.outputs)):
            raise RuntimeError("Original QSA producer changed its persistent output ABI")


class TargetMetadataState:
    def __init__(self):
        self.pending = None
        self.captured = {}
        self.capture_depth = 0
        self.stats = Counter()
        self.fallbacks = Counter()
        self.locations = Counter()
        _STATES.add(self)


@dataclass
class _Scope:
    state: TargetMetadataState
    key: tuple
    for_capture: bool
    defer: bool
    plans: list


def _state(model_state):
    state = getattr(model_state, "_qsa_target_metadata", None)
    if state is None:
        state = TargetMetadataState()
        model_state._qsa_target_metadata = state
    return state


def target_metadata_config_supported(model_state):
    config = model_state.vllm_config
    parallel = config.parallel_config
    speculative = config.speculative_config
    return bool(
        config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        and parallel.tensor_parallel_size == 8
        and parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and speculative is not None
        and speculative.num_speculative_tokens == 3
        and not config.model_config.enforce_eager
    )


@contextmanager
def target_metadata_preparation(model_state, input_batch, cg_mode, for_capture):
    """Route only this target state's builder calls, preserving their types."""
    state = _state(model_state)
    # Upstream also sets for_capture=True on dummy FULL *replays*. Only this
    # manager's active capture context means a Python model-entry call follows.
    actual_capture = bool(for_capture and state.capture_depth)
    wants_full = actual_capture or cg_mode == CUDAGraphMode.FULL
    defer = wants_full and target_metadata_config_supported(model_state)
    key = (int(input_batch.num_tokens_after_padding), int(input_batch.num_reqs_after_padding))
    scope = _Scope(state, key, actual_capture, defer, [])
    state.pending = None
    token = _SCOPE.set(scope)
    try:
        yield
    except BaseException:
        state.stats["preparation_failures"] += 1
        raise
    else:
        if scope.plans:
            state.pending = scope
            state.stats["deferred_batches"] += 1
    finally:
        _SCOPE.reset(token)


def _make_plan(common, buffers, kwargs):
    n = int(common.num_actual_tokens)
    cpu_q = common.query_start_loc_cpu
    if str(cpu_q.device) != "cpu":
        raise RuntimeError("QSA CPU query boundaries must actually be on CPU")
    mapped = int(cpu_q[-1])
    r = int(common.query_start_loc.shape[0]) - 1
    if not (0 < n <= _MAX_TARGET_TOKENS and 0 < mapped <= n and r > 0):
        raise RuntimeError("Unsupported QSA target FULL token geometry")
    # The original producer specializes on the CPU *total*. Per-request device
    # boundaries may differ under adaptive verification and are never replaced
    # by this CPU layout. Padding may have zero CPU query lengths.
    query_lens = [int(cpu_q[i + 1]) - int(cpu_q[i]) for i in range(len(cpu_q) - 1)]
    if any(q not in (0, _TARGET_QUERY_LEN) for q in query_lens):
        raise RuntimeError("Only uniform K3 target decode is audited for metadata capture")
    if getattr(common, "_token_to_req_indices_cache", None) is not None:
        raise RuntimeError("Cached external token mapping cannot be deferred safely")
    normalized = replace(
        common,
        query_start_loc=common.query_start_loc[: r + 1],
        seq_lens=common.seq_lens[:r],
        block_table_tensor=common.block_table_tensor[:r],
        slot_mapping=common.slot_mapping[:n],
        _token_to_req_indices_cache=None,
    )
    outputs = (
        buffers[0][:n],
        buffers[1][:n],
        buffers[2][:n],
        normalized.slot_mapping
        if kwargs.get("circular_buffer_size", 0) == 0 and kwargs["compress_ratio"] == 1
        else buffers[3][:n],
    )
    inputs = (normalized.query_start_loc, normalized.seq_lens, normalized.block_table_tensor, normalized.slot_mapping)
    signature = (
        n,
        mapped,
        r,
        tuple(_tensor_signature(t) for t in inputs),
        tuple(_tensor_signature(t) for t in buffers),
        tuple(sorted((k, _tensor_signature(v) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items())),
    )
    return _Plan(common, normalized, buffers, kwargs, signature, outputs)


def _invoke_original(common, buffers, kwargs, *, state=None, eligible=False, at_model_entry=False):
    # Count a misplaced eligible materialization before failing. All calls to
    # the exact original producer go through this boundary, so the snapshot's
    # zero is backed by call routing and location counters rather than a literal.
    if eligible and not at_model_entry:
        if state is not None:
            state.stats["eligible_outside_model_calls"] += 1
        raise RuntimeError("Eligible QSA producer escaped the target model entry")
    return _ORIGINAL_PRODUCER(common, *buffers, **kwargs)


def build_qsa_metadata_original_target_graph(
    common_attn_metadata,
    token_to_req_buffer,
    logical_positions_buffer,
    visible_blocks_buffer,
    slot_mapping_buffer,
    *,
    storage_block_size,
    compress_ratio,
    circular_buffer_size=0,
    k_work_metadata_buffer=None,
    request_capacity=None,
):
    """Return the existing ABI; materialize only at target graph entry."""
    buffers = (token_to_req_buffer, logical_positions_buffer, visible_blocks_buffer, slot_mapping_buffer)
    kwargs = dict(
        storage_block_size=storage_block_size,
        compress_ratio=compress_ratio,
        circular_buffer_size=circular_buffer_size,
        k_work_metadata_buffer=k_work_metadata_buffer,
        request_capacity=request_capacity,
    )
    scope = _SCOPE.get()
    if scope is None or not scope.defer:
        reason = "outside_target_scope" if scope is None else "non_full_or_unsupported_config"
        (_OUTSIDE_FALLBACKS if scope is None else scope.state.fallbacks)[reason] += 1
        (_OUTSIDE_LOCATIONS if scope is None else scope.state.locations)[
            "unscoped_original_eager" if scope is None else "target_nonfull_original_eager"
        ] += 1
        return _invoke_original(common_attn_metadata, buffers, kwargs, state=None if scope is None else scope.state)
    if any(plan.source_common is common_attn_metadata for plan in scope.plans):
        # Upstream would share its cached token map between those builders.
        # This audited lane has one QSA owner per common cache-group metadata;
        # reject a new sharing pattern rather than silently change its aliases.
        raise RuntimeError("Multiple QSA builders sharing one Common metadata need an explicit alias plan")
    plan = _make_plan(common_attn_metadata, buffers, kwargs)
    scope.plans.append(plan)
    scope.state.stats["deferred_calls"] += 1
    return plan.outputs


@contextmanager
def target_metadata_capture(model_state):
    """Track the existing target capture lifecycle; creates no graph itself."""
    state = _state(model_state)
    state.captured.clear()
    state.capture_depth += 1
    try:
        yield
    finally:
        state.capture_depth -= 1
        state.pending = None


def execute_target_metadata_at_model_entry(model_state):
    """Called by target ModelWithContext.forward within its native capture."""
    state = _state(model_state)
    scope = state.pending
    if scope is None:
        return
    if not scope.for_capture or not state.capture_depth:
        raise RuntimeError("Deferred QSA metadata reached an uncaptured model forward")
    capturing = bool(torch.npu.is_current_stream_capturing())
    try:
        for plan in scope.plans:
            plan.execute(state, capturing)
            state.stats["materialized_calls"] += 1
            if capturing:
                state.stats["capture_calls"] += 1
        if capturing:
            signatures = tuple(plan.signature for plan in scope.plans)
            if scope.key in state.captured:
                raise RuntimeError("Duplicate QSA target graph metadata geometry")
            state.captured[scope.key] = (signatures, tuple(scope.plans))
            state.stats["capture_batches"] += 1
    except BaseException:
        state.stats["capture_failures"] += 1
        raise


def validate_target_metadata_replay(model_state, desc):
    """Fail before native submission if captured pointers/geometry changed."""
    state = _state(model_state)
    scope = state.pending
    try:
        if scope is None or scope.for_capture:
            raise RuntimeError("Target FULL replay has no current QSA producer plan")
        expected_reqs = desc.num_reqs or desc.num_tokens // _TARGET_QUERY_LEN
        if scope.key != (int(desc.num_tokens), int(expected_reqs)):
            raise RuntimeError("QSA producer plan does not describe this target graph")
        captured = state.captured.get(scope.key)
        if captured is None or captured[0] != tuple(p.signature for p in scope.plans):
            raise RuntimeError("QSA target graph producer inputs or static geometry changed")
    except BaseException:
        state.stats["replay_guard_failures"] += 1
        raise
    state.stats["validated_replay_batches"] += 1


def mark_target_metadata_replay_success(model_state):
    # The model manager calls this only after the existing native replay and
    # graph-parameter updates return successfully. Client drains prove device
    # completion; this counter by itself records submission, not elapsed time.
    _state(model_state).stats["successful_replay_batches"] += 1


def target_metadata_snapshot():
    stats = Counter()
    fallbacks = Counter(_OUTSIDE_FALLBACKS)
    locations = Counter(_OUTSIDE_LOCATIONS)
    plans = []
    for state in list(_STATES):
        stats.update(state.stats)
        fallbacks.update(state.fallbacks)
        locations.update(state.locations)
        for key, (signatures, owned_plans) in state.captured.items():
            plans.append(
                dict(
                    num_tokens=key[0],
                    num_reqs=key[1],
                    num_mapped_tokens=[p.signature[1] for p in owned_plans],
                    producer_count=len(signatures),
                    signatures=signatures,
                )
            )
    fields = (
        "capture_calls",
        "capture_batches",
        "deferred_calls",
        "deferred_batches",
        "materialized_calls",
        "validated_replay_batches",
        "capture_failures",
        "successful_replay_batches",
        "replay_guard_failures",
        "preparation_failures",
    )
    return dict(
        status="verified",
        placement="inside_target_full_graph",
        original_producer_is_torch=_ORIGINAL_PRODUCER is qsa_cache._build_qsa_metadata_torch,
        fused_metadata_enabled=False,
        standalone_graphs=0,
        draft_production_inside_graph=False,
        eligible_outside_model_calls=stats["eligible_outside_model_calls"],
        materializations_by_location=dict(locations),
        eager_fallbacks_by_reason=dict(fallbacks),
        captured_plans=plans,
        **{name: stats[name] for name in fields},
    )
