# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB metrics for Model Runner V2.

Load balance mirrors the balancedness quantities that upstream ``EplbState``
logs: upstream ``step`` reduces ``expert_load_pass`` globally, logs average /
max expert load and balancedness on EP rank 0, and clears the buffer before
returning. The handler snapshots the buffer before the call, reduces the
snapshot across the EP group itself, and records only on the steps where
upstream consumed the buffer, so the series follows the upstream logging
interval instead of every ``log_stats`` step.

Weight transfer tracks the HIXL communicator: one accumulator per layer that
``set_transfer_context`` resets, ``add_send`` / ``add_recv`` fill, and
``execute`` records and drops, so a failed execute cannot leave a stale
accumulator behind.
"""

import weakref
from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    MetricSpec,
    MetricType,
    metric_enabled,
    record_metric,
    register_metrics,
    safe_recording,
)

# ---------------------------------------------------------------------------
# Expert load balance (AscendEplbState.step)
# ---------------------------------------------------------------------------

_LB_KEY = "eplb_load_balance"
_LB_RECORDERS: dict[str, Any] = {}

_AVG_TOKENS_SPEC = MetricSpec(
    "eplb:load_balance:avg_tokens",
    MetricType.GAUGE,
    ("rank", "model"),
)
_MAX_TOKENS_SPEC = MetricSpec(
    "eplb:load_balance:max_tokens",
    MetricType.GAUGE,
    ("rank", "model"),
)
_BALANCEDNESS_SPEC = MetricSpec(
    "eplb:load_balance:balancedness",
    MetricType.GAUGE,
    ("rank", "model"),
)
_LB_SPECS = (_AVG_TOKENS_SPEC, _MAX_TOKENS_SPEC, _BALANCEDNESS_SPEC)


def _log_stats_requested(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Return whether the caller asked ``EplbState.step`` to log statistics.

    ``AscendEplbState.step`` takes ``(is_dummy, is_profile, log_stats)``; the
    handler only snapshots the buffer on those steps.
    """
    if "log_stats" in kwargs:
        return bool(kwargs["log_stats"])
    return len(args) > 2 and bool(args[2])


def _snapshot_loads(state: Any) -> dict[str, tuple[Any, Any]]:
    """Clone each model's ``expert_load_pass`` before ``step`` clears it."""
    snapshots: dict[str, tuple[Any, Any]] = {}
    for name, model_state in getattr(state, "model_states", {}).items():
        expert_load_pass = getattr(model_state, "expert_load_pass", None)
        if expert_load_pass is not None and expert_load_pass.numel() > 0:
            snapshots[name] = (model_state, expert_load_pass.clone())
    return snapshots


def eplb_load_balance_handler(original_func, state, *args, **kwargs):
    """Record the globally reduced expert load that upstream ``step`` logged.

    Upstream ``EplbState.step`` consumes ``expert_load_pass`` on a balancedness
    logging step: it reduces the buffer globally, logs average / max load and
    balancedness, and clears the buffer before returning. The handler
    therefore

    * snapshots the rank-local buffer before the original call,
    * after the call checks whether upstream consumed the buffer — a cleared
      buffer marks exactly the steps upstream logs on, while an untouched
      buffer means this step stayed outside the balancedness interval, and
    * reduces the snapshot across the EP group itself, so every EP rank runs
      the reduction on those steps and only EP rank 0 records.

    The reduction duplicates the one upstream ``step`` performs for its own
    logging: upstream does not expose the reduced buffer, and the extra cost
    is confined to balancedness logging steps. There is also no metrics-level
    fast path before the reduction: the level is expected to be identical on
    every EP rank, but skipping the reduction on a rank whose level is
    momentarily disabled while the others still enter it would deadlock the
    EP group, so the snapshot and reduction run unconditionally on logging
    steps and only recording is gated by the level.

    The one host copy happens only on logging steps, not every decode step.
    """
    snapshots = _snapshot_loads(state) if _log_stats_requested(args, kwargs) else {}
    result = original_func(state, *args, **kwargs)
    if not snapshots:
        return result

    with safe_recording("eplb_load_balance"):
        consumed: dict[str, tuple[Any, Any]] = {}
        for name, (model_state, snapshot) in snapshots.items():
            expert_load_pass = getattr(model_state, "expert_load_pass", None)
            if expert_load_pass is not None and not bool(expert_load_pass.any()):
                consumed[name] = (model_state, snapshot)
        if not consumed:
            return result

        # Imported lazily so the handler module stays importable without vLLM.
        import torch
        import torch.distributed
        from vllm.distributed.parallel_state import get_ep_group

        ep_group = get_ep_group().device_group

        # The reduction below must run on every EP rank, so every gate — the
        # rank-0 gate and the recorder-availability gate — sits after the
        # collective loop: returning early on a rank whose recorder is
        # unavailable would leave the other EP ranks blocked inside
        # all_reduce.
        stats: dict[str, tuple[Any, float, float, float]] = {}
        ep_size = ep_group.size()
        for name, (model_state, snapshot) in consumed.items():
            # Each rank only counts its own experts; the sum across the EP
            # group rebuilds the same globally reduced load upstream logged.
            global_load = snapshot
            torch.distributed.all_reduce(global_load, group=ep_group)
            # Same shape as upstream: (num_moe_layers, num_physical_experts).
            tokens_per_rank = global_load.reshape(global_load.shape[0], ep_size, -1).sum(dim=-1).float()
            # One host copy for both scalars; ``step`` only reaches here on
            # balancedness logging steps, not every decode step.
            avg_tokens, max_tokens = torch.stack(
                (
                    tokens_per_rank.mean(dim=1).sum(),
                    tokens_per_rank.max(dim=1).values.sum(),
                )
            ).tolist()
            balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0
            stats[name] = (model_state, avg_tokens, max_tokens, balancedness)

        # Recorder availability is checked only after the collective loop; a
        # missing recorder disables recording, never the reduction itself.
        metrics = register_metrics(_LB_SPECS, _LB_RECORDERS, _LB_KEY)
        if metrics is None:
            return result

        if ep_group.rank() != 0:
            # Upstream logs on EP rank 0 only; mirrors that behavior after
            # the reduction has run on every rank.
            return result

        for name, (model_state, avg_tokens, max_tokens, balancedness) in stats.items():
            labels = {
                "rank": str(ep_group.rank()),
                "model": str(getattr(model_state, "model_name", name)),
            }
            for spec, value in (
                (_AVG_TOKENS_SPEC, avg_tokens),
                (_MAX_TOKENS_SPEC, max_tokens),
                (_BALANCEDNESS_SPEC, balancedness),
            ):
                if metric_enabled(spec):
                    record_metric(metrics, spec, value, labels)
    return result


# ---------------------------------------------------------------------------
# Expert weight transfer (AscendHixlEplbCommunicator)
# ---------------------------------------------------------------------------

_TF_KEY = "eplb_transfer"
_TF_RECORDERS: dict[str, Any] = {}

_SEND_EXPERTS_SPEC = MetricSpec(
    "eplb:transfer:send_experts",
    MetricType.GAUGE,
    ("rank", "layer"),
)
_RECV_EXPERTS_SPEC = MetricSpec(
    "eplb:transfer:recv_experts",
    MetricType.GAUGE,
    ("rank", "layer"),
)
_COMM_OPS_SPEC = MetricSpec(
    "eplb:transfer:comm_ops",
    MetricType.GAUGE,
    ("rank", "layer"),
)
_EST_BYTES_SPEC = MetricSpec(
    "eplb:transfer:est_bytes",
    MetricType.GAUGE,
    ("rank", "layer"),
)
_TF_SPECS = (_SEND_EXPERTS_SPEC, _RECV_EXPERTS_SPEC, _COMM_OPS_SPEC, _EST_BYTES_SPEC)

# Per-communicator accumulation for the layer that is currently being moved.
# Weak keys keep finished communicators from pinning their bookkeeping.
_pending: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = weakref.WeakKeyDictionary()


def _slot(communicator: Any) -> dict[str, Any]:
    slot = _pending.get(communicator)
    if slot is None:
        slot = {"layer": None, "send": set(), "recv": set(), "ops": 0, "bytes": 0}
        _pending[communicator] = slot
    return slot


def _reset(communicator: Any, layer: int | None) -> dict[str, Any]:
    slot = {"layer": layer, "send": set(), "recv": set(), "ops": 0, "bytes": 0}
    _pending[communicator] = slot
    return slot


def eplb_transfer_context_handler(original_func, communicator, old_indices, layer_idx, *args, **kwargs):
    """Start a clean accumulator for the layer about to be transferred."""
    result = original_func(communicator, old_indices, layer_idx, *args, **kwargs)
    _reset(communicator, layer_idx)
    return result


def eplb_transfer_send_handler(original_func, communicator, tensors, dst_rank, expert_id, *args, **kwargs):
    """Count outgoing experts, issued ops and transferred bytes."""
    result = original_func(communicator, tensors, dst_rank, expert_id, *args, **kwargs)
    slot = _slot(communicator)
    slot["send"].add(expert_id)
    slot["ops"] += len(tensors)
    slot["bytes"] += sum(tensor.nbytes for tensor in tensors)
    return result


def eplb_transfer_recv_handler(original_func, communicator, tensors, src_rank, expert_id, *args, **kwargs):
    """Count incoming experts, issued ops and transferred bytes."""
    result = original_func(communicator, tensors, src_rank, expert_id, *args, **kwargs)
    slot = _slot(communicator)
    slot["recv"].add(expert_id)
    slot["ops"] += len(tensors)
    slot["bytes"] += sum(tensor.nbytes for tensor in tensors)
    return result


def eplb_transfer_execute_handler(original_func, communicator, *args, **kwargs):
    """Record the finished layer and reset the accumulator."""
    try:
        result = original_func(communicator, *args, **kwargs)
    finally:
        # A failed execute must not leave a stale accumulator behind; the
        # slot is dropped either way and only a successful execute records.
        slot = _pending.pop(communicator, None)
    if slot is None or slot["layer"] is None:
        return result

    with safe_recording("eplb_transfer"):
        metrics = register_metrics(_TF_SPECS, _TF_RECORDERS, _TF_KEY)
        if metrics is None:
            return result

        # Imported lazily so the handler module stays importable without vLLM.
        from vllm.distributed.parallel_state import get_ep_group

        labels = {
            "rank": str(get_ep_group().device_group.rank()),
            "layer": str(slot["layer"]),
        }
        for spec, value in (
            (_SEND_EXPERTS_SPEC, len(slot["send"])),
            (_RECV_EXPERTS_SPEC, len(slot["recv"])),
            (_COMM_OPS_SPEC, slot["ops"]),
            (_EST_BYTES_SPEC, slot["bytes"]),
        ):
            if metric_enabled(spec):
                record_metric(metrics, spec, value, labels)
    return result
