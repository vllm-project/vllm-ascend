# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB metrics for Model Runner V2.

Load balance hooks ``AscendEplbState._sync_load_pass``, the pass-through
override that upstream ``EplbState.step`` calls only on balancedness logging
steps: ``log_stats`` is set and ``expert_rearrangement_step`` is a multiple of
``log_balancedness_interval``. Upstream returns the load pass already reduced
across the EP group, so the handler reuses that value instead of cloning the
buffer and reducing again: on EP rank 0 it computes the per-rank token load and
records what upstream logs on the same call. All-idle or dummy logging steps
reduce to zero and are dropped.

The series needs EPLB to be enabled and ``--eplb-config.log_balancedness true``,
which upstream defaults to false because balancedness logging adds
communication; ``log_balancedness_interval`` sets how often a sample is logged.
"""

import logging
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

logger = logging.getLogger(__name__)

_eplb_metrics_recorders: dict[str, Any] = {}

_HOTNESS_SUMMARY_METRICS = (
    "eplb:expert_hotness:current_mean",
    "eplb:expert_hotness:current_max",
    "eplb:expert_hotness:update_mean",
    "eplb:expert_hotness:update_max",
)
_IMBALANCE_METRIC = "eplb:expert_hotness:imbalance"

_SUMMARY_SPECS = tuple(MetricSpec(name, MetricType.GAUGE, ("rank", "phase")) for name in _HOTNESS_SUMMARY_METRICS)
_SUMMARY_HOTNESS_KEYS = ("current_mean", "current_max", "update_mean", "update_max")
_IMBALANCE_SPEC = MetricSpec(_IMBALANCE_METRIC, MetricType.GAUGE, ("rank", "phase", "layer"))
_EPLB_METRIC_SPECS = (*_SUMMARY_SPECS, _IMBALANCE_SPEC)

_eplb_hotness_key = "eplb_hotness"


def _register_eplb_metrics():
    return register_metrics(
        _EPLB_METRIC_SPECS,
        _eplb_metrics_recorders,
        _eplb_hotness_key,
    )


def eplb_do_update_hotness_handler(original_func, worker, *args, **kwargs):
    """Record the aggregated expert hotness exposed by the EPLB rank-0 worker."""
    result = original_func(worker, *args, **kwargs)

    with safe_recording("expert_hotness"):
        rank_id = getattr(worker, "rank_id", -1)
        if rank_id != 0:
            return result

        hotness = getattr(worker, "latest_expert_hotness", None)
        if not hotness:
            logger.debug("Skip EPLB hotness metrics because latest_expert_hotness is missing")
            return result

        metrics = _register_eplb_metrics()
        if metrics is None:
            logger.debug("Skip EPLB hotness metrics because the metric recorder is unavailable")
            return result
        rank_labels = {"rank": str(rank_id), "phase": "all"}
        for spec, hotness_key in zip(_SUMMARY_SPECS, _SUMMARY_HOTNESS_KEYS):
            if not metric_enabled(spec):
                continue
            value = hotness.get(hotness_key)
            if value is not None:
                record_metric(metrics, spec, value, rank_labels)

        if metric_enabled(_IMBALANCE_SPEC):
            for phase, key in (
                ("current", "current_imbalance_list"),
                ("update", "update_imbalance_list"),
            ):
                imbalance_values: Any = hotness.get(key)
                if imbalance_values is None:
                    continue
                for layer_index, value in enumerate(imbalance_values):
                    record_metric(
                        metrics,
                        _IMBALANCE_SPEC,
                        value,
                        {
                            "rank": str(rank_id),
                            "phase": phase,
                            "layer": str(layer_index),
                        },
                    )
    return result


# ---------------------------------------------------------------------------
# Expert load balance (AscendEplbState._sync_load_pass)
# ---------------------------------------------------------------------------

_LB_KEY = "eplb_expert_load"
_LB_RECORDERS: dict[str, Any] = {}

_AVG_TOKENS_SPEC = MetricSpec(
    "eplb:expert_load:avg_tokens",
    MetricType.GAUGE,
    ("rank", "model"),
)
_MAX_TOKENS_SPEC = MetricSpec(
    "eplb:expert_load:max_tokens",
    MetricType.GAUGE,
    ("rank", "model"),
)
_BALANCEDNESS_SPEC = MetricSpec(
    "eplb:expert_load:balancedness",
    MetricType.GAUGE,
    ("rank", "model"),
)
_LB_SPECS = (_AVG_TOKENS_SPEC, _MAX_TOKENS_SPEC, _BALANCEDNESS_SPEC)


def _per_rank_tokens(load: Any, ep_size: int) -> Any:
    """Sum expert load per EP rank, matching the shape upstream logs."""
    return load.reshape(load.shape[0], ep_size, -1).sum(dim=-1).float()


def eplb_sync_load_handler(original_func, state, *args, **kwargs):
    """Record the balancedness upstream derives from the same reduced loads.

    Upstream ``EplbState.step`` calls ``_sync_load_pass`` only when
    ``log_stats`` is set and ``expert_rearrangement_step`` is a multiple of
    ``log_balancedness_interval``, and uses the returned list to log the
    average and maximum per-rank token load plus the balancedness between them
    on EP rank 0. Hooking this call therefore needs no step predicate and no
    buffer snapshot: profile and non-logging steps never reach it, and the
    returned tensors are already reduced across the EP group, so the handler
    runs no collective of its own.

    Recording is gated to EP rank 0, matching upstream's logging rank. The gate
    may run before the host copy because nothing here is collective. Logging
    steps with no traffic anywhere still reduce to zero and are dropped.
    """
    loads = original_func(state, *args, **kwargs)

    with safe_recording("eplb_expert_load"):
        # Imported lazily so the handler module stays importable without vLLM.
        import torch
        from vllm.distributed.parallel_state import get_ep_group

        ep_group = get_ep_group().device_group
        metrics = register_metrics(_LB_SPECS, _LB_RECORDERS, _LB_KEY)
        if metrics is None or ep_group.rank() != 0:
            return loads

        ep_size = ep_group.size()
        model_states = list(getattr(state, "model_states", {}).items())
        for (name, model_state), load in zip(model_states, loads):
            if getattr(load, "numel", lambda: 0)() == 0:
                continue
            tokens_per_rank = _per_rank_tokens(load, ep_size)
            # One host copy for both scalars; ``_sync_load_pass`` only runs on
            # balancedness logging steps, not every decode step.
            avg_tokens, max_tokens = torch.stack(
                (
                    tokens_per_rank.mean(dim=1).sum(),
                    tokens_per_rank.max(dim=1).values.sum(),
                )
            ).tolist()
            if max_tokens <= 0:
                # No rank ran any tokens (all-idle or dummy step): upstream
                # logs zeros there, but there is no load to report.
                continue
            labels = {
                "rank": str(ep_group.rank()),
                "model": str(getattr(model_state, "model_name", name)),
            }
            for spec, value in (
                (_AVG_TOKENS_SPEC, avg_tokens),
                (_MAX_TOKENS_SPEC, max_tokens),
                (_BALANCEDNESS_SPEC, avg_tokens / max_tokens),
            ):
                if metric_enabled(spec):
                    record_metric(metrics, spec, value, labels)
    return loads


# ---------------------------------------------------------------------------
# Expert weight transfer (AscendHixlEplbCommunicator)
# ---------------------------------------------------------------------------

# HIXL is receiver-initiated (``receiver_initiated`` is true), so the transfer
# driver only walks the local rank's destination slots and
# ``AscendHixlEplbCommunicator.add_send`` is never called: the send handler
# does not fire and ``eplb:transfer:send_experts`` stays 0 for every HIXL
# transfer. Only ``recv_experts``, ``comm_requests`` and ``est_bytes`` carry
# values on this path. The send series stays declared for push-based
# communicators that do call ``add_send``.
#
# HIXL batches every descriptor for a source rank into one ``transfer_async``
# call, so a request is counted once per distinct peer rank per layer, not
# once per expert tensor.
#
# ``phase="all"`` is a constant label, matching the per-rank aggregate the V1
# hotness summary used, so every EPLB panel shares one phase filter; transfer
# has no current/update split to report.

_TF_KEY = "eplb_transfer"
_TF_RECORDERS: dict[str, Any] = {}

_SEND_EXPERTS_SPEC = MetricSpec(
    "eplb:transfer:send_experts",
    MetricType.GAUGE,
    ("rank", "phase", "layer"),
)
_RECV_EXPERTS_SPEC = MetricSpec(
    "eplb:transfer:recv_experts",
    MetricType.GAUGE,
    ("rank", "phase", "layer"),
)
_COMM_REQUESTS_SPEC = MetricSpec(
    "eplb:transfer:comm_requests",
    MetricType.GAUGE,
    ("rank", "phase", "layer"),
)
_EST_BYTES_SPEC = MetricSpec(
    "eplb:transfer:est_bytes",
    MetricType.GAUGE,
    ("rank", "phase", "layer"),
)
_TF_SPECS = (_SEND_EXPERTS_SPEC, _RECV_EXPERTS_SPEC, _COMM_REQUESTS_SPEC, _EST_BYTES_SPEC)

# Per-communicator accumulation for the layer that is currently being moved.
# Weak keys keep finished communicators from pinning their bookkeeping.
_pending: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = weakref.WeakKeyDictionary()


def _slot(communicator: Any) -> dict[str, Any]:
    slot = _pending.get(communicator)
    if slot is None:
        slot = {"layer": None, "send": set(), "recv": set(), "peers": set(), "bytes": 0}
        _pending[communicator] = slot
    return slot


def _reset(communicator: Any, layer: int | None) -> dict[str, Any]:
    slot: dict[str, Any] = {"layer": layer, "send": set(), "recv": set(), "peers": set(), "bytes": 0}
    _pending[communicator] = slot
    return slot


def eplb_transfer_context_handler(original_func, communicator, old_indices, layer_idx, *args, **kwargs):
    """Start a clean accumulator for the layer about to be transferred."""
    result = original_func(communicator, old_indices, layer_idx, *args, **kwargs)
    _reset(communicator, layer_idx)
    return result


def eplb_transfer_send_handler(original_func, communicator, tensors, dst_rank, expert_id, *args, **kwargs):
    """Count outgoing experts, peer ranks and transferred bytes.

    Under receiver-initiated HIXL this handler never runs (the sender side only
    serves READs), so ``eplb:transfer:send_experts`` stays 0 there; see the
    section comment above.
    """
    result = original_func(communicator, tensors, dst_rank, expert_id, *args, **kwargs)
    slot = _slot(communicator)
    slot["send"].add(expert_id)
    slot["peers"].add(dst_rank)
    slot["bytes"] += sum(tensor.nbytes for tensor in tensors)
    return result


def eplb_transfer_recv_handler(original_func, communicator, tensors, src_rank, expert_id, *args, **kwargs):
    """Count incoming experts, peer ranks and transferred bytes.

    This is the only side that fires under receiver-initiated HIXL, so
    ``recv_experts`` and the peers / bytes accumulated here describe the actual
    READs while the send series stays 0.
    """
    result = original_func(communicator, tensors, src_rank, expert_id, *args, **kwargs)
    slot = _slot(communicator)
    slot["recv"].add(expert_id)
    slot["peers"].add(src_rank)
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
            "phase": "all",
            "layer": str(slot["layer"]),
        }
        for spec, value in (
            (_SEND_EXPERTS_SPEC, len(slot["send"])),
            (_RECV_EXPERTS_SPEC, len(slot["recv"])),
            (_COMM_REQUESTS_SPEC, len(slot["peers"])),
            (_EST_BYTES_SPEC, slot["bytes"]),
        ):
            if metric_enabled(spec):
                record_metric(metrics, spec, value, labels)
    return result
