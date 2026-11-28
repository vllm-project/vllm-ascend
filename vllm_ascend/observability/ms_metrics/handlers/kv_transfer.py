# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV transfer metrics for Prefill/Decode (PD) disaggregation.

Prefill and Decode run on separate instances and the Decode side pulls the
computed KV cache through a KV connector. This module only hooks existing
connector/scheduler methods and leaves the business code in
``vllm_ascend/distributed/kv_transfer`` untouched:

- receive outcomes per request on the Decode scheduler
  (``xfer_finished_total`` and ``xfer_failed_total``);
- receive outcomes from worker-side polls where no scheduler hook exists,
  counted once per TP rank (``xfer_finished_total``);
- blocks that failed to load (``invalid_blocks_total``);
- requests still waiting for remote KV per step
  (``remote_kv_waiting_requests``).

Coverage boundary: the per-request counters require a scheduler method that
sees worker outputs already aggregated across TP ranks, which the Mooncake V2
pull and push schedulers provide. Connectors hooking a worker-side poll
(``MooncakeConnector`` and ``AscendStoreConnector``) still count once per rank,
so with TP > 1 their ``xfer_finished_total`` is inflated by the TP size.

Coverage boundary: the waiting gauge unions the queue of requests about to be
dispatched with the connector-side registrations of reads still in flight.
Only schedulers that keep such a registry (the Mooncake pull path stores it in
``_reqs_recv_info``) are counted for the whole transfer; a path that dispatches
the READs without recording them can still sample zero while the transfer is
running.
"""

from typing import Any

import regex as re

from vllm_ascend.observability.ms_metrics.handlers.common import (
    MetricSpec,
    MetricType,
    metric_enabled,
    record_observation,
    safe_recording,
)

_KEY = "kv_transfer"
_CONNECTOR_LABEL = "connector"

_FINISHED_SPEC = MetricSpec(
    "kv:xfer_finished_total",
    MetricType.COUNTER,
    (_CONNECTOR_LABEL,),
)
_FAILED_SPEC = MetricSpec(
    "kv:xfer_failed_total",
    MetricType.COUNTER,
    (_CONNECTOR_LABEL,),
)
_INVALID_BLOCKS_SPEC = MetricSpec(
    "kv:invalid_blocks_total",
    MetricType.COUNTER,
    (_CONNECTOR_LABEL,),
)
_REMOTE_WAITING_SPEC = MetricSpec(
    "kv:remote_kv_waiting_requests",
    MetricType.GAUGE,
    (_CONNECTOR_LABEL,),
)
_SPECS = (_FINISHED_SPEC, _FAILED_SPEC, _INVALID_BLOCKS_SPEC, _REMOTE_WAITING_SPEC)

_recorders: dict[str, Any] = {}

# Stable connector label values. The same class name (e.g. MooncakeConnector)
# is reused by different backend modules, so the module wins when it carries
# the disambiguating information.
_MODULE_LABELS = (
    ("mooncake_hybrid_connector", "mooncake_hybrid"),
    ("mooncake_layerwise_connector", "mooncake_layerwise"),
)
_CLASS_LABELS = {
    "MooncakeConnector": "mooncake",
    "MooncakeConnectorScheduler": "mooncake",
    "MooncakePullConnectorScheduler": "mooncake",
    "MooncakeLayerwiseConnector": "mooncake_layerwise",
    "MooncakeLayerwiseConnectorScheduler": "mooncake_layerwise",
    "MooncakeHybridConnector": "mooncake_hybrid",
    "MooncakePullConnector": "mooncake_pull",
    "MooncakePushConnector": "mooncake_push",
    "MooncakeConnectorV2": "mooncake_v2",
    "AscendStoreConnector": "ascend_store",
    "BaseKVPoolScheduler": "ascend_store",
    "KVPoolScheduler": "ascend_store",
    "LayerwiseKVPoolScheduler": "ascend_store",
    "SfaRemoteD2HConnector": "sfa",
    "SFAPDRD2HScheduler": "sfa",
    "SFAPDRD2HConsumerWorker": "sfa",
    "UCMConnectorV1": "ucm",
}
_CLASS_PREFIX_LABELS = (
    ("SFA", "sfa"),
    ("UCM", "ucm"),
)
_WAITING_ATTRIBUTES = ("_reqs_need_recv", "_reqs_recv_info", "_async_loading_req_ids")

_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return _CAMEL_CASE_BOUNDARY.sub("_", name).lower()


def _connector_label(obj: Any) -> str:
    class_name = type(obj).__name__
    module_name = type(obj).__module__
    for marker, label in _MODULE_LABELS:
        if marker in module_name:
            return label
    label = _CLASS_LABELS.get(class_name)
    if label is not None:
        return label
    for prefix, prefix_label in _CLASS_PREFIX_LABELS:
        if class_name.startswith(prefix):
            return prefix_label
    return _camel_to_snake(class_name)


def _record_finished(count: int, connector: Any) -> None:
    if not count:
        return
    record_observation(
        _recorders,
        _KEY,
        _SPECS,
        _FINISHED_SPEC,
        float(count),
        {_CONNECTOR_LABEL: _connector_label(connector)},
        "kv_transfer",
    )


def kv_transfer_finished_handler(original_func, connector, *args, **kwargs):
    """Count finished receives from a worker-side ``get_transfer_results`` poll.

    Used by connectors that have no scheduler hook seeing aggregated worker
    outputs; each TP rank reports its own receive, so the counter scales with
    the TP size.
    """
    result = original_func(connector, *args, **kwargs)

    # The poll result is untrusted connector data: a malformed field must
    # never break the caller, so the extraction stays inside safe_recording.
    finished_count = 0
    with safe_recording("kv_transfer"):
        finished_recving = getattr(result, "finished_recving", None)
        finished_count = len(finished_recving) if finished_recving else 0
    _record_finished(finished_count, connector)
    return result


def kv_transfer_outcome_handler(original_func, scheduler, connector_output, *args, **kwargs):
    """Count per-request receive outcomes from the Decode scheduler.

    ``update_connector_output`` receives worker outputs already aggregated
    across TP ranks, so one remote receive contributes a single outcome no
    matter how many ranks transferred a shard of it. ``finished_recving``
    holds every terminal receive, including the ones also listed in
    ``failed_recving``, so the failed subset is recorded separately.
    """
    result = original_func(scheduler, connector_output, *args, **kwargs)

    # Same untrusted-data rule: extract and size both collections inside
    # safe_recording so a malformed connector output cannot reach inference.
    finished_count = 0
    failed_count = 0
    with safe_recording("kv_transfer"):
        finished = getattr(connector_output, "finished_recving", None)
        if finished:
            finished_count = len(finished)
            failed = set(getattr(connector_output, "failed_recving", None) or ()) & set(finished)
            failed_count = len(failed)
    labels = {_CONNECTOR_LABEL: _connector_label(scheduler)}
    if finished_count:
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _FINISHED_SPEC,
            float(finished_count),
            labels,
            "kv_transfer",
        )
    if failed_count:
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _FAILED_SPEC,
            float(failed_count),
            labels,
            "kv_transfer",
        )
    return result


def kv_transfer_invalid_blocks_handler(original_func, connector, *args, **kwargs):
    """Count blocks that failed to load from the remote KV cache.

    ``get_block_ids_with_load_errors`` returns the pending block ids, so the
    handler only inspects the returned set.
    """
    result = original_func(connector, *args, **kwargs)

    failed_blocks = 0
    with safe_recording("kv_transfer"):
        failed_blocks = len(result) if result else 0
    if failed_blocks:
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _INVALID_BLOCKS_SPEC,
            float(failed_blocks),
            {_CONNECTOR_LABEL: _connector_label(connector)},
            "kv_transfer",
        )
    return result


def _waiting_request_count(scheduler: Any) -> int | None:
    """Return the number of requests still waiting for remote KV.

    ``_reqs_need_recv`` only holds requests queued for the current step: the
    connector clears it inside ``build_connector_meta`` when the READs are
    dispatched, while the transfer keeps running for several scheduler steps.
    Connectors that track in-flight registrations separately (for example
    ``_reqs_recv_info`` on the Mooncake pull scheduler) are unioned in, so a
    multi-step READ does not report zero until it actually completes.

    ``None`` means no known collection exists, which must not be recorded as a
    zero sample.
    """
    waiting: set[str] = set()
    seen_attribute = False
    for attribute in _WAITING_ATTRIBUTES:
        pending = getattr(scheduler, attribute, None)
        if pending is None:
            continue
        seen_attribute = True
        waiting.update(pending)
    return len(waiting) if seen_attribute else None


def kv_transfer_remote_waiting_handler(original_func, scheduler, *args, **kwargs):
    """Report per-step requests still waiting for remote KV.

    ``build_connector_meta`` drains the pending collection before returning,
    so the sample is taken before invoking the wrapped method. Zero is a valid
    sample and is recorded; a missing collection is not.

    The snapshot runs inside ``safe_recording`` and only when the level allows
    it: an unexpected attribute shape must not stop ``build_connector_meta``
    from running, and nothing is extracted when the metric is disabled.
    """
    waiting = None
    if metric_enabled(_REMOTE_WAITING_SPEC):
        with safe_recording("kv_transfer"):
            waiting = _waiting_request_count(scheduler)
    result = original_func(scheduler, *args, **kwargs)

    if waiting is not None:
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _REMOTE_WAITING_SPEC,
            float(waiting),
            {_CONNECTOR_LABEL: _connector_label(scheduler)},
            "kv_transfer",
        )
    return result
