# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PD 分离（Prefill/Decode 分离）场景下的 KV 传输指标。

Prefill 与 Decode 部署在不同实例上，Decode 侧通过 KV connector 从
Prefill 拉取已经算好的 KV cache。本模块只在既有 connector/scheduler
方法上挂点，不改动 ``vllm_ascend/distributed/kv_transfer`` 的业务代码：

- 每次轮询结果时完成的接收请求数（``xfer_finished_total``）；
- 加载失败的 block 数（``invalid_blocks_total``）；
- 每一步仍在等待远端 KV 的请求数（``remote_kv_waiting_requests``）。
"""

import re
from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    MetricSpec,
    MetricType,
    record_observation,
)

_KEY = "kv_transfer"
_CONNECTOR_LABEL = "connector"

_FINISHED_SPEC = MetricSpec(
    "kv:xfer_finished_total",
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
_SPECS = (_FINISHED_SPEC, _INVALID_BLOCKS_SPEC, _REMOTE_WAITING_SPEC)

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
_WAITING_ATTRIBUTES = ("_reqs_need_recv", "_async_loading_req_ids")

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
    """Count finished receives from ``KVConnectorTransferResults``.

    Used by connectors that implement the V2 ``get_transfer_results`` API.
    """
    result = original_func(connector, *args, **kwargs)

    finished_recving = getattr(result, "finished_recving", None)
    _record_finished(len(finished_recving) if finished_recving else 0, connector)
    return result


def kv_transfer_finished_pair_handler(original_func, connector, *args, **kwargs):
    """Count finished receives from a ``(finished_sending, finished_recving)`` pair.

    Used by connectors that only expose ``get_finished``.
    """
    result = original_func(connector, *args, **kwargs)

    finished_recving = result[1] if result else None
    _record_finished(len(finished_recving) if finished_recving else 0, connector)
    return result


def kv_transfer_invalid_blocks_handler(original_func, connector, *args, **kwargs):
    """Count blocks that failed to load from the remote KV cache.

    ``get_block_ids_with_load_errors`` returns the pending block ids, so the
    handler only inspects the returned set.
    """
    result = original_func(connector, *args, **kwargs)

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

    ``None`` means neither known collection exists, which must not be recorded
    as a zero sample.
    """
    for attribute in _WAITING_ATTRIBUTES:
        if hasattr(scheduler, attribute):
            pending = getattr(scheduler, attribute)
            return len(pending) if pending is not None else 0
    return None


def kv_transfer_remote_waiting_handler(original_func, scheduler, *args, **kwargs):
    """Report per-step requests still waiting for remote KV.

    ``build_connector_meta`` drains the pending collection before returning,
    so the sample is taken before invoking the wrapped method. Zero is a valid
    sample and is recorded; a missing collection is not.
    """
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
