# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-export layer for the Mooncake V1 layerwise connector.

The implementation lives in mooncake_v1/layerwise (connector, scheduler,
worker, threads, protocol) and mooncake_v1/common. This module keeps the
historical import path and the connector registration target stable; it
contains no implementation.
"""

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.constants import (
    KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS,
    SHUTDOWN_JOIN_TIMEOUT_SECONDS,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import (
    DONE_SENDING_MSG,
    FAILED_SENDING_MSG,
    GET_META_MSG,
    LayerMetadata,
    LayerwiseAgentMetadata,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.transport import (
    SizedDict,
    ensure_zmq_recv_with_poller,
    ensure_zmq_send,
    string_to_int64_hash,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.connector import (
    MooncakeLayerwiseConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.protocol import (
    MooncakeLayerwiseConnectorMetadata,
    ReqMeta,
    SendReqInfo,
    SendTask,
    TransferMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.scheduler import (
    MooncakeLayerwiseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.threads import (
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    group_concurrent_contiguous,
    group_contiguous_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.worker import (
    MooncakeLayerwiseConnectorWorker,
)

MooncakeAgentMetadata = LayerwiseAgentMetadata

# Back-compat alias for the historical name; still a distinct variant from the
# common ensure_zmq_recv (this one polls with a timeout before receiving).
ensure_zmq_recv = ensure_zmq_recv_with_poller

__all__ = [
    "DONE_SENDING_MSG",
    "FAILED_SENDING_MSG",
    "GET_META_MSG",
    "KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS",
    "get_external_request_id",
    "msgspec",
    "SHUTDOWN_JOIN_TIMEOUT_SECONDS",
    "KVConnectorRole",
    "KVCacheRecvingLayerThread",
    "KVCacheSendingLayerThread",
    "LayerMetadata",
    "MLAAttentionSpec",
    "MooncakeAgentMetadata",
    "MooncakeLayerwiseConnector",
    "MooncakeLayerwiseConnectorMetadata",
    "MooncakeLayerwiseConnectorScheduler",
    "MooncakeLayerwiseConnectorWorker",
    "ReqMeta",
    "SizedDict",
    "SendReqInfo",
    "SendTask",
    "TransferMeta",
    "UniformTypeKVCacheSpecs",
    "ensure_zmq_recv",
    "ensure_zmq_send",
    "group_concurrent_contiguous",
    "group_contiguous_ids",
    "string_to_int64_hash",
    "zmq_ctx",
]
