# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-export layer for the Mooncake V1 pull connector.

The implementation lives in mooncake_v1/pull (connector, scheduler, worker,
threads) and mooncake_v1/common. This module keeps the historical import path
and the connector registration target stable; it contains no implementation.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# Kept as a module attribute: the reformat mixin imports it directly now, and the
# historical entry-module name stays importable for existing callers/tests.
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first  # noqa: F401

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.block_selection import (
    GroupTransferInfo,
    build_layer_name_to_metadata_idx,
    get_prefill_pp_indices,
    resolve_remote_layer_idx,
    transfer_groups_need_independent_block_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.constants import (
    KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS,
    SHUTDOWN_JOIN_TIMEOUT_SECONDS,
    THREAD_STOP_POLL_SECONDS,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import (
    DONE_RECVING_MSG,
    GET_META_MSG,
    PullAgentMetadata,
    RemotePortInfo,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.tracker import KVCacheTaskTracker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.transport import (
    SizedDict,
    ensure_zmq_recv,
    ensure_zmq_send,
    group_concurrent_contiguous,
    split_if_not_byte_contiguous,
    string_to_int64_hash,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.connector import MooncakeConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.metadata import (
    GroupPull,
    MooncakeConnectorMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.scheduler import MooncakeConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.threads import (
    MAX_REQUESTS_PER_PEER_HANDLER,
    KVCacheRecvingThread,
    KVCacheSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.worker import (
    KV_CACHE_BUFFER_ALIGNMENT,
    MooncakeConnectorWorker,
)

MooncakeAgentMetadata = PullAgentMetadata

__all__ = [
    "is_conv_state_dim_first",
    "DONE_RECVING_MSG",
    "GET_META_MSG",
    "KV_CACHE_BUFFER_ALIGNMENT",
    "KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS",
    "SHUTDOWN_JOIN_TIMEOUT_SECONDS",
    "GroupPull",
    "GroupTransferInfo",
    "KVCacheRecvingThread",
    "KVCacheSendingThread",
    "KVCacheTaskTracker",
    "KVConnectorRole",
    "MAX_REQUESTS_PER_PEER_HANDLER",
    "MooncakeAgentMetadata",
    "MooncakeConnector",
    "MooncakeConnectorMetadata",
    "MooncakeConnectorScheduler",
    "MooncakeConnectorWorker",
    "RemotePortInfo",
    "ReqMeta",
    "SizedDict",
    "THREAD_STOP_POLL_SECONDS",
    "build_layer_name_to_metadata_idx",
    "ensure_zmq_recv",
    "ensure_zmq_send",
    "get_prefill_pp_indices",
    "group_concurrent_contiguous",
    "resolve_remote_layer_idx",
    "split_if_not_byte_contiguous",
    "string_to_int64_hash",
    "transfer_groups_need_independent_block_ids",
    "zmq_ctx",
]
