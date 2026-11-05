# SPDX-License-Identifier: Apache-2.0
"""Data models of the pull (P/D) mooncake v1 connector.

The metadata classes moved here verbatim from ``mooncake_connector`` so the pull
worker/scheduler can share them with the transfer planner that follows in a
later node. ``TransferRequest`` is a frozen dataclass view of the legacy
``trans_info`` dict used by ``KVCacheRecvingThread.add_request``; the receive
thread still enqueues ``as_legacy_dict()`` because d2rh keeps consuming the
legacy dict (``_transfer_kv_cache_all_groups(req_meta: dict)``), so the dict
seam must stay.

This module only depends on vLLM data types/base classes and the lower-level
``common.protocol`` module; it must not import the old connector modules.
"""

from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import RemotePortInfo


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds

    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_pcp_size: int
    remote_dcp_size: int
    remote_ptp_size: int | None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int
    remote_block_size: int
    local_full_block_ids: BlockIds = tuple()
    do_virtual: bool = False


@dataclass(frozen=True)
class GroupPull:
    group_id: int
    remote_tp_offset: int
    num_group_pulls: int
    prefill_pp_rank: int = 0
    is_group_transfer_end: bool = False


@dataclass(frozen=True)
class TransferRequest:
    """Frozen view of the legacy ``trans_info`` payload of one transfer request.

    Field order and defaults mirror the keyword parameters of
    ``KVCacheRecvingThread.add_request``. ``as_legacy_dict()`` reproduces the
    legacy dict exactly (same keys, same order, same normalization) because d2rh
    still consumes that dict.
    """

    request_id: str
    remote_request_id: str
    local_block_ids: BlockIds
    remote_block_ids: BlockIds
    group_pulls: list[GroupPull]
    remote_engine_id: str
    remote_host: str
    remote_handshake_port: int
    remote_block_size: Any = None
    remote_port_send_num: dict[int, RemotePortInfo] | None = None
    num_computed_tokens: int = 0
    all_task_done: bool = False
    shard_idx: int = 0
    local_block_ids_replicate_k: BlockIds | None = None
    remote_block_ids_replicate_k: BlockIds | None = None
    awaiting_kvs: bool = True

    @classmethod
    def from_legacy_kwargs(cls, **kwargs: Any) -> "TransferRequest":
        """Build a request from the legacy ``add_request`` keyword arguments."""
        return cls(**kwargs)

    @classmethod
    def from_legacy_dict(cls, payload: dict[str, Any]) -> "TransferRequest":
        """Build a request from a legacy ``trans_info`` dict payload."""
        return cls(**payload)

    def as_legacy_dict(self) -> dict[str, Any]:
        """Return the legacy ``trans_info`` dict (same keys/order/normalization)."""
        remote_port_send_num = self.remote_port_send_num
        if remote_port_send_num is None:
            remote_port_send_num = {}
        return {
            "request_id": self.request_id,
            "local_block_ids": self.local_block_ids,
            "remote_block_ids": self.remote_block_ids,
            "local_block_ids_replicate_k": self.local_block_ids_replicate_k or tuple(),
            "remote_block_ids_replicate_k": self.remote_block_ids_replicate_k or tuple(),
            "group_pulls": self.group_pulls,
            "remote_engine_id": self.remote_engine_id,
            "remote_request_id": self.remote_request_id,
            "remote_host": self.remote_host,
            "remote_handshake_port": self.remote_handshake_port,
            "num_computed_tokens": self.num_computed_tokens,
            "remote_port_send_num": remote_port_send_num,
            "all_task_done": self.all_task_done,
            "shard_idx": self.shard_idx,
            "remote_block_size": self.remote_block_size,
            "awaiting_kvs": self.awaiting_kvs,
        }


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
        local_full_block_ids: BlockIds | None = None,
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
            remote_ptp_size=kv_transfer_params.get("remote_ptp_size"),
            remote_multi_nodes_meta_mapping=kv_transfer_params.get("remote_multi_nodes_meta_mapping", {}),
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            remote_block_size=kv_transfer_params.get("remote_block_size", 0),
            local_full_block_ids=local_full_block_ids or tuple(),
            do_virtual=kv_transfer_params.get("do_virtual", False),
        )


__all__ = [
    "GroupPull",
    "MooncakeConnectorMetadata",
    "ReqMeta",
    "TransferRequest",
]
