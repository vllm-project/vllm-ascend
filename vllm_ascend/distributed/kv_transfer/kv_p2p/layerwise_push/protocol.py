# SPDX-License-Identifier: Apache-2.0
"""Wire protocol for backend-independent layerwise push."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)

BATCH_KV_TRANSFER_PARAMS = "batch_kv_transfer_params"
LAYOUT_META = b"layout_meta"
TARGET_BLOCKS = b"target_blocks"
REQUEST_DONE = b"request_done"
WRITE_FAILED = b"write_failed"


@dataclass(frozen=True)
class LayerwisePushHandshakeMetadata(KVConnectorHandshakeMetadata):
    layer_ids: tuple[int, ...]
    host: str = ""
    # D workers serve destination metadata; P workers publish layer ownership.
    port: int = 0


@dataclass(frozen=True)
class ComponentLayout:
    """One semantic cache component, containing one or more tensors."""

    name: str
    group_index: int
    block_size: int
    dtypes: tuple[str, ...]
    base_addrs: tuple[int, ...]
    block_strides: tuple[int, ...]
    block_lengths: tuple[int, ...]
    block_shapes: tuple[tuple[int, ...], ...]
    block_size_scales: tuple[int, ...]

    def __post_init__(self) -> None:
        tensor_count = len(self.base_addrs)
        field_counts = (
            len(self.dtypes),
            len(self.block_strides),
            len(self.block_lengths),
            len(self.block_shapes),
            len(self.block_size_scales),
        )
        if any(count != tensor_count for count in field_counts):
            raise ValueError(
                f"Component {self.name!r} has inconsistent tensor metadata: "
                f"base_addrs={tensor_count}, other_counts={field_counts}"
            )


@dataclass
class LayerwisePushProducerReqMeta:
    local_block_ids: list[list[int]]
    remote_tp_size: int | None
    remote_pp_size: int | None = None
    chunk_finish: bool = False
    remote_cache_tokens: int = 0
    local_computed_tokens: int = 0
    local_transed_tokens: int = 0
    chunk_start_blocks: list[int] | None = None
    # Contributor-group identity for unequal P/D TP. ratio = p_tp // d_tp P ranks
    # map to one D rank; group_member_idx is this rank's index within that group.
    # ratio == 1 (equal TP) degenerates to a single contributor.
    tp_ratio: int = 1
    group_member_idx: int = 0
    remote_endpoints: list[list[dict[str, Any]]] | None = None
    remote_topology_id: str | None = None
    layer_endpoints: dict[int, tuple[str, int]] = field(default_factory=dict)
    terminal_layers: frozenset[int] = frozenset()


class LayerwisePushProducerMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.requests: dict[str, LayerwisePushProducerReqMeta] = {}
        self.producer_pp_layers: tuple[tuple[int, ...], ...] = ()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        chunk_finish: bool = False,
        remote_cache_tokens: int = 0,
        local_computed_tokens: int = 0,
        local_transed_tokens: int = 0,
        chunk_start_blocks: list[int] | None = None,
    ) -> None:
        self.requests[request_id] = LayerwisePushProducerReqMeta(
            local_block_ids=local_block_ids,
            remote_tp_size=kv_transfer_params.get("remote_tp_size"),
            remote_pp_size=kv_transfer_params.get("remote_pp_size"),
            remote_endpoints=kv_transfer_params.get("remote_endpoints"),
            remote_topology_id=kv_transfer_params.get("remote_topology_id"),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            local_transed_tokens=local_transed_tokens,
            chunk_start_blocks=chunk_start_blocks,
        )


@dataclass
class LayerwisePushConsumerReqMeta:
    """D-side block ids for every KV cache group."""

    req_id: str
    block_ids_by_group: list[list[int]]


class LayerwisePushConsumerMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.requests: list[LayerwisePushConsumerReqMeta] = []

    def add_request(
        self,
        request_id: str,
        block_ids: list[list[int]],
    ) -> None:
        self.requests.append(
            LayerwisePushConsumerReqMeta(
                req_id=request_id,
                block_ids_by_group=[list(group) for group in block_ids],
            )
        )


@dataclass
class SendTask:
    send_request: dict[str, LayerwisePushProducerReqMeta]
    wait_event: Any | None = None
    layer_idx: int = 0
    layer_name: str = ""
    reservation_id: tuple[int, str] | None = None


def get_external_request_id(request_id: str) -> str:
    # vLLM appends a 9-character EngineCore suffix to request IDs.
    # Guard short / malformed ids so we never return an empty or garbage id
    # (which would corrupt the external_req_id -> internal_req_id map).
    return request_id[:-9] if len(request_id) > 9 else request_id
