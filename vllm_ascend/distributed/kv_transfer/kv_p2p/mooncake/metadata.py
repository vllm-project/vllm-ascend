# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Metadata types for Mooncake KV transfer connectors."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)


@dataclass(frozen=True)
class MooncakeTransferMetadata(KVConnectorHandshakeMetadata):
    """Worker transfer information exchanged during the P/D handshake.

    All per-layer fields use layer_names order; each nested list preserves
    that layer's cache tensor order.
    """

    engine_id: str
    te_rpc_port: int
    block_size: int
    # Token stripe width used by this worker's CP slot mapping.
    cp_kv_cache_interleave_size: int
    num_blocks: int
    layer_names: list[str]
    layer_block_sizes: list[int]
    group_indices: list[int]
    kv_caches_base_addr: list[list[int]]
    block_strides: list[list[int]]
    block_lens: list[list[int]]
    block_shapes: list[list[tuple[int, ...]]]
    block_size_scales: list[list[int]]
    # Per layer: 0 for non-indexer caches, 1 for unreplicated indexers,
    # >1 for replicated indexers. Distinct from kernel blocks per scheduler page.
    indexer_replication_sizes: list[int]
    local_ip: str = ""
    handshake_port: int = 0

    def __post_init__(self) -> None:
        if self.cp_kv_cache_interleave_size <= 0:
            raise ValueError("Mooncake CP interleave size must be positive.")
        num_layers = len(self.layer_names)
        if len(self.indexer_replication_sizes) != num_layers or any(
            size < 0 for size in self.indexer_replication_sizes
        ):
            raise ValueError("Invalid per-layer indexer replication sizes.")
        per_layer_fields: tuple[tuple[str, Sequence[object]], ...] = (
            ("layer_block_sizes", self.layer_block_sizes),
            ("group_indices", self.group_indices),
            ("kv_caches_base_addr", self.kv_caches_base_addr),
            ("block_strides", self.block_strides),
            ("block_lens", self.block_lens),
            ("block_shapes", self.block_shapes),
            ("block_size_scales", self.block_size_scales),
        )
        for field_name, values in per_layer_fields:
            if len(values) != num_layers:
                raise ValueError(
                    f"Mooncake transfer metadata field {field_name!r} has {len(values)} layers, expected {num_layers}."
                )

        nested_per_layer_fields: tuple[tuple[str, Sequence[Sequence[object]]], ...] = (
            ("block_strides", self.block_strides),
            ("block_lens", self.block_lens),
            ("block_shapes", self.block_shapes),
            ("block_size_scales", self.block_size_scales),
        )
        for layer_index, layer_name in enumerate(self.layer_names):
            num_addrs = len(self.kv_caches_base_addr[layer_index])
            for field_name, per_layer_values in nested_per_layer_fields:
                values = per_layer_values[layer_index]
                if len(values) != num_addrs:
                    raise ValueError(
                        f"Mooncake transfer metadata for layer {layer_name!r} "
                        f"has {len(values)} {field_name}, expected {num_addrs}."
                    )


@dataclass(frozen=True)
class MooncakeTPTransferMetadata:
    """TP-private connection and KV-cache address information."""

    te_rpc_port: int
    # PP-union layer indices physically owned by this TP rank. The address
    # table remains aligned with the PP-union layer order; unowned entries are
    # empty lists.
    layer_indices: list[int]
    kv_caches_base_addr: list[list[int]]
    local_ip: str
    handshake_port: int


@dataclass(frozen=True)
class MooncakePCPTransferMetadata:
    """Worker metadata for one PCP rank across its TP ranks."""

    metadata_by_tp_rank: dict[int, MooncakeTPTransferMetadata]


@dataclass(frozen=True)
class MooncakePPTransferMetadata:
    """Metadata shared by all PCP replicas belonging to one PP rank."""

    block_size: int
    # Token stripe width used by this worker's CP slot mapping.
    cp_kv_cache_interleave_size: int
    num_blocks: int
    layer_names: list[str]
    layer_block_sizes: list[int]
    group_indices: list[int]
    block_strides: list[list[int]]
    block_lens: list[list[int]]
    block_shapes: list[list[tuple[int, ...]]]
    block_size_scales: list[list[int]]
    metadata_by_pcp_rank: dict[int, MooncakePCPTransferMetadata]
    indexer_replication_sizes: list[int]


@dataclass(frozen=True)
class MooncakeTransferMetadataGroups:
    """PP-grouped worker transfer metadata exposed by one DP scheduler."""

    engine_id: str
    scheduler_host: str
    scheduler_port: int
    pp_size: int
    pcp_size: int
    dcp_size: int
    tp_size: int
    use_kv_pp: bool
    metadata_by_pp_rank: dict[int, MooncakePPTransferMetadata]


@dataclass
class ReqMeta:
    """Request metadata required by the initial Mooncake pull path."""

    local_block_ids: BlockIds
    local_num_prompt_tokens: int
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_num_prompt_tokens: int
    local_full_block_ids: BlockIds


@dataclass(frozen=True)
class PushReqMeta:
    """D allocation snapshot, installed before consuming completion messages."""

    request_id: str
    local_block_ids: tuple[tuple[int, ...], ...]
    num_computed_tokens: int
    num_external_tokens: int


@dataclass(frozen=True)
class PushSendReqMeta:
    """P source blocks and the prefill range scheduled for this step.

    num_computed_tokens includes the cached prefix; the worker may publish
    that prefix immediately, but must wait for layer events for new tokens.
    """

    # Full D request ID, also used by completion messages.
    request_id: str
    local_block_ids: tuple[tuple[int, ...], ...]
    num_prompt_tokens: int
    num_computed_tokens: int
    num_scheduled_tokens: int
    remote_engine_id: str
    remote_host: str
    remote_port: int
    remote_block_ids: tuple[tuple[int, ...], ...]
    remote_num_computed_tokens: int
    remote_num_external_tokens: int
    remote_worker_addresses: tuple[tuple[int, int, int, str], ...]


class MooncakeConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for Mooncake KV transfers."""

    def __init__(self) -> None:
        self.requests: dict[str, ReqMeta] = {}
        self.push_requests: dict[str, PushReqMeta] = {}
        self.push_send_requests: dict[str, PushSendReqMeta] = {}
        # P request ID -> D completion routing, including pre-dispatch aborts.
        self.push_aborted_requests: dict[str, dict[str, Any]] = {}
        # D request ID -> whether the failed metaserver call never dispatched P.
        self.push_failed_requests: dict[str, bool] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        local_num_prompt_tokens: int,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
        local_full_block_ids: BlockIds | None = None,
    ) -> None:
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            local_num_prompt_tokens=local_num_prompt_tokens,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_num_prompt_tokens=kv_transfer_params["remote_num_prompt_tokens"],
            local_full_block_ids=local_full_block_ids or tuple(),
        )

    def add_new_push_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ) -> None:
        self.push_requests[request_id] = PushReqMeta(
            request_id=request_id,
            local_block_ids=tuple(tuple(group) for group in local_block_ids),
            num_computed_tokens=kv_transfer_params["num_computed_tokens"],
            num_external_tokens=num_external_tokens,
        )

    def add_new_push_send_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_prompt_tokens: int,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        kv_transfer_params: dict[str, Any],
    ) -> None:
        """Snapshot one scheduled P chunk for the layerwise worker."""
        self.push_send_requests[request_id] = PushSendReqMeta(
            request_id=kv_transfer_params["remote_request_id"],
            local_block_ids=tuple(tuple(group) for group in local_block_ids),
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_block_ids=tuple(tuple(group) for group in kv_transfer_params["remote_block_ids"]),
            remote_num_computed_tokens=kv_transfer_params["remote_num_computed_tokens"],
            remote_num_external_tokens=kv_transfer_params["remote_num_external_tokens"],
            remote_worker_addresses=tuple(tuple(worker) for worker in kv_transfer_params["remote_worker_addresses"]),
        )


__all__ = [
    "MooncakeConnectorMetadata",
    "MooncakePCPTransferMetadata",
    "MooncakePPTransferMetadata",
    "MooncakeTPTransferMetadata",
    "MooncakeTransferMetadata",
    "MooncakeTransferMetadataGroups",
    "PushReqMeta",
    "PushSendReqMeta",
    "ReqMeta",
]
