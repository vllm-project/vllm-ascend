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
    num_blocks: int
    layer_names: list[str]
    layer_block_sizes: list[int]
    group_indices: list[int]
    kv_caches_base_addr: list[list[int]]
    block_strides: list[list[int]]
    block_lens: list[list[int]]
    block_shapes: list[list[tuple[int, ...]]]
    block_size_scales: list[list[int]]
    local_ip: str = ""
    handshake_port: int = 0

    def __post_init__(self) -> None:
        num_layers = len(self.layer_names)
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
class MooncakePPTransferMetadata:
    """Metadata shared by all TP workers belonging to one PP rank."""

    block_size: int
    num_blocks: int
    layer_names: list[str]
    layer_block_sizes: list[int]
    group_indices: list[int]
    block_strides: list[list[int]]
    block_lens: list[list[int]]
    block_shapes: list[list[tuple[int, ...]]]
    block_size_scales: list[list[int]]
    metadata_by_tp_rank: dict[int, MooncakeTPTransferMetadata]


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


class MooncakeConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for Mooncake KV transfers."""

    def __init__(self) -> None:
        self.requests: dict[str, ReqMeta] = {}
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


__all__ = [
    "MooncakeConnectorMetadata",
    "MooncakePPTransferMetadata",
    "MooncakeTPTransferMetadata",
    "MooncakeTransferMetadata",
    "MooncakeTransferMetadataGroups",
    "ReqMeta",
]
