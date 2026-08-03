# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Metadata types for Mooncake KV transfer connectors."""

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

    All per-layer fields use ``layer_names`` order. Each nested list preserves
    the cache tensor order for that layer, such as K/V or Mamba conv/SSM.
    """

    engine_id: str
    te_rpc_port: int
    block_size: int
    num_blocks: int
    layer_names: list[str]
    group_indices: list[int]
    spec_indices: list[int]
    kv_caches_base_addr: list[list[int]]
    block_strides: list[list[int]]
    block_lens: list[list[int]]
    block_shapes: list[list[tuple[int, ...]]]
    block_size_scales: list[list[int]]
    local_ip: str = ""
    handshake_port: int = 0

    def __post_init__(self) -> None:
        num_layers = len(self.layer_names)
        per_layer_fields = {
            "group_indices": self.group_indices,
            "spec_indices": self.spec_indices,
            "kv_caches_base_addr": self.kv_caches_base_addr,
            "block_strides": self.block_strides,
            "block_lens": self.block_lens,
            "block_shapes": self.block_shapes,
            "block_size_scales": self.block_size_scales,
        }
        for field_name, values in per_layer_fields.items():
            if len(values) != num_layers:
                raise ValueError(
                    f"Mooncake transfer metadata field {field_name!r} has "
                    f"{len(values)} layers, expected {num_layers}."
                )

        for layer_index, layer_name in enumerate(self.layer_names):
            num_addrs = len(self.kv_caches_base_addr[layer_index])
            for field_name in (
                "block_strides",
                "block_lens",
                "block_shapes",
                "block_size_scales",
            ):
                values = per_layer_fields[field_name][layer_index]
                if len(values) != num_addrs:
                    raise ValueError(
                        f"Mooncake transfer metadata for layer {layer_name!r} "
                        f"has {len(values)} {field_name}, expected {num_addrs}."
                    )


@dataclass
class ReqMeta:
    """Request metadata required by the initial Mooncake pull path."""

    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    num_prompt_blocks: int
    remote_block_size: int


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
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ) -> None:
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            remote_block_size=kv_transfer_params.get("remote_block_size", 0),
        )


__all__ = [
    "MooncakeConnectorMetadata",
    "MooncakeTransferMetadata",
    "ReqMeta",
]
