# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Protocol records shared by the layerwise mooncake V1 worker and threads.

These dataclasses used to live in the entry module. They are moved verbatim
because KVCacheSendingLayerThread evaluates them in method signatures at class
definition time, so the thread module needs the real objects and may not import
the entry module. The entry module re-imports them, keeping object identity.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import LayerMetadata


@dataclass
class ReqMeta:
    local_block_ids: list[list[int]]
    token_ids: list[int] | None
    # Not None if layer-wise is disabled
    remote_block_ids: list[list[int]]
    remote_block_size: list[list[int]]
    remote_engine_id: str | None
    remote_host: str | None
    remote_port: int | None
    remote_te_rpc_port: int | None
    remote_layer_metadata: dict[str, LayerMetadata] | None
    metaserver: str | None
    remote_tp_size: int | None
    remote_pcp_size: int | None
    remote_dcp_size: int | None
    chunk_finish: bool = False
    prompt_len: int = 0
    trans_count: list[int] | None = None
    remote_cache_tokens: int = 0
    local_computed_tokens: int = 0
    local_transed_tokens: int = 0
    do_virtual: bool = False


@dataclass
class SendTask:
    send_request: dict[str, ReqMeta] = field(default_factory=dict)
    # pd_head_ratio == 1 use
    wait_event: torch.npu.Event | None = None
    # pd_head_ratio > 1 use
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    # kv cache quantization layer use
    k_quant_cache: torch.Tensor | None = None
    v_quant_cache: torch.Tensor | None = None
    layer_idx: int = 0
    layer_name: str = ""
    layer_names: list[str] = field(default_factory=list)
    failed_requests: dict[str, ReqMeta] = field(default_factory=dict)
    # trans block info
    group_rearrange_block_ids: list[list[int]] | None = None
    group_num_blocks: list[int] | None = None
    group_num_tokens: list[int] | None = None
    group_block_table: list[torch.Tensor | None] | None = None
    group_block_len_tensor: list[torch.Tensor | None] | None = None
    group_seq_start_tensor: list[torch.Tensor | None] | None = None


@dataclass
class TransferMeta:
    src: list[int]
    dst: list[int]
    length: list[int]
    req_ids: list[str]


@dataclass
class SendReqInfo:
    local_block_ids: list[list[int]]
    local_transferred_tokens: int
    local_computed_tokens: int
    request: "Request"

    def extend_local_block_ids(self, new_block_ids: list[list[int]]) -> None:
        """extend local block ids for this step"""
        for i, new_block_id in enumerate(new_block_ids):
            self.local_block_ids[i].extend(new_block_id)

    def update_computed_tokens(self, computed_tokens: int) -> None:
        """update local computen tokens for this step"""
        self.local_computed_tokens = computed_tokens

    def update_transferred_tokens(self, transferred_tokens: int) -> None:
        """update transferred tokens for this step"""
        self.local_transferred_tokens = transferred_tokens

    def unpack(self):
        return (
            self.local_block_ids,
            self.local_transferred_tokens,
            self.local_computed_tokens,
            self.request,
        )


class MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.send_task: SendTask = SendTask()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        token_ids: list[int] | None = None,
        chunk_finish: bool = False,
        prompt_len: int = 0,
        remote_cache_tokens: int = 0,
        local_computed_tokens: int = 0,
        local_transed_tokens: int = 0,
    ):
        self.requests[request_id] = ReqMeta(
            token_ids=token_ids or [],
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_block_size=kv_transfer_params.get("remote_block_size", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
            remote_host=kv_transfer_params.get("remote_host"),
            remote_port=kv_transfer_params.get("remote_port"),
            remote_te_rpc_port=kv_transfer_params.get("remote_te_rpc_port"),
            remote_layer_metadata=kv_transfer_params.get("remote_layer_metadata"),
            metaserver=kv_transfer_params.get("metaserver"),
            remote_tp_size=kv_transfer_params.get("remote_tp_size"),
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size"),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size"),
            do_virtual=kv_transfer_params.get("do_virtual"),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            prompt_len=prompt_len,
            local_transed_tokens=local_transed_tokens,
            trans_count=[],
        )


if TYPE_CHECKING:
    from vllm.v1.request import Request
