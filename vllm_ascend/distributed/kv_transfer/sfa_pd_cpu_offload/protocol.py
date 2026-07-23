# SPDX-License-Identifier: Apache-2.0
"""Wire protocol helpers for SFA PD CPU offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)

BATCH_KV_TRANSFER_PARAMS = "batch_kv_transfer_params"
MF_META = b"mf_meta"
READ_READY_BATCH = b"read_ready_batch"
READ_DONE = b"read_done"
READ_FAILED = b"read_failed"


def infer_sfa_component_group_ids(kv_cache_config: Any) -> tuple[int, int]:
    """Return ``(main_group_id, indexer_group_id)`` from layer names.

    SFA layouts may keep main and indexer caches in separate groups or in one
    uniform group.  The PD protocol must not rely on a fixed group order.
    """

    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    if not groups:
        raise ValueError("SFA PD transfer requires at least one KV cache group")

    main_group_id = None
    indexer_group_id = None
    for group_id, group in enumerate(groups):
        layer_names = list(getattr(group, "layer_names", ()) or ())
        if indexer_group_id is None and any("indexer" in name.lower() for name in layer_names):
            indexer_group_id = group_id
        if main_group_id is None and any("indexer" not in name.lower() for name in layer_names):
            main_group_id = group_id

    if main_group_id is None:
        raise ValueError("SFA PD transfer did not find a main KV cache group")
    if indexer_group_id is None:
        # Some non-C8 layouts bind indexer storage to the same vLLM block
        # group as main MLA KV.
        indexer_group_id = main_group_id
    return main_group_id, indexer_group_id


@dataclass
class LayerMetadata:
    tensor_group_idx: list[int]
    kv_caches_base_addr: list[int]
    block_len: list[int]
    block_size_scale: list[int]
    # The first ``main_tensor_count`` entries belong to the main MLA cache.
    # Remaining entries, when ``has_indexer`` is true, belong to the indexer.
    # Keeping this explicit is required for sparse layers that reuse top-k
    # indices and therefore do not own an indexer cache at all.
    main_tensor_count: int = 2
    has_indexer: bool = False


@dataclass
class SfaPDProducerReqMeta:
    local_block_ids: list[list[int]]
    token_ids: list[int]
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


class SfaPDProducerMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.requests: dict[str, SfaPDProducerReqMeta] = {}

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
    ) -> None:
        self.requests[request_id] = SfaPDProducerReqMeta(
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
            do_virtual=kv_transfer_params.get("do_virtual", False),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            prompt_len=prompt_len,
            local_transed_tokens=local_transed_tokens,
            trans_count=[],
        )


@dataclass
class SfaPDConsumerReqMeta:
    """D-side destinations owned by vLLM's normal KV block manager.

    Main MLA block ids index the CPU cache owned by
    :class:`KVOffloadDecodeManager`; indexer block ids index the rank-local
    HBM cache.  Keeping the two components explicit avoids depending on a
    model-runner tuple layout.
    """

    req_id: str
    main_block_ids: list[int]
    indexer_block_ids: list[int]


class SfaPDConsumerMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.requests: list[SfaPDConsumerReqMeta] = []

    def add_request(
        self,
        request_id: str,
        main_block_ids: list[int],
        indexer_block_ids: list[int],
    ) -> None:
        self.requests.append(
            SfaPDConsumerReqMeta(
                req_id=request_id,
                main_block_ids=list(main_block_ids),
                indexer_block_ids=list(indexer_block_ids),
            )
        )


@dataclass
class SendTask:
    send_request: dict[str, SfaPDProducerReqMeta]
    wait_event: Any | None = None
    layer_idx: int = 0
    layer_name: str = ""


def get_external_request_id(request_id: str) -> str:
    # vLLM appends a 9-character EngineCore suffix to request IDs.
    # Guard short / malformed ids so we never return an empty or garbage id
    # (which would corrupt the external_req_id -> internal_req_id map).
    return request_id[:-9] if len(request_id) > 9 else request_id
