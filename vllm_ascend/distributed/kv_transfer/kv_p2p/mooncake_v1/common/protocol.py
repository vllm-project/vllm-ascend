# SPDX-License-Identifier: Apache-2.0
"""Wire protocol structures shared by the mooncake v1 KV connectors.

The pull connector (``mooncake_connector``) and the layerwise connector
(``mooncake_layerwise_connector``) exchange these message types and metadata
structures over ZMQ. The field names, field order and default values are part
of the wire format and must not change: new code has to stay compatible with
peers running the old definitions. This module intentionally depends only on
the standard library and ``msgspec``.
"""

from dataclasses import dataclass
from typing import Any, TypedDict

import msgspec

# Messages exchanged by the pull (P/D) connector handshake.
GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"

# Messages exchanged by the layerwise connector handshake.
DONE_SENDING_MSG = b"done_sending_msg"
FAILED_SENDING_MSG = b"failed_sending_msg"


class RemotePortInfo(TypedDict):
    num: int
    host: str


class PullAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]]
    block_size: int
    kv_caches_base_addr: list[list[int]]
    block_size_scale: list[list[int]]
    num_blocks: int
    block_lens: list[list[int]]
    block_strides: list[list[int]]
    local_ip: str = ""
    handshake_port: int = 0


@dataclass
class LayerMetadata:
    tensor_group_idx: list[int]
    kv_caches_base_addr: list[int]
    block_len: list[int]
    block_size_scale: list[int]


class LayerwiseAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    te_rpc_port: int
    layer_metadata: dict[str, LayerMetadata]


__all__ = [
    "DONE_RECVING_MSG",
    "DONE_SENDING_MSG",
    "FAILED_SENDING_MSG",
    "GET_META_MSG",
    "LayerMetadata",
    "LayerwiseAgentMetadata",
    "PullAgentMetadata",
    "RemotePortInfo",
]
