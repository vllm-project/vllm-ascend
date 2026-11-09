# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request handoff, separate from pool object keys and physical addresses."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorWorkerMetadata
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
)

POOL_PD_PROTOCOL = "ascend_shared_v2"
POOL_PD_MTP_PROTOCOL = "ascend_shared_mtp_v1"
HANDOFF_DEADLINE_MARGIN_S = 1.0


def prompt_digest(request) -> str:
    """Include the same salt/LoRA/MM/embedding identity as prefix caching.

    Hash the complete prompt once at the handoff boundary. A tail is never
    identified by its last few token IDs alone. No Python hash() or rank seed.
    """
    tokens = request.prompt_token_ids
    if tokens is None:
        raise ValueError("Pool PD requires prompt token IDs")
    extras, _ = generate_block_hash_extra_keys(request, 0, len(tokens), 0)
    return sha256_cbor((POOL_PD_PROTOCOL, tuple(tokens), extras)).hex()


@dataclass
class PoolPDRequest:
    request_id: str
    prompt_tokens: int
    prompt_digest: str
    block_ids: list[list[int]]
    block_hashes: list[bytes]
    local_tokens: int = 0
    cache_tokens: int | None = None
    handoff_deadline: float | None = None
    unhashed_block_ids: list[list[int]] | None = None


@dataclass
class PoolPDMetadata(KVConnectorMetadata):
    pool: AscendConnectorMetadata
    receives: list[PoolPDRequest] = field(default_factory=list)
    tails: list[PoolPDRequest] = field(default_factory=list)


@dataclass
class PoolPDWorkerMetadata(AscendStoreKVConnectorWorkerMetadata):
    failed_handoffs: set[str] = field(default_factory=set)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, PoolPDWorkerMetadata)
        events = super().aggregate(other)
        return PoolPDWorkerMetadata(events.completed_events, self.failed_handoffs | other.failed_handoffs)
