# SPDX-License-Identifier: Apache-2.0
"""Cache-key identity, wire encoding, and lazy block-hash views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashList

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerPoolKey


@dataclass
class KeyMetadata:
    """name of the LLM model"""

    model_name: str
    """ worker id when running under a distributed setting """
    head_or_tp_rank: int
    """ Initialize the current decode context model parallel rank """
    dcp_rank: int
    """ Initialize the current pipeline parallel rank """
    pp_rank: int
    """ Initialize the current kv cache group id """
    kv_cache_group_id: int = 0
    """ Differentiate kv/state keys that share the same chunk hash """
    cache_role: str = "kv"
    """ Family name for compress-aware hybrid cache layouts """
    cache_family: str = "default"


@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@dcp:{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int, layer_offset: int = 0) -> list[LayerPoolKey]:
        """Split the key into multiple keys for each layer"""
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerPoolKey

        keys = []
        for layer_id in range(layer_offset, layer_offset + num_layers):
            keys.append(
                LayerPoolKey(
                    self.key_metadata,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys


def get_block_hashes(
    block_hashes: BlockHashList | list[str],
    group_block_size: int,
    hash_block_size: int,
) -> Sequence[BlockHash | str]:
    if group_block_size == hash_block_size:
        return block_hashes
    assert group_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
    return _LazyGroupedBlockHashList(block_hashes, group_block_size // hash_block_size)


class _LazyGroupedBlockHashList(Sequence[BlockHash | str]):
    def __init__(self, block_hashes: Sequence[BlockHash | str], scale_factor: int) -> None:
        self._block_hashes = block_hashes
        self._scale_factor = scale_factor
        self._length = len(block_hashes) // scale_factor

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        # Chained hashes make the final fine-grained hash identify the
        # complete larger block. Resolve it lazily so filtered paths avoid
        # traversing hashes for chunks they will not use.
        return self._block_hashes[(index + 1) * self._scale_factor - 1]


def block_hash_to_str(block_hash: BlockHash | str) -> str:
    return block_hash if isinstance(block_hash, str) else block_hash.hex()


def block_hash_to_bytes(block_hash: BlockHash | str) -> bytes:
    if isinstance(block_hash, str):
        if len(block_hash) == 64:
            try:
                return bytes.fromhex(block_hash)
            except ValueError:
                pass
        return block_hash.encode("utf-8")
    return bytes(block_hash)
