# SPDX-License-Identifier: Apache-2.0
"""Token chunk selection and cache transfer address preparation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import compress
from typing import Any

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashList

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import GroupBufferLayout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_keys import (
    KeyMetadata,
    PoolKey,
    block_hash_to_str,
    get_block_hashes,
)


class ChunkedTokenDatabase:
    def __init__(
        self,
        metadata: list[KeyMetadata],
        block_size: list[int],
        partitions: list[int] | None,
        hash_block_size: int | None = None,
    ):
        self.metadata = metadata
        self.block_size = block_size
        self.group_layouts: dict[int, GroupBufferLayout] = {}
        self.group_cache_families: dict[str, dict[int, str]] = {
            "kv": {},
            "state": {},
        }
        self.group_num_layers: dict[str, dict[int, int]] = {
            "kv": {},
            "state": {},
        }
        self.partitions = partitions
        self.hash_block_size = self.block_size[0] if hash_block_size is None else hash_block_size
        self._key_prefix_cache: dict[tuple[int, str, str], str] = {}
        self.cache_coordinator: Any | None = None

    def _get_key_prefix(
        self,
        kv_cache_group_id: int,
        cache_role: str = "kv",
        cache_family: str | None = None,
    ) -> str:
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
        cache_key = (kv_cache_group_id, cache_role, cache_family)
        prefix = self._key_prefix_cache.get(cache_key)
        if prefix is None:
            group_metadata = self.metadata[kv_cache_group_id]
            prefix = (
                f"{group_metadata.model_name}"
                f"@dcp:{group_metadata.dcp_rank}"
                f"@head_or_tp_rank:{group_metadata.head_or_tp_rank}"
                f"@pp_rank:{group_metadata.pp_rank}"
                f"@group:{kv_cache_group_id}"
                f"@cache_role:{cache_role}"
                f"@cache_family:{cache_family}@"
            )
            self._key_prefix_cache[cache_key] = prefix
        return prefix

    def store_mask(
        self,
        aligned_token_len: int,
        num_prompt_tokens: int | None = None,
    ) -> tuple[list[bool], ...] | None:
        if self.cache_coordinator is None:
            return None
        return self.cache_coordinator.store_mask(aligned_token_len, num_prompt_tokens)

    def load_mask(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...] | None:
        if self.cache_coordinator is None:
            return None
        return self.cache_coordinator.load_mask(block_hashes, token_len)

    def mask_allows_chunk(
        self,
        masks: tuple[list[bool], ...] | None,
        kv_cache_group_id: int,
        start: int,
    ) -> bool:
        if masks is None or kv_cache_group_id >= len(masks):
            return True
        group_mask = masks[kv_cache_group_id]
        block_idx = start // self.get_block_size(kv_cache_group_id)
        return block_idx < len(group_mask) and group_mask[block_idx]

    def _make_key_by_hash(
        self,
        chunk_hash: str,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
    ):
        assert self.metadata is not None
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
        group_metadata = self.metadata[kv_cache_group_id]
        return PoolKey(
            KeyMetadata(
                model_name=group_metadata.model_name,
                head_or_tp_rank=group_metadata.head_or_tp_rank,
                dcp_rank=group_metadata.dcp_rank,
                pp_rank=group_metadata.pp_rank,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                cache_family=cache_family,
            ),
            chunk_hash,
        )

    def get_block_size(self, kv_cache_group_id: int) -> int:
        if kv_cache_group_id >= len(self.block_size):
            return self.block_size[0]
        return self.block_size[kv_cache_group_id]

    def set_group_buffers(
        self,
        group_kv_caches_base_addr: dict[int, list[int]],
        group_block_len: dict[int, list[int]],
        group_block_stride: dict[int, list[int]] | None = None,
        cache_role: str = "kv",
        group_cache_families: dict[int, str] | None = None,
        group_num_layers: dict[int, int] | None = None,
        group_layer_cache_entry_offsets: dict[int, list[int]] | None = None,
        group_cache_entry_segments: dict[int, list[tuple[int, int]]] | None = None,
    ) -> None:
        if cache_role == "state":
            # Keep the interface for future explicit state groups, but this
            # DSV4 branch stores compressor/indexer states in kv_caches.
            pass
        else:
            num_layers = group_num_layers if group_num_layers is not None else self.group_num_layers[cache_role]
            self.group_layouts = {
                group_id: GroupBufferLayout.from_buffers(
                    addresses,
                    group_block_len[group_id],
                    (group_block_stride or {}).get(group_id),
                    (group_cache_entry_segments or {}).get(group_id),
                    (group_layer_cache_entry_offsets or {}).get(group_id),
                    num_layers.get(group_id, 1),
                )
                for group_id, addresses in group_kv_caches_base_addr.items()
            }
        if group_cache_families is not None:
            self.group_cache_families[cache_role] = group_cache_families.copy()
            self._key_prefix_cache.clear()
        if group_num_layers is not None:
            self.group_num_layers[cache_role] = group_num_layers.copy()

    def prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        group_block_size = self.get_block_size(kv_cache_group_id)
        if block_id is None:
            block_idx = start // group_block_size
            if block_idx >= len(block_ids):
                return [], [], 0
            block_id = block_ids[block_idx]
        if cache_role == "state":
            return [], [], block_id
        addresses, sizes = self.group_layouts[kv_cache_group_id].address_spans(start, end, block_id, group_block_size)
        return addresses, sizes, block_id

    def prepare_value_layer(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        layer_id: int,
        kv_cache_group_id: int = 0,
    ):
        group_block_size = self.get_block_size(kv_cache_group_id)
        block_idx = start // group_block_size
        if block_idx >= len(block_ids):
            return [], [], 0
        block_id = block_ids[block_idx]
        layout = self.group_layouts[kv_cache_group_id]
        if layer_id < 0 or layer_id >= layout.num_layers:
            return [], [], 0
        entry_start, entry_end = layout.layer_range(layer_id)
        addresses, sizes = layout.address_spans(start, end, block_id, group_block_size, entry_start, entry_end)
        return addresses, sizes, block_id

    def _iter_token_chunks(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        block_ids: list[int] | None = None,
        skip_null_blocks: bool = False,
        chunk_filter: Callable[[int], bool] | None = None,
        shard_rank: int | None = None,
        shard_size: int | None = None,
        chunk_mask: Sequence[bool] | None = None,
    ) -> Iterable[tuple[int, int, BlockHash | str, int | None]]:
        # Private circular state has no prefix key, even when its block size
        # happens to be divisible by the request's hashing unit.
        if self.cache_coordinator is not None and kv_cache_group_id not in self.cache_coordinator.cacheable_group_ids:
            return
        if not block_hashes:
            return
        logical_block_size = self.get_block_size(kv_cache_group_id)
        grouped_hashes = get_block_hashes(block_hashes, logical_block_size, self.hash_block_size)
        if not grouped_hashes:
            return
        num_logical_blocks = min(len(grouped_hashes), cdiv(token_len, logical_block_size)) if token_len > 0 else 0
        block_id_offset = max(num_logical_blocks - len(block_ids), 0) if block_ids is not None else 0
        first_chunk = max(0, cdiv(mask_num, logical_block_size), block_id_offset)
        chunk_ids: Iterable[int] = range(first_chunk, num_logical_blocks)
        if chunk_mask is not None:
            # Unlike transfer-range masks, a short lookup/store mask excludes
            # chunks beyond its end. compress stops at the shorter iterator.
            chunk_ids = compress(chunk_ids, chunk_mask[first_chunk:num_logical_blocks])
        candidate_index = 0

        for chunk_id in chunk_ids:
            start_idx = chunk_id * logical_block_size
            end_idx = min(start_idx + logical_block_size, token_len)
            if end_idx <= start_idx:
                continue
            if chunk_filter is not None and not chunk_filter(start_idx):
                continue
            block_id = None
            if block_ids is not None:
                block_idx = start_idx // logical_block_size - block_id_offset
                if block_idx < 0 or block_idx >= len(block_ids):
                    continue
                block_id = block_ids[block_idx]
                if skip_null_blocks and block_id <= 0:
                    continue
            shard_allows = (
                shard_rank is None
                or shard_size is None
                or shard_size <= 1
                or candidate_index % shard_size == shard_rank
            )
            candidate_index += 1
            if not shard_allows:
                continue
            yield start_idx, end_idx, grouped_hashes[chunk_id], block_id

    def process_tokens(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process the tokens and return the corresponding cache engine keys."""
        for start, end, hash_val, _ in self._iter_token_chunks(
            token_len,
            block_hashes,
            mask_num,
            kv_cache_group_id,
        ):
            yield (
                start,
                end,
                self._make_key_by_hash(
                    block_hash_to_str(hash_val),
                    kv_cache_group_id=kv_cache_group_id,
                    cache_role=cache_role,
                    cache_family=cache_family,
                ),
            )

    def process_token_key_strings(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        chunk_filter: Callable[[int], bool] | None = None,
        chunk_mask: Sequence[bool] | None = None,
    ) -> Iterable[tuple[int, int, str, BlockHash | str]]:
        """Yield cache key strings directly without materializing PoolKey objects."""
        prefix = self._get_key_prefix(kv_cache_group_id)
        for start, end, hash_val, _ in self._iter_token_chunks(
            token_len,
            block_hashes,
            mask_num,
            kv_cache_group_id,
            chunk_filter=chunk_filter,
            chunk_mask=chunk_mask,
        ):
            yield start, end, prefix + block_hash_to_str(hash_val), hash_val

    def process_token_key_strings_with_block_ids(
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        chunk_filter: Callable[[int], bool] | None = None,
        shard_rank: int | None = None,
        shard_size: int | None = None,
        chunk_mask: Sequence[bool] | None = None,
    ) -> Iterable[tuple[int, int, str, BlockHash | str, int]]:
        """Yield cache key strings and resolved block ids without PoolKey allocation."""
        prefix = self._get_key_prefix(kv_cache_group_id)
        for start, end, hash_val, block_id in self._iter_token_chunks(
            token_len,
            block_hashes,
            mask_num,
            kv_cache_group_id,
            block_ids=block_ids,
            skip_null_blocks=skip_null_blocks,
            chunk_filter=chunk_filter,
            shard_rank=shard_rank,
            shard_size=shard_size,
            chunk_mask=chunk_mask,
        ):
            assert block_id is not None
            yield start, end, prefix + block_hash_to_str(hash_val), hash_val, block_id

    def decode_adaptor_prefill_pp(self, key, addr, size, kv_cache_group_id: int = 0, cache_role: str = "kv"):
        if self.partitions is None or len(self.partitions) == 1:
            return key, addr, size

        new_key = []
        new_addr = []
        new_size = []

        group_num_layers = self.group_num_layers.get(cache_role, {}).get(kv_cache_group_id, 0)
        for i, (addr_list, size_list) in enumerate(zip(addr, size)):
            caches_per_layer = len(addr_list) // group_num_layers if group_num_layers else 2
            caches_per_layer = max(caches_per_layer, 1)
            start = 0
            for j, part in enumerate(self.partitions):
                end = len(addr_list) if j == len(self.partitions) - 1 else start + part * caches_per_layer
                new_str = key[i].replace(  # type: ignore[attr-defined]
                    "@pp_rank:0", f"@pp_rank:{j}", 1
                )
                new_key.append(new_str)
                new_addr.append(addr_list[start:end])
                new_size.append(size_list[start:end])
                start = end
        return new_key, new_addr, new_size
