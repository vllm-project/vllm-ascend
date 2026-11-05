from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionComputeStartGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    PoolKey,
    ReqMeta,
    normalize_block_ids_by_group,
)


@dataclass(order=True)
class LayerPoolKey(PoolKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@dcp:{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@layer_id:{self.layer_id}"
            f"@{self.chunk_hash}"
        )


def get_partial_block_index(
    token_count: int,
    block_size: int,
    hash_count: int,
    enabled: bool,
) -> int | None:
    """Index of the trailing partial block to transfer, if any.

    Returns None when disabled, when the request carries no tokens, or when
    the token count aligns exactly with completed hash blocks.
    """
    if not enabled or token_count <= 0:
        return None
    full_blocks, remainder = divmod(token_count, block_size)
    if remainder:
        return full_blocks
    if full_blocks > hash_count:
        return full_blocks - 1
    return None


def masked_block_runs(
    mask: Sequence[bool] | None,
    start_block: int,
    end_block: int,
) -> list[tuple[int, int]]:
    """Split [start_block, end_block) into maximal runs of mask-allowed blocks.

    A None mask (or blocks at or beyond the mask length) disables filtering,
    so callers never transfer fewer blocks than the mask actually covers.
    """
    if end_block <= start_block:
        return []
    if mask is None:
        return [(start_block, end_block)]
    mask_len = len(mask)
    if start_block >= mask_len:
        return [(start_block, end_block)]
    scan_end = min(end_block, mask_len)
    # True/False map to b"\x01"/b"\x00"; bytes.find scans at C speed and
    # never raises (unlike list.index), so both dense and holey masks skip
    # the per-block Python loop.
    bits = bytes(mask[start_block:scan_end])
    runs: list[tuple[int, int]] = []
    pos = 0
    while True:
        hole = bits.find(b"\x00", pos)
        if hole == -1:
            runs.append((start_block + pos, end_block))
            return runs
        if hole > pos:
            runs.append((start_block + pos, start_block + hole))
        pos = bits.find(b"\x01", hole + 1)
        if pos == -1:
            # Remaining mask blocks are all disallowed, but blocks at or
            # beyond mask_len stay allowed.
            if end_block > mask_len:
                runs.append((mask_len, end_block))
            return runs


@dataclass
class LayerBatchReqMeta:
    req_ids: list[str]
    layer_id: int
    is_last_chunks: list[bool | None] = field(default_factory=list)
    addr_array: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    size_array: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    gvas_array: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    load_keys: list[str] = field(default_factory=list)


@dataclass
class LayerRangeReqMeta:
    req_ids: list[str]
    layer_id: int
    block_ids: list[int]
    keys: list[str]
    all_buffers: list[list[int]]
    all_sizes: list[list[int]]
    all_offsets: list[list[int]]
    load_keys: list[str] = field(default_factory=list)


@dataclass
class LayerBlockRange:
    request: ReqMeta
    start_block: int
    end_block: int
    partial_block_index: int | None = None


@dataclass
class SharedBlockData:
    """Pre-computed block data shared across all layers for the same request."""

    block_ids_arr: np.ndarray
    block_gvas_arr: np.ndarray | None
    req_ids: list[str]
    is_last_chunks: list[bool | None]
    block_keys: list[str] | None = None
    save_keys: list[str] = field(default_factory=list)
    load_keys: list[str] = field(default_factory=list)


@dataclass
class LayerTransferTask:
    layer_id: int
    block_ranges: list[LayerBlockRange]
    shared_block_data: SharedBlockData | None = None
    group_id: int = 0
    layer_idx_in_group: int = 0
    # Publish newly allocated GVA keys after this task completes the final
    # actual layer copy for the batch.
    write_finish_keys: list[str] = field(default_factory=list)
    # Cache for KVCacheStoreKeyLayerSendingThread:
    # maps block_range index -> list of (start, end, key_all_layers)
    cached_process_tokens: dict[int, list[tuple[int, int, list]]] | None = None
    # Block-key backends use one remote object per block/rank with per-layer ranges.
    use_key_major_ranges: bool = False
    # Group-local completion differs from the physical model layer boundary.
    final_group_layer: bool = False


@dataclass
class LayerLoadTask:
    wait_for_save_layer: int | None
    transfer_tasks: list[LayerTransferTask]
    layer_id: int
    attention_start_gate: AttentionComputeStartGate | None = None


@dataclass(init=False)
class LayerMultiBlockReqMeta:
    req_id: str
    keys: list[LayerPoolKey]
    starts: list[int]
    ends: list[int]
    block_ids_by_group: list[list[int]]
    layer_id: int
    block_hashes: Sequence[Any] = field(default_factory=list)
    is_last_chunk: bool | None = True
    current_event: torch.npu.Event | None = None
    token_ids: list[int] | None = None
    original_block_size: list[int] | int | None = None
    kv_cache_group_id: int = 0

    def __init__(
        self,
        req_id: str,
        keys: list[LayerPoolKey],
        starts: list[int],
        ends: list[int],
        block_ids_by_group: list[list[int]] | None = None,
        layer_id: int = 0,
        is_last_chunk: bool | None = True,
        current_event: torch.npu.Event | None = None,
        block_ids: list[int] | list[list[int]] | None = None,
        token_ids: list[int] | None = None,
        original_block_size: list[int] | int | None = None,
        block_hashes: Sequence[Any] | None = None,
        kv_cache_group_id: int = 0,
    ) -> None:
        self.req_id = req_id
        self.keys = keys
        self.starts = starts
        self.ends = ends
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.layer_id = layer_id
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event
        self.token_ids = token_ids
        self.original_block_size = original_block_size
        self.block_hashes = [] if block_hashes is None else block_hashes
        self.kv_cache_group_id = kv_cache_group_id

    @property
    def block_ids(self) -> list[int]:
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @block_ids.setter
    def block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.block_ids_by_group = normalize_block_ids_by_group(block_ids)
