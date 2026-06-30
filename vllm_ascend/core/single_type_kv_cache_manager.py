# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_ascend import envs


class ComputedBlockList(list[KVCacheBlock]):
    """KV blocks plus the logical token length they cover."""

    def __init__(
        self,
        blocks: Sequence[KVCacheBlock] = (),
        logical_hit_length: int | None = None,
    ) -> None:
        super().__init__(blocks)
        self.logical_hit_length = logical_hit_length


def _partial_prefix_cache(
    block_pool: BlockPool,
) -> tuple[dict[BlockHashWithGroupId, KVCacheBlock], defaultdict[int, set[BlockHashWithGroupId]]]:
    """Return the (hash -> block, block_id -> hashes) maps used to track DSv4
    partial-prefix cache entries for ``block_pool``.

    The maps are stored on the ``BlockPool`` instance (created lazily on first
    use) rather than as module-level globals so the state is scoped to a single
    engine's cache and cannot leak across engines in the same process.
    """
    hash_to_block = getattr(block_pool, "_dsv4_partial_hash_to_block", None)
    if hash_to_block is None:
        hash_to_block = {}
        block_pool._dsv4_partial_hash_to_block = hash_to_block
        block_pool._dsv4_partial_block_id_to_hashes = defaultdict(set)
    return hash_to_block, block_pool._dsv4_partial_block_id_to_hashes


def _hash_range(
    block_hashes: BlockHashList,
    hash_block_size: int,
    start_token: int,
    end_token: int,
) -> BlockHash | None:
    if end_token <= start_token:
        return None
    if start_token % hash_block_size != 0 or end_token % hash_block_size != 0:
        return None
    start = start_token // hash_block_size
    end = end_token // hash_block_size
    if end > len(block_hashes):
        return None
    return BlockHash(b"".join(block_hashes[start:end]))


def _insert_partial_cache(
    block_pool: BlockPool,
    block_hash: BlockHash,
    kv_cache_group_id: int,
    block: KVCacheBlock,
) -> None:
    hash_to_block, block_id_to_hashes = _partial_prefix_cache(block_pool)
    key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
    old_block = hash_to_block.get(key)
    if old_block is not None and old_block.block_id != block.block_id:
        block_id_to_hashes[old_block.block_id].discard(key)
    hash_to_block[key] = block
    block_id_to_hashes[block.block_id].add(key)


def get_partial_cached_block(
    block_pool: BlockPool, block_hash: BlockHash, kv_cache_group_id: int
) -> KVCacheBlock | None:
    hash_to_block, _ = _partial_prefix_cache(block_pool)
    return hash_to_block.get(make_block_hash_with_group_id(block_hash, kv_cache_group_id))


def remove_partial_cache_entries_for_block(block_pool: BlockPool, block_id: int) -> None:
    hash_to_block, block_id_to_hashes = _partial_prefix_cache(block_pool)
    for key in block_id_to_hashes.pop(block_id, set()):
        block = hash_to_block.get(key)
        if block is not None and block.block_id == block_id:
            hash_to_block.pop(key, None)


class CompressAttentionManager(FullAttentionManager):
    def __init__(self, kv_cache_spec: MLAAttentionSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block
        self.copy_block_ids: list[tuple[int, int, int]] = []
        self._copy_src_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        # Per-request bookkeeping for partial-prefix boundary registration.
        # The first compressed block's boundaries are immutable once that block
        # is full and its hashes are available, so each request is registered at
        # most once and then recorded in ``_partial_boundaries_done``. Until then
        # ``_partial_last_compressed`` dedups repeated calls at the same length.
        # This is what previously stalled the scheduler loop and collapsed
        # long-context decode throughput: the boundary hashing re-ran on every
        # decode step even though the first block never changed.
        self._partial_last_compressed: dict[str, int] = {}
        self._partial_boundaries_done: set[str] = set()

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        # assert isinstance(self.kv_cache_spec, (CompressAttentionSpec, C4IndexerSpec))

        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio
        total_computed_tokens //= self.compress_ratio

        num_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        # Partial compressed hits are copied into a private destination block
        # before the request resumes. The source block may also need to be
        # pinned if it is an eviction candidate, which super() already counts.
        num_full_hit_blocks = total_computed_tokens // self.block_size
        return num_blocks + max(0, len(new_computed_blocks) - num_full_hit_blocks)

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.
        3. (Optional) For KV connectors, allocate new blocks for external computed
            tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # It should not have any new computed blocks.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_logical_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_total_computed_tokens = num_total_logical_tokens // self.compress_ratio
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        num_full_hit_blocks = num_total_computed_tokens // self.block_size
        if len(new_computed_blocks) > num_full_hit_blocks:
            full_blocks = list(new_computed_blocks[:num_full_hit_blocks])
            partial_src_blocks = list(new_computed_blocks[num_full_hit_blocks:])
            if self.enable_caching:
                self.block_pool.touch(partial_src_blocks)
                self._copy_src_blocks[request_id].extend(partial_src_blocks)
            partial_dst_blocks = self.block_pool.get_new_blocks(len(partial_src_blocks))
            self.copy_block_ids.extend(
                (self.kv_cache_group_id, src.block_id, dst.block_id)
                for src, dst in zip(partial_src_blocks, partial_dst_blocks)
            )
            new_computed_blocks = [*full_blocks, *partial_dst_blocks]

        # Touch the computed full blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks[:num_full_hit_blocks])
        else:
            assert not any(new_computed_blocks), "Computed blocks should be empty when prefix caching is disabled"

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = min(len(req_blocks), num_full_hit_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        num_tokens //= self.compress_ratio
        ## TODO: check spec decode
        num_tokens_main_model //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        alignment_tokens: int | None = None,
    ) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
            alignment_tokens: The cache-hit alignment used by upstream vLLM
                main. v0.21.0 does not expose this argument in the base class.
        """
        compressed_tokens = num_tokens // self.compress_ratio
        # 0.20.2-style short-key convention: the key of the i-th compressed
        # block is request.block_hashes[i] (the raw 128-token chained hash).
        # When block_size == hash_block_size the base class uses the raw
        # chained hashes directly; otherwise both the base class and the
        # lookup side in find_longest_cache_hit compose them with the same
        # BlockHashListWithBlockSize(target=self.block_size), so the two
        # sides always match. A key only attests the first
        # (i+1)*hash_block_size tokens; the tail block claimed beyond
        # max_length is backed by the private copy made in
        # allocate_new_computed_blocks.
        super().cache_blocks(request, compressed_tokens)
        self._cache_partial_block_boundaries(request, compressed_tokens)

    def _cache_partial_block_boundaries(self, request: Request, compressed_tokens: int) -> None:
        req_blocks = self.req_to_blocks[request.request_id]
        if compressed_tokens <= 0 or not req_blocks:
            return
        # A request resumed from a partial-prefix hit owns private destination
        # blocks copied from an existing cached source block. Do not publish
        # those private blocks as new partial-cache sources, otherwise later
        # requests can form chained copies (dst -> next dst) and observe stale
        # KV if the copy path is batched/vectorized.
        if request.request_id in self._copy_src_blocks:
            return
        # The first block's boundaries are final once it is full; never revisit.
        if request.request_id in self._partial_boundaries_done:
            return
        # Dedup repeated calls at the same compressed length (identical entries).
        if self._partial_last_compressed.get(request.request_id) == compressed_tokens:
            return
        self._partial_last_compressed[request.request_id] = compressed_tokens

        hash_block_size = self.block_pool.hash_block_size
        logical_block_size = self.block_size * self.compress_ratio
        max_logical_tokens = compressed_tokens * self.compress_ratio
        num_blocks_with_tokens = min(cdiv(compressed_tokens, self.block_size), len(req_blocks))

        # Partial compressed hits are only needed before the first full
        # compressed KV block exists (the <16K DSv4 case). Once a prompt has a
        # full compressed-block hit, keep the normal full-block reuse path to
        # avoid adding private copy work to long-context decode.
        for block_idx in range(min(num_blocks_with_tokens, 1)):
            block = req_blocks[block_idx]
            if block.is_null:
                continue
            block_start = block_idx * logical_block_size
            block_end = min(block_start + logical_block_size, max_logical_tokens)
            boundary = block_start + hash_block_size
            while boundary <= block_end:
                # Full-block boundaries are already represented in the normal
                # prefix cache hash table.
                if boundary - block_start != logical_block_size:
                    block_hash = _hash_range(
                        request.block_hashes,
                        hash_block_size,
                        block_start,
                        boundary,
                    )
                    if block_hash is not None:
                        _insert_partial_cache(self.block_pool, block_hash, self.kv_cache_group_id, block)
                boundary += hash_block_size

        # Once the first block is full and all of its hashes are available, its
        # partial boundaries are final. Latch the request so subsequent calls
        # (every long-context decode step) return immediately above, paying zero
        # boundary-hashing cost.
        if compressed_tokens >= self.block_size and len(request.block_hashes) * hash_block_size >= logical_block_size:
            self._partial_boundaries_done.add(request.request_id)
            self._partial_last_compressed.pop(request.request_id, None)

    def take_copy_block_ids(self) -> list[tuple[int, int, int]]:
        copy_block_ids = self.copy_block_ids
        self.copy_block_ids = []
        return copy_block_ids

    def free(self, request_id: str) -> None:
        pinned_src_blocks = self._copy_src_blocks.pop(request_id, [])
        self._partial_last_compressed.pop(request_id, None)
        self._partial_boundaries_done.discard(request_id)
        super().free(request_id)
        if pinned_src_blocks:
            self.block_pool.free_blocks(reversed(pinned_src_blocks))

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        # assert isinstance(
        #     kv_cache_spec, Compress4AttentionSpec | Compress128AttentionSpec | C4IndexerSpec
        # ), (
        #     "CompressAttentionManager can only be used for compressor attention groups"
        # )
        computed_blocks: tuple[ComputedBlockList, ...] = tuple(
            ComputedBlockList() for _ in range(len(kv_cache_group_ids))
        )
        raw_alignment_tokens = alignment_tokens
        block_size = kv_cache_spec.block_size
        hash_block_size = block_pool.hash_block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        full_block_hashes = (
            block_hashes
            if block_size == hash_block_size
            else BlockHashListWithBlockSize(block_hashes, hash_block_size, block_size)
        )
        for block_hash in itertools.islice(full_block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()

        # Keep the original compressed full-block path for prompts that can hit
        # normally. Only fall back to a private partial copy when the prompt is
        # shorter than one compressed cache block and would otherwise get 0 hit.
        logical_block_size = kv_cache_spec.block_size * kv_cache_spec.compress_ratio
        # Gate the partial copy on the matched hit length: for short prompts the
        # fixed cost of copying cached KV into a private block plus boundary
        # hashing can exceed the prefill it saves. 0 disables the gate.
        min_partial_hit_tokens = envs.VLLM_ASCEND_DSV4_PARTIAL_MIN_HIT_TOKENS
        if not computed_blocks[0]:
            candidate = min(
                max_length // raw_alignment_tokens * raw_alignment_tokens,
                logical_block_size - raw_alignment_tokens,
            )
            while candidate > 0:
                if min_partial_hit_tokens and candidate < min_partial_hit_tokens:
                    # The best achievable partial hit is below the break-even
                    # threshold; recompute instead of paying the copy.
                    break
                partial_hash = _hash_range(block_hashes, hash_block_size, 0, candidate)
                if partial_hash is None:
                    candidate -= raw_alignment_tokens
                    continue
                partial_blocks = [
                    block
                    for group_id in kv_cache_group_ids
                    if (block := get_partial_cached_block(block_pool, partial_hash, group_id)) is not None
                ]
                if len(partial_blocks) != len(kv_cache_group_ids):
                    candidate -= raw_alignment_tokens
                    continue
                for computed, cached in zip(computed_blocks, partial_blocks):
                    computed.append(cached)
                    computed.logical_hit_length = candidate
                break

        # NOTE: Div the compress ratio when finding the longest cache hit token length.
        alignment_tokens = cdiv(alignment_tokens, kv_cache_spec.compress_ratio)
        while (
            logical_block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * logical_block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """Build the per-spec KV cache manager.

    For DSv4 / DSA path (``MLAAttentionSpec`` with ``compress_ratio>1``), align
    the runtime admission gate with the startup pool-sizing bound the same way
    vLLM PR #40946 does for ``SlidingWindowSpec`` / ``ChunkedLocalAttentionSpec``.
    Without this cap, an admitted request can demand more blocks than the pool
    was sized to back, and ``allocate_slots`` silently returns ``None`` from
    the ``full_sequence_must_fit`` branch, leaving long-input requests stuck
    in the waiting queue (see vLLM issue #40863, observed on DSv4 + MTP with
    cc>=1 and prompt>=32K).

    The compressed-MLA peak per request is bounded by
    ``cdiv(max_model_len // compress_ratio, block_size)`` (it does not shrink
    via recycling like SWA, but neither does it ever exceed this). Capping at
    this value matches the pool sizer and makes admission consistent with the
    block budget actually held.
    """
    manager_class = spec_manager_map[type(kv_cache_spec)]
    if isinstance(kv_cache_spec, MLAAttentionSpec) and kv_cache_spec.compress_ratio > 1:
        manager_class = CompressAttentionManager
        if max_model_len is not None:
            # Compressed-MLA peak in blocks: ceil(max_model_len/compress/block).
            compress_ratio = kv_cache_spec.compress_ratio
            block_size = kv_cache_spec.block_size
            max_compressed_tokens = max_model_len // compress_ratio
            kwargs["max_admission_blocks_per_request"] = cdiv(max_compressed_tokens, block_size) + 1
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        # Replicate the upstream PR #40946 cap setting for recycling specs.
        # We override the vLLM factory above, so the upstream block that does
        # this lives in dead code (never reached); without re-applying it here
        # SlidingWindowMLASpec / ChunkedLocalAttentionSpec groups have no cap
        # and ``full_sequence_must_fit`` admission reserves the full
        # ``max_model_len`` worth of blocks per request, exhausting the pool
        # at cc>=2 on DSv4 (see vLLM issue #40863).
        if max_num_batched_tokens is not None and max_model_len is not None:
            kwargs["max_admission_blocks_per_request"] = kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
