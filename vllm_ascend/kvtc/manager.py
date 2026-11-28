"""Scheduler for separate raw-sink and compressed KVTC CPU pools.

Shares byte accounting and calibration schemas with the worker. Blocks become
CPU cache hits only after every TP worker has completed its store.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import logger
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.manager import TransferMeta as BaseTransferMeta
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

from vllm_ascend.kvtc.quant_layout import (
    PoolGeometry,
    checkpoint_layouts,
)
from vllm_ascend.utils import vllm_version_is

from .kvtc_config import KVTCConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, torch.Tensor)):
        raise ValueError(f"{name} must be a host integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@dataclass
class KVTCOffloadMetadata(SimpleCPUOffloadMetadata):
    store_block_token_starts: list[int] = field(default_factory=list)
    load_block_token_starts: list[int] = field(default_factory=list)
    kvtc_layout_fingerprint: str = ""


@dataclass
class TransferMeta(BaseTransferMeta):
    block_token_starts: list[int] = field(default_factory=list)


@dataclass
class LoadRequestState:
    request: Request
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False


@dataclass
class StoreRequestState:
    request: Request
    block_ids: tuple[list[int], ...]
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False
    sink_cursor: int = 0
    window_cursor: int = 0
    purge_done: bool = False


@dataclass
class _PendingHit:
    blocks: tuple[list[KVCacheBlock], ...]
    hit_tokens: int
    gpu_prefix_tokens: int


class KVTCBlockHashMap(BlockHashToBlockMap):
    """Allow raw and compressed copies of a prefix, preferring raw on loads."""

    def __init__(self, blocks: list[KVCacheBlock], compressed_boundary: int):
        if vllm_version_is("0.33.0"):
            super().__init__()
        else:
            super().__init__(blocks)
        self.compressed_boundary = compressed_boundary

    def get_representation(self, key, *, raw: bool):
        blocks = self._cache.get(key)
        if blocks is None:
            return None
        # vLLM main indexes the block table by ID, including duplicate hashes.
        if isinstance(blocks, int):
            block_ids = self._dup_ids.get(key, (blocks,))
            return next(
                (self._blocks[bid] for bid in block_ids if (bid >= self.compressed_boundary) == raw),
                None,
            )
        candidates = blocks.values() if isinstance(blocks, dict) else (blocks,)  # vLLM 0.33
        return next((block for block in candidates if (block.block_id >= self.compressed_boundary) == raw), None)

    def get_one_block(self, key):
        return self.get_representation(key, raw=True) or super().get_one_block(key)


class KVTCDualBlockPool(BlockPool):
    """Separate eviction queues sharing vLLM's hash index and block objects."""

    def __init__(
        self,
        num_gpu_blocks,
        enable_caching,
        hash_block_size,
        max_comp_blocks,
        enable_kv_cache_events=False,
        metrics_collector=None,
    ):
        if not 0 <= max_comp_blocks <= num_gpu_blocks:
            raise ValueError("Compressed boundary is outside the CPU pool")
        super().__init__(
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
            medium=MEDIUM_CPU,
        )
        self.shift_threshold = max_comp_blocks
        self.cached_block_hash_to_block = KVTCBlockHashMap(self.blocks, max_comp_blocks)
        free = self.free_block_queue.popleft_n(self.free_block_queue.num_free_blocks)
        self.free_comp_queue = FreeKVCacheBlockQueue([block for block in free if block.block_id < max_comp_blocks])
        self.free_sink_queue = FreeKVCacheBlockQueue([block for block in free if block.block_id >= max_comp_blocks])

    def _queue_for(self, block):
        return self.free_comp_queue if block.block_id < self.shift_threshold else self.free_sink_queue

    def _check_block(self, block):
        bid = block.block_id
        if not 0 <= bid < len(self.blocks) or self.blocks[bid] is not block:
            raise RuntimeError("Block object belongs to a different pool")

    def get_new_blocks(self, num_blocks: int):
        if num_blocks == 0:
            return []
        raise RuntimeError("KVTC must allocate an explicit compressed or raw slot")

    def get_num_free_blocks(self) -> int:
        return self.free_comp_queue.num_free_blocks + self.free_sink_queue.num_free_blocks

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        for block in blocks:
            self._check_block(block)
            if block.is_null:
                continue
            if block.ref_cnt < 0:
                raise RuntimeError("Negative CPU block reference count")
            if block.ref_cnt == 0:
                self._queue_for(block).remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        blocks = list(ordered_blocks)
        counts = Counter()
        for block in blocks:
            self._check_block(block)
            if not block.is_null:
                counts[block.block_id] += 1
        # Validate the whole batch before changing refs or queue membership.
        for bid, count in counts.items():
            if count > self.blocks[bid].ref_cnt:
                raise RuntimeError(f"CPU block {bid}: unbalanced release of {count} refs")
        comp, sink = [], []
        for block in blocks:
            if block.is_null:
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                (comp if block.block_id < self.shift_threshold else sink).append(block)
        if comp:
            self.free_comp_queue.append_n(comp)
        if sink:
            self.free_sink_queue.append_n(sink)

    def _allocate(self, queue):
        if queue.num_free_blocks == 0:
            return None
        block = queue.popleft()
        if block.ref_cnt != 0 or block.is_null:
            raise RuntimeError("Referenced/null CPU block appeared in an eviction queue")
        self._maybe_evict_cached_block(block)
        block.ref_cnt = 1
        if self.metrics_collector:
            self.metrics_collector.on_block_allocated(block)
        return block

    def allocate_comp_block(self):
        return self._allocate(self.free_comp_queue)

    def allocate_sink_block(self):
        return self._allocate(self.free_sink_queue)

    def reset_prefix_cache(self) -> bool:
        if not super().reset_prefix_cache():
            return False
        self.cached_block_hash_to_block = KVTCBlockHashMap(self.blocks, self.shift_threshold)
        return True


class KVTCOffloadScheduler(SimpleCPUOffloadScheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
    ):
        if lazy_offload:
            raise ValueError("KVTC supports eager offload; lazy eviction lacks RoPE token positions")
        if kv_cache_config is None:
            raise ValueError("KVTC requires KVCacheConfig")
        block_size = _integer(scheduler_block_size, "scheduler_block_size", minimum=1)
        hash_size = _integer(hash_block_size, "hash_block_size", minimum=1)
        capacity = _integer(cpu_capacity_bytes, "cpu_capacity_bytes", minimum=1)
        if block_size % hash_size:
            raise ValueError("scheduler_block_size must be divisible by hash_block_size")
        groups = kv_cache_config.kv_cache_groups
        if (
            len(groups) != 1
            or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec)
            or groups[0].kv_cache_spec.block_size != block_size
        ):
            raise ValueError("The KVTC worker requires one uniform full-attention block group")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("The KVTC manager/worker layout requires BF16")
        parallel = vllm_config.parallel_config
        if (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or getattr(parallel, "pipeline_parallel_size", 1) != 1
        ):
            raise ValueError("KVTC supports TP with PP=DCP=PCP=1")
        transfer = vllm_config.kv_transfer_config
        extra = (transfer.kv_connector_extra_config or {}) if transfer else {}
        config = KVTCConfig.from_extra(extra)
        self.kcr = config.kcr
        self.vcr = config.vcr
        self.uncompressed_ratio = config.uncompressed_ratio
        self.sink_blocks = config.sink_blocks
        self.window_blocks = config.window_blocks
        self.svd_path = config.svd_path
        self.use_kvtc = config.compression_enabled
        self.quant_enabled = self.use_kvtc and config.kvtc_quantization
        _integer(kv_cache_config.num_blocks, "GPU blocks", minimum=1)
        # Descriptors can refer to the same backing allocation. Logical layer
        # pages determine the bytes copied by this uniform full-attention worker.
        raw = groups[0].kv_cache_spec.page_size_bytes * len(groups[0].layer_names)
        if raw % (4 * block_size):
            raise ValueError("Expected uniform BF16 K/V")
        layouts = None
        if self.quant_enabled:
            params = torch.load(self.svd_path, map_location="cpu", weights_only=True)
            layouts = checkpoint_layouts(
                params,
                tp_size=parallel.tensor_parallel_size,
                k=raw // (4 * block_size),
                block_size=block_size,
                kcr=self.kcr,
                vcr=self.vcr,
            )
            del params
        self._layout = PoolGeometry.calculate(
            capacity=capacity,
            raw_block_bytes=raw,
            block_size=block_size,
            kcr=self.kcr,
            vcr=self.vcr,
            raw_ratio=self.uncompressed_ratio,
            quantized=self.quant_enabled,
            layouts=layouts,
        )
        self.max_comp_blocks = self._layout.max_comp_blocks
        self.max_sink_blocks = self._layout.max_sink_blocks
        self.shift_threshold = self.max_comp_blocks
        self._compress_blocks = self.use_kvtc and self.max_comp_blocks > 0
        if self._compress_blocks and (self.sink_blocks or self.window_blocks) and not self.max_sink_blocks:
            raise ValueError("sink_blocks or window_blocks > 0 requires a nonempty raw CPU pool")

        # Reuse the base initialization and completion/cancellation
        # bookkeeping instead of duplicating all of its internal state.
        super().__init__(
            vllm_config,
            kv_cache_config,
            capacity,
            block_size,
            hash_size,
            lazy_offload=False,
        )
        old_pool = self.cpu_coordinator.block_pool
        self.cpu_block_pool = KVTCDualBlockPool(
            num_gpu_blocks=self.num_cpu_blocks,
            enable_caching=True,
            hash_block_size=hash_size,
            max_comp_blocks=self.max_comp_blocks,
            enable_kv_cache_events=self.enable_kv_cache_events,
            metrics_collector=old_pool.metrics_collector,
        )
        self.cpu_coordinator.block_pool = self.cpu_block_pool
        # The coordinator also constructs per-group managers holding the pool.
        # Rebind them at initialization, while they have no request state.
        for manager in self.cpu_coordinator.single_type_managers:
            if manager.block_pool is not old_pool:
                raise RuntimeError("Unexpected coordinator pool ownership")
            manager.block_pool = self.cpu_block_pool
            if getattr(manager, "_null_block", None) is old_pool.null_block:
                manager._null_block = self.cpu_block_pool.null_block
        self._in_flight_store_hashes: set = set()
        logger.info(
            "[KVTC MANAGER] K=%d compressed quota=%d bytes/block, CPU compressed=%d raw=%d "
            "shift=%d; usable compressed=%d raw=%d (one null slot reserved)",
            self._layout.k,
            self._layout.compressed_block_bytes,
            self.max_comp_blocks,
            self.max_sink_blocks,
            self.shift_threshold,
            self.cpu_block_pool.free_comp_queue.num_free_blocks,
            self.cpu_block_pool.free_sink_queue.num_free_blocks,
        )

    def _derive_cpu_config(
        self,
        gpu_config: KVCacheConfig,
        cpu_capacity_bytes: int,
        num_cpu_blocks: int | None = None,
        aligned_num_cpu_blocks: int | None = None,
    ) -> KVCacheConfig:
        """Schedule CPU slots using groups; the worker owns mixed-size buffers.

        Raw and compressed slots have different byte strides, so they cannot
        share a KVCacheTensor placement descriptor. The CPU coordinator only
        needs the block count and group specs. PoolGeometry budgets the buffers.
        """
        block_count = self._layout.num_cpu_blocks
        if any(count is not None and count != block_count for count in (num_cpu_blocks, aligned_num_cpu_blocks)):
            raise ValueError("CPU block count disagrees with KVTC pool geometry")
        return replace(gpu_config, num_blocks=block_count, kv_cache_tensors=[])

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        if len(gpu_block_pool.blocks) != self.kv_cache_config.num_blocks:
            raise ValueError("GPU block pool size differs from KVCacheConfig")
        super().bind_gpu_block_pool(gpu_block_pool)

    def _free_pending_cpu_hit(self, pending: _PendingHit) -> None:
        self.cpu_block_pool.free_blocks(block for group in pending.blocks for block in group if not block.is_null)

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        if getattr(request, "skip_reading_prefix_cache", False):
            return 0, False

        # KVTC transfers complete cache blocks.
        if num_computed_tokens % self.block_size:
            return 0, False

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False

        # Recompute at least the last token.
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False

        cpu_hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(remaining_hashes, max_hit_len)
        # Newer coordinators may include a partial tail in a prefix hit.
        # A KVTC transfer and its RoPE position always cover whole blocks.
        hit_blocks = hit_length // self.block_size
        cpu_hit_blocks = (cpu_hit_blocks[0][:hit_blocks],)
        hit_length = hit_blocks * self.block_size

        if self._compress_blocks and self.window_blocks:
            # Protection is relative to the requested prefix, not the transfer
            # start. If only a compressed version remains, recompute that suffix.
            window_start = max(self.sink_blocks, request.num_tokens // self.block_size - self.window_blocks)
            first_block = num_computed_tokens // self.block_size
            for index, block in enumerate(cpu_hit_blocks[0]):
                if first_block + index >= window_start and block.block_id < self.shift_threshold:
                    cpu_hit_blocks = (cpu_hit_blocks[0][:index],)
                    hit_length = min(hit_length, index * self.block_size)
                    break

        if hit_length > 0:
            pin_blocks = [blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = _PendingHit(
                blocks=cpu_hit_blocks,
                hit_tokens=hit_length,
                gpu_prefix_tokens=num_computed_tokens,
            )
            return hit_length, True

        return 0, False

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        req_id = request.request_id
        pending = self._pending_cpu_hits.pop(req_id, None)
        try:
            groups = blocks.get_block_ids()
            if len(groups) != 1:
                raise ValueError("KVTC only accepts one group of GPU block IDs")
            if req_id not in self._reqs_to_store:
                self._reqs_to_store[req_id] = StoreRequestState(
                    request=request,
                    block_ids=([],),
                    num_stored_blocks=[self.sink_blocks if self._compress_blocks else 0],
                )
            external = _integer(num_external_tokens, "num_external_tokens")
            if external == 0:
                return
            if pending is None:
                raise RuntimeError("External tokens were accepted without a pinned CPU cache hit")
            if external % self.block_size or external > pending.hit_tokens:
                raise ValueError("External tokens exceed the hit or are not block-aligned")
            if req_id in self._reqs_to_load:
                raise RuntimeError(f"Request {req_id} already has a pending load")
            if self._gpu_block_pool is None:
                raise RuntimeError("GPU block pool must be bound before loading")
            count = external // self.block_size
            start = pending.gpu_prefix_tokens // self.block_size
            gpu_ids = list(groups[0][start : start + count])
            if len(gpu_ids) != count or len(set(gpu_ids)) != count:
                raise ValueError("GPU destination coverage is incomplete or overlaps")
            gpu_blocks = []
            for bid in gpu_ids:
                bid = _integer(bid, "GPU destination ID")
                if bid >= len(self._gpu_block_pool.blocks):
                    raise ValueError("GPU destination ID is out of range")
                block = self._gpu_block_pool.blocks[bid]
                if block.is_null:
                    raise ValueError("Cannot restore into the GPU null block")
                gpu_blocks.append(block)
            cpu_blocks = pending.blocks[0][:count]
            self.cpu_block_pool.touch(cpu_blocks)
            try:
                self._gpu_block_pool.touch(gpu_blocks)
            except Exception:
                self.cpu_block_pool.free_blocks(cpu_blocks)
                raise
            starts = [pending.gpu_prefix_tokens + i * self.block_size for i in range(count)]
            self._reqs_to_load[req_id] = LoadRequestState(
                request, TransferMeta(gpu_ids, [b.block_id for b in cpu_blocks], block_token_starts=starts)
            )
        finally:
            if pending is not None:
                self._free_pending_cpu_hit(pending)

    def force_purge_all_free_gpu_blocks(self) -> int:
        """Evict only free GPU entries; preserve live hashes and reverse indexes."""
        pool = self._gpu_block_pool
        if pool is None:
            return 0
        purged = 0
        for block in pool.blocks:
            if not block.is_null and block.ref_cnt == 0 and block.block_hash is not None:
                if pool._maybe_evict_cached_block(block):
                    purged += 1
        logger.debug("[KVTC MANAGER] Purged %d free GPU cache entries", purged)
        return purged

    def _store_range(self, state, begin, end, *, raw, loading, gpu_out, cpu_out, starts_out):
        """Advance one pool's cursor independently of the other pool."""
        pool = self.cpu_block_pool
        gpu_pool = self._gpu_block_pool
        index = begin
        retry_index = None
        while index < end:
            bid = state.block_ids[0][index]
            block = gpu_pool.blocks[bid]
            bhash = block.block_hash
            if block.is_null or bhash is None:
                # Full attention may not have published its hash yet. Unlike
                # SWA gaps, this position must be retried rather than lost.
                break
            cache_index = pool.cached_block_hash_to_block
            cached = (
                cache_index.get_representation(bhash, raw=raw)
                if self.window_blocks and self._compress_blocks
                else cache_index.get_one_block(bhash)
            )
            if cached is not None:
                index += 1
                continue
            if bid in self._in_flight_store_gpu_blocks or bhash in self._in_flight_store_hashes:
                if self.window_blocks and self._compress_blocks:
                    # Retain the earliest unfinished conversion, but submit
                    # independent later blocks before this request can finish.
                    if retry_index is None:
                        retry_index = index
                    index += 1
                    continue
                index += 1
                continue
            if bid in loading:
                break  # Its restore can still be running on another stream.
            cpu_block = pool.allocate_sink_block() if raw else pool.allocate_comp_block()
            if cpu_block is None:
                break
            gpu_pool.touch([block])
            self._in_flight_store_gpu_blocks.add(bid)
            self._in_flight_store_hashes.add(bhash)
            gpu_out.append(bid)
            cpu_out.append(cpu_block.block_id)
            starts_out.append(index * self.block_size)
            index += 1
        return index if retry_index is None else retry_index

    def _store_request_blocks(self, state, loading, gpu_ids, cpu_ids, starts):
        confirmed = max(
            0,
            state.request.num_computed_tokens - state.request.num_output_placeholders,
        )
        ready = min(confirmed // self.block_size, len(state.block_ids[0]))
        raw_end = min(self.sink_blocks, ready) if self._compress_blocks else ready
        state.sink_cursor = self._store_range(
            state,
            state.sink_cursor,
            raw_end,
            raw=True,
            loading=loading,
            gpu_out=gpu_ids,
            cpu_out=cpu_ids,
            starts_out=starts,
        )
        if self._compress_blocks:
            compressed_end = max(self.sink_blocks, ready - self.window_blocks)
            if self.window_blocks:
                state.window_cursor = self._store_range(
                    state,
                    max(state.window_cursor, compressed_end),
                    ready,
                    raw=True,
                    loading=loading,
                    gpu_out=gpu_ids,
                    cpu_out=cpu_ids,
                    starts_out=starts,
                )
            state.num_stored_blocks[0] = self._store_range(
                state,
                state.num_stored_blocks[0],
                compressed_end,
                raw=False,
                loading=loading,
                gpu_out=gpu_ids,
                cpu_out=cpu_ids,
                starts_out=starts,
            )

    def _prepare_eager_store_specs(self, scheduler_output: SchedulerOutput):
        gpu_ids, cpu_ids, starts, req_ids = [], [], [], []
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None:
            return gpu_ids, cpu_ids, starts, req_ids
        loading = {bid for state in self._reqs_to_load.values() for bid in state.transfer_meta.gpu_block_ids}
        for req_id, new_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue
            if preempted:
                state.block_ids = ([],)
                state.sink_cursor = 0
                state.window_cursor = 0
                state.num_stored_blocks = [self.sink_blocks if self._compress_blocks else 0]
            if new_groups:
                if len(new_groups) != 1:
                    raise ValueError("KVTC scheduler received multiple KV groups")
                if new_groups[0] is not None:
                    new_ids = [_integer(bid, "GPU store ID") for bid in new_groups[0]]
                    if any(bid >= len(gpu_pool.blocks) for bid in new_ids):
                        raise ValueError("GPU store ID is out of range")
                    state.block_ids[0].extend(new_ids)
            params = state.request.kv_transfer_params
            if isinstance(params, dict) and params.get("purge_cache") is True and not state.purge_done:
                self.force_purge_all_free_gpu_blocks()
                state.purge_done = True
            before = len(gpu_ids)
            self._store_request_blocks(state, loading, gpu_ids, cpu_ids, starts)
            if len(gpu_ids) != before:
                req_ids.append(req_id)
        return gpu_ids, cpu_ids, starts, req_ids

    def prepare_store_specs(self, scheduler_output: SchedulerOutput):
        return self._prepare_eager_store_specs(scheduler_output)

    def _queue_finished_eager_store(self, request: Request, block_ids: tuple[list[int], ...]) -> None:
        """Flush confirmed full blocks through the same sink/window policy."""
        state = self._reqs_to_store.get(request.request_id)
        if state is None or self._gpu_block_pool is None:
            return
        state.block_ids = block_ids
        loading = {bid for load in self._reqs_to_load.values() for bid in load.transfer_meta.gpu_block_ids}
        gpu_ids, cpu_ids, starts = [], [], []
        self._store_request_blocks(state, loading, gpu_ids, cpu_ids, starts)
        if gpu_ids:
            self._pending_finished_stores.append(TransferMeta(gpu_ids, cpu_ids, block_token_starts=starts))

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVTCOffloadMetadata:
        store_gpu, store_cpu, store_starts, store_reqs = self.prepare_store_specs(scheduler_output)
        for transfer in self._pending_finished_stores:
            store_gpu.extend(transfer.gpu_block_ids)
            store_cpu.extend(transfer.cpu_block_ids)
            store_starts.extend(transfer.block_token_starts)
        self._pending_finished_stores.clear()
        store_event = -1
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu, block_token_starts=store_starts
            )
            self._store_event_to_reqs[store_event] = list(dict.fromkeys(store_reqs))
            for req_id in self._store_event_to_reqs[store_event]:
                self._reqs_to_store[req_id].store_events.add(store_event)
        load_gpu, load_cpu, load_starts, load_reqs = [], [], [], []
        for req_id, state in self._reqs_to_load.items():
            if state.load_event is None:
                load_gpu.extend(state.transfer_meta.gpu_block_ids)
                load_cpu.extend(state.transfer_meta.cpu_block_ids)
                load_starts.extend(state.transfer_meta.block_token_starts)
                load_reqs.append(req_id)
        load_event = -1
        if load_reqs:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            self._load_event_to_reqs[load_event] = list(load_reqs)
            for req_id in load_reqs:
                self._reqs_to_load[req_id].load_event = load_event
        return KVTCOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=list(load_gpu),
            load_cpu_blocks=list(load_cpu),
            load_event_to_reqs={idx: list(reqs) for idx, reqs in self._load_event_to_reqs.items()},
            store_event=store_event,
            store_gpu_blocks=list(store_gpu),
            store_cpu_blocks=list(store_cpu),
            need_flush=bool(scheduler_output.preempted_req_ids),
            store_block_token_starts=list(store_starts),
            load_block_token_starts=list(load_starts),
            kvtc_layout_fingerprint=self._layout.fingerprint,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        # vLLM aggregates finished_recving across workers before this callback.
        for req_id in connector_output.finished_recving or ():
            state = self._reqs_to_load.get(req_id) or self._abandoned_reqs_to_load.get(req_id)
            if state is not None and state.load_event is not None:
                self._interval_load_blocks_completed += self._cleanup_load_request(req_id)
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event, count in meta.completed_store_events.items():
            event = _integer(event, "store completion event")
            count = _integer(count, "store completion count", minimum=1)
            if event not in self._store_event_to_blocks and event not in self._abandoned_store_event_to_blocks:
                if event < self._store_event_counter:
                    continue  # Completed earlier; never release refs twice.
                raise RuntimeError(f"Completion for unissued KVTC store event {event}")
            total = self._store_event_pending_counts.get(event, 0) + count
            if total > self._expected_worker_count:
                raise RuntimeError("Store completion count exceeds the TP worker count")
            if total == self._expected_worker_count:
                self._store_event_pending_counts.pop(event, None)
                self._process_store_event(event)
            else:
                self._store_event_pending_counts[event] = total

    def _process_store_event(self, event_idx: int) -> None:
        transfer = self._store_event_to_blocks.get(event_idx) or self._abandoned_store_event_to_blocks.get(event_idx)
        if transfer is None:
            return
        hashes = [self._gpu_block_pool.blocks[bid].block_hash for bid in transfer.gpu_block_ids]
        # Base publishes CPU hashes, releases GPU/CPU pins and cleans finished
        # request state. Cancellation does not release an in-flight transfer.
        super()._process_store_event(event_idx)
        self._in_flight_store_hashes.difference_update(hashes)

    def reset(self) -> bool:
        for pending in self._pending_cpu_hits.values():
            self._free_pending_cpu_hit(pending)
        self._pending_cpu_hits.clear()
        # Queued finish-time stores have not reached a worker. Upstream releases
        # their refs immediately; submitted stores stay pinned until completion.
        queued_hashes = {
            self._gpu_block_pool.blocks[bid].block_hash
            for transfer in self._pending_finished_stores
            for bid in transfer.gpu_block_ids
        }
        result = super().reset()
        self._in_flight_store_hashes.difference_update(queued_hashes)
        return result
