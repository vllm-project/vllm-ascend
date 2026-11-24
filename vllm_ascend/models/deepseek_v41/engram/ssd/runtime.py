# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EngramSSDRuntime: engine + cache + planes behind one prepare() call.

Single-consumer (DP1) first version, per docs/04 §6/§9.2. Each step:

1. select this rank's hash columns ``[T, L, C_local]`` from ``[T, L, C]``;
2. async D2H into a reused pinned buffer, wait on the copy event;
3. per layer, fold valid (non-dead, in-range) int32 rows into packed
   global ids; both layers share one cache lease (hits pinned first);
4. fill misses through the libaio engine into the shared payload planes,
   publish the lease (directory visibility switches atomically);
5. rebuild per-layer slot matrices in request order, H2D, UVA gather into
   the reused BF16 output buffer (dead slots read row 0, which stays
   zero-initialized);
6. ``finish_execution`` waits on the gather event before releasing the
   lease so a later fill can never overwrite a slot an in-flight gather
   still reads.

The demand path reuses its staging buffers. Prefill lookahead owns
independent buffers until the next batch accepts or discards them.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch

from vllm_ascend.ascend_config import EngramSSDConfig
from vllm_ascend.models.deepseek_v41.engram.npu import (
    gather_dequantize_host_uva,
)
from vllm_ascend.models.deepseek_v41.engram.ssd.cache import (
    CacheLease,
    SSDCacheError,
    make_ssd_row_cache,
)
from vllm_ascend.models.deepseek_v41.engram.ssd.engine import SSDFillEngine
from vllm_ascend.models.deepseek_v41.engram.ssd.manifest import SSDManifest
from vllm_ascend.models.deepseek_v41.engram.ssd.planes import SSDPlanes

DEAD_ID = -1


@dataclass
class PreparedEngram:
    """One step's lookup outputs; backing is owned by the runtime."""

    step_id: int
    layer_outputs: dict[int, torch.Tensor]   # [T, C_local, 256] BF16 (device)
    mask: np.ndarray | None                  # [T] bool, 有效 token
    gather_event: torch.npu.Event | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LayeredStaging:
    hashes: torch.Tensor
    mask: torch.Tensor | None
    host_slots: torch.Tensor
    device_slots: torch.Tensor


@dataclass
class _LayeredLookup:
    tokens: int
    stream: torch.npu.Stream | None
    ready: tuple[Future, Future]
    retired: Future | None
    producer: Future
    step: int
    wait_ms: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefillLookahead:
    key: tuple[str, int, int, int]
    token_history: torch.Tensor
    segments: tuple[slice, ...]
    padded_tokens: int


@dataclass
class _PrefetchedChunk:
    key: tuple[str, int, int, int]
    lookup: _LayeredLookup
    graph_inputs: dict[str, Any] | None
    inputs_ready: torch.npu.Event | None


class EngramSSDRuntime:
    """Model-level facade owning both layers' SSD state for this rank."""

    def __init__(self, cfg: EngramSSDConfig, manifest: SSDManifest, *, device,
                 num_layers: int, total_hash_cols: int, col_start: int,
                 col_count: int, max_tokens: int, tp_size: int = 1,
                 planes=None, engine=None, cache=None, protocol=None):
        if col_start < 0 or col_count <= 0 or col_start + col_count > max(total_hash_cols, 1):
            raise ValueError(f"bad column shard [{col_start},{col_start + col_count}) "
                             f"of {total_hash_cols}")
        self.cfg = cfg
        self.manifest = manifest
        self.device = device
        self.num_layers = num_layers
        self.total_hash_cols = total_hash_cols
        self.col_start = col_start
        self.col_count = col_count
        self.max_tokens = max_tokens
        self.tp_size = tp_size
        self.layer_ids = [s.layer_id for s in manifest.specs]
        self.layer_bases = [s.packed_global_base for s in manifest.specs]
        self.layer_rows = [s.logical_rows for s in manifest.specs]

        self.capacity = cfg.capacity()
        self.layerwise_overlap = getattr(cfg, "layerwise_overlap", False)
        if self.layerwise_overlap and (
            tp_size != 1 or protocol is None or num_layers != 2
            or self.layer_ids[0] >= self.layer_ids[1]
            or self.layer_bases[0] + self.layer_rows[0] > self.layer_bases[1]
            or max_tokens * num_layers * col_count * protocol.size > self.capacity
        ):
            raise ValueError("layerwise Engram SSD requires TP1, two ordered layers, and shared slots "
                             "for a full group batch")
        self.prefill_lookahead = getattr(cfg, "prefill_lookahead", False)
        self._layered_lookup: _LayeredLookup | None = None
        self._prefetched_chunk: _PrefetchedChunk | None = None
        self._layered_executor: ThreadPoolExecutor | None = None
        self.is_leader = engine is not None or cache is not None
        if planes is not None:
            self.planes = planes                    # 共享装配(调用方创建)
        else:
            # aclrtMallocHost 需要设备上下文已初始化(裸进程直接分配返回 107002)
            torch.zeros(1, device=device)
            self.planes = SSDPlanes(self.capacity, device)
        codes_np, scales_np = self.planes.numpy_views()
        if not self.planes.shared:
            # 私有平面初始化为安全值(row0 会被无效 slot 读取)
            codes_np.fill(0)
            scales_np.fill(0)
        self._codes_np, self._scales_np = codes_np, scales_np

        # 装配矩阵: leader(engine+cache 给定) / client(仅 protocol, 双 None) /
        # 单消费者默认(protocol=None 且双 None → 自建)
        is_client = protocol is not None and engine is None and cache is None
        if engine is not None:
            self.engine = engine
        elif is_client:
            self.engine = None                      # client: 无 IO 引擎
        else:
            cpu_base = cfg.io_cpu_list[0] if cfg.io_cpu_list else -1
            self.engine = SSDFillEngine(
                cfg.table_paths, layer_rows=self.layer_rows,
                layer_bases=self.layer_bases,
                threads_per_disk=cfg.threads_per_disk, qd=cfg.queue_depth,
                io_nr=cfg.io_setup_events, cpu_base=cpu_base)
        if cache is not None:
            self.cache = cache
        elif self.engine is not None:
            rebuild = cfg.rebuild_threshold or None
            self.cache = make_ssd_row_cache(
                self.capacity,
                row_limit=max(manifest.total_logical_rows,
                              self.layer_bases[-1] + self.layer_rows[-1]),
                rebuild_threshold=rebuild)
        else:
            self.cache = None                       # client: 无本地目录
        # 预分配: pinned D2H / 设备槽位 + host 槽位 / BF16 输出
        # 显式 cpu: 模型 init 可能处于 torch.device('npu') 上下文
        self._d2h = torch.empty((max_tokens, num_layers, col_count), dtype=torch.int32,
                                pin_memory=True, device="cpu")
        self._mask_d2h = (torch.empty(max_tokens, dtype=torch.bool, pin_memory=True, device="cpu")
                          if self.layerwise_overlap else None)
        self._slot_dev = torch.empty((num_layers, max_tokens, col_count), dtype=torch.int32,
                                     device=device)
        self._slot_host = torch.empty((num_layers, max_tokens, col_count), dtype=torch.int32,
                                      device="cpu", pin_memory=self.layerwise_overlap)
        self._output = torch.empty((num_layers, max_tokens, col_count, 256),
                                   dtype=torch.bfloat16, device=device)
        # 超过该 miss 数走 extent 合并读(decode 逐行阈值以下)
        self.extents_threshold = (
            float("inf") if getattr(self.cache, "forced_read_mode", None) else 64)
        self._lease: CacheLease | None = None
        # DP 协议(可选): epoch>=1 时行→槽位走组协议
        self.protocol = protocol
        self._gather_event = torch.npu.Event()
        self._closed = False
        self._step = 0

    # ---------- 主路径 ----------

    def _cache_observation(self, lease: CacheLease | None) -> dict[str, Any]:
        """A shared cache is observable only on its owner, not each consumer."""
        if lease is not None:
            cache = dict(lease.stats)
            cache['unique_misses'] = len(lease.miss_rows)
            scope = 'local'
        else:
            observed = self.protocol.last_cache_stats if self.protocol is not None else None
            cache = dict(observed) if observed is not None else {}
            scope = 'owner_group' if observed is not None else 'owner_group_unavailable'
        return dict(cache_scope=scope, cache=cache, hits=cache.get('hit_rows'),
                    misses=cache.get('unique_misses'), query_misses=cache.get('miss_rows'))

    def prepare(self, hashes: torch.Tensor, token_mask: torch.Tensor | None,
                *, step_id: int = -1, output_buffers: dict | None = None,
                retire_after_gather: bool = False,
                graph_ready_events: dict | None = None,
                skip_layerwise: bool = False) -> PreparedEngram:
        """hashes: [T, L, C] int32 device; mask: [T] bool 或 None(全有效)。

        output_buffers: {layer_id: [max_tokens, C_local, 256] BF16}(调用方固定
        地址缓冲, FULL replay 依赖); 缺省写 runtime 内部固定缓冲。
        """
        if self._closed:
            raise RuntimeError("EngramSSDRuntime closed")
        if self._lease is not None or self._layered_lookup is not None:
            raise RuntimeError("previous lease not finished (call finish_execution)")
        t0 = time.perf_counter()
        T = int(hashes.shape[0])
        assert hashes.shape[1] == self.num_layers and hashes.shape[2] == self.total_hash_cols
        assert T <= self.max_tokens
        # 1) 选列 + D2H
        local = hashes[:, :, self.col_start:self.col_start + self.col_count]
        local_c = local.contiguous()
        d2h = self._d2h[:T]
        ta = time.perf_counter()
        d2h.copy_(local_c, non_blocking=True)
        tb = time.perf_counter()
        d2h_event = torch.npu.Event()
        if not skip_layerwise and self.layerwise_active:
            stream = torch.npu.current_stream()
            mask_host = None
            if token_mask is not None:
                mask_host = self._mask_d2h[:T]
                mask_host.copy_(token_mask, non_blocking=True)
            d2h_event.record(stream)
            outputs = {
                layer: (output_buffers[layer][:T].view(T, self.col_count, 256)
                        if output_buffers is not None and layer in output_buffers else self._output[li, :T])
                for li, layer in enumerate(self.layer_ids)
            }
            self._layered_lookup = self._queue_layered(
                T, outputs, stream=stream, d2h_event=d2h_event, mask_host=mask_host,
                retire_after_gather=retire_after_gather, graph_ready_events=graph_ready_events,
            )
            return PreparedEngram(step_id, outputs, None)
        d2h_event.record()
        d2h_event.synchronize()
        t1 = time.perf_counter()
        self._d2h_enqueue_ms = (tb - ta) * 1e3
        self._d2h_wait_ms = (t1 - tb) * 1e3
        host = d2h.numpy()

        mask_np = (np.ones(T, bool) if token_mask is None
                   else token_mask.detach().cpu().numpy().astype(bool))

        layer_valid, ids_parts, ids = self._pack_rows(host, mask_np)
        t2 = time.perf_counter()

        if self._requires_shared_windows():
            return self._prepare_shared_windowed(mask_np, host, layer_valid, T,
                                                 step_id=step_id, output_buffers=output_buffers)
        if self._needs_windows(layer_valid):
            return self._prepare_windowed(hashes, mask_np, host, layer_valid, T,
                                          step_id=step_id)

        # DP 模式(epoch>=1): 行→槽位经组协议(leader 侧事务在本进程
        # cache/engine 上完成); 单消费者路径(epoch 0)直接本地事务。
        protocol_slots = None
        if self.protocol is not None and self.protocol.epoch_fn() >= 1:
            protocol_slots = self.protocol.step(ids, num_tokens=int(mask_np.sum()))
        if protocol_slots is None and self.cache is None:
            raise RuntimeError(
                "client runtime without protocol epoch: hashed step before "
                "activation (should not happen; check the epoch-file watcher)")
        lease = self.cache.acquire(ids) if protocol_slots is None else None
        fill_stats: dict[str, Any] = {}
        if lease is not None and len(lease.miss_rows) >= self.extents_threshold:
            # prefill 大批: 排序去重后贪心合并为 4K 对齐 extent(≤16K)
            from vllm_ascend.models.deepseek_v41.engram.ssd.engine import plan_row_extents

            tf0 = time.perf_counter()
            ext, sct, _ = plan_row_extents(
                lease.miss_rows, lease.miss_slots,
                base1=self.layer_bases[1] if self.num_layers > 1 else 0,
                dual=self.engine.dual)
            st = self.engine.read_extents(ext, sct, self._codes_np, self._scales_np)
            fill_stats = dict(st)
            fill_stats["fill_wall_ms"] = (time.perf_counter() - tf0) * 1e3
        elif lease is not None and len(lease.miss_rows):
            tf0 = time.perf_counter()
            self.engine.fill(lease.miss_rows, lease.miss_slots,
                             self._codes_np, self._scales_np)
            fill_stats = self.engine.stats()
            fill_stats["fill_wall_ms"] = (time.perf_counter() - tf0) * 1e3
        if lease is not None:
            self.cache.publish(lease)
            self._lease = lease
        t3 = time.perf_counter()

        # 4) 槽位矩阵(请求顺序还原) + H2D + UVA gather
        offsets = np.cumsum([0] + [len(p) for p in ids_parts])
        request_slots = protocol_slots if protocol_slots is not None else lease.slots
        slot_dev = self._slot_dev[:, :T]
        slot_np = self._slot_host.numpy()[:, :T]
        for li in range(self.num_layers):
            m = slot_np[li]
            m.fill(-1)
            valid = layer_valid[li]
            vp = np.nonzero(valid.ravel())[0]
            if len(vp):
                m.reshape(-1)[vp] = request_slots[offsets[li]:offsets[li + 1]]
        slot_dev.copy_(self._slot_host[:, :T], non_blocking=True)
        t4 = time.perf_counter()

        outputs: dict[int, torch.Tensor] = {}
        for li in range(self.num_layers):
            ids_dev = slot_dev[li].reshape(-1)
            if output_buffers is not None and self.layer_ids[li] in output_buffers:
                out = output_buffers[self.layer_ids[li]][:T]
                out_slice = out
            else:
                out = self._output[li, :T]
                out_slice = out
            gather_dequantize_host_uva(
                self.planes.codes, self.planes.scales, ids_dev,
                head_start=0, local_heads=1, pad_heads=1, output=out_slice.reshape(-1, 256),
                vocab_start=0, vocab_end=self.capacity)
            outputs[self.layer_ids[li]] = out_slice
        self._gather_event.record()
        t5 = time.perf_counter()

        observation = self._cache_observation(lease)
        cache_fill_ms = (observation['cache'].get('fill_wall_ms') if protocol_slots is not None
                         else fill_stats.get('fill_wall_ms', 0.0))

        self._step += 1
        if self._step % 200 == 0:
            proto = ""
            if self.protocol is not None:
                ph = self.protocol.stats.phase_ms
                n = max(self.protocol.stats.steps, 1)
                proto = (f" proto[p0={ph.get('p0_header',0)/n:.2f} p1={ph.get('p1_gather',0)/n:.2f}"
                         f" p3={ph.get('p3_scatter',0)/n:.2f} p4={ph.get('p4_ack',0)/n:.2f}ms]"
                         f" rounds={self.protocol.stats.steps} failed={self.protocol.stats.rounds_failed}")
            logging.getLogger(__name__).info(
                "engram-ssd step=%d T=%d rows=%d cache_scope=%s miss=%s d2h=%.2f(enq=%.2f,wait=%.2f) "
                "plan=%.2f protocol=%.2f fill=%s h2d=%.2f uva=%.2f total=%.2fms%s",
                self._step, T, len(ids),
                observation['cache_scope'], observation['misses'],
                (t1-t0)*1e3, getattr(self, "_d2h_enqueue_ms", 0),
                getattr(self, "_d2h_wait_ms", 0),
                (t2-t1)*1e3, (t3-t2)*1e3 if protocol_slots is not None else 0,
                'n/a' if cache_fill_ms is None else f'{cache_fill_ms:.2f}',
                (t4-t3)*1e3, (t5-t4)*1e3,
                (t5-t0)*1e3, proto)
        stats = dict(
            step=self._step, step_id=step_id, tokens=T,
            rows=len(ids),
            **observation,
            d2h_ms=(t1 - t0) * 1e3, plan_ms=(t2 - t1) * 1e3,
            fill_ms=cache_fill_ms,
            protocol_ms=(t3 - t2) * 1e3 if protocol_slots is not None else 0.0,
            h2d_ms=(t4 - t3) * 1e3,
            uva_ms=(t5 - t4) * 1e3, total_ms=(t5 - t0) * 1e3,
            engine=fill_stats,
            protocol=protocol_slots is not None)
        return PreparedEngram(step_id=step_id, layer_outputs=outputs,
                              mask=mask_np, gather_event=self._gather_event,
                              stats=stats)

    @property
    def layerwise_active(self) -> bool:
        return self.layerwise_overlap and self.protocol.epoch_fn() >= 1

    def _pack_rows(self, host: np.ndarray, mask_np: np.ndarray):
        layer_valid: list[np.ndarray] = []
        ids_parts: list[np.ndarray] = []
        for li in range(self.num_layers):
            h = host[:, li, :]                       # [T, C_local] int32
            valid = mask_np[:, None] & (h != DEAD_ID) & (h >= 0) & (h < self.layer_rows[li])
            rows = (h.astype(np.uint64) + np.uint64(self.layer_bases[li]))[valid]
            ids_parts.append(rows)
            layer_valid.append(valid)
        ids = (np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.uint64))
        return layer_valid, ids_parts, ids

    def _queue_layered(
        self, tokens, outputs, *, stream=None, d2h_event=None, mask_host=None,
        staging=None, retire_after_gather=False, staging_failed=False, graph_ready_events=None,
    ) -> _LayeredLookup:
        if self._closed:
            raise RuntimeError("EngramSSDRuntime closed")
        if self._layered_executor is None:
            self._layered_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-ssd")
        ready = (Future(), Future())
        if graph_ready_events is not None:
            for layer, result in zip(self.layer_ids, ready, strict=True):
                result.add_done_callback(partial(
                    self._release_failed_graph_layer, stream, outputs[layer], graph_ready_events[layer],
                ))
        retired = None if retire_after_gather else Future()
        producer = self._layered_executor.submit(
            self._produce_layers, tokens, outputs, stream, d2h_event, mask_host,
            ready, retired, staging, staging_failed, graph_ready_events,
        )
        self._step += 1
        return _LayeredLookup(tokens, stream, ready, retired, producer, self._step)

    def _release_failed_graph_layer(self, stream, out, event, result) -> None:
        if result.exception() is None:
            return
        # FULL replay may already be waiting here. Finish the device graph
        # with defined data, then surface the failure during step retirement.
        try:
            with torch.inference_mode(), torch.npu.device(self.device), torch.npu.stream(stream):
                out.zero_()
                event.record(stream)
        except Exception:
            logging.getLogger(__name__).exception("Cannot release failed Engram SSD graph wait")
            # An outstanding wait cannot be recovered by resetting the event.
            os._exit(87)

    def _produce_layers(
        self, tokens, outputs, stream, d2h_event, mask_host, ready, retired,
        staging, staging_failed, graph_ready_events,
    ) -> None:
        try:
            rows = np.empty(0, np.uint64)
            first_count = -1 if staging_failed else 0
            num_tokens = 0
            submit_layer = None
            if d2h_event is not None:
                try:
                    with torch.npu.device(self.device):
                        d2h_event.synchronize()
                    mask = np.ones(tokens, bool) if mask_host is None else mask_host.numpy()
                    host = self._d2h if staging is None else staging.hashes
                    valid, parts, rows = self._pack_rows(host[:tokens].numpy(), mask)
                    first_count = len(parts[0])
                    num_tokens = int(mask.sum())
                except Exception:
                    # Peers must reach the same P0 even when local staging fails.
                    logging.getLogger(__name__).exception("Engram SSD failed to stage hash rows")
                    first_count = -1
                else:
                    def submit_layer(index, slots):
                        layer = self.layer_ids[index]
                        graph_event = None if graph_ready_events is None else graph_ready_events[layer]
                        return self._submit_layer(
                            stream, outputs[layer], valid[index], index, slots, staging, graph_event
                        )

            options = {}
            if retired is None:
                options["synchronize_gathers"] = lambda: self._synchronize_gathers(stream, ready)
            self.protocol.step_layered(
                rows, first_count, self.layer_bases[1], ready, retired,
                num_tokens=num_tokens, submit_layer=submit_layer, **options,
            )
        except BaseException as exc:
            for result in ready:
                if not result.done():
                    result.set_exception(exc)
            raise

    def _synchronize_gathers(self, stream, ready) -> bool:
        try:
            with torch.npu.device(self.device):
                if any(result.exception() is not None for result in ready):
                    # A failed submission may have enqueued device work without its event.
                    if stream is not None:
                        stream.synchronize()
                else:
                    for result in reversed(ready):
                        event = result.result()
                        if event is not None:
                            event.synchronize()
                            break
        except Exception:
            logging.getLogger(__name__).exception("Failed to retire Engram UVA reads")
            return False
        return True

    def _submit_layer(self, stream, out, valid, index, slots, staging=None, graph_event=None):
        tokens = out.shape[0]
        if tokens == 0:
            if graph_event is not None:
                with torch.npu.device(self.device), torch.npu.stream(stream):
                    graph_event.record(stream)
            return None
        host_slots = self._slot_host if staging is None else staging.host_slots
        device_slots = self._slot_dev if staging is None else staging.device_slots
        slot_host = host_slots[index, :tokens]
        slot_np = slot_host.numpy()
        slot_np.fill(-1)
        slot_np[valid] = slots
        with torch.inference_mode(), torch.npu.device(self.device), torch.npu.stream(stream):
            slot_dev = device_slots[index, :tokens]
            slot_dev.copy_(slot_host, non_blocking=True)
            gather_dequantize_host_uva(
                self.planes.codes, self.planes.scales, slot_dev.view(-1),
                head_start=0, local_heads=1, pad_heads=1, output=out.view(-1, 256),
                vocab_start=0, vocab_end=self.capacity,
            )
            event = torch.npu.Event()
            event.record(stream)
            if graph_event is not None:
                graph_event.record(stream)
        return event

    def queue_prefill(self, key, hashes, mask, padded_tokens, gather_stream) -> None:
        # Stage the next exact chunk while the current model runs.
        tokens = hashes.shape[0]
        assert self._prefetched_chunk is None and tokens <= padded_tokens <= self.max_tokens
        staging = _LayeredStaging(
            torch.empty(hashes.shape, dtype=torch.int32, device="cpu", pin_memory=True),
            torch.empty(tokens, dtype=torch.bool, device="cpu", pin_memory=True),
            torch.empty((self.num_layers, tokens, self.col_count),
                        dtype=torch.int32, device="cpu", pin_memory=True),
            torch.empty((self.num_layers, tokens, self.col_count),
                        dtype=torch.int32, device=self.device),
        )
        mask_padded = mask.new_zeros(padded_tokens)
        mask_padded[:tokens].copy_(mask)
        outputs = {layer: hashes.new_empty((padded_tokens, self.col_count, 256), dtype=torch.bfloat16)
                   for layer in self.layer_ids}
        for out in outputs.values():
            out[tokens:].zero_()
        staging.hashes.copy_(hashes, non_blocking=True)
        staging.mask.copy_(mask, non_blocking=True)
        inputs_ready = torch.npu.Event()
        inputs_ready.record()
        lookup = self._queue_layered(
            tokens, {layer: out[:tokens] for layer, out in outputs.items()},
            stream=gather_stream, d2h_event=inputs_ready, mask_host=staging.mask,
            staging=staging, retire_after_gather=True,
        )
        graph_inputs = {
            "engram_lookups": {layer: out.flatten(1) for layer, out in outputs.items()},
            "engram_mask": mask_padded,
            "engram_mask_ready_event": inputs_ready,
        }
        self._prefetched_chunk = _PrefetchedChunk(key, lookup, graph_inputs, inputs_ready)

    def queue_failed_prefill(self, key) -> None:
        # Report a local staging failure through the speculative round's P0.
        # The current forward can still consume its own lookup on every rank.
        lookup = self._queue_layered(0, {}, retire_after_gather=True, staging_failed=True)
        self._prefetched_chunk = _PrefetchedChunk(key, lookup, None, None)

    def take_prefill(self, key) -> dict[str, Any] | None:
        pending = self._prefetched_chunk
        if pending is None:
            return None
        if pending.key != key:
            self._discard_prefill()
            return None
        self._prefetched_chunk = None
        self._layered_lookup = pending.lookup
        graph_inputs = pending.graph_inputs
        if graph_inputs is None:
            pending.lookup.producer.result()
            raise RuntimeError("Engram SSD prefill staging failed")
        main = torch.npu.current_stream()
        for tensor in (*graph_inputs["engram_lookups"].values(), graph_inputs["engram_mask"]):
            tensor.record_stream(main)
        if self.protocol.rank == 0:
            logging.getLogger(__name__).info(
                "engram-ssd-prefill-lookahead hit step=%d start=%d end=%d",
                pending.lookup.step, key[2], key[3],
            )
        return graph_inputs

    def _discard_prefill(self, *, failed: bool = False) -> None:
        pending = self._prefetched_chunk
        if pending is None:
            return
        self._prefetched_chunk = None
        if self.protocol.rank == 0:
            logging.getLogger(__name__).info(
                "engram-ssd-prefill-lookahead discard step=%d start=%d end=%d",
                pending.lookup.step, pending.key[2], pending.key[3],
            )
        self._join_layered(pending.lookup, failed=failed)

    def consume_layer(self, layer: int) -> None:
        lookup = self._layered_lookup
        if lookup is None or lookup.tokens == 0:
            return
        before = time.perf_counter()
        event = lookup.ready[self.layer_ids.index(layer)].result()
        lookup.wait_ms[layer] = (time.perf_counter() - before) * 1e3
        if event is not None:
            torch.npu.current_stream().wait_event(event)

    def _finish_layered(self, *, failed: bool) -> None:
        lookup = self._layered_lookup
        if lookup is None:
            return
        try:
            self._join_layered(lookup, failed=failed)
        finally:
            self._layered_lookup = None
        if lookup.wait_ms and (lookup.tokens >= 512 or lookup.step % 200 == 0):
            logging.getLogger(__name__).info(
                "engram-ssd-layer-consumer step=%d cpu_wait_ms=%s", lookup.step, lookup.wait_ms)

    def _join_layered(self, lookup, *, failed: bool) -> None:
        if lookup.retired is None:
            # The producer already owns the UVA fence and can move on to the
            # next chunk independently of the main forward.
            try:
                lookup.producer.result()
                if not failed:
                    for result in lookup.ready:
                        result.result()
            except Exception:
                if not failed:
                    raise
                logging.getLogger(__name__).exception("Engram SSD producer failed during retirement")
            return
        # The default path retires on the caller after all device submissions.
        wait(lookup.ready)
        safe = False
        try:
            if lookup.stream is not None:
                lookup.stream.synchronize()
            safe = True
        finally:
            lookup.retired.set_result(safe)
            try:
                lookup.producer.result()
                if safe and not failed:
                    for result in lookup.ready:
                        result.result()
            except Exception:
                if not failed and safe:
                    raise
                logging.getLogger(__name__).exception("Engram SSD producer failed during retirement")

    def preload(self, rows_path: str, *, artifact_version: str = "") -> dict[str, Any]:
        """预热预载(04 §4.1): 行列表绑定 artifact identity, 预算封顶。

        预载行是普通缓存内容——后续正常 lease 可淘汰; 与业务热行预载是
        两件事。行列表 npz 含 ``ids``(uint64 packed global ids)与
        ``artifact_version``(str), 不匹配即拒绝。
        """
        if self._closed:
            raise RuntimeError("EngramSSDRuntime closed")
        data = np.load(rows_path)
        ids = np.ascontiguousarray(data["ids"], dtype=np.uint64)
        got_version = str(data["artifact_version"]) if "artifact_version" in data else ""
        expected = artifact_version or self.manifest.version
        if got_version != expected:
            raise SSDInputError(1, f"preload artifact mismatch: {got_version!r} != {expected!r}")
        ids = np.unique(ids)
        budget_rows = self.cfg.preload_max_bytes // 288 if self.cfg.preload_max_bytes else len(ids)
        ids = ids[:min(len(ids), budget_rows, self.capacity)]
        filled = 0
        batch = max(1, self.capacity)
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            lease = self.cache.acquire(chunk)
            if len(lease.miss_rows):
                self._fill_misses(lease)
            self.cache.publish(lease)
            self.cache.release(lease)
            filled += len(lease.miss_rows)
        self._log_preload(rows_path, len(ids), filled)
        return {"requested": int(len(data["ids"])), "unique_budgeted": int(len(ids)),
                "filled": int(filled)}

    def _log_preload(self, path: str, rows: int, filled: int) -> None:
        import logging

        logging.getLogger(__name__).info(
            "engram ssd preload: %s -> %d rows (%d filled)", path, rows, filled)

    def _needs_windows(self, layer_valid: list[np.ndarray]) -> bool:
        """最坏估计(全部有效行唯一)超过容量 → 窗口分块(04 §7)。"""
        est = sum(int(v.sum()) for v in layer_valid)
        return est > self.capacity

    def _requires_shared_windows(self) -> bool:
        """Use a group-invariant bound: local sizes can differ, including zero."""
        return (self.protocol is not None and self.protocol.epoch_fn() >= 1
                and self.max_tokens * self.num_layers * self.col_count * self.protocol.size > self.capacity)

    def prepare_idle(self, *, skip_layerwise: bool = False) -> None:
        """Join the same planning and row rounds as active DP/PCP peers."""
        if self.protocol is None or self.protocol.epoch_fn() < 1:
            return
        if not skip_layerwise and self.layerwise_active:
            if self._layered_lookup is not None:
                raise RuntimeError("Previous Engram SSD layerwise lookup is still active")
            self._layered_lookup = self._queue_layered(0, {})
            return
        windows = (self.protocol.token_windows(0, rows_per_token=self.num_layers * self.col_count,
                                               capacity=self.capacity)
                   if self._requires_shared_windows() else [(0, 0)])
        for _ in windows:
            self.protocol.step(np.zeros(0, np.uint64), num_tokens=0)

    def _prepare_shared_windowed(self, mask_np, host, layer_valid, T, *, step_id,
                                 output_buffers=None):
        """Prepare group-bounded windows, retiring UVA reads before the next P0.

        The protocol releases its lease at P4. No subsequent fill is permitted
        until every consumer enters the next P0, after its gather has completed.
        This ordering also holds for empty windows and idle ranks.
        """
        windows = self.protocol.token_windows(T, rows_per_token=self.num_layers * self.col_count,
                                              capacity=self.capacity)
        outputs = {layer: (output_buffers[layer][:T] if output_buffers is not None and layer in output_buffers
                           else self._output[li, :T]) for li, layer in enumerate(self.layer_ids)}
        for lo, hi in windows:
            valids = [v[lo:hi] for v in layer_valid]
            parts = [(host[lo:hi, li, :].astype(np.uint64) + np.uint64(self.layer_bases[li]))[v]
                     for li, v in enumerate(valids)]
            ids = np.concatenate(parts)
            slots = self.protocol.step(ids, num_tokens=int(mask_np[lo:hi].sum()))
            if lo == hi:
                continue
            offset = 0
            for li, valid in enumerate(valids):
                target = self._slot_host[li, lo:hi].numpy()
                target.fill(-1)
                count = int(valid.sum())
                target[valid] = slots[offset:offset + count]
                offset += count
            self._slot_dev[:, lo:hi].copy_(self._slot_host[:, lo:hi], non_blocking=True)
            for li, layer in enumerate(self.layer_ids):
                gather_dequantize_host_uva(
                    self.planes.codes, self.planes.scales, self._slot_dev[li, lo:hi].reshape(-1),
                    head_start=0, local_heads=1, pad_heads=1,
                    output=outputs[layer][lo:hi].reshape(-1, 256),
                    vocab_start=0, vocab_end=self.capacity)
            self._gather_event.record()
            self._gather_event.synchronize()
        self._gather_event.record()
        self._step += 1
        return PreparedEngram(step_id=step_id, layer_outputs=outputs, mask=mask_np,
                              gather_event=self._gather_event,
                              stats={"windowed": True, "protocol": True, "windows": len(windows)})

    def _prepare_windowed(self, hashes, mask_np, host, layer_valid, T, *, step_id):
        """token 窗口轮次: 每窗 填充→gather→release, 输出写固定缓冲切片。"""
        win = max(1, self.capacity // (self.num_layers * max(self.col_count, 1)))
        if win < 1:
            raise SSDCacheError(2, "capacity too small for one token window")
        outputs: dict[int, torch.Tensor] = {}
        stats_acc: dict[str, float] = {}
        for li in range(self.num_layers):
            outputs[self.layer_ids[li]] = self._output[li, :T]
        for w0 in range(0, T, win):
            w1 = min(T, w0 + win)
            parts, valids = [], []
            for li in range(self.num_layers):
                v = layer_valid[li][w0:w1]
                h = host[w0:w1, li, :]
                rows = (h.astype(np.uint64) + np.uint64(self.layer_bases[li]))[v]
                parts.append(rows)
                valids.append(v)
            ids = np.concatenate(parts) if parts else np.zeros(0, np.uint64)
            lease = self.cache.acquire(ids)
            if len(lease.miss_rows):
                self._fill_misses(lease)
            self.cache.publish(lease)
            self._lease = lease
            slot_np = self._slot_host.numpy()
            base_li = 0   # 本窗口内层序拼接的行基址(lease.slots 与该序对齐)
            for li in range(self.num_layers):
                m = slot_np[li, w0:w1]
                m.fill(-1)
                v = valids[li]
                vp = np.nonzero(v.ravel())[0]
                if len(vp):
                    m.reshape(-1)[vp] = lease.slots[base_li:base_li + len(vp)]
                base_li += len(vp)
            self._slot_dev[:, w0:w1].copy_(self._slot_host[:, w0:w1], non_blocking=True)
            for li in range(self.num_layers):
                ids_dev = self._slot_dev[li, w0:w1].reshape(-1)
                gather_dequantize_host_uva(
                    self.planes.codes, self.planes.scales, ids_dev,
                    head_start=0, local_heads=1, pad_heads=1,
                    output=outputs[self.layer_ids[li]][w0:w1].reshape(-1, 256),
                    vocab_start=0, vocab_end=self.capacity)
            self._gather_event.record()
            self._gather_event.synchronize()
            self.cache.release(lease)
            self._lease = None
        self._gather_event.record()
        self._step += 1
        return PreparedEngram(step_id=step_id, layer_outputs=outputs,
                              mask=mask_np, gather_event=self._gather_event,
                              stats={"windowed": True, "windows": (T + win - 1) // win,
                                     "window_tokens": win})

    def _fill_misses(self, lease) -> None:
        """统一 miss 填充: 大批走 extent 合并, 小批逐行。"""
        from vllm_ascend.models.deepseek_v41.engram.ssd.engine import plan_row_extents

        if len(lease.miss_rows) >= self.extents_threshold:
            ext, sct, _ = plan_row_extents(
                lease.miss_rows, lease.miss_slots,
                base1=self.layer_bases[1] if self.num_layers > 1 else 0,
                dual=self.engine.dual)
            self.engine.read_extents(ext, sct, self._codes_np, self._scales_np)
        else:
            self.engine.fill(lease.miss_rows, lease.miss_slots,
                             self._codes_np, self._scales_np)

    def finish_execution(self, *, failed: bool = False) -> None:
        """gather 完成后释放租约; 步失败时 abort(不清 state unknown 的 poison 判定)。"""
        if self._layered_lookup is not None or getattr(self, "_prefetched_chunk", None) is not None:
            try:
                self._finish_layered(failed=failed)
            finally:
                if getattr(self, "prefill_lookahead", False):
                    if failed:
                        self._discard_prefill(failed=True)
                    elif self._prefetched_chunk is not None:
                        # The next runner step can recycle the request's token
                        # history. Fence that read only, independently of SSD.
                        inputs_ready = self._prefetched_chunk.inputs_ready
                        if inputs_ready is not None:
                            torch.npu.current_stream().wait_event(inputs_ready)
            return
        lease = self._lease
        if self.protocol is not None and self.protocol.epoch_fn() >= 1:
            # 协议路径: 租约由协议在 P4 后统一释放
            self._gather_event.synchronize()
            return
        if lease is None:
            return
        self._lease = None
        if failed:
            # 已知失败(引擎捕获的异常): 本批未完整消费, abort 移除 pending;
            # 不在此 raise——错误处理路径不得被二次异常掩盖原始错误。
            try:
                self._gather_event.synchronize()
            except Exception:
                self.cache.poison("gather sync failed on failed step")
                self._closed = True
                return
            if lease.published:
                self.cache.release(lease)
            else:
                self.cache.abort(lease)
            return
        # 等待 gather 真正完成, 之后才允许未来批次覆写 victim 槽位
        self._gather_event.synchronize()
        self.cache.release(lease)

    # ---------- 生命周期 ----------

    def close(self) -> None:
        if self._closed:
            return
        if self._layered_executor is not None:
            self._finish_layered(failed=True)
            self._discard_prefill(failed=True)
            self._layered_executor.shutdown()
        if self._lease is not None and self.cache is not None:
            self.cache.abort(self._lease)
            self._lease = None
        self._closed = True
        if self.engine is not None:
            self.engine.close()
        if self.cache is not None:
            self.cache.close()
        self.planes.close()
