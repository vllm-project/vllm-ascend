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

All per-step buffers are preallocated; steady state allocates nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    SSDRowCache,
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
            self.cache = SSDRowCache(
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
        self._slot_dev = torch.empty((num_layers, max_tokens, col_count), dtype=torch.int32,
                                     device=device)
        self._slot_host = torch.empty((num_layers, max_tokens, col_count), dtype=torch.int32,
                                      device="cpu")
        self._output = torch.empty((num_layers, max_tokens, col_count, 256),
                                   dtype=torch.bfloat16, device=device)
        # 超过该 miss 数走 extent 合并读(decode 逐行阈值以下)
        self.extents_threshold = 64
        self._lease: CacheLease | None = None
        # DP 协议(可选): epoch>=1 时行→槽位走组协议
        self.protocol = protocol
        self._gather_event = torch.npu.Event()
        self._closed = False
        self._step = 0

    # ---------- 主路径 ----------

    def prepare(self, hashes: torch.Tensor, token_mask: torch.Tensor | None,
                *, step_id: int = -1, output_buffers: dict | None = None) -> PreparedEngram:
        """hashes: [T, L, C] int32 device; mask: [T] bool 或 None(全有效)。

        output_buffers: {layer_id: [max_tokens, C_local, 256] BF16}(调用方固定
        地址缓冲, FULL replay 依赖); 缺省写 runtime 内部固定缓冲。
        """
        if self._closed:
            raise RuntimeError("EngramSSDRuntime closed")
        if self._lease is not None:
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
        d2h_event.record()
        d2h_event.synchronize()
        t1 = time.perf_counter()
        self._d2h_enqueue_ms = (tb - ta) * 1e3
        self._d2h_wait_ms = (t1 - tb) * 1e3
        host = d2h.numpy()

        mask_np = (np.ones(T, bool) if token_mask is None
                   else token_mask.detach().cpu().numpy().astype(bool))

        # 2) 按层拼有效行 → packed global ids(请求顺序); 超容量则 token 窗口分块
        layer_valid: list[np.ndarray] = []
        ids_parts: list[np.ndarray] = []
        for li in range(self.num_layers):
            h = host[:, li, :]                       # [T, C_local] int32
            valid = mask_np[:, None] & (h != DEAD_ID) & (h >= 0) & (h < self.layer_rows[li])
            rows = (h.astype(np.uint64) + np.uint64(self.layer_bases[li]))[valid]
            ids_parts.append(rows)
            layer_valid.append(valid)
        ids = (np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.uint64))
        t2 = time.perf_counter()

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

        self._step += 1
        if self._step % 200 == 0:
            import logging

            proto = ""
            if self.protocol is not None:
                ph = self.protocol.stats.phase_ms
                n = max(self.protocol.stats.steps, 1)
                proto = (f" proto[p0={ph.get('p0_header',0)/n:.2f} p1={ph.get('p1_gather',0)/n:.2f}"
                         f" p3={ph.get('p3_scatter',0)/n:.2f} p4={ph.get('p4_ack',0)/n:.2f}ms]"
                         f" rounds={self.protocol.stats.steps} failed={self.protocol.stats.rounds_failed}")
            logging.getLogger(__name__).info(
                "engram-ssd step=%d T=%d rows=%d miss=%d d2h=%.2f(enq=%.2f,wait=%.2f) "
                "plan=%.2f fill=%.2f h2d=%.2f uva=%.2f total=%.2fms%s",
                self._step, T, len(ids),
                len(lease.miss_rows) if lease is not None else 0,
                (t1-t0)*1e3, getattr(self, "_d2h_enqueue_ms", 0),
                getattr(self, "_d2h_wait_ms", 0),
                (t2-t1)*1e3, (t3-t2)*1e3, (t4-t3)*1e3, (t5-t4)*1e3,
                (t5-t0)*1e3, proto)
        stats = dict(
            step=self._step, step_id=step_id, tokens=T,
            rows=len(ids),
            misses=len(lease.miss_rows) if lease is not None else 0,
            hits=lease.stats.get("hit_rows", 0) if lease is not None else 0,
            d2h_ms=(t1 - t0) * 1e3, plan_ms=(t2 - t1) * 1e3,
            fill_ms=(t3 - t2) * 1e3, h2d_ms=(t4 - t3) * 1e3,
            uva_ms=(t5 - t4) * 1e3, total_ms=(t5 - t0) * 1e3,
            cache=lease.stats if lease is not None else {},
            engine=fill_stats,
            protocol=protocol_slots is not None)
        return PreparedEngram(step_id=step_id, layer_outputs=outputs,
                              mask=mask_np, gather_event=self._gather_event,
                              stats=stats)

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
        if self._lease is not None and self.cache is not None:
            self.cache.abort(self._lease)
            self._lease = None
        self._closed = True
        if self.engine is not None:
            self.engine.close()
        if self.cache is not None:
            self.cache.close()
        self.planes.close()
