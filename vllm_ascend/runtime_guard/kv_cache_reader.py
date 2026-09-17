#
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

"""Read paged KV cache blocks for a request (direct D2H, no msprobe)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# Default extra free space required on top of the estimated dump payload.
DEFAULT_DUMP_FREE_HEADROOM_BYTES = 5 * 1024 * 1024 * 1024


def _pp_start_layer(runner: Any) -> int:
    """First global transformer layer index on this PP rank (0 if unknown)."""
    roots: list[Any] = [getattr(runner, "model", None)]
    getter = getattr(runner, "get_model", None)
    if callable(getter):
        try:
            roots.append(getter())
        except Exception:
            pass
    for root in roots:
        for obj in (root, getattr(root, "model", None) if root is not None else None):
            try:
                return int(obj.start_layer)  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError):
                continue
    return 0


def _kv_cache_config_layer_names(runner: Any) -> list[str]:
    """Layer names in the same order ``bind_kv_cache`` appends to the list.

    Last-PP only holds a suffix of the model; these names are already global
    (e.g. ``model.layers.14.self_attn``), unlike list-index ``layer_0``.
    """
    cfg = getattr(runner, "kv_cache_config", None)
    groups = getattr(cfg, "kv_cache_groups", None) if cfg is not None else None
    if not groups:
        return []
    names: list[str] = []
    for group in groups:
        for name in getattr(group, "layer_names", None) or []:
            names.append(str(name))
    if not names:
        return []
    try:
        from vllm.v1.worker.utils import extract_layer_index

        names.sort(key=lambda n: (extract_layer_index(n), n))
    except Exception:
        names.sort()
    return names


def _list_layer_name(i: int, *, names: list[str], start_layer: int) -> str:
    if 0 <= i < len(names):
        return names[i]
    return f"layer_{start_layer + i}"


def _iter_kv_tensors(
    kv_caches: Any,
    *,
    list_names: list[str] | None = None,
    start_layer: int = 0,
) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    if kv_caches is None:
        return out
    names = list_names or []
    if isinstance(kv_caches, dict):
        for name, val in kv_caches.items():
            if isinstance(val, torch.Tensor):
                out.append((str(name), val))
            elif isinstance(val, (list, tuple)):
                for i, t in enumerate(val):
                    if isinstance(t, torch.Tensor):
                        out.append((f"{name}[{i}]", t))
    elif isinstance(kv_caches, list):
        for i, val in enumerate(kv_caches):
            layer_name = _list_layer_name(i, names=names, start_layer=start_layer)
            if isinstance(val, torch.Tensor):
                out.append((layer_name, val))
            elif isinstance(val, (list, tuple)):
                for j, t in enumerate(val):
                    if isinstance(t, torch.Tensor):
                        out.append((f"{layer_name}[{j}]", t))
    return out


def _slice_blocks(tensor: torch.Tensor, block_ids: list[int]) -> tuple[torch.Tensor, list[int]] | None:
    """Select blocks along dim0, keeping ``[n_sel_blocks, block_size, ...]``.

    Returns ``None`` when ``block_ids`` is empty so callers refuse a full-cache
    copy (empty ids must not mean "dump everything"). Returns the payload
    together with the block ids actually used (B'5c: partial out-of-range ids
    must not be recorded as if fully dumped).
    """
    if not block_ids:
        return None
    if tensor.ndim < 2:
        return tensor.detach().cpu(), list(block_ids)
    # Common paged layout: [num_blocks, block_size, ...]
    max_block = int(tensor.shape[0]) - 1
    valid = [int(b) for b in block_ids if 0 <= int(b) <= max_block]
    if not valid:
        return None
    # Advanced index keeps [n_sel_blocks, block_size, ...] (same rank as dump_all).
    sliced = tensor[valid]
    return sliced.detach().cpu(), valid


def free_bytes_at(path: Path) -> int | None:
    """Available bytes on the filesystem that contains ``path``.

    Walks up to an existing ancestor. Returns ``None`` if the volume cannot be
    queried (caller should not skip the dump solely for that).
    """
    probe = path
    try:
        for _ in range(16):
            if probe.exists():
                break
            parent = probe.parent
            if parent == probe:
                break
            probe = parent
        if not probe.exists():
            return None
        st = os.statvfs(probe)
        return int(st.f_bavail) * int(st.f_frsize)
    except OSError:
        return None


def _block_payload_bytes(tensor: torch.Tensor, block_ids: list[int]) -> int:
    nbytes = int(tensor.nbytes)
    if tensor.ndim < 2:
        return nbytes
    nblocks = int(tensor.shape[0])
    if nblocks <= 0:
        return 0
    valid = sum(1 for b in block_ids if 0 <= int(b) < nblocks)
    return (nbytes // nblocks) * valid


@dataclass
class KvDumpSnapshot:
    """CPU-side KV payload ready for async ``torch.save``."""

    path: Path
    payload: dict[str, Any]


class KvCacheReader:
    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def _kv_sources(self) -> list[tuple[str, Any]]:
        runner = self._runner
        sources: list[tuple[str, Any]] = []
        kv_dict = getattr(runner, "kv_caches", None)
        if isinstance(kv_dict, dict) and kv_dict:
            sources.append(("kv_caches", kv_dict))
        kv_list = getattr(runner, "kv_caches", None)
        if isinstance(kv_list, list) and kv_list and not sources:
            sources.append(("kv_caches", kv_list))
        return sources

    def _list_layer_meta(self) -> tuple[list[str], int]:
        return (
            _kv_cache_config_layer_names(self._runner),
            _pp_start_layer(self._runner),
        )

    def estimate_dump_bytes(self, *, block_ids: list[int]) -> int:
        """Host-side size estimate of a request dump (no D2H)."""
        ids = list(block_ids)
        names, start = self._list_layer_meta()
        total = 0
        for _src, kv_caches in self._kv_sources():
            for _name, tensor in _iter_kv_tensors(
                kv_caches, list_names=names, start_layer=start
            ):
                total += _block_payload_bytes(tensor, ids)
        return total

    def iter_request_snapshots(
        self,
        *,
        req_id: str,
        block_ids: list[int],
        out_dir: Path,
    ) -> "Iterator[KvDumpSnapshot]":
        """Yield per-layer CPU snapshots for one request (D2H per layer).

        Only the request's ``block_ids`` are dumped (never the full KV pool).
        B'3: a generator lets callers enqueue each layer for async save as soon
        as it lands on host.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = list(block_ids)
        produced = False
        # Record which shard produced the dump so cross-rank comparisons can
        # stitch TP heads / PP layers / CP token interleaves.
        from vllm_ascend.runtime_guard.rank_gate import (
            dump_rank_tag,
            runner_cp_rank,
            runner_pp_rank,
            runner_tp_rank,
        )

        try:
            tp_rank = runner_tp_rank(self._runner)
        except Exception:
            tp_rank = 0
        rank_tag = dump_rank_tag(self._runner)
        pp_rank = runner_pp_rank(self._runner)
        cp_rank = runner_cp_rank(self._runner)
        names, start = self._list_layer_meta()
        for src_name, kv_caches in self._kv_sources():
            for layer_name, tensor in _iter_kv_tensors(
                kv_caches, list_names=names, start_layer=start
            ):
                num_kv_heads = int(tensor.shape[-2]) if tensor.dim() >= 3 else None
                if not ids:
                    logger.warning(
                        "[runtime_guard dump_kv] skip layer=%s req_id=%s: empty block_ids "
                        "(refusing full-cache D2H)",
                        layer_name,
                        req_id,
                    )
                    continue
                sliced = _slice_blocks(tensor, ids)
                if sliced is None:
                    logger.warning(
                        "[runtime_guard dump_kv] skip layer=%s req_id=%s: no valid blocks in %s",
                        layer_name,
                        req_id,
                        ids,
                    )
                    continue
                payload_tensor, used_ids = sliced
                safe_layer = layer_name.replace("/", "_")
                path = out_dir / f"{req_id}_{safe_layer}_req.pt"
                produced = True
                yield KvDumpSnapshot(
                    path=path,
                    payload={
                        "req_id": req_id,
                        "block_ids": used_ids,
                        "layer": layer_name,
                        "source": src_name,
                        "rank_tag": rank_tag,
                        "tp_rank": tp_rank,
                        "pp_rank": pp_rank,
                        "cp_rank": cp_rank,
                        "num_kv_heads": num_kv_heads,
                        "tensor": payload_tensor,
                    },
                )
        if not produced:
            logger.warning("[runtime_guard dump_kv] no kv tensors found req_id=%s", req_id)

    def snapshot_request_blocks(
        self,
        *,
        req_id: str,
        block_ids: list[int],
        out_dir: Path,
    ) -> list[KvDumpSnapshot]:
        """Sync D2H of KV for one request. Does not write files."""
        return list(
            self.iter_request_snapshots(
                req_id=req_id,
                block_ids=block_ids,
                out_dir=out_dir,
            )
        )

    @staticmethod
    def write_snapshots(snapshots: list[KvDumpSnapshot]) -> list[str]:
        """Async-safe: write previously snapshotted CPU tensors to disk."""
        written: list[str] = []
        for snap in snapshots:
            try:
                snap.path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snap.payload, snap.path)
                written.append(str(snap.path))
            except Exception as exc:
                logger.exception(
                    "[runtime_guard dump_kv] torch.save failed path=%s",
                    snap.path,
                )
                payload = snap.payload if isinstance(snap.payload, dict) else {}
                dump_root = payload.get("dump_root")
                req_id = str(payload.get("req_id") or "")
                incident_type = str(payload.get("incident_type") or "unknown")
                rank_tag = str(payload.get("rank_tag") or "")
                wave = payload.get("dump_arm_wave")
                try:
                    wave_i = int(wave) if wave is not None else None
                except (TypeError, ValueError):
                    wave_i = None
                if dump_root and req_id:
                    from vllm_ascend.runtime_guard.dump_io import write_kv_dump_skipped

                    write_kv_dump_skipped(
                        dump_root,
                        req_id=req_id,
                        incident_type=incident_type,
                        reason="torch_save_failed",
                        stage="drain",
                        rank_tag=rank_tag,
                        wave=wave_i,
                        detail={
                            "error": f"{type(exc).__name__}: {exc}",
                            "path": str(snap.path),
                            "layer": payload.get("layer"),
                        },
                    )
        if written:
            req_id = snapshots[0].payload.get("req_id")
            logger.info(
                "[runtime_guard dump_kv] wrote req_id=%s pt_files=%d dir=%s",
                req_id,
                len(written),
                snapshots[0].path.parent,
            )
        return written
