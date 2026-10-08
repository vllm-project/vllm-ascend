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

"""KV dump path: block ids, D2H reader, on-disk skipped/request_info writers."""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_guard.rank_gate import (
    dump_rank_tag,
    runner_cp_rank,
    runner_pp_rank,
    runner_tp_rank,
)

logger = init_logger_ascend(__name__)

# When True on request_info / .pt / report: req was already finished at arm or
# drain time. Blocks may have been freed or reused — treat tensors as suspect.
REQUEST_FINISHED_AT_DUMP_KEY = "request_finished_at_dump"

# ---- dump path / free space ----

# Cap how far we walk toward an existing ancestor when probing free space for
# a path that does not exist yet (mkdir happens later).
_FREE_BYTES_MAX_PARENT_WALK = 16


def kv_dump_wave_dirname(wave: int | None) -> str:
    """Subdir under ``{type}/{req_id}/`` separating dumps across steps."""
    if wave is None:
        return "wave_unknown"
    return f"wave_{int(wave)}"


def parse_dump_arm_wave(wave: Any) -> int | None:
    """Coerce ``dump_arm_wave`` / job ``wave`` fields to ``int`` (else ``None``)."""
    try:
        return int(wave) if wave is not None else None
    except (TypeError, ValueError):
        return None


def free_bytes_at(path: Path) -> int | None:
    """Available bytes on the filesystem that contains ``path``.

    Walks up to an existing ancestor (at most ``_FREE_BYTES_MAX_PARENT_WALK``
    levels). Returns ``None`` if the volume cannot be queried (caller should
    not skip the dump solely for that).
    """
    probe = path
    try:
        for _ in range(_FREE_BYTES_MAX_PARENT_WALK):
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


def write_kv_dump_skipped(
    dump_root: str | Path,
    *,
    req_id: str,
    incident_type: str,
    reason: str,
    stage: str = "arm",
    rank_tag: str = "",
    wave: int | None = None,
    detail: dict[str, Any] | None = None,
) -> Path | None:
    """Record why KV dump did not fully succeed on this rank.

    Layout (same tree as ``.pt`` when ``rank_tag`` is set)::

      {dump_root}/{incident_type}/{req_id}/wave_<N>/[rank_tag/]dump_skipped.json

    Arm skips (leader) and drain skips (each last-PP TP) all use this helper.
    ``dump_skipped`` means incomplete / failed — ``request_info`` or some
    ``.pt`` files may already exist beside it.
    """
    if not req_id:
        return None
    out_dir = Path(dump_root) / str(incident_type or "unknown") / str(req_id) / kv_dump_wave_dirname(wave)
    if rank_tag:
        out_dir = out_dir / str(rank_tag)
    path = out_dir / "dump_skipped.json"
    payload: dict[str, Any] = {
        "reason": str(reason or "unknown"),
        "req_id": str(req_id),
        "incident_type": str(incident_type or "unknown"),
        "stage": str(stage or "arm"),
        "rank_tag": str(rank_tag or ""),
        "dump_arm_wave": int(wave) if wave is not None else None,
        "ts": time.time(),
    }
    if detail:
        payload["detail"] = dict(detail)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.warning(
            "[runtime_guard dump_kv] skipped reason=%s stage=%s req_id=%s type=%s rank=%s marker=%s",
            reason,
            stage,
            req_id,
            incident_type,
            rank_tag or "-",
            path,
        )
        return path
    except OSError as exc:
        logger.warning(
            "[runtime_guard dump_kv] failed to write skip marker req_id=%s path=%s: %s",
            req_id,
            path,
            exc,
        )
        return None


def write_kv_dump_request_info(
    dump_root: str | Path,
    *,
    req_id: str,
    incident_type: str,
    detail: dict[str, Any] | None,
    rank_tag: str = "",
    wave: int | None = None,
    block_ids: list[int] | None = None,
    tokenizer: Any | None = None,
    save_sensitive_info: bool = False,
    decode_token_ids: bool = True,
    max_prompt_token_ids: int = 100000,
    max_output_token_ids: int = 100000,
    request_finished_at_dump: bool = False,
) -> Path | None:
    """Last-PP TP0: write report-like request metadata next to KV ``.pt`` shards.

    Path: ``{dump_root}/{incident_type}/{req_id}/{wave_N}/request_info.json``

    ``request_finished_at_dump``: req was already finished when dump armed;
    KV may already be freed/reused — treat ``.pt`` as suspect.
    """
    if not req_id:
        return None
    from vllm_ascend.observability.runtime_guard.report import dumps_report_json, sanitize_report_detail

    out_dir = Path(dump_root) / str(incident_type or "unknown") / str(req_id) / kv_dump_wave_dirname(wave)
    path = out_dir / "request_info.json"
    safe_detail = sanitize_report_detail(
        detail,
        save_sensitive_info=save_sensitive_info,
        max_prompt_token_ids=max_prompt_token_ids,
        max_output_token_ids=max_output_token_ids,
        decode_token_ids=decode_token_ids and save_sensitive_info,
        tokenizer=tokenizer if (decode_token_ids and save_sensitive_info) else None,
    )
    payload = {
        "ts": time.time(),
        "incident_type": str(incident_type or "unknown"),
        "req_id": str(req_id),
        "rank": str(rank_tag or ""),
        "dump_arm_wave": int(wave) if wave is not None else None,
        "block_ids": list(block_ids) if block_ids is not None else safe_detail.get("block_ids"),
        REQUEST_FINISHED_AT_DUMP_KEY: bool(request_finished_at_dump),
        "decode_token_ids": bool(decode_token_ids and save_sensitive_info),
        "max_prompt_token_ids": int(max_prompt_token_ids),
        "max_output_token_ids": int(max_output_token_ids),
        "detail": safe_detail,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_report_json(payload, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "[runtime_guard dump_kv] request_info req_id=%s type=%s path=%s",
            req_id,
            incident_type,
            path,
        )
        return path
    except OSError as exc:
        logger.warning(
            "[runtime_guard dump_kv] failed to write request_info req_id=%s path=%s: %s",
            req_id,
            path,
            exc,
        )
        return None


# ---- KV block metadata ----


def block_ids_for_request(
    runner: Any,
    req_id: str,
    req_idx: int | None = None,
    *,
    kv_cache_group: int = 0,
    input_batch: Any = None,
) -> list[int]:
    """Return logical GPU block ids for ``req_id`` (group 0 by default).

    MRV2 clears ``execute_model_state`` at the start of ``sample_tokens``, so
    end-of-wave dump often has no live ``input_batch``. Fall back to persistent
    ``req_states`` + ``runner.block_tables`` (BUG-4).
    """
    if not req_id or runner is None:
        return []

    requests = getattr(runner, "requests", None)
    if requests is not None:
        state = requests.get(req_id)
        if state is not None:
            raw = getattr(state, "block_ids", None)
            parsed = _normalize_block_ids(raw, kv_cache_group=kv_cache_group)
            if parsed:
                return parsed

    if input_batch is None:
        # V2: runner.input_batch stays None; prefer execute_model_state batch
        # while it still exists (pre-sample / mid-execute).
        input_batch = _runner_input_batch(runner)

    if input_batch is not None:
        idx = req_idx
        if idx is None:
            mapping = getattr(input_batch, "req_id_to_index", None)
            if isinstance(mapping, dict) and req_id in mapping:
                idx = int(mapping[req_id])
            else:
                req_ids = list(getattr(input_batch, "req_ids", None) or [])
                try:
                    idx = req_ids.index(req_id)
                except ValueError:
                    idx = None
        if idx is not None:
            idx = int(idx)
            table = _block_table_for_group(input_batch, kv_cache_group)
            if table is not None:
                try:
                    num_blocks = int(table.num_blocks_per_row[idx])
                    if num_blocks <= 0:
                        return []
                    row = table.block_table.np[idx, :num_blocks]
                    return [int(x) for x in row.tolist()]
                except Exception:
                    # Not a host-mirrored v1 table (e.g. wrong idx); try v2 paths.
                    pass
            else:
                # ModelRunner V2: batch row → persistent state index → block_tables.
                got = _block_ids_from_v2_block_tables(
                    runner,
                    state_idx=_state_idx_from_batch(input_batch, idx),
                    kv_cache_group=kv_cache_group,
                )
                if got:
                    return got

    # Post-sample MRV2: no input_batch; resolve via req_states slot.
    return _block_ids_from_v2_block_tables(
        runner,
        state_idx=_state_idx_from_req_states(runner, req_id),
        kv_cache_group=kv_cache_group,
    )


def _runner_input_batch(runner: Any) -> Any | None:
    batch = getattr(runner, "input_batch", None)
    if batch is not None:
        return batch
    state = getattr(runner, "execute_model_state", None)
    return getattr(state, "input_batch", None) if state is not None else None


def _state_idx_from_batch(input_batch: Any, batch_idx: int) -> int | None:
    idx_mapping = getattr(input_batch, "idx_mapping_np", None)
    if idx_mapping is None:
        return None
    try:
        return int(idx_mapping[int(batch_idx)])
    except Exception:
        return None


def _state_idx_from_req_states(runner: Any, req_id: str) -> int | None:
    req_states = getattr(runner, "req_states", None)
    id_map = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if not isinstance(id_map, dict) or req_id not in id_map:
        return None
    try:
        return int(id_map[req_id])
    except (TypeError, ValueError):
        return None


def _block_ids_from_v2_block_tables(
    runner: Any,
    *,
    state_idx: int | None,
    kv_cache_group: int,
) -> list[int]:
    """Read one request's block row from MRV2 ``runner.block_tables``."""
    if state_idx is None:
        return []
    block_tables = getattr(runner, "block_tables", None)
    if block_tables is None:
        return []
    try:
        num_blocks = int(block_tables.num_blocks.np[kv_cache_group, state_idx])
        if num_blocks <= 0:
            return []
        # v2 rows live in StagedWriteTensor (``.gpu``, no host ``.np`` mirror):
        # prefer a host numpy mirror when present, else sync the device row.
        entry = block_tables.block_tables[kv_cache_group]
        host = getattr(entry, "np", None)
        if host is not None:
            row = host[state_idx, :num_blocks]
        else:
            row = entry.gpu[state_idx, :num_blocks].cpu()
        return [int(x) for x in row.tolist()]
    except Exception:
        return []


def _normalize_block_ids(raw: Any, *, kv_cache_group: int) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, tuple):
        if not raw or kv_cache_group >= len(raw):
            return []
        return [int(x) for x in raw[kv_cache_group]]
    if isinstance(raw, list):
        if not raw:
            return []
        if isinstance(raw[0], (list, tuple)):
            if kv_cache_group >= len(raw):
                return []
            return [int(x) for x in raw[kv_cache_group]]
        return [int(x) for x in raw]
    return []


def _block_table_for_group(input_batch: Any, kv_cache_group: int) -> Any | None:
    multi = getattr(input_batch, "block_table", None)
    if multi is None:
        return None
    tables = getattr(multi, "block_tables", None)
    if tables is not None:
        if kv_cache_group >= len(tables):
            return None
        return tables[kv_cache_group]
    try:
        return multi[kv_cache_group]
    except Exception:
        return multi if kv_cache_group == 0 else None


# ---- KV cache reader ----


def _pp_start_layer(runner: Any) -> int:
    """First global transformer layer index on this PP rank (0 if unknown)."""
    roots: list[Any] = [getattr(runner, "model", None)]
    getter = getattr(runner, "get_model", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            roots.append(getter())
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


def _append_tensor_or_parts(out: list[tuple[str, torch.Tensor]], base_name: str, val: Any) -> None:
    """Append a tensor, or expand list/tuple parts as ``base_name[i]``."""
    if isinstance(val, torch.Tensor):
        out.append((base_name, val))
    elif isinstance(val, (list, tuple)):
        for i, t in enumerate(val):
            if isinstance(t, torch.Tensor):
                out.append((f"{base_name}[{i}]", t))


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
            _append_tensor_or_parts(out, str(name), val)
    elif isinstance(kv_caches, list):
        for i, val in enumerate(kv_caches):
            layer_name = _list_layer_name(i, names=names, start_layer=start_layer)
            _append_tensor_or_parts(out, layer_name, val)
    return out


def _slice_blocks(tensor: torch.Tensor, block_ids: list[int]) -> tuple[torch.Tensor, list[int]] | None:
    """Select blocks along dim0, keeping ``[n_sel_blocks, block_size, ...]``.

    Returns ``None`` when ``block_ids`` is empty so callers refuse a full-cache
    copy (empty ids must not mean "dump everything"). Returns the payload
    together with the block ids actually used: out-of-range ids are dropped so
    metadata matches what was copied (not the caller's full request list).
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
        kv = getattr(self._runner, "kv_caches", None)
        if isinstance(kv, dict) and kv:
            return [("kv_caches", kv)]
        if isinstance(kv, list) and kv:
            return [("kv_caches", kv)]
        return []

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
            for _name, tensor in _iter_kv_tensors(kv_caches, list_names=names, start_layer=start):
                total += _block_payload_bytes(tensor, ids)
        return total

    def iter_request_snapshots(
        self,
        *,
        req_id: str,
        block_ids: list[int],
        out_dir: Path,
    ) -> Iterator[KvDumpSnapshot]:
        """Yield per-layer CPU snapshots for one request (D2H per layer).

        Only the request's ``block_ids`` are dumped (never the full KV pool).
        Yielding per layer lets callers enqueue each layer for async save as
        soon as it lands on host.
        """
        ids = list(block_ids)
        if not ids:
            logger.warning(
                "[runtime_guard dump_kv] skip req_id=%s: empty block_ids (refusing full-cache D2H)",
                req_id,
            )
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        produced = False
        # Record which shard produced the dump so cross-rank comparisons can
        # stitch TP heads / PP layers / CP token interleaves.
        try:
            tp_rank = runner_tp_rank(self._runner)
        except Exception:
            tp_rank = 0
        rank_tag = dump_rank_tag(self._runner)
        pp_rank = runner_pp_rank(self._runner)
        cp_rank = runner_cp_rank(self._runner)
        names, start = self._list_layer_meta()
        for src_name, kv_caches in self._kv_sources():
            for layer_name, tensor in _iter_kv_tensors(kv_caches, list_names=names, start_layer=start):
                num_kv_heads = int(tensor.shape[-2]) if tensor.dim() >= 3 else None
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
                if dump_root and req_id:
                    write_kv_dump_skipped(
                        dump_root,
                        req_id=req_id,
                        incident_type=str(payload.get("incident_type") or "unknown"),
                        reason="torch_save_failed",
                        stage="drain",
                        rank_tag=str(payload.get("rank_tag") or ""),
                        wave=parse_dump_arm_wave(payload.get("dump_arm_wave")),
                        detail={
                            "error": f"{type(exc).__name__}: {exc}",
                            "path": str(snap.path),
                            "layer": payload.get("layer"),
                        },
                    )
        if written:
            req_id = str(snapshots[0].payload.get("req_id") or "")
            logger.info(
                "[runtime_guard dump_kv] wrote req_id=%s pt_files=%d dir=%s",
                req_id,
                len(written),
                snapshots[0].path.parent,
            )
        return written
