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
    manager_block_ids_by_group: list[list[int]] | None = None,
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
    if manager_block_ids_by_group is not None:
        payload["manager_block_ids_by_group"] = manager_block_ids_by_group
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
    manager_ids: bool = False,
) -> list[int]:
    """Return kernel-page block ids for ``req_id`` (group 0 by default).

    With ``manager_ids=True``, keep scheduler IDs for later per-layer expansion.

    Id space contract (Ascend / Runtime Guard dump is MRV2-oriented):

    - ``requests[].block_ids`` (mainly ``_310p``; absent on pinned main MRV2)
      are scheduler/manager ids and are expanded by ``blocks_per_kv_block``
      before return.
    - ``runner.block_tables``: kernel ids when the table exposes
      ``blocks_per_kv_block`` (_310p); otherwise manager ids expanded here
      (current main keeps public tables at manager granularity).

    The MRV1 ``input_batch`` host-mirror table path is intentionally omitted so
    callers never silently receive manager-granularity ids.

    MRV2 clears ``execute_model_state`` at the start of ``sample_tokens``, so
    end-of-wave dump often has no live ``input_batch``. Fall back to persistent
    ``req_states`` + ``runner.block_tables`` (BUG-4).
    """
    if not req_id:
        return []

    requests = getattr(runner, "requests", None)
    if requests is not None:
        state = requests.get(req_id)
        if state is not None:
            raw = getattr(state, "block_ids", None)
            parsed = _normalize_block_ids(raw, kv_cache_group=kv_cache_group)
            if parsed:
                if manager_ids:
                    return parsed
                # ``requests[].block_ids`` are scheduler/manager ids. Ascend
                # kernel-paged caches use dim0 = manager_id * R + k; expand here.
                # ``runner.block_tables`` path expands only when tables are
                # still manager-granularity (no ``blocks_per_kv_block`` attr).
                ratio, _unknown = _blocks_per_kv_block_for_group(runner, kv_cache_group)
                return expand_manager_block_ids(parsed, ratio)

    if input_batch is None:
        # V2: runner.input_batch stays None; prefer execute_model_state batch
        # while it still exists (pre-sample / mid-execute).
        input_batch = _runner_input_batch(runner)

    if input_batch is not None:
        idx = req_idx
        # Negative markers mean "unknown"; look up by req_id.
        if idx is not None and int(idx) < 0:
            idx = None
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
            # ModelRunner V2: batch row → persistent state index → block_tables
            # (already kernel-addressed). No MRV1 host-mirror table read.
            got = _block_ids_from_v2_block_tables(
                runner,
                state_idx=_state_idx_from_batch(input_batch, int(idx)),
                manager_ids=manager_ids,
                kv_cache_group=kv_cache_group,
            )
            if got:
                return got

    # Post-sample MRV2: no input_batch; resolve via req_states slot.
    return _block_ids_from_v2_block_tables(
        runner,
        state_idx=_state_idx_from_req_states(runner, req_id),
        manager_ids=manager_ids,
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
    manager_ids: bool = False,
) -> list[int]:
    """Read one request's block row from MRV2 ``runner.block_tables``.

    ``_310p`` tables store already-expanded kernel ids (and expose
    ``blocks_per_kv_block``). Current main keeps manager-granularity rows, so
    expand with the same R used for ``requests[].block_ids``.
    """
    if state_idx is None:
        return []
    block_tables = getattr(runner, "block_tables", None)
    if block_tables is None:
        return []
    try:
        # 310P owns expanded rows on CPU. Other MRV2 tables may only expose a device row.
        cpu_tables = getattr(block_tables, "block_tables_cpu", None)
        if cpu_tables is not None:
            count = int(block_tables.num_blocks_np[kv_cache_group, state_idx])
            row = cpu_tables[kv_cache_group][state_idx, :count]
        else:
            count = int(block_tables.num_blocks.np[kv_cache_group, state_idx])
            if count <= 0:
                return []
            entry = block_tables.block_tables[kv_cache_group]
            host = getattr(entry, "np", None)
            row = host[state_idx, :count] if host is not None else entry.gpu[state_idx, :count].cpu()
        ids = [int(x) for x in row.tolist()]
    except Exception:
        return []
    # Attribute present ⇒ table already kernel-expanded (_310p / older paths).
    if getattr(block_tables, "blocks_per_kv_block", None) is not None:
        return _manager_ids_from_pages(runner, ids, kv_cache_group) if manager_ids else ids
    if manager_ids:
        return ids
    ratio, _unknown = _blocks_per_kv_block_for_group(runner, kv_cache_group)
    return expand_manager_block_ids(ids, ratio)


def _manager_ids_from_pages(runner: Any, pages: list[int], gid: int) -> list[int]:
    ratio, _ = _blocks_per_kv_block_for_group(runner, gid)
    return [page // ratio for page in pages[::ratio]] if ratio > 1 else list(pages)


def group0_pages(runner: Any, rows: list[list[int]]) -> list[int]:
    """Keep the existing group-0 kernel-page field for reports and legacy readers."""
    if not rows or not rows[0]:
        return []
    ratio, _ = _blocks_per_kv_block_for_group(runner, 0)
    return expand_manager_block_ids(rows[0], ratio)


def request_manager_blocks(
    runner: Any, req_id: str, req_idx: int | None = None, *, group0_pages: list[int] | None = None
) -> list[list[int]]:
    """Freeze one manager row per group. A supplied group-0 row is already in kernel pages."""
    groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None)
    group0 = (
        _manager_ids_from_pages(runner, group0_pages, 0)
        if group0_pages
        else block_ids_for_request(runner, req_id, req_idx, manager_ids=True)
    )
    return [group0] + [
        block_ids_for_request(runner, req_id, req_idx, kv_cache_group=gid, manager_ids=True)
        for gid in range(1, len(groups or ()))
    ]


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
        return [int(x) for x in raw] if kv_cache_group == 0 else []
    return []


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


def _layer_name_to_kv_cache_group(runner: Any) -> dict[str, int]:
    """Map logical layer name → ``kv_cache_groups`` index."""
    cfg = getattr(runner, "kv_cache_config", None)
    groups = getattr(cfg, "kv_cache_groups", None) if cfg is not None else None
    if not groups:
        return {}
    out: dict[str, int] = {}
    for gi, group in enumerate(groups):
        for name in getattr(group, "layer_names", None) or []:
            out[str(name)] = int(gi)
    return out


def _logical_layer_name(layer_name: str) -> str:
    """Strip dump part suffix ``name[i]`` back to the config layer name."""
    if layer_name.endswith("]") and "[" in layer_name:
        return layer_name[: layer_name.rfind("[")]
    return layer_name


def _group_for_layer(layer_name: str, layer_to_group: dict[str, int]) -> int | None:
    """Return the layer's ``kv_cache_group`` index, or ``None`` if unknown.

    Distinguishes ``group 0`` from ``not in map`` so dump never silently
    reuses group-0 page lists for unresolved layers.
    """
    logical = _logical_layer_name(layer_name)
    if logical not in layer_to_group:
        return None
    return int(layer_to_group[logical])


def _warn_underivable_blocks_per_kv(kv_cache_group: int, *, reason: str) -> None:
    logger.warning_once(
        "[runtime_guard dump_kv] group=%s blocks_per_kv_block underivable (%s); "
        "falling back to R=1 — dump pages may be wrong for compressed/C8 layouts",
        kv_cache_group,
        reason,
    )


def _derive_blocks_per_kv_block(runner: Any, kv_cache_group: int, layer_name: str | None = None) -> tuple[int, bool]:
    """Match the physical rows per manager block to the rows in a kernel page.

    Public ``AscendBlockTables`` on main no longer carries
    ``blocks_per_kv_block``; dump tensors are allocated with the *storage*
    ratio in ``attn_utils``, which differs from logical
    ``block_size // kernel_block_size`` for compressed MLA / C8 groups.

    Returns ``(ratio, expansion_unknown)``. ``expansion_unknown`` is True when
    we fall back to ``R=1`` despite evidence that expansion may be required
    (silent R=1 would reintroduce wrong-page dumps).
    """
    cfg = getattr(runner, "kv_cache_config", None)
    groups = getattr(cfg, "kv_cache_groups", None) if cfg is not None else None
    if not groups or kv_cache_group < 0 or kv_cache_group >= len(groups):
        return 1, False
    group = groups[kv_cache_group]
    spec = getattr(group, "kv_cache_spec", None)
    if layer_name is None:
        layer_name = next(iter(getattr(group, "layer_names", ()) or ()), None)
    if layer_name is not None:
        spec = getattr(spec, "kv_cache_specs", {}).get(layer_name, spec)
    if spec is None:
        _warn_underivable_blocks_per_kv(kv_cache_group, reason="missing kv_cache_spec")
        return 1, True
    try:
        storage = int(spec.num_states)
    except Exception:
        _warn_underivable_blocks_per_kv(kv_cache_group, reason="num_states unavailable")
        return 1, True
    kbs: int | None = None
    sizes = getattr(runner, "kernel_block_sizes", None)
    if sizes is not None:
        try:
            kbs = int(sizes[kv_cache_group])
        except (IndexError, TypeError, ValueError):
            kbs = None
    # The bound attention backend may override the group's default kernel size.
    target_groups = getattr(runner, "attn_groups", None) or ()
    draft_groups = getattr(getattr(runner, "speculator", None), "attn_groups", None) or ()
    for groups in (*target_groups, *draft_groups):
        for ag in groups:
            if ag.kv_cache_group_id == kv_cache_group and (layer_name is None or layer_name in ag.layer_names):
                kbs = ag.kernel_block_size or kbs
                break
        else:
            continue
        break
    if not kbs or kbs <= 0:
        # Compressed specs often have storage != logical block_size; missing
        # kernel size then makes R underivable.
        block_size = int(getattr(spec, "block_size", 0) or 0)
        if storage > 0 and block_size > 0 and storage != block_size:
            _warn_underivable_blocks_per_kv(
                kv_cache_group,
                reason="missing kernel_block_size for compressed spec",
            )
            return 1, True
        return 1, False
    if getattr(spec, "indexes_kv_by_block_stride", False):
        # Both sizes must count physical rows (compressed keys or a complete tail ring).
        kernel_states = getattr(spec, "get_num_kernel_states", None)
        if kernel_states is None:
            _warn_underivable_blocks_per_kv(kv_cache_group, reason="missing kernel-state layout")
            return 1, True
        kbs = int(kernel_states(kbs))
    if kbs <= 0 or storage <= 0 or storage % kbs:
        _warn_underivable_blocks_per_kv(
            kv_cache_group,
            reason=f"storage={storage} not divisible by kernel={kbs}",
        )
        return 1, True
    return max(1, storage // kbs), False


def _blocks_per_kv_block_for_group(runner: Any, kv_cache_group: int, layer_name: str | None = None) -> tuple[int, bool]:
    """Kernel pages per manager block for ``kv_cache_group``.

    Returns ``(ratio, expansion_unknown)``.

    Prefer ``runner.block_tables.blocks_per_kv_block`` when present (_310p /
    tests):

    - **Scalar** (Python ``int``/``float`` or numpy/tensor scalar): one R
      **shared by all groups**.
    - **Sequence**: per-group ``ratios[kv_cache_group]``.
    - A scalar must not be treated as ``R=1`` via a failed subscript.

    When the attribute is missing (current main ``AscendBlockTables``), derive
    R from storage block size so dump indexing matches tensor dim0 for
    compressed / C8 groups. Underivable paths warn once and set
    ``expansion_unknown``.
    """
    block_tables = getattr(runner, "block_tables", None)
    ratios = getattr(block_tables, "blocks_per_kv_block", None) if block_tables is not None else None
    if ratios is None:
        return _derive_blocks_per_kv_block(runner, kv_cache_group, layer_name)
    # Scalar ratio shared by all groups (Python int/float; not bool).
    if isinstance(ratios, (int, float)) and not isinstance(ratios, bool):
        return max(1, int(ratios)), False
    # Numpy / tensor 0-d scalars are subscriptable but must share one R
    # (do not treat ratios[group] IndexError as missing expansion).
    if getattr(ratios, "ndim", None) == 0:
        try:
            return max(1, int(ratios)), False
        except (TypeError, ValueError):
            _warn_underivable_blocks_per_kv(
                kv_cache_group,
                reason="0-d blocks_per_kv_block not convertible to int",
            )
            return 1, True
    try:
        return max(1, int(ratios[kv_cache_group])), False
    except IndexError:
        _warn_underivable_blocks_per_kv(
            kv_cache_group,
            reason="blocks_per_kv_block index out of range",
        )
        return 1, True
    except (TypeError, ValueError):
        try:
            return max(1, int(ratios)), False
        except (TypeError, ValueError):
            _warn_underivable_blocks_per_kv(
                kv_cache_group,
                reason="blocks_per_kv_block not convertible to int",
            )
            return 1, True


def expand_manager_block_ids(block_ids: list[int], blocks_per_kv_block: int) -> list[int]:
    """Expand scheduler/manager block ids to kernel dim0 indices."""
    ratio = max(1, int(blocks_per_kv_block))
    if ratio == 1:
        return [int(b) for b in block_ids]
    out: list[int] = []
    for bid in block_ids:
        base = int(bid) * ratio
        out.extend(range(base, base + ratio))
    return out


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
        """Prefer dict/tuple KV (preserves C8/TQ4 part names) over flattened list.

        Ascend binds a flattened ``runner.kv_caches`` list for attention kernels.
        ``compilation_config.static_forward_context[layer].kv_cache`` still holds
        the original tensor or ``(k, v[, scale…])`` tuple per logical layer.
        """
        runner = self._runner
        ctx = None
        try:
            cfg = getattr(runner, "compilation_config", None)
            ctx = getattr(cfg, "static_forward_context", None) if cfg is not None else None
        except Exception:
            ctx = None
        if isinstance(ctx, dict) and ctx:
            rebuilt: dict[str, Any] = {}
            for layer_name, module in ctx.items():
                kv = getattr(module, "kv_cache", None)
                if kv is None:
                    continue
                # Some bindings store a list of virtual-engine caches; take [0].
                if isinstance(kv, list) and kv and not isinstance(kv[0], torch.Tensor):
                    # list of tuples / tensors per VE
                    first = kv[0] if kv else None
                    if first is not None:
                        rebuilt[str(layer_name)] = first
                elif isinstance(kv, list) and kv and isinstance(kv[0], torch.Tensor):
                    rebuilt[str(layer_name)] = tuple(kv) if len(kv) > 1 else kv[0]
                else:
                    rebuilt[str(layer_name)] = kv
            if rebuilt:
                flat = getattr(runner, "kv_caches", None)
                flat_n = len(flat) if isinstance(flat, (list, dict)) else 0
                ctx_n = len(rebuilt)
                # Flattened bind may pack C8 parts as separate list entries, so
                # unequal counts are expected for some layouts — still surface
                # the coverage change so a silent layer drop is visible.
                if flat_n and flat_n != ctx_n:
                    logger.warning_once(
                        "[runtime_guard dump_kv] static_forward_context has %s "
                        "layers with kv_cache; runner.kv_caches has %s entries — "
                        "dump uses forward_context only (layers missing from "
                        "context will be omitted)",
                        ctx_n,
                        flat_n,
                    )
                return [("static_forward_context", rebuilt)]

        kv = getattr(runner, "kv_caches", None)
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

    def _request_tensors(self, rows: list[list[int]]) -> Iterator[tuple[str, str, torch.Tensor, int, list[int], bool]]:
        """Resolve physical pages for each bound tensor from the frozen manager rows."""
        names, start = self._list_layer_meta()
        layer_to_group = _layer_name_to_kv_cache_group(self._runner)
        pages: dict[tuple[int, int], list[int]] = {}
        for source, caches in self._kv_sources():
            for layer, tensor in _iter_kv_tensors(caches, list_names=names, start_layer=start):
                gid = _group_for_layer(layer, layer_to_group)
                if gid is None:
                    logger.warning("[runtime_guard dump_kv] skip layer=%s: kv_cache_group unresolved", layer)
                    continue
                if gid >= len(rows) or not rows[gid]:
                    logger.warning_once(
                        "[runtime_guard dump_kv] skip layer=%s: no saved blocks for group %s", layer, gid
                    )
                    continue
                ratio, unknown = _blocks_per_kv_block_for_group(self._runner, gid, _logical_layer_name(layer))
                key = (gid, ratio)
                if key not in pages:
                    pages[key] = expand_manager_block_ids(rows[gid], ratio)
                yield source, layer, tensor, gid, pages[key], unknown

    def estimate_dump_bytes(self, *, manager_block_ids_by_group: list[list[int]]) -> int:
        """Count the same tensor pages that the reader will copy, without D2H."""
        return sum(
            _block_payload_bytes(tensor, ids)
            for _, _, tensor, _, ids, _ in self._request_tensors(manager_block_ids_by_group)
        )

    def iter_request_snapshots(
        self,
        *,
        req_id: str,
        manager_block_ids_by_group: list[list[int]],
        out_dir: Path,
    ) -> Iterator[KvDumpSnapshot]:
        """Copy this request's frozen manager blocks using each layer's physical layout."""
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
        for src_name, layer_name, tensor, gid, ids, expansion_unknown in self._request_tensors(
            manager_block_ids_by_group
        ):
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
            payload: dict[str, Any] = {
                "req_id": req_id,
                "block_ids": used_ids,
                "layer": layer_name,
                "kv_cache_group": int(gid),
                "source": src_name,
                "rank_tag": rank_tag,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "cp_rank": cp_rank,
                "num_kv_heads": num_kv_heads,
                "tensor": payload_tensor,
            }
            if expansion_unknown:
                payload["expansion"] = "unknown"
            yield KvDumpSnapshot(
                path=path,
                payload=payload,
            )
        if not produced:
            logger.warning("[runtime_guard dump_kv] no matching KV pages for req_id=%s", req_id)

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
                manager_block_ids_by_group=request_manager_blocks(self._runner, req_id, group0_pages=block_ids),
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
