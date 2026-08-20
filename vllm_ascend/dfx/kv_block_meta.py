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

"""Per-block KV write metadata for DFX reports (wave + last writer)."""

from __future__ import annotations

from contextlib import suppress
from typing import Any


def block_ids_for_request(
    runner: Any,
    req_id: str,
    req_idx: int | None = None,
    *,
    kv_cache_group: int = 0,
) -> list[int]:
    """Return logical GPU block ids for ``req_id`` (group 0 by default)."""
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

    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return []
    idx = req_idx
    if idx is None:
        mapping = getattr(input_batch, "req_id_to_index", None)
        if isinstance(mapping, dict) and req_id in mapping:
            idx = int(mapping[req_id])
        else:
            return []
    idx = int(idx)
    table = _block_table_for_group(input_batch, kv_cache_group)
    if table is None:
        return []
    try:
        num_blocks = int(table.num_blocks_per_row[idx])
    except Exception:
        return []
    if num_blocks <= 0:
        return []
    try:
        row = table.block_table.np[idx, :num_blocks]
        return [int(x) for x in row.tolist()]
    except Exception:
        return []


def touched_block_ids(
    block_ids: list[int],
    *,
    block_size: int,
    num_computed_before: int,
    num_scheduled: int,
) -> list[int]:
    """Block ids whose KV slots are written in ``[computed, computed+scheduled)``."""
    if not block_ids or num_scheduled <= 0 or block_size <= 0:
        return []
    start = max(0, int(num_computed_before))
    end = start + int(num_scheduled)
    first = start // int(block_size)
    last = (end - 1) // int(block_size)
    if first >= len(block_ids):
        return [block_ids[-1]] if block_ids else []
    last = min(last, len(block_ids) - 1)
    if last < first:
        return []
    return list(block_ids[first : last + 1])


def slot_mapping_for_request(
    runner: Any,
    req_id: str,
    req_idx: int | None = None,
    *,
    kv_cache_group: int = 0,
    scheduler_output: Any | None = None,
) -> tuple[list[int], tuple[int, int]] | None:
    """D2H this wave's GPU ``slot_mapping`` slice for ``req_id``.

    Returns ``(values, (start, end))`` in the packed batch, or ``None`` if the
    live tensor / query span cannot be resolved. Never raises.
    """
    if not req_id or runner is None:
        return None
    try:
        batch = _runner_input_batch(runner)
        idx = _resolve_req_idx(runner, batch, req_id, req_idx)
        if idx is None:
            return None
        span = _query_span(runner, batch, idx, scheduler_output)
        if span is None:
            return None
        start, end = span
        gpu = _slot_mapping_gpu(batch, kv_cache_group)
        if gpu is None:
            return None
        values = _d2h_int_list(gpu[start:end])
        return values, (start, end)
    except Exception:
        return None


def _runner_input_batch(runner: Any) -> Any | None:
    batch = getattr(runner, "input_batch", None)
    if batch is not None:
        return batch
    state = getattr(runner, "execute_model_state", None)
    return getattr(state, "input_batch", None) if state is not None else None


def _resolve_req_idx(
    runner: Any,
    batch: Any | None,
    req_id: str,
    req_idx: int | None,
) -> int | None:
    if req_idx is not None and int(req_idx) >= 0:
        return int(req_idx)
    mapping = getattr(batch, "req_id_to_index", None) if batch is not None else None
    if isinstance(mapping, dict) and req_id in mapping:
        return int(mapping[req_id])
    req_ids = getattr(batch, "req_ids", None) if batch is not None else None
    if req_ids:
        try:
            return list(req_ids).index(req_id)
        except ValueError:
            pass
    req_states = getattr(runner, "req_states", None)
    id_map = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if isinstance(id_map, dict) and req_id in id_map:
        return int(id_map[req_id])
    return None


def _query_span(
    runner: Any,
    batch: Any | None,
    req_idx: int,
    scheduler_output: Any | None,
) -> tuple[int, int] | None:
    for qsl in (
        getattr(runner, "query_start_loc", None),
        getattr(getattr(runner, "input_buffers", None), "query_start_loc", None),
        getattr(getattr(runner, "execute_model_state", None), "query_start_loc", None),
    ):
        span = _span_from_qsl(qsl, req_idx)
        if span is not None:
            return span
    return _span_from_scheduler(batch, req_idx, scheduler_output)


def _span_from_qsl(qsl: Any, req_idx: int) -> tuple[int, int] | None:
    if qsl is None:
        return None
    arr = getattr(qsl, "np", None)
    if arr is None:
        cpu = getattr(qsl, "cpu", None)
        arr = cpu if cpu is not None else qsl
    try:
        start = int(arr[req_idx].item() if hasattr(arr[req_idx], "item") else arr[req_idx])
        nxt = arr[req_idx + 1]
        end = int(nxt.item() if hasattr(nxt, "item") else nxt)
    except Exception:
        return None
    if 0 <= start < end:
        return start, end
    return None


def _span_from_scheduler(
    batch: Any | None,
    req_idx: int,
    scheduler_output: Any | None,
) -> tuple[int, int] | None:
    if batch is None or scheduler_output is None:
        return None
    req_ids = getattr(batch, "req_ids", None)
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not req_ids or not isinstance(num_scheduled, dict):
        return None
    if req_idx < 0 or req_idx >= len(req_ids):
        return None
    start = 0
    for i, rid in enumerate(req_ids):
        n = int(num_scheduled.get(rid, 0) or 0)
        if i == req_idx:
            return (start, start + n) if n > 0 else None
        start += max(n, 0)
    return None


def _slot_mapping_gpu(input_batch: Any, kv_cache_group: int) -> Any | None:
    multi = getattr(input_batch, "block_table", None)
    if multi is None:
        return None
    slots = getattr(multi, "slot_mappings", None)
    if slots is not None:
        try:
            if int(getattr(slots, "ndim", 1) or 1) >= 2:
                return slots[int(kv_cache_group)]
            return slots
        except Exception:
            return None
    table = _block_table_for_group(input_batch, kv_cache_group)
    if table is None:
        return None
    sm = getattr(table, "slot_mapping", None)
    if sm is None:
        return None
    return getattr(sm, "gpu", sm)


def _d2h_int_list(gpu_slice: Any) -> list[int]:
    """Blocking copy of a GPU (or CPU) 1-D integer tensor to ``list[int]``."""
    if gpu_slice is None:
        return []
    if isinstance(gpu_slice, (list, tuple)):
        return [int(x) for x in gpu_slice]
    t = gpu_slice
    detach = getattr(t, "detach", None)
    if callable(detach):
        t = detach()
    to_fn = getattr(t, "to", None)
    if callable(to_fn):
        try:
            t = to_fn("cpu")
        except Exception:
            cpu_fn = getattr(t, "cpu", None)
            t = cpu_fn() if callable(cpu_fn) else t
    elif callable(getattr(t, "cpu", None)):
        t = t.cpu()
    reshape = getattr(t, "reshape", None)
    if callable(reshape):
        with suppress(Exception):
            t = reshape(-1)
    if hasattr(t, "tolist"):
        raw = t.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = [x for row in raw for x in row]
        return [int(x) for x in raw]
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


class KvBlockMetaTracker:
    """Sparse per-block last-write wave and writer req_id (process-local)."""

    _instance: KvBlockMetaTracker | None = None

    def __init__(self) -> None:
        self._wave: dict[int, int] = {}
        self._writer: dict[int, str] = {}

    @classmethod
    def get(cls) -> KvBlockMetaTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    def record_writes(self, req_id: str, block_ids: list[int], wave: int) -> None:
        if not req_id or not block_ids:
            return
        w = int(wave)
        rid = str(req_id)
        for bid in block_ids:
            b = int(bid)
            self._wave[b] = w
            self._writer[b] = rid

    def last_write_wave(self, block_id: int) -> int | None:
        return self._wave.get(int(block_id))

    def last_writer_req_id(self, block_id: int) -> str | None:
        return self._writer.get(int(block_id))

    def blocks_detail(
        self,
        block_ids: list[int],
        *,
        include_wave: bool,
        include_writer: bool,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bid in block_ids:
            b = int(bid)
            entry: dict[str, Any] = {"block_id": b}
            if include_wave:
                entry["last_write_wave"] = self._wave.get(b)
            if include_writer:
                entry["last_writer_req_id"] = self._writer.get(b)
            out.append(entry)
        return out
