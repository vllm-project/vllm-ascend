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
