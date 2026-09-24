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

"""Request KV location helpers for reports (``block_ids``)."""

from __future__ import annotations

from typing import Any


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
