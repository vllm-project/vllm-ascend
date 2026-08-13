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

"""Report-time I/O views over :class:`RequestDfxStore` + runner prompts.

Ownership split:

- **Data** (cumulative ``output_token_ids``, append dedupe): ``RequestDfxStore``
- **View** (this module): normalize token rows, ``snapshot()`` for reports,
  same-wave snapshot cache only

Detectors / ``DetectorManager`` call :meth:`append_output` /
:meth:`append_batch` as a thin normalize→Store adapter. They must not treat
this manager as a second per-req state owner — finish cleanup is
``RequestDfxStore.clear`` only.

Async scheduling leaves runner ``req_output_token_ids`` as ``-1`` placeholders
unless logits processors need real ids; DFX therefore prefers Store-built
cumulative ids for counts and report bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.dfx.input_filters import prompt_token_ids_for_request
from vllm_ascend.dfx.request_state import RequestDfxStore


def normalize_token_ids(token_ids: Any) -> list[int]:
    """Normalize tensor / nested tensors / sequences to ``list[int]``."""
    if token_ids is None:
        return []
    if torch.is_tensor(token_ids):
        return [int(x) for x in token_ids.tolist()]
    out: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, torch.Tensor):
            out.append(int(token_id.item()))
        else:
            out.append(int(token_id))
    return out


def _filter_valid_token_ids(token_ids: Any) -> list[int]:
    """Normalize and drop async / pad placeholders (``-1``)."""
    return [tid for tid in normalize_token_ids(token_ids) if tid != -1]


def _raw_output_token_ids(runner: Any, req_id: str, req_idx: int | None) -> Any:
    if runner is None:
        return None
    input_batch = getattr(runner, "input_batch", None)
    req_output_token_ids = getattr(input_batch, "req_output_token_ids", None) if input_batch else None
    if req_output_token_ids is not None and req_idx is not None and 0 <= int(req_idx) < len(req_output_token_ids):
        return req_output_token_ids[int(req_idx)]
    requests = getattr(runner, "requests", None)
    req_state = requests.get(req_id) if requests is not None else None
    if req_state is not None:
        return getattr(req_state, "output_token_ids", None)
    return None


def output_token_count_for_request(runner: Any, req_id: str, req_idx: int | None = None) -> int:
    """Length of cumulative output; prefer DFX Store list when present."""
    built = RequestDfxStore.get().cumulative_output_count(req_id)
    if built > 0:
        return built
    raw = _raw_output_token_ids(runner, req_id, req_idx)
    if raw is None:
        return 0
    try:
        ids = normalize_token_ids(raw)
    except TypeError:
        return 0
    return sum(1 for tid in ids if tid != -1)


def prompt_token_count_for_request(runner: Any, req_id: str, req_idx: int | None = None) -> int:
    ids = prompt_token_ids_for_request(runner, req_id, req_idx) if runner is not None else None
    return len(ids) if ids is not None else 0


@dataclass(frozen=True)
class RequestIoSnapshot:
    """Prompt / cumulative-output view for one request at report time."""

    req_id: str
    prompt_token_count: int
    output_token_count: int
    prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] | None = None

    def as_detail_fields(self) -> dict[str, Any]:
        """Fields merged into anomaly report detail (counts always; ids optional)."""
        out: dict[str, Any] = {
            "prompt_token_count": self.prompt_token_count,
            "output_token_count": self.output_token_count,
        }
        if self.prompt_token_ids is not None:
            out["prompt_token_ids"] = self.prompt_token_ids
        if self.output_token_ids is not None:
            out["output_token_ids"] = self.output_token_ids
        return out


class RequestIoSnapshotManager:
    """Report I/O view helper (not a second per-req state owner).

    - Accumulates via Store (``append_*`` = normalize + ``RequestDfxStore``).
    - Builds :class:`RequestIoSnapshot` for anomaly / dump_finish reports.
    - Keeps a same-wave snapshot cache only; cleared each ``clear_wave_cache``
      and on Store.clear via registered hook.
    """

    _instance: RequestIoSnapshotManager | None = None

    def __init__(self) -> None:
        self._cache: dict[str, RequestIoSnapshot] = {}
        RequestDfxStore.get().register_on_clear(self.clear_req_cache)

    @classmethod
    def get(cls) -> RequestIoSnapshotManager:
        if cls._instance is None:
            cls._instance = cls()
        else:
            # Store may have been reset_for_tests while this singleton survived.
            RequestDfxStore.get().register_on_clear(cls._instance.clear_req_cache)
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
        RequestDfxStore.reset_for_tests()

    def clear_wave_cache(self) -> None:
        """Drop snapshot cache and same-wave append dedupe on Store.

        Called at the start of each ``sync_for_step`` / ``refresh_config`` wave
        so content-identical chunks from a later step are not swallowed.
        """
        self._cache.clear()
        RequestDfxStore.get().clear_wave_append_frontier()

    def clear_req_cache(self, req_id: str) -> None:
        """Drop wave snapshot cache entries for ``req_id`` (Store on_clear hook)."""
        if not req_id:
            return
        prefix = f"{req_id}|"
        stale = [k for k in self._cache if k.startswith(prefix)]
        for key in stale:
            self._cache.pop(key, None)

    def clear_req(self, req_id: str) -> None:
        """Compatibility: finish cleanup via :meth:`RequestDfxStore.clear`."""
        if not req_id:
            return
        RequestDfxStore.get().clear(str(req_id))

    def cumulative_output_ids(self, req_id: str) -> list[int]:
        return RequestDfxStore.get().cumulative_output_ids(req_id)

    def cumulative_output_count(self, req_id: str) -> int:
        return RequestDfxStore.get().cumulative_output_count(req_id)

    def append_output(self, req_id: str, token_ids: Any) -> None:
        """Normalize accepted ids and append into :class:`RequestDfxStore`.

        Drops ``-1`` placeholders. Within one engine wave, duplicate chunks
        for the same req are skipped (Store ``last_append_chunk``). Across
        waves, identical chunks are kept after :meth:`clear_wave_cache`.
        """
        if not req_id:
            return
        new_ids = _filter_valid_token_ids(token_ids)
        if not new_ids:
            return
        RequestDfxStore.get().append_output_ids(req_id, new_ids)

    def append_batch(
        self,
        req_ids: list[str] | None,
        sampled_rows: Any,
    ) -> None:
        """Append one step of per-req sampled / accepted token rows."""
        if not req_ids or sampled_rows is None:
            return
        for i, req_id in enumerate(req_ids):
            if not req_id:
                continue
            try:
                row = sampled_rows[i]
            except (IndexError, TypeError, KeyError):
                continue
            self.append_output(req_id, row)

    def snapshot(
        self,
        runner: Any,
        req_id: str,
        req_idx: int | None = None,
        *,
        include_token_ids: bool = False,
        use_cache: bool = True,
    ) -> RequestIoSnapshot:
        """Build prompt + cumulative-output view for ``req_id``.

        Output prefers Store cumulative ids (async-safe). Prompt is read from
        the runner. ``include_token_ids`` attaches full id lists when
        ``report.save_sensitive_info`` is on.
        """
        if not req_id:
            return RequestIoSnapshot(req_id="", prompt_token_count=0, output_token_count=0)
        cache_key = f"{req_id}|{int(bool(include_token_ids))}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt_ids: list[int] | None = None
        store = RequestDfxStore.get()
        built_output = store.cumulative_output_ids(req_id)
        if include_token_ids and runner is not None:
            raw_prompt = prompt_token_ids_for_request(runner, req_id, req_idx)
            prompt_ids = list(raw_prompt) if raw_prompt is not None else []
            prompt_count = len(prompt_ids)
            if built_output:
                output_ids = built_output
            else:
                output_ids = _filter_valid_token_ids(_raw_output_token_ids(runner, req_id, req_idx))
            output_count = len(output_ids)
        else:
            prompt_count = prompt_token_count_for_request(runner, req_id, req_idx)
            output_count = (
                len(built_output) if built_output else output_token_count_for_request(runner, req_id, req_idx)
            )
            output_ids = None

        snap = RequestIoSnapshot(
            req_id=req_id,
            prompt_token_count=prompt_count,
            output_token_count=output_count,
            prompt_token_ids=prompt_ids,
            output_token_ids=output_ids if include_token_ids else None,
        )
        if use_cache:
            self._cache[cache_key] = snap
        return snap

    def merge_into_detail(
        self,
        detail: dict[str, Any] | None,
        snapshot: RequestIoSnapshot,
    ) -> dict[str, Any]:
        """Overlay I/O fields onto detector detail (I/O wins on key clash)."""
        out = dict(detail or {})
        out.update(snapshot.as_detail_fields())
        return out
