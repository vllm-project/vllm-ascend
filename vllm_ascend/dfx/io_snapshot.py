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

"""Request prompt/output token snapshots for DFX reports.

Detectors own anomaly metrics only. Full ``prompt_token_ids`` /
``output_token_ids`` are attached once at report time by
``RequestIoSnapshotManager`` (process-wide singleton) — not in model_runner
and not duplicated per detector.

Async scheduling leaves ``req_output_token_ids`` as ``-1`` placeholders unless
logits processors need real ids. DFX therefore **self-accumulates** accepted
output token ids on the detect rank (via :meth:`append_output`) and prefers
that cumulative list for report / counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.dfx.input_filters import prompt_token_ids_for_request


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
    """Length of cumulative output; prefer DFX self-built list when present."""
    mgr = RequestIoSnapshotManager.get()
    built = mgr.cumulative_output_count(req_id)
    if built > 0:
        return built
    raw = _raw_output_token_ids(runner, req_id, req_idx)
    if raw is None:
        return 0
    # Avoid counting async ``-1`` placeholders as real tokens.
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
    """Process-wide helper: accumulate outputs + snapshot I/O for report write.

    One copy per detect-rank process (last-PP TP0 under pending-OR) is enough;
    peers do not need a mirror.
    """

    _instance: RequestIoSnapshotManager | None = None

    def __init__(self) -> None:
        # Optional same-wave cache: cache_key → snapshot (cleared by processor).
        self._cache: dict[str, RequestIoSnapshot] = {}
        # Self-built cumulative accepted output ids (detect rank only).
        self._output_by_req: dict[str, list[int]] = {}
        # Last appended chunk per req (dedupe spec + token_logprob same step).
        self._last_append: dict[str, tuple[int, ...]] = {}

    @classmethod
    def get(cls) -> RequestIoSnapshotManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    def clear_wave_cache(self) -> None:
        self._cache.clear()

    def clear_req(self, req_id: str) -> None:
        """Drop cumulative output + wave cache entries for a finished request."""
        if not req_id:
            return
        self._output_by_req.pop(req_id, None)
        self._last_append.pop(req_id, None)
        stale = [k for k in self._cache if k.startswith(f"{req_id}|")]
        for key in stale:
            self._cache.pop(key, None)

    def cumulative_output_ids(self, req_id: str) -> list[int]:
        return list(self._output_by_req.get(req_id, ()))

    def cumulative_output_count(self, req_id: str) -> int:
        return len(self._output_by_req.get(req_id, ()))

    def append_output(self, req_id: str, token_ids: Any) -> None:
        """Extend cumulative output for ``req_id`` with accepted token ids.

        Drops ``-1`` placeholders. If ``token_ids`` equals the previous append
        for this req (spec + token_logprob recording the same step), skip.
        """
        if not req_id:
            return
        new_ids = _filter_valid_token_ids(token_ids)
        if not new_ids:
            return
        chunk = tuple(new_ids)
        if self._last_append.get(req_id) == chunk:
            return
        self._output_by_req.setdefault(req_id, []).extend(new_ids)
        self._last_append[req_id] = chunk

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

        Output prefers self-built cumulative ids (async-safe). Prompt is still
        read from the runner. ``include_token_ids`` controls whether full id
        lists are attached (``report.save_sensitive_info``).
        """
        if not req_id:
            return RequestIoSnapshot(req_id="", prompt_token_count=0, output_token_count=0)
        cache_key = f"{req_id}|{int(bool(include_token_ids))}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt_ids: list[int] | None = None
        built_output = self.cumulative_output_ids(req_id)
        if include_token_ids and runner is not None:
            raw_prompt = prompt_token_ids_for_request(runner, req_id, req_idx)
            prompt_ids = list(raw_prompt) if raw_prompt is not None else []
            prompt_count = len(prompt_ids)
            if built_output:
                output_ids = built_output
            else:
                # Fallback: runner list, stripping async placeholders.
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
