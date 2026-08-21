#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.config_backed import ConfigBackedDetector
from vllm_ascend.dfx.dfx_types import ILL_TYPE_REPEAT
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager, normalize_token_ids
from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def normalize_ignore_token_ids(raw: Any) -> list[int]:
    """Validate config ``ignore_token_ids`` as a flat list of ints."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"ignore_token_ids must be a list of ints, got {type(raw).__name__}")
    out: list[int] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"ignore_token_ids[{i}] must be int, got {item!r}")
        out.append(int(item))
    return out


@dataclass
class TokenRepeatState:
    """Per-request sliding window for local token re-reading detection."""

    content: deque[int] = field(default_factory=deque)
    scores: deque[int] = field(default_factory=deque)
    freq: dict[int, int] = field(default_factory=dict)
    repeat_sum: int = 0
    content_tokens_seen: int = 0
    consecutive_hits: int = 0


def push_token_repeat(
    state: TokenRepeatState,
    tid: int,
    *,
    window: int,
    ignore: frozenset[int] | set[int],
) -> int:
    """Fold one token into the sliding window; return this step's score.

    Score = how many times ``tid`` already appears in the prior content window
    (ignored tokens score 0 and are not stored). ``repeat_sum`` is the sum of
    the last ``window`` scores.
    """
    if tid in ignore:
        score = 0
    else:
        score = int(state.freq.get(tid, 0))
        state.content.append(tid)
        state.freq[tid] = score + 1
        state.content_tokens_seen += 1
        while len(state.content) > window:
            old = state.content.popleft()
            left = state.freq.get(old, 0) - 1
            if left <= 0:
                state.freq.pop(old, None)
            else:
                state.freq[old] = left

    state.scores.append(score)
    state.repeat_sum += score
    while len(state.scores) > window:
        state.repeat_sum -= state.scores.popleft()
    return score


class TokenRepeatDetector(ConfigBackedDetector):
    """Detect local token re-reading via sliding-window ``repeat_sum``.

    Consumes the same cumulative output stream as ``OutputSubstringDetector``
    (``RequestIoSnapshotManager``), including tokens recorded by
    ``check_after_spec``. A per-req cursor tracks how far the stream has been
    folded into the repeat window so each id is pushed once.
    """

    anomaly_type = "token_repeat"
    section_key = "token_repeat"

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=False)
        self._window = 32
        self._repeat_sum_threshold = 64
        self._min_tokens = 32
        self._consecutive_hits_thresh = 1
        self._ignore_token_ids: frozenset[int] = frozenset()
        self._states: dict[str, TokenRepeatState] = {}
        # How many cumulative output ids have already been pushed for each req.
        self._consumed_len: dict[str, int] = {}
        self._alerted: set[str] = set()
        if dfx_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        new_window = max(1, int(getter("window", self._window)))
        if new_window != self._window and self._states:
            # Shrinking/growing the window invalidates in-flight sliding windows.
            self._states.clear()
            self._consumed_len.clear()
        self._window = new_window
        self._repeat_sum_threshold = max(0, int(getter("repeat_sum_threshold", self._repeat_sum_threshold)))
        self._min_tokens = max(0, int(getter("min_tokens", self._min_tokens)))
        self._consecutive_hits_thresh = max(1, int(getter("consecutive_hits", self._consecutive_hits_thresh)))
        try:
            ignore = normalize_ignore_token_ids(getter("ignore_token_ids", []))
        except ValueError as exc:
            logger.error("[Anomaly token_repeat] invalid ignore_token_ids: %s; keeping previous", exc)
            return
        self._ignore_token_ids = frozenset(ignore)

    def clear_finished(self, req_id: str) -> None:
        self._states.pop(req_id, None)
        self._consumed_len.pop(req_id, None)
        self._alerted.discard(req_id)

    def check_all(
        self,
        sampled_token_ids: list[list[int]] | None,
        req_ids: list[str] | None = None,
        skip_req_ids: set[str] | None = None,
    ) -> list[AnomalyAlert]:
        """Fold new cumulative output ids into per-req windows.

        Prefer ``sampled_token_ids=None`` after
        ``DetectorManager.check_after_sample`` / ``check_after_spec`` have
        already written into the IO buffer (same path as substring). When
        provided (standalone callers / unit tests), tokens are appended here
        first so the stream matches substring.
        """
        if not self._precheck():
            return []

        runner = self._runner
        log_leader = int(getattr(runner, "tp_rank", 0) if runner is not None else 0) == 0
        if req_ids is None:
            input_batch = getattr(runner, "input_batch", None) if runner is not None else None
            req_ids = list(getattr(input_batch, "req_ids", None) or [])
        if not req_ids:
            return []

        io_mgr = RequestIoSnapshotManager.get()
        if sampled_token_ids is not None:
            io_mgr.append_batch(req_ids, sampled_token_ids)

        alerts: list[AnomalyAlert] = []
        for batch_idx, req_id in enumerate(req_ids):
            if not req_id:
                continue
            cumulative = io_mgr.cumulative_output_ids(req_id)
            total = len(cumulative)
            consumed = self._consumed_len.get(req_id, 0)
            if consumed > total:
                # Buffer was reset / replaced; restart from current stream.
                consumed = 0
                self._states.pop(req_id, None)
            new_ids = cumulative[consumed:]
            # Advance cursor even when skipping/alerted so we never re-push.
            self._consumed_len[req_id] = total

            if req_id in self._alerted:
                continue
            if skip_req_ids and req_id in skip_req_ids:
                continue
            if not new_ids:
                continue
            if not self._passes_input_filter(req_id, batch_idx):
                continue
            alert = self.check_one(
                req_idx=batch_idx,
                req_id=req_id,
                token_ids=new_ids,
                log_leader=log_leader,
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def check_one(
        self,
        req_idx: int,
        req_id: str,
        token_ids: Iterable[int],
        *,
        log_leader: bool = False,
    ) -> AnomalyAlert | None:
        state = self._states.get(req_id)
        if state is None:
            state = TokenRepeatState()
            self._states[req_id] = state

        ids_list = normalize_token_ids(token_ids)
        window = self._window
        thresh = self._repeat_sum_threshold
        ignore = self._ignore_token_ids
        hit = False
        last_score = 0
        for tid in ids_list:
            last_score = push_token_repeat(state, int(tid), window=window, ignore=ignore)
            warmed = state.content_tokens_seen >= self._min_tokens
            over = warmed and state.repeat_sum > thresh
            if over:
                state.consecutive_hits += 1
            else:
                state.consecutive_hits = 0
            if state.consecutive_hits >= self._consecutive_hits_thresh:
                hit = True
                # Keep pushing remaining ids so window state matches the full
                # cumulative delta already committed via _consumed_len.

        if not hit:
            return None

        self._alerted.add(req_id)
        detail = {
            "repeat_sum": state.repeat_sum,
            "repeat_sum_threshold": thresh,
            "window": window,
            "content_tokens_seen": state.content_tokens_seen,
            "last_score": last_score,
            "consecutive_hits": state.consecutive_hits,
            "chunk_len": len(ids_list),
            "recent_token_ids": ids_list[-min(32, len(ids_list)) :],
        }
        if log_leader:
            logger.error(
                "[Anomaly token_repeat] req=%s repeat_sum=%s threshold=%s window=%s content_seen=%s consecutive=%s",
                req_id,
                state.repeat_sum,
                thresh,
                window,
                state.content_tokens_seen,
                state.consecutive_hits,
            )
        return AnomalyAlert(
            anomaly_type=self.anomaly_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=ILL_TYPE_REPEAT,
            detail=detail,
        )
