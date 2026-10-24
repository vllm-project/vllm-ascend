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
from typing import TYPE_CHECKING, Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config.schema import ConfigField, DetectorSchema, coerce_list_int
from vllm_ascend.observability.runtime_guard.detector.base import ConfigBackedDetector, resolve_batch_req_ids
from vllm_ascend.observability.runtime_guard.io import RequestIoSnapshotManager, normalize_token_ids
from vllm_ascend.observability.runtime_guard.rank_gate import runner_tp_rank
from vllm_ascend.observability.runtime_guard.state import ILL_TYPE_REPEAT, Incident

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


def _ids_already_ints(token_ids: Iterable[int]) -> list[int]:
    """Skip ``normalize_token_ids`` when Store already handed ``list[int]``."""
    if isinstance(token_ids, list):
        if not token_ids or isinstance(token_ids[0], int):
            return token_ids
    return normalize_token_ids(token_ids)


@dataclass
class TokenRepeatWindow:
    """One sliding window: its own content, scores, freq and counters."""

    window: int
    threshold: int = 0
    min_tokens: int = 0
    content: deque[int] = field(default_factory=deque)
    scores: deque[int] = field(default_factory=deque)
    freq: dict[int, int] = field(default_factory=dict)
    repeat_sum: int = 0
    content_tokens_seen: int = 0
    consecutive_hits: int = 0


@dataclass
class TokenRepeatState:
    """Per-request state: several differently-sized sliding windows."""

    windows: list[TokenRepeatWindow] = field(default_factory=list)


def push_token_repeat_window(
    ws: TokenRepeatWindow,
    tid: int,
    *,
    ignore: frozenset[int] | set[int],
) -> int:
    """Fold one token into a single window; return this window's score.

    Score = how many times ``tid`` already appears in the window's content
    (ignored tokens score 0 and are not stored). ``repeat_sum`` is the sum of
    the last ``window`` scores.
    """
    if tid in ignore:
        score = 0
    else:
        score = int(ws.freq.get(tid, 0))
        ws.content.append(tid)
        ws.freq[tid] = score + 1
        ws.content_tokens_seen += 1
        while len(ws.content) > ws.window:
            old = ws.content.popleft()
            left = ws.freq.get(old, 0) - 1
            if left <= 0:
                ws.freq.pop(old, None)
            else:
                ws.freq[old] = left

    ws.scores.append(score)
    ws.repeat_sum += score
    while len(ws.scores) > ws.window:
        ws.repeat_sum -= ws.scores.popleft()
    return score


def push_token_repeat(
    state: TokenRepeatState,
    tid: int,
    *,
    ignore: frozenset[int] | set[int],
) -> list[int]:
    """Fold one token into every window; return per-window scores."""
    return [push_token_repeat_window(ws, tid, ignore=ignore) for ws in state.windows]


def _as_int_list(
    raw: Any,
    *,
    default: list[int],
    field: str,
    min_value: int,
    max_value: int | None = None,
) -> list[int]:
    """Normalize a scalar-or-list config value into a ``list[int]``.

    Direct ``_data`` edits (unit tests / startup overlays that bypass full
    validation) still funnel through here, so the detector stays robust.
    Out-of-range values are rejected (not silently clamped) so a bad window
    keeps the previous config instead of quietly changing behavior.
    """
    if raw is None:
        return list(default)
    if isinstance(raw, bool):
        raise ValueError(f"{field} must be an int or list of ints, got {raw!r}")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{field} must be an integer, got {raw!r}")
        _check_window_value(int(raw), field, min_value, max_value)
        return [int(raw)]
    if isinstance(raw, (list, tuple)):
        out: list[int] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{field} entries must be int, got {item!r}")
            _check_window_value(int(item), field, min_value, max_value)
            out.append(int(item))
        return out
    raise ValueError(f"{field} must be an int or list of ints, got {type(raw).__name__}")


def _check_window_value(value: int, field: str, min_value: int, max_value: int | None) -> None:
    """Reject window sizes outside ``[min_value, max_value]``."""
    if value < min_value:
        raise ValueError(f"{field} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{field} must be <= {max_value}, got {value}")


# Bounds for the multi-window config surface. A window smaller than this is a
# degenerate "two identical adjacent tokens" detector that W=8 already covers,
# and more/larger windows only add linear CPU/memory without extra coverage.
_MIN_WINDOW = 8
_MAX_WINDOW = 512
_MAX_WINDOWS = 4


class TokenRepeatDetector(ConfigBackedDetector):
    """Detect local token re-reading via one or more sliding-window ``repeat_sum``.

    Consumes the cumulative output stream from ``RequestIoSnapshotManager``
    (IO is appended once in after-sample, not in ``check_after_spec``). A
    per-req cursor tracks how far the stream has been folded into the repeat
    window so each id is pushed once. ``window`` / ``repeat_sum_threshold`` /
    ``min_tokens`` each accept a scalar (legacy single window) or a list of the
    same length (multi-window); any window crossing its own threshold arms the
    incident.
    """

    incident_type = "token_repeat"
    section_key = "token_repeat"
    schema = DetectorSchema(
        section_key="token_repeat",
        stage="after_sample",
        help="Sliding-window local token re-read (no logprobs); multi-window supported.",
        fields=(
            ConfigField("enabled", False, "bool", help="Master switch"),
            ConfigField(
                "window",
                32,
                "int_or_list_int",
                min_value=_MIN_WINDOW,
                max_value=_MAX_WINDOW,
                min_items=1,
                max_items=_MAX_WINDOWS,
                help="Sliding content window size(s); scalar or list for multi-window",
            ),
            ConfigField(
                "repeat_sum_threshold",
                64,
                "int_or_list_int",
                min_value=0,
                help="Repeat-sum alert threshold(s); scalar or list matching window",
            ),
            ConfigField(
                "min_tokens",
                32,
                "int_or_list_int",
                min_value=0,
                help="Minimum content tokens per window before alerting (scalar or list)",
            ),
            ConfigField(
                "consecutive_hits",
                1,
                "int",
                min_value=1,
                help="Required consecutive over-threshold steps",
            ),
            ConfigField(
                "ignore_token_ids",
                [],
                "list_int",
                help="Token ids excluded from window scoring",
            ),
        ),
    )

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        _section = self.schema.defaults_dict()
        self._windows: list[int] = [int(_section["window"])]
        self._repeat_sum_thresholds: list[int] = [int(_section["repeat_sum_threshold"])]
        self._min_tokens: list[int] = [int(_section["min_tokens"])]
        self._consecutive_hits_thresh = int(_section["consecutive_hits"])
        self._ignore_token_ids: frozenset[int] = frozenset()
        self._states: dict[str, TokenRepeatState] = {}
        # How many cumulative output ids have already been pushed for each req.
        self._consumed_len: dict[str, int] = {}
        self._alerted: set[str] = set()
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        try:
            new_windows = _as_int_list(
                getter("window", self._windows),
                default=self._windows,
                field="token_repeat.window",
                min_value=_MIN_WINDOW,
                max_value=_MAX_WINDOW,
            )
            if not new_windows:
                raise ValueError("token_repeat: at least one window is required")
            if len(new_windows) > _MAX_WINDOWS:
                raise ValueError(f"token_repeat: at most {_MAX_WINDOWS} windows, got {len(new_windows)}")
            new_thresholds = _as_int_list(
                getter("repeat_sum_threshold", self._repeat_sum_thresholds),
                default=self._repeat_sum_thresholds,
                field="token_repeat.repeat_sum_threshold",
                min_value=0,
            )
            new_min_tokens = _as_int_list(
                getter("min_tokens", self._min_tokens),
                default=self._min_tokens,
                field="token_repeat.min_tokens",
                min_value=0,
            )
            # A scalar threshold / min_tokens applies to every window.
            if len(new_thresholds) == 1:
                new_thresholds = new_thresholds * len(new_windows)
            if len(new_min_tokens) == 1:
                new_min_tokens = new_min_tokens * len(new_windows)
            if len(new_thresholds) != len(new_windows) or len(new_min_tokens) != len(new_windows):
                raise ValueError(
                    "token_repeat: window/repeat_sum_threshold/min_tokens lengths must match "
                    f"(window={len(new_windows)}, thresholds={len(new_thresholds)}, "
                    f"min_tokens={len(new_min_tokens)})"
                )
            if len(set(new_windows)) != len(new_windows):
                raise ValueError("token_repeat: window sizes must be unique")
        except (TypeError, ValueError) as exc:
            logger.error("[runtime_guard: token_repeat] invalid config: %s; keeping previous", exc)
            return

        windows_changed = new_windows != self._windows
        levels_changed = new_thresholds != self._repeat_sum_thresholds or new_min_tokens != self._min_tokens
        if windows_changed and self._states:
            # B12 fix: also clear _alerted — a req already alerted under the
            # old windows would never be re-evaluated under the new windows.
            self._states.clear()
            self._consumed_len.clear()
            self._alerted.clear()
        elif levels_changed and self._states:
            # Window sizes are unchanged, so in-flight windows keep their content,
            # scores and hit streaks; only the per-window threshold / warmup captured
            # at state creation have to follow the hot-reloaded config, otherwise
            # ongoing requests keep alerting on the stale values until they finish.
            for state in self._states.values():
                for ws, threshold, min_tokens in zip(state.windows, new_thresholds, new_min_tokens):
                    ws.threshold = threshold
                    ws.min_tokens = min_tokens
        self._windows = new_windows
        self._repeat_sum_thresholds = new_thresholds
        self._min_tokens = new_min_tokens
        self._consecutive_hits_thresh = max(1, int(getter("consecutive_hits", self._consecutive_hits_thresh)))
        try:
            ignore = coerce_list_int(getter("ignore_token_ids", []), "ignore_token_ids")
        except ValueError as exc:
            logger.error("[runtime_guard: token_repeat] invalid ignore_token_ids: %s; keeping previous", exc)
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
    ) -> list[Incident]:
        """Fold new cumulative output ids into per-req windows.

        Prefer ``sampled_token_ids=None`` after
        ``DetectorManager.after_sample_hot_path`` / ``check_after_spec`` have
        already written into the IO buffer. When provided (standalone callers /
        unit tests), tokens are appended here first.

        Always advances via Store ``cumulative_output_since`` + ``_consumed_len``
        so a second call in the same wave cannot re-score ids that append
        dedupe skipped (W1-1 / R-04).
        """
        if not self._precheck():
            return []

        runner = self._runner
        log_leader = runner_tp_rank(runner) == 0
        req_ids = resolve_batch_req_ids(runner, req_ids)
        if not req_ids:
            return []

        io_mgr = RequestIoSnapshotManager.get()
        if sampled_token_ids is not None:
            io_mgr.append_batch(req_ids, sampled_token_ids)

        alerts: list[Incident] = []
        for batch_idx, req_id in enumerate(req_ids):
            if not req_id:
                continue
            consumed = self._consumed_len.get(req_id, 0)
            total, new_ids = io_mgr.cumulative_output_since(req_id, consumed)
            if consumed > total:
                # Buffer was reset / replaced; restart from current stream.
                consumed = 0
                self._states.pop(req_id, None)
                total, new_ids = io_mgr.cumulative_output_since(req_id, 0)
            # Advance cursor even when skipping/alerted so we never re-push.
            self._consumed_len[req_id] = total

            if req_id in self._alerted:
                continue
            if skip_req_ids and req_id in skip_req_ids:
                continue
            if not new_ids:
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
    ) -> Incident | None:
        state = self._states.get(req_id)
        if state is None:
            state = TokenRepeatState(
                windows=[
                    TokenRepeatWindow(window=w, threshold=t, min_tokens=m)
                    for w, t, m in zip(self._windows, self._repeat_sum_thresholds, self._min_tokens)
                ]
            )
            self._states[req_id] = state

        ids_list = _ids_already_ints(token_ids)
        ignore = self._ignore_token_ids
        cons_thresh = self._consecutive_hits_thresh
        hit = False
        fired_windows: set[int] = set()
        for tid in ids_list:
            for ws in state.windows:
                push_token_repeat_window(ws, int(tid), ignore=ignore)
                warmed = ws.content_tokens_seen >= ws.min_tokens
                over = warmed and ws.repeat_sum > ws.threshold
                if over:
                    ws.consecutive_hits += 1
                else:
                    ws.consecutive_hits = 0
                if ws.consecutive_hits >= cons_thresh:
                    fired_windows.add(ws.window)
                    hit = True
                    # Keep pushing remaining ids so window state matches the
                    # full cumulative delta already committed via _consumed_len.

        if not hit:
            return None

        self._alerted.add(req_id)
        by_window = {ws.window: ws for ws in state.windows}
        primary = by_window[min(fired_windows)]
        detail = {
            "window": primary.window,
            "repeat_sum": primary.repeat_sum,
            "repeat_sum_threshold": primary.threshold,
            "content_tokens_seen": primary.content_tokens_seen,
            "consecutive_hits": primary.consecutive_hits,
            "fired_windows": sorted(fired_windows),
            "window_details": [
                {
                    "window": ws.window,
                    "repeat_sum": ws.repeat_sum,
                    "repeat_sum_threshold": ws.threshold,
                    "content_tokens_seen": ws.content_tokens_seen,
                }
                for ws in state.windows
            ],
            "chunk_len": len(ids_list),
            "recent_token_ids": ids_list[-min(32, len(ids_list)) :],
        }
        if log_leader:
            logger.error(
                "[runtime_guard: token_repeat] req=%s fired_windows=%s window=%s "
                "repeat_sum=%s threshold=%s content_seen=%s consecutive=%s",
                req_id,
                sorted(fired_windows),
                primary.window,
                primary.repeat_sum,
                primary.threshold,
                primary.content_tokens_seen,
                primary.consecutive_hits,
            )
        return Incident(
            incident_type=self.incident_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=ILL_TYPE_REPEAT,
            detail=detail,
        )
