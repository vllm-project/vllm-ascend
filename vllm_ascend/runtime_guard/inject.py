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

"""Env-gated anomaly injection for live detector wire-up verification.

``RG_INJECT=scenario[:step][:param]`` — see ``vllm_ascend/runtime_guard/test/ANOMALY_INJECTION.md``.
Double-gated for shipped safety: injection stays dead unless (1) this file's
``_INJECT_MASTER_SWITCH`` is flipped to ``True`` (verification builds only;
requires a source edit) and (2) ``RG_INJECT`` is set. Either missing →
:data:`ENABLED` is False (zero prod overhead). Errors are logged, never raised.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import regex as re

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

SCENARIOS = ("nan_logits", "inf_logits", "forbidden_substring", "token_loop", "spec_all_reject")
DEFAULT_STEP = 5
DEFAULT_TEXT = "李白"
DEFAULT_LOOP_WAVES = 40
_IDS_RE = re.compile(r"\d+(?:,\d+)*")


class _Plan:
    __slots__ = ("scenario", "step", "param")

    def __init__(self, scenario: str, step: int, param: str | None) -> None:
        self.scenario = scenario
        self.step = step
        self.param = param


def _parse(raw: str) -> _Plan:
    parts = raw.split(":")
    scenario = parts[0]
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r} (choose from {SCENARIOS})")
    if len(parts) > 3:
        raise ValueError("expected scenario[:step][:param]")
    step = DEFAULT_STEP
    if len(parts) >= 2 and parts[1]:
        step = int(parts[1])
        if step < 1:
            raise ValueError("step must be >= 1")
    param = parts[2] if len(parts) >= 3 and parts[2] else None
    return _Plan(scenario, step, param)


# Source-level master switch, shipped False: production builds ignore RG_INJECT
# entirely. Arming = edit this constant to True in a verification build.
_INJECT_MASTER_SWITCH = False


def _load() -> _Plan | None:
    if not _INJECT_MASTER_SWITCH:
        return None
    raw = os.environ.get("RG_INJECT", "")
    if not raw:
        return None
    try:
        return _parse(raw)
    except ValueError as exc:
        logger.error("[INJECT] invalid RG_INJECT=%r (%s); injection disabled", raw, exc)
        return None


_plan = _load()
ENABLED = _plan is not None

_wave = 0
# Set by inject_before_sample (canonical). After-sample / after-spec lazily
# advance when the pre-sample wrap was skipped (e.g. logits_finite off).
_wave_opened = False
_fired: set[str] = set()
_last_tok: int | None = None
_loop_waves_left = 0
_pattern: list[int] | None = None
_pattern_pos = 0


def _open_wave(*, from_before: bool) -> None:
    """Advance ``_wave`` once per engine step.

    ``inject_before_sample`` always opens a new wave (and always increments).
    Post-sample hooks open only when before-sample did not run this step, so
    ``RG_INJECT=token_loop`` still works with ``logits_finite`` disabled.
    """
    global _wave, _wave_opened
    if from_before:
        _wave += 1
        _wave_opened = True
        return
    if not _wave_opened:
        _wave += 1
        _wave_opened = True


def _close_wave() -> None:
    global _wave_opened
    _wave_opened = False


def _set_row0_col(data: Any, col: int, value: float) -> None:
    """Set ``data[0, col]`` on a tensor or nested-list logits view."""
    if hasattr(data, "shape"):
        data[0, col] = value
    else:
        data[0][col] = float(value)


def _row0_ids(sampled: Any) -> list[int]:
    row = sampled[0]
    out: list[int] = []
    for x in row:
        if hasattr(x, "item"):
            with contextlib.suppress(Exception):
                x = x.item()
        out.append(int(x))
    return out


def _overwrite_row0_tokens(sampled: Any, token_id: int) -> None:
    """Pin every track of the first sampling row to ``token_id``."""
    if hasattr(sampled, "shape"):
        sampled[0, :] = token_id
        return
    row = sampled[0]
    if hasattr(row, "fill_"):
        row.fill_(token_id)
        return
    for i in range(len(row)):
        row[i] = token_id


def _resolve_pattern(param: str | None, runner: Any) -> list[int] | None:
    """Token ids for ``forbidden_substring``: raw ``"id,id"`` or encoded text.

    Returns None (retry next wave) while the tokenizer is not yet available.
    """
    global _pattern
    if _pattern is not None:
        return _pattern
    if param and _IDS_RE.fullmatch(param):
        _pattern = [int(x) for x in param.split(",")]
        return _pattern
    text = param or DEFAULT_TEXT
    from vllm_ascend.runtime_guard.token_utils import load_model_tokenizer

    try:
        tok = load_model_tokenizer(runner)
        if tok is None:
            return None
        ids = [int(t) for t in tok.encode(text, add_special_tokens=False)]
    except Exception as exc:
        logger.warning("[INJECT] tokenizer failed (%s); retrying next wave", exc)
        return None
    if not ids:
        logger.warning("[INJECT] pattern %r encoded to 0 tokens; injection disabled", text)
        _pattern = []
        return None
    _pattern = ids
    return _pattern


def inject_before_sample(logits: Any) -> None:
    """``nan_logits`` / ``inf_logits``: corrupt row 0 pre-sample (one-shot).

    Always opens the wave counter so post-sample hooks in the same step share
    the same ``_wave`` value.
    """
    _open_wave(from_before=True)
    plan = _plan
    if plan is None or plan.scenario not in ("nan_logits", "inf_logits"):
        return
    if plan.scenario in _fired or _wave < plan.step:
        return
    col = 5 if plan.scenario == "nan_logits" else 3
    value = float("nan") if plan.scenario == "nan_logits" else float("inf")
    try:
        _set_row0_col(logits, col, value)
    except Exception as exc:
        logger.warning("[INJECT] %s write failed wave=%d error=%s", plan.scenario, _wave, exc)
        return
    _fired.add(plan.scenario)
    logger.info("[INJECT] scenario=%s wave=%d hook=before_sample col=%d", plan.scenario, _wave, col)


def inject_after_spec(accepted_token_nums: Any) -> None:
    """``spec_all_reject``: zero accepted counts pre-detect (one-shot)."""
    _open_wave(from_before=False)
    plan = _plan
    if plan is None or plan.scenario != "spec_all_reject":
        return
    if plan.scenario in _fired or _wave < plan.step:
        return
    try:
        if hasattr(accepted_token_nums, "zero_"):
            accepted_token_nums.zero_()
        else:
            for i in range(len(accepted_token_nums)):
                accepted_token_nums[i] = 0
    except Exception as exc:
        logger.warning("[INJECT] spec_all_reject write failed wave=%d error=%s", _wave, exc)
        return
    _fired.add(plan.scenario)
    logger.info(
        "[INJECT] scenario=spec_all_reject wave=%d hook=after_spec bs=%d",
        _wave,
        len(accepted_token_nums) if hasattr(accepted_token_nums, "__len__") else -1,
    )


def inject_after_sample(sampled_token_ids: Any, runner: Any = None) -> None:
    """``forbidden_substring`` / ``token_loop``: corrupt row 0 post-sample."""
    global _last_tok, _loop_waves_left, _pattern_pos
    _open_wave(from_before=False)
    try:
        plan = _plan
        if plan is None or plan.scenario not in ("forbidden_substring", "token_loop"):
            return
        if _wave < plan.step:
            return
        if plan.scenario == "forbidden_substring":
            ids = _resolve_pattern(plan.param, runner)
            if not ids:
                return
            token_id = ids[_pattern_pos % len(ids)]
            _pattern_pos += 1
            try:
                _overwrite_row0_tokens(sampled_token_ids, token_id)
            except Exception as exc:
                logger.warning("[INJECT] forbidden_substring write failed wave=%d error=%s", _wave, exc)
                return
            logger.info(
                "[INJECT] scenario=forbidden_substring wave=%d hook=after_sample token=%d (%d/%d)",
                _wave,
                token_id,
                (_pattern_pos - 1) % len(ids) + 1,
                len(ids),
            )
            return
        # token_loop: anchor = the trigger wave's own row-0 last sampled token.
        if plan.scenario in _fired:
            return
        if _last_tok is None:
            ids = _row0_ids(sampled_token_ids)
            if not ids:
                return
            _last_tok = ids[-1]
            try:
                _loop_waves_left = int(plan.param) if plan.param and plan.param.isdigit() else DEFAULT_LOOP_WAVES
            except ValueError:
                _loop_waves_left = DEFAULT_LOOP_WAVES
        try:
            _overwrite_row0_tokens(sampled_token_ids, _last_tok)
        except Exception as exc:
            logger.warning("[INJECT] token_loop write failed wave=%d error=%s", _wave, exc)
            return
        _loop_waves_left -= 1
        if _loop_waves_left <= 0:
            _fired.add("token_loop")
        logger.info(
            "[INJECT] scenario=token_loop wave=%d hook=after_sample token=%d waves_left=%d",
            _wave,
            _last_tok,
            max(_loop_waves_left, 0),
        )
    finally:
        # Last inject hook in the sync sample path; async get_output also ends here.
        _close_wave()
