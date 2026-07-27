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
# This file is a part of the vllm-ascend project.
"""Console markers and the final bisect report."""

import json
from pathlib import Path

from tools.bisect.config import BisectInput, Candidate, TrialResult

# ANSI colors (mirrors the style used in multi_node/scripts/run.sh).
_GREEN = "\033[0;32m"
_RED = "\033[0;31m"
_YELLOW = "\033[0;33m"
_BLUE = "\033[0;34m"
_NC = "\033[0m"

_MARK = {
    "PASS": (_GREEN, "PASS"),
    "FAIL": (_RED, "FAIL"),
    "SKIP": (_YELLOW, "SKIP"),
}


def print_verdict(result: TrialResult) -> None:
    """The obvious per-trial good/bad marker, e.g. ``[FAIL] PR-10538``."""
    color, tag = _MARK[result.verdict]
    rebuilt = "rebuilt" if result.rebuilt else "checkout-only"
    print(
        f"{color}[{tag}] {result.candidate.label}{_NC}  "
        f"({result.candidate.short}, {rebuilt}, {result.duration_s:.0f}s) "
        f"- {result.note}",
        flush=True,
    )


def print_progress(round_idx: int, lo: int, hi: int, candidate: Candidate) -> None:
    remaining = max(hi - lo, 1)
    import math

    print(
        f"{_BLUE}=== Round {round_idx}: testing {candidate.label} "
        f"({candidate.short}); window=[{lo},{hi}] {remaining} left, "
        f"~{math.ceil(math.log2(remaining + 1))} rounds to go ==={_NC}",
        flush=True,
    )


def print_endpoint_check(role: str, candidate: Candidate, verdict: str, ok: bool) -> None:
    color = _GREEN if ok else _RED
    status = "OK" if ok else "UNEXPECTED"
    print(
        f"{color}[{role} check] {candidate.label} ({candidate.short}) -> {verdict} [{status}]{_NC}",
        flush=True,
    )


def print_conclusion(first_bad: Candidate, trials: list[TrialResult]) -> None:
    print(f"\n{_RED}{'=' * 64}{_NC}", flush=True)
    print(f"{_RED}  FIRST BAD COMMIT: {first_bad.commit}{_NC}", flush=True)
    print(f"{_RED}  FIRST BAD PR:     {first_bad.label}{_NC}", flush=True)
    print(f"{_RED}  Subject:          {first_bad.subject}{_NC}", flush=True)
    print(f"{_RED}{'=' * 64}{_NC}\n", flush=True)
    print("Trial history:", flush=True)
    for t in trials:
        _, tag = _MARK[t.verdict]
        print(f"  [{tag}] {t.candidate.short} {t.candidate.label} - {t.note}", flush=True)


def write_report_json(
    path: Path,
    *,
    inp: BisectInput,
    good: Candidate | str,
    bad: Candidate,
    first_bad: Candidate | None,
    trials: list[TrialResult],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _cand(c) -> dict | str:
        if isinstance(c, Candidate):
            return {"commit": c.commit, "pr": c.pr_number, "subject": c.subject}
        return c

    payload = {
        "case_key": inp.case_key,
        "scene": inp.scene,
        "config_yaml": inp.config_yaml,
        "name": inp.name,
        "good": _cand(good),
        "bad": _cand(bad),
        "first_bad_commit": first_bad.commit if first_bad else None,
        "first_bad_pr": first_bad.pr_number if first_bad else None,
        "first_bad_subject": first_bad.subject if first_bad else None,
        "trials": [
            {
                "commit": t.candidate.commit,
                "pr": t.candidate.pr_number,
                "verdict": t.verdict,
                "rebuilt": t.rebuilt,
                "duration_s": round(t.duration_s, 1),
                "exit_code": t.exit_code,
                "log": t.log_path,
                "note": t.note,
            }
            for t in trials
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBisect report written to {path}", flush=True)
    return path
