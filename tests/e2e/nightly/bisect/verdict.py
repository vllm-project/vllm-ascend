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
"""Normalise a trial run into PASS / FAIL / SKIP.

Note: a benchmark *baseline/threshold* miss is recorded in the results JSON's
``pass_fail`` field but does NOT (unless ``benchmark_comparisons`` is enabled)
make pytest exit non-zero. So a perf/accuracy regression is only visible by
reading the JSON -- we must check both signals, not just the exit code.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.nightly.bisect.config import Verdict

logger = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """Raw signals produced by a runner for one trial."""

    exit_code: int
    results_json: Path | None  # benchmark results file, if the case writes one
    infra_error: bool = False  # set when the failure is environmental (-> SKIP)


def _read_pass_fail(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read results json %s: %s", path, exc)
        return None
    return data.get("pass_fail")


def evaluate(outcome: RunOutcome) -> tuple[Verdict, str]:
    """Map a run outcome to a verdict plus a short human reason."""
    if outcome.infra_error:
        return "SKIP", "infra/environment error - cannot judge this commit"

    if outcome.exit_code != 0:
        return "FAIL", f"pytest exited non-zero (rc={outcome.exit_code})"

    # Exit code 0: still inspect the benchmark verdict for silent regressions.
    pass_fail = _read_pass_fail(outcome.results_json)
    if pass_fail == "fail":
        return "FAIL", "benchmark pass_fail=fail (baseline/threshold miss)"
    if pass_fail == "pass":
        return "PASS", "pytest ok and benchmark pass_fail=pass"
    return "PASS", "pytest ok (no benchmark verdict file)"
