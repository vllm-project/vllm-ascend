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

from tools.bisect.config import Verdict

logger = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """Raw signals produced by a runner for one trial."""

    exit_code: int
    # Directory holding this trial's benchmark JSON(s). A whole-YAML run writes
    # one file per case, so we scan the directory rather than a single file.
    results_dir: Path | None = None
    infra_error: bool = False  # set when the failure is environmental (-> SKIP)
    skip_reason: str = ""  # human reason when infra_error is set (e.g. vllm skew)


def _any_benchmark_failed(results_dir: Path | None) -> bool:
    """True if any benchmark JSON in ``results_dir`` reports pass_fail=fail."""
    if not results_dir or not results_dir.exists():
        return False
    for path in results_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read results json %s: %s", path, exc)
            continue
        if data.get("pass_fail") == "fail":
            logger.info("Benchmark regression in %s (pass_fail=fail)", path.name)
            return True
    return False


# pytest exit codes that mean "the test never actually ran" rather than a
# genuine functional pass/fail. These are NOT a clean bisect signal, so they map
# to SKIP (like ``git bisect skip``):
#   2 = interrupted, 3 = internal error, 4 = usage/collection error
#       (e.g. a conftest ImportError -> vllm/vllm-ascend version mismatch),
#   5 = no tests collected (e.g. -k matched nothing).
# 124 is our own timeout sentinel (see runner) -> also unjudgeable.
_INFRA_EXIT_CODES = {2, 3, 4, 5, 124}


def evaluate(outcome: RunOutcome) -> tuple[Verdict, str]:
    """Map a run outcome to a verdict plus a short human reason."""
    if outcome.infra_error:
        return "SKIP", outcome.skip_reason or "infra/environment error - cannot judge this commit"

    rc = outcome.exit_code
    if rc in _INFRA_EXIT_CODES:
        if rc == 124:
            return "SKIP", "trial timed out - cannot judge this commit"
        return "SKIP", (
            f"pytest could not collect/run the test (rc={rc}); likely an "
            "environment issue such as a conftest ImportError / vllm mismatch"
        )
    if rc != 0:
        # A real crash during a running test (assertion, segfault, ...) -> FAIL.
        return "FAIL", f"pytest exited non-zero (rc={rc})"

    # Exit code 0: still inspect the benchmark verdicts for silent regressions
    # (a baseline/threshold miss is recorded in JSON without failing pytest).
    if _any_benchmark_failed(outcome.results_dir):
        return "FAIL", "benchmark pass_fail=fail (baseline/threshold miss)"
    return "PASS", "pytest ok (no benchmark regression)"
