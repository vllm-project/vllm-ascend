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
"""Persist bisect progress so a preempted/timed-out run can resume."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BisectState:
    """Serializable search state: endpoints + window + cached verdicts.

    The state file lives on the shared PVC and persists across CI runs, so it is
    tagged with the good/bad endpoints it belongs to. ``load()`` discards a state
    whose endpoints differ from the current run (a new/changed failure of the
    same yaml must not resume a stale, unrelated search).
    """

    good: str = ""
    bad: str = ""
    lo: int = 0
    hi: int = 0
    round_idx: int = 0
    # commit sha -> "PASS"/"FAIL"/"SKIP". Only PASS/FAIL are honoured on resume;
    # SKIP is kept for the report but always re-attempted (see Bisector._judge).
    verdicts: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, good: str, bad: str) -> "BisectState | None":
        """Load resumable state for this (good, bad) pair, else None.

        Returns None when the file is missing or belongs to a different
        good/bad range (stale state from a previous, unrelated failure).
        """
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        prev_good, prev_bad = data.get("good", ""), data.get("bad", "")
        if prev_good != good or prev_bad != bad:
            logger.info(
                "Ignoring stale state at %s (was good=%s bad=%s, now good=%s bad=%s); starting fresh.",
                path,
                prev_good[:12],
                prev_bad[:12],
                good[:12],
                bad[:12],
            )
            return None
        return cls(
            good=good,
            bad=bad,
            lo=data.get("lo", 0),
            hi=data.get("hi", 0),
            round_idx=data.get("round_idx", 0),
            verdicts=data.get("verdicts", {}),
        )
