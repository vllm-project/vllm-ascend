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
"""Read/update the "last known good" table.

The table is a small CSV so it can be opened in any editor / spreadsheet and
read at a glance. One row per nightly case::

    case_key,scene,config_yaml,case_name,last_good_commit,last_good_pr,updated_at

``case_key`` is the join of scene + yaml + case (see ``BisectInput.case_key``)
and is the lookup key. Reads tolerate a missing file (returns no match);
updates create the file/parent dirs on demand and rewrite atomically.
"""

import csv
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import filelock

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "case_key",
    "scene",
    "config_yaml",
    "case_name",
    "last_good_commit",
    "last_good_pr",
    "updated_at",
]


@dataclass(frozen=True)
class GoodEntry:
    case_key: str
    scene: str
    config_yaml: str
    case_name: str
    last_good_commit: str
    last_good_pr: str = ""
    updated_at: str = ""


class GoodTable:
    """CSV-backed store mapping a case to its last passing commit."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = filelock.FileLock(str(self.path) + ".lock")

    # ----------------------------------------------------------------- read
    def _read_all(self) -> dict[str, GoodEntry]:
        if not self.path.exists():
            logger.warning("Good table not found at %s", self.path)
            return {}
        entries: dict[str, GoodEntry] = {}
        with self.path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("case_key") or "").strip()
                if not key:
                    continue
                entries[key] = GoodEntry(
                    case_key=key,
                    scene=(row.get("scene") or "").strip(),
                    config_yaml=(row.get("config_yaml") or "").strip(),
                    case_name=(row.get("case_name") or "").strip(),
                    last_good_commit=(row.get("last_good_commit") or "").strip(),
                    last_good_pr=(row.get("last_good_pr") or "").strip(),
                    updated_at=(row.get("updated_at") or "").strip(),
                )
        return entries

    def lookup(self, case_key: str) -> GoodEntry | None:
        """Return the last-good entry for ``case_key`` or None."""
        entry = self._read_all().get(case_key)
        if entry is None:
            logger.warning("No good-table entry for case_key=%s", case_key)
        elif not entry.last_good_commit:
            logger.warning("Good-table entry for %s has empty commit", case_key)
            return None
        return entry

    # ---------------------------------------------------------------- write
    def update(
        self,
        *,
        case_key: str,
        scene: str,
        config_yaml: str,
        case_name: str,
        last_good_commit: str,
        last_good_pr: str = "",
    ) -> None:
        """Insert or replace the row for ``case_key`` (atomic, lock-guarded)."""
        with self._lock:
            entries = self._read_all()
            entries[case_key] = GoodEntry(
                case_key=case_key,
                scene=scene,
                config_yaml=config_yaml,
                case_name=case_name,
                last_good_commit=last_good_commit,
                last_good_pr=last_good_pr,
                updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            self._write_all(entries)
        logger.info("Updated good table: %s -> %s", case_key, last_good_commit[:12])

    def _write_all(self, entries: dict[str, GoodEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".csv")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                for entry in sorted(entries.values(), key=lambda e: e.case_key):
                    writer.writerow(
                        {
                            "case_key": entry.case_key,
                            "scene": entry.scene,
                            "config_yaml": entry.config_yaml,
                            "case_name": entry.case_name,
                            "last_good_commit": entry.last_good_commit,
                            "last_good_pr": entry.last_good_pr,
                            "updated_at": entry.updated_at,
                        }
                    )
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
