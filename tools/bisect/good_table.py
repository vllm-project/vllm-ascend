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
"""Read the nightly status table to find a case's last-known-good commit.

The table is produced by the nightly pipeline (one row per case per run) with
these columns::

    name, yaml/path, link, status, vLLM Git information, vLLM-Ascend Git information, time

Example rows (columns abbreviated)::

    test_custom_op_multi_card, .../multicard_ops_a2/, <link>, success, <vllm>, <vllm_ascend>, <time>
    Qwen3.5-397B-A17B-w4a8-mtp, .../Qwen3.5-...-A2.yaml, <link>, failure, <vllm>, <vllm_ascend>, <time>

For a given case (matched by ``name`` or by ``yaml/path``) the last-known-good
vllm-ascend commit is the ``vLLM-Ascend Git information`` of the most recent row
whose ``status`` is ``success``. That row also records the paired vLLM commit,
which lets us keep vLLM in sync while bisecting vllm-ascend.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Column headers, kept tolerant of case/whitespace differences on read.
COL_NAME = "name"
COL_PATH = "yaml/path"
COL_LINK = "link"
COL_STATUS = "status"
COL_VLLM = "vLLM Git information"
COL_VLLM_ASCEND = "vLLM-Ascend Git information"
COL_TIME = "time"

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class GoodEntry:
    name: str
    path: str
    link: str
    status: str
    vllm_commit: str
    vllm_ascend_commit: str
    time: str

    @property
    def is_success(self) -> bool:
        return self.status.strip().lower() in ("success", "pass", "passed", "ok")


def _coerce(value: object) -> str:
    """Normalise a DictReader cell to a stripped string.

    ``csv.DictReader`` yields a *list* under the ``None`` key for surplus columns
    when a row has more fields than the header (e.g. an unquoted value contains a
    comma). Joining keeps the data readable; ``None`` (a short row's missing
    cell) becomes "".
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(v) for v in value).strip()
    return str(value).strip()


def _norm(row: dict[str | None, object]) -> tuple[dict[str, str], bool]:
    """Return (lower-cased/stripped row, had_surplus_columns).

    ``had_surplus_columns`` is True when DictReader produced a ``None`` overflow
    key, i.e. that data row has more columns than the header (misaligned CSV).
    """
    surplus = row.get(None)
    had_surplus = surplus is not None and surplus != [] and surplus != ""
    normalised: dict[str, str] = {}
    for key, value in row.items():
        if key is None:  # surplus-column overflow; not a real named field
            continue
        normalised[key.strip().lower()] = _coerce(value)
    return normalised, had_surplus


def _parse_time(value: str) -> datetime:
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Unparsable timestamps sort oldest so they never shadow a real success.
    return datetime.min.replace(tzinfo=None)


class GoodTable:
    """Read-only accessor over the nightly status CSV."""

    def __init__(self, path: str):
        self.path = Path(path)

    def _read_all(self) -> list[GoodEntry]:
        if not self.path.exists():
            logger.warning("Good table not found at %s", self.path)
            return []
        entries: list[GoodEntry] = []
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            for line_no, raw in enumerate(csv.DictReader(f), start=2):
                row, had_surplus = _norm(raw)
                if had_surplus:
                    logger.warning(
                        "Good table line %d has more columns than the header "
                        "(misaligned CSV, likely an unquoted comma in a field); "
                        "parsing the named columns best-effort. name=%r",
                        line_no,
                        row.get(COL_NAME.lower(), ""),
                    )
                name = row.get(COL_NAME.lower(), "")
                if not name and not row.get(COL_PATH.lower()):
                    continue
                entries.append(
                    GoodEntry(
                        name=name,
                        path=row.get(COL_PATH.lower(), ""),
                        link=row.get(COL_LINK.lower(), ""),
                        status=row.get(COL_STATUS.lower(), ""),
                        vllm_commit=row.get(COL_VLLM.lower(), ""),
                        vllm_ascend_commit=row.get(COL_VLLM_ASCEND.lower(), ""),
                        time=row.get(COL_TIME.lower(), ""),
                    )
                )
        return entries

    @staticmethod
    def _matches(entry: GoodEntry, name: str | None, config_yaml: str | None) -> bool:
        if name:
            return entry.name == name
        if config_yaml:
            p = entry.path.rstrip("/")
            q = config_yaml.rstrip("/")
            return p.endswith(q) or Path(p).name == Path(q).name
        return False

    def lookup_last_good(self, *, name: str | None = None, config_yaml: str | None = None) -> GoodEntry | None:
        """Latest ``success`` row for the case, or None.

        Match by ``name`` when given, otherwise by ``yaml/path`` ending with
        ``config_yaml`` (or its basename). The newest success row by ``time``
        wins; its ``vllm_ascend_commit`` is the good bisect endpoint.
        """
        rows = [
            e for e in self._read_all() if self._matches(e, name, config_yaml) and e.is_success and e.vllm_ascend_commit
        ]
        if not rows:
            logger.warning("No successful good-table row for name=%r config_yaml=%r", name, config_yaml)
            return None
        best = max(rows, key=lambda e: _parse_time(e.time))
        logger.info(
            "Good baseline from table: %s @ %s (vllm-ascend=%s, vllm=%s)",
            best.name,
            best.time,
            best.vllm_ascend_commit[:12],
            best.vllm_commit[:12],
        )
        return best
