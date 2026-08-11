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
"""Read a scheduled-test status table to find a case's last-known-good commit.

The table is maintained as one latest-success row per
``soc + scene + yaml/path`` key with these columns::

    name, yaml/path, link, status, vLLM Git information,
    vLLM-Ascend Git information, soc, scene, time

Example rows (columns abbreviated)::

    qwen3-30b-acc, .../test_qwen3_30b_acc.py, <link>, success, <vllm>, <vllm_ascend>, <time>
    Qwen3.5-397B-A17B-w4a8-mtp, .../Qwen3.5-...-A2.yaml, <link>, failure, <vllm>, <vllm_ascend>, <time>

For a given case (matched by the supplied identity dimensions) the last-known-good
vllm-ascend commit is the ``vLLM-Ascend Git information`` of the most recent row
whose ``status`` is ``success``. That row also records the paired vLLM commit,
which lets us keep vLLM in sync while bisecting vllm-ascend.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Column headers, kept tolerant of case/whitespace differences on read.
COL_NAME = "name"
COL_PATH = "yaml/path"
COL_LINK = "link"
COL_STATUS = "status"
COL_VLLM = "vLLM Git information"
COL_VLLM_ASCEND = "vLLM-Ascend Git information"
COL_SOC = "soc"
COL_SCENE = "scene"
COL_TIME = "time"

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S")
VALID_SCENES = ("single_node", "multi_node")
INVALID_SOC_VALUES = {"", "unknown", "none", "null"}


def valid_soc(value: str) -> str:
    """Return a stripped soc that is safe to use in the composite key."""
    value = value.strip()
    if not value or value.lower() in INVALID_SOC_VALUES:
        raise ValueError(f"valid soc is required for good-table lookup, got {value!r}")
    return value


def valid_scene(value: str) -> str:
    value = value.strip()
    if value not in VALID_SCENES:
        raise ValueError(f"invalid scene {value!r}; expected one of {VALID_SCENES}")
    return value


@dataclass(frozen=True)
class GoodEntry:
    name: str
    path: str
    link: str
    status: str
    vllm_commit: str
    vllm_ascend_commit: str
    soc: str
    scene: str
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
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        # Normalise to naive UTC so tz-aware and tz-less rows stay comparable;
        # otherwise ``max()`` below raises on mixed formats.
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    # Unparsable timestamps sort oldest so they never shadow a real success.
    return datetime.min


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
                        soc=row.get(COL_SOC.lower(), ""),
                        scene=row.get(COL_SCENE.lower(), ""),
                        time=row.get(COL_TIME.lower(), ""),
                    )
                )
        return entries

    @staticmethod
    def _matches(
        entry: GoodEntry,
        name: str | None,
        config_yaml: str | None,
    ) -> bool:
        if name and entry.name != name:
            return False
        if config_yaml:
            p = entry.path.rstrip("/")
            q = config_yaml.rstrip("/")
            if not (p.endswith(q) or Path(p).name == Path(q).name):
                return False
        return bool(name or config_yaml)

    def lookup_last_good(
        self,
        *,
        soc: str,
        scene: str,
        name: str | None = None,
        config_yaml: str | None = None,
    ) -> GoodEntry | None:
        """Latest ``success`` row for the case, or None.

        ``soc`` and ``scene`` are required request dimensions. Rows in the new
        nine-column schema are matched exactly; legacy seven-column rows
        (without soc/scene) are only used as a fallback while they migrate and
        produce a warning. The newest success row by ``time`` wins; its
        ``vllm_ascend_commit`` is the good bisect endpoint.
        """
        soc = valid_soc(soc)
        scene = valid_scene(scene)
        exact: list[GoodEntry] = []
        legacy: list[GoodEntry] = []
        for entry in self._read_all():
            if not self._matches(entry, name, config_yaml):
                continue
            if not entry.is_success or not entry.vllm_ascend_commit:
                continue
            if entry.soc == soc and entry.scene == scene:
                exact.append(entry)
            elif not entry.soc and not entry.scene:
                legacy.append(entry)
            # Rows with a partially filled key (only one of soc/scene) never
            # match; the writer always stores both.
        if exact:
            best = max(exact, key=lambda e: _parse_time(e.time))
        elif legacy:
            logger.warning(
                "No exact good-table row for soc=%r scene=%r; falling back to a "
                "legacy row (it will be migrated on the next successful write)",
                soc,
                scene,
            )
            best = max(legacy, key=lambda e: _parse_time(e.time))
        else:
            logger.warning(
                "No successful good-table row for name=%r config_yaml=%r soc=%r scene=%r",
                name,
                config_yaml,
                soc,
                scene,
            )
            return None
        logger.info(
            "Good baseline from table: %s @ %s (vllm-ascend=%s, vllm=%s)",
            best.name,
            best.time,
            best.vllm_ascend_commit[:12],
            best.vllm_commit[:12],
        )
        return best
