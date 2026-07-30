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
"""Read the nightly environment table used by auto-bisect."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from tools.bisect import git_ops
from tools.bisect.config import BisectInput, Candidate

logger = logging.getLogger(__name__)

COL_NAME = "name"
COL_PATH = "yaml/path"
COL_LINK = "link"
COL_STATUS = "status"
COL_VLLM = "vLLM Git information"
COL_VLLM_ASCEND = "vLLM-Ascend Git information"
COL_CANN = "CANN Version"
COL_TORCH_NPU = "torch-npu Version"
COL_TIME = "time"

HEADER = [
    COL_NAME,
    COL_PATH,
    COL_LINK,
    COL_STATUS,
    COL_VLLM,
    COL_VLLM_ASCEND,
    COL_CANN,
    COL_TORCH_NPU,
    COL_TIME,
]

UNKNOWN_VALUES = {"", "N/A", "n/a", "unknown", "None", "none"}


@dataclass(frozen=True)
class RuntimeEnv:
    vllm_ref: str = ""
    cann_version: str = ""
    torch_npu_version: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.vllm_ref or self.cann_version or self.torch_npu_version)

    def to_dict(self) -> dict[str, str]:
        return {
            "vllm_ref": self.vllm_ref,
            "cann_version": self.cann_version,
            "torch_npu_version": self.torch_npu_version,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "RuntimeEnv | None":
        if not data:
            return None
        env = cls(
            vllm_ref=str(data.get("vllm_ref") or ""),
            cann_version=str(data.get("cann_version") or ""),
            torch_npu_version=str(data.get("torch_npu_version") or ""),
        )
        return None if env.is_empty else env


@dataclass(frozen=True)
class EnvEntry:
    name: str
    path: str
    link: str
    status: str
    vllm_commit: str
    vllm_ascend_commit: str
    cann_version: str
    torch_npu_version: str
    time: str

    @property
    def runtime_env(self) -> RuntimeEnv:
        return RuntimeEnv(
            vllm_ref="" if self.vllm_commit in UNKNOWN_VALUES else self.vllm_commit,
            cann_version="" if self.cann_version in UNKNOWN_VALUES else self.cann_version,
            torch_npu_version="" if self.torch_npu_version in UNKNOWN_VALUES else self.torch_npu_version,
        )


def _coerce(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(v) for v in value).strip()
    return str(value).strip()


def _norm(row: dict[str | None, object]) -> dict[str, str]:
    normalised: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalised[key.strip().lower()] = _coerce(value)
    return normalised


def _matches(entry: EnvEntry, name: str | None, config_yaml: str | None) -> bool:
    if name:
        return entry.name == name
    if config_yaml:
        p = entry.path.rstrip("/")
        q = config_yaml.rstrip("/")
        return p.endswith(q) or Path(p).name == Path(q).name
    return False


def _same_commit(lhs: str, rhs: str) -> bool:
    lhs = lhs.strip()
    rhs = rhs.strip()
    return bool(lhs and rhs and (lhs.startswith(rhs) or rhs.startswith(lhs)))


class EnvTable:
    """Read-only accessor over the nightly runtime environment CSV."""

    def __init__(self, path: str):
        self.path = Path(path)

    def _read_all(self) -> list[EnvEntry]:
        if not self.path.exists():
            logger.warning("Environment table not found at %s", self.path)
            return []
        entries: list[EnvEntry] = []
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            for raw in csv.DictReader(f):
                row = _norm(raw)
                name = row.get(COL_NAME.lower(), "")
                vllm_ascend_commit = row.get(COL_VLLM_ASCEND.lower(), "")
                if not name and not vllm_ascend_commit:
                    continue
                entries.append(
                    EnvEntry(
                        name=name,
                        path=row.get(COL_PATH.lower(), ""),
                        link=row.get(COL_LINK.lower(), ""),
                        status=row.get(COL_STATUS.lower(), ""),
                        vllm_commit=row.get(COL_VLLM.lower(), ""),
                        vllm_ascend_commit=vllm_ascend_commit,
                        cann_version=row.get(COL_CANN.lower(), ""),
                        torch_npu_version=row.get(COL_TORCH_NPU.lower(), ""),
                        time=row.get(COL_TIME.lower(), ""),
                    )
                )
        return entries

    @staticmethod
    def _closest_ancestor(repo: Path, target_commit: str, entries: list[EnvEntry]) -> EnvEntry | None:
        best: EnvEntry | None = None
        for entry in entries:
            commit = entry.vllm_ascend_commit
            if not commit or commit in UNKNOWN_VALUES:
                continue
            try:
                is_ancestor = git_ops.is_ancestor(repo, commit, target_commit)
            except Exception as exc:  # noqa: BLE001 - stale table rows should not abort bisect
                logger.warning("Ignoring env row with unresolved commit %s: %s", commit, exc)
                continue
            if not is_ancestor:
                continue
            if best is None or git_ops.is_ancestor(repo, best.vllm_ascend_commit, commit):
                best = entry
        return best

    def resolve_for_commits(
        self,
        repo: Path,
        inp: BisectInput,
        commits: list[Candidate],
    ) -> dict[str, RuntimeEnv]:
        """Resolve the runtime env for each commit from yaml status rows.

        Exact rows win. For commits that do not have their own nightly status
        row, use the closest preceding row for the same yaml/name in the
        first-parent history. This matches the daily-image model: the env active
        at the latest observed yaml status is used until a newer status row says
        otherwise.
        """
        rows = [entry for entry in self._read_all() if _matches(entry, inp.name, inp.config_yaml)]
        if not rows:
            logger.warning("No environment rows match name=%r config_yaml=%r", inp.name, inp.config_yaml)
            return {}

        resolved: dict[str, RuntimeEnv] = {}
        for candidate in commits:
            exact = next((entry for entry in rows if _same_commit(entry.vllm_ascend_commit, candidate.commit)), None)
            entry = exact or self._closest_ancestor(repo, candidate.commit, rows)
            if entry is not None:
                env = entry.runtime_env
                if not env.is_empty:
                    resolved[candidate.commit] = env

        logger.info(
            "Resolved runtime env for %d/%d bisect commits from %s",
            len(resolved),
            len(commits),
            self.path,
        )
        return resolved
