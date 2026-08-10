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
"""Version history support for nightly/weekly bisect.

The bisect signal should come from vllm-ascend changes, not from moving
external dependencies. This module records, per vllm-ascend branch and target,
the historical vLLM release tag, torch-npu pin, and CANN base version extracted
from the version-controlled files that drive the nightly/weekly image builds.
CANN is record-only because the toolkit is baked into the test image and is
not switched during a bisect run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import regex as re
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback uses regex below
    tomllib = None  # type: ignore[assignment]

from tools.bisect import git_ops
from tools.bisect.config import DEFAULT_VERSION_TABLE, REPO_ROOT, Candidate

logger = logging.getLogger(__name__)

VLLM_TAG_FILE = ".github/vllm-release-tag.commit"
REQUIREMENTS_FILE = "requirements.txt"
PYPROJECT_FILE = "pyproject.toml"

TARGET_DOCKERFILES = {
    "a2": "Dockerfile",
    "a3": "Dockerfile.a3",
    "310p": "Dockerfile.310p",
}

FIELDNAMES = (
    "branch",
    "target",
    "commit",
    "vllm_release_tag",
    "torch_npu_version",
    "cann_version",
)


class VersionHistoryError(RuntimeError):
    pass


class VersionSyncError(RuntimeError):
    pass


@dataclass(frozen=True)
class VersionProfile:
    branch: str
    target: str
    commit: str
    vllm_release_tag: str
    torch_npu_version: str
    cann_version: str

    def same_versions(self, other: VersionProfile) -> bool:
        """Return whether the runtime-switchable dependencies are identical."""
        return self.vllm_release_tag == other.vllm_release_tag and self.torch_npu_version == other.torch_npu_version

    def version_key(self) -> tuple[str, str, str]:
        """Return the complete key used to record historical changes."""
        return (self.vllm_release_tag, self.torch_npu_version, self.cann_version)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> VersionProfile:
        return cls(
            branch=data["branch"],
            target=data["target"],
            commit=data["commit"],
            vllm_release_tag=data["vllm_release_tag"],
            torch_npu_version=data["torch_npu_version"],
            cann_version=data["cann_version"],
        )


def _strip(content: str | None) -> str:
    return (content or "").strip()


def _read_file_at(repo: Path, commit: str, rel_path: str) -> str:
    content = git_ops.file_at_commit(repo, commit, rel_path)
    if content is None:
        raise VersionHistoryError(f"{rel_path} is missing at {commit[:12]}")
    return content


def _extract_arg(content: str, name: str) -> str | None:
    match = re.search(rf"^\s*ARG\s+{re.escape(name)}=(?:\"([^\"]+)\"|'([^']+)'|([^\s#]+))", content, re.MULTILINE)
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def _extract_torch_npu(requirements: str, pyproject: str | None = None) -> str:
    req_match = re.search(r"^\s*torch-npu\s*==\s*([^\s#;]+)", requirements, re.MULTILINE)
    if req_match:
        return req_match.group(1)
    if pyproject:
        if tomllib is not None:
            try:
                data = tomllib.loads(pyproject)
            except tomllib.TOMLDecodeError:
                data = {}

            requirement_values: list[str] = []
            build_system = data.get("build-system", {})
            requirement_values.extend(build_system.get("requires", []))
            project = data.get("project", {})
            requirement_values.extend(project.get("dependencies", []))
            requirement_values.extend(
                requirement
                for requirements_group in project.get("optional-dependencies", {}).values()
                for requirement in requirements_group
            )
            for requirement_text in requirement_values:
                try:
                    requirement = Requirement(requirement_text)
                except InvalidRequirement:
                    continue
                if requirement.name.lower() == "torch-npu":
                    for specifier in requirement.specifier:
                        if specifier.operator == "==":
                            return specifier.version

        pyproject_match = re.search(r"[\"']torch-npu\s*==\s*([^\"']+)[\"']", pyproject)
        if pyproject_match:
            return pyproject_match.group(1).strip()
    raise VersionHistoryError("torch-npu pin is missing from requirements.txt/pyproject.toml")


def extract_profile_at(repo: Path, commit: str, branch: str, target: str) -> VersionProfile:
    """Extract the profile that the nightly/weekly image files define at commit."""
    dockerfile = TARGET_DOCKERFILES.get(target)
    if dockerfile is None:
        expected_targets = sorted(TARGET_DOCKERFILES)
        raise VersionHistoryError(f"unsupported version target {target!r}; expected one of {expected_targets}")

    vllm_tag = _strip(git_ops.file_at_commit(repo, commit, VLLM_TAG_FILE))
    if not vllm_tag:
        dockerfile_content = _read_file_at(repo, commit, dockerfile)
        vllm_tag = _extract_arg(dockerfile_content, "VLLM_TAG") or ""
    else:
        dockerfile_content = _read_file_at(repo, commit, dockerfile)
    if not vllm_tag:
        raise VersionHistoryError(f"vLLM release tag is missing at {commit[:12]}")

    requirements = _read_file_at(repo, commit, REQUIREMENTS_FILE)
    pyproject = git_ops.file_at_commit(repo, commit, PYPROJECT_FILE)
    torch_npu = _extract_torch_npu(requirements, pyproject)

    cann_version = _extract_arg(dockerfile_content, "CANN_VERSION")
    if not cann_version:
        raise VersionHistoryError(f"CANN_VERSION is missing from {dockerfile} at {commit[:12]}")

    return VersionProfile(
        branch=branch,
        target=target,
        commit=commit,
        vllm_release_tag=vllm_tag,
        torch_npu_version=torch_npu,
        cann_version=cann_version,
    )


def infer_branch(repo: Path) -> str:
    env_branch = os.getenv("BISECT_VERSION_BRANCH") or os.getenv("VLLM_ASCEND_BRANCH")
    if env_branch:
        return env_branch.replace("/", "-")
    branch = git_ops.current_branch(repo)
    if branch and branch != "HEAD":
        return branch.replace("/", "-")
    ref = os.getenv("GITHUB_REF_NAME")
    if ref:
        return ref.replace("/", "-")
    return "main"


def infer_branch_from_good_table(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part == "vllm-ascend" and idx + 1 < len(parts):
            branch = parts[idx + 1]
            if branch:
                return branch.replace("/", "-")
    return None


def infer_target(config_yaml: str | None = None) -> str:
    env_target = os.getenv("BISECT_VERSION_TARGET")
    if env_target:
        return env_target.lower()
    soc = os.getenv("SOC_VERSION", "").lower()
    runner = os.getenv("RUNNER_NAME", "").lower()
    probe = " ".join(part for part in (soc, runner, config_yaml or "") if part).lower()
    if "310p" in probe:
        return "310p"
    if "a3" in probe or "9391" in probe:
        return "a3"
    return "a2"


class VersionHistory:
    def __init__(self, path: str, repo: Path, branch: str, target: str):
        self.path = Path(path)
        self.repo = repo
        self.branch = branch
        self.target = target

    def read_rows(self) -> list[VersionProfile]:
        if not self.path.exists():
            return []
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [
                VersionProfile.from_dict(row)
                for row in reader
                if row.get("branch") == self.branch and row.get("target") == self.target
            ]

    def append_missing(self, profiles: list[VersionProfile]) -> None:
        if not profiles:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = {(p.branch, p.target, p.commit) for p in self.read_rows()}
        missing = [p for p in profiles if (p.branch, p.target, p.commit) not in existing]
        if not missing:
            return
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            for profile in missing:
                writer.writerow(profile.to_dict())
        logger.info("[versions] appended %d version history row(s) to %s", len(missing), self.path)

    def lookup(self, commit: str) -> VersionProfile | None:
        rows = self.read_rows()
        reachable = [row for row in rows if git_ops.is_ancestor(self.repo, row.commit, commit)]
        if not reachable:
            return None
        # Pick the latest change point: among reachable rows, it is the one that
        # is not an ancestor of another reachable row.
        latest = reachable[0]
        for row in reachable[1:]:
            if git_ops.is_ancestor(self.repo, latest.commit, row.commit):
                latest = row
        return VersionProfile(
            branch=latest.branch,
            target=latest.target,
            commit=commit,
            vllm_release_tag=latest.vllm_release_tag,
            torch_npu_version=latest.torch_npu_version,
            cann_version=latest.cann_version,
        )

    def record_range(self, good: str, candidates: list[Candidate]) -> bool:
        commits = [good, *(candidate.commit for candidate in candidates)]
        profiles = [extract_profile_at(self.repo, commit, self.branch, self.target) for commit in commits]
        change_points: list[VersionProfile] = []
        previous_key: tuple[str, str, str] | None = None
        for profile in profiles:
            if profile.version_key() != previous_key:
                change_points.append(profile)
                previous_key = profile.version_key()
        self.append_missing(change_points)

        good_profile = profiles[0]
        bad_profile = profiles[-1]
        if good_profile.same_versions(bad_profile):
            logger.info(
                "[versions] endpoints use the same vLLM/torch-npu versions; "
                "CANN history recorded, per-commit version sync disabled"
            )
            return False

        logger.info(
            "[versions] vLLM/torch-npu endpoints differ; per-commit sync enabled "
            "(CANN is record-only, %d change point(s))",
            len(change_points),
        )
        return True


def _version_base(value: str) -> str:
    try:
        return Version(value).base_version
    except InvalidVersion:
        return value.lstrip("v")


def _installed_dist_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str], log_file: Path | None, cwd: Path | None = None) -> None:
    logger.info("[versions] running: %s", " ".join(cmd))
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as out:
            out.write(f"\n$ {' '.join(cmd)}\n")
            proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=out, stderr=subprocess.STDOUT, text=True)
            tail = "(see trial log)"
    else:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
        tail = (proc.stdout or proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        raise VersionSyncError(f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n{tail}")


class ExternalVersionManager:
    def __init__(
        self,
        history: VersionHistory,
        sync_enabled: bool = True,
        active: bool = False,
        vllm_repo_dir: Path | None = None,
    ):
        self.history = history
        self.sync_enabled = sync_enabled
        self.active = active
        self.vllm_repo_dir = vllm_repo_dir or Path(os.getenv("BISECT_VLLM_REPO_DIR", "/vllm-workspace/vllm"))
        self.env_overrides: dict[str, str] = {}

    def _set_env(self, key: str, value: str) -> None:
        self.env_overrides[key] = value
        os.environ[key] = value

    def prepare_range(self, good: str, candidates: list[Candidate]) -> None:
        self.active = self.sync_enabled and self.history.record_range(good, candidates)

    def profile_for_commit(self, commit: str) -> VersionProfile | None:
        if not self.active:
            return None
        return self.history.lookup(commit) or extract_profile_at(
            self.history.repo, commit, self.history.branch, self.history.target
        )

    def sync_for_commit(self, commit: str, log_file: Path | None = None) -> VersionProfile | None:
        profile = self.profile_for_commit(commit)
        if profile is None:
            return None
        self.sync_profile(profile, log_file)
        return profile

    def sync_profile(self, profile: VersionProfile, log_file: Path | None = None) -> None:
        if not self.sync_enabled:
            return
        self._sync_vllm(profile, log_file)
        self._sync_torch_npu(profile, log_file)

    def _sync_vllm(self, profile: VersionProfile, log_file: Path | None) -> None:
        installed = os.getenv("VLLM_VERSION") or _installed_dist_version("vllm")
        expected = profile.vllm_release_tag
        if installed and _version_base(installed) == _version_base(expected):
            logger.info("[versions] vLLM already matches %s", expected)
            self._set_env("VLLM_VERSION", expected.lstrip("v"))
            return

        vllm_dir = self.vllm_repo_dir
        if vllm_dir.exists():
            _run(["git", "-C", str(vllm_dir), "fetch", "--tags", "--quiet", "origin", expected], log_file)
            _run(["git", "-C", str(vllm_dir), "checkout", "--force", expected], log_file)
            _run(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    f"{vllm_dir}[audio]",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cpu/",
                    "--no-input",
                    "--disable-pip-version-check",
                ],
                log_file,
            )
            _run(["python", "-m", "pip", "uninstall", "-y", "triton"], log_file)
        else:
            _run(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    f"vllm=={expected.lstrip('v')}",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cpu/",
                    "--no-input",
                    "--disable-pip-version-check",
                ],
                log_file,
            )
        self._set_env("VLLM_VERSION", expected.lstrip("v"))

    def _sync_torch_npu(self, profile: VersionProfile, log_file: Path | None) -> None:
        installed = _installed_dist_version("torch-npu")
        expected = profile.torch_npu_version
        if installed == expected:
            logger.info("[versions] torch-npu already matches %s", expected)
            return
        _run(
            [
                "python",
                "-m",
                "pip",
                "install",
                f"torch-npu=={expected}",
                "--force-reinstall",
                "--extra-index-url",
                "https://repo.huaweicloud.com/ascend/repos/pypi",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu/",
                "--no-input",
                "--disable-pip-version-check",
            ],
            log_file,
        )


def generate_table(repo: Path, branch: str, target: str, output: str, since: str, until: str) -> None:
    good = git_ops.resolve_commit(repo, since)
    bad = git_ops.resolve_commit(repo, until)
    candidates = git_ops.candidate_list(repo, good, bad)
    VersionHistory(output, repo, branch, target).record_range(good, candidates)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bisect external-version history rows.")
    parser.add_argument("--repo-dir", default=str(REPO_ROOT))
    parser.add_argument("--branch", default=None)
    parser.add_argument("--target", default=None, choices=sorted(TARGET_DOCKERFILES))
    parser.add_argument("--output", default=DEFAULT_VERSION_TABLE)
    parser.add_argument("--since", required=True, help="old endpoint commit/ref")
    parser.add_argument("--until", default="HEAD", help="new endpoint commit/ref")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = Path(args.repo_dir)
    branch = args.branch or infer_branch(repo)
    target = args.target or infer_target()
    generate_table(repo, branch, target, args.output, args.since, args.until)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
