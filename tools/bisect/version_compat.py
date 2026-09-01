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
"""Version discovery and dependency adaptation for bisect trials.

Nightly images are built from the version metadata kept in each
``vllm-ascend`` commit:

* ``.github/vllm-release-tag.commit`` selects the vLLM release;
* ``requirements.txt`` selects the torch-npu distribution.
"""

import importlib.metadata
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from tools.bisect import git_ops

logger = logging.getLogger(__name__)

VLLM_PACKAGE = "vllm"
TORCH_NPU_PACKAGE = "torch-npu"

ALL_PACKAGES = (VLLM_PACKAGE, TORCH_NPU_PACKAGE)

VLLM_TAG_FILE = ".github/vllm-release-tag.commit"
TORCH_NPU_REQUIREMENTS_FILE = "requirements.txt"
TORCH_NPU_PYPROJECT_FILE = "pyproject.toml"


@dataclass(frozen=True)
class PackageVersions:
    """Versions declared by one vllm-ascend commit."""

    vllm: str | None = None
    torch_npu: str | None = None

    def get(self, package: str) -> str | None:
        if package == VLLM_PACKAGE:
            return self.vllm
        if package == TORCH_NPU_PACKAGE:
            return self.torch_npu
        raise ValueError(f"unsupported package {package!r}")

    def as_dict(self, packages: tuple[str, ...] = ALL_PACKAGES) -> dict[str, str]:
        return {package: value for package in packages if (value := self.get(package)) is not None}

    @classmethod
    def from_dict(cls, values: dict[str, str] | None) -> "PackageVersions":
        values = values or {}
        return cls(
            vllm=values.get(VLLM_PACKAGE),
            torch_npu=values.get(TORCH_NPU_PACKAGE),
        )


@dataclass(frozen=True)
class VersionPolicy:
    """Packages whose versions changed between the bisect endpoints."""

    checked_packages: tuple[str, ...] = ()
    good: PackageVersions = PackageVersions()
    bad: PackageVersions = PackageVersions()

    @classmethod
    def between(cls, good: PackageVersions, bad: PackageVersions) -> "VersionPolicy":
        checked = tuple(package for package in ALL_PACKAGES if not versions_equal(good.get(package), bad.get(package)))
        return cls(checked_packages=checked, good=good, bad=bad)

    def checks(self, package: str) -> bool:
        return package in self.checked_packages

    @property
    def enabled(self) -> bool:
        return bool(self.checked_packages)


class VersionAdaptationError(RuntimeError):
    """Raised when a requested package version cannot be installed."""


def versions_equal(left: str | None, right: str | None) -> bool:
    """Compare public versions while ignoring only local build suffixes."""
    if left is None or right is None:
        return left == right
    try:
        return Version(left).public == Version(right).public
    except InvalidVersion:
        return left.strip().lstrip("v") == right.strip().lstrip("v")


def _read_torch_npu_requirement(content: str | None) -> str | None:
    if not content:
        return None
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-r", "--")):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        if requirement.name.lower().replace("_", "-") != TORCH_NPU_PACKAGE:
            continue
        for specifier in requirement.specifier:
            if specifier.operator == "==":
                return specifier.version
        return str(requirement.specifier) or None
    return None


def _read_torch_npu_from_pyproject(content: str | None) -> str | None:
    if not content:
        return None
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return None
    requires = data.get("build-system", {}).get("requires", [])
    return _read_torch_npu_requirement("\n".join(requires))


def _file_content(repo: Path, commit: str | None, relative_path: str) -> str | None:
    if commit is None:
        path = repo / relative_path
        return path.read_text(encoding="utf-8") if path.exists() else None
    return git_ops.file_at_commit(repo, commit, relative_path)


def expected_versions(repo: Path, commit: str | None = None) -> PackageVersions:
    """Versions declared by the working tree, or by ``commit`` when given."""
    vllm_tag = (_file_content(repo, commit, VLLM_TAG_FILE) or "").strip() or None
    torch_npu = _read_torch_npu_requirement(
        _file_content(repo, commit, TORCH_NPU_REQUIREMENTS_FILE),
    ) or _read_torch_npu_from_pyproject(
        _file_content(repo, commit, TORCH_NPU_PYPROJECT_FILE),
    )
    return PackageVersions(vllm=vllm_tag, torch_npu=torch_npu)


def installed_package_version(package: str) -> str | None:
    """Read the installed distribution version without importing NPU modules."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - broken metadata must not abort discovery
        logger.warning("Could not read installed %s version: %s", package, exc)
        return None


def installed_vllm_version() -> str | None:
    """Return the active vLLM version, retaining the existing env override."""
    env = os.getenv("VLLM_VERSION")
    if env:
        return env.strip()
    return installed_package_version(VLLM_PACKAGE)


def installed_torch_npu_version() -> str | None:
    return installed_package_version(TORCH_NPU_PACKAGE)


def installed_versions() -> PackageVersions:
    return PackageVersions(
        vllm=installed_vllm_version(),
        torch_npu=installed_torch_npu_version(),
    )


class VersionAdapter:
    """Bring switchable packages to the versions declared by a candidate."""

    def __init__(self, options):
        self.opt = options
        self.vllm_dir = Path(options.vllm_dir)
        self._overrides: dict[str, str] = {}

    def targets_at(self, repo: Path, commit: str, policy: VersionPolicy) -> dict[str, str]:
        return expected_versions(repo, commit).as_dict(policy.checked_packages)

    def ensure_at_commit(
        self,
        repo: Path,
        commit: str,
        policy: VersionPolicy,
        log_file: Path | None = None,
    ) -> dict[str, str]:
        return self.ensure_targets(self.targets_at(repo, commit, policy), policy.checked_packages, log_file)

    def ensure_targets(
        self,
        targets: dict[str, str],
        checked_packages: tuple[str, ...] | list[str] = ALL_PACKAGES,
        log_file: Path | None = None,
    ) -> dict[str, str]:
        for package in checked_packages:
            expected = targets.get(package)
            if not expected:
                logger.info("[version] no declared %s version; skipping adaptation", package)
                continue
            installed = self._installed_version(package)
            if _matches_expected(package, expected, installed):
                logger.info("[version] %s matches (installed=%s expected=%s)", package, installed, expected)
                continue
            logger.info("[version] switching %s from %s to %s", package, installed, expected)
            if package == VLLM_PACKAGE:
                self._switch_vllm(expected, log_file)
            elif package == TORCH_NPU_PACKAGE:
                self._switch_torch_npu(expected, log_file)
            else:
                raise VersionAdaptationError(f"version switching is not supported for {package}")
        return targets

    def _installed_version(self, package: str) -> str | None:
        if package in self._overrides:
            return self._overrides[package]
        if package == VLLM_PACKAGE:
            return installed_vllm_version()
        if package == TORCH_NPU_PACKAGE:
            return installed_torch_npu_version()
        raise ValueError(f"unsupported package {package!r}")

    def _switch_vllm(self, expected: str, log_file: Path | None) -> None:
        if self.vllm_dir.is_dir():
            try:
                target = git_ops.resolve_commit(self.vllm_dir, expected)
                git_ops.checkout(self.vllm_dir, target)
            except Exception as exc:  # noqa: BLE001 - add context for bisect logs
                raise VersionAdaptationError(f"could not checkout vLLM {expected}: {exc}") from exc
            command = [
                "pip",
                "install",
                "-e",
                str(self.vllm_dir),
                "--no-deps",
                "--no-input",
                "--disable-pip-version-check",
            ]
        else:
            command = [
                "pip",
                "install",
                f"vllm=={expected.lstrip('v')}",
                "--no-input",
                "--disable-pip-version-check",
            ]
        self._run(command, log_file, f"install vllm {expected}")
        self._overrides[VLLM_PACKAGE] = expected
        os.environ["VLLM_VERSION"] = expected.lstrip("v")

    def _switch_torch_npu(self, expected: str, log_file: Path | None) -> None:
        requirement = expected if expected[:1] in "<>!=" else f"=={expected}"
        command = [
            "pip",
            "install",
            f"torch-npu{requirement}",
            "--force-reinstall",
            "--no-deps",
            "--no-input",
            "--disable-pip-version-check",
        ]
        self._run(command, log_file, f"install torch-npu {expected}")
        self._overrides[TORCH_NPU_PACKAGE] = expected

    @staticmethod
    def _run(command: list[str], log_file: Path | None, label: str) -> None:
        logger.info("[version] running: %s", " ".join(command))
        try:
            if log_file is not None:
                with open(log_file, "a", encoding="utf-8") as out:
                    proc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, text=True)
                tail = "(see version adaptation log)"
            else:
                proc = subprocess.run(command, capture_output=True, text=True)
                tail = (proc.stdout or "")[-2000:]
        except OSError as exc:
            raise VersionAdaptationError(f"Failed to execute command {' '.join(command)}: {exc}") from exc
        if proc.returncode != 0:
            raise VersionAdaptationError(f"{label} failed (rc={proc.returncode}):\n{tail}")


def _matches_expected(package: str, expected: str, installed: str | None) -> bool:
    if installed is None:
        return False
    if package == TORCH_NPU_PACKAGE and any(op in expected for op in ("<", ">", "=", "!")):
        try:
            return Version(installed) in SpecifierSet(expected)
        except InvalidVersion:
            return False
    return versions_equal(expected, installed)
