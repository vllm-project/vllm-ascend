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
"""Runtime environment switching for auto-bisect."""

import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

from tools.bisect.env_table import UNKNOWN_VALUES, RuntimeEnv

logger = logging.getLogger(__name__)

DEFAULT_VLLM_REPO_DIR = "/vllm-workspace/vllm"
DEFAULT_VLLM_REMOTE_URL = "https://github.com/vllm-project/vllm.git"


class EnvSwitchError(RuntimeError):
    pass


def _run(cmd: list[str], log_file: Path | None, label: str, cwd: Path | None = None) -> None:
    logger.info("[env] running: %s", " ".join(cmd))
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as out:
            out.write(f"\n$ {' '.join(cmd)}\n")
            out.flush()
            proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=out, stderr=subprocess.STDOUT, text=True)
        tail = "(see env/build log)"
    else:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
        tail = (proc.stdout or proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        raise EnvSwitchError(f"{label} failed (rc={proc.returncode}):\n{tail}")


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise EnvSwitchError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _known(value: str) -> bool:
    return value.strip() not in UNKNOWN_VALUES


def _same_ref(current: str, target: str) -> bool:
    current = current.strip()
    target = target.strip()
    return bool(current and target and (current.startswith(target) or target.startswith(current)))


def _installed_package_version(*package_names: str) -> str:
    for name in package_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def _read_cann_version_from_info(info_file: Path) -> str:
    if not info_file.exists():
        return ""
    for line in info_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("version="):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def _installed_cann_version() -> str:
    env = os.getenv("CANN_VERSION")
    if env:
        return env.strip()
    machine = platform.machine()
    ascend_home = Path(os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    for info_file in (
        ascend_home / f"{machine}-linux" / "ascend_toolkit_install.info",
        ascend_home / "ascend_toolkit_install.info",
    ):
        version = _read_cann_version_from_info(info_file)
        if version:
            return version
    return ""


def _source_env_file(source_file: Path) -> None:
    script = f"source {source_file} >/dev/null 2>&1; env -0"
    proc = subprocess.run(["bash", "-c", script], capture_output=True)
    if proc.returncode != 0:
        raise EnvSwitchError(f"source CANN env failed: {source_file}")
    for chunk in proc.stdout.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        os.environ[key.decode()] = value.decode(errors="ignore")


def _cann_source_candidates(version: str) -> list[Path]:
    roots = [
        Path(f"/usr/local/Ascend/cann-{version}"),
        Path(f"/usr/local/Ascend/ascend-toolkit/{version}"),
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "set_env.sh",
                root / "ascend-toolkit" / "set_env.sh",
                root / "share" / "info" / "ascendnpu-ir" / "bin" / "set_env.sh",
            ]
        )
    return candidates


def _is_shallow_repo(repo: Path) -> bool:
    return _git(repo, "rev-parse", "--is-shallow-repository", check=False).strip() == "true"


def _current_vllm_ref(vllm_repo: Path) -> str:
    if vllm_repo.exists():
        current = _git(vllm_repo, "rev-parse", "HEAD", check=False)
        if current:
            return current
    return os.getenv("VLLM_VERSION", "")


class EnvironmentManager:
    def __init__(self, vllm_repo_dir: str | None = None, vllm_remote_url: str | None = None):
        self.vllm_repo = Path(vllm_repo_dir or os.getenv("VLLM_REPO_DIR", DEFAULT_VLLM_REPO_DIR))
        self.vllm_remote_url = vllm_remote_url or os.getenv("VLLM_REMOTE_URL", DEFAULT_VLLM_REMOTE_URL)

    def ensure(self, target: RuntimeEnv | None, log_file: Path | None = None) -> bool:
        """Make the current process env match ``target``.

        Returns True when something changed, so callers can rebuild vllm-ascend
        against the newly selected runtime.
        """
        if target is None or target.is_empty:
            return False
        changed = False
        changed |= self._ensure_cann(target.cann_version)
        changed |= self._ensure_torch_npu(target.torch_npu_version, log_file)
        changed |= self._ensure_vllm(target.vllm_ref, log_file)
        return changed

    def _ensure_vllm(self, target_ref: str, log_file: Path | None) -> bool:
        if not _known(target_ref):
            return False
        current = _current_vllm_ref(self.vllm_repo)
        if _same_ref(current, target_ref):
            logger.info("[env] vLLM already matches %s", target_ref[:12])
            return False
        if not self.vllm_repo.exists():
            self.vllm_repo.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", self.vllm_remote_url, str(self.vllm_repo)], log_file, "clone vLLM")
        _git(self.vllm_repo, "fetch", "--tags", "origin", check=False)
        if _is_shallow_repo(self.vllm_repo):
            _git(self.vllm_repo, "fetch", "--unshallow", "origin", check=False)
        _git(self.vllm_repo, "fetch", "origin", target_ref, check=False)
        try:
            _git(self.vllm_repo, "checkout", "--force", target_ref)
        except EnvSwitchError:
            _git(self.vllm_repo, "checkout", "--force", "FETCH_HEAD")
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                str(self.vllm_repo),
                "--no-input",
                "--disable-pip-version-check",
            ],
            log_file,
            "install vLLM from source",
        )
        os.environ["VLLM_VERSION"] = target_ref
        return True

    def _ensure_torch_npu(self, version: str, log_file: Path | None) -> bool:
        if not _known(version):
            return False
        current = _installed_package_version("torch-npu", "torch_npu")
        if current == version:
            logger.info("[env] torch-npu already matches %s", version)
            return False
        package = os.getenv("BISECT_TORCH_NPU_PACKAGE", "torch-npu")
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"{package}=={version}",
                "--force-reinstall",
                "--no-input",
                "--disable-pip-version-check",
            ],
            log_file,
            "install torch-npu",
        )
        os.environ["TORCH_NPU_VERSION"] = version
        return True

    def _ensure_cann(self, version: str) -> bool:
        if not _known(version):
            return False
        current = _installed_cann_version()
        if current == version:
            logger.info("[env] CANN already matches %s", version)
            return False
        for source_file in _cann_source_candidates(version):
            if source_file.exists():
                _source_env_file(source_file)
                os.environ["CANN_VERSION"] = version
                logger.info("[env] sourced CANN %s from %s", version, source_file)
                return True
        raise EnvSwitchError(
            f"CANN {version} is not available under /usr/local/Ascend; "
            "the daily image cannot be changed, so mount/install this CANN runtime before bisecting"
        )
