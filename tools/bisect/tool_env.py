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
"""Isolate bisect's build parser packages from the system model environment."""

import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

from tools.bisect.config import DEFAULT_WORK_DIR

_ANALYZER_REQUIREMENTS = Path(__file__).parent / "build_analyzer" / "requirements.txt"
_PIP_INDEX_URL = "https://mirrors.huaweicloud.com/repository/pypi/simple"


def ensure_tool_environment() -> None:
    """Re-enter the CLI with parser packages local to its own venv.

    System site packages stay visible so version discovery and psutil work.
    PATH is deliberately unchanged: pip, python and python3 invoked by the
    trial runner still target the container's model environment.
    """
    if os.name != "posix":
        return
    venv_dir = Path(DEFAULT_WORK_DIR).expanduser().resolve().parent / "tool_venv"
    python = venv_dir / "bin" / "python"
    stamp = venv_dir / ".requirements-sha256"
    # Include the installation policy so environments previously stamped after
    # --no-deps are repaired even when the requirements file is unchanged.
    required_hash = hashlib.sha256(b"with-dependencies-v1\n" + _ANALYZER_REQUIREMENTS.read_bytes()).hexdigest()

    in_tool_environment = Path(sys.prefix).resolve() == venv_dir
    if in_tool_environment and stamp.is_file() and stamp.read_text(encoding="ascii") == required_hash:
        return

    if not python.is_file():
        print(f"[bisect] creating tool Python environment: {venv_dir}", flush=True)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(venv_dir)
    if not stamp.is_file() or stamp.read_text(encoding="ascii") != required_hash:
        print(f"[bisect] installing analyzer requirements: {_ANALYZER_REQUIREMENTS}", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--ignore-installed", "-r", str(_ANALYZER_REQUIREMENTS)],
            check=True,
            env={**os.environ, "PIP_INDEX_URL": os.environ.get("PIP_INDEX_URL", _PIP_INDEX_URL)},
        )
        stamp.write_text(required_hash, encoding="ascii")

    if not in_tool_environment:
        os.execv(str(python), [str(python), "-m", "tools.bisect.auto_bisect", *sys.argv[1:]])
