# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ensure a source checkout can import the package before build metadata exists."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_source_checkout_imports_without_generated_build_info(tmp_path):
    package = tmp_path / "vllm_ascend"
    package.mkdir()
    source_root = Path(__file__).parents[2]
    shutil.copy2(source_root / "vllm_ascend/__init__.py", package / "__init__.py")
    shutil.copy2(source_root / "vllm_ascend/logger.py", package / "logger.py")
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    (vllm / "__init__.py").write_text("")
    (vllm / "envs.py").write_text("")
    (vllm / "logger.py").write_text("import logging\n_METHODS_TO_PATCH = {}\n_VllmLogger = logging.Logger\n")
    (vllm / "logging_utils.py").write_text(
        "import logging\nColoredFormatter = logging.Formatter\nNewLineFormatter = logging.Formatter\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_ascend; print(vllm_ascend._build_info.__soc_version__)"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
