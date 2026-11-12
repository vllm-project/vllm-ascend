"""Exercise installer success and failure paths without CANN or downloads."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["", "download", "install", "extract", "extension"])
def test_batch_invariant_installer(tmp_path, failure):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required")
    mock_env = tmp_path / "mock_env.sh"
    mock_env.write_text(
        """uname() { echo aarch64; }
curl() {
    [[ "$FAIL_STAGE" != download ]] || return 22
    local output
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == -o ]]; then output="$2"; shift; fi
        shift
    done
    if [[ "$output" == *.run ]]; then
        printf '#!/bin/bash\n[[ "$FAIL_STAGE" != install ]]\n' > "$output"
    else
        touch "$output"
    fi
}
python() {
    [[ "$FAIL_STAGE" != extract ]] || return 1
    mkdir -p torch_ops_extension/batch_invariant_ops
    printf '#!/bin/bash\n[[ "$FAIL_STAGE" != extension ]]\n' > \
        torch_ops_extension/batch_invariant_ops/build_and_install.sh
}
""",
        encoding="utf-8",
    )
    installer = Path(__file__).resolve().parents[4] / "csrc/build_batch_invariant_ops.sh"
    env = {**os.environ, "BASH_ENV": mock_env.as_posix(), "FAIL_STAGE": failure}
    result = subprocess.run(
        [bash, str(installer), "ascend910b"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) == (failure == ""), result.stdout + result.stderr
    assert ("installation completed" in result.stdout) == (failure == "")
