# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="Requires a POSIX shell")
@pytest.mark.parametrize(
    ("soc_version", "soc_arg", "has_sfa"),
    [
        ("ascend950", "ascend950", True),
        ("ascend950dt", "ascend950", True),
        ("ascend910b", "ascend910b", True),
        ("ascend910_93", "ascend910_93", True),
        ("ascend310p", "ascend310p", False),
    ],
)
def test_custom_op_build_selects_sparse_flash_attention(tmp_path, soc_version, soc_arg, has_sfa):
    """Exercise the real build entry point without a compiler or a CANN install."""
    script = Path(__file__).resolve().parents[2] / "csrc" / "build_aclnn.sh"
    root = tmp_path / "repo"
    csrc = root / "csrc"
    (csrc / "third_party" / "catlass" / "include").mkdir(parents=True)
    # Do not let dependency setup modify the user's global git configuration.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/bash\nexit 0\n")
    git.chmod(0o755)
    (csrc / "build.sh").write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" > build_args.txt\n"
        "mkdir -p build\n"
        "printf '#!/bin/bash\\nexit 0\\n' > build/cann-ops-transformer-test.run\n"
    )
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    env["VLLM_BATCH_INVARIANT"] = "0"
    completed = subprocess.run(
        ["bash", str(script), str(root), soc_version],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    args = (csrc / "build_args.txt").read_text().splitlines()
    assert f"--soc={soc_arg}" in args
    ops = next(arg.removeprefix("--ops=") for arg in args if arg.startswith("--ops=")).split(";")
    assert ("sparse_flash_attention" in ops) == has_sfa
    assert len(ops) == len(set(ops))
