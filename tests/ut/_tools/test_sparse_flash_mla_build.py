# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "soc,version,expected",
    [
        ("ascend950pr_9579", "9.2.0", 1024),
        ("ascend950pr_9579", "9.2.1", 1024),
        ("ascend950pr_9579", "9.3.0", 2048),
        ("ascend950dt_9599", "9.3.0", 2048),
        ("ascend910b1", "9.3.0", 1024),
        ("ascend910_9391", "9.3.0", 1024),
        ("ascend950pr_9579", None, 1024),
        ("ascend950pr_9579", "invalid", None),
    ],
)
def test_sparse_flash_mla_metadata_build_contract(tmp_path, soc, version, expected):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is unavailable")
    # Component upgrades can leave the toolkit installation directory unchanged.
    cann_root = tmp_path / "cann-9.2.1"
    version_file = cann_root / "share/info/ops_transformer/version.info"
    if version is not None:
        version_file.parent.mkdir(parents=True)
        version_file.write_text(f"Version={version}\n", encoding="utf-8")
    result = tmp_path / "size.txt"
    script = tmp_path / "check.cmake"
    module = REPO_ROOT / "cmake/sparse_flash_mla.cmake"
    script.write_text(
        f'include("{module.as_posix()}")\n'
        f'get_sparse_flash_mla_metadata_size(SIZE "{soc}" "{cann_root.as_posix()}")\n'
        f'file(WRITE "{result.as_posix()}" "${{SIZE}}")\n',
        encoding="utf-8",
    )
    proc = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    if expected is None:
        assert proc.returncode != 0
        assert "Cannot read ops-transformer version" in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
        assert int(result.read_text(encoding="utf-8")) == expected
