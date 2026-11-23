# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise selected-test reporting with real pytest exit codes."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).with_name("run_selected_tests.sh")
BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="selected-tests runner requires Bash")
@pytest.mark.parametrize(
    ("source", "exit_code", "summary"),
    [
        pytest.param(
            "import pytest\npytest.skip('unavailable', allow_module_level=True)\n", 0, "SKIPPED", id="module-skip"
        ),
        pytest.param("import pytest\ndef test_skip():\n    pytest.skip('unavailable')\n", 0, "SKIPPED", id="item-skip"),
        pytest.param(
            "import pytest\n@pytest.mark.xfail\ndef test_xfail():\n    assert False\n", 0, "PASSED", id="xfail"
        ),
        pytest.param("pass\n", 5, "FAILED", id="empty"),
        pytest.param(
            "import pytest\ndef test_fail():\n    assert False\ndef test_skip():\n    pytest.skip('unavailable')\n",
            1,
            "FAILED",
            id="failure-and-skip",
        ),
        pytest.param(
            "import pytest\ndef test_pass():\n    pass\ndef test_skip():\n    pytest.skip('unavailable')\n",
            0,
            "PASSED",
            id="pass-and-skip",
        ),
    ],
)
def test_selected_tests_summary(tmp_path, source, exit_code, summary):
    (tmp_path / "test_first.py").write_text(source, encoding="utf-8")
    (tmp_path / "test_next.py").write_text("def test_next():\n    pass\n", encoding="utf-8")
    python_bin = str(Path(sys.executable).parent)
    env = {
        **os.environ,
        "CI": "false",
        "ENABLE_COVERAGE": "false",
        "PATH": f"{python_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "RUNNER_TEMP": tmp_path.as_posix(),
    }
    result = subprocess.run(
        [BASH, RUNNER.resolve().as_posix(), "a2", "1", "with-device", "test_first.py", "test_next.py"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert result.returncode == exit_code, result.stdout
    assert f"{summary}: test_first.py" in result.stdout
    assert ("PASSED: test_next.py" in result.stdout) == (exit_code == 0)
