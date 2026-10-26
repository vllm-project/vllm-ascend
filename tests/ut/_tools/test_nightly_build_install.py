# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/_e2e_nightly_single_node.yaml"


def test_pr_checkout_disables_isolation_for_native_extension(tmp_path):
    """Exercise the main-branch install command against PR-owned uv settings."""
    uv = os.environ.get("UV_TEST_BINARY") or shutil.which("uv")
    if not uv:
        pytest.skip("uv is required for the build-isolation regression")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    settings = project["tool"]["uv"]["pip"]
    base = (
        '[build-system]\nrequires = ["nightly-test-missing-build-dependency"]\n'
        'build-backend = "probe_backend"\nbackend-path = ["."]\n'
        '[project]\nname = "vllm-ascend"\nversion = "0.0.0"\n'
    )
    # The unavailable build dependency reproduces isolated-build resolution.
    # A backend sentinel identifies execution with the target interpreter.
    (tmp_path / "probe_backend.py").write_text(
        'import sys\ndef build_editable(*args, **kwargs):\n raise RuntimeError("TARGET_BUILD_ENV:" + sys.prefix)\n',
        encoding="utf-8",
    )
    project_file = tmp_path / "pyproject.toml"
    command = [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(tmp_path / "target"),
        "-e",
        ".",
        "--no-deps",
        "--no-index",
    ]
    project_file.write_text(base, encoding="utf-8")
    isolated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert isolated.returncode != 0
    assert "TARGET_BUILD_ENV:" not in isolated.stderr
    project_file.write_text(
        base
        + "\n[tool.uv.pip]\nno-build-isolation-package = "
        + json.dumps(settings["no-build-isolation-package"])
        + "\n",
        encoding="utf-8",
    )
    native = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert native.returncode != 0  # Intentional backend sentinel, no package is installed.
    assert "TARGET_BUILD_ENV:" + sys.prefix in native.stderr


@pytest.mark.parametrize(("build_status", "import_status"), [(17, 0), (0, 13), (0, 0)])
def test_install_failure_prevents_native_smoke_success(tmp_path, build_status, import_status):
    """Run the actual install shell, including the tee pipeline failure path."""
    bash = os.environ.get("BASH_TEST_BINARY") or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required for the install-shell regression")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["e2e-nightly"]["steps"]
    install = next(step for step in steps if step.get("id") == "install-ascend")
    yaml_test = next(step for step in steps if step.get("id") == "yaml-test")
    assert yaml_test["if"] == "${{ success() && inputs.config_file_path != '' }}"
    (tmp_path / "requirements.txt").write_text("torch==2.10.0\npybind11\n", encoding="utf-8")
    script = r"""git() { :; }
pip() { :; }
uv() {
    case " $* " in
        *" -e "*) echo "native build output"; return "$BUILD_STATUS" ;;
        *) return 0 ;;
    esac
}
python3() { touch smoke-started; return "$IMPORT_STATUS"; }
"""
    script += install["run"].replace("/tmp/test-logs", "test-logs")
    script += "\ntouch install-succeeded\n"
    env = dict(os.environ, BUILD_STATUS=str(build_status), IMPORT_STATUS=str(import_status))
    result = subprocess.run(
        [bash, "-e", "-s"], input=script, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert (result.returncode == 0) == (build_status == import_status == 0), result.stderr
    assert (tmp_path / "smoke-started").exists() == (build_status == 0)
    assert (tmp_path / "install-succeeded").exists() == (build_status == import_status == 0)
    assert "native build output" in (tmp_path / "test-logs/ascend-install.log").read_text()
