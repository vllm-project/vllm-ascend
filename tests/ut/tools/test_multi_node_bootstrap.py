import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "e2e/common/multi_node/run.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required")


def shell_function(name):
    source = SCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}", source)
    assert match, f"Missing shell function: {name}"
    return match.group()


@pytest.mark.parametrize("exit_code", [0, 23])
def test_bootstrap_exit_is_visible_to_log_stream(exit_code):
    program = "\n".join(
        [
            "set -euo pipefail",
            shell_function("report_bootstrap_failure"),
            shell_function("finish_without_aop"),
            "backup_ascend_logs() { return 1; }",
            "backup_requested_directory() { return 1; }",
            "FAIL_TAG=FAIL_TAG_test.yaml",
            "trap finish_without_aop EXIT",
            f"exit {exit_code}",
        ]
    )
    result = subprocess.run([BASH, "-c", program], text=True, capture_output=True)
    assert result.returncode == exit_code
    assert ("FAIL_TAG_test.yaml" in result.stdout) == bool(exit_code)


def test_early_bootstrap_failure_is_visible():
    prefix = SCRIPT.read_text(encoding="utf-8").split("BISECT_SOC=", 1)[0]
    result = subprocess.run(
        [BASH, "-c", prefix + "\nFAIL_TAG=FAIL_TAG_early.yaml\nexit 17\n"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 17
    assert "FAIL_TAG_early.yaml" in result.stdout


@pytest.mark.parametrize("primary_status,fallback_status", [(0, 0), (1, 0), (1, 19)])
def test_git_network_fallback(primary_status, fallback_status):
    program = "\n".join(
        [
            shell_function("git_network"),
            "git() {",
            '  printf "config=%s\\n" "${GIT_CONFIG_GLOBAL:-default}"',
            f'  if [ "${{GIT_CONFIG_GLOBAL:-}}" = /dev/null ]; then return {fallback_status}; fi',
            f"  return {primary_status}",
            "}",
            "unset GIT_CONFIG_GLOBAL",
            "git_network fetch origin test-ref",
        ]
    )
    result = subprocess.run([BASH, "-c", program], text=True, capture_output=True)
    assert result.returncode == (fallback_status if primary_status else 0)
    assert result.stdout.splitlines() == (
        ["config=default", "config=/dev/null"] if primary_status else ["config=default"]
    )


@pytest.mark.parametrize("missing_pod", ["test-0-1", "test-0"])
def test_pod_readiness_retries_transient_lookup_failure(tmp_path, missing_pod):
    root = SCRIPT.parents[4]
    workflow = yaml.safe_load((root / ".github/workflows/_e2e_nightly_multi_node.yaml").read_text(encoding="utf-8"))
    wait_script = next(
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Wait for pods ready"
    ).replace("${{ inputs.size }}", "2")
    program = (
        r"""
NAMESPACE=test
LWS_NAME=test
LEADER_POD=test-0
kubectl() {
    if [[ "$*" == *containerStatuses* ]]; then
        if [ "$3" = "$MISSING_POD" ] && [ ! -f readiness-retried ]; then
            touch readiness-retried
            return 1
        fi
        echo true
    else
        echo Running
    fi
}
sleep() { :; }
"""
        + wait_script
    )
    result = subprocess.run(
        [BASH, "-c", program],
        cwd=tmp_path,
        env={**os.environ, "MISSING_POD": missing_pod},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "All pods are Running and Ready." in result.stdout
    assert (tmp_path / "readiness-retried").exists()
