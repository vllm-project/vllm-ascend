# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Tests for the ``examples/external_online_dp`` launcher.

The launcher is a user-facing example script, so it is exercised as a
subprocess against a stub ``run_dp_template.sh`` that records the positional
arguments it receives. This keeps the test independent of any NPU hardware
and of ``vllm serve``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _find_example_dir() -> Path:
    """Locate ``examples/external_online_dp`` by walking up from this file."""
    override = os.environ.get("VLLM_ASCEND_EXTERNAL_ONLINE_DP_DIR")
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "examples" / "external_online_dp"
        if (candidate / "launch_online_dp.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"examples/external_online_dp/launch_online_dp.py not found above {Path(__file__).resolve()}"
    )


EXAMPLE_DIR = _find_example_dir()
LAUNCHER = EXAMPLE_DIR / "launch_online_dp.py"
TEMPLATE = EXAMPLE_DIR / "run_dp_template.sh"

# The stub records the arguments it was called with in a file named after its
# own PID. The launcher starts the ranks in parallel, so each rank writes its
# own file instead of appending to a shared one, which would race.
STUB_TEMPLATE = """#!/usr/bin/env bash
echo "$@" > "$RECORD_DIR/$$"
"""


def _run_launcher(tmp_path, extra_args, num_ranks=2):
    """Run the launcher with the stub template and return the recorded calls.

    Returns a list of argument lists, one per recorded invocation.
    """
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    template = tmp_path / "run_dp_template.sh"
    template.write_text(STUB_TEMPLATE)
    template.chmod(0o755)

    env = dict(os.environ)
    env["RECORD_DIR"] = str(record_dir)

    cmd = [
        sys.executable,
        str(LAUNCHER),
        "--dp-size",
        str(num_ranks),
        "--dp-size-local",
        str(num_ranks),
        "--dp-address",
        "127.0.0.1",
    ] + extra_args

    # The launcher calls "./run_dp_template.sh" relative to its CWD.
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    calls = []
    for record in sorted(record_dir.iterdir()):
        calls.append(record.read_text().split())
    # The ranks start in parallel, so order the calls by their data parallel
    # rank to make the assertions deterministic.
    calls.sort(key=lambda call: int(call[3]))
    return calls


def test_pcp_size_argument_is_accepted(tmp_path):
    """``--pcp-size`` must be a recognized launcher argument."""
    calls = _run_launcher(tmp_path, ["--pcp-size", "4"])
    assert len(calls) == 2


def test_default_dp_rpc_port_is_a_string(tmp_path):
    """The default ``--dp-rpc-port`` must not crash the subprocess call.

    ``subprocess.run`` rejects non-string arguments, so an int default makes
    every launcher invocation fail unless the user overrides the port.
    """
    calls = _run_launcher(tmp_path, [])
    for call in calls:
        assert call[5] == "12345"


def test_explicit_dp_rpc_port_is_forwarded(tmp_path):
    """An explicitly given ``--dp-rpc-port`` reaches the template."""
    calls = _run_launcher(tmp_path, ["--dp-rpc-port", "12399"])
    for call in calls:
        assert call[5] == "12399"


def test_pcp_size_defaults_to_one(tmp_path):
    """Without ``--pcp-size`` the template still receives a pcp size of 1."""
    calls = _run_launcher(tmp_path, [])
    for call in calls:
        assert call[7] == "1"


def test_pcp_size_is_passed_as_eighth_positional_argument(tmp_path):
    """The pcp size is forwarded to the template as the 8th argument."""
    calls = _run_launcher(tmp_path, ["--pcp-size", "4"])
    for call in calls:
        assert len(call) == 8
        assert call[7] == "4"


def test_template_argument_order(tmp_path):
    """The positional arguments keep the documented order."""
    calls = _run_launcher(
        tmp_path,
        ["--tp-size", "2", "--pcp-size", "4", "--dp-rank-start", "3", "--vllm-start-port", "9100"],
    )
    for index, call in enumerate(calls):
        visible_devices, port, dp_size, dp_rank = call[0:4]
        dp_address, dp_rpc_port, tp_size, pcp_size = call[4:8]
        assert visible_devices
        assert port == str(9100 + index)
        assert dp_size == "2"
        assert dp_rank == str(3 + index)
        assert dp_address == "127.0.0.1"
        assert dp_rpc_port == "12345"
        assert tp_size == "2"
        assert pcp_size == "4"


def test_visible_devices_account_for_pcp_size(tmp_path):
    """Each DP rank gets ``tp_size * pcp_size`` contiguous devices."""
    calls = _run_launcher(tmp_path, ["--tp-size", "1", "--pcp-size", "4"])
    devices = [call[0].split(",") for call in calls]
    assert devices == [["0", "1", "2", "3"], ["4", "5", "6", "7"]]


def test_visible_devices_without_pcp(tmp_path):
    """The default layout is unchanged: one device per rank when tp_size=1."""
    calls = _run_launcher(tmp_path, ["--tp-size", "1"])
    devices = [call[0].split(",") for call in calls]
    assert devices == [["0"], ["1"]]


def test_visible_devices_with_tp_only(tmp_path):
    """A tp_size of 4 with no PCP still gives each rank 4 devices."""
    calls = _run_launcher(tmp_path, ["--tp-size", "4"])
    devices = [call[0].split(",") for call in calls]
    assert devices == [["0", "1", "2", "3"], ["4", "5", "6", "7"]]


def test_visible_devices_with_tp_and_pcp(tmp_path):
    """``tp_size * pcp_size`` devices are handed to each rank."""
    calls = _run_launcher(tmp_path, ["--tp-size", "2", "--pcp-size", "2"])
    devices = [call[0].split(",") for call in calls]
    assert devices == [["0", "1", "2", "3"], ["4", "5", "6", "7"]]


@pytest.mark.parametrize(
    "template_body,expected_flag",
    [
        (
            "vllm serve m --port $2 --tensor-parallel-size $7 --prefill-context-parallel-size $8",
            "--prefill-context-parallel-size",
        ),
    ],
)
def test_repository_template_wires_pcp_flag(template_body, expected_flag):
    """The shipped template forwards ``$8`` to the PCP vLLM flag."""
    content = TEMPLATE.read_text()
    assert expected_flag in content
    assert "--prefill-context-parallel-size $8" in content


def test_repository_template_keeps_existing_flags():
    """The template keeps the flags it already had."""
    content = TEMPLATE.read_text()
    for flag in (
        "--data-parallel-size $3",
        "--data-parallel-rank $4",
        "--data-parallel-address $5",
        "--data-parallel-rpc-port $6",
        "--tensor-parallel-size $7",
    ):
        assert flag in content


def test_launcher_and_template_exist():
    """Guard against the example files being moved or renamed."""
    assert LAUNCHER.is_file()
    assert TEMPLATE.is_file()
