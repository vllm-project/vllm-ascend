# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Temporary probe: A5 UB fabric topology needed by HIXL/mooncake.

The installation guide builds the UB topology from /etc/hixlep, /lib/route.conf
and /etc/hccl_rootinfo.json, and ``urma_admin show topo`` reports
``Failed to recv netlink msg`` / ``Failed to get ubagg topo`` when the stack is
not ready. The a5-8 pod exposes /dev/uburma and /dev/ummu but the pool runs die
during HIXL store init, so record what the documented check returns.
"""

import os
import shutil
import subprocess

import pytest

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
PROBE_TIMEOUT_SECONDS = 60
PROBE_PATHS = (
    "/etc/hixlep",
    "/lib/route.conf",
    "/etc/hccl_rootinfo.json",
    "/dev/uburma",
    "/dev/ummu",
)
PROBE_COMMANDS = (
    ["urma_admin", "show"],
    ["urma_admin", "show", "topo"],
)


def _run(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return f"{command[0]}: not installed"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"{' '.join(command)}: timeout after {PROBE_TIMEOUT_SECONDS}s"
    output = (completed.stdout + completed.stderr).strip()
    return f"{' '.join(command)}: rc={completed.returncode} output={output[:2000]!r}"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching",
    parallel="DP",
    deploy="pd_mix",
    hardware="A5",
)
def test_a5_ub_topology_probe() -> None:
    for path in PROBE_PATHS:
        print(f"[ub_probe] {path} exists={os.path.exists(path)}", flush=True)
    for command in PROBE_COMMANDS:
        print(f"[ub_probe] {_run(command)}", flush=True)
