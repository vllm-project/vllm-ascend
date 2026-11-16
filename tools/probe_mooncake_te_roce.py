#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Reproduce the Ascend KV-pool TransferEngine handshake on bare Mooncake.

The A2 KV-pool e2e legs fail with::

    EndpointGenerator ... Device roce endpoint is unavailable, phy_device_id:7
    RuntimeError: TransferEngine initialization failed with ret_value: -1

That message comes from the Mooncake/HIXL transport, not from vLLM, so this
script drives the same ``TransferEngine.initialize()`` call the connector makes
(``vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py``)
without loading a model, a KV cache or any vllm_ascend code. It answers the
question a failing CI run cannot: is the host missing device RoCE, or is the
process environment misconfigured?

Two presets are compared, each in its own child process because a failed native
initialization cannot be retried in the same process (Mooncake leaks transport
resources and ``GlobalTE`` latches the failure):

``bare``
    Only ``HCCL_INTRA_ROCE_ENABLE=1`` with ``ASCEND_ENABLE_USE_FABRIC_MEM``
    removed, i.e. what the A2 pool test armed before CI reported the endpoint
    failure. HIXL derives the endpoint list itself (``AutoGenEndpointList``).

``roce-store``
    Additionally pins ``ASCEND_GLOBAL_RESOURCE_CONFIG`` to
    ``protocol_desc=["roce:device"]`` and sets ``HCCL_INTRA_PCIE_ENABLE=0``,
    the trio the repo's validated RoCE Store path uses
    (``tests/e2e/pull_request/four_card/test_ec_mooncake_cross_encoder_store.py``).

Usage, inside the CI container (or any NPU shell)::

    python tools/probe_mooncake_te_roce.py                 # run both presets
    python tools/probe_mooncake_te_roce.py --preset bare   # one preset, in-process
    python tools/probe_mooncake_te_roce.py --hostname 10.0.0.165

Exit code 0 means at least one preset initialized the transport. A non-zero
code means no preset worked: the devices have no usable RoCE endpoint, which is
a host/container provisioning problem (``/etc/hccn.conf`` visible in the
container, per-NPU RoCE IP and link up), not a test-side one.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PRESETS = ("bare", "roce-store")
# A failed initialize() can take a while on a misconfigured device; keep the
# child bounded so a hung driver does not block the probe.
CHILD_TIMEOUT_SECONDS = 180
ROCE_RESOURCE_CONFIG = json.dumps(
    {
        "comm_resource_config": {"protocol_desc": ["roce:device"]},
        "store": {"comm_resource_config": {"protocol_desc": ["roce:device"]}},
    },
    separators=(",", ":"),
)
HUGEPAGES_2MB_PATH = Path("/proc/sys/vm/nr_hugepages")
HCCN_CONF_PATH = Path("/etc/hccn.conf")


def _preset_env(preset: str) -> dict[str, str]:
    """Return the environment a preset arms on top of the inherited one."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
    env = dict(os.environ)
    # Mooncake's native code treats the presence of this variable as true even
    # when it is "0", so it has to be removed, not set to a false value.
    env.pop("ASCEND_ENABLE_USE_FABRIC_MEM", None)
    env["HCCL_INTRA_ROCE_ENABLE"] = "1"
    if preset == "roce-store":
        env["ASCEND_GLOBAL_RESOURCE_CONFIG"] = ROCE_RESOURCE_CONFIG
        env["HCCL_INTRA_PCIE_ENABLE"] = "0"
    else:
        env.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        env.pop("HCCL_INTRA_PCIE_ENABLE", None)
    return env


def _attempt_transfer_engine(hostname: str, device_name: str) -> int:
    """Run the connector's initialize() call and return its native status."""
    try:
        from mooncake.engine import TransferEngine  # type: ignore[import-not-found]
    except ImportError as error:
        print(f"mooncake is not importable in this interpreter: {error}", file=sys.stderr)
        return -2
    engine = TransferEngine()
    return engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)


def _print_environment(hostname: str, device_name: str, preset: str) -> None:
    """Print the inputs that decide whether device RoCE can be built."""
    print(f"preset={preset} hostname={hostname!r} device_name={device_name!r}")
    print(f"  python={platform.python_version()} platform={platform.platform()}")
    for name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "HCCL_INTRA_ROCE_ENABLE",
        "HCCL_INTRA_PCIE_ENABLE",
        "ASCEND_ENABLE_USE_FABRIC_MEM",
        "ASCEND_GLOBAL_RESOURCE_CONFIG",
        "MOONCAKE_CONFIG_PATH",
    ):
        print(f"  {name}={os.environ.get(name)!r}")
    print(f"  {HCCN_CONF_PATH} exists={HCCN_CONF_PATH.exists()}")
    try:
        print(f"  {HUGEPAGES_2MB_PATH} = {HUGEPAGES_2MB_PATH.read_text(encoding='utf-8').strip()}")
    except OSError as error:
        print(f"  {HUGEPAGES_2MB_PATH} unreadable ({error})")


def _run_child(preset: str, hostname: str, device_name: str) -> tuple[bool, str]:
    """Run one preset in a fresh process; return (ok, captured output)."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preset",
        preset,
        "--hostname",
        hostname,
        "--device-name",
        device_name,
    ]
    try:
        completed = subprocess.run(
            command,
            env=_preset_env(preset),
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"preset {preset}: initialize() did not return within {CHILD_TIMEOUT_SECONDS}s"
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, f"preset {preset}: exit={completed.returncode}\n{output}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=PRESETS, help="run one preset in this process instead of both")
    parser.add_argument("--hostname", default="127.0.0.1", help="TransferEngine local hostname (default: 127.0.0.1)")
    parser.add_argument("--device-name", default="", help="TransferEngine device name (default: empty, as vLLM passes)")
    args = parser.parse_args()

    if args.preset is not None:
        # Resolve the preset against the inherited environment first: the
        # parent already armed it, and this keeps a hand-run preset identical.
        resolved = _preset_env(args.preset)
        os.environ.clear()
        os.environ.update(resolved)
        _print_environment(args.hostname, args.device_name, args.preset)
        status = _attempt_transfer_engine(args.hostname, args.device_name)
        print(f"TransferEngine.initialize() -> {status} (0 means the transport is usable)")
        return 0 if status == 0 else 1

    print(f"Comparing {len(PRESETS)} presets; each runs in its own process.\n")
    results = [_run_child(preset, args.hostname, args.device_name) for preset in PRESETS]
    for ok, report in results:
        print(f"[{'OK' if ok else 'FAIL'}] {report}\n")
    if any(ok for ok, _ in results):
        print("At least one preset initialized the Ascend transport; compare the presets above.")
        return 0
    print(
        "No preset initialized the Ascend transport. The NPUs visible to this process expose no usable "
        "RoCE endpoint: verify /etc/hccn.conf is mounted, the per-NPU RoCE IP and link are up "
        "(hccn_tool -i <phy_id> -ip -g / -link -g), and inspect the CANN plog under "
        "/root/ascend/log/debug/plog.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
