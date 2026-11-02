# SPDX-License-Identifier: Apache-2.0
"""Launch one arm of the saved A5 C8/MTP3 single-D comparison."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("on", "off"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8029)
    parser.add_argument("--host-ip", help="Actual data-network IP for this host, when required")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("reference_configuration.json").read_text())
    argv = list(config[args.arm])
    for flag, value in [("--model", args.model), ("--host", args.host), ("--port", str(args.port))]:
        argv[argv.index(flag) + 1] = value
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *argv]
    if args.print_command:
        print(json.dumps(command, indent=2))
        return
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ASCEND_HISPARSE_PROFILE"] = "a5_c8_62592"
    if args.host_ip:
        os.environ["VLLM_HOST_IP"] = args.host_ip
    from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import optimized_profile_enabled

    optimized_profile_enabled()
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
