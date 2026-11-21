#!/usr/bin/env python3
"""Launch one shell template per local DP rank.

The template receives devices, API port, DP size, DP rank, DP address,
DP RPC port, TP size, PCP size, and PP size, in that order.
"""

import argparse
import os
import shlex
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-size", type=int, required=True, help="Global data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pcp-size", type=int, default=1, help="Prefill context parallel size.")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local DP size; defaults to --dp-size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="First global DP rank on this node.")
    parser.add_argument("--dp-address", required=True, help="IP address of the DP master node.")
    parser.add_argument("--dp-rpc-port", type=int, default=12345, help="DP RPC port.")
    parser.add_argument("--vllm-start-port", type=int, default=9000, help="First API port on this node.")
    parser.add_argument("--template", "--script", default="./run_dp_template.sh", help="Shell template path.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting services.")
    args = parser.parse_args(argv)

    if args.dp_size_local == -1:
        args.dp_size_local = args.dp_size
    if min(args.dp_size, args.dp_size_local, args.tp_size, args.pcp_size, args.pp_size) < 1:
        parser.error("Parallel sizes must be positive.")
    if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
        parser.error("Local DP ranks exceed the global DP range.")
    if not os.path.isfile(args.template):
        parser.error(f"Template does not exist: {args.template}")
    return args


def main(argv=None):
    args = parse_args(argv)
    devices_per_rank = args.tp_size * args.pcp_size * args.pp_size
    processes = []

    for i in range(args.dp_size_local):
        dp_rank = args.dp_rank_start + i
        port = args.vllm_start_port + i
        devices = ",".join(str(x) for x in range(i * devices_per_rank, (i + 1) * devices_per_rank))
        command = [
            "bash",
            "-o",
            "pipefail",
            args.template,
            devices,  # $1: visible devices
            str(port),  # $2: API port
            str(args.dp_size),  # $3: global DP size
            str(dp_rank),  # $4: global DP rank
            args.dp_address,  # $5: DP address
            str(args.dp_rpc_port),  # $6: DP RPC port
            str(args.tp_size),  # $7: TP
            str(args.pcp_size),  # $8: PCP
            str(args.pp_size),  # $9: PP
        ]

        print(f"DP rank={dp_rank}, devices={devices}, API port={port}", flush=True)
        if args.dry_run:
            print(shlex.join(command))
        else:
            # Start all local ranks before waiting for any of them.
            processes.append(subprocess.Popen(command))

    # Wait for every rank and propagate failure to the launcher exit status.
    exit_codes = [process.wait() for process in processes]
    return 1 if any(code != 0 for code in exit_codes) else 0


if __name__ == "__main__":
    sys.exit(main())
