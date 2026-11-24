#!/usr/bin/env python3
"""Launch one shell template per local DP rank."""

import argparse
import os
import signal
import subprocess
import sys
import time

POLL_INTERVAL = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Global data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pcp-size", type=int, default=1, help="Prefill context parallel size (default: 1).")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size (default: 1).")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local DP size; defaults to --dp-size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="First global DP rank on this node.")
    parser.add_argument("--dp-address", required=True, help="IP address of the DP master node.")
    parser.add_argument("--dp-rpc-port", type=int, default=12345, help="DP RPC port.")
    parser.add_argument("--vllm-start-port", type=int, default=9000, help="First API port on this node.")
    parser.add_argument("--template", "--script", default="./run_dp_template.sh", help="Shell template path.")
    args = parser.parse_args()

    if args.dp_size_local == -1:
        args.dp_size_local = args.dp_size
    if min(args.dp_size, args.dp_size_local, args.tp_size, args.pcp_size, args.pp_size) < 1:
        parser.error("Parallel sizes must be positive.")
    if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
        parser.error("Local DP ranks exceed the global DP range.")
    if not os.path.isfile(args.template):
        parser.error(f"Template does not exist: {args.template}")
    return args


def handle_sigterm(signum, frame):
    raise SystemExit()


def main():
    args = parse_args()
    devices_per_rank = args.tp_size * args.pcp_size * args.pp_size
    processes = []
    try:
        signal.signal(signal.SIGTERM, handle_sigterm)
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
            # Start all local ranks before waiting; each rank owns a process group.
            processes.append(subprocess.Popen(command, start_new_session=True))

        # Check every rank so a running rank cannot hide another rank's failure.
        while True:
            running = False
            for i, process in enumerate(processes):
                code = process.poll()
                if code is None:
                    running = True
                elif code != 0:
                    raise RuntimeError(f"DP rank {args.dp_rank_start + i} exited with code {code}")
            if not running:
                return 0
            time.sleep(POLL_INTERVAL)
    except BaseException:
        # Notify every recorded rank, including workers whose shell has exited.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        for process in processes:
            process.wait()
        raise


if __name__ == "__main__":
    sys.exit(main())
