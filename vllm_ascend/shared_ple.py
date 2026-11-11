# SPDX-License-Identifier: Apache-2.0
"""Start one immutable Qwen PLE owner backed by a shared BM pool."""

import argparse
import json
import os
import signal
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", required=True, help="Full checkpoint prefix of the PLE embedding table")
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--root", required=True, help="New UID-private shared-pool generation directory")
    parser.add_argument("--host", required=True, help="Reachable owner control address")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=65)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41.engram.shared_pool import SharedPoolOwner
    from vllm_ascend.models.qwen4_exp.ple_bm_table import load_table
    from vllm_ascend.models.qwen4_exp.ple_checkpoint import PLECheckpoint, checkpoint_identity

    torch.npu.set_device(args.device)
    checkpoint = PLECheckpoint(args.model, args.prefix, args.parts, args.width)
    identity = checkpoint_identity(args.model, args.prefix)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    owner = SharedPoolOwner(
        args.root, identity, device=f"npu:{args.device}", host=args.host, world_size=args.world_size
    )
    try:
        layout = load_table(owner, checkpoint, args.prefix)
        owner.ready()
        print(
            json.dumps(
                {"stage": "PLE_OWNER_READY", "pid": os.getpid(), "rows": layout.rows, "bytes": layout.allocation_bytes}
            ),
            flush=True,
        )
        stop.wait()
    finally:
        while True:
            try:
                owner.close(timeout=30)
                break
            except TimeoutError:
                print("PLE_OWNER_DRAINING: waiting for consumer leases", flush=True)
    print("PLE_OWNER_CLOSED", flush=True)


if __name__ == "__main__":
    main()
