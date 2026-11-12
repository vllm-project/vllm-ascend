# SPDX-License-Identifier: Apache-2.0
"""Serve a Qwen PLE table through the shared Engram HIXL row protocol."""

import argparse
import json
import os
import signal
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--endpoint", required=True, help="Reachable control address:port")
    parser.add_argument("--engine-endpoint", required=True, help="Distinct reachable HIXL address:port")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library-path")
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.models.qwen4_exp.ple_rdma import PLERemoteOwner

    torch.npu.set_device(args.device)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    owner = PLERemoteOwner(
        args.model,
        args.prefix,
        args.parts,
        args.width,
        args.endpoint,
        args.engine_endpoint,
        device=args.device,
        library_path=args.library_path,
    )
    service = threading.Thread(target=owner.serve_forever, name="ple-rdma-owner")
    service.start()
    print(json.dumps({"stage": "PLE_RDMA_OWNER_READY", "pid": os.getpid(), "metadata": owner.metadata}), flush=True)
    stop.wait()
    owner.shutdown()
    service.join()
    while True:
        with owner.lock:
            sessions, retained = len(owner.sessions), len(owner.retained)
        if not sessions and not retained:
            break
        print(json.dumps({"stage": "PLE_RDMA_OWNER_DRAINING", "sessions": sessions, "retained": retained}), flush=True)
        threading.Event().wait(30)
    owner.server_close()
    print("PLE_RDMA_OWNER_CLOSED", flush=True)


if __name__ == "__main__":
    main()
