# SPDX-License-Identifier: Apache-2.0
"""Standalone immutable Engram owner for A2 HIXL consumers."""

import argparse
import json
import signal
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True, help="Reachable control IPv4:port")
    parser.add_argument("--engine-endpoint", required=True, help="Distinct reachable HIXL IPv4:port")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library-path")
    parser.add_argument("--ready-file")
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41.engram.shared_rdma import SharedOwner

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    owner = SharedOwner(
        args.model,
        args.endpoint,
        args.engine_endpoint,
        device=args.device,
        library_path=args.library_path,
    )
    service = threading.Thread(target=owner.serve_forever, name="engram-owner")
    service.start()
    ready = dict(
        stage="POOL_READY",
        generation=owner.generation,
        endpoint=args.endpoint,
        metadata=owner.metadata,
        table_bytes=sum(t.numel() for t in owner.tables.values()),
    )
    if args.ready_file:
        path = Path(args.ready_file)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ready, indent=2))
        temporary.replace(path)
    print(json.dumps(ready), flush=True)
    previous_clients = None
    while not stop.wait(5):
        with owner.lock:
            clients = (len(owner.sessions), len(owner.retained))
        if clients != previous_clients:
            print(json.dumps(dict(stage="POOL_CLIENTS", sessions=clients[0], retained=clients[1])), flush=True)
            previous_clients = clients
    owner.shutdown()
    service.join()
    while True:
        with owner.lock:
            sessions, retained = len(owner.sessions), len(owner.retained)
        if not sessions and not retained:
            break
        print(json.dumps(dict(stage="POOL_DRAINING", sessions=sessions, retained=retained)), flush=True)
        threading.Event().wait(5)
    owner.server_close()
    print("POOL_CLOSED", flush=True)


if __name__ == "__main__":
    main()
