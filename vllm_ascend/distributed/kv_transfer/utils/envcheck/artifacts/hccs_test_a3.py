#!/usr/bin/env python3
"""Offline A3 single-node HCCL all-to-all correctness probe."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys
import time
import traceback
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3 single-node HCCL all-to-all data correctness test")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--elements-per-peer", type=int, default=16384)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.world_size < 2:
        parser.error("--world-size must be at least 2")
    if args.elements_per_peer <= 0 or args.iterations <= 0 or args.timeout_seconds <= 0:
        parser.error("Test parameters must be positive integers")
    return args


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def worker(rank: int, world_size: int, elements_per_peer: int, iterations: int, timeout_seconds: int, master_port: int, queue: Any) -> None:
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401 - registers the NPU backend on torch

    started = time.monotonic()
    result: Dict[str, Any] = {"rank": rank, "ok": False}
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        device = f"npu:{rank}"
        for iteration in range(iterations):
            blocks = [
                torch.full(
                    (elements_per_peer,),
                    float(iteration * world_size * world_size + rank * world_size + target),
                    dtype=torch.float32,
                    device=device,
                )
                for target in range(world_size)
            ]
            send = torch.cat(blocks)
            receive = torch.empty_like(send)
            dist.all_to_all_single(receive, send)
            expected = torch.cat([
                torch.full(
                    (elements_per_peer,),
                    float(iteration * world_size * world_size + source * world_size + rank),
                    dtype=torch.float32,
                    device=device,
                )
                for source in range(world_size)
            ])
            if not bool(torch.equal(receive, expected)):
                raise RuntimeError(f"rank {rank} iteration {iteration} data verification failed")
        torch.npu.synchronize()
        result.update(ok=True, elapsed_seconds=round(time.monotonic() - started, 3))
    except Exception as exc:  # noqa: BLE001 - this is a standalone diagnostic boundary
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-3000:],
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        queue.put(result)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    try:
        import torch
        import torch.multiprocessing as mp
        import torch_npu  # noqa: F401

        visible_devices = int(torch.npu.device_count())
        if visible_devices != args.world_size:
            raise RuntimeError(f"Visible NPU dies={visible_devices}, expected={args.world_size}")
        context = mp.get_context("spawn")
        queue = context.SimpleQueue()
        master_port = free_local_port()
        mp.spawn(
            worker,
            args=(args.world_size, args.elements_per_peer, args.iterations, args.timeout_seconds, master_port, queue),
            nprocs=args.world_size,
            join=True,
        )
        ranks: List[Dict[str, Any]] = sorted((queue.get() for _ in range(args.world_size)), key=lambda item: item["rank"])
        ok = len(ranks) == args.world_size and all(item.get("ok") for item in ranks)
        result = {
            "schema_version": "1.0",
            "ok": ok,
            "collective": "all_to_all_single",
            "world_size": args.world_size,
            "elements_per_peer": args.elements_per_peer,
            "iterations": args.iterations,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "ranks": ranks,
        }
    except Exception as exc:  # noqa: BLE001 - emit one machine-readable failure record
        result = {
            "schema_version": "1.0",
            "ok": False,
            "collective": "all_to_all_single",
            "world_size": args.world_size,
            "elements_per_peer": args.elements_per_peer,
            "iterations": args.iterations,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-3000:],
        }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
