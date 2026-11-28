# SPDX-License-Identifier: Apache-2.0
"""Standalone immutable Engram owner for A5 CANN consumers."""

import argparse
import json
import os
import tempfile
import threading
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from cann_ops_transformer.ops.mc2.common.elastic_buffer import ElasticBuffer
from safetensors import safe_open

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.shared_cann import (
    BF16_ELEMS_PER_ROW,
    CODE_BYTES,
    MAX_READERS,
    _group,
    _owner_only_ranges,
)
from vllm_ascend.models.deepseek_v41.engram.shared_rdma import checkpoint_metadata


def _available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    else:
        raise RuntimeError("cannot determine available host memory")
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        limit_file, usage_file = Path(limit_path), Path(usage_path)
        if limit_file.is_file() and usage_file.is_file():
            limit = limit_file.read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit) - int(usage_file.read_text())))
            break
    return available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True, help="Reachable control IPv4:port")
    parser.add_argument("--readers", required=True, type=int)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.readers <= MAX_READERS:
        raise ValueError(f"CANN Engram owner supports 1-{MAX_READERS} fixed readers")
    torch.npu.set_device(args.device)
    meta = checkpoint_metadata(args.model)
    if any(v["code_dtype"] != "F8_E4M3" or v["scale_dtype"] != "F8_E8M0" for v in meta["tables"].values()):
        raise ValueError("CANN Engram owner requires FP8/E8M0 tables")
    root = Path(args.model)
    index = json.loads((root / meta["index_file"]).read_text())["weight_map"]
    fd, path = tempfile.mkstemp(prefix="engram-cann-owner-")
    os.close(fd)
    os.unlink(path)
    dist.init_process_group("gloo", init_method="file://" + path, rank=0, world_size=1, timeout=timedelta(seconds=1800))
    host, port = args.endpoint.rsplit(":", 1)
    store = dist.TCPStore(
        host,
        int(port),
        args.readers + 1,
        True,
        timeout=timedelta(seconds=1800),
        wait_for_workers=False,
    )
    store.set("reader-count", str(args.readers))
    store.set("metadata", json.dumps(meta))
    print("CANN_OWNER_LISTENING", flush=True)
    buffers = []
    groups = []
    try:
        store.wait([f"ready-{i}" for i in range(1, args.readers + 1)], timedelta(seconds=1800))
        for layer, spec in meta["tables"].items():
            for shard, (first, last) in enumerate(_owner_only_ranges(spec["rows"], args.readers + 1)):
                size = ElasticBuffer.get_engram_storage_size_hint(last - first, BF16_ELEMS_PER_ROW, torch.bfloat16)
                required = (last - first) * spec["row_bytes"] + size + 16 * 1024**3
                if _available_memory() < required:
                    raise MemoryError("insufficient host or cgroup memory for CANN Engram shard")
                packed = torch.empty((last - first, spec["row_bytes"]), dtype=torch.uint8, device="cpu")
                for suffix, start, end in (
                    ("weight", 0, CODE_BYTES),
                    ("scale", CODE_BYTES, spec["row_bytes"]),
                ):
                    key = f"layers.{layer}.engram.embed.{suffix}"
                    with safe_open(root / index[key], framework="pt", device="cpu") as source:
                        tensor = source.get_slice(key)
                        for row in range(first, last, 1 << 20):
                            stop = min(row + (1 << 20), last)
                            packed[row - first : stop - first, start:end].copy_(
                                tensor[row:stop].contiguous().view(torch.uint8)
                            )
                store.set(f"prepared-{layer}-{shard}", "1")
                group = _group(
                    store,
                    f"engram_{port}_{layer}_{shard}",
                    0,
                    args.readers + 1,
                    {0: 0},
                )
                groups.append(group)
                buf = ElasticBuffer(group, num_cpu_bytes=size, explicitly_destroy=True)
                buffers.append(buf)
                buf.engram_write(packed.view(torch.bfloat16), owner_only_num_entries=last - first)
                del packed
                print("CANN_OWNER_LAYER_READY", layer, shard, first, last, size, flush=True)
        store.wait([f"done-{i}" for i in range(1, args.readers + 1)], timedelta(seconds=86400))
        for buf in reversed(buffers):
            buf.destroy()
        for group in groups:
            dist.distributed_c10d._world.pg_group_ranks.pop(group, None)
        dist.destroy_process_group()
        print("CANN_OWNER_CLOSED", flush=True)
    except BaseException as exc:
        # A peer failure does not establish that device accesses have retired.
        print("CANN_OWNER_HOLD", repr(exc), flush=True)
        threading.Event().wait()


if __name__ == "__main__":
    main()
