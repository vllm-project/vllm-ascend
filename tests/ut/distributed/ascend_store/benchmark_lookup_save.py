# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare lookup/save CPU overhead against a Git revision, without NPU I/O.

Run in the unit-test environment from the repository root:
    python -m tests.ut.distributed.ascend_store.benchmark_lookup_save \
        --baseline origin/main --hash-blocks 32768 --repeat 7

Only baseline classes/methods are compiled; both versions use identical buffer
metadata and a synthetic backend. Results are CPU timings, not serving gains.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import coordinator, kv_transfer, metadata, pool_worker
# isort: on


class _Backend:
    requires_exists_before_put = True

    def __init__(self, present: bool):
        self.present = present
        self.saved = []

    def exists(self, keys):
        return [int(self.present)] * len(keys)

    def put(self, keys, addrs, sizes):
        self.saved.append((keys, addrs, sizes))


def _baseline_definition(ref, module, class_name, method_name=None):
    repo = Path(__file__).resolve().parents[4]
    path = f"vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/{module.__name__.rsplit('.', 1)[1]}.py"
    source = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    definition = next(
        node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    if method_name is not None:
        definition = next(
            node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    tree = ast.fix_missing_locations(
        ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definition],
            type_ignores=[],
        )
    )
    namespace = vars(module).copy()
    exec(compile(tree, f"{ref}:{path}", "exec"), namespace)
    return namespace[method_name or class_name]


def _workloads(db_cls, sender_cls, lookup, hashes):
    sizes = [128, 128, 512, 512, 16384, 16384]
    token_len = len(hashes) * 128
    db = db_cls([metadata.KeyMetadata("benchmark", 0, 0, 0)] * len(sizes), sizes, None, hash_block_size=128)
    db.set_group_buffers(
        {i: [1000 * (i + 1)] for i in range(len(sizes))},
        {i: [size] for i, size in enumerate(sizes)},
        group_num_layers={i: 1 for i in range(len(sizes))},
    )
    coord = coordinator.AscendStoreCoordinator(
        [
            KVCacheGroupSpec(
                [f"layer.{i}"],
                FullAttentionSpec(block_size=size, num_kv_heads=1, head_size=1, dtype=torch.float32),
            )
            for i, size in enumerate(sizes)
        ],
        16384,
        128,
        sizes,
        ["c1", "c1", "c4", "c4", "c128", "c128"],
    )
    masks = tuple([False] * (token_len // size - 2) + [True, True] if i < 2 else None for i, size in enumerate(sizes))
    worker = object.__new__(pool_worker.KVPoolWorker)
    worker.num_kv_cache_groups = len(sizes)
    worker.cache_coordinator = coord
    worker.token_database = db
    worker.m_store = _Backend(True)

    def run_lookup():
        with patch.object(coord, "lookup_mask", return_value=masks):
            return lookup(worker, token_len, hashes, list(range(len(sizes))), False, False, token_len - 16384)

    backend = _Backend(False)
    sender = sender_cls(
        backend, db, sizes, tp_rank=0, put_step=2, pcp_rank=0, pcp_size=2, ready_event=threading.Event()
    )
    store_masks = tuple([True] * (token_len // size) if mask is None else mask for size, mask in zip(sizes, masks))
    request = metadata.ReqMeta(
        req_id="benchmark",
        token_len_chunk=token_len,
        block_ids_by_group=[list(range(1, 17)) for _ in sizes],
        block_hashes=hashes,
        kv_cache_group_ids=list(range(len(sizes))),
    )

    def run_save():
        backend.saved.clear()
        with patch.object(db, "store_mask", return_value=store_masks):
            sender._handle_stored_request(request)
        return backend.saved.copy()

    return run_lookup, run_save


def _median_ms(workload, repeat):
    workload()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        workload()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hash-blocks", type=int, default=32768)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if args.hash_blocks < 256 or args.hash_blocks % 128 or args.repeat <= 0:
        parser.error("hash-blocks must be a multiple of 128 and at least 256; repeat must be positive")
    hashes = [i.to_bytes(32, "big").hex() for i in range(args.hash_blocks)]
    before = _workloads(
        _baseline_definition(args.baseline, metadata, "ChunkedTokenDatabase"),
        _baseline_definition(args.baseline, kv_transfer, "KVCacheStoreSendingThread"),
        _baseline_definition(args.baseline, pool_worker, "KVPoolWorker", "_lookup_with_coordinator"),
        hashes,
    )
    after = _workloads(
        metadata.ChunkedTokenDatabase,
        kv_transfer.KVCacheStoreSendingThread,
        pool_worker.KVPoolWorker._lookup_with_coordinator,
        hashes,
    )
    results = {"hash_blocks": args.hash_blocks, "groups": 6, "backend": "synthetic; no NPU I/O"}
    for name, old, new in zip(("lookup", "save"), before, after, strict=True):
        assert old() == new(), f"{name} results differ from baseline"
        old_ms, new_ms = _median_ms(old, args.repeat), _median_ms(new, args.repeat)
        results[name] = {"before_ms": old_ms, "after_ms": new_ms, "speedup": old_ms / new_ms}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
