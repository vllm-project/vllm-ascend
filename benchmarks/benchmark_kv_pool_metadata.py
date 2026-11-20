# SPDX-License-Identifier: Apache-2.0
"""Compare actual KV-pool CPU preparation methods with a mock store.

Run in an installed vLLM/Ascend environment; no model, store service, or NPU
operation is created. Both revisions use the same dependency stack and run
CPU metadata preparation in separate processes.

Example:
    python benchmarks/benchmark_kv_pool_metadata.py \
        --baseline-root /tmp/kv-pool-baseline \
        --candidate-root /tmp/kv-pool-candidate --output /tmp/kv-pool-perf.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import platform
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEDULER_PATH = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"
WORKER_PATH = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/pool_worker.py"
GVA_PATH = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/planes/gva.py"


def load_source(root: Path, relative_path: str) -> Any:
    path = (root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module = import_module(relative_path.removesuffix(".py").replace("/", "."))
    if Path(module.__file__).resolve() != path:
        raise RuntimeError(f"Expected {path}, loaded {module.__file__}")
    return module


class MockInfo:
    def __init__(self, address: int) -> None:
        self.address = address

    def size(self) -> int:
        return 128

    def gva_list(self) -> list[int]:
        return [self.address]


class MockStore:
    """Immediate deterministic metadata/lease replies; counts real method calls."""

    def __init__(self, infos: dict[str, MockInfo]) -> None:
        self.infos = infos
        self.calls: dict[str, int] = {}

    def batch_get_key_info(self, keys: list[str], **kwargs: Any) -> list[MockInfo]:
        self.calls["metadata_calls"] = self.calls.get("metadata_calls", 0) + 1
        self.calls["metadata_keys"] = self.calls.get("metadata_keys", 0) + len(keys)
        return [self.infos[key] for key in keys]

    def batch_add_lease(self, keys: list[str], ttl: int) -> list[int]:
        self.calls["lease_calls"] = self.calls.get("lease_calls", 0) + 1
        self.calls["lease_keys"] = self.calls.get("lease_keys", 0) + len(keys)
        return [0] * len(keys)


class SchedulerCase:
    def __init__(self, module: Any, count: int, all_cached: bool, *, unfinished_ids: list[str] | None = None) -> None:
        self.ids = [f"request-{index}" for index in range(count)]
        cached_ids = self.ids if all_cached else self.ids[::2]
        unfinished_ids = self.ids if unfinished_ids is None else unfinished_ids
        cached_id_set = set(cached_ids)
        self.expected_pending = [req_id for req_id in unfinished_ids if req_id not in cached_id_set]
        self.calls: list[str] = []
        self.scheduler = module.KVPoolScheduler.__new__(module.KVPoolScheduler)
        for name, value in {
            "kv_role": "kv_both",
            "consumer_is_to_put": False,
            "can_put": False,
            "_request_trackers": {},
            "_unfinished_requests": {req_id: (None, []) for req_id in unfinished_ids},
            "_preempted_req_ids": set(),
            "_loading_req_ids": set(),
            "_async_loading_req_ids": set(),
        }.items():
            setattr(self.scheduler, name, value)
        self.scheduler._process_async_load_request = self.process_pending
        self.output = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=cached_ids),
        )

    def process_pending(self, req_id: str, *_: Any) -> None:
        self.calls.append(req_id)

    def reset(self) -> None:
        self.calls.clear()

    def run(self) -> Any:
        return self.scheduler.build_connector_meta(self.output)

    def verify(self, result: Any) -> dict[str, Any]:
        assert result.requests == []
        assert self.calls == self.expected_pending
        return {
            "pending_callbacks": len(self.calls),
            "order_sha256": hashlib.sha256("\n".join(self.calls).encode()).hexdigest(),
        }


class LeaseCase:
    def __init__(self, module: Any, count: int, gva_plane: Any) -> None:
        import numpy as np

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
            LoadSpec,
            ReqMeta,
            block_hash_to_str,
        )

        block_hashes = [index.to_bytes(8, "big") for index in range(count)]
        self.keys = [self.make_key(0, block_hash_to_str(block_hash)) for block_hash in block_hashes]
        self.addresses = np.arange(1_000_000, 1_000_000 + count * 128, 128, dtype=np.int64)
        self.store = MockStore(
            {key: MockInfo(int(address)) for key, address in zip(self.keys, self.addresses, strict=True)}
        )
        self.worker = module.LayerwiseKVPoolWorker.__new__(module.LayerwiseKVPoolWorker)
        for name, value in {
            "use_layerwise_transfer": True,
            "use_eagle": False,
            "num_kv_cache_groups": 1,
            "cacheable_group_ids": [0],
            "grouped_block_size": [128],
            "group_uses_align_state": [False],
            "hash_block_size": 128,
            "layerwise_offload": True,
            "layerwise_max_transfer_blocks": 0,
            "_step_load_keys": set(),
            "m_store": self.store,
            "_compute_reachable_load_masks": lambda *_: None,
            "_layerwise_load_snapshot_start": lambda *_: None,
            "_resolve_layerwise_load_partial": lambda *_: (None, None),
            "_make_layerwise_full_key": self.make_key,
        }.items():
            setattr(self.worker, name, value)
        load_owner = gva_plane(self.worker) if gva_plane is not None else self.worker
        self.prepare_load = getattr(load_owner, "_prepare_load_gvas_for_request", None) or load_owner._prepare_load_gvas
        ids = np.arange(count, dtype=np.int64)
        self.request = ReqMeta(
            req_id="benchmark-load",
            token_len_chunk=count * 128,
            target_token_len=count * 128,
            block_ids=ids.tolist(),
            block_ids_by_group=[ids.tolist()],
            block_ids_np=ids,
            block_ids_by_group_np=[ids],
            block_hashes=block_hashes,
            load_spec=LoadSpec(0, count * 128, True),
        )

    @staticmethod
    def make_key(group: int, block_hash: str) -> str:
        return f"benchmark-model@group:{group}@head_or_tp_rank:0@{block_hash}"

    def reset(self) -> None:
        self.store.calls.clear()
        self.worker._step_load_keys.clear()

    def run(self) -> Any:
        return self.prepare_load([self.request])

    def verify(self, result: Any) -> dict[str, Any]:
        import numpy as np

        assert result is None
        assert self.request.load_keys == self.keys
        np.testing.assert_array_equal(self.request.load_block_gvas_by_group_np[0], self.addresses)
        assert self.worker._step_load_keys == set(self.keys)
        assert self.store.calls == {
            "metadata_calls": 2,
            "metadata_keys": 2 * len(self.keys),
            "lease_calls": 1,
            "lease_keys": len(self.keys),
        }
        return {**self.store.calls, "gva_sha256": hashlib.sha256(self.addresses.tobytes()).hexdigest()}


def summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * 0.95
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return {
        "samples": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[low] + (ordered[high] - ordered[low]) * (position - low),
        "raw_ms": samples,
    }


def measure(case: SchedulerCase | LeaseCase, warmup: int, rounds: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmup):
        case.reset()
        fingerprint = case.verify(case.run())
    samples = []
    for _ in range(rounds):
        for _ in range(2 * iterations):
            case.reset()
            start = time.perf_counter_ns()
            result = case.run()
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            assert case.verify(result) == fingerprint
            samples.append(elapsed_ms)
    return {"timing": summarize(samples), "verified_operations": fingerprint}


def git_revision(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run_revision(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    logging.getLogger("vllm").setLevel(logging.WARNING)
    scheduler_module = load_source(root, SCHEDULER_PATH)
    worker_module = load_source(root, WORKER_PATH)
    gva_module = load_source(root, GVA_PATH) if (root / GVA_PATH).is_file() else None
    gva_plane = gva_module.GvaDataPlane if gva_module is not None else None
    counts = (args.warmup, args.rounds, args.iterations)
    cases = []
    for count in (128, 512, 1024, 4096):
        for all_cached in (True, False):
            cases.append(
                {
                    "phase": "build_connector_meta_filter",
                    "active_requests": count,
                    "all_cached": all_cached,
                    **measure(SchedulerCase(scheduler_module, count, all_cached), *counts),
                }
            )
    # Decoding can schedule many cached requests with no or very few unfinished
    # connector requests. Include early/late matches as well as pending misses.
    for count in (128, 4096):
        for unfinished_count in (0, 1, 4):
            placements = ("empty",) if unfinished_count == 0 else ("head", "tail", "pending")
            for placement in placements:
                if placement == "head":
                    unfinished_ids = [f"request-{index}" for index in range(unfinished_count)]
                elif placement == "tail":
                    unfinished_ids = [f"request-{index}" for index in range(count - unfinished_count, count)]
                else:
                    unfinished_ids = [f"pending-{index}" for index in range(unfinished_count)]
                cases.append(
                    {
                        "phase": "build_connector_meta_sparse",
                        "scheduled_cached_requests": count,
                        "unfinished_requests": unfinished_count,
                        "unfinished_placement": placement,
                        **measure(SchedulerCase(scheduler_module, count, True, unfinished_ids=unfinished_ids), *counts),
                    }
                )
    for count in (128, 512, 2048, 8192):
        cases.append(
            {
                "phase": "prepare_layerwise_load_gvas",
                "blocks": count,
                **measure(LeaseCase(worker_module, count, gva_plane), *counts),
            }
        )
    for name, module in tuple(sys.modules.items()):
        if name == "vllm_ascend" or name.startswith("vllm_ascend."):
            path = getattr(module, "__file__", None)
            if path is not None and not Path(path).resolve().is_relative_to(root):
                raise RuntimeError(f"Module {name} was loaded from outside {root}: {path}")
    source_paths = [SCHEDULER_PATH, WORKER_PATH]
    if gva_module is not None:
        source_paths.append(GVA_PATH)
    return {
        "source": {
            "root": str(root),
            "git_head": git_revision(root),
            "files": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths},
            "loaded_scheduler": inspect.getsourcefile(scheduler_module.KVPoolScheduler),
            "loaded_worker": inspect.getsourcefile(worker_module.LayerwiseKVPoolWorker),
            "loaded_gva": inspect.getsourcefile(gva_plane) if gva_plane is not None else None,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--baseline-root", type=Path)
    target.add_argument("--revision-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.rounds, args.iterations) < 1:
        parser.error("warmup, rounds and iterations must be positive")
    if args.revision_root is not None:
        with redirect_stdout(sys.stderr):
            report = run_revision(args.revision_root.resolve(), args)
        print(json.dumps(report))
        return
    if args.candidate_root is None or args.output is None:
        parser.error("--candidate-root and --output are required")

    reports = {}
    for label, root in (("baseline", args.baseline_root.resolve()), ("candidate", args.candidate_root.resolve())):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--revision-root",
                str(root),
                "--warmup",
                str(args.warmup),
                "--rounds",
                str(args.rounds),
                "--iterations",
                str(args.iterations),
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            check=True,
        )
        reports[label] = json.loads(result.stdout)

    cases = []
    for baseline, candidate in zip(reports["baseline"]["cases"], reports["candidate"]["cases"], strict=True):
        baseline_timing = baseline.pop("timing")
        candidate_timing = candidate.pop("timing")
        if baseline != candidate:
            raise RuntimeError(f"Benchmark inputs or verified operations differ: {baseline} vs {candidate}")
        cases.append({**baseline, "baseline": baseline_timing, "candidate": candidate_timing})
    report = {
        "scope": "Native source modules and real NumPy, immediate mock store; CPU preparation only",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "order": "baseline process, then candidate process",
        "warmup_per_case_variant": args.warmup,
        "rounds": args.rounds,
        "iterations_per_revision_per_round": 2 * args.iterations,
        "sources": {label: result["source"] for label, result in reports.items()},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
