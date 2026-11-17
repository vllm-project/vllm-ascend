# SPDX-License-Identifier: Apache-2.0
"""Compare actual KV-pool CPU preparation methods with a mock store.

Run in an installed vLLM/Ascend environment; no model, store service, or NPU
operation is created. Both source trees must use the same dependency stack.
This measures CPU metadata preparation, not a full scheduler tick or E2E serving.

Example:
    python benchmarks/benchmark_kv_pool_metadata.py \
        --baseline-root /tmp/kv-pool-baseline \
        --candidate-root /tmp/kv-pool-candidate --output /tmp/kv-pool-perf.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import logging
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEDULER_PATH = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"
WORKER_PATH = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/pool_worker.py"


def load_source(root: Path, relative_path: str, label: str) -> Any:
    path = root / relative_path
    name = f"kv_pool_benchmark_{label}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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
    def __init__(self, module: Any, count: int) -> None:
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
            "hash_block_size": 128,
            "layerwise_offload": True,
            "_step_load_keys": set(),
            "m_store": self.store,
            "_compute_reachable_load_masks": lambda *_: None,
            "_layerwise_load_snapshot_start": lambda *_: None,
            "_resolve_layerwise_load_partial": lambda *_: (None, None),
            "_make_layerwise_full_key": self.make_key,
        }.items():
            setattr(self.worker, name, value)
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
        return self.worker._prepare_load_gvas_for_request([self.request])

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


def measure(cases: dict[str, Any], warmup: int, rounds: int, iterations: int) -> dict[str, Any]:
    fingerprints = {}
    samples: dict[str, list[float]] = {label: [] for label in cases}
    for label, case in cases.items():
        for _ in range(warmup):
            case.reset()
            fingerprints[label] = case.verify(case.run())
    assert fingerprints["baseline"] == fingerprints["candidate"]
    for _ in range(rounds):
        for label in ("baseline", "candidate", "candidate", "baseline"):
            case = cases[label]
            for _ in range(iterations):
                case.reset()
                start = time.perf_counter_ns()
                result = case.run()
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                assert case.verify(result) == fingerprints[label]
                samples[label].append(elapsed_ms)
    return {
        **{label: summarize(values) for label, values in samples.items()},
        "verified_operations": fingerprints["baseline"],
    }


def git_revision(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.rounds, args.iterations) < 1:
        parser.error("warmup, rounds and iterations must be positive")
    roots = {"baseline": args.baseline_root.resolve(), "candidate": args.candidate_root.resolve()}
    sys.path.insert(0, str(roots["candidate"]))
    logging.getLogger("vllm").setLevel(logging.WARNING)
    modules = {
        label: (load_source(root, SCHEDULER_PATH, label), load_source(root, WORKER_PATH, label))
        for label, root in roots.items()
    }
    report: dict[str, Any] = {
        "scope": "Native source modules and real NumPy, immediate mock store; CPU preparation only, no NPU/E2E claim",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "order": "ABBA: baseline,candidate,candidate,baseline per round",
        "warmup_per_case_variant": args.warmup,
        "rounds": args.rounds,
        "iterations_per_ABBA_block": args.iterations,
        "sources": {},
        "cases": [],
    }
    for label, (scheduler_module, worker_module) in modules.items():
        report["sources"][label] = {
            "root": str(roots[label]),
            "git_head": git_revision(roots[label]),
            "files": {
                relative: hashlib.sha256((roots[label] / relative).read_bytes()).hexdigest()
                for relative in (SCHEDULER_PATH, WORKER_PATH)
            },
            "loaded_scheduler": inspect.getsourcefile(scheduler_module.KVPoolScheduler),
            "loaded_worker": inspect.getsourcefile(worker_module.LayerwiseKVPoolWorker),
        }
    cases: dict[str, Any]
    for count in (128, 512, 1024, 4096):
        for all_cached in (True, False):
            cases = {label: SchedulerCase(pair[0], count, all_cached) for label, pair in modules.items()}
            report["cases"].append(
                {
                    "phase": "build_connector_meta_filter",
                    "active_requests": count,
                    "all_cached": all_cached,
                    **measure(cases, args.warmup, args.rounds, args.iterations),
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
                cases = {
                    label: SchedulerCase(pair[0], count, True, unfinished_ids=unfinished_ids)
                    for label, pair in modules.items()
                }
                report["cases"].append(
                    {
                        "phase": "build_connector_meta_sparse",
                        "scheduled_cached_requests": count,
                        "unfinished_requests": unfinished_count,
                        "unfinished_placement": placement,
                        **measure(cases, args.warmup, args.rounds, args.iterations),
                    }
                )
    for count in (128, 512, 2048, 8192):
        cases = {label: LeaseCase(pair[1], count) for label, pair in modules.items()}
        report["cases"].append(
            {
                "phase": "prepare_layerwise_load_gvas",
                "blocks": count,
                **measure(cases, args.warmup, args.rounds, args.iterations),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
