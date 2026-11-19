# SPDX-License-Identifier: Apache-2.0
"""CPU receive preparation: indexed address lookup versus known KV identities.

Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0. Network READ, landing copies and NPU
events are no-ops; this measures Python preparation, not transport bandwidth.
Both variants use the same current code and registry, with/without local target
descriptors. This isolates bypassing lookups from the earlier index improvement.
"""

import argparse
import importlib
import json
import os
import platform
import statistics
import sys
import threading
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from benchmark_dsa_pd_mooncake_registry import _measure_pair

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.transfer_targets import MooncakeKVRegistry

registry = MooncakeKVRegistry()

ALIGNMENT = 2 * 1024 * 1024
LAYERS = 78
BLOCKS = 33
DIRECT_TENSORS = 21
MAIN_BLOCK_BYTES = 128 * 1024
ROPE_BLOCK_BYTES = 16 * 1024


class _Event:
    def record(self, _stream):
        pass

    def synchronize(self):
        pass


class _Engine:
    def batch_transfer_sync_read(self, _session, _local, _remote, _lengths):
        return 0


def _make_reader(reader_type):
    reader = reader_type.__new__(reader_type)
    reader.engine = _Engine()
    reader.host_kv = SimpleNamespace(land=lambda *_args: None)
    reader.kv_registry = registry
    reader.kv_caches = {f"direct.{i}": torch.empty(1024 * 1024, dtype=torch.int8) for i in range(DIRECT_TENSORS)}
    for layer in range(LAYERS):
        pair = (
            torch.empty(BLOCKS * MAIN_BLOCK_BYTES, dtype=torch.int8),
            torch.empty(BLOCKS * ROPE_BLOCK_BYTES, dtype=torch.int8),
        )
        reader.kv_caches[f"host.{layer}"] = pair
        for tensor in pair:
            registry.register_offloaded_tensor(tensor)
    reader.block_len_per_addr = [[MAIN_BLOCK_BYTES, ROPE_BLOCK_BYTES]]
    reader._dsa_pd_mooncake_staging_lock = threading.Lock()
    reader._dsa_pd_mooncake_staging_tensor = torch.empty(16 * ALIGNMENT, dtype=torch.int8)
    reader._dsa_pd_mooncake_staging_bytes = 16 * ALIGNMENT
    reader._pipeline_landing_enabled = True
    reader._pipeline_banks = 2
    # Measure preparation on the caller, excluding executor scheduling costs.
    reader._get_dsa_pd_mooncake_bank_workers = lambda _banks: 0
    reader._landing_streams = [SimpleNamespace(synchronize=lambda: None) for _ in range(2)]
    reader._landing_stream = reader._landing_streams[0]
    reader._get_dsa_pd_mooncake_staging_window = lambda: 16
    return reader


def _counts(reader, operation):
    book = registry
    with (
        patch.object(book, "resolve_offloaded_tensor", wraps=book.resolve_offloaded_tensor) as tensors,
        patch.object(book, "get_register_location", wraps=book.get_register_location) as locations,
        patch.object(book, "resolve_offloaded_byte_tensor", wraps=book.resolve_offloaded_byte_tensor) as views,
        patch.object(reader.engine, "batch_transfer_sync_read", wraps=reader.engine.batch_transfer_sync_read) as reads,
        patch.object(reader.host_kv, "land", wraps=reader.host_kv.land) as landings,
    ):
        assert operation() == 0
        return dict(
            tensor_queries=tensors.call_count,
            cpu_location_queries=locations.call_count,
            byte_tensor_queries=views.call_count,
            read_batches=reads.call_count,
            copies=sum(len(call.args[0]) for call in landings.call_args_list),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--min-sample-ms", type=float, default=10)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3 or args.min_sample_ms <= 0:
        parser.error("samples must be >= 3 and min-sample-ms must be positive")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    fake_npu = SimpleNamespace(Event=_Event, stream=lambda _stream: nullcontext(), synchronize=lambda: None)
    engine_name = "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine"
    engine_module = ModuleType(engine_name)
    engine_module.global_te = SimpleNamespace(register_buffer=lambda *_args: None)
    with patch.object(torch, "npu", fake_npu, create=True), patch.dict(sys.modules, {engine_name: engine_module}):
        # The staging module's evaluated NPU annotations need this CPU stub.
        staging = importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.staging")
        # Keep log I/O out of both CPU preparation measurements.
        staging.logger.info = lambda *_args, **_kwargs: None
        registry.clear()
        reader = _make_reader(staging.MooncakeSFAStagingReader)
        layers = [list(reader._iter_kv_cache_tensors(value)) for value in reader.kv_caches.values()]
        bases = [[tensor.data_ptr() for tensor in layer] for layer in layers]
        tensors = [tensor for layer in layers for tensor in layer]
        local = [tensor.data_ptr() for tensor in tensors]
        remote = [(100 + i * 8) * ALIGNMENT for i in range(len(tensors))]
        lengths = [tensor.nbytes for tensor in tensors]
        construction = []
        for _ in range(3):
            start = time.perf_counter_ns()
            table = reader.get_transfer_targets(list(reader.kv_caches), bases)
            construction.append((time.perf_counter_ns() - start) / 1000)
            assert table is not None
        targets = [target for layer in table for target in layer]
        operations = {
            "baseline": partial(reader.batch_transfer_sync_read, "peer", local, remote, lengths, "indexed"),
            "optimized": partial(
                reader.batch_transfer_sync_read, "peer", local, remote, lengths, "tagged", transfer_targets=targets
            ),
        }
        counts = {name: _counts(reader, operation) for name, operation in operations.items()}
        assert counts["baseline"]["tensor_queries"] == 199 and counts["baseline"]["cpu_location_queries"] == 21
        assert counts["optimized"]["tensor_queries"] == counts["optimized"]["cpu_location_queries"] == 0
        assert counts["baseline"]["read_batches"] == counts["optimized"]["read_batches"] == 40
        assert counts["baseline"]["copies"] == counts["optimized"]["copies"] == 156
        result = {
            "environment": dict(
                host=platform.node(),
                machine=platform.machine(),
                python=sys.version,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                samples=args.samples,
                min_sample_ms=args.min_sample_ms,
                profiler=False,
            ),
            "scope": "Same indexed implementation, staging-reader entry to return, no-op network/copy/events.",
            "geometry": dict(
                host_tensors=156,
                direct_tensors=21,
                staging_bytes=16 * ALIGNMENT,
                banks=2,
                host_bytes=LAYERS * BLOCKS * (MAIN_BLOCK_BYTES + ROPE_BLOCK_BYTES),
            ),
            "counts_untimed": counts,
            "target_table_build_median_us": statistics.median(construction),
            "timings": _measure_pair(operations, args.samples, args.min_sample_ms * 1_000_000),
            "limitations": [
                "No model, native transfer engine, actual landing copy or NPU is exercised.",
                "Upstream address construction and per-request target-list appends are excluded.",
                "Target-table construction is startup work and excluded from warm receive timings.",
                "Counting instrumentation is used in separate calls, not in timed samples.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
