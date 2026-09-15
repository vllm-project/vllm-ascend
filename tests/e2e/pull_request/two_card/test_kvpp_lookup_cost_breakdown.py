# SPDX-License-Identifier: Apache-2.0
"""Lookup transport proof and serving cost attribution for the payload RFC."""

import json
import statistics
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.utils.network_utils import get_open_port
from vllm.v1.serial_utils import MsgpackEncoder

from tests.e2e.pull_request.two_card import test_kvpp_lookup_payload_e2e as serving
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupHashMode

RPC_WARMUP_PAIRS = 80
RPC_SAMPLE_PAIRS = 1000

pytestmark = pytest.mark.e2e_model(serving.MODEL)


class LookupStub:
    def __init__(self, expected_calls: int):
        self.expected_calls = expected_calls
        self.calls = 0
        self.hash_counts: dict[LookupHashMode, list[int]] = defaultdict(list)
        self.server: Any = None

    def lookup_scheduler(
        self,
        token_len,
        block_hashes,
        kv_cache_group_ids,
        use_layerwise,
        hbm_hit_tokens,
        lookup_hash_mode,
    ):
        self.calls += 1
        self.hash_counts[lookup_hash_mode].append(len(block_hashes))
        if self.calls == self.expected_calls:
            assert self.server is not None
            self.server.running = False
        return token_len


def lookup_config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"lookup_rpc_port": get_open_port()},
        ),
    )


def encoded_hash_bytes(block_hashes: list[bytes]) -> int:
    frames = MsgpackEncoder().encode([block_hash.hex() for block_hash in block_hashes])
    return sum(len(frame) for frame in frames)


def test_lookup_rpc_payload_latency():
    """Measure the exact Full and Suffix payloads used by the E2E scenario."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import LookupKeyServer
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import LookupKeyClient

    hbm_levels = (serving.HBM_BLOCKS,)
    full_hashes = [index.to_bytes(32, "big") for index in range(serving.TOTAL_BLOCKS)]
    measured_pairs = RPC_SAMPLE_PAIRS
    expected_calls = 2 * (RPC_WARMUP_PAIRS + measured_pairs)
    stub = LookupStub(expected_calls)
    config = lookup_config()
    server = LookupKeyServer(stub, config)
    stub.server = server
    client = LookupKeyClient(config)
    samples: dict[int, dict[LookupHashMode, list[float]]] = {
        hbm: {LookupHashMode.FULL: [], LookupHashMode.SUFFIX: []} for hbm in hbm_levels
    }

    def call(mode: LookupHashMode, hbm_blocks: int) -> float:
        hashes = full_hashes if mode is LookupHashMode.FULL else full_hashes[hbm_blocks:]
        started = time.perf_counter()
        assert (
            client.lookup(
                serving.INPUT_TOKENS,
                hashes,
                hbm_hit_tokens=hbm_blocks * serving.BLOCK_SIZE,
                lookup_hash_mode=mode,
            )
            == serving.INPUT_TOKENS
        )
        return time.perf_counter() - started

    try:
        for pair_index in range(RPC_WARMUP_PAIRS):
            hbm_blocks = hbm_levels[pair_index % len(hbm_levels)]
            order = (LookupHashMode.FULL, LookupHashMode.SUFFIX)
            if pair_index % 2:
                order = tuple(reversed(order))
            for mode in order:
                call(mode, hbm_blocks)

        for pair_index in range(measured_pairs):
            hbm_blocks = hbm_levels[pair_index % len(hbm_levels)]
            order = (LookupHashMode.FULL, LookupHashMode.SUFFIX)
            if pair_index % 2:
                order = tuple(reversed(order))
            for mode in order:
                samples[hbm_blocks][mode].append(call(mode, hbm_blocks))
    finally:
        client.close()
        server.thread.join(timeout=5)
        server.close()

    levels = []
    for hbm_blocks in hbm_levels:
        full = samples[hbm_blocks][LookupHashMode.FULL]
        suffix = samples[hbm_blocks][LookupHashMode.SUFFIX]
        full_bytes = encoded_hash_bytes(full_hashes)
        suffix_bytes = encoded_hash_bytes(full_hashes[hbm_blocks:])
        level = {
            "hbm_blocks": hbm_blocks,
            "hbm_hit_percent": hbm_blocks / serving.TOTAL_BLOCKS * 100,
            "full_hashes": serving.TOTAL_BLOCKS,
            "suffix_hashes": serving.TOTAL_BLOCKS - hbm_blocks,
            "full_encoded_hash_bytes": full_bytes,
            "suffix_encoded_hash_bytes": suffix_bytes,
            "payload_reduction_percent": (1 - suffix_bytes / full_bytes) * 100,
            "full_latency": serving.value_distribution([value * 1e6 for value in full]),
            "suffix_latency": serving.value_distribution([value * 1e6 for value in suffix]),
            "mean_latency_reduction_percent": (1 - statistics.fmean(suffix) / statistics.fmean(full)) * 100,
        }
        levels.append(level)
        assert len(full) == RPC_SAMPLE_PAIRS
        assert len(suffix) == RPC_SAMPLE_PAIRS
        assert suffix_bytes < full_bytes
        assert statistics.fmean(suffix) < statistics.fmean(full)

    report = {
        "config": {
            "warmup_pairs": RPC_WARMUP_PAIRS,
            "sample_pairs": RPC_SAMPLE_PAIRS,
            "total_blocks": serving.TOTAL_BLOCKS,
            "hbm_blocks": serving.HBM_BLOCKS,
        },
        "levels": levels,
    }
    print("\nKVPP_LOOKUP_RPC_PAYLOAD_LATENCY=" + json.dumps(report, sort_keys=True))


def lookup_cost_breakdown(runs: list[dict[str, Any]]) -> dict[str, Any]:
    duration_metric = "vllm:ascend_store_lookup_duration_seconds_total"
    pairs = []
    errors = []
    for round_index in range(len(serving.SERVING_MODE_ORDERS)):
        by_mode = {run["mode"]: run for run in runs if run["round"] == round_index}
        full_run = by_mode[LookupHashMode.FULL.value]
        suffix_run = by_mode[LookupHashMode.SUFFIX.value]
        for full, suffix in zip(full_run["samples"], suffix_run["samples"], strict=True):
            if full["manifest_digest"] != suffix["manifest_digest"]:
                errors.append(f"round={round_index}, scenario={full['scenario_id']}: manifest mismatch")
                continue
            full_lookup = full[duration_metric]
            suffix_lookup = suffix[duration_metric]
            pairs.append(
                {
                    "round": round_index,
                    "scenario_id": full["scenario_id"],
                    "hbm_hit_percent": full["hbm_hit_percent"],
                    "external_hit_percent_of_suffix": full["external_hit_percent_of_suffix"],
                    "full_batch_wall_seconds": full["batch_wall_seconds"],
                    "suffix_batch_wall_seconds": suffix["batch_wall_seconds"],
                    "full_lookup_seconds": full_lookup,
                    "suffix_lookup_seconds": suffix_lookup,
                    "full_lookup_share_percent": full_lookup / full["batch_wall_seconds"] * 100,
                    "suffix_lookup_share_percent": suffix_lookup / suffix["batch_wall_seconds"] * 100,
                    "lookup_time_reduction_percent": (1 - suffix_lookup / full_lookup) * 100,
                    "lookup_savings_share_of_full_e2e_percent": (
                        (full_lookup - suffix_lookup) / full["batch_wall_seconds"] * 100
                    ),
                    "observed_batch_wall_reduction_percent": (
                        1 - suffix["batch_wall_seconds"] / full["batch_wall_seconds"]
                    )
                    * 100,
                }
            )

    full_wall = sum(pair["full_batch_wall_seconds"] for pair in pairs)
    suffix_wall = sum(pair["suffix_batch_wall_seconds"] for pair in pairs)
    full_lookup = sum(pair["full_lookup_seconds"] for pair in pairs)
    suffix_lookup = sum(pair["suffix_lookup_seconds"] for pair in pairs)
    return {
        "pairs": len(pairs),
        "expected_pairs": len(serving.SERVING_MODE_ORDERS) * serving.SAMPLES_PER_INSTANCE,
        "validation_errors": errors,
        "aggregate": {
            "full_lookup_share_percent": full_lookup / full_wall * 100,
            "suffix_lookup_share_percent": suffix_lookup / suffix_wall * 100,
            "lookup_time_reduction_percent": (1 - suffix_lookup / full_lookup) * 100,
            "lookup_savings_share_of_full_e2e_percent": (full_lookup - suffix_lookup) / full_wall * 100,
            "observed_batch_wall_reduction_percent": (1 - suffix_wall / full_wall) * 100,
            "full_lookup_seconds": full_lookup,
            "suffix_lookup_seconds": suffix_lookup,
            "full_batch_wall_seconds": full_wall,
            "suffix_batch_wall_seconds": suffix_wall,
        },
        "distributions": {
            field: serving.value_distribution([pair[field] for pair in pairs])
            for field in (
                "full_lookup_share_percent",
                "suffix_lookup_share_percent",
                "lookup_time_reduction_percent",
                "lookup_savings_share_of_full_e2e_percent",
                "observed_batch_wall_reduction_percent",
            )
            if pairs
        },
        "raw_pairs": pairs,
    }


@serving.wait_until_npu_memory_free()
def test_vllm_serve_lookup_cost_breakdown(tmp_path):
    """Attribute batch wall time using the opt-in in-path lookup timer."""
    pytest.importorskip("memcache_hybrid")
    report = serving.run_benchmark(tmp_path, profile_lookup=True)
    breakdown = lookup_cost_breakdown(report["runs"])
    result = {
        "config": report["config"],
        "serving_validation": report["paired"],
        "cost_breakdown": breakdown,
        "runs": report["runs"],
    }
    result["valid"] = (
        report["valid"] and not breakdown["validation_errors"] and breakdown["pairs"] == breakdown["expected_pairs"]
    )
    print("\nKVPP_LOOKUP_COST_BREAKDOWN=" + json.dumps(result, sort_keys=True))
    assert result["valid"], json.dumps(
        {
            "serving_errors": report["paired"]["validation_errors"],
            "breakdown_errors": breakdown["validation_errors"],
            "breakdown_pairs": breakdown["pairs"],
            "expected_pairs": breakdown["expected_pairs"],
        },
        sort_keys=True,
    )
