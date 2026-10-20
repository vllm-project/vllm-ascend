# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare AscendC KDA verify against RecoverSSM verify plus commit.

Run on an Ascend NPU with five speculative tokens (six verified positions).
Memory figures describe recurrent tensors, not an allocated serving KV pool.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.kda import run_recurrent_kda
from vllm_ascend.ops.triton.kda.recoverssm import (
    RECOVERSSM_VALUE_TILE,
    kda_recoverssm_commit,
    kda_recoverssm_verify,
)


def graph_time_us(fn, inner=20, samples=7):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / inner)
    return {"median_us": statistics.median(times), "min_us": min(times), "samples_us": times}


@torch.inference_mode()
def case(batch, state_dtype, accept_count):
    torch.manual_seed(20261017 + batch)
    device, heads, dim, length = "npu", 8, 128, 6
    q, k, v, raw_g = [torch.randn(1, batch * length, heads, dim, device=device, dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(1, batch * length, heads, device=device, dtype=torch.bfloat16)
    a_log = torch.randn(heads, device=device) * 0.2
    bias = torch.randn(heads * dim, device=device) * 0.1
    starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * length
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    native_indices = torch.arange(1, batch * length + 1, device=device, dtype=torch.int32).reshape(batch, length)
    accepted = torch.full((batch,), accept_count, device=device, dtype=torch.int32)
    native_previous = torch.ones(batch, device=device, dtype=torch.int32)
    initial = (torch.randn(batch + 1, heads, dim, dim, device=device) * 0.05).to(state_dtype)
    checkpoint = initial.clone()
    native_state = torch.zeros(batch * length + 1, heads, dim, dim, device=device, dtype=state_dtype)
    native_state[native_indices[:, 0].long()] = initial[1:]
    correction = torch.empty(batch + 1, heads, length, dim, device=device)
    kg = torch.empty(batch + 1, heads, length, dim * 2, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(v)

    def verify():
        return kda_recoverssm_verify(
            q, k, v, raw_g, beta, a_log, bias, -5.0, checkpoint, correction, kg, starts, indices, length, out=out
        )

    def commit():
        kda_recoverssm_commit(checkpoint, correction, kg, a_log, bias, -5.0, starts, indices, accepted)

    def recover_round():
        verify()
        commit()

    def native():
        return run_recurrent_kda(
            q,
            k,
            v,
            raw_g,
            beta,
            native_state,
            starts,
            native_indices,
            a_log,
            bias,
            lower_bound=-5.0,
            beta_is_preprocessed=False,
            num_accepted_tokens=native_previous,
        )

    # Compare two rounds before timing, including the next round's checkpoint.
    errors = []
    for step in range(2):
        original = checkpoint.clone()
        expected = native()
        actual = verify()
        torch.testing.assert_close(checkpoint, original, atol=0, rtol=0)
        torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
        commit()
        selected = native_indices[:, accept_count - 1].long()
        native_committed = native_state[selected]
        torch.testing.assert_close(checkpoint[1:].float(), native_committed.float(), atol=2e-2, rtol=2e-2)
        errors.append(
            {
                "round": step + 1,
                "output_max_abs": (actual.float() - expected.float()).abs().max().item(),
                "state_max_abs": (checkpoint[1:].float() - native_committed.float()).abs().max().item(),
            }
        )
        native_previous.fill_(accept_count)

    timings = {
        "native_ascendc_verify": graph_time_us(native),
        "recover_verify": graph_time_us(verify),
        "recover_commit": graph_time_us(commit),
        "recover_verify_plus_commit": graph_time_us(recover_round),
    }
    state_bytes = torch.empty((), dtype=state_dtype).element_size()
    native_bytes = batch * length * heads * dim * dim * state_bytes
    recover_bytes = batch * heads * (dim * dim * state_bytes + length * dim * 4 + length * 2 * dim * 2)
    return {
        "batch": batch,
        "local_heads": heads,
        "head_dim": dim,
        "num_speculative_tokens": 5,
        "verified_positions": length,
        "accepted_positions": accept_count,
        "state_dtype": str(state_dtype),
        "activation_dtype": "torch.bfloat16",
        "errors_vs_native_ascendc": errors,
        "persistent_recurrent_cache_native_bytes": native_bytes,
        "persistent_recurrent_cache_recover_bytes": recover_bytes,
        "recurrent_cache_reduction": 1 - recover_bytes / native_bytes,
        "timings": timings,
        "recover_total_over_native": timings["recover_verify_plus_commit"]["median_us"]
        / timings["native_ascendc_verify"]["median_us"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark_kda_recoverssm.json"))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--accepted", nargs="+", type=int, default=[1, 3, 6])
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batches):
        parser.error("batch sizes must be positive")
    if any(count < 1 or count > 6 for count in args.accepted):
        parser.error("accepted positions must be in [1, 6]")
    results = {
        "hardware": torch.npu.get_device_name(),
        "torch": torch.__version__,
        "value_tile": RECOVERSSM_VALUE_TILE,
        "torch_npu": torch_npu.__version__,
        "triton": triton.__version__,
        "method": "Median of seven NPU graph replays, each containing twenty calls.",
        "scope": "Recurrence only; excludes projections, convolution, sampling, metadata and serving.",
        "native_commit": "AscendC stores each position; the next verify selects its accepted checkpoint.",
        "memory_scope": "Logical tensor bytes; excludes padding, convolution and allocator alignment.",
        "cases": [],
    }
    started = time.monotonic()
    for state_dtype in (torch.float32, torch.bfloat16):
        for batch in args.batches:
            for accept_count in args.accepted:
                record = case(batch, state_dtype, accept_count)
                results["cases"].append(record)
                results["elapsed_seconds"] = time.monotonic() - started
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
