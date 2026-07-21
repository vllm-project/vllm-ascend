"""Standalone correctness + performance harness for the CopyAndExpandDflashInputs
custom operator (Ascend 310P).

Runnable directly on an NPU host:

    python tests/e2e/nightly/single_node/ops/singlecard_ops/bench_copy_and_expand_dflash_inputs.py

1. Correctness (standalone reproduction of the golden accuracy test):
   validates the AscendC kernel against a CPU NumPy golden reference (bit-exact)
   for both DFlash (sample_from_anchor=False) and DSpark (sample_from_anchor=True)
   across all benchmarked batch sizes.

2. Performance of npu_copy_and_expand_dflash_inputs across two variants, over a
   range of batch sizes (num_reqs). Timing is measured with the torch_npu
   profiler: each case is profiled to its own directory, then the kernel's
   "Avg Time(us)" (column 7) is read back from
   ASCEND_PROFILER_OUTPUT/op_statistic.csv and aggregated into a table + CSV.
   Variants (columns):
   - dflash : sample_from_anchor=False, num_query_per_req = 1 + K
   - dspark : sample_from_anchor=True,  num_query_per_req = K
"""

import csv
import glob
import os

import numpy as np
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEVICE = "npu"
SEED = 42

# --------------------------------------------------------------------------- #
# Benchmark configuration
# --------------------------------------------------------------------------- #
# Primary sweep axis: batch size (number of requests). num_context scales as
# num_reqs * CTX_PER_REQ, so this also sweeps the dominant context-copy work.
SIZES = [1, 4, 8, 16, 32, 64, 128]

# Variants compared side by side (analogous to the reference script's "modes").
MODES = ["dflash", "dspark"]

# Fixed workload knobs (kept constant across the sweep for a clean 2D table).
NUM_SPECULATIVE_TOKENS = 4
BLOCK_SIZE = 128
CTX_PER_REQ = 256
MAX_REJECTED_PER_REQ = 4

PROF_BASE_DIR = "./prof_copy_expand_dflash"
PERF_SUMMARY_CSV = "./copy_expand_dflash_perf_summary.csv"

PROFILE_ITERS = 40


# --------------------------------------------------------------------------- #
# Golden reference (CPU, pure NumPy) -- mirrors the Triton kernel semantics.
# --------------------------------------------------------------------------- #
def golden_copy_and_expand_dflash(
    next_token_ids,
    target_positions,
    context_slot_mapping,
    query_start_loc,
    seq_lens,
    block_table,
    num_rejected_tokens,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    sample_from_anchor,
):
    num_reqs = len(next_token_ids)
    num_context = len(target_positions)
    num_query_total = num_reqs * num_query_per_req

    out_input_ids = np.zeros(num_query_total, dtype=np.int32)
    out_query_positions = np.zeros(num_query_total, dtype=np.int32)
    out_query_slot_mapping = np.zeros(num_query_total, dtype=np.int32)
    out_context_positions = np.zeros(num_context, dtype=np.int32)
    out_context_slot_mapping = np.zeros(num_context, dtype=np.int32)
    out_token_indices = np.zeros(num_reqs * num_speculative_tokens, dtype=np.int32)

    for req in range(num_reqs):
        ctx_start = int(query_start_loc[req])
        ctx_end = int(query_start_loc[req + 1])

        for j in range(ctx_start, ctx_end):
            out_context_positions[j] = target_positions[j]
            out_context_slot_mapping[j] = context_slot_mapping[j]

        num_rejected = int(num_rejected_tokens[req])
        if num_rejected < 0:
            num_rejected = 0
        valid_ctx_end = ctx_end - num_rejected

        seq_len = int(seq_lens[req])
        effective_seq_len = seq_len - num_rejected
        last_pos = int(target_positions[valid_ctx_end - 1])

        for q in range(num_query_per_req):
            query_pos = last_pos + 1 + q
            query_out_idx = req * num_query_per_req + q
            out_query_positions[query_out_idx] = query_pos

            query_cache_pos = effective_seq_len + q
            block_num = query_cache_pos // block_size
            block_id = int(block_table[req, block_num])
            slot = block_id * block_size + (query_cache_pos % block_size)
            out_query_slot_mapping[query_out_idx] = slot

            if q == 0:
                out_input_ids[query_out_idx] = int(next_token_ids[req])
            else:
                out_input_ids[query_out_idx] = parallel_drafting_token_id

            if sample_from_anchor:
                out_token_indices[req * num_speculative_tokens + q] = query_out_idx
            elif q > 0:
                out_token_indices[req * num_speculative_tokens + (q - 1)] = query_out_idx

    return (
        out_input_ids,
        out_query_positions,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_token_indices,
    )


# --------------------------------------------------------------------------- #
# Test case generator (fixed context length for reproducible timing).
# --------------------------------------------------------------------------- #
def generate_test_case(
    rng,
    num_reqs,
    num_speculative_tokens,
    sample_from_anchor,
    block_size=BLOCK_SIZE,
    ctx_per_req=CTX_PER_REQ,
    max_rejected_per_req=MAX_REJECTED_PER_REQ,
):
    parallel_drafting_token_id = 100
    num_query_per_req = num_speculative_tokens if sample_from_anchor else (1 + num_speculative_tokens)

    # Fixed per-request context length so the profiled workload is deterministic.
    ctx_counts = np.full(num_reqs, ctx_per_req, dtype=np.int32)
    rejected_per_req = np.array(
        [rng.integers(0, min(max_rejected_per_req, int(ctx_counts[i]) - 1) + 1) for i in range(num_reqs)],
        dtype=np.int32,
    )

    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    for i in range(num_reqs):
        query_start_loc[i + 1] = query_start_loc[i] + ctx_counts[i]
    num_context = int(query_start_loc[num_reqs])

    target_positions = np.zeros(num_context, dtype=np.int32)
    seq_lens = np.zeros(num_reqs, dtype=np.int32)
    for i in range(num_reqs):
        base = int(rng.integers(0, 32))
        qs = int(query_start_loc[i])
        n = int(ctx_counts[i])
        for j in range(n):
            target_positions[qs + j] = base + j
        seq_lens[i] = base + n

    context_slot_mapping = rng.integers(0, 1_000_000, size=num_context, dtype=np.int32)
    next_token_ids = rng.integers(1, 50000, size=num_reqs, dtype=np.int32)

    max_cache_pos = 0
    for i in range(num_reqs):
        eff = int(seq_lens[i]) - int(rejected_per_req[i])
        max_cache_pos = max(max_cache_pos, eff + num_query_per_req - 1)
    max_blocks = max_cache_pos // block_size + 2
    block_table = rng.integers(0, 10000, size=(num_reqs, max_blocks), dtype=np.int32)

    return {
        "next_token_ids": next_token_ids,
        "target_positions": target_positions,
        "context_slot_mapping": context_slot_mapping,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "block_table": block_table,
        "num_rejected_tokens": rejected_per_req,
        "parallel_drafting_token_id": parallel_drafting_token_id,
        "block_size": block_size,
        "num_query_per_req": num_query_per_req,
        "num_speculative_tokens": num_speculative_tokens,
        "sample_from_anchor": sample_from_anchor,
    }


_GOLDEN_KEYS = (
    "next_token_ids", "target_positions", "context_slot_mapping", "query_start_loc",
    "seq_lens", "block_table", "num_rejected_tokens", "parallel_drafting_token_id",
    "block_size", "num_query_per_req", "num_speculative_tokens", "sample_from_anchor",
)


def _to_npu_case(case):
    """Move the tensor inputs to NPU once so timing excludes H2D transfers."""
    npu_case = dict(case)
    for k in ("next_token_ids", "target_positions", "context_slot_mapping",
              "query_start_loc", "seq_lens", "block_table", "num_rejected_tokens"):
        npu_case[k] = torch.from_numpy(case[k]).to(torch.int32).npu()
    return npu_case


def _run_op(npu_case):
    return torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs(
        npu_case["next_token_ids"],
        npu_case["target_positions"],
        npu_case["context_slot_mapping"],
        npu_case["query_start_loc"],
        npu_case["seq_lens"],
        npu_case["block_table"],
        npu_case["num_rejected_tokens"],
        npu_case["parallel_drafting_token_id"],
        npu_case["block_size"],
        npu_case["num_query_per_req"],
        npu_case["num_speculative_tokens"],
        npu_case["sample_from_anchor"],
    )


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def test_correctness():
    names = [
        "out_input_ids", "out_query_positions", "out_query_slot_mapping",
        "out_context_positions", "out_context_slot_mapping", "out_token_indices",
    ]
    for mode in MODES:
        sample_from_anchor = mode == "dspark"
        for n in SIZES:
            rng = np.random.default_rng(SEED + n)
            case = generate_test_case(rng, n, NUM_SPECULATIVE_TOKENS, sample_from_anchor)
            golden = golden_copy_and_expand_dflash(**{k: case[k] for k in _GOLDEN_KEYS})
            npu_out = tuple(t.cpu() for t in _run_op(_to_npu_case(case)))
            for name, got, ref in zip(names, npu_out, golden):
                torch.testing.assert_close(
                    got, torch.from_numpy(ref), atol=0, rtol=0,
                    msg=f"{mode} num_reqs={n}: {name} mismatch",
                )
    print(f"PASS  test_correctness ({', '.join(MODES)} x {len(SIZES)} sizes)")


# --------------------------------------------------------------------------- #
# Performance (torch_npu profiler based)
# --------------------------------------------------------------------------- #
def _case_dir(mode, n):
    return os.path.join(PROF_BASE_DIR, f"cxd_{mode}_n{n}")


def _profile_case(fn, prof_dir, iters=PROFILE_ITERS):
    """Profile fn under the torch_npu profiler, dumping to prof_dir.

    Schedule: skip_first=1, wait=1, warmup=1, active=20.
    """
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=1, warmup=1, active=20, repeat=1, skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
    ):
        for _ in range(iters):
            fn()
            torch_npu.npu.synchronize()


def _avg_time_us_from_op_statistic(case_dir):
    """Avg Time(us) (column 7) of the kernel row in op_statistic.csv, from the
    newest profiling run under this case directory."""
    pattern = os.path.join(case_dir, "**", "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    with open(newest, newline="") as fp:
        body = list(csv.reader(fp))[1:]
    body = [r for r in body if len(r) > 6]
    if not body:
        return None
    # Prefer the CopyAndExpandDflashInputs kernel row; otherwise the dominant row.
    def _is_target(name):
        low = name.lower()
        return "dflash" in low or "copyandexpand" in low or "copy_and_expand" in low

    row = next((r for r in body if _is_target(r[0])), body[0])
    return row[6]


def _profile_mode(mode):
    sample_from_anchor = mode == "dspark"
    for n in SIZES:
        rng = np.random.default_rng(SEED + n)
        case = generate_test_case(rng, n, NUM_SPECULATIVE_TOKENS, sample_from_anchor)
        npu_case = _to_npu_case(case)

        def _fn(_c=npu_case):
            return _run_op(_c)

        # A couple of untimed warmups before the profiler schedule kicks in.
        for _ in range(3):
            _fn()
        torch_npu.npu.synchronize()

        _profile_case(_fn, _case_dir(mode, n))


def bench_all():
    print("\n== profiling npu_copy_and_expand_dflash_inputs (torch_npu profiler) ==")
    print(
        f"   config: num_speculative_tokens={NUM_SPECULATIVE_TOKENS}, "
        f"block_size={BLOCK_SIZE}, ctx_per_req={CTX_PER_REQ}"
    )
    for mode in MODES:
        print(f"  profiling variant: {mode}")
        _profile_mode(mode)

    table = {n: {} for n in SIZES}
    for mode in MODES:
        for n in SIZES:
            table[n][mode] = _avg_time_us_from_op_statistic(_case_dir(mode, n))

    header = f"{'num_reqs':>10} " + " ".join(f"{m:>13}" for m in MODES) + "   (Avg Time us)"
    print("\n== npu_copy_and_expand_dflash_inputs performance (Avg Time us) ==")
    print(header)
    print("-" * len(header))
    for n in SIZES:
        cells = " ".join(f"{(table[n][m] or 'NA'):>13}" for m in MODES)
        print(f"{n:>10} {cells}")

    with open(PERF_SUMMARY_CSV, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["num_reqs"] + MODES)
        for n in SIZES:
            writer.writerow([n] + [table[n][m] if table[n][m] is not None else "NA" for m in MODES])
    print(f"\nperf summary written to {PERF_SUMMARY_CSV}")


if __name__ == "__main__":
    test_correctness()
    bench_all()
