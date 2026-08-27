"""On-board accuracy test for CopyAndExpandDflashInputs custom operator (310P).

Validates the AscendC kernel against a CPU golden reference that mirrors the
Triton kernel ``copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid``
exactly. Covers DFlash (sample_from_anchor=False, num_query_per_req = 1 + K)
and DSpark (sample_from_anchor=True, num_query_per_req = K), with and without
rejected tokens, across multiple batch sizes.

Run on a 310P card:
    pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_copy_and_expand_dflash_inputs.py
"""

import numpy as np
import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure NumPy) -- mirrors the Triton kernel semantics.
# ---------------------------------------------------------------------------


def golden_copy_and_expand_dflash(
    next_token_ids: np.ndarray,
    target_positions: np.ndarray,
    context_slot_mapping: np.ndarray,
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    block_table: np.ndarray,
    num_rejected_tokens: np.ndarray,
    parallel_drafting_token_id: int,
    block_size: int,
    num_query_per_req: int,
    num_speculative_tokens: int,
    sample_from_anchor: bool,
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

        # 1. Context identity copy.
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

        # 2. Query block.
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

            # 3. Sample indices.
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


# ---------------------------------------------------------------------------
# NPU operator wrapper
# ---------------------------------------------------------------------------


def npu_op_exec(case):
    result = torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs(
        torch.from_numpy(case["next_token_ids"]).to(torch.int32).npu(),
        torch.from_numpy(case["target_positions"]).to(torch.int32).npu(),
        torch.from_numpy(case["context_slot_mapping"]).to(torch.int32).npu(),
        torch.from_numpy(case["query_start_loc"]).to(torch.int32).npu(),
        torch.from_numpy(case["seq_lens"]).to(torch.int32).npu(),
        torch.from_numpy(case["block_table"]).to(torch.int32).npu(),
        torch.from_numpy(case["num_rejected_tokens"]).to(torch.int32).npu(),
        case["parallel_drafting_token_id"],
        case["block_size"],
        case["num_query_per_req"],
        case["num_speculative_tokens"],
        case["sample_from_anchor"],
    )
    return tuple(t.cpu() for t in result)


# ---------------------------------------------------------------------------
# Test case generator
# ---------------------------------------------------------------------------


def generate_test_case(
    rng,
    num_reqs,
    num_speculative_tokens,
    sample_from_anchor,
    block_size=128,
    min_ctx_per_req=2,
    max_ctx_per_req=64,
    max_rejected_per_req=4,
):
    parallel_drafting_token_id = 100
    num_query_per_req = num_speculative_tokens if sample_from_anchor else (1 + num_speculative_tokens)

    ctx_per_req = rng.integers(min_ctx_per_req, max_ctx_per_req + 1, size=num_reqs)
    rejected_per_req = np.array(
        [rng.integers(0, min(max_rejected_per_req, ctx_per_req[i] - 1) + 1) for i in range(num_reqs)],
        dtype=np.int32,
    )

    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    for i in range(num_reqs):
        query_start_loc[i + 1] = query_start_loc[i] + ctx_per_req[i]
    num_context = int(query_start_loc[num_reqs])

    # Per-request context positions start at a random base and run contiguously.
    target_positions = np.zeros(num_context, dtype=np.int32)
    seq_lens = np.zeros(num_reqs, dtype=np.int32)
    for i in range(num_reqs):
        base = int(rng.integers(0, 32))
        qs = int(query_start_loc[i])
        n = int(ctx_per_req[i])
        for j in range(n):
            target_positions[qs + j] = base + j
        # Absolute sequence length up to (and including) the last context token.
        seq_lens[i] = base + n

    context_slot_mapping = rng.integers(0, 1_000_000, size=num_context, dtype=np.int32)
    next_token_ids = rng.integers(1, 50000, size=num_reqs, dtype=np.int32)

    # block_table must cover the largest query cache position accessed.
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


def _assert_all_close(npu_out, golden_out):
    names = [
        "out_input_ids",
        "out_query_positions",
        "out_query_slot_mapping",
        "out_context_positions",
        "out_context_slot_mapping",
        "out_token_indices",
    ]
    for name, n, g in zip(names, npu_out, golden_out):
        torch.testing.assert_close(n, torch.from_numpy(g), atol=0, rtol=0, msg=f"{name} mismatch")


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_reqs", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3, 5])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("seed_offset", [0, 1])
def test_copy_and_expand_dflash_inputs(num_reqs, num_speculative_tokens, sample_from_anchor, seed_offset):
    rng = np.random.default_rng(SEED + seed_offset)
    case = generate_test_case(rng, num_reqs, num_speculative_tokens, sample_from_anchor)
    golden = golden_copy_and_expand_dflash(**{k: case[k] for k in (
        "next_token_ids", "target_positions", "context_slot_mapping", "query_start_loc",
        "seq_lens", "block_table", "num_rejected_tokens", "parallel_drafting_token_id",
        "block_size", "num_query_per_req", "num_speculative_tokens", "sample_from_anchor")})
    npu_out = npu_op_exec(case)
    _assert_all_close(npu_out, golden)


@pytest.mark.parametrize("num_reqs", [1, 4, 8])
def test_no_rejected_tokens(num_reqs):
    """DFlash decode steps with zero rejected tokens (num_rejected all 0)."""
    rng = np.random.default_rng(SEED + 400)
    case = generate_test_case(rng, num_reqs, num_speculative_tokens=3, sample_from_anchor=False, max_rejected_per_req=0)
    golden = golden_copy_and_expand_dflash(**{k: case[k] for k in (
        "next_token_ids", "target_positions", "context_slot_mapping", "query_start_loc",
        "seq_lens", "block_table", "num_rejected_tokens", "parallel_drafting_token_id",
        "block_size", "num_query_per_req", "num_speculative_tokens", "sample_from_anchor")})
    npu_out = npu_op_exec(case)
    _assert_all_close(npu_out, golden)


@pytest.mark.parametrize("num_reqs", [3, 7, 13])
def test_large_context(num_reqs):
    """Larger per-request context to exercise the bulk context copy path."""
    rng = np.random.default_rng(SEED + 200)
    case = generate_test_case(
        rng, num_reqs, num_speculative_tokens=4, sample_from_anchor=False,
        min_ctx_per_req=100, max_ctx_per_req=512, max_rejected_per_req=8,
    )
    golden = golden_copy_and_expand_dflash(**{k: case[k] for k in (
        "next_token_ids", "target_positions", "context_slot_mapping", "query_start_loc",
        "seq_lens", "block_table", "num_rejected_tokens", "parallel_drafting_token_id",
        "block_size", "num_query_per_req", "num_speculative_tokens", "sample_from_anchor")})
    npu_out = npu_op_exec(case)
    _assert_all_close(npu_out, golden)


@pytest.mark.parametrize("block_size", [64, 128])
def test_block_sizes(block_size):
    """Both supported 310P kernel block sizes."""
    rng = np.random.default_rng(SEED + 500)
    case = generate_test_case(
        rng, num_reqs=8, num_speculative_tokens=2, sample_from_anchor=False, block_size=block_size,
    )
    golden = golden_copy_and_expand_dflash(**{k: case[k] for k in (
        "next_token_ids", "target_positions", "context_slot_mapping", "query_start_loc",
        "seq_lens", "block_table", "num_rejected_tokens", "parallel_drafting_token_id",
        "block_size", "num_query_per_req", "num_speculative_tokens", "sample_from_anchor")})
    npu_out = npu_op_exec(case)
    _assert_all_close(npu_out, golden)
