"""E2E accuracy test for CopyAndExpandEagleInputs custom operator.

Tests the Ascend C kernel against a CPU golden reference implementation
with parametrized test cases covering various configurations.
"""

import numpy as np
import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure Python/NumPy)
# ---------------------------------------------------------------------------

def golden_copy_and_expand(
    target_token_ids: np.ndarray,
    target_positions: np.ndarray,
    next_token_ids: np.ndarray,
    query_start_loc: np.ndarray,
    query_end_loc: np.ndarray,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    num_padding_slots: int,
    shift_input_ids: bool,
):
    """CPU golden reference for CopyAndExpandEagleInputs.

    Returns:
        (out_input_ids, out_positions, out_is_rejected_token_mask,
         out_is_masked_token_mask, out_new_token_indices,
         out_hidden_state_mapping)
    """
    num_reqs = len(next_token_ids)

    # Compute total_draft_tokens
    total_draft_tokens = 0
    for r in range(num_reqs):
        qs = query_start_loc[r]
        nqs = query_start_loc[r + 1]
        qe = query_end_loc[r]
        num_rejected = max(nqs - qe - 1, 0)
        if shift_input_ids:
            num_valid = max(qe - qs, 0)
        else:
            num_valid = max(qe - qs + 1, 0)
        total_draft_tokens += num_valid + num_padding_slots + num_rejected

    out_ids = np.zeros(total_draft_tokens, dtype=np.int32)
    out_pos = np.zeros(total_draft_tokens, dtype=np.int32)
    out_rej = np.zeros(total_draft_tokens, dtype=np.int8)
    out_msk = np.zeros(total_draft_tokens, dtype=np.int8)
    out_nti = np.zeros(num_reqs * num_padding_slots, dtype=np.int32)
    total_input_tokens = len(target_token_ids)
    out_hsm = np.zeros(total_input_tokens, dtype=np.int32)

    for r in range(num_reqs):
        qs = query_start_loc[r]
        nqs = query_start_loc[r + 1]
        qe = query_end_loc[r]

        num_rejected = max(nqs - qe - 1, 0)

        if shift_input_ids:
            num_valid = max(qe - qs, 0)
            output_start = qs + r * (num_padding_slots - 1)
        else:
            num_valid = max(qe - qs + 1, 0)
            output_start = qs + r * num_padding_slots

        start_pos = target_positions[qs]
        next_token_id = next_token_ids[r]

        # Valid region
        if shift_input_ids:
            read_start = qs + 1
            read_count = min(num_valid, total_input_tokens - read_start)
            if read_count < 0:
                read_count = 0
            for j in range(num_valid):
                idx = min(j, read_count - 1) if read_count > 0 else 0
                out_ids[output_start + j] = target_token_ids[read_start + idx] if read_count > 0 else 0
                out_pos[output_start + j] = start_pos + j
                out_rej[output_start + j] = 0
                out_msk[output_start + j] = 0
        else:
            num_input = nqs - qs
            for j in range(num_valid):
                idx = min(j, num_input - 1)
                out_ids[output_start + j] = target_token_ids[qs + idx]
                out_pos[output_start + j] = start_pos + j
                out_rej[output_start + j] = 0
                out_msk[output_start + j] = 0

        # Bonus token
        out_ids[output_start + num_valid] = next_token_id
        out_pos[output_start + num_valid] = start_pos + num_valid
        out_rej[output_start + num_valid] = 0
        out_msk[output_start + num_valid] = 0

        # Parallel draft tokens
        for k in range(1, num_padding_slots):
            j = num_valid + k
            out_ids[output_start + j] = parallel_drafting_token_id
            out_pos[output_start + j] = start_pos + j
            out_rej[output_start + j] = 0
            out_msk[output_start + j] = 1

        # Rejected tokens
        for k in range(num_rejected):
            j = num_valid + num_padding_slots + k
            out_ids[output_start + j] = padding_token_id
            out_pos[output_start + j] = 0
            out_rej[output_start + j] = 1
            out_msk[output_start + j] = 0

        # New token indices
        for k in range(num_padding_slots):
            out_nti[r * num_padding_slots + k] = output_start + num_valid + k

        # Hidden state mapping (shift_input_ids=true only)
        if shift_input_ids:
            num_input = nqs - qs
            for j in range(num_input):
                out_hsm[qs + j] = output_start + j

    return out_ids, out_pos, out_rej, out_msk, out_nti, out_hsm


# ---------------------------------------------------------------------------
# NPU operator wrapper
# ---------------------------------------------------------------------------

def npu_op_exec(
    target_token_ids, target_positions, next_token_ids,
    query_start_loc, query_end_loc,
    padding_token_id, parallel_drafting_token_id,
    num_padding_slots, shift_input_ids, total_draft_tokens,
):
    """Execute the custom Ascend NPU operator."""
    result = torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(
        target_token_ids.to(torch.int32).npu(),
        target_positions.to(torch.int32).npu(),
        next_token_ids.to(torch.int32).npu(),
        query_start_loc.to(torch.int32).npu(),
        query_end_loc.to(torch.int32).npu(),
        padding_token_id,
        parallel_drafting_token_id,
        num_padding_slots,
        shift_input_ids,
        total_draft_tokens,
    )
    return tuple(t.cpu() for t in result)


# ---------------------------------------------------------------------------
# Test case generator
# ---------------------------------------------------------------------------

def generate_test_case(rng, num_reqs, num_padding_slots, shift_input_ids,
                       min_tokens_per_req=2, max_tokens_per_req=64,
                       max_rejected_per_req=5):
    """Generate a random test case.

    Returns dict with all input arrays and expected parameters.
    """
    padding_token_id = 0
    parallel_drafting_token_id = 100

    # Generate per-request token counts
    tokens_per_req = rng.integers(min_tokens_per_req, max_tokens_per_req + 1,
                                  size=num_reqs)
    rejected_per_req = rng.integers(0, max_rejected_per_req + 1, size=num_reqs)

    # Build query_start_loc (cumulative)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    for i in range(num_reqs):
        query_start_loc[i + 1] = query_start_loc[i] + tokens_per_req[i] + rejected_per_req[i]

    total_input_tokens = int(query_start_loc[num_reqs])

    # Build query_end_loc: queryEnd = queryStart + numAccepted - 1
    # where numAccepted = tokens_per_req[i]
    # For shift=false: numValid = queryEnd - queryStart + 1 = tokens_per_req[i]
    # For shift=true: numValid = queryEnd - queryStart = tokens_per_req[i] - 1
    query_end_loc = np.zeros(num_reqs, dtype=np.int32)
    for i in range(num_reqs):
        if shift_input_ids:
            query_end_loc[i] = query_start_loc[i] + tokens_per_req[i]
        else:
            query_end_loc[i] = query_start_loc[i] + tokens_per_req[i] - 1

    # Generate input tokens and positions
    target_token_ids = rng.integers(1, 50000, size=total_input_tokens, dtype=np.int32)
    target_positions = np.zeros(total_input_tokens, dtype=np.int32)
    for i in range(num_reqs):
        qs = query_start_loc[i]
        nqs = query_start_loc[i + 1]
        for j in range(nqs - qs):
            target_positions[qs + j] = j

    next_token_ids = rng.integers(1, 50000, size=num_reqs, dtype=np.int32)

    # Compute total_draft_tokens
    total_draft_tokens = 0
    for r in range(num_reqs):
        qs = query_start_loc[r]
        nqs = query_start_loc[r + 1]
        qe = query_end_loc[r]
        num_rejected = max(nqs - qe - 1, 0)
        if shift_input_ids:
            num_valid = max(qe - qs, 0)
        else:
            num_valid = max(qe - qs + 1, 0)
        total_draft_tokens += num_valid + num_padding_slots + num_rejected

    return {
        "target_token_ids": target_token_ids,
        "target_positions": target_positions,
        "next_token_ids": next_token_ids,
        "query_start_loc": query_start_loc,
        "query_end_loc": query_end_loc,
        "padding_token_id": padding_token_id,
        "parallel_drafting_token_id": parallel_drafting_token_id,
        "num_padding_slots": num_padding_slots,
        "shift_input_ids": shift_input_ids,
        "total_draft_tokens": total_draft_tokens,
    }


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_reqs", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("num_padding_slots", [1, 2, 3, 5])
@pytest.mark.parametrize("shift_input_ids", [False, True])
@pytest.mark.parametrize("seed_offset", [0, 1])
def test_copy_and_expand_eagle_inputs(num_reqs, num_padding_slots,
                                       shift_input_ids, seed_offset):
    """Test CopyAndExpandEagleInputs with parametrized configurations."""
    rng = np.random.default_rng(SEED + seed_offset)

    case = generate_test_case(rng, num_reqs, num_padding_slots,
                              shift_input_ids)

    # Golden reference
    g_ids, g_pos, g_rej, g_msk, g_nti, g_hsm = golden_copy_and_expand(
        case["target_token_ids"],
        case["target_positions"],
        case["next_token_ids"],
        case["query_start_loc"],
        case["query_end_loc"],
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
    )

    # NPU execution
    n_ids, n_pos, n_rej, n_msk, n_nti, n_hsm = npu_op_exec(
        torch.from_numpy(case["target_token_ids"]),
        torch.from_numpy(case["target_positions"]),
        torch.from_numpy(case["next_token_ids"]),
        torch.from_numpy(case["query_start_loc"]),
        torch.from_numpy(case["query_end_loc"]),
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
        case["total_draft_tokens"],
    )

    # Convert golden to tensors
    g_ids_t = torch.from_numpy(g_ids)
    g_pos_t = torch.from_numpy(g_pos)
    g_rej_t = torch.from_numpy(g_rej)
    g_msk_t = torch.from_numpy(g_msk)
    g_nti_t = torch.from_numpy(g_nti)
    g_hsm_t = torch.from_numpy(g_hsm)

    # Compare outputs
    torch.testing.assert_close(n_ids, g_ids_t, atol=0, rtol=0,
                               msg="out_input_ids mismatch")
    torch.testing.assert_close(n_pos, g_pos_t, atol=0, rtol=0,
                               msg="out_positions mismatch")
    torch.testing.assert_close(n_rej, g_rej_t, atol=0, rtol=0,
                               msg="out_is_rejected_token_mask mismatch")
    torch.testing.assert_close(n_msk, g_msk_t, atol=0, rtol=0,
                               msg="out_is_masked_token_mask mismatch")
    torch.testing.assert_close(n_nti, g_nti_t, atol=0, rtol=0,
                               msg="out_new_token_indices mismatch")

    if shift_input_ids:
        torch.testing.assert_close(n_hsm, g_hsm_t, atol=0, rtol=0,
                                   msg="out_hidden_state_mapping mismatch")


@pytest.mark.parametrize("num_reqs", [1])
@pytest.mark.parametrize("num_padding_slots", [1])
@pytest.mark.parametrize("shift_input_ids", [False, True])
def test_minimal_case(num_reqs, num_padding_slots, shift_input_ids):
    """Test with minimal input (1 request, 1 padding slot)."""
    rng = np.random.default_rng(SEED + 100)
    case = generate_test_case(rng, num_reqs, num_padding_slots,
                              shift_input_ids, min_tokens_per_req=2,
                              max_tokens_per_req=3, max_rejected_per_req=1)

    g_ids, g_pos, g_rej, g_msk, g_nti, g_hsm = golden_copy_and_expand(
        case["target_token_ids"],
        case["target_positions"],
        case["next_token_ids"],
        case["query_start_loc"],
        case["query_end_loc"],
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
    )

    n_ids, n_pos, n_rej, n_msk, n_nti, n_hsm = npu_op_exec(
        torch.from_numpy(case["target_token_ids"]),
        torch.from_numpy(case["target_positions"]),
        torch.from_numpy(case["next_token_ids"]),
        torch.from_numpy(case["query_start_loc"]),
        torch.from_numpy(case["query_end_loc"]),
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
        case["total_draft_tokens"],
    )

    torch.testing.assert_close(n_ids, torch.from_numpy(g_ids), atol=0, rtol=0)
    torch.testing.assert_close(n_pos, torch.from_numpy(g_pos), atol=0, rtol=0)
    torch.testing.assert_close(n_rej, torch.from_numpy(g_rej), atol=0, rtol=0)
    torch.testing.assert_close(n_msk, torch.from_numpy(g_msk), atol=0, rtol=0)
    torch.testing.assert_close(n_nti, torch.from_numpy(g_nti), atol=0, rtol=0)


@pytest.mark.parametrize("num_reqs", [3, 7, 13])
def test_large_tokens_per_request(num_reqs):
    """Test with larger token counts per request."""
    rng = np.random.default_rng(SEED + 200)
    case = generate_test_case(rng, num_reqs, num_padding_slots=3,
                              shift_input_ids=False,
                              min_tokens_per_req=100,
                              max_tokens_per_req=512,
                              max_rejected_per_req=10)

    g_ids, g_pos, g_rej, g_msk, g_nti, g_hsm = golden_copy_and_expand(
        case["target_token_ids"],
        case["target_positions"],
        case["next_token_ids"],
        case["query_start_loc"],
        case["query_end_loc"],
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
    )

    n_ids, n_pos, n_rej, n_msk, n_nti, n_hsm = npu_op_exec(
        torch.from_numpy(case["target_token_ids"]),
        torch.from_numpy(case["target_positions"]),
        torch.from_numpy(case["next_token_ids"]),
        torch.from_numpy(case["query_start_loc"]),
        torch.from_numpy(case["query_end_loc"]),
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
        case["total_draft_tokens"],
    )

    torch.testing.assert_close(n_ids, torch.from_numpy(g_ids), atol=0, rtol=0)
    torch.testing.assert_close(n_pos, torch.from_numpy(g_pos), atol=0, rtol=0)
    torch.testing.assert_close(n_rej, torch.from_numpy(g_rej), atol=0, rtol=0)
    torch.testing.assert_close(n_msk, torch.from_numpy(g_msk), atol=0, rtol=0)
    torch.testing.assert_close(n_nti, torch.from_numpy(g_nti), atol=0, rtol=0)


@pytest.mark.parametrize("num_reqs", [3, 7, 13])
def test_large_tokens_shift_true(num_reqs):
    """Test with larger token counts and shift_input_ids=True."""
    rng = np.random.default_rng(SEED + 300)
    case = generate_test_case(rng, num_reqs, num_padding_slots=4,
                              shift_input_ids=True,
                              min_tokens_per_req=50,
                              max_tokens_per_req=256,
                              max_rejected_per_req=8)

    g_ids, g_pos, g_rej, g_msk, g_nti, g_hsm = golden_copy_and_expand(
        case["target_token_ids"],
        case["target_positions"],
        case["next_token_ids"],
        case["query_start_loc"],
        case["query_end_loc"],
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
    )

    n_ids, n_pos, n_rej, n_msk, n_nti, n_hsm = npu_op_exec(
        torch.from_numpy(case["target_token_ids"]),
        torch.from_numpy(case["target_positions"]),
        torch.from_numpy(case["next_token_ids"]),
        torch.from_numpy(case["query_start_loc"]),
        torch.from_numpy(case["query_end_loc"]),
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
        case["total_draft_tokens"],
    )

    torch.testing.assert_close(n_ids, torch.from_numpy(g_ids), atol=0, rtol=0)
    torch.testing.assert_close(n_pos, torch.from_numpy(g_pos), atol=0, rtol=0)
    torch.testing.assert_close(n_rej, torch.from_numpy(g_rej), atol=0, rtol=0)
    torch.testing.assert_close(n_msk, torch.from_numpy(g_msk), atol=0, rtol=0)
    torch.testing.assert_close(n_nti, torch.from_numpy(g_nti), atol=0, rtol=0)
    torch.testing.assert_close(n_hsm, torch.from_numpy(g_hsm), atol=0, rtol=0)


@pytest.mark.parametrize("num_reqs", [1, 4, 8])
def test_no_rejected_tokens(num_reqs):
    """Test cases with zero rejected tokens."""
    rng = np.random.default_rng(SEED + 400)
    case = generate_test_case(rng, num_reqs, num_padding_slots=2,
                              shift_input_ids=False,
                              min_tokens_per_req=5,
                              max_tokens_per_req=20,
                              max_rejected_per_req=0)

    g_ids, g_pos, g_rej, g_msk, g_nti, g_hsm = golden_copy_and_expand(
        case["target_token_ids"],
        case["target_positions"],
        case["next_token_ids"],
        case["query_start_loc"],
        case["query_end_loc"],
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
    )

    n_ids, n_pos, n_rej, n_msk, n_nti, n_hsm = npu_op_exec(
        torch.from_numpy(case["target_token_ids"]),
        torch.from_numpy(case["target_positions"]),
        torch.from_numpy(case["next_token_ids"]),
        torch.from_numpy(case["query_start_loc"]),
        torch.from_numpy(case["query_end_loc"]),
        case["padding_token_id"],
        case["parallel_drafting_token_id"],
        case["num_padding_slots"],
        case["shift_input_ids"],
        case["total_draft_tokens"],
    )

    torch.testing.assert_close(n_ids, torch.from_numpy(g_ids), atol=0, rtol=0)
    torch.testing.assert_close(n_pos, torch.from_numpy(g_pos), atol=0, rtol=0)
    torch.testing.assert_close(n_rej, torch.from_numpy(g_rej), atol=0, rtol=0)
    torch.testing.assert_close(n_msk, torch.from_numpy(g_msk), atol=0, rtol=0)
    torch.testing.assert_close(n_nti, torch.from_numpy(g_nti), atol=0, rtol=0)
