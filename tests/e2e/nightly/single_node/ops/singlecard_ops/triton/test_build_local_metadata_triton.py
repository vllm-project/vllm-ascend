import gc

import pytest
import torch
from vllm.triton_utils import HAS_TRITON, triton

from vllm_ascend.ops.triton.dsa_cp import build_local_metadata_triton

MAX_NUM_SEQS = 1024
NUM_REQS_LIST = [1, 7, 32, 1024]
TP_SIZES = [1, 8]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 0
DEFAULT_RTOL = 0


def _run_native(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    local_start: int,
    local_end: int,
    num_reqs: int,
    compute_start_pos: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    local_query_start = torch.clamp(query_start_loc[:-1], min=local_start, max=local_end)
    local_query_end = torch.clamp(query_start_loc[1:], min=local_start, max=local_end)
    local_query_lens = local_query_end - local_query_start

    local_query_start_loc = torch.zeros(MAX_NUM_SEQS + 1, dtype=torch.int32, device=query_start_loc.device)
    local_query_start_loc[1 : num_reqs + 1] = torch.cumsum(local_query_lens, dim=0)

    offset = query_start_loc[1:] - local_query_end
    local_seq_lens = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=query_start_loc.device)
    local_seq_lens[:num_reqs] = (local_query_lens > 0) * (seq_lens - offset)

    start_pos = None
    if compute_start_pos:
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        start_pos = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=query_start_loc.device)
        start_pos[:num_reqs] = seq_lens[:num_reqs] - seq_lens_q

    return local_query_start_loc, local_seq_lens, start_pos


def _run_triton(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    local_start: int,
    local_end: int,
    num_reqs: int,
    compute_start_pos: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    local_query_start_loc = torch.zeros(MAX_NUM_SEQS + 1, dtype=torch.int32, device=query_start_loc.device)
    local_seq_lens = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=query_start_loc.device)
    zero_i32 = torch.tensor([0], device=query_start_loc.device, dtype=torch.int32)
    start_pos_out = (
        torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=query_start_loc.device) if compute_start_pos else zero_i32
    )

    build_local_metadata_triton[(1,)](
        query_start_loc,
        seq_lens,
        local_query_start_loc,
        local_seq_lens,
        local_start,
        local_end,
        num_reqs,
        start_pos_out,
        BLOCK_NUM_REQS=triton.next_power_of_2(num_reqs),
        COMPUTE_START_POS=compute_start_pos,
    )

    return local_query_start_loc, local_seq_lens, start_pos_out if compute_start_pos else None


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@pytest.mark.parametrize("num_reqs", NUM_REQS_LIST)
@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("tp_rank", range(max(TP_SIZES)))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_build_local_metadata_triton(
    num_reqs: int,
    tp_size: int,
    tp_rank: int,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)

    # only test first rank and last rank
    if tp_rank == 0 or tp_rank == tp_size - 1:
        return

    # Build random seq_lens and compute local_start/local_end as the builder does.
    num_input_tokens = max(num_reqs * 4, 32)
    max_tpr = num_input_tokens // max(num_reqs, 1) * 2
    seq_lens_list = torch.randint(1, max_tpr + 1, (num_reqs,)).tolist()
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    for i in range(num_reqs):
        query_start_loc[i + 1] = query_start_loc[i] + seq_lens_list[i]
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    local_start = tp_rank * tokens_per_rank
    local_end = local_start + tokens_per_rank

    for compute_start_pos in [True, False]:
        trt_qsl, trt_sl, trt_sp = _run_triton(
            query_start_loc,
            seq_lens,
            local_start,
            local_end,
            num_reqs,
            compute_start_pos=compute_start_pos,
        )
        ref_qsl, ref_sl, ref_sp = _run_native(
            query_start_loc,
            seq_lens,
            local_start,
            local_end,
            num_reqs,
            compute_start_pos=compute_start_pos,
        )

        torch.testing.assert_close(
            trt_qsl[: num_reqs + 1],
            ref_qsl[: num_reqs + 1],
            atol=DEFAULT_ATOL,
            rtol=DEFAULT_RTOL,
        )
        torch.testing.assert_close(
            trt_sl[:num_reqs],
            ref_sl[:num_reqs],
            atol=DEFAULT_ATOL,
            rtol=DEFAULT_RTOL,
        )
        if compute_start_pos:
            assert trt_sp is not None and ref_sp is not None
            torch.testing.assert_close(
                trt_sp[:num_reqs],
                ref_sp[:num_reqs],
                atol=DEFAULT_ATOL,
                rtol=DEFAULT_RTOL,
            )
        else:
            assert trt_sp is None and ref_sp is None

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
