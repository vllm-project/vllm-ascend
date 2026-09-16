import gc

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.dsa_local_metadata import (
    DSA_LOCAL_METADATA_BLOCK,
    build_local_metadata,
    build_local_metadata_kernel,
)

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
    valid_local_req = (local_query_lens > 0) & (seq_lens > 0)
    local_seq_lens[:num_reqs] = torch.where(
        valid_local_req,
        torch.clamp_min(seq_lens - offset, 0),
        torch.zeros_like(seq_lens),
    )

    start_pos = None
    if compute_start_pos:
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        start_pos = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=query_start_loc.device)
        start_pos[:num_reqs] = seq_lens[:num_reqs] - seq_lens_q

    return local_query_start_loc, local_seq_lens, start_pos


def _run_fused(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    local_start: int,
    local_end: int,
    num_reqs: int,
    compute_start_pos: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    local_query_start_loc = torch.zeros(MAX_NUM_SEQS + 1, dtype=torch.int32, device=query_start_loc.device)
    local_seq_lens = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=query_start_loc.device)
    start_pos_out = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=query_start_loc.device)

    build_local_metadata(
        query_start_loc,
        seq_lens,
        local_query_start_loc,
        local_seq_lens,
        local_start,
        local_end,
        num_reqs,
        start_pos_out=start_pos_out if compute_start_pos else None,
        block=MAX_NUM_SEQS,
    )

    return (
        local_query_start_loc,
        local_seq_lens,
        start_pos_out if compute_start_pos else None,
    )


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
        trt_qsl, trt_sl, trt_sp = _run_fused(
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


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_build_local_metadata_triton_masks_graph_padding(device: str) -> None:
    torch.set_default_device(device)

    # TP8, graph size 80 and MTP3 produce 20 four-token request slots. With
    # nine real requests, rank 6 splits a padded slot at its local boundary.
    tp_size = 8
    tp_rank = 6
    num_input_tokens = 80
    num_reqs = 20
    num_actual_reqs = 9
    tokens_per_rank = num_input_tokens // tp_size
    local_start = tp_rank * tokens_per_rank
    local_end = local_start + tokens_per_rank

    query_start_loc = torch.arange(0, num_input_tokens + 1, 4, dtype=torch.int32, device=device)
    seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    seq_lens[:num_actual_reqs] = torch.arange(
        128,
        128 + num_actual_reqs,
        dtype=torch.int32,
        device=device,
    )

    trt_qsl, trt_sl, _ = _run_fused(
        query_start_loc,
        seq_lens,
        local_start,
        local_end,
        num_reqs,
        compute_start_pos=False,
    )
    ref_qsl, ref_sl, _ = _run_native(
        query_start_loc,
        seq_lens,
        local_start,
        local_end,
        num_reqs,
        compute_start_pos=False,
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
    assert ref_sl[17].item() == 0
    assert torch.all(trt_sl[:num_reqs] >= 0)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_build_local_metadata_no_recompile_across_num_reqs(device: str) -> None:
    """The whole point of the fixed-capacity kernel: sweeping num_reqs
    across power-of-2 boundaries (the old next_power_of_2 blocking
    re-JITed at each crossing) must not add kernel cache entries."""
    torch.set_default_device(device)

    def entry_count() -> int:
        return sum(len(v) for v in build_local_metadata_kernel.cache.values())

    cache_before = entry_count()
    for num_reqs in [1, 2, 3, 5, 8, 17, 32, 33, 64, 100, 128, 129, 256, 512, 1024]:
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        for i in range(num_reqs):
            query_start_loc[i + 1] = query_start_loc[i] + 4
        seq_lens = torch.full((num_reqs,), 64, dtype=torch.int32, device=device)
        local_query_start_loc = torch.zeros(MAX_NUM_SEQS + 1, dtype=torch.int32, device=device)
        local_seq_lens = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=device)
        start_pos_out = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=device)
        build_local_metadata(
            query_start_loc,
            seq_lens,
            local_query_start_loc,
            local_seq_lens,
            0,
            64,
            num_reqs,
            start_pos_out=start_pos_out,
            block=MAX_NUM_SEQS,
        )
    assert entry_count() - cache_before <= 1, (
        f"num_reqs sweep must not grow the kernel cache (before={cache_before}, after={entry_count()})"
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_build_local_metadata_empty_batch(device: str) -> None:
    """num_reqs == 0 short-circuits on the host: outputs must be all
    zeros (the [0] prefix plus an empty cumsum) for every output tensor,
    including start_pos_out, without touching the input pointers."""
    torch.set_default_device(device)

    query_start_loc = torch.zeros(1, dtype=torch.int32, device=device)
    seq_lens = torch.zeros(0, dtype=torch.int32, device=device)
    local_query_start_loc = torch.full((MAX_NUM_SEQS + 1,), 7, dtype=torch.int32, device=device)
    local_seq_lens = torch.full((MAX_NUM_SEQS,), 7, dtype=torch.int32, device=device)
    start_pos_out = torch.full((MAX_NUM_SEQS,), 7, dtype=torch.int32, device=device)

    build_local_metadata(
        query_start_loc,
        seq_lens,
        local_query_start_loc,
        local_seq_lens,
        0,
        32,
        0,
        start_pos_out=start_pos_out,
        block=MAX_NUM_SEQS,
    )

    assert torch.all(local_query_start_loc == 0)
    assert torch.all(local_seq_lens == 0)
    assert torch.all(start_pos_out == 0)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_build_local_metadata_tail_overwrites_stale_data(device: str) -> None:
    """The kernel overwrites the full capacity region, so a smaller batch
    following a larger one must not observe stale values beyond num_reqs
    (this is what lets the caller skip its fill_(0) pre-zeroing)."""
    torch.set_default_device(device)

    # Step 1: large batch writes non-zero metadata into the buffers.
    num_reqs_large = 64
    query_start_loc = torch.zeros(num_reqs_large + 1, dtype=torch.int32, device=device)
    for i in range(num_reqs_large):
        query_start_loc[i + 1] = query_start_loc[i] + 8
    seq_lens = torch.full((num_reqs_large,), 256, dtype=torch.int32, device=device)
    local_query_start_loc = torch.zeros(MAX_NUM_SEQS + 1, dtype=torch.int32, device=device)
    local_seq_lens = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=device)
    start_pos_out = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=device)
    build_local_metadata(
        query_start_loc,
        seq_lens,
        local_query_start_loc,
        local_seq_lens,
        0,
        512,
        num_reqs_large,
        start_pos_out=start_pos_out,
        block=MAX_NUM_SEQS,
    )
    assert local_query_start_loc[num_reqs_large + 1 :].abs().max().item() == 0
    assert local_seq_lens[num_reqs_large:].abs().max().item() == 0

    # Step 2: small batch into the SAME (dirty) buffers, no pre-zeroing.
    num_reqs_small = 5
    query_start_loc = torch.zeros(num_reqs_small + 1, dtype=torch.int32, device=device)
    for i in range(num_reqs_small):
        query_start_loc[i + 1] = query_start_loc[i] + 4
    seq_lens = torch.full((num_reqs_small,), 64, dtype=torch.int32, device=device)
    build_local_metadata(
        query_start_loc,
        seq_lens,
        local_query_start_loc,
        local_seq_lens,
        0,
        32,
        num_reqs_small,
        start_pos_out=start_pos_out,
        block=MAX_NUM_SEQS,
    )
    assert local_query_start_loc[num_reqs_small + 1 :].abs().max().item() == 0
    assert local_seq_lens[num_reqs_small:].abs().max().item() == 0
    assert start_pos_out[num_reqs_small:].abs().max().item() == 0

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_build_local_metadata_default_block_matches_capacity(device: str) -> None:
    """The default block is the production scheduler capacity: the builder
    allocates local_query_start_loc with max_num_seqs + 1 elements, so the
    kernel's default BLOCK must divide it evenly (SUB_N=8 fold)."""
    assert DSA_LOCAL_METADATA_BLOCK == 512
    assert (DSA_LOCAL_METADATA_BLOCK + 1) == 513
    assert DSA_LOCAL_METADATA_BLOCK % 8 == 0
