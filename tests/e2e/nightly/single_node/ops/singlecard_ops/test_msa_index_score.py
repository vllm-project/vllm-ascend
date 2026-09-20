# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

BLOCK_SIZE = 128
HEAD_DIM = 128
NUM_HEADS = 8
SCORE_ALIGNMENT = 16
MASK_SIZE = 2048
FILL_THRESHOLD = -1.0e30
BOOST_THRESHOLD = 1.0e28


def _reference_scores(query, key, block_table, q_lens, kv_lens, sparse_mode, init_blocks, local_blocks):
    """CPU FP32 QK, right-aligned causal masking, then max over each KV block."""
    width = math.ceil(block_table.shape[1] / SCORE_ALIGNMENT) * SCORE_ALIGNMENT
    scores = torch.full((NUM_HEADS, sum(q_lens), width), -torch.inf)
    q_begin = 0
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        block_count = math.ceil(kv_len / BLOCK_SIZE)
        if q_len and block_count:
            pages = key[block_table[batch, :block_count].long(), :, 0, :]
            logits = torch.einsum("qhd,bkd->hqbk", query[q_begin : q_begin + q_len], pages)
            positions = torch.arange(block_count * BLOCK_SIZE).view(block_count, BLOCK_SIZE)
            visible = torch.full((q_len,), kv_len)
            if sparse_mode == 3:
                visible = (kv_len - q_len + torch.arange(q_len) + 1).clamp(0, kv_len)
            logits.masked_fill_(positions[None, None] >= visible[None, :, None, None], -torch.inf)
            block_scores = logits.amax(dim=-1)
            block_scores[..., :init_blocks] = 1.0e30
            if local_blocks:
                block_scores[..., max(0, block_count - local_blocks) :] = 1.0e29
            scores[:, q_begin : q_begin + q_len, :block_count] = block_scores
        q_begin += q_len
    return scores


def _assert_scores_close(actual, expected, dtype):
    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape
    actual = actual.cpu()
    # Kernels may represent masked/forced scores with finite sentinels or inf.
    fill = expected <= FILL_THRESHOLD
    boost = expected >= BOOST_THRESHOLD
    torch.testing.assert_close(actual <= FILL_THRESHOLD, fill, rtol=0, atol=0)
    torch.testing.assert_close(actual >= BOOST_THRESHOLD, boost, rtol=0, atol=0)
    valid = ~(fill | boost)
    tolerance = 2.0e-2 if dtype == torch.float8_e4m3fn else 1.0e-3
    torch.testing.assert_close(actual[valid], expected[valid], rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "table_width", "page_axis_gap", "sparse_mode", "init_blocks", "local_blocks"),
    [
        pytest.param((32, 17), (300, 130), 8, 1, 3, 0, 0, id="prefill"),
        pytest.param((32, 17), (300, 130), 8, 2, 3, 0, 0, id="strided-page-axis"),
        pytest.param((1, 2), (900, 257), 8, 1, 3, 0, 0, id="decode"),
        pytest.param((4,), (32769,), 275, 1, 3, 0, 0, id="long-kv-wide-table"),
        pytest.param((2,), (5,), 257, 1, 3, 0, 0, id="wide-table-padding"),
        pytest.param((0, 1), (129, 0), 2, 1, 3, 0, 0, id="empty-query-and-kv"),
        pytest.param((2,), (513,), 8, 1, 3, 1, 2, id="forced-blocks"),
        pytest.param((2,), (129,), 8, 1, 0, 0, 0, id="dense-tp-chunk"),
    ],
)
@torch.inference_mode()
def test_msa_index_score_precision(
    dtype, q_lens, kv_lens, table_width, page_axis_gap, sparse_mode, init_blocks, local_blocks
):
    """Compare the custom operator with an independent reference, without model weights."""
    if dtype == torch.float8_e4m3fn and not get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
        pytest.skip("MsaIndexScore FP8 requires FP8 attention support")

    generator = torch.Generator().manual_seed(2026)
    num_pages = max(math.ceil(max(kv_lens) / BLOCK_SIZE) + 3, 4)
    # Binary fractions are exactly representable in all tested input dtypes.
    query_cpu = torch.randint(-8, 9, (sum(q_lens), NUM_HEADS, HEAD_DIM), generator=generator).float() / 8
    key_cpu = torch.randint(-8, 9, (num_pages, BLOCK_SIZE, 1, HEAD_DIM), generator=generator).float() / 8
    block_table = torch.zeros((len(q_lens), table_width), dtype=torch.int32)
    for batch, kv_len in enumerate(kv_lens):
        blocks = math.ceil(kv_len / BLOCK_SIZE)
        block_table[batch, :blocks] = torch.randperm(num_pages, generator=generator)[:blocks].int()

    query = query_cpu.npu().to(dtype)
    storage = torch.full((num_pages * page_axis_gap, BLOCK_SIZE, 1, HEAD_DIM), 7.0, device="npu")
    storage[::page_axis_gap].copy_(key_cpu)
    key = storage.to(dtype)[::page_axis_gap]
    assert key.stride(0) == page_axis_gap * BLOCK_SIZE * HEAD_DIM

    cu_seqlens = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32).npu()
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="npu")
    start_loc = torch.tensor(
        [max(0, math.ceil(length / BLOCK_SIZE) - 1) for length in kv_lens], dtype=torch.int32, device="npu"
    )
    mask = torch.zeros((MASK_SIZE, MASK_SIZE), dtype=torch.int8, device="npu") if sparse_mode == 3 else None
    expected = _reference_scores(
        query_cpu, key_cpu, block_table, q_lens, kv_lens, sparse_mode, init_blocks, local_blocks
    )
    actual = torch.ops._C_ascend.npu_msa_index_score(
        query,
        key,
        block_table.npu(),
        start_loc,
        atten_mask=mask,
        actual_seq_qlen=cu_seqlens,
        actual_seq_klen=seq_lens,
        layout_key="BBND",
        sparse_mode=sparse_mode,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    _assert_scores_close(actual, expected, dtype)
