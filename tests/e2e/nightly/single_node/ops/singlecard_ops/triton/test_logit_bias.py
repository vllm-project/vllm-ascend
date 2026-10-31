# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[Role] Validate the Ascend MRv2 bias override through upstream callers.
[Invariants] Imported wrapper aliases stay valid; finite stop logits alone restore.
[Timeouts & Locks] Run under the bounded NPU test runner; no new kernel barriers.
[Side-Effect Policy] Allocates small test tensors on NPU0 and compares CPU results.
"""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias

import vllm_ascend.patch.worker  # noqa: F401
from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import _apply_grammar_bitmask_kernel
from vllm_ascend.ops.triton.v2.sample.logit_bias import _bias_kernel


@pytest.mark.parametrize("check_all", [True, False])
def test_stop_restore_through_existing_upstream_alias(check_all):
    torch.npu.set_device(0)
    assert apply_logit_bias.__globals__["_bias_kernel"] is _bias_kernel
    rows, vocab = 4, 129280
    logits = torch.full((rows, vocab), -float("inf"))
    logits[:, 1] = 2
    logits[:, 17] = -3
    logits[:, 33] = float("inf")
    logits[:, 65] = float("nan")
    logits[1, 200] = 0
    expected = logits.clone()
    expected[:3, [1, 17, 33, 65]] = -float("inf")
    if check_all:
        expected[0, 1] = 2
        expected[0, 17] = -3

    req = torch.arange(rows, dtype=torch.int64)
    pos = torch.tensor([0, 0, 0, 100], dtype=torch.int64)
    counts = torch.zeros(rows, dtype=torch.int32)
    ids = torch.zeros((rows, 1024), dtype=torch.int32)
    bias = torch.zeros((rows, 1024), dtype=torch.float32)
    min_len = torch.full((rows,), 100, dtype=torch.int32)
    stop_n = torch.full((rows,), 4, dtype=torch.int32)
    restore = torch.tensor([1, 1, 0, 1], dtype=torch.int32)
    stop_ids = torch.zeros((rows, 128), dtype=torch.int32)
    stop_ids[:, :4] = torch.tensor([1, 17, 33, 65])
    actual = logits.to("npu:0")
    args = [req, pos, counts, ids, counts, ids, bias, min_len, stop_n, restore, stop_ids]
    apply_logit_bias(actual, *[arg.to("npu:0") for arg in args], check_all_masked_rows=check_all)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)


def test_grammar_bias_stream_overlap():
    """Consecutive lock-bearing launches must not corrupt a reused workspace."""
    torch.npu.set_device(0)
    rows, mask_stride, vocab, block_size = 18, 6, 129280, 8192
    reqs = rows // mask_stride
    logits = torch.zeros((rows, vocab), dtype=torch.bfloat16, device="npu:0")
    req_idx = torch.arange(rows, dtype=torch.int32) // mask_stride
    position = torch.arange(rows, dtype=torch.int32) % mask_stride
    mapping_cpu = req_idx * mask_stride + position
    mapping_npu = mapping_cpu.to("npu:0")
    cu_num_logits = torch.arange(0, (reqs + 1) * mask_stride, mask_stride, dtype=torch.int32, device="npu:0")
    bitmask_cpu = torch.full((rows, (vocab + 31) // 32), -1, dtype=torch.int32, pin_memory=True)
    bitmask_cpu[:, 1::2] = 0
    bitmask_cpu[:, 2::4] = 0x55555555
    bitmask_npu = torch.empty_like(bitmask_cpu, device="npu:0")
    expanded_idx = req_idx.to(torch.int64).to("npu:0")
    pos = position.to(torch.int64).to("npu:0")
    counts = torch.zeros(reqs, dtype=torch.int32, device="npu:0")
    ids = torch.zeros((reqs, 1024), dtype=torch.int32, device="npu:0")
    biases = torch.zeros((reqs, 1024), dtype=torch.float32, device="npu:0")
    min_lens = torch.full((reqs,), vocab, dtype=torch.int32, device="npu:0")
    stop_n = torch.ones(reqs, dtype=torch.int32, device="npu:0")
    restore = torch.ones(reqs, dtype=torch.int32, device="npu:0")
    stop_ids = torch.zeros((reqs, 128), dtype=torch.int32, device="npu:0")
    stop_ids[:, 0] = 1
    copy_stream = torch.npu.Stream()
    main_stream = torch.npu.current_stream()

    for _ in range(20):
        with torch.npu.stream(copy_stream):
            copy_stream.wait_stream(main_stream)
            bitmask_npu.copy_(bitmask_cpu, non_blocking=True)
            mapping_npu.copy_(mapping_cpu, non_blocking=True)
        main_stream.wait_stream(copy_stream)
        _apply_grammar_bitmask_kernel[(rows, triton.cdiv(vocab, block_size))](
            logits,
            logits.stride(0),
            mapping_npu,
            cu_num_logits,
            bitmask_npu,
            bitmask_npu.stride(0),
            vocab,
            MASK_STRIDE=mask_stride,
            BLOCK_SIZE=block_size,
        )
        biased_logits = logits.float()
        apply_logit_bias(
            biased_logits,
            expanded_idx,
            pos,
            counts,
            ids,
            counts,
            ids,
            biases,
            min_lens,
            stop_n,
            restore,
            stop_ids,
            check_all_masked_rows=True,
        )
        copy_stream.wait_stream(main_stream)

    torch.npu.synchronize()
    token = torch.arange(vocab)
    blocked = ((bitmask_cpu[:, token // 32].to(torch.int64) >> (token % 32)) & 1) == 0
    expected = torch.zeros((rows, vocab), dtype=torch.bfloat16)
    expected.masked_fill_(blocked, -float("inf"))
    torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
    expected_bias = expected.float()
    expected_bias[:, 1] = -float("inf")
    torch.testing.assert_close(biased_logits.cpu(), expected_bias, rtol=0, atol=0)
