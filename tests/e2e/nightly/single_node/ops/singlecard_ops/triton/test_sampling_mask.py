# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for sampling-mask packing on non-contiguous logits."""

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.sampling_mask import sampling_mask_from_logits_npu

_HAS_COMPACT_IDS = "token_ids" in SamplingMaskTensors._fields
# vLLM #59359 added cu_num_logits and request-major multi-row output slots.
_HAS_ROWS_PER_REQUEST = "rows_per_request" in SamplingMaskTensors._fields


def _from_logits(logits, num_sampled_tokens, max_num_kept=512):
    if _HAS_ROWS_PER_REQUEST:
        cu_num_logits = torch.arange(logits.shape[0] + 1, dtype=torch.int32, device=logits.device)
        return sampling_mask_from_logits_npu(
            SamplingMaskTensors,
            logits,
            cu_num_logits,
            num_sampled_tokens,
            max_num_kept,
        )
    if _HAS_COMPACT_IDS:
        return sampling_mask_from_logits_npu(
            SamplingMaskTensors,
            logits,
            num_sampled_tokens,
            max_num_kept=max_num_kept,
        )
    return sampling_mask_from_logits_npu(
        SamplingMaskTensors,
        logits,
        num_sampled_tokens,
    )


def _to_nested_list(tensors, num_sampled_tokens):
    cpu_tensors = tensors.to_cpu_nonblocking()
    torch.npu.synchronize()
    if _HAS_COMPACT_IDS:
        assert tensors.token_ids.shape == (len(num_sampled_tokens), 0)
        return cpu_tensors.tolists().to_nested_list()
    return cpu_tensors.tolists(num_sampled_tokens.cpu().numpy()).to_nested_list()


@pytest.mark.parametrize("max_num_kept", [512, 20_001])
def test_sampling_mask_matches_finite_support(max_num_kept):
    """Cover the support sizes and fallback boundaries used by upstream."""
    init_device_properties_triton()

    vocab_size = 20_001
    sizes = [0, 1, 7, 511, 512, 513, 2048, 2053, vocab_size, 40, 3]
    generator = torch.Generator().manual_seed(0)
    logits = torch.full((len(sizes), vocab_size), -float("inf"))
    expected = []
    for row, size in enumerate(sizes):
        kept = torch.randperm(vocab_size, generator=generator)[:size].sort().values
        logits[row, kept] = torch.randn(size, generator=generator)
        expected.append(kept.tolist())

    num_sampled = torch.ones(len(sizes), dtype=torch.int32)
    num_sampled[[1, 9]] = 0
    num_sampled[-1] = 2
    expected[1] = expected[9] = []
    sampled_flags = num_sampled.tolist()

    tensors = _from_logits(
        logits.to("npu"),
        num_sampled.to("npu"),
        max_num_kept=max_num_kept,
    )
    torch.testing.assert_close(
        tensors.counts,
        torch.tensor(
            [size if sampled else 0 for size, sampled in zip(sizes, sampled_flags)],
            dtype=torch.int32,
            device="npu",
        ),
        rtol=0,
        atol=0,
    )

    result = _to_nested_list(tensors, num_sampled)
    if not _HAS_COMPACT_IDS:
        expected = [row for row, sampled in zip(expected, sampled_flags) if sampled]
    assert result == expected


def test_sampling_mask_matches_processed_top_k_top_p_support():
    """Mirror the processed-logits integration case from vLLM v0.29.0."""
    init_device_properties_triton()
    init_ascend_config(VllmConfig())
    try:
        processed_logits = apply_top_k_top_p(
            logits=torch.tensor(
                [[6.0, 5.0, 4.0, 4.0, 4.0, 2.0, 1.0, 0.0]],
                device="npu",
            ),
            k=torch.tensor([3], device="npu"),
            p=torch.tensor([0.9], device="npu"),
        )
        expected_ids = torch.isfinite(processed_logits[0]).nonzero().flatten().tolist()
        assert 0 < len(expected_ids) < processed_logits.shape[1]

        num_sampled = torch.tensor([1], dtype=torch.int32, device="npu")
        tensors = _from_logits(processed_logits, num_sampled, max_num_kept=3)

        assert _to_nested_list(tensors, num_sampled) == [expected_ids]
    finally:
        clear_ascend_config()


def test_sampling_mask_non_contiguous_vocab_dimension():
    init_device_properties_triton()

    num_reqs = 4
    vocab_size = 128256
    base = torch.full(
        (num_reqs, vocab_size * 2),
        -float("inf"),
        dtype=torch.float32,
        device="npu",
    )
    logits = base[:, ::2]
    logits[:, :64] = torch.randn(
        (num_reqs, 64),
        dtype=logits.dtype,
        device=logits.device,
    )
    logits[:, 64] = float("nan")
    logits[:, 65] = float("inf")

    assert logits.shape == (num_reqs, vocab_size)
    assert logits.stride(1) == 2

    num_sampled = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device="npu")
    tensors = _from_logits(logits, num_sampled, max_num_kept=32)

    # The wrapper rebinds only its local argument; the caller's view is unchanged.
    assert logits.stride(1) == 2
    torch.npu.synchronize()
    torch.testing.assert_close(
        tensors.counts,
        torch.tensor([64, 0, 64, 0], dtype=torch.int32, device="npu"),
        rtol=0,
        atol=0,
    )
    expected_ids = list(range(64))
    result = _to_nested_list(tensors, num_sampled)
    if _HAS_COMPACT_IDS:
        expected = [expected_ids, [], expected_ids, []]
    else:
        expected = [expected_ids] * 2

    assert result == expected


@pytest.mark.skipif(not _HAS_ROWS_PER_REQUEST, reason="requires the request-major SamplingMaskTensors API")
def test_sampling_mask_uses_request_boundaries():
    init_device_properties_triton()

    logits = torch.full((6, 8), -float("inf"), device="npu")
    logits[0, [1, 3]] = 0
    logits[2, [2, 4, 6]] = 0
    logits[5, [0, 7]] = 0

    tensors = sampling_mask_from_logits_npu(
        SamplingMaskTensors,
        logits,
        torch.tensor([0, 2, 5, 6], dtype=torch.int32, device="npu"),
        torch.tensor([1, 0, 1], dtype=torch.int32, device="npu"),
        3,
    )
    torch.npu.synchronize()

    assert tensors.tolists().to_nested_list() == [[1, 3], [], [0, 7]]


@pytest.mark.skipif(not _HAS_ROWS_PER_REQUEST, reason="requires the request-major SamplingMaskTensors API")
def test_sampling_mask_multirow_request_layout():
    """Spec-decode layout: each request owns a fixed number of output slots."""
    init_device_properties_triton()

    rows_per_request = 4
    vocab_size = 64
    row_sizes = [3, 5, 1, 2, 9, 4, 6]
    generator = torch.Generator().manual_seed(1)
    logits = torch.full((len(row_sizes), vocab_size), -float("inf"))
    for row, size in enumerate(row_sizes):
        kept = torch.randperm(vocab_size, generator=generator)[:size]
        logits[row, kept] = torch.randn(size, generator=generator)

    cu_num_logits = torch.tensor([0, 3, 4, 7], dtype=torch.int32)
    num_sampled = torch.tensor([2, 1, 3], dtype=torch.int32)
    expected = [
        torch.isfinite(logits[int(cu_num_logits[req]) + slot]).nonzero().flatten().tolist()
        for req in range(len(num_sampled))
        for slot in range(int(num_sampled[req]))
    ]

    tensors = sampling_mask_from_logits_npu(
        SamplingMaskTensors,
        logits.to("npu"),
        cu_num_logits.to("npu"),
        num_sampled.to("npu"),
        3,
        rows_per_request,
    )
    torch.npu.synchronize()

    assert tensors.rows_per_request == rows_per_request
    assert tensors.counts.shape == (len(num_sampled) * rows_per_request,)
    assert tensors.tolists(num_sampled.numpy()).to_nested_list() == expected
