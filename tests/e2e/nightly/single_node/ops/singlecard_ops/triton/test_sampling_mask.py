# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for sampling-mask packing on non-contiguous logits."""

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.sampling_mask import (
    sampling_mask_from_logits_npu,
    sampling_mask_from_logits_npu_main,
)

_HAS_COMPACT_IDS = "token_ids" in SamplingMaskTensors._fields
# Main keys each output row to a request-major slot and resolves its source
# logits row through `cu_num_logits`; the release has one row per request.
_SLOT_LAYOUT = "rows_per_request" in SamplingMaskTensors._fields


def _from_logits(
    logits,
    num_sampled_tokens,
    max_num_kept=512,
    cu_num_logits=None,
    rows_per_request=1,
):
    if _SLOT_LAYOUT:
        if cu_num_logits is None:
            cu_num_logits = torch.arange(
                num_sampled_tokens.shape[0],
                dtype=torch.int32,
                device=logits.device,
            )
        return sampling_mask_from_logits_npu_main(
            SamplingMaskTensors,
            logits,
            cu_num_logits,
            num_sampled_tokens,
            max_num_kept,
            rows_per_request,
        )
    return sampling_mask_from_logits_npu(
        SamplingMaskTensors,
        logits,
        num_sampled_tokens,
        max_num_kept=max_num_kept,
    )


def _to_nested_list(tensors, num_sampled_tokens):
    cpu_tensors = tensors.to_cpu_nonblocking()
    torch.npu.synchronize()
    rows_per_request = getattr(tensors, "rows_per_request", 1)
    if _HAS_COMPACT_IDS:
        assert tensors.token_ids.shape == (
            len(num_sampled_tokens) * rows_per_request,
            0,
        )
    if not _HAS_COMPACT_IDS:
        # Three-field release API: `tolists` needs the sampled counts and only
        # emits rows that have one.
        return cpu_tensors.tolists(num_sampled_tokens.cpu().numpy()).to_nested_list()
    if rows_per_request != 1:
        return cpu_tensors.tolists(num_sampled_tokens.cpu().numpy()).to_nested_list()
    return cpu_tensors.tolists().to_nested_list()


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
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


@pytest.mark.parametrize(
    "cu_num_logits_per_req,num_sampled_list,num_logits_rows",
    [
        # Non-zero base, an unused gap at rows 4..8, and a request that commits
        # nothing, so source_row never equals the output row.
        ([1, 4, 9], [3, 0, 3], 12),
        # Partial commit: request 1 fills only two of its three slots, so the
        # trailing slot must be dropped rather than read.
        ([0, 3], [3, 2], 6),
    ],
    ids=["non-identity-mapping", "partial-commit-trailing-slot"],
)
def test_sampling_mask_slot_layout_maps_committed_rows(cu_num_logits_per_req, num_sampled_list, num_logits_rows):
    """One support per committed speculative slot (vLLM #59359).

    Each output row belongs to a request-major slot whose logits row is
    ``cu_num_logits[req] + slot``, and a slot past the request's sampled count is
    reported as empty rather than reading an uncommitted row.

    The first parameter set makes ``source_row`` differ from the output row
    everywhere, so an implementation that simply read the output row fails; a
    contiguous ``[0, rpr, 2 * rpr]`` layout would let such an implementation
    pass. The second keeps the partial-commit case, where only the trailing slot
    of a request is dead.
    """
    if not _SLOT_LAYOUT:
        pytest.skip(
            "Requires a vLLM build whose SamplingMaskTensors carries the slot API "
            "(v0.32.0 or newer); this build uses the per-request layout."
        )

    init_device_properties_triton()

    vocab_size = 1024
    rows_per_request = 3
    num_sampled = torch.tensor(num_sampled_list, dtype=torch.int32, device="npu")
    cu_num_logits = torch.tensor(cu_num_logits_per_req, dtype=torch.int32, device="npu")
    # None marks a slot past its request's committed count.
    source_rows = [
        cu_num_logits_per_req[req] + slot if slot < num_sampled_list[req] else None
        for req in range(len(num_sampled_list))
        for slot in range(rows_per_request)
    ]

    generator = torch.Generator().manual_seed(7)
    logits = torch.full((num_logits_rows, vocab_size), -float("inf"))
    kept_by_row = {}
    for row in range(num_logits_rows):
        kept = torch.randperm(vocab_size, generator=generator)[: 4 + row].sort().values
        logits[row, kept] = torch.randn(kept.numel(), generator=generator)
        kept_by_row[row] = kept.tolist()

    expected = [kept_by_row[source] if source is not None else [] for source in source_rows]

    tensors = _from_logits(
        logits.to("npu"),
        num_sampled,
        max_num_kept=64,
        cu_num_logits=cu_num_logits,
        rows_per_request=rows_per_request,
    )
    torch.npu.synchronize()

    assert tensors.rows_per_request == rows_per_request
    assert tensors.counts.numel() == num_sampled.numel() * rows_per_request
    torch.testing.assert_close(
        tensors.counts,
        torch.tensor([len(row) for row in expected], dtype=torch.int32, device="npu"),
        rtol=0,
        atol=0,
    )

    # Dead slots are dropped, so the CSR holds exactly the six live supports in
    # request-major order.
    assert _to_nested_list(tensors, num_sampled) == [row for row in expected if row]
