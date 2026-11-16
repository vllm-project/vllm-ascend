# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for request-major sampling masks on Ascend."""

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.sampling_mask import sampling_mask_from_logits_npu


def _from_logits(logits, num_sampled_tokens, max_num_kept=512, cu_num_logits=None, rows_per_request=1):
    if cu_num_logits is None:
        cu_num_logits = torch.arange(len(num_sampled_tokens) + 1, dtype=torch.int32, device=logits.device)
    return sampling_mask_from_logits_npu(
        SamplingMaskTensors,
        logits,
        cu_num_logits,
        num_sampled_tokens,
        max_num_kept,
        rows_per_request,
    )


def _to_nested_list(tensors, num_sampled_tokens):
    cpu_tensors = tensors.to_cpu_nonblocking()
    torch.npu.synchronize()
    assert tensors.token_ids.shape == (len(num_sampled_tokens) * tensors.rows_per_request, 0)
    return cpu_tensors.tolists(num_sampled_tokens.cpu().numpy()).to_nested_list()


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
    num_sampled[-1] = 1
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
    assert result == expected


def _ragged_logits(vocab_size, rows_per_request, strided=False):
    spans = [rows_per_request + 2, rows_per_request, rows_per_request + 3, 0]
    cu = np.array([2, *[2 + sum(spans[:i]) for i in range(1, 5)]], dtype=np.int32)
    sampled = np.array([rows_per_request, rows_per_request - 1, 0, 0], dtype=np.int32)
    cpu = torch.full((int(cu[-1]), vocab_size), -float("inf"))
    for row in range(cpu.shape[0]):
        indices = (torch.arange(min(vocab_size, 23)) * 97 + row * 13) % vocab_size
        cpu[row, indices] = 1.0
    cpu[int(cu[0]) + rows_per_request - 1] = 0.0
    if vocab_size > 9:
        cpu[int(cu[0]), 0] = float("nan")
        cpu[int(cu[0]), 1] = float("inf")
    if strided:
        base = torch.empty((cpu.shape[0], vocab_size * 2), device="npu")
        logits = base[:, ::2]
        logits.copy_(cpu.to("npu"))
    else:
        logits = cpu.to("npu")
    return logits, cpu, cu, sampled


def _assert_request_major(tensors, cpu_logits, cu, sampled):
    rows_per_request = tensors.rows_per_request
    expected_rows = []
    for req in range(len(sampled)):
        for slot in range(rows_per_request):
            if slot < sampled[req]:
                source = cpu_logits[int(cu[req]) + slot].numpy()
                expected_rows.append(np.flatnonzero(np.isfinite(source)).tolist())
            else:
                expected_rows.append([])
    cpu = tensors.to_cpu_nonblocking()
    torch.npu.synchronize()
    assert cpu.token_ids.shape == (len(sampled) * rows_per_request, 0)
    assert cpu.counts.tolist() == [len(row) for row in expected_rows]
    bits = np.unpackbits(cpu.packed_mask.numpy(), axis=1, count=cpu.vocab_size, bitorder="little")
    assert [np.flatnonzero(row).tolist() for row in bits] == expected_rows
    csr = cpu.tolists(sampled)
    if rows_per_request == 1:
        expected = expected_rows
        assert csr.cu_num_generated_tokens is None
    else:
        expected = [
            expected_rows[req * rows_per_request + slot]
            for req in range(len(sampled))
            for slot in range(int(sampled[req]))
        ]
        assert csr.cu_num_generated_tokens == np.cumsum(np.r_[0, sampled]).tolist()
    assert csr.to_nested_list() == expected
    for req, count in enumerate(sampled):
        for kept in range(int(count) + 1):
            wanted = expected_rows[req * rows_per_request : req * rows_per_request + kept]
            assert csr.slice_request(req, kept).to_nested_list() == wanted


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("rows_per_request", [1, 2, 4])
@pytest.mark.parametrize("vocab_size", [9, 4097, 248320])
@pytest.mark.parametrize("strided", [False, True])
def test_sampling_mask_request_major_ragged_rows(rows_per_request, vocab_size, strided):
    """Counts select committed slots; row offsets may contain gaps and empty spans."""
    init_device_properties_triton()
    logits, cpu, cu, sampled = _ragged_logits(vocab_size, rows_per_request, strided)
    tensors = _from_logits(
        logits,
        torch.from_numpy(sampled).to("npu"),
        max_num_kept=8,
        cu_num_logits=torch.from_numpy(cu).to("npu"),
        rows_per_request=rows_per_request,
    )
    assert tensors.rows_per_request == rows_per_request
    _assert_request_major(tensors, cpu, cu, sampled)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
def test_sampling_mask_request_major_chunk_cat():
    """Chunk-local request metadata retains offsets into the shared logits tensor."""
    init_device_properties_triton()
    logits, cpu, cu, sampled = _ragged_logits(4097, 3)
    chunks = []
    for lo, hi in [(0, 1), (1, 4)]:
        chunks.append(
            _from_logits(
                logits,
                torch.from_numpy(sampled[lo:hi]).to("npu"),
                cu_num_logits=torch.from_numpy(cu[lo : hi + 1]).to("npu"),
                rows_per_request=3,
            )
        )
    _assert_request_major(SamplingMaskTensors.cat(chunks), cpu, cu, sampled)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
def test_sampling_mask_empty_request_batch():
    tensors = _from_logits(
        torch.empty((0, 9), device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        cu_num_logits=torch.zeros(1, dtype=torch.int32, device="npu"),
        rows_per_request=3,
    )
    assert tensors.packed_mask.shape == (0, 2)
    assert tensors.counts.shape == (0,)
    assert tensors.rows_per_request == 3
    assert _to_nested_list(tensors, torch.empty(0, dtype=torch.int32, device="npu")) == []


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
def test_sampling_mask_graph_replay_updates_row_metadata():
    """A captured layout must read new device offsets/counts on replay."""
    init_device_properties_triton()
    logits, cpu, cu, sampled = _ragged_logits(4097, 3)
    cu_device = torch.from_numpy(cu).to("npu")
    sampled_device = torch.from_numpy(sampled).to("npu")
    for _ in range(3):
        _from_logits(logits, sampled_device, cu_num_logits=cu_device, rows_per_request=3)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        tensors = _from_logits(logits, sampled_device, cu_num_logits=cu_device, rows_per_request=3)
    graph.replay()
    _assert_request_major(tensors, cpu, cu, sampled)
    changed_cu = cu.copy()
    changed_cu[0] += 1
    changed_sampled = np.array([1, 2, 1, 0], dtype=np.int32)
    cu_device.copy_(torch.from_numpy(changed_cu))
    sampled_device.copy_(torch.from_numpy(changed_sampled))
    graph.replay()
    _assert_request_major(tensors, cpu, changed_cu, changed_sampled)


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
    expected = [expected_ids, [], expected_ids, []]

    assert result == expected
