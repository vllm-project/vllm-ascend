# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU correctness tests for fused C2 metadata; no model weights needed."""

import pytest
import torch
import torch_npu  # noqa: F401 -- register the NPU backend

from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata


@pytest.fixture(autouse=True)
def require_real_npu():
    # Fail rather than report mock/CPU execution as NPU correctness evidence.
    assert torch.npu.is_available() is True, "Run this file on an actual Ascend NPU"


def _inputs(query_lens, *, position_offset=0, strided=False):
    starts = [0]
    seq_lens, positions = [], []
    for request, length in enumerate(query_lens):
        first_position = request * 7 + position_offset
        starts.append(starts[-1] + length)
        seq_lens.append(first_position + length)
        positions.extend(range(first_position, first_position + length))
    num_reqs, num_tokens = len(query_lens), len(positions)
    width = 64
    rows = max(positions, default=0) + 2
    values = torch.arange(rows * width, dtype=torch.float32).reshape(rows, 1, 1, width)
    result = dict(
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        positions=torch.tensor(positions, dtype=torch.int64),
        block_table=(torch.arange(num_reqs, dtype=torch.int32) * 3 + 7).reshape(num_reqs, 1),
        # Distinct, nonzero row-zero values detect unwritten inactive RoPE rows.
        source_cos=values / 128 + 0.25,
        source_sin=-values / 64 - 0.5,
    )
    if strided:
        # Copy on the device after allocation so .to() cannot collapse strides.
        strides = {
            "query_start_loc": (2,),
            "seq_lens": (2,),
            "positions": (2,),
            "block_table": (5, 2),
            "source_cos": (width * 4, width * 4, width * 4, 2),
            "source_sin": (width * 4, width * 4, width * 4, 2),
        }
    else:
        strides = {name: tensor.stride() for name, tensor in result.items()}
    device_inputs = {}
    for name, tensor in result.items():
        device_tensor = torch.empty_strided(tensor.shape, strides[name], dtype=tensor.dtype, device="npu")
        device_tensor.copy_(tensor)
        device_inputs[name] = device_tensor
    assert device_inputs["positions"].numel() == num_tokens
    return result, device_inputs


def _reference(inputs, num_actual_reqs, num_actual_tokens, skip_update):
    """Scalar oracle for the eager builder's controls and RoPE gathers."""
    starts = inputs["query_start_loc"].tolist()
    seq_lens = inputs["seq_lens"].tolist()
    positions = inputs["positions"].tolist()
    blocks = inputs["block_table"][:, 0].tolist()
    num_reqs = len(seq_lens)
    ring = torch.zeros((5, num_reqs), dtype=torch.int32)
    for request in range(num_reqs):
        begin, end = starts[request : request + 2]
        used = max(min(end, num_actual_tokens) - begin, 0)
        if request >= num_actual_reqs or skip_update:
            used = 0
        ring[:, request] = torch.tensor(
            [max(seq_lens[request] - (end - begin), 0), used, begin, begin, blocks[request] if used else 0],
            dtype=torch.int32,
        )
    valid_end = min(starts[num_actual_reqs], num_actual_tokens)
    complete = [token < valid_end and position % 2 == 1 and not skip_update for token, position in enumerate(positions)]
    source_positions = torch.tensor(
        [position - 1 if valid else 0 for position, valid in zip(positions, complete)], dtype=torch.int64
    )
    return dict(
        ring_metadata_output=ring,
        complete_mask_output=torch.tensor(complete, dtype=torch.bool),
        source_positions_output=source_positions,
        cos_output=inputs["source_cos"].index_select(0, source_positions),
        sin_output=inputs["source_sin"].index_select(0, source_positions),
    )


def _outputs(inputs):
    num_reqs = inputs["seq_lens"].numel()
    num_tokens = inputs["positions"].numel()
    width = inputs["source_cos"].shape[-1]
    return dict(
        ring_metadata_output=torch.full((5, num_reqs), -777, dtype=torch.int32, device="npu"),
        complete_mask_output=torch.ones(num_tokens, dtype=torch.bool, device="npu"),
        source_positions_output=torch.full((num_tokens,), -777, dtype=torch.int64, device="npu"),
        cos_output=torch.full((num_tokens, 1, 1, width), float("nan"), device="npu", dtype=torch.float32),
        sin_output=torch.full((num_tokens, 1, 1, width), float("nan"), device="npu", dtype=torch.float32),
    )


def _run_and_compare(cpu_inputs, device_inputs, outputs, num_actual_reqs, num_actual_tokens, skip_update):
    build_c2_ring_metadata(
        **device_inputs,
        num_reqs=device_inputs["seq_lens"].numel(),
        num_tokens=device_inputs["positions"].numel(),
        num_actual_reqs=num_actual_reqs,
        num_actual_tokens=num_actual_tokens,
        skip_update=skip_update,
        **outputs,
    )
    torch.npu.synchronize()
    expected = _reference(cpu_inputs, num_actual_reqs, num_actual_tokens, skip_update)
    for name, actual in outputs.items():
        # The kernel only computes integer controls and copies FP32 rows.
        torch.testing.assert_close(
            actual.cpu(), expected[name], rtol=0, atol=0, msg=lambda msg, key=name: f"{key}: {msg}"
        )


@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
@pytest.mark.parametrize("skip_update", [False, True], ids=["update", "skip"])
@pytest.mark.parametrize(
    "query_lens,num_actual_reqs,num_actual_tokens,position_offset",
    [
        pytest.param([], 0, 0, 0, id="empty"),
        pytest.param([0, 0], 0, 0, 0, id="requests-with-no-tokens"),
        pytest.param([1], 1, 1, 0, id="single-even"),
        pytest.param([1], 1, 1, 1, id="single-odd"),
        pytest.param([2, 1, 5, 0, 3], 5, 11, 0, id="mixed-prefill-decode"),
        pytest.param([2, 3, 1], 3, 4, 0, id="token-cut-inside-request"),
        pytest.param([2, 3, 1], 1, 6, 0, id="padded-requests"),
        pytest.param([2, 3, 1], 3, 0, 0, id="zero-actual-tokens"),
        pytest.param([2, 3, 1], 0, 6, 0, id="zero-actual-requests"),
        pytest.param([1] * 63, 61, 62, 0, id="req-block-minus-one"),
        pytest.param([1] * 64, 64, 64, 0, id="req-block"),
        pytest.param([1] * 65, 63, 64, 0, id="req-block-plus-one"),
        pytest.param([15], 1, 15, 0, id="token-block-minus-one"),
        pytest.param([16], 1, 16, 0, id="token-block"),
        pytest.param([17], 1, 17, 0, id="token-block-plus-one"),
    ],
)
def test_c2_ring_metadata_matches_reference(
    query_lens, num_actual_reqs, num_actual_tokens, position_offset, skip_update, strided
):
    cpu_inputs, device_inputs = _inputs(query_lens, position_offset=position_offset, strided=strided)
    _run_and_compare(
        cpu_inputs, device_inputs, _outputs(device_inputs), num_actual_reqs, num_actual_tokens, skip_update
    )


@pytest.mark.parametrize("strided", [False, True])
def test_c2_ring_metadata_overwrites_reused_buffers(strided):
    query_lens = [2, 3, 1]
    cpu_inputs, device_inputs = _inputs(query_lens, strided=strided)
    outputs = _outputs(device_inputs)
    pointers = {name: value.data_ptr() for name, value in outputs.items()}
    # Keep physical batch dimensions fixed while active counts shrink and grow.
    for offset, actual_reqs, actual_tokens, skip in [
        (0, 3, 6, False),
        (1, 1, 1, False),
        (2, 0, 0, False),
        (3, 3, 6, True),
        (4, 3, 6, False),
    ]:
        next_cpu, next_device = _inputs(query_lens, position_offset=offset, strided=strided)
        # Keep scheduler input buffers stable as well; RoPE tables are immutable
        # within each call, but the new positions require longer tables.
        for name in ("query_start_loc", "seq_lens", "positions", "block_table"):
            device_inputs[name].copy_(next_device[name])
        for name in ("source_cos", "source_sin"):
            device_inputs[name] = next_device[name]
        _run_and_compare(next_cpu, device_inputs, outputs, actual_reqs, actual_tokens, skip)
        assert {name: value.data_ptr() for name, value in outputs.items()} == pointers
