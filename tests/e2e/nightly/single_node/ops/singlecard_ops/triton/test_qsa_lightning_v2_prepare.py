# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.qwen4_exp.lightning_v2_prepare import prepare_qsa_lightning_v2_inputs

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")

_HEAD_DIM = 128
_COMPRESS_RATIO = 4


def _inputs(rows: int, requests: int, heads: int, dtype: torch.dtype, strided: bool):
    query = torch.zeros((rows, heads, _HEAD_DIM), dtype=torch.bfloat16, device="npu")
    # Full requests, a partial final request and padded requests share one
    # boundary buffer. The wrapper clamps unused boundaries to the live rows.
    query_chunk = max(1, (rows + requests - 1) // max(1, requests))
    starts = torch.arange(requests + 1, dtype=dtype, device="npu") * query_chunk
    lengths = torch.arange(requests, dtype=dtype, device="npu") * 5 + query_chunk
    lengths = torch.where(starts[:-1] < rows, lengths, 0)
    if strided:
        lengths = torch.stack((lengths, lengths), dim=1)[:, 0]
        starts = torch.stack((starts, starts), dim=1)[:, 0]
    return query, lengths, starts


def _reference(query, lengths, starts):
    # Preserve the original BF16 rounding before promotion to V2's FP32 weights.
    weight = torch.tensor(1 / _HEAD_DIM**0.5, dtype=torch.bfloat16, device=query.device).float()
    return (
        weight.expand(query.shape[:2]).clone(),
        starts[: lengths.numel() + 1].clamp_max(query.shape[0]),
        lengths.floor_divide(_COMPRESS_RATIO),
        lengths.remainder(_COMPRESS_RATIO),
    )


def _assert_outputs(actual, expected):
    assert len(actual) == len(expected) == 4
    for result, reference in zip(actual, expected):
        assert result.is_contiguous()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("rows,requests", [(0, 0), (0, 3), (1, 1), (4, 8), (32, 1), (32, 8), (128, 32), (512, 129)])
@pytest.mark.parametrize("heads", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_qsa_lightning_v2_prepare_matches_reference(rows, requests, heads, dtype, strided):
    inputs = _inputs(rows, requests, heads, dtype, strided)
    snapshots = tuple(tensor.clone() for tensor in inputs)
    actual = prepare_qsa_lightning_v2_inputs(*inputs, _COMPRESS_RATIO)
    _assert_outputs(actual, _reference(*inputs))
    for tensor, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_qsa_lightning_v2_prepare_replay_uses_updated_metadata(dtype):
    inputs = _inputs(32, 8, 4, dtype, strided=True)
    for _ in range(3):
        prepare_qsa_lightning_v2_inputs(*inputs, _COMPRESS_RATIO)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = prepare_qsa_lightning_v2_inputs(*inputs, _COMPRESS_RATIO)
    for increment in (1, 7, 4):
        # Keep addresses and strides fixed while changing device-side values.
        inputs[1].add_(increment)
        inputs[2][1:].add_(increment)
        graph.replay()
        _assert_outputs(captured, _reference(*inputs))
