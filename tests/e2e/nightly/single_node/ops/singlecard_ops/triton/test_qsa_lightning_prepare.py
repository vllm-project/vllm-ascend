# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.qwen4_exp.lightning_prepare import prepare_qsa_lightning_inputs

pytestmark = [
    pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]


def _reference(query, positions, lengths, owners, table):
    rows, heads, width = query.shape
    padded = next(h for h in (8, 16, 24, 32, 64) if h >= heads)
    q = torch.zeros(rows, padded, width, dtype=query.dtype, device=query.device)
    q[:, :heads] = query
    weights = torch.zeros(rows, padded, dtype=query.dtype, device=query.device)
    weights[:, :heads] = 1 / math.sqrt(width)
    visible = torch.minimum((positions + 1).floor_divide(4), lengths[owners.long()].floor_divide(4)).int()
    return q, weights, visible, torch.arange(1, rows + 1, dtype=torch.int32, device=query.device), table[owners.long()]


@pytest.mark.parametrize("heads", [1, 4, 8, 9, 17, 24, 33, 64])
@pytest.mark.parametrize("rows", [0, 1, 32])
@torch.inference_mode()
def test_lightning_prepare_matches_torch(heads, rows):
    query = torch.randn(rows, heads + 1, 128, dtype=torch.bfloat16, device="npu")[:, :heads]
    table = torch.arange(3 * 2210, dtype=torch.int32, device="npu").view(3, 2210)[:, ::2]
    lengths = torch.tensor([0, 16384, 131073], dtype=torch.int32, device="npu")
    owners = torch.arange(rows, dtype=torch.int32, device="npu") % 3
    positions = torch.arange(rows * 2, dtype=torch.int64, device="npu")[::2] * 4096 - 5
    inputs = (query, positions, lengths, owners, table)
    expected = _reference(*inputs)
    actual = prepare_qsa_lightning_inputs(*inputs)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_lightning_prepare_replay_updates_metadata():
    query = torch.randn(32, 4, 128, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(32, dtype=torch.int64, device="npu") * 17
    lengths = torch.tensor([16384, 131072], dtype=torch.int64, device="npu")
    owners = torch.arange(32, dtype=torch.int64, device="npu") % 2
    table = torch.arange(14, dtype=torch.int64, device="npu").view(2, 7)
    args = (query, positions, lengths, owners, table)
    for _ in range(3):
        prepare_qsa_lightning_inputs(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = prepare_qsa_lightning_inputs(*args)
    for step in (1, 2, 3):
        query.add_(0.125)
        positions.add_(16384)
        lengths.sub_(4)
        owners.copy_((torch.arange(32, device="npu") + step) % 2 - 2)
        table.add_(7)
        graph.replay()
        for output, expected in zip(actual, _reference(*args)):
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_lightning_prepare_invalid_contract():
    query = torch.empty(2, 4, 128, dtype=torch.bfloat16, device="npu")
    positions = torch.zeros(2, dtype=torch.int64, device="npu")
    lengths = torch.ones(2, dtype=torch.int32, device="npu")
    table = torch.zeros(2, 3, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match="metadata"):
        prepare_qsa_lightning_inputs(query, positions.float(), lengths, positions, table)
    with pytest.raises(ValueError, match="match query rows"):
        prepare_qsa_lightning_inputs(query, positions[:1], lengths, positions, table)
    with pytest.raises(ValueError, match="nonempty"):
        prepare_qsa_lightning_inputs(query, positions, lengths, positions, table[:, :0])
