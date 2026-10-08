# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accepted lookback and UVA contracts; no NPU/hash-kernel execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _load_common():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/common.py"
    spec = importlib.util.spec_from_file_location("v41_pcp_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hash_mod = _load_common()


@pytest.mark.parametrize("start", [0, 1, 3, 5, 8])
@pytest.mark.parametrize("barrier", [17, 98, 99])
def test_chunk_and_decode_history_is_newest_first(start, barrier):
    tokens = torch.tensor([0, 5, 9, 13, barrier, 19, 23, 29, 31])
    actual = hash_mod.gather_engram_lookback(
        torch.arange(start, tokens.numel()),
        torch.tensor([0, tokens.numel() - start]),
        torch.tensor([0]),
        tokens[None],
        torch.tensor([tokens.numel()]),
        3,
    )
    expected = [int(tokens[start - 1 - column]) if start - 1 - column >= 0 else -1 for column in range(3)]
    # Image IDs remain raw here; NgramHashState receives their dead mask.
    assert actual.tolist() == [expected]


def test_mixed_requests_use_current_slot_mapping_and_accepted_tokens():
    token_table = torch.full((3, 12), 97)
    token_table[2, :7] = torch.tensor([0, 5, 9, 13, 17, 19, 23])
    token_table[0, :5] = torch.tensor([41, 43, 47, 53, 59])
    actual = hash_mod.gather_engram_lookback(
        torch.tensor([3, 4, 5, 6, 4]),
        torch.tensor([0, 4, 5]),
        torch.tensor([2, 0]),
        token_table,
        torch.tensor([5, 0, 7]),
        3,
    )
    assert actual.tolist() == [[9, 5, 0], [53, 47, 43]]


def test_recycled_request_tail_and_unaccepted_tokens_are_excluded():
    actual = hash_mod.gather_engram_lookback(
        torch.tensor([3]),
        torch.tensor([0, 1]),
        torch.tensor([1]),
        torch.tensor([[71, 73, 79, 83], [5, 9, 97, 97]]),
        torch.tensor([4, 2]),
        3,
    )
    assert actual.tolist() == [[-1, 9, 5]]


def test_empty_batch_has_empty_lookback():
    actual = hash_mod.gather_engram_lookback(
        torch.empty(0, dtype=torch.long),
        torch.tensor([0]),
        torch.empty(0, dtype=torch.long),
        torch.zeros((2, 8), dtype=torch.long),
        torch.tensor([0, 0]),
        3,
    )
    assert actual.shape == (0, 3)


def test_uva_cpu_history_waits_for_pending_device_writes(monkeypatch):
    token_table = torch.tensor([[0, 5, 97, 13]])
    stream = SimpleNamespace(synchronize=MagicMock(side_effect=lambda: token_table.__setitem__((0, 2), 9)))
    device = SimpleNamespace(type="npu")
    npu = SimpleNamespace(current_stream=MagicMock(return_value=stream))
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    actual = hash_mod.gather_engram_lookback(
        torch.tensor([3]),
        torch.tensor([0, 1]),
        torch.tensor([0]),
        token_table,
        torch.tensor([4]),
        3,
        execution_device=device,
    )
    assert actual.tolist() == [[9, 5, 0]]
    npu.current_stream.assert_called_once_with(device)
    stream.synchronize.assert_called_once_with()


def test_uva_moves_only_short_control_vectors_to_cpu(monkeypatch):
    npu_device = SimpleNamespace(type="npu")
    moved_shapes = []

    class DeviceVector:
        device = npu_device

        def __init__(self, tensor):
            self.tensor = tensor

        def __getitem__(self, key):
            return DeviceVector(self.tensor[key])

        def to(self, *, device, dtype=None):
            if device.type == "cpu":
                moved_shapes.append(self.tensor.shape)
                return self.tensor.to(dtype=dtype)
            assert device is npu_device
            return DeviceVector(self.tensor.to(dtype=dtype))

        def index_select(self, dim, indices):
            return DeviceVector(self.tensor.index_select(dim, indices.tensor))

    stream = SimpleNamespace(synchronize=MagicMock())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda device: stream), raising=False)
    actual = hash_mod.gather_engram_lookback(
        DeviceVector(torch.tensor([3, 2, 3])),
        DeviceVector(torch.tensor([0, 1, 3])),
        DeviceVector(torch.tensor([0, 1])),
        torch.tensor([[0, 5, 9, 13], [41, 43, 47, 53]]),
        DeviceVector(torch.tensor([4, 4])),
        3,
    )
    assert actual.tolist() == [[9, 5, 0], [43, 41, -1]]
    assert moved_shapes == [torch.Size([2])] * 3
    stream.synchronize.assert_called_once_with()


def test_device_history_keeps_indices_on_its_device(monkeypatch):
    device = torch.device("meta")
    stream = MagicMock()
    monkeypatch.setattr(torch, "meta", SimpleNamespace(current_stream=stream), raising=False)
    result = hash_mod.gather_engram_lookback(
        torch.tensor([3]),
        torch.tensor([0, 1]),
        torch.tensor([0]),
        torch.empty((1, 4), dtype=torch.long, device=device),
        torch.tensor([4]),
        3,
        execution_device=device,
    )
    assert result.device == device and result.shape == (1, 3)
    stream.assert_not_called()
