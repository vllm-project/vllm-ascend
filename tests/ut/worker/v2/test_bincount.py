# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def penalties(monkeypatch):
    # Exercise real CPU tensor initialization while mocking only NPU launches.
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.triton = SimpleNamespace(
        jit=lambda function: MagicMock(name=function.__name__),
        cdiv=lambda value, block: (value + block - 1) // block,
    )
    triton_utils.tl = MagicMock()
    device_utils = ModuleType("vllm_ascend.ops.triton.triton_utils")
    device_utils.init_device_properties_triton = MagicMock()
    device_utils.get_vectorcore_num = MagicMock(return_value=17)
    monkeypatch.setitem(sys.modules, triton_utils.__name__, triton_utils)
    monkeypatch.setitem(sys.modules, device_utils.__name__, device_utils)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/sample/penalties.py"
    spec = importlib.util.spec_from_file_location("_bincount_wrapper_ut", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "requests,max_prefill,vocab_size,expected_programs",
    [(1, 1, 33, 1), (2, 1024, 4097, 2), (1, 40961, 33, 17), (64, 1048576, 33, 17)],
)
def test_bincount_initialization_and_launch(penalties, dtype, requests, max_prefill, vocab_size, expected_programs):
    rows = requests + 2
    mapping = torch.arange(requests, 0, -1, dtype=dtype)
    words = (vocab_size + 31) // 32
    masks = torch.full((rows, words), -7, dtype=torch.int32)
    storage = torch.full((rows, vocab_size + 4), -7, dtype=torch.int32)
    counts = storage[:, 2:-2]
    expected = storage.clone()
    expected[mapping.long(), 2:-2] = 0
    # The mocked launch does not dereference token storage, so large context
    # metadata can test launch limits without allocating a 1M-token input.
    tokens = MagicMock(spec=torch.Tensor)
    tokens.stride.side_effect = lambda axis: max_prefill + 17 if axis == 0 else 1
    lengths = torch.full((rows,), max_prefill, dtype=torch.int32)

    penalties.bincount(mapping, tokens, lengths, lengths, masks, counts, max_prefill)

    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    assert torch.all(masks == -7)  # Packing is mocked; there is no redundant mask clear.
    assert penalties._bincount_kernel.__getitem__.call_args.args[0] == (expected_programs,)
    launch = penalties._bincount_kernel.__getitem__.return_value.call_args
    assert launch.args[2] == max_prefill + 17
    flags = launch.args[5]
    assert flags.shape == (requests, words * 32)
    assert flags.dtype == torch.int8
    assert torch.count_nonzero(flags) == 0
    assert launch.args[8] == counts.stride(0)
    assert penalties._pack_prompt_bin_mask_kernel.__getitem__.call_count == 1


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bincount_empty_mapping_does_not_launch(penalties, dtype):
    mapping = torch.empty(0, dtype=dtype)
    tensor = torch.full((2, 33), -7, dtype=torch.int32)
    lengths = torch.zeros(2, dtype=torch.int32)
    penalties.bincount(mapping, tensor, lengths, lengths, tensor, tensor, 0)
    assert torch.all(tensor == -7)
    penalties._bincount_kernel.__getitem__.assert_not_called()
    penalties._pack_prompt_bin_mask_kernel.__getitem__.assert_not_called()
    penalties.get_vectorcore_num.assert_not_called()
