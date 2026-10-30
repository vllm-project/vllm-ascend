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
    # Run real tensor allocation, indexing and typed views on CPU. Only the
    # device capability query and Triton launches are substituted.
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.__dict__.update(
        triton=SimpleNamespace(
            jit=lambda function: MagicMock(name=function.__name__),
            cdiv=lambda value, block: (value + block - 1) // block,
        ),
        tl=SimpleNamespace(constexpr=lambda value: SimpleNamespace(value=value)),
    )
    device_utils = ModuleType("vllm_ascend.ops.triton.triton_utils")
    device_utils.__dict__.update(
        init_device_properties_triton=MagicMock(),
        get_vectorcore_num=MagicMock(return_value=17),
    )
    monkeypatch.setitem(sys.modules, triton_utils.__name__, triton_utils)
    monkeypatch.setitem(sys.modules, device_utils.__name__, device_utils)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/sample/penalties.py"
    spec = importlib.util.spec_from_file_location("_bincount_wrapper_ut", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "requests,max_prefill,vocab_size", [(1, 1, 33), (2, 1024, 4097), (1, 40961, 151936), (64, 1048576, 151936)]
)
def test_bincount_initialization_and_launch(penalties, dtype, requests, max_prefill, vocab_size):
    rows = requests + 2
    mapping = torch.arange(requests, 0, -1, dtype=dtype)
    words = (vocab_size + 31) // 32
    masks = torch.full((rows, words), -7, dtype=torch.int32)
    storage = torch.full((rows, vocab_size + 4), -7, dtype=torch.int32)
    counts = storage[:, 2:-2]
    expected = storage.clone()
    expected[mapping.long(), 2:-2] = 0
    # Large metadata tests the host launch bound without allocating 1M tokens.
    tokens = MagicMock(spec=torch.Tensor)
    tokens.stride.return_value = max_prefill + 17
    lengths = torch.full((rows,), max_prefill, dtype=torch.int32)
    penalties.bincount(mapping, tokens, lengths, lengths, masks, counts, max_prefill)

    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    assert torch.all(masks == -7)
    statistics = penalties._bincount_kernel.__getitem__
    assert statistics.call_args.args[0] == (min(17, requests * ((max_prefill + 1023) // 1024)),)
    launch = statistics.return_value.call_args
    assert launch.args[2] == max_prefill + 17
    flags = launch.args[5]
    assert flags.shape == (requests, words * 32)
    assert flags.dtype == torch.int8
    assert torch.count_nonzero(flags) == 0
    assert launch.args[8] == counts.stride(0)

    packing = penalties._pack_prompt_bin_mask_kernel.__getitem__
    word_blocks = (words + penalties.WORDS_PER_BLOCK - 1) // penalties.WORDS_PER_BLOCK
    assert packing.call_args.args[0] == (min(17, requests * word_blocks),)
    packed_launch = packing.return_value.call_args
    flag_words = packed_launch.args[1]
    assert flag_words.dtype == torch.int32
    assert flag_words.data_ptr() == flags.data_ptr()
    assert flag_words.shape == (requests, words * 8)
    assert packed_launch.args[2] == flag_words.stride(0)
    assert packed_launch.args[4] == masks.stride(0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bincount_empty_mapping(penalties, dtype):
    mapping = torch.empty(0, dtype=dtype)
    tensor = torch.full((2, 33), -7, dtype=torch.int32)
    lengths = torch.zeros(2, dtype=torch.int32)
    penalties.bincount(mapping, tensor, lengths, lengths, tensor, tensor, 0)
    assert torch.all(tensor == -7)
    penalties._bincount_kernel.__getitem__.assert_not_called()
    penalties._pack_prompt_bin_mask_kernel.__getitem__.assert_not_called()
    penalties.get_vectorcore_num.assert_not_called()


def test_bincount_zero_history(penalties):
    mapping = torch.tensor([1], dtype=torch.int32)
    tensor = torch.full((2, 33), -7, dtype=torch.int32)
    lengths = torch.zeros(2, dtype=torch.int32)
    penalties.bincount(mapping, tensor, lengths, lengths, tensor, tensor, 0)
    assert torch.all(tensor[0] == -7)
    assert torch.count_nonzero(tensor[1]) == 0
    penalties._bincount_kernel.__getitem__.assert_not_called()
    assert penalties._pack_prompt_bin_mask_kernel.__getitem__.call_count == 1
