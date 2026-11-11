# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.multi_module_mtp import speculator as module


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("target_prefilling", [False, True])
def test_full_draft_graph_uses_decode_padding(monkeypatch, mode, target_prefilling):
    """A one-token prefix remainder must not suppress padded draft FIA rows."""
    batch = object.__new__(AscendInputBatch)
    batch.is_prefilling_np = np.array([target_prefilling])
    speculator = object.__new__(module.AscendMultiModuleMTPSpeculator)
    speculator.input_batch = batch
    speculator.input_buffers = SimpleNamespace(positions=torch.arange(8))
    speculator.vllm_config = SimpleNamespace(parallel_config=object())
    factory = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", factory)
    upstream = MagicMock(return_value={"draft": object()})
    monkeypatch.setattr(module.MultiModuleMTPSpeculator, "_build_attn_metadata", upstream)
    descriptor = SimpleNamespace(cg_mode=mode, num_tokens=8)
    result = speculator._build_attn_metadata(1, descriptor, np.array([0, 4]), torch.tensor([129]), 0)
    assert result is upstream.return_value
    draft_prefilling = target_prefilling and mode != CUDAGraphMode.FULL
    assert bool(factory.call_args.args[2][0]) is draft_prefilling
    expected = AscendAttentionState.ChunkedPrefill if draft_prefilling else AscendAttentionState.DecodeOnly
    assert factory.call_args.kwargs["attn_state"] == expected
    assert bool(batch.is_prefilling_np[0]) is target_prefilling
