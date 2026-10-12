# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def xlite_module(monkeypatch):
    native = MagicMock(Model=type("Model", (), {}), ModelConfig=type("ModelConfig", (), {}))
    monkeypatch.setitem(sys.modules, "xlite._C", native)
    return importlib.import_module("vllm_ascend.xlite.xlite")


@pytest.mark.parametrize(("c8", "expected_mask"), [(False, []), (True, [False, True, False])])
def test_index_c8_mask(xlite_module, c8, expected_mask):
    adapter = xlite_module.DeepseekV32XliteModel.__new__(xlite_module.DeepseekV32XliteModel)
    adapter.xlite_config = SimpleNamespace(index_c8_mask=[])
    config = SimpleNamespace(
        index_head_dim=128, index_n_heads=32, index_topk=2048, indexer_types=["full", "full", "shared"]
    )
    adapter.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config))
    layers = [MagicMock(), MagicMock(), MagicMock()]
    layers[0].self_attn.mla_attn.mla_attn.impl.indexer.enable_sparse_li_c8 = False
    layers[1].self_attn.mla_attn.mla_attn.impl.indexer.enable_sparse_li_c8 = c8
    layers[2].self_attn.mla_attn.mla_attn.impl.indexer = None
    with (
        patch.object(xlite_module.DeepseekV3XliteModel, "_build_model_config"),
        patch.object(adapter, "_get_layers_and_model_prefix", return_value=(layers, "")),
    ):
        adapter._build_model_config()
    assert adapter.xlite_config.index_c8_mask == expected_mask
    assert adapter.xlite_config.index_full_mask == [True, True, False]


@pytest.mark.parametrize("index_c8_mask", [[False, True, False], [False, True]])
def test_extract_indexer_cache(xlite_module, index_c8_mask):
    adapter = xlite_module.DeepseekV32XliteModel.__new__(xlite_module.DeepseekV32XliteModel)
    adapter.xlite_config = SimpleNamespace(n_layers=3, index_full_mask=[True, True, False], index_c8_mask=index_c8_mask)
    mla_caches = [(torch.empty(1, dtype=torch.bfloat16), torch.empty(1, dtype=torch.bfloat16)) for _ in range(3)]
    indexer_cache = (torch.empty(1, dtype=torch.bfloat16),)
    indexer_c8_cache = (torch.empty(1, dtype=torch.int8), torch.empty(1, dtype=torch.float16))
    caches = [indexer_cache, mla_caches[0], mla_caches[1], indexer_c8_cache, mla_caches[2]]
    actual = adapter.extract_kv_cache(caches)
    dummy = importlib.import_module("vllm_ascend.xlite.utils")._DUMMY_TENSOR
    expected = [mla_caches[0] + indexer_cache, mla_caches[1] + indexer_c8_cache, mla_caches[2] + (dummy,)]
    assert len(actual) == len(expected)
    for actual_cache, expected_cache in zip(actual, expected):
        assert len(actual_cache) == len(expected_cache)
        assert all(value is expected_value for value, expected_value in zip(actual_cache, expected_cache))
