# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.dsv41_a5 import indexer_qw


@pytest.fixture
def weights(monkeypatch):
    vendor = ModuleType("vllm_ascend.ops.pythondsl.indexer_prologue_qw_interleaved")
    vendor.to_nz = lambda tensor: tensor.clone()
    calls = []

    def kernel(x, q, wqb, ww, qs, ws, sin, cos, **kwargs):
        calls.append((x, q, wqb, ww, qs, ws, sin, cos, kwargs))
        return (
            torch.zeros(q.shape[0], 32, 64, dtype=torch.uint8),
            torch.zeros(q.shape[0], 32, 2, 2, dtype=torch.uint8),
            torch.full((q.shape[0], 32), 0.123456),
        )

    vendor.indexer_prologue_qw = kernel
    monkeypatch.setitem(sys.modules, vendor.__name__, vendor)
    wq = SimpleNamespace(
        weight=torch.zeros(1280, 4096, dtype=torch.float8_e4m3fn),
        weight_scale=torch.zeros(20, 4096, 2, dtype=torch.uint8),
        quant_method=SimpleNamespace(quant_method=SimpleNamespace(dynamic_mx_quant_scale_alg=0)),
    )
    ww = SimpleNamespace(weight=torch.zeros(32, 5120, dtype=torch.bfloat16))
    return wq, ww, calls


def test_static_weight_pack_and_paired_scale_layout(weights):
    wq, ww, _ = weights
    adapter = indexer_qw.IndexerQWFusion(wq, ww, 128**-0.5 * 32**-0.5)
    assert adapter.wqb_nz.shape == (4096, 1280)
    assert adapter.wqb_scale.shape == (4096, 20, 2)
    assert adapter.ww_nz.shape == (32, 5120)
    assert not adapter.state_dict()  # no checkpoint schema changes
    assert wq.weight.shape == (1280, 4096)  # eager fallback weights untouched


def test_dynamic_inputs_keep_scale_algorithm_and_bf16_weights_rounding(weights, monkeypatch):
    wq, ww, calls = weights
    adapter = indexer_qw.IndexerQWFusion(wq, ww, 1 / 64)
    algorithms = []

    def quant(value, *, dst_type, scale_alg):
        algorithms.append(scale_alg)
        return value.to(dst_type), torch.zeros(value.shape[0], 20, 2, dtype=torch.uint8)

    monkeypatch.setattr(indexer_qw.torch_npu, "npu_dynamic_mx_quant", quant)
    q, scale, weights_out = adapter(
        torch.zeros(3, 5120, dtype=torch.bfloat16),
        torch.zeros(3, 1280, dtype=torch.bfloat16),
        torch.ones(3, 1, 1, 64),
        torch.zeros(3, 1, 1, 64),
    )
    assert algorithms == [0]
    assert q.shape == (3, 32, 64)
    assert scale.shape == (3, 32, 4)
    assert calls[0][6].shape == (3, 64)
    assert calls[0][7].dtype == torch.float32
    assert torch.equal(weights_out, torch.full((3, 32), 0.123456).bfloat16().float())


def test_unqualified_geometry_fails_before_serving(weights):
    wq, ww, _ = weights
    wq.weight = torch.zeros(64, 128, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="postprocessed Flash"):
        indexer_qw.IndexerQWFusion(wq, ww, 1 / 64)


@pytest.mark.parametrize("backend", [None, object()])
@pytest.mark.parametrize("compatible", [False, True])
def test_fusion_is_automatic_only_for_supported_a5_weights(weights, backend, compatible):
    from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer

    wq, ww, _ = weights
    if not compatible:
        wq.weight = torch.zeros(1280, 4096, dtype=torch.bfloat16)
    indexer = SimpleNamespace(dsv41_backend=backend, wq_b=wq, weights_proj=ww, weights_scale=1 / 64, qw_fusion=None)
    DeepseekV41Indexer.prepare_qw_fusion(indexer)
    assert (indexer.qw_fusion is not None) == (backend is not None and compatible)


def test_model_postload_initializes_indexer_without_config_switch(weights):
    from unittest.mock import patch

    from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer
    from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM

    indexer = object.__new__(DeepseekV41Indexer)
    model = SimpleNamespace(model=SimpleNamespace(modules=lambda: [indexer]))
    with patch.object(DeepseekV41Indexer, "prepare_qw_fusion") as prepare:
        AscendDeepseekV41LLMForCausalLM.process_weights_after_loading(model)
    prepare.assert_called_once_with()
