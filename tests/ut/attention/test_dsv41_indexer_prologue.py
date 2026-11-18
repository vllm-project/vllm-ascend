# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused QW reuses quantized input and preserves the native K path."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl
from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer
from vllm_ascend.ops import dsv41_indexer_prologue


def test_preparation_uses_fused_outputs_without_native_projections():
    packed = torch.zeros(6, 32, 64, dtype=torch.uint8)
    scale = torch.ones(6, 32, 2, 2, dtype=torch.uint8)
    weights = torch.ones(6, 32)
    qr, qs = torch.zeros(6, 1280, dtype=torch.uint8), torch.ones(6, 20, 2, dtype=torch.uint8)
    indexer = SimpleNamespace(
        fused_prologue_qw=object(),
        project_fused=Mock(return_value=(packed, scale, weights)),
        project_query=Mock(side_effect=AssertionError("duplicate projection")),
        project_weights=Mock(side_effect=AssertionError("duplicate weights")),
        quantize_query=Mock(side_effect=AssertionError("duplicate quantization")),
    )
    impl = AscendDSAV41Impl.__new__(AscendDSAV41Impl)
    impl.role = SimpleNamespace(has_long_context=True, is_index_source=True)
    hidden, cos, sin = torch.zeros(6, 5120), torch.zeros(6, 1, 1, 64), torch.zeros(6, 1, 1, 64)
    prepared = impl._prepare_indexer_inputs(SimpleNamespace(indexer=indexer), hidden, (qr, qs), cos, sin)
    assert prepared.query is prepared.quantized_query is packed
    assert prepared.query_scale is scale
    assert prepared.weights is weights
    indexer.project_fused.assert_called_once_with(hidden, qr, qs, cos, sin)
    assert not impl._should_quantize_indexer(prepared, None)


def test_qw_weight_preparation_preserves_native_k_storage_and_dtype(monkeypatch):
    indexer = SimpleNamespace(
        weights_proj=torch.nn.Linear(5120, 32, bias=False, dtype=torch.bfloat16),
        wk=torch.nn.Linear(512, 128, bias=False, dtype=torch.bfloat16),
        k_norm=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))),
        owns_k=True,
    )
    k_ptr = indexer.wk.weight.data_ptr()
    gamma_ptr = indexer.k_norm.weight.data_ptr()
    qw_ptr = indexer.weights_proj.weight.data_ptr()
    cast = Mock(side_effect=lambda value, fmt: value.clone())
    monkeypatch.setattr(dsv41_indexer_prologue.torch_npu, "npu_format_cast", cast)
    dsv41_indexer_prologue.prepare_indexer_weights(indexer)
    assert cast.call_count == 1
    assert cast.call_args.args[0].data_ptr() == qw_ptr
    assert indexer.wk.weight.data_ptr() == k_ptr
    assert indexer.k_norm.weight.data_ptr() == gamma_ptr
    assert indexer.wk.weight.dtype == indexer.k_norm.weight.dtype == torch.bfloat16


def test_qw_wrapper_passes_tensor_aliases_and_only_rope_views():
    indexer = DeepseekV41Indexer.__new__(DeepseekV41Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.rope_width = 64
    indexer.weights_scale = 1.0 / 64
    indexer.wq_b = torch.nn.Linear(1280, 4096, bias=False)
    indexer.wq_b.weight_scale = torch.nn.Parameter(torch.ones(4096, 20, 2), requires_grad=False)
    indexer.weights_proj = torch.nn.Linear(5120, 32, bias=False)
    indexer.fused_prologue_qw = Mock(return_value=(1, 2, 3))
    hidden, qr, qs = torch.zeros(6, 5120), torch.zeros(6, 1280), torch.zeros(6, 20, 2)
    cos, sin = torch.zeros(6, 1, 1, 64), torch.zeros(6, 1, 1, 64)
    assert indexer.project_fused(hidden, qr, qs, cos, sin) == (1, 2, 3)
    call = indexer.fused_prologue_qw.call_args
    assert call.args[0] is hidden and call.args[1] is qr and call.args[4] is qs
    assert type(call.args[2]) is torch.Tensor
    assert call.args[2].data_ptr() == indexer.wq_b.weight.data_ptr()
    assert call.args[6].data_ptr() == sin.data_ptr()
    assert call.args[7].data_ptr() == cos.data_ptr()
