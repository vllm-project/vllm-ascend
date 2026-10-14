# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.config import CompilationMode
from vllm.model_executor.models import deepseek_v2

from vllm_ascend.models import deepseek_mtp
from vllm_ascend.models.deepseek_mtp import AscendDSACPDecoderLayer, AscendDSACPModel, AscendGlmMoeDsaForCausalLM


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_consecutive_mixed_layers_keep_local_tokens_and_residual(monkeypatch, rank, dtype):
    # Global order is decode followed by prefill. Some ranks own decode only;
    # model layers must keep the same shard in either part of the mixed batch.
    hidden = torch.arange(rank * 12, (rank + 1) * 12, dtype=dtype).reshape(4, 3)
    residual = hidden.clone()
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = True
    layer.use_mha = False
    layer.routed_scaling_factor = 1.0
    layer.layer_idx = 1
    layer.input_layernorm = Mock(side_effect=lambda x, r: (x, r))
    layer.post_attention_layernorm = Mock(side_effect=lambda x, r: (x, r))
    wrapper = SimpleNamespace(
        supports_sequence_parallel_input=True,
        forward_sequence_parallel=Mock(side_effect=lambda positions, x, *, output_scale: (x + 1) * output_scale),
    )
    layer.self_attn = Mock()
    layer.self_attn.mla_attn = wrapper
    layer.mlp = Mock(side_effect=lambda x, **kw: x * 2)
    gather, scatter = Mock(), Mock()
    monkeypatch.setattr(deepseek_v2, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(deepseek_v2, "tensor_model_parallel_reduce_scatter", scatter)

    actual, actual_residual = layer(torch.arange(14), hidden, residual)
    actual, actual_residual = layer(torch.arange(14), actual, actual_residual)

    torch.testing.assert_close(actual, hidden * 4 + 6)
    assert actual.shape == hidden.shape
    assert actual_residual is residual
    assert wrapper.forward_sequence_parallel.call_count == 2
    assert all(call.args[1].shape == hidden.shape for call in wrapper.forward_sequence_parallel.call_args_list)
    assert all(call.kwargs == {"already_sequence_parallel": True} for call in layer.mlp.call_args_list)
    layer.self_attn.assert_not_called()
    gather.assert_not_called()
    scatter.assert_not_called()


@pytest.mark.parametrize("boundary", ["replicated", "first_layer", "dense", "unsupported", "query_scaling"])
def test_unsupported_layouts_preserve_upstream_behavior(monkeypatch, boundary):
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = boundary != "dense"
    layer.use_mha = False
    layer.self_attn = SimpleNamespace(
        mla_attn=SimpleNamespace(supports_sequence_parallel_input=boundary != "unsupported")
    )
    hidden = torch.ones(4 if boundary == "replicated" else 2, 3, dtype=torch.bfloat16)
    positions = torch.arange(4)
    residual = None if boundary == "first_layer" else hidden.clone()
    scaling = torch.ones(4) if boundary == "query_scaling" else None
    expected = (hidden, residual)
    upstream = Mock(return_value=expected)
    monkeypatch.setattr(deepseek_v2.DeepseekV2DecoderLayer, "forward", upstream)

    assert layer.forward(positions, hidden, residual, scaling) is expected
    upstream.assert_called_once_with(positions, hidden, residual, scaling)


def test_local_attention_uses_explicit_sequence_parallel_interface():
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = True
    layer.input_layernorm = Mock(side_effect=lambda x, r: (x, r))
    layer.post_attention_layernorm = Mock(side_effect=lambda x, r: (x, r))
    attention = SimpleNamespace(
        supports_sequence_parallel_input=True,
        forward_sequence_parallel=Mock(side_effect=lambda positions, x, *, output_scale: (x + 1) * output_scale),
    )
    layer.self_attn = SimpleNamespace(mla_attn=attention)
    layer.mlp = Mock(side_effect=lambda x, **kwargs: x)
    hidden = torch.ones(2, 3, dtype=torch.float32)
    residual = hidden.clone()
    positions = torch.arange(4)

    actual, actual_residual = layer(positions, hidden, residual)

    torch.testing.assert_close(actual, hidden + 1)
    assert actual_residual is residual
    attention.forward_sequence_parallel.assert_called_once_with(positions, hidden, output_scale=1.0)


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_local_fp16_preserves_upstream_overflow_scaling(layer_idx):
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = True
    layer.use_mha = False
    layer.layer_idx = layer_idx
    layer.routed_scaling_factor = 2.0
    layer.input_layernorm = Mock(side_effect=lambda x, r: (x, r))
    layer.post_attention_layernorm = Mock(side_effect=lambda x, r: (x + r, r))
    layer.self_attn = SimpleNamespace(
        mla_attn=SimpleNamespace(
            supports_sequence_parallel_input=True,
            forward_sequence_parallel=Mock(side_effect=lambda positions, x, *, output_scale: x * 4 * output_scale),
        )
    )
    layer.mlp = Mock(side_effect=lambda x, **kwargs: x * 2)
    hidden = torch.tensor([[2, 4, 6], [8, 10, 12]], dtype=torch.float16)
    residual = hidden + 2
    original_residual = residual.clone()
    expected_residual = original_residual / 2 if layer_idx == 0 else original_residual
    expected = (hidden * 2 + expected_residual) * 2

    actual, actual_residual = layer(torch.arange(4), hidden, residual)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_residual, expected_residual)
    assert actual_residual is residual
    layer.mlp.assert_called_once()
    torch.testing.assert_close(layer.mlp.call_args.args[0], hidden * 2 + expected_residual)
    assert layer.mlp.call_args.kwargs == {"already_sequence_parallel": True}


@pytest.mark.parametrize(
    "v2,pp,sp,dsa_cp,expected",
    [
        (False, 1, True, True, AscendDSACPModel),
        (False, 1, True, False, deepseek_v2.DeepseekV2Model),
        (False, 1, False, True, deepseek_v2.DeepseekV2Model),
        (True, 1, True, True, deepseek_v2.DeepseekV2Model),
        (False, 2, True, True, deepseek_v2.DeepseekV2Model),
    ],
)
def test_registered_glm_model_selects_native_v1_cp_model(monkeypatch, v2, pp, sp, dsa_cp, expected):
    selected = []

    def init(self, *, vllm_config, prefix):
        selected.append(self.model_cls)
        nn.Module.__init__(self)
        self.moe_layers = []

    monkeypatch.setattr(deepseek_v2.GlmMoeDsaForCausalLM, "__init__", init)
    monkeypatch.setattr(deepseek_mtp, "enable_dsa_cp", lambda: dsa_cp)
    config = SimpleNamespace(
        use_v2_model_runner=v2,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp, use_sequence_parallel_moe=sp),
    )
    AscendGlmMoeDsaForCausalLM(vllm_config=config, prefix="model")

    assert selected == [expected]
    assert deepseek_v2.GlmMoeDsaForCausalLM.model_cls is deepseek_v2.DeepseekV2Model


def test_native_constructor_preserves_weight_names_and_shared_index_buffer(monkeypatch):
    layer_args = []

    def init_layer(self, *, vllm_config, prefix, topk_indices_buffer, index_group_builder):
        nn.Module.__init__(self)
        self.mlp = nn.Linear(3, 3, bias=False)
        self.self_attn = SimpleNamespace(topk_indices_buffer=topk_indices_buffer)
        layer_args.append((type(self), prefix, topk_indices_buffer, index_group_builder))

    def make_layers(count, factory, prefix):
        return 0, count, nn.ModuleList(factory(f"{prefix}.{index}") for index in range(count))

    monkeypatch.setattr(deepseek_v2.DeepseekV2DecoderLayer, "__init__", init_layer)
    monkeypatch.setattr(deepseek_v2, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    for module in (deepseek_v2, deepseek_mtp):
        monkeypatch.setattr(module, "current_platform", SimpleNamespace(device_type="cpu"))
        monkeypatch.setattr(module, "VocabParallelEmbedding", lambda vocab, hidden, **kw: nn.Embedding(vocab, hidden))
        monkeypatch.setattr(module, "RMSNorm", lambda hidden, **kw: nn.Identity())
        monkeypatch.setattr(module, "make_layers", make_layers)
        monkeypatch.setattr(module, "get_sparse_mla_index_group_max_rows", lambda config: 16)
        monkeypatch.setattr(module, "SparseMLAIndexGroupBuilder", lambda *args: object())
        monkeypatch.setattr(module, "is_model_fused_shared_expert_compatible", lambda *args: True)
    config = SimpleNamespace(
        use_v2_model_runner=False,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=3,
                vocab_size=8,
                num_hidden_layers=2,
                index_topk=2,
                rms_norm_eps=1e-5,
                qk_nope_head_dim=2,
                qk_rope_head_dim=2,
                model_type="glm_moe_dsa",
            )
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        parallel_config=SimpleNamespace(eplb_config=SimpleNamespace(num_redundant_experts=0)),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )
    upstream = deepseek_v2.DeepseekV2Model(vllm_config=config, prefix="model")
    layer_args.clear()
    native = AscendDSACPModel(vllm_config=config, prefix="model")

    assert set(native.state_dict()) == set(upstream.state_dict())
    assert [tuple(t.shape) for t in native.state_dict().values()] == [
        tuple(t.shape) for t in upstream.state_dict().values()
    ]
    assert all(kind is AscendDSACPDecoderLayer for kind, *_ in layer_args)
    assert [prefix for _, prefix, *_ in layer_args] == ["model.layers.0", "model.layers.1"]
    assert all(buffer is native.topk_indices_buffer for _, _, buffer, _ in layer_args)
    assert layer_args[0][3] is layer_args[1][3]
    assert native.is_fused_shared_expert_enabled == upstream.is_fused_shared_expert_enabled
    assert native._use_upstream_aux_relay is False
    assert native.aux_hidden_state_layers == upstream.aux_hidden_state_layers
    assert native.do_not_compile is True
    assert (
        native.make_empty_intermediate_tensors(4, torch.bfloat16, "cpu").tensors.keys()
        == upstream.make_empty_intermediate_tensors(4, torch.bfloat16, "cpu").tensors.keys()
    )
