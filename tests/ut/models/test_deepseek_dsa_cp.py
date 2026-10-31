# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.config import CompilationMode
from vllm.model_executor.models import deepseek_v2
from vllm.sequence import IntermediateTensors

from vllm_ascend.models import deepseek_mtp
from vllm_ascend.models.deepseek_mtp import AscendDSACPDecoderLayer, AscendDSACPModel, AscendGlmMoeDsaForCausalLM
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
from vllm_ascend.patch.worker import patch_deepseek_v2


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 14])
def test_consecutive_mixed_layers_keep_local_tokens_and_residual(monkeypatch, rank, dtype, num_tokens):
    # Global order is decode followed by prefill. Some ranks own decode only;
    # model layers must keep the same shard in either part of the mixed batch.
    local_tokens = (num_tokens + 3) // 4
    hidden = torch.arange(rank * local_tokens * 3, (rank + 1) * local_tokens * 3, dtype=dtype).reshape(local_tokens, 3)
    residual = hidden.clone()
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = True
    layer.input_is_sequence_parallel = True
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
    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = True
    moe.experts = Mock(side_effect=lambda hidden_states, router_logits: hidden_states * 2)
    layer.mlp = Mock(wraps=moe.forward)
    gather, scatter, chunk = Mock(), Mock(), Mock()
    monkeypatch.setattr(deepseek_v2, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(deepseek_mtp, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(deepseek_v2, "tensor_model_parallel_reduce_scatter", scatter)
    monkeypatch.setattr(deepseek_v2, "sequence_parallel_chunk", chunk)

    actual, actual_residual = layer(torch.arange(num_tokens), hidden, residual)
    actual, actual_residual = layer(torch.arange(num_tokens), actual, actual_residual)

    torch.testing.assert_close(actual, hidden * 4 + 6)
    assert actual.shape == hidden.shape
    assert actual_residual is residual
    assert wrapper.forward_sequence_parallel.call_count == 2
    assert all(call.args[1].shape == hidden.shape for call in wrapper.forward_sequence_parallel.call_args_list)
    assert all(call.kwargs == {"already_sequence_parallel": True} for call in layer.mlp.call_args_list)
    layer.self_attn.assert_not_called()
    gather.assert_not_called()
    scatter.assert_not_called()
    chunk.assert_not_called()


@pytest.mark.parametrize("boundary", ["replicated", "first_layer", "dense", "unsupported", "query_scaling"])
def test_unsupported_layouts_preserve_upstream_behavior(monkeypatch, boundary):
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = boundary != "dense"
    layer.input_is_sequence_parallel = boundary in {"unsupported", "query_scaling"}
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

    actual, actual_residual = layer.forward(positions, hidden, residual, scaling)
    assert actual is expected[0] and actual_residual is expected[1]
    upstream.assert_called_once_with(positions, hidden, residual, scaling)


def test_local_attention_uses_explicit_sequence_parallel_interface():
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = True
    layer.input_is_sequence_parallel = True
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
    layer.input_is_sequence_parallel = True
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
        (False, 2, True, True, AscendDSACPModel),
        (False, 3, True, True, AscendDSACPModel),
        (True, 2, True, True, deepseek_v2.DeepseekV2Model),
        (False, 2, False, True, deepseek_v2.DeepseekV2Model),
        (False, 2, True, False, deepseek_v2.DeepseekV2Model),
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


@pytest.mark.parametrize("pp_rank,pp_size", [(0, 1), (0, 3), (1, 3), (2, 3)])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("projection", ["local", "unsupported", "custom", "dense", "mixed", "query_scaling"])
def test_native_constructor_preserves_weight_names_and_shared_index_buffer(
    monkeypatch, pp_rank, pp_size, sp, projection
):
    layer_args = []

    class MoE(nn.Linear):
        pass

    def init_layer(self, *, vllm_config, prefix, topk_indices_buffer, index_group_builder):
        nn.Module.__init__(self)
        dense = projection == "dense" or (projection == "mixed" and int(prefix.rsplit(".", 1)[1]) % 2 == 1)
        self.mlp = (nn.Linear if dense else MoE)(3, 3, bias=False)
        self.mlp.is_sequence_parallel = sp
        self.use_sequence_parallel_moe = pp_size == 1 and sp and not dense
        o_proj = SimpleNamespace(
            reduce_results=not self.use_sequence_parallel_moe,
            custom_op=object() if projection == "custom" else None,
        )
        wrapper = AscendMultiHeadLatentAttention.__new__(AscendMultiHeadLatentAttention)
        nn.Module.__init__(wrapper)
        wrapper.output_token_shard_size = 1
        wrapper.mla_attn = SimpleNamespace(
            impl=SimpleNamespace(supports_sequence_parallel_input=projection != "unsupported", o_proj=o_proj)
        )
        self.self_attn = SimpleNamespace(topk_indices_buffer=topk_indices_buffer, o_proj=o_proj, mla_attn=wrapper)
        layer_args.append((type(self), prefix, topk_indices_buffer, index_group_builder))

    def make_layers(count, factory, prefix):
        start, end = pp_rank * count // pp_size, (pp_rank + 1) * count // pp_size
        return (
            start,
            end,
            nn.ModuleList(
                factory(f"{prefix}.{index}") if start <= index < end else deepseek_mtp.PPMissingLayer()
                for index in range(count)
            ),
        )

    monkeypatch.setattr(deepseek_v2.DeepseekV2DecoderLayer, "__init__", init_layer)
    monkeypatch.setattr(deepseek_mtp, "DeepseekV2MoE", MoE)
    for module in (deepseek_v2, deepseek_mtp):
        monkeypatch.setattr(
            module,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=pp_rank == 0, is_last_rank=pp_rank == pp_size - 1),
        )
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
                num_hidden_layers=6,
                index_topk=2,
                rms_norm_eps=1e-5,
                qk_nope_head_dim=2,
                qk_rope_head_dim=2,
                model_type="glm_moe_dsa",
                llama_4_scaling=(
                    {"original_max_position_embeddings": 8192, "beta": 1.0} if projection == "query_scaling" else None
                ),
            )
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            use_sequence_parallel_moe=sp,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )
    config.model_config.hf_text_config = config.model_config.hf_config
    upstream = deepseek_v2.DeepseekV2Model(vllm_config=config, prefix="model")
    layer_args.clear()
    native = AscendDSACPModel(vllm_config=config, prefix="model")

    # Each PP stage must retain upstream checkpoint keys and index-buffer ownership.
    assert set(native.state_dict()) == set(upstream.state_dict())
    assert [tuple(t.shape) for t in native.state_dict().values()] == [
        tuple(t.shape) for t in upstream.state_dict().values()
    ]
    assert all(kind is AscendDSACPDecoderLayer for kind, *_ in layer_args)
    assert [prefix for _, prefix, *_ in layer_args] == [
        f"model.layers.{index}" for index in range(native.start_layer, native.end_layer)
    ]
    assert isinstance(native.embed_tokens, deepseek_mtp.PPMissingLayer) is (pp_rank != 0)
    assert isinstance(native.norm, deepseek_mtp.PPMissingLayer) is (pp_rank != pp_size - 1)
    previous_sp = False
    for index, layer in enumerate(native.layers[native.start_layer : native.end_layer], native.start_layer):
        dense = projection == "dense" or (projection == "mixed" and index % 2 == 1)
        expected_sp = sp and not dense and projection in {"local", "mixed"}
        assert layer.use_sequence_parallel_moe is expected_sp
        assert layer.self_attn.o_proj.reduce_results is (not expected_sp)
        assert layer.input_is_sequence_parallel is (previous_sp and expected_sp)
        assert layer.gather_sequence_parallel_output is (
            expected_sp and (index + 1 == native.end_layer or projection == "mixed")
        )
        previous_sp = expected_sp
    assert config.parallel_config.pipeline_parallel_size == pp_size
    assert config.parallel_config.use_sequence_parallel_moe is sp
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


@pytest.mark.parametrize("num_tokens", [1, 3, 4])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("pp_rank,pp_size", [(0, 1), (0, 2), (1, 2), (1, 3)])
@pytest.mark.parametrize(
    "layout", [(True, True, False), (True, False, True), (True, True, True), (False, False, False)]
)
def test_model_forward_gathers_each_sp_segment_once(monkeypatch, num_tokens, rank, pp_rank, pp_size, layout):
    local_tokens = (num_tokens + 1) // 2
    full_hidden = torch.arange(num_tokens * 3, dtype=torch.float32).reshape(num_tokens, 3)
    full_residual = full_hidden if pp_rank == 0 else full_hidden + 20
    shard = slice(rank * local_tokens, (rank + 1) * local_tokens)

    def padded(states):
        return torch.nn.functional.pad(states, (0, 0, 0, 2 * local_tokens - num_tokens))

    # Fake only operators/collectives; retain the model and both decoder forwards.
    local_layers: list[AscendDSACPDecoderLayer] = []
    for sp in layout:
        layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
        nn.Module.__init__(layer)
        layer.use_sequence_parallel_moe = False
        layer.use_mha = False
        layer.routed_scaling_factor = 1.0
        layer.layer_idx = pp_rank * len(layout) + len(local_layers)
        layer.input_layernorm = Mock(side_effect=lambda x, r=None: x if r is None else (x, r))
        layer.post_attention_layernorm = Mock(side_effect=lambda x, r: (x, r))
        o_proj = SimpleNamespace(reduce_results=True, custom_op=None)
        wrapper = AscendMultiHeadLatentAttention.__new__(AscendMultiHeadLatentAttention)
        nn.Module.__init__(wrapper)
        wrapper.output_token_shard_size = 1
        wrapper.mla_attn = SimpleNamespace(impl=SimpleNamespace(supports_sequence_parallel_input=True, o_proj=o_proj))
        wrapper.forward_sequence_parallel = Mock(side_effect=lambda positions, x, **kw: x)
        layer.self_attn = Mock(side_effect=lambda positions, x, scaling: x)
        layer.self_attn.o_proj = o_proj
        layer.self_attn.mla_attn = wrapper
        layer.mlp = nn.Identity()
        if sp:
            layer.mlp = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
            nn.Module.__init__(layer.mlp)
            layer.mlp.is_sequence_parallel = True
            layer.mlp.experts = Mock(side_effect=lambda *, hidden_states, router_logits: hidden_states + 1)
        local_layers.append(layer)

    start, end = pp_rank * len(layout), (pp_rank + 1) * len(layout)
    layers = nn.ModuleList(
        local_layers[index - start] if start <= index < end else deepseek_mtp.PPMissingLayer()
        for index in range(pp_size * len(layout))
    )
    monkeypatch.setattr(deepseek_mtp, "make_layers", lambda *a, **kw: (start, end, layers))
    monkeypatch.setattr(deepseek_mtp, "VocabParallelEmbedding", lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(deepseek_mtp, "RMSNorm", lambda *a, **kw: Mock(side_effect=lambda x, r: (x, r)))
    monkeypatch.setattr(deepseek_mtp, "is_model_fused_shared_expert_compatible", lambda *a: False)
    pp_group = SimpleNamespace(is_first_rank=pp_rank == 0, is_last_rank=pp_rank == pp_size - 1)
    for module in (deepseek_mtp, deepseek_v2):
        monkeypatch.setattr(module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(deepseek_v2, "sequence_parallel_chunk", lambda x: padded(x)[shard])
    monkeypatch.setattr(deepseek_v2, "tensor_model_parallel_reduce_scatter", lambda x, dim: x[shard])
    config = SimpleNamespace(
        use_v2_model_runner=False,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=3,
                vocab_size=8,
                num_hidden_layers=pp_size * len(layout),
                rms_norm_eps=1e-5,
                model_type="glm_moe_dsa",
            )
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size, eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )
    model = AscendDSACPModel(vllm_config=config, prefix="model")
    assert model.forward.__func__ is patch_deepseek_v2._patched_forward
    model.aux_hidden_state_layers = (end,)
    monkeypatch.setitem(model.forward.__globals__, "get_pp_group", lambda: pp_group)
    segment_ends = [i for i, sp in enumerate(layout) if sp and (i + 1 == len(layout) or not layout[i + 1])]

    def gather_states(states, dim):
        assert dim == 0
        assert gather.call_count <= len(segment_ends), "Repeated gather at the same SP boundary"
        hidden = padded(full_hidden + sum(layout[: segment_ends[gather.call_count - 1] + 1]))
        combined = torch.cat([hidden, padded(full_residual)], dim=-1)
        assert states.shape == (local_tokens, 6)
        valid_rows = max(0, min(local_tokens, num_tokens - rank * local_tokens))
        torch.testing.assert_close(states[:valid_rows], combined[shard][:valid_rows])
        return combined

    gather = Mock(side_effect=gather_states)
    model_gather = Mock(wraps=gather)
    monkeypatch.setattr(deepseek_mtp, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(deepseek_v2, "tensor_model_parallel_all_gather", model_gather)
    monkeypatch.setitem(model.forward.__globals__, "tensor_model_parallel_all_gather", model_gather)
    result = model.forward(
        input_ids=None,
        positions=torch.arange(num_tokens),
        intermediate_tensors=IntermediateTensors({"hidden_states": full_hidden, "residual": full_residual}),
        inputs_embeds=full_hidden if pp_rank == 0 else None,
    )

    if not pp_group.is_last_rank:
        hidden, residual = result["hidden_states"], result["residual"]
        aux_hidden_states = deepseek_mtp.pp_transport.get_pp_transport_tensors(
            result, deepseek_mtp.pp_transport.PPTransportDataType.AUX_HIDDEN_STATES
        )
    else:
        hidden, aux_hidden_states = result
        residual = model.norm.call_args.args[1]
    torch.testing.assert_close(hidden, full_hidden + sum(layout))
    torch.testing.assert_close(residual, full_residual)
    assert len(aux_hidden_states) == 1
    torch.testing.assert_close(aux_hidden_states[0], full_hidden + sum(layout) + full_residual)
    assert gather.call_count == len(segment_ends)
    model_gather.assert_not_called()
    for index, layer in enumerate(local_layers):
        if not layout[index]:
            torch.testing.assert_close(layer.self_attn.call_args.args[1], full_hidden + sum(layout[:index]))
            torch.testing.assert_close(layer.post_attention_layernorm.call_args.args[1], full_residual)


@pytest.mark.parametrize("num_tokens", [1, 3, 4])
@pytest.mark.parametrize("rank", [0, 1])
def test_sp_segment_end_restores_full_states_even_when_local_rows_equal_global(monkeypatch, num_tokens, rank):
    local_tokens = (num_tokens + 1) // 2
    full_hidden = torch.arange(local_tokens * 6, dtype=torch.float32).reshape(local_tokens * 2, 3)
    full_residual = full_hidden + 20
    shard = slice(rank * local_tokens, (rank + 1) * local_tokens)
    local_hidden, local_residual = full_hidden[shard], full_residual[shard]
    layer = AscendDSACPDecoderLayer.__new__(AscendDSACPDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel_moe = True
    layer.gather_sequence_parallel_output = True
    parent = Mock(return_value=(local_hidden, local_residual))
    gather = Mock(return_value=torch.cat([full_hidden, full_residual], dim=-1))
    monkeypatch.setattr(deepseek_v2.DeepseekV2DecoderLayer, "forward", parent)
    monkeypatch.setattr(deepseek_mtp, "tensor_model_parallel_all_gather", gather)
    positions = torch.arange(num_tokens)

    hidden, residual = layer(positions, local_hidden, local_residual)

    torch.testing.assert_close(hidden, full_hidden[:num_tokens])
    torch.testing.assert_close(residual, full_residual[:num_tokens])
    assert hidden.is_contiguous() and residual.is_contiguous()
    parent.assert_called_once_with(positions, local_hidden, local_residual, None)
    gather.assert_called_once()
    torch.testing.assert_close(gather.call_args.args[0], torch.cat([full_hidden[shard], full_residual[shard]], dim=-1))
    assert gather.call_args.args[1] == 0
