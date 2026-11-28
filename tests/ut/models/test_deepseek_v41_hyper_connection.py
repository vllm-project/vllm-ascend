# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch
from vllm.config import CUDAGraphMode
from vllm.model_executor.models.utils import PPMissingLayer

from tests.deepseek_v41_utils import hc_mixes_reference, hc_post_reference
from vllm_ascend.models.deepseek_v41 import model as deepseek_v41_module
from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer, _PendingHcPost


def _layer(layer_cls=DeepseekV41DecoderLayer) -> DeepseekV41DecoderLayer:
    layer = layer_cls.__new__(layer_cls)
    torch.nn.Module.__init__(layer)
    layer.hc_mult = 4
    layer.hc_sinkhorn_iters = 3
    layer.norm_eps = 1e-6
    layer.hc_eps = 1e-6
    return layer


def test_v41_hc_pre_dispatches_fused_operator_with_pre_mix():
    layer = _layer()
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, 32, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)
    pre_mix = torch.randn(2, 4, dtype=torch.float32)
    expected = (
        torch.randn(2, 8, dtype=torch.bfloat16),
        torch.randn(2, 4),
        torch.randn(2, 4, 4),
        torch.randn(2, 4),
    )

    with patch.object(
        torch.ops._C_ascend,
        "npu_hc_pre_v3",
        create=True,
        return_value=expected,
    ) as op:
        actual = layer.hc_pre(x, hc_fn, hc_scale, hc_base, pre_mix)

    assert actual is expected
    op.assert_called_once_with(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_mix,
        hc_mult=4,
        hc_sinkhorn_iters=3,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )


def test_v41_forward_threads_pre_mix_through_fused_hc_pre():
    layer = _layer()
    hidden_states = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    incoming_pre = torch.randn(2, 4, dtype=torch.float32)
    attn_pre = torch.randn(2, 4, dtype=torch.float32)
    ffn_pre = torch.randn(2, 4, dtype=torch.float32)
    post = torch.randn(2, 4, dtype=torch.float32)
    comb = torch.randn(2, 4, 4, dtype=torch.float32)
    collapsed = torch.randn(2, 8, dtype=torch.bfloat16)
    layer.hc_attn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer.hc_attn_scale = torch.nn.Parameter(torch.empty(3))
    layer.hc_attn_base = torch.nn.Parameter(torch.empty(24))
    layer.hc_ffn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer.hc_ffn_scale = torch.nn.Parameter(torch.empty(3))
    layer.hc_ffn_base = torch.nn.Parameter(torch.empty(24))
    layer.hc_pre = MagicMock(
        side_effect=[
            (collapsed, post, comb, attn_pre),
            (collapsed, post, comb, ffn_pre),
        ]
    )
    layer.input_layernorm = MagicMock(side_effect=lambda value: value)
    normalized = torch.randn_like(collapsed)
    layer.post_attention_layernorm = MagicMock(return_value=normalized)
    layer.self_attn = MagicMock(side_effect=lambda _positions, value, _scaling: value)
    layer.mlp = MagicMock(side_effect=lambda value, **_kwargs: value)
    layer.hc_post = MagicMock(side_effect=lambda _x, residual, _post, _comb: residual)

    input_ids = torch.tensor([11, 22])
    with patch.object(deepseek_v41_module.DeviceOperator, "rms_norm_cast", return_value=None):
        output, next_pre = layer.forward(torch.arange(2), hidden_states, incoming_pre, input_ids=input_ids)

    assert output is hidden_states
    assert next_pre is ffn_pre
    assert layer.hc_pre.call_args_list[0].args[-1] is incoming_pre
    assert layer.hc_pre.call_args_list[1].args[-1] is attn_pre
    layer.post_attention_layernorm.assert_called_once_with(collapsed)
    layer.mlp.assert_called_once()
    assert layer.mlp.call_args.args[0] is normalized
    assert layer.mlp.call_args.kwargs["input_ids"] is input_ids
    assert layer.mlp.call_args.kwargs["already_sequence_parallel"] is False
    torch.testing.assert_close(layer.mlp.call_args.kwargs["hidden_states_fp32"], normalized.float())


def test_v41_forward_gathers_attention_and_keeps_moe_sharded(monkeypatch):
    layer = _layer()
    layer.use_sequence_parallel = True
    hidden_states = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    collapsed = torch.randn(2, 8, dtype=torch.bfloat16)
    post = torch.randn(2, 4, dtype=torch.float32)
    comb = torch.randn(2, 4, 4, dtype=torch.float32)
    pre = torch.randn(2, 4, dtype=torch.float32)
    layer.hc_attn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer.hc_attn_scale = torch.nn.Parameter(torch.empty(3))
    layer.hc_attn_base = torch.nn.Parameter(torch.empty(24))
    layer.hc_ffn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer.hc_ffn_scale = torch.nn.Parameter(torch.empty(3))
    layer.hc_ffn_base = torch.nn.Parameter(torch.empty(24))
    layer.hc_pre = MagicMock(side_effect=[(collapsed, post, comb, pre)] * 2)
    layer.input_layernorm = MagicMock(side_effect=lambda value: value)
    layer.post_attention_layernorm = MagicMock(return_value=collapsed)
    layer.self_attn = MagicMock(side_effect=lambda _positions, value, _scaling: value)
    layer.mlp = MagicMock(side_effect=lambda value, **_kwargs: value)
    layer.hc_post = MagicMock(side_effect=lambda _x, residual, _post, _comb: residual)
    all_gather = MagicMock(return_value=collapsed)
    reduce_scatter = MagicMock(return_value=collapsed)
    monkeypatch.setattr(deepseek_v41_module, "sp_all_gather", all_gather)
    monkeypatch.setattr(deepseek_v41_module, "sp_reduce_scatter", reduce_scatter)

    with patch.object(deepseek_v41_module.DeviceOperator, "rms_norm_cast", return_value=None):
        layer.forward(torch.arange(2), hidden_states, pre, input_ids=torch.tensor([1, 2]))

    all_gather.assert_called_once_with(collapsed)
    reduce_scatter.assert_called_once()
    torch.testing.assert_close(reduce_scatter.call_args.args[0], collapsed)
    assert layer.mlp.call_args.kwargs["already_sequence_parallel"] is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v41_rms_norm_cast_preserves_rounded_routing_input(dtype):
    layer = _layer()
    x = torch.randn(2, 8, dtype=dtype)
    normalized = torch.randn_like(x)
    normalized_fp32 = normalized.float()
    norm = MagicMock(return_value=normalized)
    norm.weight = torch.ones(8, dtype=dtype)
    norm.variance_epsilon = 1e-6
    layer.post_attention_layernorm = norm

    with (
        patch("vllm_ascend.utils.enable_custom_op", return_value=True),
        patch("vllm_ascend.device.device_op.get_current_hardware_profile") as profile,
        patch.object(
            torch.ops._C_ascend,
            "npu_rms_norm_cast",
            create=True,
            return_value=(normalized, normalized_fp32),
        ) as op,
    ):
        profile.return_value.supports.return_value = True
        actual, actual_fp32 = layer.rms_norm_cast(x)

    assert actual is normalized
    torch.testing.assert_close(actual_fp32, normalized.float(), rtol=0, atol=0)
    op.assert_called_once_with(x, norm.weight, norm.variance_epsilon)
    assert actual_fp32 is normalized_fp32
    norm.assert_not_called()


@pytest.mark.parametrize("use_sequence_parallel", [False, True])
@pytest.mark.parametrize("draft_layer", [False, True])
def test_v41_forward_uses_fused_attention_to_ffn_boundary(monkeypatch, use_sequence_parallel, draft_layer):
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkDecoderLayer

    layer = _layer(DeepseekV41DSparkDecoderLayer if draft_layer else DeepseekV41DecoderLayer)
    layer.use_hc_post_pre_norm = True
    layer.use_sequence_parallel = use_sequence_parallel
    hidden_size = 5120
    residual = torch.empty(2, 4, hidden_size, dtype=torch.bfloat16)
    hidden = torch.empty(2, hidden_size, dtype=torch.bfloat16)
    attention_output = torch.empty_like(hidden)
    sharded_output = torch.empty_like(hidden)
    incoming_pre, attn_pre, attn_post, ffn_pre, ffn_post = [torch.empty(2, 4) for _ in range(5)]
    attn_comb, ffn_comb = torch.empty(2, 4, 4), torch.empty(2, 4, 4)
    new_residual = torch.empty_like(residual)
    y = torch.empty_like(hidden)
    y_fp32 = torch.empty_like(y, dtype=torch.float32)
    moe_output = torch.empty_like(y)
    layer.hc_attn_fn = torch.empty(24, 4 * hidden_size)
    layer.hc_attn_scale = torch.empty(3)
    layer.hc_attn_base = torch.empty(24)
    layer.hc_ffn_fn = torch.empty_like(layer.hc_attn_fn)
    layer.hc_ffn_scale = torch.empty_like(layer.hc_attn_scale)
    layer.hc_ffn_base = torch.empty_like(layer.hc_attn_base)
    layer.hc_pre = MagicMock(return_value=(hidden, attn_post, attn_comb, attn_pre))
    layer.input_layernorm = MagicMock(return_value=hidden)
    layer.post_attention_layernorm = MagicMock(
        weight=torch.empty(hidden_size, dtype=torch.bfloat16), variance_epsilon=1e-20
    )
    layer.self_attn = MagicMock(return_value=attention_output)
    layer.mlp = MagicMock(return_value=moe_output)
    layer.hc_post = MagicMock(return_value=new_residual)
    layer.rms_norm_cast = MagicMock()
    monkeypatch.setattr(deepseek_v41_module, "sp_all_gather", lambda x: x)
    reduce_scatter = MagicMock(return_value=sharded_output)
    monkeypatch.setattr(deepseek_v41_module, "sp_reduce_scatter", reduce_scatter)

    with patch.object(
        torch.ops._C_ascend,
        "npu_hc_post_pre_norm",
        create=True,
        return_value=(y, y_fp32, new_residual, ffn_post, ffn_comb, ffn_pre),
    ) as fused:
        output, next_pre = layer.forward(torch.arange(2), residual, incoming_pre)

    assert output is new_residual
    assert next_pre is ffn_pre
    assert layer.hc_pre.call_count == 1
    if use_sequence_parallel:
        reduce_scatter.assert_called_once_with(attention_output)
    else:
        reduce_scatter.assert_not_called()
    fused.assert_called_once_with(
        sharded_output if use_sequence_parallel else attention_output,
        residual,
        attn_post,
        attn_comb,
        attn_pre,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        layer.post_attention_layernorm.weight,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        hc_norm_eps=1e-6,
        ffn_norm_eps=1e-20,
    )
    layer.rms_norm_cast.assert_not_called()
    layer.mlp.assert_called_once_with(
        y, input_ids=None, hidden_states_fp32=y_fp32, already_sequence_parallel=use_sequence_parallel
    )
    layer.hc_post.assert_called_once_with(moe_output, new_residual, ffn_post, ffn_comb)


@pytest.mark.parametrize("num_tokens", [36, 37])
@pytest.mark.parametrize(
    ("route", "limited"),
    [
        ("plain", False),
        ("draft", True),
        ("eager_verify", True),
        ("eager_prefill", False),
        ("eager_decode", False),
        ("full_graph", True),
        ("piecewise_graph", True),
    ],
)
def test_v41_dspark_cross_layer_token_limit(monkeypatch, num_tokens, route, limited):
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkDecoderLayer

    layer = _layer(DeepseekV41DSparkDecoderLayer if route == "draft" else DeepseekV41DecoderLayer)
    layer._uses_dspark = route != "plain"
    layer.self_attn = SimpleNamespace(v41_impl=SimpleNamespace(swa_prefix="swa"))
    num_decodes, num_decode_tokens = {
        "eager_verify": (6, 36),
        "eager_decode": (36, 36),
    }.get(route, (0, 0))
    graph_mode = {
        "full_graph": CUDAGraphMode.FULL,
        "piecewise_graph": CUDAGraphMode.PIECEWISE,
    }.get(route, CUDAGraphMode.NONE)
    context = SimpleNamespace(
        cudagraph_runtime_mode=graph_mode,
        attn_metadata={"swa": SimpleNamespace(num_decodes=num_decodes, num_decode_tokens=num_decode_tokens)},
    )
    monkeypatch.setattr(deepseek_v41_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(deepseek_v41_module, "get_forward_context", lambda: context)

    x = torch.zeros(num_tokens, 5120, dtype=torch.bfloat16)
    residual = x.unsqueeze(1).repeat(1, 4, 1)
    post = pre_mix = torch.zeros(num_tokens, 4)
    comb = torch.zeros(num_tokens, 4, 4)
    pending = _PendingHcPost(x, residual, post, comb)
    layer.hc_attn_fn = torch.empty(24, 4 * 5120)
    layer.hc_attn_scale = torch.empty(3)
    layer.hc_attn_base = torch.empty(24)
    layer.hc_post = MagicMock(return_value=residual)
    layer.hc_pre = MagicMock(return_value=(x, post, comb, pre_mix))
    layer.input_layernorm = MagicMock(
        return_value=x, weight=torch.ones(5120, dtype=torch.bfloat16), variance_epsilon=1e-6, bias_loaded=False
    )
    with patch.object(
        torch.ops._C_ascend,
        "npu_hc_post_pre_norm",
        create=True,
        return_value=(x, None, residual, post, comb, pre_mix),
    ) as fused:
        assert layer._prepare_attention(pending, pre_mix) == (x, residual, post, comb, pre_mix)

    if limited and num_tokens > 36:
        fused.assert_not_called()
        layer.hc_post.assert_called_once_with(x, residual, post, comb)
        layer.hc_pre.assert_called_once_with(
            residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, pre_mix
        )
        layer.input_layernorm.assert_called_once_with(x)
    else:
        fused.assert_called_once()
        assert fused.call_args.kwargs["return_fp32"] is False
        layer.hc_post.assert_not_called()
        layer.hc_pre.assert_not_called()
        layer.input_layernorm.assert_not_called()


def test_v41_hc_reference_supports_hidden_size_5120():
    torch.manual_seed(7)
    layer = _layer()
    x = torch.randn(2, 4, 5120, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, 4 * 5120, dtype=torch.float32) / 5120
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)

    pre, post, comb = hc_mixes_reference(layer, x, hc_fn, hc_scale, hc_base)
    y = layer.hc_collapse(x, pre)

    assert y.shape == (2, 5120)
    assert y.dtype == torch.bfloat16
    assert post.shape == (2, 4)
    assert post.dtype == torch.float32
    assert comb.shape == (2, 4, 4)
    assert comb.dtype == torch.float32
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 4), atol=2e-5, rtol=2e-5)

    restored = hc_post_reference(y, x, post, comb)
    assert restored.shape == x.shape
    assert restored.dtype == x.dtype


def test_v41_hc_post_matches_reference_equation():
    torch.manual_seed(11)
    x = torch.randn(3, 5, dtype=torch.bfloat16)
    residual = torch.randn(3, 4, 5, dtype=torch.bfloat16)
    post = torch.randn(3, 4, dtype=torch.float32)
    comb = torch.randn(3, 4, 4, dtype=torch.float32)

    actual = hc_post_reference(x, residual, post, comb)
    expected = (post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)).to(
        x.dtype
    )
    torch.testing.assert_close(actual, expected)


def test_v41_hc_post_dispatches_fused_operator_with_batch_dimension():
    layer = _layer()
    x = torch.randn(3, 5, dtype=torch.bfloat16)
    residual = torch.randn(3, 4, 5, dtype=torch.bfloat16)
    post = torch.randn(3, 4, dtype=torch.float32)
    comb = torch.randn(3, 4, 4, dtype=torch.float32)
    expected = torch.randn_like(residual).unsqueeze(0)

    with patch.object(
        torch.ops._C_ascend,
        "npu_hc_post",
        create=True,
        return_value=expected,
    ) as op:
        actual = layer.hc_post(x, residual, post, comb)

    torch.testing.assert_close(actual, expected.squeeze(0))
    op.assert_called_once()
    for actual_arg, expected_arg in zip(
        op.call_args.args,
        (
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ),
    ):
        torch.testing.assert_close(actual_arg, expected_arg)


@pytest.mark.parametrize("cross_layer", [False, True])
@pytest.mark.parametrize("num_layers", [1, 3])
def test_v41_dspark_propagates_delayed_mix_and_collapses_final_stream(cross_layer, num_layers):
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel

    model = DeepseekV41DSparkModel.__new__(DeepseekV41DSparkModel)
    torch.nn.Module.__init__(model)
    model.hc_mult = 2
    model.needs_moe_input_ids = False
    model.decoder_replay_layers = None
    model.use_sequence_parallel_moe = False
    model.use_cross_layer_hc_post_pre_norm = cross_layer
    model.embed_tokens = torch.nn.Embedding(4, 3)
    seen = []

    class Layer(torch.nn.Module):
        hc_collapse = staticmethod(DeepseekV41DecoderLayer.hc_collapse)

        def forward(self, positions, hidden, pre_mix, llama_4_scaling=None, input_ids=None, *, defer_ffn_post=False):
            pending = isinstance(hidden, _PendingHcPost)
            seen.append((pre_mix[0].tolist(), pending, defer_ffn_post))
            if pending:
                hidden = hidden.residual + hidden.output
            if defer_ffn_post:
                return _PendingHcPost(torch.ones_like(hidden), hidden, pre_mix, pre_mix), pre_mix.flip(-1)
            return hidden + 1, pre_mix.flip(-1)

    model.layers = torch.nn.ModuleDict({str(40 + i): Layer() for i in range(num_layers)})
    ids = torch.tensor([0, 1])
    result = model(ids, torch.tensor([2, 3]))
    torch.testing.assert_close(result, model.embed_tokens(ids) + num_layers)
    assert seen == [
        ([1, 0] if i % 2 == 0 else [0, 1], cross_layer and i > 0, cross_layer and i + 1 < num_layers)
        for i in range(num_layers)
    ]


def test_v41_cross_layer_fusion_uses_global_indices_on_later_pp_stage():
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.start_layer = 2
    model.end_layer = 4
    model.use_cross_layer_hc_post_pre_norm = True
    model.aux_hidden_state_layers = ()
    model.decoder_replay_layers = None
    next_layer = _layer()
    next_layer.engram = None
    model.layers = torch.nn.ModuleList(
        [PPMissingLayer(), PPMissingLayer(), _layer(), next_layer, PPMissingLayer(), PPMissingLayer()]
    )

    assert model._can_defer_ffn_post(3)
    assert not model._can_defer_ffn_post(1)
    assert not model._can_defer_ffn_post(4)

    next_layer.engram = torch.nn.Identity()
    assert not model._can_defer_ffn_post(3)


def test_v41_target_emits_input_residual_for_selected_aux_layers():
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.decoder_replay_layers = None
    model.hc_mult = 2
    model.start_layer = 0
    model.end_layer = 3
    model.decoder_replay_start = 3
    model.decoder_replay_layers = None
    model.use_sequence_parallel = False
    model.needs_moe_input_ids = False
    model.embed_tokens = torch.nn.Embedding(4, 3)
    model.norm = torch.nn.Identity()
    model.shared_attention_state = MagicMock()
    # forward reads the MTP buffer (None = collapse locally), which __init__
    # normally allocates; the __new__-built fixture skips that.
    model._mtp_hidden_buffer = None
    model._set_aux_hidden_state_layers((0, 2))

    class Layer(torch.nn.Module):
        hc_collapse = staticmethod(DeepseekV41DecoderLayer.hc_collapse)

        def __init__(self, idx):
            super().__init__()
            self.layer_idx = idx
            self.engram = None

        def forward(self, positions, hidden, pre_mix, scaling, input_ids=None):
            return hidden + 1, pre_mix

    model.layers = torch.nn.ModuleList([Layer(i) for i in range(3)])
    ids = torch.tensor([0, 1])
    with patch(
        "vllm_ascend.models.deepseek_v41.model.get_pp_group",
        return_value=MagicMock(is_first_rank=True, is_last_rank=True),
    ):
        output, aux = model.forward(
            ids,
            torch.tensor([0, 1]),
            None,
            engram_lookups={},
            engram_mask=torch.empty(0, dtype=torch.bool),
        )
    embedded = model.embed_tokens(ids)
    torch.testing.assert_close(output, embedded + 3)
    assert len(aux) == 2
    torch.testing.assert_close(aux[0], embedded)
    torch.testing.assert_close(aux[1], embedded + 2)


def test_v41_dspark_decoder_uses_draft_experts_instead_of_target_config():
    from contextlib import ExitStack
    from types import SimpleNamespace

    import vllm_ascend.models.deepseek_v41.dspark as shared
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel

    draft = SimpleNamespace(
        hc_mult=4,
        hidden_size=8,
        head_dim=4,
        dspark_block_size=5,
        dspark_markov_rank=4,
        num_nextn_predict_layers=3,
        dspark_target_layer_ids=[37, 38, 39],
        num_hidden_layers=40,
        vocab_size=16,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        n_routed_experts=128,
        num_experts_per_tok=3,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(n_routed_experts=384)),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_text_config=draft)),
        quant_config=None,
    )

    def make_layer(*args, **kwargs):
        layer = torch.nn.Module()
        layer.mlp = SimpleNamespace(gate=SimpleNamespace(tid2eid=None, bias_vl=None))
        layer.self_attn = SimpleNamespace(dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=object())))
        return layer

    factory = MagicMock(side_effect=make_layer)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                shared,
                "get_ascend_config",
                return_value=SimpleNamespace(ascend_compilation_config=SimpleNamespace(fuse_mhc=False)),
            )
        )
        for name in (
            "VocabParallelEmbedding",
            "ColumnParallelLinear",
            "MergedColumnParallelLinear",
            "RMSNorm",
            "DSparkMarkovHead",
            "DSparkConfidenceHead",
        ):
            stack.enter_context(patch.object(shared, name, side_effect=lambda *args, **kwargs: torch.nn.Identity()))
        stack.enter_context(patch.object(shared, "DeepseekV41DSparkDecoderLayer", factory))
        model = DeepseekV41DSparkModel(vllm_config=config)
    assert len(model.layers) == factory.call_count == 3
    for call in factory.call_args_list:
        assert call.kwargs["config"] is draft
        assert call.kwargs["config"].n_routed_experts == 128
        assert call.kwargs["is_draft_layer"]
    assert config.model_config.hf_config.n_routed_experts == 384
