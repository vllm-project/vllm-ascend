# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
    set_eagle3_aux_hidden_state_layers,
)

from tests.deepseek_v41_utils import hc_mixes_reference, hc_post_reference
from vllm_ascend.models.deepseek_v41 import model as deepseek_v41_module
from vllm_ascend.models.deepseek_v41.model import (
    AscendDeepseekV41LLMForCausalLM,
    DeepseekV41DecoderLayer,
    DeepseekV41Model,
)
from vllm_ascend.worker.v2.pp_transport import make_empty_intermediate_tensors


class _PipelineLayer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.layer_idx = index
        self.engram = None

    def forward(self, positions, hidden, pre_mix, scaling, input_ids=None):
        del positions, scaling, input_ids
        hidden = (hidden.float() + pre_mix.unsqueeze(-1) * (self.layer_idx + 1)).to(hidden.dtype)
        return hidden, pre_mix + (self.layer_idx + 1) / 8

    def hc_collapse(self, hidden, pre_mix):
        return (hidden.float() * pre_mix.unsqueeze(-1)).sum(1).to(hidden.dtype)


def _pipeline_stage(start, end, embedding, use_sp, *, num_layers=6):
    model = object.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    # __new__ bypasses compile-decorator initialization; keep the real eager forward.
    model.do_not_compile = True
    model.config = SimpleNamespace(hidden_size=4, engram_layer_ids=())
    model.hc_mult = 4
    model.decoder_replay_layers = None
    model.has_engram = False
    model.start_layer, model.end_layer = start, end
    model.use_sequence_parallel = use_sp
    model.embed_tokens = embedding if start == 0 else PPMissingLayer()
    model.layers = torch.nn.ModuleList(
        [_PipelineLayer(i) if start <= i < end else PPMissingLayer() for i in range(num_layers)]
    )
    model.norm = torch.nn.Identity() if end == num_layers else PPMissingLayer()
    model.shared_attention_state = SimpleNamespace(reset=lambda: None)
    model._mtp_hidden_buffer = None
    model.aux_hidden_state_layers = (1, 4)
    return model


@pytest.mark.parametrize("partition", [(3, 3), (1, 2, 2, 1)])
@pytest.mark.parametrize("use_sp", [False, True])
def test_v41_pp_preserves_hc_pre_mix_and_aux_states(monkeypatch, partition, use_sp):
    """Splitting the same layers must preserve HC results and Eagle inputs."""
    embedding = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
    ids, positions = torch.full((3,), 2), torch.arange(3)
    if use_sp:
        # Equal rows let a single CPU process model two TP shards, including
        # padding of the three-token batch to an even shard size.
        monkeypatch.setattr(deepseek_v41_module, "sp_shard", lambda value: value[: (value.shape[0] + 1) // 2])
        monkeypatch.setattr(deepseek_v41_module, "sp_all_gather", lambda value: torch.cat((value, value)))
    monkeypatch.setattr(
        deepseek_v41_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)
    )
    expected, expected_aux = _pipeline_stage(0, 6, embedding, use_sp)(ids, positions, None)
    incoming: IntermediateTensors | tuple[torch.Tensor, Sequence[torch.Tensor]] | None = None
    start = 0
    for rank, count in enumerate(partition):
        first, last = rank == 0, rank == len(partition) - 1
        monkeypatch.setattr(
            deepseek_v41_module,
            "get_pp_group",
            lambda first=first, last=last: SimpleNamespace(is_first_rank=first, is_last_rank=last),
        )
        stage = _pipeline_stage(start, start + count, embedding, use_sp)
        incoming = stage(ids, positions, incoming)
        if not last:
            assert isinstance(incoming, IntermediateTensors)
            assert incoming["hidden_states"].shape == (3, 4, 4)
            assert incoming["pre_mix"].shape == (3, 4)
            assert incoming["pre_mix"].dtype == torch.float32
        start += count
    assert isinstance(incoming, tuple)
    actual, actual_aux = incoming
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(actual_aux) == len(expected_aux) == 2
    for actual_state, expected_state in zip(actual_aux, expected_aux):
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


def test_v41_pp_receive_buffer_preserves_pre_mix_precision():
    model = _pipeline_stage(3, 6, None, False)
    buffers = model._make_empty_intermediate_tensors(5, torch.bfloat16, "cpu")
    assert buffers["hidden_states"].shape == (5, 4, 4)
    assert buffers["hidden_states"].dtype == torch.bfloat16
    assert buffers["pre_mix"].shape == (5, 4)
    assert buffers["pre_mix"].dtype == torch.float32
    assert torch.isfinite(buffers["hidden_states"]).all()
    torch.testing.assert_close(buffers["pre_mix"].sum(-1), torch.ones(5))


@pytest.mark.parametrize("partition", [(20, 20), (24, 16), (8, 12, 20)])
@pytest.mark.parametrize("use_sp", [False, True])
def test_mrv2_checkpoint_aux_boundaries_survive_native_pp_receive_and_relay(monkeypatch, partition, use_sp):
    """MRV2 checkpoint IDs include PP boundary 20 and final boundary 40."""
    config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41", dspark_target_layer_ids=[1, 8, 20, 37, 40]),
        ),
    )
    assert get_eagle3_aux_layers_from_config(config) == (1, 8, 20, 37, 40)
    embedding = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
    ids, positions = torch.full((3,), 2), torch.arange(3)
    if use_sp:
        monkeypatch.setattr(deepseek_v41_module, "sp_shard", lambda value: value[: (value.shape[0] + 1) // 2])
        monkeypatch.setattr(deepseek_v41_module, "sp_all_gather", lambda value: torch.cat((value, value)))
    monkeypatch.setattr("vllm.distributed.parallel_state.model_parallel_is_initialized", lambda: True)

    def configure(start, end, group):
        monkeypatch.setattr(deepseek_v41_module, "get_pp_group", lambda: group)
        monkeypatch.setattr("vllm.distributed.parallel_state.get_pp_group", lambda: group)
        inner = _pipeline_stage(start, end, embedding, use_sp, num_layers=40)
        inner.make_empty_intermediate_tensors = make_empty_intermediate_tensors(
            inner, inner._make_empty_intermediate_tensors
        )
        model = AscendDeepseekV41LLMForCausalLM.__new__(AscendDeepseekV41LLMForCausalLM)
        torch.nn.Module.__init__(model)
        model.model = inner
        model.make_empty_intermediate_tensors = inner.make_empty_intermediate_tensors
        set_eagle3_aux_hidden_state_layers(model, config)
        assert inner.aux_hidden_state_layers == (1, 8, 20, 37, 40)
        return model

    baseline = configure(0, 40, SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True))
    expected, expected_aux = baseline.model(ids, positions, None)
    incoming: IntermediateTensors | tuple[torch.Tensor, Sequence[torch.Tensor]] | None = None
    start = 0
    for rank, count in enumerate(partition):
        first, last = rank == 0, rank == len(partition) - 1
        group = SimpleNamespace(world_size=len(partition), is_first_rank=first, is_last_rank=last)
        model = configure(start, start + count, group)
        if not first:
            assert isinstance(incoming, IntermediateTensors)
            buffers = model.make_empty_intermediate_tensors(5, torch.bfloat16, "cpu")
            assert buffers.tensors.keys() == incoming.tensors.keys()
            for name, tensor in buffers.tensors.items():
                tensor[:3].copy_(incoming[name])
            incoming = buffers[:3]
        output = model.model(ids, positions, incoming)
        if not last:
            handler = PPHandler.__new__(PPHandler)
            handler.configure_aux_hidden_state_relay(model)
            output = handler.relay_aux_hidden_states(incoming, output)
            assert output["pre_mix"].dtype == torch.float32
            expected_count = sum(layer <= start + count for layer in model.model.aux_hidden_state_layers)
            aux_keys = [key for key in output.tensors if key.startswith(model.model.AUX_HIDDEN_STATE_KEY)]
            assert len(aux_keys) == expected_count
        incoming = output
        start += count

    assert isinstance(incoming, tuple)
    actual, actual_aux = incoming
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(actual_aux) == len(expected_aux) == 5
    for actual_state, expected_state in zip(actual_aux, expected_aux):
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


def _layer() -> DeepseekV41DecoderLayer:
    layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
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
    normalized_fp32 = normalized.float()
    layer.rms_norm_cast = MagicMock(return_value=(normalized, normalized_fp32))
    layer.self_attn = MagicMock(side_effect=lambda _positions, value, _scaling: value)
    layer.mlp = MagicMock(side_effect=lambda value, **_kwargs: value)
    layer.hc_post = MagicMock(side_effect=lambda _x, residual, _post, _comb: residual)

    input_ids = torch.tensor([11, 22])
    output, next_pre = layer.forward(torch.arange(2), hidden_states, incoming_pre, input_ids=input_ids)

    assert output is hidden_states
    assert next_pre is ffn_pre
    assert layer.hc_pre.call_args_list[0].args[-1] is incoming_pre
    assert layer.hc_pre.call_args_list[1].args[-1] is attn_pre
    layer.rms_norm_cast.assert_called_once_with(collapsed)
    layer.mlp.assert_called_once_with(
        normalized,
        input_ids=input_ids,
        hidden_states_fp32=normalized_fp32,
        already_sequence_parallel=False,
    )


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
    layer.rms_norm_cast = MagicMock(return_value=(collapsed, collapsed.float()))
    layer.self_attn = MagicMock(side_effect=lambda _positions, value, _scaling: value)
    layer.mlp = MagicMock(side_effect=lambda value, **_kwargs: value)
    layer.hc_post = MagicMock(side_effect=lambda _x, residual, _post, _comb: residual)
    all_gather = MagicMock(return_value=collapsed)
    reduce_scatter = MagicMock(return_value=collapsed)
    monkeypatch.setattr(deepseek_v41_module, "sp_all_gather", all_gather)
    monkeypatch.setattr(deepseek_v41_module, "sp_reduce_scatter", reduce_scatter)

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


def test_v41_dspark_propagates_delayed_mix_and_collapses_final_stream():
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel

    model = DeepseekV41DSparkModel.__new__(DeepseekV41DSparkModel)
    torch.nn.Module.__init__(model)
    model.hc_mult = 2
    model.needs_moe_input_ids = False
    model.decoder_replay_layers = None
    model.use_sequence_parallel_moe = False
    model.embed_tokens = torch.nn.Embedding(4, 3)
    seen = []

    class Layer(torch.nn.Module):
        hc_collapse = staticmethod(DeepseekV41DecoderLayer.hc_collapse)

        def forward(self, positions, hidden, pre_mix, llama_4_scaling=None, input_ids=None):
            seen.append(pre_mix.clone())
            return hidden + 1, pre_mix.flip(-1)

    model.layers = torch.nn.ModuleDict({"40": Layer(), "41": Layer(), "42": Layer()})
    ids = torch.tensor([0, 1])
    result = model(ids, torch.tensor([2, 3]))
    torch.testing.assert_close(result, model.embed_tokens(ids) + 3)
    assert [x[0].tolist() for x in seen] == [[1, 0], [0, 1], [1, 0]]


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize("engram_offset", [0, 10])
def test_v41_target_preserves_runner_specific_aux_boundaries(use_v2_model_runner, engram_offset):
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.decoder_replay_layers = None
    model.hc_mult = 2
    model.start_layer = 0
    model.end_layer = 3
    model.decoder_replay_start = 3
    model.use_sequence_parallel = False
    model._capture_aux_after_layer = use_v2_model_runner
    model.engram_rotated = False
    model.needs_moe_input_ids = False
    model.embed_tokens = torch.nn.Embedding(4, 3)
    model.norm = torch.nn.Identity()
    model.shared_attention_state = MagicMock()
    # forward reads the MTP buffer (None = collapse locally), which __init__
    # normally allocates; the __new__-built fixture skips that.
    model._mtp_hidden_buffer = None
    model._set_aux_hidden_state_layers((1, 3))

    class Layer(torch.nn.Module):
        hc_collapse = staticmethod(DeepseekV41DecoderLayer.hc_collapse)

        def __init__(self, idx):
            super().__init__()
            self.layer_idx = idx
            self.engram = (
                (lambda hidden, lookup, mask, rotation: hidden + engram_offset) if idx == 0 and engram_offset else None
            )

        def forward(self, positions, hidden, pre_mix, scaling, input_ids=None):
            return hidden + 1, pre_mix

    model.layers = torch.nn.ModuleList([Layer(i) for i in range(3)])
    model.start_layer, model.end_layer = 0, len(model.layers)
    ids = torch.tensor([0, 1])
    with patch(
        "vllm_ascend.models.deepseek_v41.model.get_pp_group",
        return_value=MagicMock(is_first_rank=True, is_last_rank=True),
    ):
        output, aux = model.forward(
            ids,
            torch.tensor([0, 1]),
            None,
            engram_lookups={0: torch.zeros((2, 1))} if engram_offset else {},
            engram_mask=torch.ones(2, dtype=torch.bool) if engram_offset else torch.empty(0, dtype=torch.bool),
        )
    embedded = model.embed_tokens(ids)
    torch.testing.assert_close(output, embedded + engram_offset + 3)
    assert len(aux) == 2
    offset = int(use_v2_model_runner)
    torch.testing.assert_close(aux[0], embedded + offset * (engram_offset + 1))
    torch.testing.assert_close(aux[1], embedded + engram_offset + 2 + offset)


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
