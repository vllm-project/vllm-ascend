# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import StageMissingLayer

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForCausalLM,
    AscendKimiK3ForConditionalGeneration,
    KimiK3MLP,
    KimiK3MoE,
    KimiK3MultiModalProjector,
    KimiK3TextModel,
    KimiK3VisionEncoderLayer,
    _move_module_to_device,
    _resolve_packed_expert_weight_name,
    _routed_latent_quant_config,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.ops.activation import AscendSituAndMul, SituActivationConfig
from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    KimiK3Config,
    KimiK3VisionConfig,
)


def test_kimi_k3_model_declares_checkpoint_packing_contract():
    assert AscendKimiK3ForCausalLM.packed_modules_mapping["fused_qkv"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    assert AscendKimiK3ForCausalLM.packed_modules_mapping["experts"] == [
        "experts.0.w1",
        "experts.0.w3",
        "experts.0.w2",
    ]


def test_kimi_k3_loads_qkv_checkpoint_shards_into_fused_linear():
    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=0)
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = nn.Module()
    model.layers[0].self_attn.fused_qkv = nn.Module()

    fused_weight = nn.Parameter(torch.empty(1))
    fused_weight.weight_loader = MagicMock()
    model.layers[0].self_attn.fused_qkv.register_parameter("weight", fused_weight)
    weights = [(f"layers.0.self_attn.{name}.weight", torch.empty(1)) for name in ("q_proj", "k_proj", "v_proj")]

    with (
        patch("vllm_ascend.models.kimi_k3.get_spec_layer_idx_from_weight_name", return_value=None),
        patch("vllm_ascend.models.kimi_k3.fused_moe_make_expert_params_mapping", return_value=[]),
        patch("vllm_ascend.models.kimi_k3.is_pp_missing_parameter", return_value=False),
    ):
        loaded = model.load_weights(weights)

    assert [call.args[2] for call in fused_weight.weight_loader.call_args_list] == ["q", "k", "v"]
    assert loaded == {"layers.0.self_attn.fused_qkv.weight"}


@pytest.mark.parametrize(
    ("quant_name", "uses_quantized_latent_projections"),
    [
        ("ascend", True),
        ("compressed-tensors", False),
        ("other", False),
    ],
)
def test_kimi_k3_quantizes_latent_projections_only_for_modelslim(
    quant_name: str,
    uses_quantized_latent_projections: bool,
):
    quant_config = MagicMock()
    quant_config.get_name.return_value = quant_name

    actual = _routed_latent_quant_config(quant_config)

    if uses_quantized_latent_projections:
        assert actual is quant_config
    else:
        assert actual is None


def test_kimi_k3_unquantized_model_keeps_latent_projections_unquantized():
    assert _routed_latent_quant_config(None) is None


def test_kimi_k3_projector_registers_rotation_for_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    class StubReplicatedLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", StubReplicatedLinear)
    monkeypatch.setattr(kimi_k3, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(kimi_k3, "get_act_fn", lambda *args, **kwargs: nn.Identity())
    config = KimiK3VisionConfig(
        mm_hidden_size=2,
        text_hidden_size=8,
        merge_kernel_size=(2, 2),
    )
    projector = KimiK3MultiModalProjector(config)

    assert projector.rot_proj is not None


@pytest.mark.parametrize(
    ("loaded_weights", "has_rot_proj"),
    [
        ({"mm_projector.rot_proj.weight"}, True),
        ({"mm_projector.linear_1.weight"}, False),
    ],
)
def test_kimi_k3_enables_projector_rotation_only_when_weight_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    loaded_weights: set[str],
    has_rot_proj: bool,
):
    class StubLoader:
        def __init__(self, model, *, skip_prefixes):
            assert model is wrapper
            assert skip_prefixes == []

        def load_weights(self, weights, *, mapper):
            assert list(weights) == []
            assert mapper is wrapper.hf_to_vllm_mapper
            return loaded_weights

    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", StubLoader)
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.mm_projector = nn.Module()
    wrapper.mm_projector.rot_proj = nn.Linear(1, 1, bias=False)

    actual = wrapper.load_weights(iter(()))

    assert actual == loaded_weights
    assert hasattr(wrapper.mm_projector, "rot_proj") is has_rot_proj
    assert ("mm_projector.rot_proj.weight" in dict(wrapper.named_parameters())) is has_rot_proj


def test_kimi_k3_deletes_unused_rot_proj_when_projector_is_placeholder(
    monkeypatch: pytest.MonkeyPatch,
):
    # Text-only serving (--language-model-only, or --limit-mm-per-prompt at 0
    # for all tower modalities) wraps tower components in StageMissingLayer.
    # Its __getattr__ delegates to the wrapped projector, but del acts on the
    # placeholder's own registries (empty by design), so deleting
    # mm_projector.rot_proj directly raises AttributeError. The deletion must
    # target the wrapped module instead.
    class StubLoader:
        def __init__(self, model, *, skip_prefixes):
            assert model is wrapper
            assert skip_prefixes == []

        def load_weights(self, weights, *, mapper):
            assert list(weights) == []
            assert mapper is wrapper.hf_to_vllm_mapper
            return {"mm_projector.linear_1.weight"}

    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", StubLoader)
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    projector = nn.Module()
    projector.rot_proj = nn.Linear(1, 1, bias=False)
    wrapper.mm_projector = StageMissingLayer("vision_tower", projector)

    actual = wrapper.load_weights(iter(()))

    assert actual == {"mm_projector.linear_1.weight"}
    # The unused rotation was released from the wrapped projector...
    assert hasattr(projector, "rot_proj") is False
    # ...and lookups through the placeholder no longer find it either.
    assert hasattr(wrapper.mm_projector, "rot_proj") is False


def test_kimi_k3_projector_applies_rotation_only_after_weight_load():
    class PassthroughLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    class ScaleLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2, None

    projector = KimiK3MultiModalProjector.__new__(KimiK3MultiModalProjector)
    nn.Module.__init__(projector)
    projector.input_size = 2
    projector.linear_1 = PassthroughLinear()
    projector.linear_2 = PassthroughLinear()
    projector.act = nn.Identity()
    projector.post_norm = nn.Identity()
    image_features = torch.tensor([[1.0, 2.0]])

    projector.rot_proj = ScaleLinear()
    del projector.rot_proj
    assert not hasattr(projector, "rot_proj")
    torch.testing.assert_close(projector(image_features), image_features)

    projector.rot_proj = ScaleLinear()
    torch.testing.assert_close(projector(image_features), image_features * 2)


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight": object()},
            "layers.1.experts.w13_weight",
        ),
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_packed",
        ),
        (
            "layers.1.experts.w2_weight",
            {"layers.1.experts.w2_weight_packed": object()},
            "layers.1.experts.w2_weight_packed",
        ),
        (
            "layers.1.experts.w13_weight_scale",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_scale",
        ),
    ],
)
def test_kimi_k3_resolves_packed_expert_checkpoint_names(
    name: str,
    params: dict[str, object],
    expected: str,
):
    assert _resolve_packed_expert_weight_name(name, params) == expected


def test_kimi_k3_config_normalizes_checkpoint_schema_for_vllm():
    """Cover only the non-pass-through checkpoint-to-vLLM adaptations."""
    config = KimiK3Config(
        text_config={"hidden_size": 4096},
        vision_config={
            "vt_num_attention_heads": 12,
            "vt_num_hidden_layers": 7,
            "vt_hidden_size": 1024,
            "vt_intermediate_size": 3584,
            "text_hidden_size": 1024,
        },
        use_unified_vision_chunk=True,
    )

    # Old plugin checkpoints used vision_chunk, while vLLM consumes image.
    assert not hasattr(config, "use_unified_vision_chunk")
    # MoonViT consumers use canonical names instead of checkpoint vt_* names.
    assert config.vision_config.num_attention_heads == 12
    assert config.vision_config.hidden_size == 1024
    # The projector output must follow the text model, not stale vision config.
    assert config.vision_config.text_hidden_size == config.text_config.hidden_size


def test_kimi_k3_model_uses_image_placeholder_from_upstream_contract():
    assert AscendKimiK3ForConditionalGeneration.get_placeholder_str("image", 0) == "<|kimi_image_placeholder|>"
    with pytest.raises(ValueError, match="does not support modality"):
        AscendKimiK3ForConditionalGeneration.get_placeholder_str(
            "vision_chunk",
            0,
        )


def test_kimi_k3_weight_mapper_adds_inner_language_model_prefix():
    mapper = AscendKimiK3ForConditionalGeneration.hf_to_vllm_mapper

    assert (
        mapper._map_name("language_model.layers.12.self_attn.q_proj.weight")
        == "language_model.model.layers.12.self_attn.q_proj.weight"
    )
    assert (
        mapper._map_name("language_model.model.layers.12.self_attn.q_proj.weight")
        == "language_model.model.layers.12.self_attn.q_proj.weight"
    )
    assert mapper._map_name("mm_projector.proj.0.weight") == "mm_projector.linear_1.weight"


@pytest.mark.parametrize(
    ("weight_name", "expected_layer"),
    [
        ("model.layers.93.self_attn.q_proj.weight", 93),
        ("layers.94.self_attn.q_proj.weight", 94),
        ("language_model.layers.93.mlp.gate_proj.weight", 93),
        ("language_model.model.layers.94.mlp.up_proj.weight", 94),
        ("model.layers.92.self_attn.q_proj.weight", None),
        ("model.layers.95.self_attn.q_proj.weight", None),
    ],
)
def test_kimi_k3_spec_layer_detection_accepts_loader_prefixes(
    weight_name: str,
    expected_layer: int | None,
):
    config = SimpleNamespace(
        num_hidden_layers=93,
        num_nextn_predict_layers=2,
    )

    assert get_spec_layer_idx_from_weight_name(config, weight_name) == expected_layer


def test_kimi_k3_spec_layer_detection_allows_missing_nextn_config():
    config = SimpleNamespace(num_hidden_layers=93)

    assert (
        get_spec_layer_idx_from_weight_name(
            config,
            "model.layers.93.self_attn.q_proj.weight",
        )
        is None
    )


def test_kimi_k3_vision_tp16_falls_back_to_data_parallel(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.models import vision as vision_utils

    class StubModule(nn.Module):
        pass

    qkv_kwargs: dict[str, object] = {}
    output_kwargs: dict[str, object] = {}

    def fake_qkv(*args, **kwargs):
        del args
        qkv_kwargs.update(kwargs)
        return StubModule()

    def fake_output(*args, **kwargs):
        del args
        output_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(
        vision_utils,
        "get_tensor_model_parallel_world_size",
        lambda: 16,
    )
    monkeypatch.setattr(
        kimi_k3,
        "get_tensor_model_parallel_world_size",
        lambda: 16,
    )
    monkeypatch.setattr(
        kimi_k3,
        "KimiK3VisionMLP",
        lambda *args, **kwargs: StubModule(),
    )
    monkeypatch.setattr(kimi_k3, "get_act_fn", lambda name: nn.Identity())
    monkeypatch.setattr(kimi_k3, "QKVParallelLinear", fake_qkv)
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", fake_output)
    monkeypatch.setattr(
        kimi_k3,
        "MMEncoderAttention",
        lambda *args, **kwargs: StubModule(),
    )

    layer = KimiK3VisionEncoderLayer(
        KimiK3VisionConfig(vt_num_attention_heads=12),
        quant_config=None,
        prefix="vision_tower.encoder.blocks.0",
    )

    assert layer.use_data_parallel is True
    assert layer.tp_size == 1
    assert layer.num_local_heads == 12
    assert qkv_kwargs["disable_tp"] is True
    assert output_kwargs["disable_tp"] is True


def test_kimi_k3_vit_dp_compat_calls_release_helper_without_num_heads(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[None] = []

    def release_helper():
        calls.append(None)
        return False

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: version == "0.25.1")
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", release_helper)

    assert kimi_k3._is_vit_use_data_parallel(8) is False
    assert calls == [None]


def test_kimi_k3_vit_dp_compat_recreates_release_tp_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_release_helper():
        pytest.fail("The release helper must not run after the TP fallback")

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: version == "0.25.1")
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", unexpected_release_helper)

    assert kimi_k3._is_vit_use_data_parallel(12) is True


def test_kimi_k3_vit_dp_compat_passes_num_heads_to_main_helper(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def main_helper(num_heads):
        calls.append(num_heads)
        return True

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: False)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", main_helper)

    assert kimi_k3._is_vit_use_data_parallel(12) is True
    assert calls == [12]


def test_kimi_k3_skips_explicit_move_for_meta_modules():
    module = nn.Linear(4, 4, device="meta")

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.is_meta for parameter in module.parameters())


def test_kimi_k3_moves_non_meta_modules():
    module = nn.Linear(4, 4)

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_kimi_k3_passes_situ_parameters_through_activation_config(monkeypatch):
    class StubModule(nn.Module):
        pass

    fused_moe_kwargs = {}

    def fake_replicated_linear(*args, **kwargs):
        return StubModule()

    def fake_fused_moe(**kwargs):
        fused_moe_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", fake_replicated_linear)
    monkeypatch.setattr(kimi_k3, "FusedMoE", fake_fused_moe)
    config = SimpleNamespace(
        hidden_act="situ",
        hidden_size=32,
        routed_expert_hidden_size=16,
        num_shared_experts=0,
        num_experts=8,
        rms_norm_eps=1e-6,
        latent_moe_use_norm=False,
        moe_intermediate_size=12,
        num_experts_per_token=2,
        moe_renormalize=True,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=2.5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    KimiK3MoE(config, prefix="model.layers.1.block_sparse_moe")

    activation = fused_moe_kwargs["activation"]
    assert isinstance(activation, SituActivationConfig)
    assert activation.beta == 4.0
    assert activation.linear_beta == 25.0


def test_kimi_k3_dense_mlp_uses_callable_situ(monkeypatch):
    class StubLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "MergedColumnParallelLinear", lambda *args, **kwargs: StubLinear())
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", lambda *args, **kwargs: StubLinear())
    config = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    mlp = KimiK3MLP(config, hidden_size=4, intermediate_size=2)
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    output = mlp(hidden_states)

    assert isinstance(mlp.act_fn, AscendSituAndMul)
    assert output.shape == (1, 2)


def test_text_model_captures_materialized_dspark_aux_stream(monkeypatch: pytest.MonkeyPatch):
    residual_calls: list[tuple[int, torch.Tensor]] = []
    consumed_inputs: list[torch.Tensor] = []

    class Marker(nn.Module):
        def __init__(self, value: int) -> None:
            super().__init__()
            self.value = value

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.self_attention_res_proj = Marker(layer_idx)
            self.self_attention_res_norm = nn.Identity()

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            block_residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del positions
            if block_residual.shape[1] > 0:
                hidden_states = kimi_k3._apply_attention_residual(
                    hidden_states,
                    block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            consumed_inputs.append(hidden_states.clone())
            if block_residual.shape[1] == 0:
                block_residual = hidden_states.unsqueeze(1)
            return hidden_states + 10, block_residual

    def fake_attention_residual(
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        projection: Marker,
        norm: nn.Module,
    ) -> torch.Tensor:
        del block_residual, norm
        residual_calls.append((projection.value, hidden_states.clone()))
        return hidden_states + projection.value * 100

    monkeypatch.setattr(kimi_k3, "_apply_attention_residual", fake_attention_residual)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.do_not_compile = True
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([FakeLayer(0), FakeLayer(1)])
    model.embed_input_ids = MagicMock(return_value=torch.tensor([[1.0]]))
    model.output_attn_res_proj = Marker(0)
    model.output_attn_res_norm = nn.Identity()
    model.norm = nn.Identity()
    model.dspark_aux_capture_materialized = True
    model.attn_res_mode = "original"
    model._set_aux_hidden_state_layers((0, 1))

    hidden_states, aux_hidden_states = model(
        torch.tensor([1]),
        torch.tensor([0]),
        None,
    )

    torch.testing.assert_close(aux_hidden_states[0], consumed_inputs[0])
    torch.testing.assert_close(aux_hidden_states[1], consumed_inputs[1])
    torch.testing.assert_close(aux_hidden_states[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(aux_hidden_states[1], torch.tensor([[111.0]]))
    torch.testing.assert_close(hidden_states, torch.tensor([[121.0]]))
    assert [layer_idx for layer_idx, _ in residual_calls] == [1, 1, 0]

    residual_calls.clear()
    consumed_inputs.clear()
    model.dspark_aux_capture_materialized = False
    model._set_aux_hidden_state_layers((1,))

    _, raw_aux_hidden_states = model(
        torch.tensor([1]),
        torch.tensor([0]),
        None,
    )

    torch.testing.assert_close(raw_aux_hidden_states[0], torch.tensor([[11.0]]))
    assert [layer_idx for layer_idx, _ in residual_calls] == [1, 0]


def test_two_phase_phase1_phase2_matches_single_softmax():
    """The two-phase Online Softmax helpers are numerically identical to the
    single-pass softmax over ``[completed blocks..., running partial]``."""

    def single_pass(rows, partial_vec, query, eps):
        values = torch.cat((rows, partial_vec.unsqueeze(1)), dim=1).float()
        q = query.float()
        inv_rms = torch.rsqrt(values.square().mean(-1) + eps)
        logits = torch.matmul(values, q) * inv_rms
        weights = logits.softmax(-1)
        return torch.matmul(weights.unsqueeze(1), values).squeeze(1)

    torch.manual_seed(3)
    token_count, num_blocks, hidden_size = 2, 3, 6
    eps = 1e-6
    block_residual = torch.randn(token_count, num_blocks, hidden_size)
    queries = torch.randn(4, hidden_size)

    phase1 = kimi_k3._prepare_attn_res_phase1(block_residual, queries, eps)
    slot = kimi_k3.AttnResPhase2Slot(
        queries[0],
        phase1.inter_numerator[0],
        phase1.inter_max[0],
        phase1.inter_exp_sum[0],
    )

    partial = torch.randn(token_count, hidden_size)
    expected = single_pass(block_residual, partial, queries[0], eps)
    torch.testing.assert_close(
        kimi_k3._merge_attn_res_partial(partial, slot, eps),
        expected,
        atol=1e-4,
        rtol=1e-4,
    )

    delta = torch.randn(token_count, hidden_size)
    updated = partial + delta
    torch.testing.assert_close(
        kimi_k3._update_attn_res_phase2(partial.clone(), delta, slot, eps),
        single_pass(block_residual, updated, queries[0], eps),
        atol=1e-4,
        rtol=1e-4,
    )

    torch.testing.assert_close(
        kimi_k3._merge_attn_res_slot(updated.float(), slot, eps),
        single_pass(block_residual, updated, queries[0], eps),
        atol=1e-4,
        rtol=1e-4,
    )


def test_two_phase_matches_original_text_model_flow(monkeypatch: pytest.MonkeyPatch):
    """Running the text model with the two-phase backend reproduces the
    original single-pass residual flow (and both DSpark aux capture modes)."""

    def single_pass(prefix_sum, block_residual, projection, norm):
        values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
        weights = projection.weight.squeeze(0).float() * norm.weight.float()
        inv_rms = torch.rsqrt(values.square().mean(-1) + norm.variance_epsilon)
        logits = torch.matmul(values, weights) * inv_rms
        probabilities = logits.softmax(-1)
        return torch.matmul(probabilities.unsqueeze(1), values).squeeze(1).to(prefix_sum.dtype)

    class FakeLinear:
        def __init__(self, hidden_size):
            self.weight = nn.Parameter(torch.randn(1, hidden_size))

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx, block_size, hidden_size):
            super().__init__()
            self.layer_idx = layer_idx
            self.block_size = block_size
            self.self_attention_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.mlp_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.self_attention_res_proj = FakeLinear(hidden_size)
            self.mlp_res_proj = FakeLinear(hidden_size)

        def _attention(self, positions, attention_input):
            del positions
            return attention_input * 0.5

        def _mlp(self, mlp_input):
            return mlp_input + 10.0

        def forward(self, positions, hidden_states, block_residual):
            del positions
            prefix_sum = hidden_states
            if block_residual.shape[1] > 0:
                hidden_states = single_pass(
                    prefix_sum,
                    block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            if self.layer_idx % self.block_size == 0:
                block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
                prefix_sum = None
            attention_output = self._attention(None, hidden_states)
            prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
            hidden_states = single_pass(
                prefix_sum,
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
            )
            hidden_states = self._mlp(hidden_states)
            return prefix_sum + hidden_states, block_residual

    monkeypatch.setattr(kimi_k3, "_apply_attention_residual", single_pass)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    hidden_size, num_layers, block_size = 4, 3, 2

    def build_model(mode: str) -> KimiK3TextModel:
        torch.manual_seed(11)
        model = KimiK3TextModel.__new__(KimiK3TextModel)
        nn.Module.__init__(model)
        model.do_not_compile = True
        model.start_layer = 0
        model.end_layer = num_layers
        model.config = SimpleNamespace(
            hidden_size=hidden_size,
            rms_norm_eps=1e-6,
            attn_res_block_size=block_size,
        )
        model.layers = nn.ModuleList([FakeLayer(layer_idx, block_size, hidden_size) for layer_idx in range(num_layers)])
        model.output_attn_res_proj = FakeLinear(hidden_size)
        model.output_attn_res_norm = SimpleNamespace(
            weight=nn.Parameter(torch.randn(hidden_size)),
            variance_epsilon=1e-6,
        )
        model.norm = nn.Identity()
        model.dspark_aux_capture_materialized = True
        model.attn_res_mode = mode
        model.attn_res_effective_queries = None
        model._set_aux_hidden_state_layers((0, 2))
        return model

    original = build_model("original")
    two_phase = build_model("two_phase")
    embeds = torch.randn(1, hidden_size)
    positions = torch.tensor([0])

    orig_hidden, orig_aux = original(None, positions, None, inputs_embeds=embeds)
    two_hidden, two_aux = two_phase(None, positions, None, inputs_embeds=embeds)

    torch.testing.assert_close(two_hidden, orig_hidden, atol=1e-4, rtol=1e-4)
    assert len(two_aux) == len(orig_aux)
    for captured_two, captured_orig in zip(two_aux, orig_aux):
        torch.testing.assert_close(captured_two, captured_orig, atol=1e-4, rtol=1e-4)

    for model in (original, two_phase):
        model.dspark_aux_capture_materialized = False
        model._set_aux_hidden_state_layers((1, 3))

    orig_hidden, orig_aux = original(None, positions, None, inputs_embeds=embeds)
    two_hidden, two_aux = two_phase(None, positions, None, inputs_embeds=embeds)

    torch.testing.assert_close(two_hidden, orig_hidden, atol=1e-4, rtol=1e-4)
    for captured_two, captured_orig in zip(two_aux, orig_aux):
        torch.testing.assert_close(captured_two, captured_orig, atol=1e-4, rtol=1e-4)


def test_kimi_k3_attn_res_fused_availability_gating(monkeypatch: pytest.MonkeyPatch):
    """The fused AttnRes backend requires both the CANNBot DSL operators and
    the opt-in environment variable."""
    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", object())
    monkeypatch.setattr(kimi_k3, "_block_attn_res_update_fused", object())
    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED", "1")
    assert kimi_k3._use_fused_attn_res() is True

    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED", "0")
    assert kimi_k3._use_fused_attn_res() is False

    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", None)
    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED", "1")
    assert kimi_k3._use_fused_attn_res() is False


class _AttnResFusedDispatchStub(KimiK3TextModel):
    """Minimal model shim exercising the fused dispatch methods."""

    attn_res_mode = "fused"

    def __init__(self) -> None:
        pass


def test_fused_phase1_and_phase2_dispatch_to_cannbot_dsl(monkeypatch: pytest.MonkeyPatch):
    """The fused Phase-1/Phase-2 dispatch forwards the reference operator
    interfaces (valid_blocks scalar, fp32 partial, bf16 delta) and consumes
    the returned ``(h, partial_blocks)`` buffers correctly."""
    token_count, num_blocks, hidden_size = 2, 3, 6
    eps = 1e-6
    torch.manual_seed(7)
    v = torch.randn(token_count, num_blocks, hidden_size).contiguous()
    queries = torch.randn(4, hidden_size).contiguous()
    prepare_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    update_calls: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
    ] = []

    def fused_prepare(v_arg, eq_arg, valid_blocks_arg, eps=1e-6):
        prepare_calls.append((v_arg, eq_arg, valid_blocks_arg, eps))
        assert v_arg.dtype == torch.float32 and v_arg.is_contiguous()
        assert eq_arg.dtype == torch.float32 and eq_arg.dim() == 2
        assert valid_blocks_arg.dtype == torch.int64 and valid_blocks_arg.numel() == 1
        assert int(valid_blocks_arg) == v_arg.shape[1]
        phase1 = kimi_k3._prepare_attn_res_phase1(v_arg, eq_arg, eps)
        return phase1.inter_numerator, phase1.inter_max, phase1.inter_exp_sum

    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", fused_prepare)

    stub = _AttnResFusedDispatchStub()
    phase1 = stub._run_attn_res_phase1(v, queries, eps)
    assert len(prepare_calls) == 1
    assert int(prepare_calls[0][2]) == num_blocks
    expected = kimi_k3._prepare_attn_res_phase1(v, queries, eps)
    for actual, ref in zip(phase1, expected):
        torch.testing.assert_close(actual, ref, atol=1e-4, rtol=1e-4)

    partial = torch.randn(token_count, hidden_size, dtype=torch.float32).contiguous()
    delta = torch.randn(token_count, hidden_size, dtype=torch.bfloat16).contiguous()
    slot = kimi_k3.AttnResPhase2Slot(
        queries[0],
        phase1.inter_numerator[0],
        phase1.inter_max[0],
        phase1.inter_exp_sum[0],
    )

    def fused_update(
        partial_block,
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon,
    ):
        update_calls.append(
            (partial_block, partial_delta, effective_query, inter_max, inter_exp_sum, inter_numerator, epsilon)
        )
        assert partial_block.dtype == torch.float32 and partial_block.is_contiguous()
        assert partial_delta.dtype in (torch.bfloat16, torch.float16)
        assert partial_delta.shape == partial_block.shape
        assert effective_query.dtype == torch.float32 and effective_query.shape == (hidden_size,)
        assert inter_max.dtype == torch.float32 and inter_max.shape == (token_count,)
        assert inter_exp_sum.dtype == torch.float32 and inter_exp_sum.shape == (token_count,)
        assert inter_numerator.dtype == torch.float32 and inter_numerator.shape == (token_count, hidden_size)
        updated = partial_block.clone().float() + partial_delta.float()
        merged = kimi_k3._update_attn_res_phase2(
            partial_block,
            partial_delta,
            kimi_k3.AttnResPhase2Slot(effective_query, inter_numerator, inter_max, inter_exp_sum),
            epsilon,
        )
        return merged.to(partial_delta.dtype), updated

    monkeypatch.setattr(kimi_k3, "_block_attn_res_update_fused", fused_update)

    partial_before = partial.clone()
    output = stub._run_attn_res_phase2(partial, delta, slot, eps)

    assert len(update_calls) == 1
    assert torch.equal((partial_before.float() + delta.float()).to(torch.float32), partial)
    expected_merged = kimi_k3._update_attn_res_phase2(partial_before.clone(), delta, slot, eps)
    torch.testing.assert_close(output, expected_merged.to(torch.bfloat16), atol=1e-2, rtol=1e-2)


def test_fused_mode_text_model_flow_matches_two_phase(monkeypatch: pytest.MonkeyPatch):
    """With the CANNBot DSL operators installed, the fused backend follows the
    same dynamic block flow as the pure-Torch two-phase backend and produces
    matching results (DSpark aux capture included)."""

    def single_pass(prefix_sum, block_residual, projection, norm):
        values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
        weights = projection.weight.squeeze(0).float() * norm.weight.float()
        inv_rms = torch.rsqrt(values.square().mean(-1) + norm.variance_epsilon)
        logits = torch.matmul(values, weights) * inv_rms
        probabilities = logits.softmax(-1)
        return torch.matmul(probabilities.unsqueeze(1), values).squeeze(1).to(prefix_sum.dtype)

    class FakeLinear:
        def __init__(self, hidden_size):
            self.weight = nn.Parameter(torch.randn(1, hidden_size))

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx, block_size, hidden_size):
            super().__init__()
            self.layer_idx = layer_idx
            self.block_size = block_size
            self.self_attention_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.mlp_res_norm = SimpleNamespace(
                weight=nn.Parameter(torch.randn(hidden_size)),
                variance_epsilon=1e-6,
            )
            self.self_attention_res_proj = FakeLinear(hidden_size)
            self.mlp_res_proj = FakeLinear(hidden_size)

        def _attention(self, positions, attention_input):
            del positions
            return attention_input * 0.5

        def _mlp(self, mlp_input):
            return mlp_input + 10.0

        def forward(self, positions, hidden_states, block_residual):
            del positions
            prefix_sum = hidden_states
            if block_residual.shape[1] > 0:
                hidden_states = single_pass(
                    prefix_sum,
                    block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            if self.layer_idx % self.block_size == 0:
                block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
                prefix_sum = None
            attention_output = self._attention(None, hidden_states)
            prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
            hidden_states = single_pass(
                prefix_sum,
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
            )
            hidden_states = self._mlp(hidden_states)
            return prefix_sum + hidden_states, block_residual

    monkeypatch.setattr(kimi_k3, "_apply_attention_residual", single_pass)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    hidden_size, num_layers, block_size = 4, 3, 2
    prepare_calls: list[int] = []
    update_calls: list[int] = []

    def fused_prepare(v, effective_queries, valid_blocks, eps=1e-6):
        prepare_calls.append(int(valid_blocks))
        assert v.dtype == torch.float32 and int(valid_blocks) == v.shape[1]
        phase1 = kimi_k3._prepare_attn_res_phase1(v, effective_queries, eps)
        return phase1.inter_numerator, phase1.inter_max, phase1.inter_exp_sum

    def fused_update(partial_block, partial_delta, effective_query, inter_max, inter_exp_sum, inter_numerator, epsilon):
        update_calls.append(partial_delta.shape[0])
        assert partial_block.dtype == torch.float32
        merged = kimi_k3._update_attn_res_phase2(
            partial_block,
            partial_delta,
            kimi_k3.AttnResPhase2Slot(effective_query, inter_numerator, inter_max, inter_exp_sum),
            epsilon,
        )
        return merged, partial_block

    monkeypatch.setattr(kimi_k3, "_block_attn_res_prepare_fused", fused_prepare)
    monkeypatch.setattr(kimi_k3, "_block_attn_res_update_fused", fused_update)

    def build_model() -> KimiK3TextModel:
        model = KimiK3TextModel.__new__(KimiK3TextModel)
        nn.Module.__init__(model)
        model.do_not_compile = True
        model.start_layer = 0
        model.end_layer = num_layers
        model.config = SimpleNamespace(
            hidden_size=hidden_size,
            rms_norm_eps=1e-6,
            attn_res_block_size=block_size,
        )
        model.layers = nn.ModuleList([FakeLayer(layer_idx, block_size, hidden_size) for layer_idx in range(num_layers)])
        model.output_attn_res_proj = FakeLinear(hidden_size)
        model.output_attn_res_norm = SimpleNamespace(
            weight=nn.Parameter(torch.randn(hidden_size)),
            variance_epsilon=1e-6,
        )
        model.norm = nn.Identity()
        model.dspark_aux_capture_materialized = True
        model.attn_res_effective_queries = None
        model._set_aux_hidden_state_layers((0, 2))
        return model

    torch.manual_seed(11)
    two_phase = build_model()
    two_phase.attn_res_mode = "two_phase"
    torch.manual_seed(11)
    fused = build_model()
    fused.attn_res_mode = "fused"

    embeds = torch.randn(1, hidden_size)
    positions = torch.tensor([0])

    two_hidden, two_aux = two_phase(None, positions, None, inputs_embeds=embeds)
    fused_hidden, fused_aux = fused(None, positions, None, inputs_embeds=embeds)

    torch.testing.assert_close(fused_hidden, two_hidden, atol=1e-4, rtol=1e-4)
    assert len(fused_aux) == len(two_aux)
    for captured_fused, captured_two in zip(fused_aux, two_aux):
        torch.testing.assert_close(captured_fused, captured_two, atol=1e-4, rtol=1e-4)
    assert prepare_calls == [1, 2]
    assert len(update_calls) == 4


def test_kimi_k3_dspark_aux_capture_mode_is_forwarded():
    causal_model = AscendKimiK3ForCausalLM.__new__(AscendKimiK3ForCausalLM)
    nn.Module.__init__(causal_model)
    causal_model.model = SimpleNamespace(dspark_aux_capture_materialized=False)

    causal_model.set_dspark_aux_capture_materialized(True)

    assert causal_model.model.dspark_aux_capture_materialized is True

    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = MagicMock()

    wrapper.set_dspark_aux_capture_materialized(True)

    wrapper.language_model.set_dspark_aux_capture_materialized.assert_called_once_with(True)
