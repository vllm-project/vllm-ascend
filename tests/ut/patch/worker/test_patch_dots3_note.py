from importlib import reload
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.model_executor.layers import mla as mla_module
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.models import deepseek_v2

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    Dots3NoteMLAAttentionSpec,
    Dots3NoteSlidingWindowMLASpec,
)
from vllm_ascend.patch.dots3_note_config import Dots3NoteConfig
from vllm_ascend.patch.worker import patch_dots3_note


def test_dots3_note_dynamic_rsf_handles_zero_weights():
    weights, _ = patch_dots3_note._dots3_note_noaux_tc_topk(
        hidden_states=torch.empty(1, 1),
        gating_output=torch.full((1, 2), -torch.inf),
        topk=2,
        renormalize=False,
        e_score_correction_bias=torch.zeros(2),
        routed_scaling_factor=1.0,
        use_dynamic_rsf=True,
    )

    assert torch.isfinite(weights).all()
    assert torch.count_nonzero(weights) == 0


def test_non_dots3_note_deepseek_patches_delegate(monkeypatch):
    config = SimpleNamespace(model_type="deepseek_v3")
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=config))
    instance = SimpleNamespace()

    original_moe = MagicMock()
    monkeypatch.setattr(patch_dots3_note, "_original_moe_init", original_moe)
    patch_dots3_note._moe_init(
        instance,
        config,
        "parallel",
        "quant",
        prefix="model.layers.0.mlp",
    )
    original_moe.assert_called_once_with(
        instance,
        config=config,
        parallel_config="parallel",
        quant_config="quant",
        reduce_results=True,
        prefix="model.layers.0.mlp",
        apply_routed_scale_to_output=False,
    )

    original_decoder = MagicMock()
    monkeypatch.setattr(patch_dots3_note, "_original_decoder_init", original_decoder)
    topk_buffer = torch.empty(0)
    patch_dots3_note._decoder_init(
        instance,
        vllm_config,
        "model.layers.0",
        config,
        topk_buffer,
    )
    original_decoder.assert_called_once_with(
        instance,
        vllm_config,
        "model.layers.0",
        config,
        topk_buffer,
    )

    original_attention = MagicMock()
    monkeypatch.setattr(
        patch_dots3_note,
        "_original_mla_attention_init",
        original_attention,
    )
    patch_dots3_note._deepseek_mla_attention_init(
        instance,
        vllm_config,
        config,
        16,
        2,
        4,
        2,
        4,
        None,
        8,
        non_causal_multi_token_decode=True,
        sliding_window=511,
        rope_parameters={"rope_type": "default"},
    )
    assert original_attention.call_count == 1
    assert original_attention.call_args.kwargs["non_causal_multi_token_decode"] is True
    assert "sliding_window" not in original_attention.call_args.kwargs
    assert "rope_parameters" not in original_attention.call_args.kwargs


def test_dots3_note_routing_is_injected_during_single_construction(monkeypatch):
    config = SimpleNamespace(
        model_type="dots3_note",
        topk_method="noaux_tc",
        use_dynamic_rsf=False,
    )
    instance = SimpleNamespace()
    fused_moe = MagicMock(return_value=object())

    def original_moe_init(self, *_args, **_kwargs):
        self.gate = SimpleNamespace(e_score_correction_bias=torch.tensor([0.0, 0.0, 2.0]))
        self.routed_scaling_factor = 1.0
        self.experts = deepseek_v2.FusedMoEFactory(
            use_grouped_topk=True,
            routed_scaling_factor=2.0,
        )

    monkeypatch.setattr(patch_dots3_note, "_original_moe_init", original_moe_init)
    monkeypatch.setattr(deepseek_v2, "FusedMoEFactory", fused_moe)

    patch_dots3_note._moe_init(instance, config, "parallel")

    fused_moe.assert_called_once()
    kwargs = fused_moe.call_args.kwargs
    assert kwargs["use_grouped_topk"] is False
    assert kwargs["routed_scaling_factor"] == 1.0
    assert deepseek_v2.FusedMoEFactory is fused_moe
    weights, indices = kwargs["custom_routing_function"](
        torch.empty(1, 1),
        torch.tensor([[0.0, 1.0, -1.0]]),
        2,
        True,
    )
    assert set(indices[0].tolist()) == {1, 2}
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))


def test_dots3_note_headwise_gate_uses_replicated_linear(monkeypatch):
    config = SimpleNamespace(
        model_type="dots3_note",
        apply_mla_qkv_lora_rescale=False,
        k_rope_only_layernorm=False,
        sdpa_gate_type="headwise",
        attention_gate_type="headwise",
    )
    instance = SimpleNamespace()

    def original_init(self, **kwargs):
        self.hidden_size = kwargs["hidden_size"]
        self.num_heads = kwargs["num_heads"]
        self.num_local_heads = kwargs["num_heads"] // 2
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = kwargs["v_head_dim"]
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.scaling = self.qk_head_dim**-0.5
        self.mla_attn = SimpleNamespace(skip_topk=False)
        self.is_v32 = False
        self.indexer = None
        self.indexer_rope_emb = None
        self.kv_a_layernorm = object()
        self.kv_b_proj = object()
        self.rotary_emb = object()
        self.o_proj = object()
        self.fused_qkv_a_proj = object()
        self.kv_a_proj_with_mqa = None
        self.q_a_layernorm = object()
        self.q_b_proj = object()
        self.q_proj = None

    replicated = MagicMock(return_value=object())
    column_parallel = MagicMock(return_value=object())
    wrapper = MagicMock(return_value=object())
    monkeypatch.setattr(patch_dots3_note, "_original_mla_attention_init", original_init)
    monkeypatch.setattr(deepseek_v2, "ReplicatedLinear", replicated)
    monkeypatch.setattr(deepseek_v2, "ColumnParallelLinear", column_parallel)
    monkeypatch.setattr(mla_module, "MultiHeadLatentAttentionWrapper", wrapper)
    monkeypatch.setattr(
        patch_dots3_note,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
    )

    patch_dots3_note._deepseek_mla_attention_init(
        instance,
        vllm_config=SimpleNamespace(),
        config=config,
        hidden_size=16,
        num_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        prefix="model.layers.0.self_attn",
    )

    replicated.assert_called_once_with(
        16,
        4,
        bias=False,
        quant_config=None,
        prefix="model.layers.0.self_attn.g_proj",
    )
    column_parallel.assert_not_called()


def test_non_dots3_note_mla_and_weight_patches_delegate(monkeypatch):
    instance = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v3"))
    weights = [("model.norm.weight", torch.ones(1))]
    original_load = MagicMock(return_value={"model.norm.weight"})
    monkeypatch.setattr(patch_dots3_note, "_original_load_weights", original_load)
    assert patch_dots3_note._load_weights(instance, weights) == {"model.norm.weight"}
    original_load.assert_called_once_with(instance, weights)

    original_mla_init = MagicMock()
    monkeypatch.setattr(patch_dots3_note, "_original_mla_init", original_mla_init)
    mla_instance = SimpleNamespace()
    patch_dots3_note._mla_init(mla_instance, "arg", key="value")
    original_mla_init.assert_called_once_with(mla_instance, "arg", key="value")
    assert not hasattr(mla_instance, "_vllm_ascend_dots3_note")

    base_spec = object()
    original_spec = MagicMock(return_value=base_spec)
    monkeypatch.setattr(patch_dots3_note, "_original_get_kv_cache_spec", original_spec)
    assert patch_dots3_note._get_kv_cache_spec(mla_instance, "config") is base_spec


def test_dots3_note_main_model_weight_filter(monkeypatch):
    instance = SimpleNamespace(
        config=SimpleNamespace(
            model_type="dots3_note",
            moe_layer_freq=[1, 0],
            num_hidden_layers=2,
        )
    )
    weights = [
        ("model.layers.0.input_layernorm.weight", torch.ones(1)),
        ("model.layers.2.self_attn.q_proj.weight", torch.ones(1)),
        ("model.mtp.embed_tokens.weight", torch.ones(1)),
        ("model.norm.weight", torch.ones(1)),
    ]
    captured: list[str] = []

    def load(_, filtered_weights):
        captured.extend(name for name, _ in filtered_weights)
        return set(captured)

    monkeypatch.setattr(patch_dots3_note, "_original_load_weights", load)
    result = patch_dots3_note._load_weights(instance, weights)

    assert result == {
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
    }


def test_dots3_note_decoder_projects_layer_specific_config(monkeypatch):
    config = SimpleNamespace(
        model_type="dots3_note",
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        q_lora_rank=1024,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        attention_gate_type="headwise",
        swa_num_attention_heads=8,
        swa_num_key_value_heads=8,
        swa_q_lora_rank=768,
        swa_qk_nope_head_dim=64,
        swa_qk_rope_head_dim=32,
        swa_v_head_dim=96,
        swa_attention_gate_type="elementwise",
        swa_kv_lora_rank=256,
        sliding_window_size=512,
        rope_theta=10_000,
        swa_rope_theta=1_000,
        layer_types=["full_attention", "sliding_attention"],
        moe_layer_freq=[1, 0],
        n_routed_experts=8,
    )
    original_decoder = MagicMock()
    monkeypatch.setattr(patch_dots3_note, "_original_decoder_init", original_decoder)

    patch_dots3_note._decoder_init(
        SimpleNamespace(),
        SimpleNamespace(model_config=SimpleNamespace(hf_config=config)),
        "model.layers.1",
        config,
    )

    projected = original_decoder.call_args.args[3]
    assert projected is not config
    assert projected.num_attention_heads == 8
    assert projected.num_key_value_heads == 8
    assert projected.q_lora_rank == 768
    assert projected.qk_nope_head_dim == 64
    assert projected.qk_rope_head_dim == 32
    assert projected.v_head_dim == 96
    assert projected.kv_lora_rank == 256
    assert projected.sdpa_gate_type == "elementwise"
    assert projected._dots3_note_sliding_window == 511
    assert projected.rope_parameters == {"rope_type": "default", "rope_theta": 1_000}
    assert projected.n_routed_experts is None


def test_dots3_note_full_attention_uses_release_gate(monkeypatch):
    config = SimpleNamespace(
        model_type="dots3_note",
        num_hidden_layers=1,
        attention_gate_type="headwise",
        layer_types=["full_attention"],
        moe_layer_freq=1,
    )
    original_decoder = MagicMock()
    monkeypatch.setattr(patch_dots3_note, "_original_decoder_init", original_decoder)

    patch_dots3_note._decoder_init(
        SimpleNamespace(),
        SimpleNamespace(model_config=SimpleNamespace(hf_config=config)),
        "model.layers.0",
        config,
    )

    assert original_decoder.call_args.args[3].sdpa_gate_type == "headwise"


def test_dots3_note_mtp_layer_uses_dense_mlp(monkeypatch):
    config = Dots3NoteConfig(
        num_hidden_layers=46,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_routed_experts=256,
    )
    original_decoder = MagicMock()
    monkeypatch.setattr(patch_dots3_note, "_original_decoder_init", original_decoder)

    patch_dots3_note._decoder_init(
        SimpleNamespace(),
        SimpleNamespace(model_config=SimpleNamespace(hf_config=config)),
        "model.layers.46",
        config,
    )

    projected = original_decoder.call_args.args[3]
    assert projected.first_k_dense_replace == 47
    assert config.n_routed_experts == 256
    assert config.first_k_dense_replace == 1


def test_deepseek_indexer_uses_leading_rope(monkeypatch):
    indexer = deepseek_v2.Indexer.__new__(deepseek_v2.Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_head = 1
    indexer.head_dim = 4
    indexer.rope_dim = 2
    indexer.softmax_scale = 1.0
    indexer.n_head_scale = 1.0
    indexer.quant_block_size = 4
    indexer.scale_fmt = None
    indexer.use_fused_indexer_q = False
    indexer.wq_b = lambda _: (torch.tensor([[1.0, 2.0, 10.0, 20.0]]), None)
    indexer.wk_weights_proj = lambda _: (torch.tensor([[3.0, 4.0, 30.0, 40.0, 1.0]]), None)
    indexer.k_norm = lambda value: value

    captured = {}

    def rotary_emb(_, q_pe, k_pe):
        captured["q_pe"] = q_pe.clone()
        captured["k_pe"] = k_pe.clone()
        return q_pe + 100, k_pe + 200

    def indexer_op(_, q, k, weights):
        captured["q"] = q.clone()
        captured["k"] = k.clone()
        return weights

    indexer.indexer_op = indexer_op
    monkeypatch.setattr(deepseek_v2.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        deepseek_v2,
        "per_token_group_quant_fp8",
        lambda q, *_args, **_kwargs: (q, torch.ones((q.shape[0], 1))),
    )

    indexer(torch.zeros((1, 4)), torch.zeros((1, 1)), torch.tensor([0]), rotary_emb)

    torch.testing.assert_close(captured["q_pe"].flatten(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(captured["k_pe"].flatten(), torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(captured["q"].flatten(), torch.tensor([101.0, 102.0, 10.0, 20.0]))
    torch.testing.assert_close(captured["k"].flatten(), torch.tensor([203.0, 204.0, 30.0, 40.0]))


def test_dots3_note_mla_cache_specs_keep_marker(monkeypatch):
    base_spec = AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=80,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    monkeypatch.setattr(
        patch_dots3_note,
        "_original_get_kv_cache_spec",
        MagicMock(return_value=base_spec),
    )

    full_attention = SimpleNamespace(
        _vllm_ascend_dots3_note=True,
        sliding_window=None,
    )
    assert isinstance(
        patch_dots3_note._get_kv_cache_spec(full_attention, "config"),
        Dots3NoteMLAAttentionSpec,
    )

    sliding_attention = SimpleNamespace(
        _vllm_ascend_dots3_note=True,
        sliding_window=511,
    )
    sliding_spec = patch_dots3_note._get_kv_cache_spec(sliding_attention, "config")
    assert isinstance(sliding_spec, Dots3NoteSlidingWindowMLASpec)
    assert sliding_spec.sliding_window == 511


def test_dots3_note_noaux_tc_routing():
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    bias = torch.tensor([0.0, 0.0, 2.0])
    weights, indices = patch_dots3_note._dots3_note_noaux_tc_topk(
        hidden_states=torch.empty(1, 1),
        gating_output=logits,
        topk=2,
        renormalize=True,
        e_score_correction_bias=bias,
        routed_scaling_factor=1.0,
    )

    assert set(indices[0].tolist()) == {1, 2}
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))


def test_worker_patch_reload_is_idempotent():
    original_moe_init = patch_dots3_note._original_moe_init
    reload(patch_dots3_note)

    assert deepseek_v2.DeepseekV2MoE.__init__ is patch_dots3_note._moe_init
    assert mla_attention.MLAAttention.__init__ is patch_dots3_note._mla_init
    assert patch_dots3_note._original_moe_init is original_moe_init
