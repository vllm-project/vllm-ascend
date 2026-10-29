# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Deferred torch checks for DeepSeek V4.1 DSpark model/cache integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs
from vllm_ascend.models.deepseek_v41 import dspark as deepseek_v41_dspark_module
from vllm_ascend.models.deepseek_v41.dspark import (
    DeepseekV41DSparkAttention,
    DeepseekV41DSparkDecoderLayer,
    DeepseekV41DSparkModel,
    DeepseekV41DSparkSWACache,
    DSparkDeepseekV41ForCausalLM,
)
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41SWACache, DeepseekV41Model
from vllm_ascend.worker.v1.model_runner import NPUModelRunner


def test_draft_cache_uses_v41_backend_and_explicit_deepseek_v41_spec(monkeypatch):
    spec = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v41",
    )
    with patch.object(AscendDeepseekV41SWACache, "get_kv_cache_spec", return_value=spec):
        cache = DeepseekV41DSparkSWACache.__new__(DeepseekV41DSparkSWACache)
        draft = cache.get_kv_cache_spec(None)
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend

    assert DeepseekV41DSparkSWACache.get_attn_backend(None) is DeepseekV41CacheBackend
    assert type(draft) is AscendSlidingWindowMLASpec
    assert draft.page_size_bytes == 131072
    assert DeepseekV41DSparkDecoderLayer.attention_cls is DeepseekV41DSparkAttention
    assert DeepseekV41DSparkAttention.swa_cache_cls is DeepseekV41DSparkSWACache
    registrations = {}

    def record(kvcache_spec_cls, manager_class, uniform_type_base_spec):
        registrations[kvcache_spec_cls] = manager_class

    monkeypatch.setattr(KVCacheSpecRegistry, "register", record)
    register_ascend_kv_cache_specs()
    assert registrations[type(draft)] is SlidingWindowManager


def test_composite_config_selects_checkpoint_aux_layers():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    text = SimpleNamespace(dspark_target_layer_ids=[37, 38, 39])
    runner.speculative_config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=text)),
    )
    with patch.object(GPUModelRunner, "_get_eagle3_aux_layers_from_config", return_value=None):
        assert runner._get_eagle3_aux_layers_from_config() == (38, 39, 40)


def test_target_exports_residual_entering_selected_layers(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v41.model.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    class Layer:
        def __init__(self, index):
            self.layer_idx = index
            self.engram = None

        def __call__(self, positions, hidden, pre_mix, unused, input_ids):
            return hidden + self.layer_idx + 1, pre_mix

        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden.mean(dim=1)

    model = SimpleNamespace(
        hc_mult=4,
        decoder_replay_layers=None,
        needs_moe_input_ids=False,
        _mtp_hidden_buffer=None,
        prepare_engram=lambda input_ids, positions: ({}, torch.empty(0, dtype=torch.bool)),
        aux_hidden_state_layers=(1, 3),
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        layers=[Layer(i) for i in range(3)],
        norm=lambda hidden: hidden,
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output, aux = DeepseekV41Model.forward(
        model,
        torch.arange(3),
        torch.arange(3),
        None,
        inputs_embeds=hidden,
        engram_lookups={},
        engram_mask=torch.empty(0, dtype=torch.bool),
    )
    torch.testing.assert_close(aux[0], hidden)
    torch.testing.assert_close(aux[1], hidden + 3)
    torch.testing.assert_close(output, hidden + 6)


@pytest.mark.parametrize("cp", [False, True])
def test_v41_draft_routes_to_v41(cp):
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41SWAAttention

    draft_backend = SimpleNamespace()

    def initialize_base(instance, **kwargs):
        torch.nn.Module.__init__(instance)
        instance.compress_ratio = 0
        instance.scale = 512**-0.5
        instance.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=draft_backend))

    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41CPImpl
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl

    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    with (
        patch.object(DeepseekV41SWAAttention, "__init__", initialize_base),
        patch("vllm_ascend.attention.context_parallel.dsa_v41_cp.enable_dsa_cp", return_value=cp),
        patch.object(
            deepseek_v41_dspark_module.DeviceOperator,
            "get_dsv41_packed_cache_ops",
            return_value=draft_backend,
        ),
    ):
        draft = DeepseekV41DSparkAttention(vllm_config=config, prefix="mtp.0.self_attn")
    assert type(draft.v41_impl) is (AscendDSAV41CPImpl if cp else AscendDSAV41Impl)
    assert config.compilation_config.static_forward_context[draft.v41_layer_name] is draft
    assert draft.packed_cache_ops is draft_backend
    assert draft.softmax_scale == 512**-0.5


def test_v41_draft_sequence_parallel_shards_inputs_and_restores_output(monkeypatch):
    class Layer:
        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden.mean(dim=1)

        def __call__(self, positions, hidden, pre_mix, llama_4_scaling, input_ids):
            return hidden, pre_mix

    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    input_ids = torch.tensor([11, 12, 13, 14])
    padding = torch.tensor([False, True, False, False])
    forward_context = SimpleNamespace(is_padding=padding)
    sharded_hidden = hidden[:2].unsqueeze(1).repeat(1, 4, 1)
    sharded_ids = input_ids[:2]
    gathered = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    sp_shard = MagicMock(side_effect=[sharded_hidden, sharded_ids])
    sp_all_gather = MagicMock(return_value=gathered)
    padding_mask = MagicMock(return_value=torch.tensor([False, True]))
    monkeypatch.setattr(deepseek_v41_dspark_module, "sp_shard", sp_shard)
    monkeypatch.setattr(deepseek_v41_dspark_module, "sp_all_gather", sp_all_gather)
    monkeypatch.setattr(deepseek_v41_dspark_module, "sp_padding_mask", padding_mask)
    monkeypatch.setattr(deepseek_v41_dspark_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(deepseek_v41_dspark_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(deepseek_v41_dspark_module.envs, "VLLM_MOE_SKIP_PADDING", True)

    model = SimpleNamespace(
        embed_tokens=MagicMock(return_value=hidden),
        hc_mult=4,
        use_sequence_parallel_moe=True,
        needs_moe_input_ids=False,
        layers={"40": Layer()},
    )

    output = DeepseekV41DSparkModel.forward(model, input_ids, torch.arange(4))

    padding_mask.assert_called_once()
    assert forward_context.is_padding.tolist() == [False, True]
    assert sp_shard.call_args_list[0].args[0].shape == (4, 4, 4)
    assert sp_shard.call_args_list[1].args[0] is input_ids
    sp_all_gather.assert_called_once()
    torch.testing.assert_close(output, gathered[:4])


def test_v41_draft_context_store_uses_physical_pairs_and_preserves_padding():
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel

    cache = torch.empty(3, 128, 1, 8)
    attn = SimpleNamespace(
        packed_cache_ops=None,
        dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(block_size=128, kv_cache=[cache])),
    )
    values = torch.randn(3, 1, 8)
    with patch("vllm_ascend.models.deepseek_v41.dspark.scatter_cache_sk") as store:
        DeepseekV41DSparkModel._store_standard_swa_kv(None, values, torch.tensor([129, -1, 258]), attn)
    actual_cache, slots, updates = store.call_args.args
    assert actual_cache is cache
    assert slots.tolist() == [[1, 1], [-1, -1], [2, 2]]
    torch.testing.assert_close(updates, values.squeeze(1))


def test_v41_draft_context_store_routes_packed_a5_cache_to_device_writer():
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel

    cache = torch.empty(3, 128, 1, 8)
    writer = MagicMock()
    attn = SimpleNamespace(
        packed_cache_ops=SimpleNamespace(write_attention_cache=writer),
        dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(block_size=128, kv_cache=[cache])),
    )
    slots = torch.tensor([129, -1, 258])
    values = torch.randn(3, 1, 8)

    DeepseekV41DSparkModel._store_standard_swa_kv(None, values, slots, attn)

    actual_cache, actual_slots, updates = writer.call_args.args
    assert actual_cache is cache
    assert actual_slots is slots
    assert writer.call_args.kwargs == {"kind": "win"}
    torch.testing.assert_close(updates, values.squeeze(1))


@pytest.mark.parametrize("draft_vocab_size", [None, 16])
def test_draft_constructor_uses_upstream_head_contracts(monkeypatch, draft_vocab_size):
    config = SimpleNamespace(
        hc_mult=4,
        hidden_size=8,
        head_dim=4,
        vocab_size=32,
        draft_vocab_size=draft_vocab_size,
        dspark_block_size=6,
        dspark_target_layer_ids=[37, 38, 39],
        dspark_markov_rank=4,
        num_hidden_layers=40,
        rms_norm_eps=1e-6,
    )
    quant_config = object()
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_text_config=config,
                hf_config=SimpleNamespace(quantization_config={"quant_method": "fp8"}),
            )
        ),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        model_config=SimpleNamespace(dtype=torch.float16),
        quant_config=quant_config,
    )
    module = deepseek_v41_dspark_module
    draft_impls = []

    def layer(*args, **kwargs):
        result = torch.nn.Module()
        result.mlp = SimpleNamespace(gate=SimpleNamespace(tid2eid=None, bias_vl=None))
        impl = MagicMock(spec=module.AscendDSAImpl)
        draft_impls.append(impl)
        result.self_attn = SimpleNamespace(dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=impl)))
        return result

    monkeypatch.setattr(module, "DeepseekV41DSparkDecoderLayer", layer)
    monkeypatch.setattr(module, "VocabParallelEmbedding", lambda *a, **kw: torch.nn.Identity())
    linear = MagicMock(return_value=torch.nn.Identity())
    monkeypatch.setattr(module, "ColumnParallelLinear", linear)
    monkeypatch.setattr(
        module,
        "MergedColumnParallelLinear",
        MagicMock(return_value=torch.nn.Identity()),
    )
    monkeypatch.setattr(module, "RMSNorm", lambda *a, **kw: torch.nn.Identity())
    with (
        patch.object(
            module.DeviceOperator,
            "get_dsv41_packed_cache_ops",
            return_value=object(),
        ),
        patch.object(module, "DSparkMarkovHead", autospec=True, return_value=torch.nn.Identity()) as markov,
        patch.object(module, "DSparkConfidenceHead", autospec=True, return_value=torch.nn.Identity()) as confidence,
    ):
        model = DeepseekV41DSparkModel(vllm_config=vllm_config, prefix="model")
    markov.assert_called_once_with(32, draft_vocab_size or 32, 4, prefix="model.layers.42.markov_head")
    confidence.assert_called_once_with(input_dim=12, prefix="model.confidence_head", bias=False, with_markov=True)
    assert linear.call_args.kwargs["quant_config"] is quant_config
    assert model.layers["42"].markov_head is model.markov_head
    assert len(draft_impls) == 3
    for impl in draft_impls:
        assert impl._oproj_buffer_pool is model._oproj_buffer_pool
        impl.reserve_oproj_buffers.assert_called_once_with(
            torch.device(module.current_platform.device_type), torch.float16
        )


def test_context_wkv_weights_load_into_the_matching_merged_shard(monkeypatch):
    load_calls = []

    def weight_loader(param, loaded_weight, shard_id=None):
        load_calls.append((param, loaded_weight, shard_id))

    original_param = torch.nn.Parameter(torch.empty(4))
    original_param.weight_loader = weight_loader
    merged_param = torch.nn.Parameter(torch.empty(4))
    merged_param.weight_loader = weight_loader
    params = {
        "model.layers.40.self_attn.wkv.weight": original_param,
        "model.context_wkv_proj.weight": merged_param,
    }
    model = SimpleNamespace(
        model=SimpleNamespace(num_dspark_layers=1, get_expert_mapping=lambda: []),
        config=SimpleNamespace(num_attention_heads=1),
        rotation_path=None,
        named_parameters=lambda: params.items(),
    )

    def remap(name):
        if name.startswith("context_wkv_proj."):
            return f"model.{name}"
        return "model.layers.40.self_attn.wkv.weight"

    model._remap_dspark_name = remap
    module = deepseek_v41_dspark_module
    monkeypatch.setattr(module, "process_eagle_weight", lambda *_: None)
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    loaded_weight = torch.arange(4)
    loaded_params = DSparkDeepseekV41ForCausalLM.load_weights(
        model,
        [("mtp.0.attn.wkv.weight", loaded_weight)],
    )

    assert loaded_params == set(params)
    assert [call[2] for call in load_calls] == [None, 0]
    assert load_calls[0][0] is original_param
    assert load_calls[1][0] is merged_param
    assert load_calls[1][1].data_ptr() == loaded_weight.data_ptr()


def test_context_kv_uses_one_stacked_wkv_projection(monkeypatch):
    projection_calls = []
    rope_calls = []
    rope = object()
    stacked_output = torch.arange(24, dtype=torch.float32).view(2, 12)

    class StackedProjection:
        def __call__(self, context_states):
            projection_calls.append(context_states)
            return stacked_output

    attns = [SimpleNamespace(rotary_emb=SimpleNamespace(layername=f"layer-{layer_idx}")) for layer_idx in range(3)]
    layers = {str(layer_idx + 40): SimpleNamespace(self_attn=attn) for layer_idx, attn in enumerate(attns)}

    def get_rope(positions, layer_names):
        rope_calls.append((positions, layer_names))
        return rope

    monkeypatch.setattr(deepseek_v41_dspark_module, "get_cos_and_sin_dsa", get_rope)

    projected = []
    stored = []
    model = SimpleNamespace(
        context_wkv_proj=StackedProjection(),
        num_dspark_layers=3,
        config=SimpleNamespace(head_dim=4),
        layers=layers,
    )

    def project_shared_kv(kv, positions, attn, rope=None):
        projected.append((kv.clone(), positions, attn, rope))
        return kv.unsqueeze(1)

    def store_standard_swa_kv(shared_kv, slot_mapping, attn):
        stored.append((shared_kv.clone(), slot_mapping, attn))

    model._project_shared_kv = project_shared_kv
    model._store_standard_swa_kv = store_standard_swa_kv
    context_states = torch.zeros(2, 5)
    positions = torch.tensor([7, 8])
    slot_mappings = [torch.tensor([0, 1]), None, torch.tensor([4, 5])]

    DeepseekV41DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        slot_mappings,
    )

    assert len(projection_calls) == 1
    assert projection_calls[0] is context_states
    assert len(rope_calls) == 1
    assert rope_calls[0][0] is positions
    assert rope_calls[0][1] == ["layer-0", "layer-1", "layer-2"]
    expected = stacked_output.view(2, 3, 4)
    torch.testing.assert_close(projected[0][0], expected[:, 0])
    torch.testing.assert_close(projected[1][0], expected[:, 2])
    assert [call[3] for call in projected] == [rope, rope]
    assert stored[0][1] is slot_mappings[0]
    assert stored[1][1] is slot_mappings[2]
