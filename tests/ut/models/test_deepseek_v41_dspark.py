# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred torch checks for Aurora DSpark model/cache integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.core.deepseek_v41 import DeepseekV41DraftSWASpec
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.deepseek_v41 import dspark as deepseek_v41_dspark_module
from vllm_ascend.models.deepseek_v41.dspark import (
    DeepseekV41DSparkAttention,
    DeepseekV41DSparkDecoderLayer,
    DeepseekV41DSparkModel,
    DeepseekV41DSparkSWACache,
)
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_draft_cache_uses_v41_backend_and_explicit_aurora_spec(monkeypatch):
    spec = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v4",
    )
    with patch.object(AscendDeepseekV4SWACache, "get_kv_cache_spec", return_value=spec):
        cache = DeepseekV41DSparkSWACache.__new__(DeepseekV41DSparkSWACache)
        draft = cache.get_kv_cache_spec(None)
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend

    assert DeepseekV41DSparkSWACache.get_attn_backend(None) is DeepseekV41CacheBackend
    assert type(draft) is DeepseekV41DraftSWASpec
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
        needs_moe_input_ids=False,
        prepare_engram=lambda input_ids, positions: ({}, torch.empty(0, dtype=torch.bool)),
        aux_hidden_state_layers=(1, 3),
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        layers=[Layer(i) for i in range(3)],
        norm=lambda hidden: hidden,
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output, aux = DeepseekV41Model.forward(model, torch.arange(3), torch.arange(3), None, inputs_embeds=hidden)
    torch.testing.assert_close(aux[0], hidden)
    torch.testing.assert_close(aux[1], hidden + 3)
    torch.testing.assert_close(output, hidden + 6)


@pytest.mark.parametrize("cp", [False, True])
def test_v41_draft_routes_to_v41_and_disables_post_projection_q_norm(cp):
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention

    ordinary_backend = SimpleNamespace(apply_q_norm=True)
    draft_backend = SimpleNamespace(apply_q_norm=True)

    def initialize_base(instance, **kwargs):
        torch.nn.Module.__init__(instance)
        instance.compress_ratio = 0
        instance.scale = 512**-0.5
        instance.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=draft_backend))

    from vllm_ascend.attention.context_parallel.dsa_v41_cp import DeepseekV41CPImpl
    from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl

    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    with (
        patch.object(DeepseekV4Attention, "__init__", initialize_base),
        patch("vllm_ascend.attention.context_parallel.dsa_v41_cp.enable_dsa_cp", return_value=cp),
        patch("vllm_ascend.attention.context_parallel.dsa_v41_cp.enable_pcp", return_value=False),
    ):
        draft = DeepseekV41DSparkAttention(vllm_config=config, prefix="mtp.0.self_attn")
    assert type(draft.v41_impl) is (DeepseekV41CPImpl if cp else DeepseekV41EagerAttentionImpl)
    assert config.compilation_config.static_forward_context[draft.v41_layer_name] is draft
    assert draft.softmax_scale == 512**-0.5
    assert draft.dsa_attn.dsa_attn.impl.apply_q_norm is False
    assert ordinary_backend.apply_q_norm is True


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
        use_sequence_parallel=True,
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
    attn = SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(block_size=128, kv_cache=[cache])))
    values = torch.randn(3, 1, 8)
    with patch("vllm_ascend.models.deepseek_v41.dspark.scatter_cache_sk") as store:
        DeepseekV41DSparkModel._store_standard_swa_kv(None, values, torch.tensor([129, -1, 258]), attn)
    actual_cache, slots, updates = store.call_args.args
    assert actual_cache is cache
    assert slots.tolist() == [[1, 1], [-1, -1], [2, 2]]
    torch.testing.assert_close(updates, values.squeeze(1))
