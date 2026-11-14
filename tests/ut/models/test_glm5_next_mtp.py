# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import pytest
import torch
import vllm.config.speculative as speculative_config
from transformers import Glm5NextTextConfig

from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, PreprocessType
from vllm_ascend.models.glm5next import mtp as mtp_module
from vllm_ascend.models.glm5next.attention import Glm5NextMLAAttention
from vllm_ascend.models.glm5next.model import (
    Glm5NextForConditionalGeneration,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.models.glm5next.mtp import Glm5NextMTP, Glm5NextMultiTokenPredictorLayer
from vllm_ascend.patch.platform.patch_speculative_config import (
    _normalize_legacy_qwen3_dspark_config,
)


def test_get_spec_layer_idx_accepts_checkpoint_prefixes():
    config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=2,
    )

    assert get_spec_layer_idx_from_weight_name(config, "model.layers.45.enorm.weight") == 45
    assert get_spec_layer_idx_from_weight_name(config, "layers.46.self_attn.q_a_proj.weight") == 46
    assert get_spec_layer_idx_from_weight_name(config, "model.layers.44.mlp.weight") is None
    assert get_spec_layer_idx_from_weight_name(config, "rot.weight") is None


def test_mtp_rewrites_layer_and_shared_weight_names():
    mtp = object.__new__(Glm5NextMTP)

    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.self_attn.q_a_proj.weight",
        )
        == "model.layers.45.mtp_block.self_attn.q_a_proj.weight"
    )
    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.shared_head.norm.weight",
        )
        == "model.layers.45.shared_head.norm.weight"
    )


def test_multimodal_mapper_flattens_modelslim_forget_gate_prefix():
    weight_name = "model.language_model.layers.0.self_attn.forget_gate.f_b_proj.weight"

    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name(weight_name)
        == "language_model.model.layers.0.self_attn.f_b_proj.weight"
    )

    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name("model.language_model.layers.1.attn_hc.fn")
        == "language_model.model.layers.1.hc_attn_fn"
    )
    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name("model.language_model.layers.1.ffn_hc.scale")
        == "language_model.model.layers.1.hc_ffn_scale"
    )


def test_glm5_speculative_config_selects_mtp_architecture():
    config = Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        num_nextn_predict_layers=2,
    )

    result = _normalize_legacy_qwen3_dspark_config(config)

    assert result.model_type == "glm5_next_mtp"
    assert result.n_predict == 2
    assert result.architectures == ["Glm5NextMTPModel"]
    assert "glm5_next_mtp" in get_args(speculative_config.MTPModelTypes)


@pytest.mark.parametrize("num_decodes", [0, 1])
def test_mtp_cache_only_prefill_writes_main_and_indexer_caches_before_transfer(monkeypatch, num_decodes):
    hidden = torch.ones(4, 8)
    slots = torch.arange(4)
    lengths = torch.tensor([4])
    blocks = torch.zeros(1, 1, dtype=torch.int32)
    sfa = AscendSFAMetadata(4, slots, lengths, lengths, lengths, blocks, None, None, num_decodes=num_decodes)
    indexer = AscendIndexerKPoolMetadata(blocks, slots, lengths, slots, 128, 4, num_actual_tokens=4)
    events = MagicMock()
    events.kv_values.return_value = (None, None, None)
    events.transfer_window.return_value = nullcontext()
    impl = object.__new__(AscendSFAImpl)
    impl.preprocess_type = PreprocessType.NATIVE
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl.is_kv_producer = True
    impl.has_indexer = True
    impl.q_lora_rank = impl.kv_lora_rank = 4
    impl.layerwise_kv_cache_hook = None
    impl.fused_qkv_a_proj = lambda x: (x,)
    impl._compose_sfa_kv_cache = lambda cache: cache
    impl.exec_kv = events.kv_values
    impl._store_packed_kv_cache = events.write_main
    impl.indexer = events.write_indexer
    impl.indexer.k_cache = SimpleNamespace(prefix="indexer")
    cache_layer = SimpleNamespace(impl=impl, layer_name="attention", kv_cache=(torch.empty(1),))
    attention = MagicMock(spec=Glm5NextMLAAttention, mla_attn=SimpleNamespace(mla_attn=cache_layer))
    layer = SimpleNamespace(mtp_block=SimpleNamespace(self_attn=attention, input_layernorm=lambda x: x))
    monkeypatch.setattr(
        mtp_module, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"attention": sfa, "indexer": indexer})
    )
    monkeypatch.setattr(mtp_module, "wait_for_kv_layer_from_connector", events.wait)
    monkeypatch.setattr(mtp_module, "notify_kv_cache_written", events.notify)
    monkeypatch.setattr(mtp_module, "attention_transfer_window", events.transfer_window)
    monkeypatch.setattr(mtp_module, "maybe_save_kv_layer_to_connector", events.save)

    assert Glm5NextMultiTokenPredictorLayer._write_prefill_cache(layer, hidden) is (num_decodes == 0)
    if num_decodes:
        assert not events.mock_calls
    else:
        assert [call[0] for call in events.mock_calls] == [
            "wait",
            "kv_values",
            "write_main",
            "write_indexer",
            "notify",
            "transfer_window",
            "save",
        ]
        assert events.kv_values.call_args.args[0].shape == (4, 4)
        assert events.write_indexer.call_args.args[2] is hidden
        assert events.write_indexer.call_args.kwargs == {"compute_topk": False}
