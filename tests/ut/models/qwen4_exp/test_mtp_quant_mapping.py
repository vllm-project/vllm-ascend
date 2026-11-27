# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn
from vllm.config import CompilationConfig, CompilationMode

from vllm_ascend.models.qwen4_exp import mtp


def _make_long_context_config(
    *,
    draft_rope_type: str = "default",
    draft_max_position_embeddings: int = 262144,
    draft_architecture: str = "Qwen4ExpMTP",
) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, object]]:
    target_rope_parameters: dict[str, object] = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262144,
    }
    target_model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            rope_parameters=target_rope_parameters,
        ),
        max_model_len=1010000,
    )
    draft_rope_parameters: dict[str, object] = {
        "rope_type": draft_rope_type,
    }
    draft_model_config = SimpleNamespace(
        architectures=[draft_architecture],
        hf_text_config=SimpleNamespace(
            rope_parameters=draft_rope_parameters,
            max_position_embeddings=draft_max_position_embeddings,
        ),
        max_model_len=draft_max_position_embeddings,
    )
    vllm_config = SimpleNamespace(
        model_config=draft_model_config,
        speculative_config=SimpleNamespace(
            target_model_config=target_model_config,
            draft_model_config=draft_model_config,
        ),
    )
    return vllm_config, draft_model_config, target_rope_parameters


def test_qwen4_exp_mtp_position_cache_matches_long_context_target() -> None:
    vllm_config, draft_model_config, target_rope_parameters = _make_long_context_config()
    draft_rope_parameters = draft_model_config.hf_text_config.rope_parameters

    mtp._align_qwen4_exp_mtp_position_cache(vllm_config)

    assert draft_model_config.hf_text_config.max_position_embeddings == 1010000
    assert draft_model_config.max_model_len == 1010000
    assert draft_model_config.hf_text_config.rope_parameters is draft_rope_parameters
    assert draft_rope_parameters == {"rope_type": "default"}
    assert draft_rope_parameters is not target_rope_parameters


def test_qwen4_exp_mtp_position_cache_preserves_non_default_rope() -> None:
    vllm_config, draft_model_config, _ = _make_long_context_config(
        draft_rope_type="yarn",
    )

    mtp._align_qwen4_exp_mtp_position_cache(vllm_config)

    assert draft_model_config.hf_text_config.max_position_embeddings == 262144
    assert draft_model_config.max_model_len == 262144


def test_qwen4_exp_mtp_position_cache_does_not_shrink() -> None:
    vllm_config, draft_model_config, _ = _make_long_context_config(
        draft_max_position_embeddings=1048576,
    )

    mtp._align_qwen4_exp_mtp_position_cache(vllm_config)

    assert draft_model_config.hf_text_config.max_position_embeddings == 1048576
    assert draft_model_config.max_model_len == 1048576


def test_qwen4_exp_mtp_position_cache_ignores_other_architectures() -> None:
    vllm_config, draft_model_config, _ = _make_long_context_config(
        draft_architecture="OtherMTP",
    )

    mtp._align_qwen4_exp_mtp_position_cache(vllm_config)

    assert draft_model_config.hf_text_config.max_position_embeddings == 262144
    assert draft_model_config.max_model_len == 262144


def _rope_model_config(uses_mrope: bool) -> SimpleNamespace:
    rope_parameters = {"mrope_section": [24, 20, 20], "mrope_interleaved": True} if uses_mrope else {}
    hf_config = SimpleNamespace(rope_parameters=rope_parameters)
    return SimpleNamespace(
        uses_mrope=uses_mrope,
        hf_config=hf_config,
        hf_text_config=hf_config,
    )


def test_modelslim_mtp_layer_ids_are_remapped_by_qwen_model_adapter() -> None:
    quant_config = SimpleNamespace(
        quant_description={
            "mtp.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
            "mtp.layers.1.mlp.experts.gate_up_proj": "FLOAT",
            "model.embed_tokens.weight": "FLOAT",
        }
    )

    mtp._remap_modelslim_mtp_layer_ids(
        quant_config,
        mtp_start_layer_idx=48,
        num_mtp_layers=2,
    )

    assert quant_config.quant_description == {
        "mtp.layers.48.self_attn.q_proj.weight": "W8A8_DYNAMIC",
        "mtp.layers.49.mlp.experts.gate_up_proj": "FLOAT",
        "model.embed_tokens.weight": "FLOAT",
    }


def test_modelslim_mtp_layer_remap_is_idempotent() -> None:
    quant_config = SimpleNamespace(
        quant_description={
            "mtp.layers.48.self_attn.q_proj.weight": "W8A8_DYNAMIC",
        }
    )

    mtp._remap_modelslim_mtp_layer_ids(
        quant_config,
        mtp_start_layer_idx=48,
        num_mtp_layers=1,
    )

    assert "mtp.layers.48.self_attn.q_proj.weight" in quant_config.quant_description


@pytest.mark.parametrize("num_mtp_layers", [None, 1])
def test_draft_config_remaps_modelslim_description(monkeypatch, num_mtp_layers) -> None:
    draft_config = SimpleNamespace(
        quant_config=SimpleNamespace(
            quant_description={
                "mtp.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
            }
        )
    )
    target_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(mtp_num_hidden_layers=1),
        ),
        speculative_config=None,
    )
    calls = []

    def make_draft_release(config, start):
        calls.append((config, start))
        return draft_config

    def make_draft_main(config, start, layers):
        calls.append((config, start, layers))
        return draft_config

    monkeypatch.setattr(
        mtp,
        "_upstream_make_draft_vllm_config",
        make_draft_release if num_mtp_layers is None else make_draft_main,
    )

    args = (target_config, 48)
    if num_mtp_layers is not None:
        args += (num_mtp_layers,)
    actual = mtp._make_draft_vllm_config(*args)

    assert actual is draft_config
    assert calls == [args]
    assert draft_config.quant_config.quant_description == {
        "mtp.layers.48.self_attn.q_proj.weight": "W8A8_DYNAMIC",
    }


@pytest.mark.parametrize("v_mode", ["QK_INT8_DYNAMIC_V_INT8", "QK_INT8_DYNAMIC_V_INT8_PER_CHANNEL"])
def test_predictor_constructor_passes_remapped_c8_description_to_draft_layers(monkeypatch, v_mode) -> None:
    # Exercise the installed vLLM constructor and its real draft-config
    # call site. Stub model allocation so both supported ABIs run on CPU.
    upstream = mtp.upstream_mtp
    target_config, draft_model_config, _ = _make_long_context_config()
    text_config = draft_model_config.hf_text_config
    text_config.num_hidden_layers = 48
    text_config.mtp_num_hidden_layers = 2
    text_config.vocab_size = 16
    text_config.hidden_size = 8
    text_config.hc_count = 2
    text_config.hc_lowrank = 2
    text_config.rms_norm_eps = 1e-6
    target_config.parallel_config = SimpleNamespace(enable_eplb=False)
    target_config.compilation_config = CompilationConfig(mode=CompilationMode.NONE)
    quant_description = {
        "mtp.layers.0.self_attn.quant_type": v_mode,
        "mtp.layers.1.self_attn.indexer.quant_type": "QK_INT8_DYNAMIC",
    }
    quant_config = SimpleNamespace(quant_description=quant_description, get_name=lambda: "ascend")
    monkeypatch.setattr(upstream, "get_draft_quant_config", lambda config: quant_config)
    monkeypatch.setattr(upstream, "configure_quant_config", Mock())
    monkeypatch.setattr(upstream, "replace", lambda config, **changes: SimpleNamespace(**(vars(config) | changes)))
    monkeypatch.setattr(upstream, "set_current_vllm_config", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(upstream, "is_hc_sequence_parallel_enabled", lambda config: False)
    for name in ("VocabParallelEmbedding", "ColumnParallelLinear", "GemmaRMSNorm", "GatedResidual"):
        monkeypatch.setattr(upstream, name, Mock(side_effect=lambda *args, **kwargs: nn.Identity()))
    for name in ("get_fused_shared_expert_layers", "is_model_fused_shared_expert_compatible"):
        if hasattr(upstream, name):
            monkeypatch.setattr(upstream, name, Mock())
    decoder = Mock(side_effect=lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(upstream, "Qwen4ExpDecoderLayer", decoder)

    predictor = mtp.AscendQwen4ExpMultiTokenPredictor(vllm_config=target_config, prefix="mtp")

    assert len(predictor.layers) == 2
    assert [call.kwargs["prefix"] for call in decoder.call_args_list] == ["mtp.layers.48", "mtp.layers.49"]
    for call in decoder.call_args_list:
        assert call.args[0].quant_config.quant_description == {
            "mtp.layers.48.self_attn.quant_type": v_mode,
            "mtp.layers.49.self_attn.indexer.quant_type": "QK_INT8_DYNAMIC",
        }
        assert call.args[0].model_config is draft_model_config
    assert draft_model_config.max_model_len == 1010000


def test_position_cache_is_aligned_before_upstream_draft_config(monkeypatch) -> None:
    vllm_config, draft_model_config, _ = _make_long_context_config()
    draft_model_config.hf_text_config.mtp_num_hidden_layers = 1
    draft_config = SimpleNamespace(quant_config=None)

    def make_draft_config(_config, _start, num_mtp_layers):
        assert num_mtp_layers == 1
        assert draft_model_config.hf_text_config.max_position_embeddings == 1010000
        assert draft_model_config.max_model_len == 1010000
        return draft_config

    monkeypatch.setattr(
        mtp,
        "_upstream_make_draft_vllm_config",
        make_draft_config,
    )

    actual = mtp._make_draft_vllm_config(
        vllm_config,
        mtp_start_layer_idx=48,
        num_mtp_layers=1,
    )

    assert actual is draft_config


def test_text_target_strips_mrope_from_mtp_draft() -> None:
    target_model_config = _rope_model_config(uses_mrope=False)
    draft_model_config = _rope_model_config(uses_mrope=True)
    config = SimpleNamespace(
        model_config=target_model_config,
        speculative_config=SimpleNamespace(
            target_model_config=target_model_config,
            draft_model_config=draft_model_config,
        ),
    )

    mtp._align_draft_rope_with_target(config)

    assert draft_model_config.hf_text_config.rope_parameters == {}


def test_mrope_target_preserves_mrope_on_mtp_draft() -> None:
    target_model_config = _rope_model_config(uses_mrope=True)
    draft_model_config = _rope_model_config(uses_mrope=True)
    config = SimpleNamespace(
        model_config=target_model_config,
        speculative_config=SimpleNamespace(
            target_model_config=target_model_config,
            draft_model_config=draft_model_config,
        ),
    )

    mtp._align_draft_rope_with_target(config)

    assert draft_model_config.hf_text_config.rope_parameters == {
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": True,
    }
