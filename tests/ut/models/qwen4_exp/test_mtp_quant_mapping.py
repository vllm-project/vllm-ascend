# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

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


def test_draft_config_remaps_modelslim_description(monkeypatch) -> None:
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
    monkeypatch.setattr(
        mtp,
        "_upstream_make_draft_vllm_config",
        lambda _config, _start: draft_config,
    )

    actual = mtp._make_draft_vllm_config(target_config, mtp_start_layer_idx=48)

    assert actual is draft_config
    assert draft_config.quant_config.quant_description == {
        "mtp.layers.48.self_attn.q_proj.weight": "W8A8_DYNAMIC",
    }


def test_position_cache_is_aligned_before_upstream_draft_config(monkeypatch) -> None:
    vllm_config, draft_model_config, _ = _make_long_context_config()
    draft_model_config.hf_text_config.mtp_num_hidden_layers = 1
    draft_config = SimpleNamespace(quant_config=None)

    def make_draft_config(_config, _start):
        assert draft_model_config.hf_text_config.max_position_embeddings == 1010000
        assert draft_model_config.max_model_len == 1010000
        return draft_config

    monkeypatch.setattr(
        mtp,
        "_upstream_make_draft_vllm_config",
        make_draft_config,
    )

    actual = mtp._make_draft_vllm_config(vllm_config, mtp_start_layer_idx=48)

    assert actual is draft_config
