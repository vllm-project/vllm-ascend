# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.models.qwen4_exp import mtp


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
        )
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
