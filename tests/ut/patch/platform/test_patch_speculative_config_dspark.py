from types import SimpleNamespace

import pytest
from transformers import Qwen3Config
from vllm.config.speculative import SpeculativeConfig

from vllm_ascend.patch.platform import patch_speculative_config


def test_legacy_qwen3_dspark_config_uses_qwen3_loader():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
    )

    normalized = SpeculativeConfig.hf_config_override(config)

    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]
    assert normalized.block_size == 7


def _make_k3_dspark_config(dcp_size: int = 8):
    draft_hf_config = SimpleNamespace(
        ptd_token_id=163839,
        dspark_noise_token_id=163839,
        mask_token_id=None,
    )
    return SimpleNamespace(
        method="dspark",
        target_parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
        ),
        draft_model_config=SimpleNamespace(
            architectures=["K3DSparkModel"],
            hf_config=draft_hf_config,
        ),
        use_dspark=lambda: True,
    )


def test_k3_dspark_dcp_is_hidden_only_during_upstream_validation(monkeypatch):
    config = _make_k3_dspark_config()
    observed_dcp_sizes = []

    def validate_without_gpu_guard(candidate):
        observed_dcp_sizes.append(
            candidate.target_parallel_config.decode_context_parallel_size
        )

    monkeypatch.setattr(
        patch_speculative_config, "_orig_post_init", validate_without_gpu_guard
    )

    patch_speculative_config._dspark_post_init(config)

    assert observed_dcp_sizes == [1]
    assert config.target_parallel_config.decode_context_parallel_size == 8


@pytest.mark.parametrize(
    ("dcp_size", "message"),
    [
        (1, "upstream speculative config error"),
        (8, "some other speculative config error"),
    ],
)
def test_k3_dspark_dcp_restores_config_and_propagates_validation_errors(
    monkeypatch, dcp_size, message
):
    config = _make_k3_dspark_config(dcp_size=dcp_size)
    observed_dcp_sizes = []

    def raise_validation_error(candidate):
        observed_dcp_sizes.append(
            candidate.target_parallel_config.decode_context_parallel_size
        )
        raise ValueError(message)

    monkeypatch.setattr(
        patch_speculative_config, "_orig_post_init", raise_validation_error
    )

    with pytest.raises(ValueError, match=message):
        patch_speculative_config._dspark_post_init(config)

    assert observed_dcp_sizes == [1]
    assert config.target_parallel_config.decode_context_parallel_size == dcp_size
