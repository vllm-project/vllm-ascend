from types import SimpleNamespace

import pytest
from transformers import Qwen3Config
from vllm.config.speculative import SpeculativeConfig

from vllm_ascend.patch.platform import patch_speculative_config

_UPSTREAM_K3_DSPARK_DCP_ERROR = (
    patch_speculative_config._UPSTREAM_K3_DSPARK_DCP_ERROR_FRAGMENT
    + "; set decode_context_parallel_size=1."
)


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


def test_k3_dspark_dcp_bypasses_upstream_gpu_guard(monkeypatch):
    config = _make_k3_dspark_config()

    def raise_upstream_guard(_config):
        raise ValueError(_UPSTREAM_K3_DSPARK_DCP_ERROR)

    monkeypatch.setattr(
        patch_speculative_config, "_orig_post_init", raise_upstream_guard
    )

    patch_speculative_config._dspark_post_init(config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            _make_k3_dspark_config(dcp_size=1),
            _UPSTREAM_K3_DSPARK_DCP_ERROR,
        ),
        (_make_k3_dspark_config(), "some other speculative config error"),
    ],
)
def test_k3_dspark_dcp_does_not_hide_other_validation_errors(
    monkeypatch, config, message
):
    def raise_validation_error(_config):
        raise ValueError(message)

    monkeypatch.setattr(
        patch_speculative_config, "_orig_post_init", raise_validation_error
    )

    with pytest.raises(ValueError, match=message):
        patch_speculative_config._dspark_post_init(config)
