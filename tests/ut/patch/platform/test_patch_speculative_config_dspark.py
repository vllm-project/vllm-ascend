from types import SimpleNamespace

import pytest
from transformers import Qwen3Config
from vllm.config import SpeculativeConfig

from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.patch.platform.patch_speculative_config import (
    _dspark_post_init,
    hf_config_override,
)


def test_legacy_qwen3_dspark_config_is_normalized_before_model_inspection():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
        pad_token_id=163839,
    )

    normalized = hf_config_override(config)

    assert SpeculativeConfig.hf_config_override is hf_config_override
    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]
    assert normalized.block_size == 7
    assert normalized.pad_token_id == 163839


@pytest.mark.parametrize(
    ("block_size", "num_speculative_tokens"),
    [(7, 1), (7, 3), (7, 8), (5, 7)],
)
def test_qwen3_dspark_requires_num_speculative_tokens_to_match_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    num_speculative_tokens: int,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    draft_hf_config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        block_size=block_size,
        mask_token_id=163824,
        ptd_token_id=None,
    )
    config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=num_speculative_tokens,
    )

    with pytest.raises(ValueError, match=r"trained block_size"):
        _dspark_post_init(config)


@pytest.mark.parametrize("block_size", [5, 7])
def test_qwen3_dspark_accepts_checkpoint_block_size(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    draft_hf_config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        block_size=block_size,
        mask_token_id=163824,
        ptd_token_id=None,
    )
    config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=block_size,
    )

    _dspark_post_init(config)

    assert draft_hf_config.ptd_token_id == 163824


@pytest.mark.parametrize("block_size", [None, 0, -1, "7", True])
def test_qwen3_dspark_requires_positive_integer_checkpoint_block_size(
    monkeypatch: pytest.MonkeyPatch,
    block_size,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    draft_hf_config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        block_size=block_size,
        mask_token_id=163824,
        ptd_token_id=None,
    )
    config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=7,
    )

    with pytest.raises(ValueError, match=r"positive integer block_size"):
        _dspark_post_init(config)
