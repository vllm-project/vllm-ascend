from types import SimpleNamespace
from unittest.mock import patch

from transformers import Qwen3Config
from vllm.config.speculative import SpeculativeConfig

import vllm_ascend.patch.platform.patch_speculative_config as patch_speculative_config


def test_ascend_speculative_config_field_defaults_to_false():
    field_name = "skip_parallel_drafting_seq_lens_override"

    assert SpeculativeConfig.__dataclass_fields__[field_name].default is False
    assert SpeculativeConfig.__pydantic_fields__[field_name].default is False


def test_seq_lens_override_warns_about_host_side_metadata():
    config = SimpleNamespace(
        skip_parallel_drafting_seq_lens_override=True,
        use_dspark=lambda: False,
    )

    with (
        patch.object(patch_speculative_config, "_orig_post_init"),
        patch.object(patch_speculative_config.logger, "warning_once") as warning,
    ):
        patch_speculative_config._dspark_post_init(config)

    warning.assert_called_once()
    assert "host-side sequence lengths" in warning.call_args.args[0]


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
