# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

from transformers import PretrainedConfig


def test_dspark_deepseek_v4_hf_config_override():
    repo_root = Path(__file__).parents[3]
    patch_module = runpy.run_path(str(repo_root / "vllm_ascend/patch/platform/patch_speculative_config.py"))

    hf_config = PretrainedConfig(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
    )

    patched = patch_module["hf_config_override"](hf_config)

    assert patched.model_type == "deepseek_mtp"
    assert patched.architectures == ["DeepSeekV4DSparkMTPModel"]
    assert patched.n_predict == 5
    assert patched.ptd_token_id == 128799
