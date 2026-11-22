# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform import patch_glm_moe_dsa_hnd as patch


@pytest.mark.parametrize(
    "model,method,layout,backend,allowed",
    [
        ("glm_moe_dsa", "dflash", "HND", "LBHNC", True),
        ("glm_moe_dsa", "dflash", "LBHNC", "LBHNC", True),
        ("glm_moe_dsa", "dflash", "HND", "BLHNC", False),
        ("glm5_next", "dflash", "HND", "LBHNC", False),
        ("glm_moe_dsa", "eagle3", "HND", "LBHNC", False),
    ],
)
def test_engine_layout_negotiation(monkeypatch, model, method, layout, backend, allowed):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model)),
        speculative_config=SimpleNamespace(method=method),
        cache_config=CacheConfig(),
        kv_transfer_config=None,
    )
    target = AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    draft = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, dcp_sharded=False, is_draft=True
    )
    monkeypatch.setattr(patch.envs, "VLLM_KV_CACHE_LAYOUT", layout)
    inputs = [target, draft]
    resolve = patch.engine_core.resolve_kv_cache_layout
    if allowed:
        assert resolve(config, [[backend]], inputs).name == "LBHNC"
    else:
        reason = "valid layouts" if backend == "BLHNC" else "requires a block-outer"
        with pytest.raises(ValueError, match=reason):
            resolve(config, [[backend]], inputs)
    assert inputs == [target, draft]
