# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import vllm.envs as vllm_envs
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.patch.worker.patch_v2 import patch_dspark


@pytest.mark.parametrize("inherits_quant", [True, False])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("fail", [True, False])
def test_dspark_draft_partition_isolation(monkeypatch, inherits_quant, pp_size, fail):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(model="target", architecture="GlmMoeDsaForCausalLM"),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=SimpleNamespace(model="target" if inherits_quant else "draft")
        ),
        quant_config=object(),
    )
    get_pp_group = lambda: SimpleNamespace(world_size=pp_size)
    monkeypatch.setattr(patch_dspark.dspark_utils, "get_pp_group", get_pp_group, raising=False)
    monkeypatch.setattr(vllm_envs, "VLLM_PP_LAYER_PARTITION", "42,36")
    should_share = eagle_utils._should_share
    get_quant_config = patch_dspark.model_utils.get_draft_quant_config
    assert "_should_share" not in vars(patch_dspark.dspark_utils)

    def load(target, received_config):
        assert received_config is config
        assert config.parallel_config.pipeline_parallel_size == pp_size
        expected_partition = None if pp_size > 1 else "42,36"
        assert expected_partition == vllm_envs.VLLM_PP_LAYER_PARTITION
        assert patch_dspark.dspark_utils.get_pp_group().world_size == pp_size
        # Native PP loading and sharing are not overridden.
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

        assert _should_share is should_share
        assert "_should_share" not in vars(patch_dspark.dspark_utils)
        if inherits_quant:
            assert patch_dspark.model_utils.get_draft_quant_config(config) is config.quant_config
        else:
            assert patch_dspark.model_utils.get_draft_quant_config is get_quant_config
        if fail:
            raise RuntimeError("draft load failed")
        return target

    monkeypatch.setattr(patch_dspark, "_original_load_dspark_model", load)
    target = object()
    if fail:
        with pytest.raises(RuntimeError, match="draft load failed"):
            patch_dspark._load_dspark_model_with_target_quant(target, config)
    else:
        assert patch_dspark._load_dspark_model_with_target_quant(target, config) is target
    assert config.parallel_config.pipeline_parallel_size == pp_size
    assert vllm_envs.VLLM_PP_LAYER_PARTITION == "42,36"
    assert eagle_utils._should_share is should_share
    assert patch_dspark.model_utils.get_draft_quant_config is get_quant_config
    assert patch_dspark.dspark_utils.get_pp_group is get_pp_group
    assert "_should_share" not in vars(patch_dspark.dspark_utils)
