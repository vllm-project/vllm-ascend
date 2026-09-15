# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.worker.patch_v2 import patch_dspark
from vllm_ascend.worker.v2 import pp_utils


@pytest.mark.parametrize("version", ["0.28.0", "0.29.0", "0.30.0"])
@pytest.mark.parametrize("fail", [True, False])
def test_dspark_draft_partition_isolation(monkeypatch, version, fail):
    legacy = version != "0.30.0"
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(model="target", architecture="GlmMoeDsaForCausalLM"),
        speculative_config=SimpleNamespace(method="dspark", draft_model_config=SimpleNamespace(model="draft")),
    )
    monkeypatch.setattr(pp_utils, "use_legacy_spec_pp", lambda: legacy)
    monkeypatch.setattr(patch_dspark, "use_legacy_spec_pp", lambda: legacy)
    monkeypatch.setattr(patch_dspark, "vllm_version_is", lambda target: target == version)
    get_pp_group = lambda: SimpleNamespace(world_size=2)
    monkeypatch.setattr(patch_dspark.dspark_utils, "get_pp_group", get_pp_group, raising=False)
    monkeypatch.setattr(pp_utils.vllm_envs, "VLLM_PP_LAYER_PARTITION", "42,36")
    should_share = patch_dspark.eagle_utils._should_share
    monkeypatch.setattr(patch_dspark.dspark_utils, "_should_share", should_share, raising=False)

    def load(target, received_config):
        assert received_config is config
        assert config.parallel_config.pipeline_parallel_size == (1 if legacy else 2)
        assert pp_utils.vllm_envs.VLLM_PP_LAYER_PARTITION is None
        assert patch_dspark.dspark_utils.get_pp_group().world_size == (1 if version == "0.28.0" else 2)
        if not legacy:
            assert patch_dspark.eagle_utils._should_share is should_share
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
    assert config.parallel_config.pipeline_parallel_size == 2
    assert pp_utils.vllm_envs.VLLM_PP_LAYER_PARTITION == "42,36"
    assert patch_dspark.eagle_utils._should_share is should_share
    assert patch_dspark.dspark_utils.get_pp_group is get_pp_group
    assert patch_dspark.dspark_utils._should_share is should_share
