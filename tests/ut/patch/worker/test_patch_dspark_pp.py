# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.worker.patch_v2 import patch_dspark
from vllm_ascend.worker.v2 import pp_utils


@pytest.mark.parametrize("version", ["0.29.0", "0.1.dev1+g84030bbe3.empty"])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("fail", [True, False])
def test_dspark_draft_partition_isolation(monkeypatch, version, pp_size, fail):
    legacy = version == "0.29.0"
    bypass_pp_guard = legacy and pp_size > 1
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(model="target", architecture="GlmMoeDsaForCausalLM"),
        speculative_config=SimpleNamespace(method="dspark", draft_model_config=SimpleNamespace(model="draft")),
    )
    monkeypatch.setattr(pp_utils, "use_legacy_spec_pp", lambda: legacy)
    monkeypatch.setattr(patch_dspark, "use_legacy_spec_pp", lambda: legacy)
    get_pp_group = lambda: SimpleNamespace(world_size=pp_size)
    monkeypatch.setattr(patch_dspark.dspark_utils, "get_pp_group", get_pp_group, raising=False)
    monkeypatch.setattr(pp_utils.vllm_envs, "VLLM_PP_LAYER_PARTITION", "42,36")
    should_share = patch_dspark.eagle_utils._should_share
    assert "_should_share" not in vars(patch_dspark.dspark_utils)

    def load(target, received_config):
        assert received_config is config
        assert config.parallel_config.pipeline_parallel_size == (1 if bypass_pp_guard else pp_size)
        expected_partition = None if pp_size > 1 else "42,36"
        assert expected_partition == pp_utils.vllm_envs.VLLM_PP_LAYER_PARTITION
        assert patch_dspark.dspark_utils.get_pp_group().world_size == (1 if bypass_pp_guard else pp_size)
        # Both upstream loaders resolve this helper inside load_dspark_model.
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

        assert _should_share is patch_dspark.eagle_utils._should_share
        assert "_should_share" not in vars(patch_dspark.dspark_utils)
        if bypass_pp_guard:
            assert _should_share is not should_share
        else:
            assert _should_share is should_share
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
    assert pp_utils.vllm_envs.VLLM_PP_LAYER_PARTITION == "42,36"
    assert patch_dspark.eagle_utils._should_share is should_share
    assert patch_dspark.dspark_utils.get_pp_group is get_pp_group
    assert "_should_share" not in vars(patch_dspark.dspark_utils)
