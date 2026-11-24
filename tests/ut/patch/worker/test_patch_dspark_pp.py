# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config import get_current_vllm_config, set_current_vllm_config

from vllm_ascend.attention.attention_v1 import AscendKimiK31GQADraftAttentionBackend
from vllm_ascend.patch.worker import patch_qwen3_dflash
from vllm_ascend.patch.worker.patch_v2 import patch_dspark
from vllm_ascend.worker.v2 import pp_transport


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("fail", [True, False])
def test_dspark_draft_partition_isolation(monkeypatch, legacy, pp_size, fail):
    bypass_pp_guard = legacy and pp_size > 1
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(model="target", architecture="GlmMoeDsaForCausalLM"),
        speculative_config=SimpleNamespace(method="dspark", draft_model_config=SimpleNamespace(model="draft")),
    )
    monkeypatch.setattr(pp_transport, "use_legacy_spec_pp", lambda: legacy)
    monkeypatch.setattr(patch_dspark, "use_legacy_spec_pp", lambda: legacy)
    get_pp_group = lambda: SimpleNamespace(world_size=pp_size)
    monkeypatch.setattr(patch_dspark.dspark_utils, "get_pp_group", get_pp_group, raising=False)
    monkeypatch.setattr(pp_transport.vllm_envs, "VLLM_PP_LAYER_PARTITION", "42,36")
    should_share = patch_dspark.eagle_utils._should_share
    assert "_should_share" not in vars(patch_dspark.dspark_utils)

    def load(target, received_config):
        assert received_config is config
        assert config.parallel_config.pipeline_parallel_size == (1 if bypass_pp_guard else pp_size)
        expected_partition = None if pp_size > 1 else "42,36"
        assert expected_partition == pp_transport.vllm_envs.VLLM_PP_LAYER_PARTITION
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
    assert pp_transport.vllm_envs.VLLM_PP_LAYER_PARTITION == "42,36"
    assert patch_dspark.eagle_utils._should_share is should_share
    assert patch_dspark.dspark_utils.get_pp_group is get_pp_group
    assert "_should_share" not in vars(patch_dspark.dspark_utils)


def test_kimi_k31_dcp8_loads_gqa_draft_with_its_own_parallel_config(monkeypatch):
    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=8,
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=1664,
        ),
        model_config=SimpleNamespace(
            model="target",
            architecture="KimiK31ForConditionalGeneration",
            hf_config=SimpleNamespace(model_type="kimi_k31"),
        ),
        cache_config=SimpleNamespace(block_size=1664),
        compilation_config=object(),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=SimpleNamespace(model="draft", architectures=["Qwen3DSparkModel"])
        ),
    )
    monkeypatch.setattr(patch_qwen3_dflash, "get_dcp_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(patch_qwen3_dflash, "_original_dflash_attention", lambda **kwargs: kwargs)

    def load(target, draft_config):
        assert draft_config is not target_config
        assert draft_config.parallel_config.tensor_parallel_size == 8
        assert draft_config.parallel_config.decode_context_parallel_size == 1
        assert draft_config.parallel_config.cp_kv_cache_interleave_size == 1
        assert draft_config.compilation_config is target_config.compilation_config
        assert draft_config.cache_config is not target_config.cache_config
        with set_current_vllm_config(draft_config):
            return patch_qwen3_dflash.qwen3_dflash.Attention()

    monkeypatch.setattr(patch_dspark, "_original_load_dspark_model", load)
    with set_current_vllm_config(target_config):
        assert patch_qwen3_dflash.qwen3_dflash.Attention() == {}
        assert patch_dspark._load_dspark_model_with_target_quant(object(), target_config) == {
            "attn_backend": AscendKimiK31GQADraftAttentionBackend
        }
        assert get_current_vllm_config() is target_config
        assert target_config.parallel_config.decode_context_parallel_size == 8
        assert target_config.parallel_config.cp_kv_cache_interleave_size == 1664
