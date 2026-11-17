# SPDX-License-Identifier: Apache-2.0

from copy import copy, deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import Qwen3MoeConfig
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.config.parallel import ParallelConfig

from vllm_ascend import ascend_config
from vllm_ascend import logger as ascend_logger
from vllm_ascend.patch.worker import patch_dspark
from vllm_ascend.platform import _validate_eplb_config


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_invalid_target_is_still_rejected(phase):
    config = SimpleNamespace(
        additional_config={"eplb_config": {"load_collection_phase": phase}},
        parallel_config=SimpleNamespace(enable_eplb=False),
        speculative_config=object(),
        use_v2_model_runner=True,
    )
    with pytest.raises(ValueError, match="requires --enable-eplb"):
        _validate_eplb_config(config)


def test_draft_marker_restored_after_nested_failure():
    assert not ascend_config.is_draft_config_context()
    with ascend_config.draft_config_context():
        with pytest.raises(RuntimeError, match="draft failed"), ascend_config.draft_config_context():
            assert ascend_config.is_draft_config_context()
            raise RuntimeError("draft failed")
        assert ascend_config.is_draft_config_context()
    assert not ascend_config.is_draft_config_context()


@pytest.fixture
def dspark_config(tmp_path, monkeypatch, phase, draft_supports_eplb):
    from vllm.v1.worker.gpu.spec_decode import utils as spec_decode_utils

    # Scope platform initialization state and suppress log-file creation only.
    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", None)
    monkeypatch.setattr(ascend_config, "_INIT_VLLM_CONFIG", None)
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", VllmConfig.use_v2_model_runner)
    monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", VllmConfig._validate_v2_model_runner)
    monkeypatch.setattr(ascend_logger, "configure_ascend_file_logging", lambda: None)
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    # Config reconstruction runs before distributed worker initialization.
    monkeypatch.setattr(spec_decode_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    # A local config is sufficient: no checkpoint/tokenizer download or weights.
    Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=4,
        num_experts_per_tok=1,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True, max_model_len=128, dtype="bfloat16"
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config={"num_redundant_experts": 0, "use_async": True, "communicator": "torch_gloo"},
    )
    # Initialize the real SpeculativeConfig without downloading a draft, then
    # supply the resolved DSpark model/parallel metadata used by the loader.
    speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
        target_model_config=model_config,
        target_parallel_config=parallel_config,
    )
    speculative_config.method = "dspark"
    draft_model_config = copy(model_config)
    draft_model_config.hf_config = deepcopy(model_config.hf_config)
    draft_model_config.hf_config.model_type = "deepseek_v4" if draft_supports_eplb else "qwen3_moe"
    speculative_config.draft_model_config = draft_model_config
    speculative_config.draft_parallel_config = parallel_config
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        additional_config={"eplb_config": {"load_collection_phase": phase, "stair_config": {"load_window_bins": 32}}},
    )


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("draft_supports_eplb", [False, True])
def test_real_vllm_config_draft_reconstruction(monkeypatch, dspark_config, phase, draft_supports_eplb):
    from vllm.model_executor import model_loader

    config = dspark_config
    original_additional_config = deepcopy(config.additional_config)
    get_model = Mock(side_effect=RuntimeError("reached model construction"))
    monkeypatch.setattr(model_loader, "get_model", get_model)
    # Keep the real upstream loader and VllmConfig validation; stop before weights.
    if not draft_supports_eplb:
        with pytest.raises(ValueError, match="requires --enable-eplb"):
            patch_dspark._original_load_dspark_model(object(), config)
        get_model.assert_not_called()

    get_parallel_config = Mock(wraps=patch_dspark.dspark_utils._get_dspark_parallel_config)
    monkeypatch.setattr(patch_dspark.dspark_utils, "_get_dspark_parallel_config", get_parallel_config)
    with pytest.raises(RuntimeError, match="reached model construction"):
        patch_dspark._load_dspark_model_with_target_quant(object(), config)
    get_parallel_config.assert_called_once()
    get_model.assert_called_once()
    draft = get_model.call_args.kwargs["vllm_config"]
    assert get_model.call_args.kwargs["model_config"] is config.speculative_config.draft_model_config
    assert isinstance(draft, VllmConfig)
    assert draft is not config
    assert draft.parallel_config.enable_eplb is draft_supports_eplb
    assert draft.additional_config == config.additional_config == original_additional_config
    assert ascend_config.get_ascend_config().eplb_config.load_collection_phase == phase
    assert config.parallel_config.enable_eplb
