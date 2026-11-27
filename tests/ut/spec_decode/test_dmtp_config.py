# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.spec_decode.speculator_config import get_custom_speculator_config_class

from vllm_ascend import register_model
from vllm_ascend.worker.v2.spec_decode.dmtp.config import AscendDMTPSpeculatorConfig, validate_dmtp_engine_config


def test_custom_method_registration_and_request_validation():
    from vllm_ascend.patch.platform import patch_dmtp_sampling  # noqa: F401

    register_model()
    assert get_custom_speculator_config_class("dmtp") is AscendDMTPSpeculatorConfig
    spec = SimpleNamespace(custom_method="dmtp", num_speculative_tokens=7)
    assert AscendDMTPSpeculatorConfig(spec).num_lookahead_tokens == 8
    SamplingParams(temperature=0)._validate_spec_decode(spec)
    with pytest.raises(VLLMValidationError, match="temperature=0"):
        SamplingParams(temperature=0.8)._validate_spec_decode(spec)
    SamplingParams(temperature=0.8)._validate_spec_decode(SimpleNamespace(custom_method=None))


def test_dmtp_checkpoint_cannot_fall_through_to_vanilla_dflash():
    draft = SimpleNamespace(architectures=["DFlash2T1ResidualDraftModel"])
    spec = SimpleNamespace(custom_method=None, draft_model_config=draft)
    with pytest.raises(ValueError, match="method='dmtp'"):
        validate_dmtp_engine_config(SimpleNamespace(speculative_config=spec))


@pytest.mark.parametrize("count", [1, 6, 8])
def test_rejects_mismatched_proposal_count(count):
    spec = SimpleNamespace(
        custom_method="dmtp",
        draft_model_config=SimpleNamespace(architectures=["DFlash2T1ResidualDraftModel"]),
        num_speculative_tokens=count,
    )
    with pytest.raises(ValueError, match="num_speculative_tokens=7"):
        AscendDMTPSpeculatorConfig.apply_options(spec)


def test_non_eager_requires_fa3_and_keeps_scheduling_guards(monkeypatch):
    from vllm_ascend.worker.v2.spec_decode.dmtp import config as module

    parallel = SimpleNamespace(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(custom_method="dmtp", draft_parallel_config=parallel),
        model_config=SimpleNamespace(enforce_eager=False, max_model_len=3072, multimodal_config=None),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        parallel_config=parallel,
        kv_transfer_config=None,
        lora_config=None,
    )
    monkeypatch.setattr(module, "find_spec", lambda _: None)
    with pytest.raises(ValueError, match="requires flash_attn_npu_3"):
        validate_dmtp_engine_config(config)
    config.model_config.enforce_eager = True
    validate_dmtp_engine_config(config)
    config.model_config.enforce_eager = False
    monkeypatch.setattr(module, "find_spec", lambda _: object())
    monkeypatch.setattr(
        module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(flash_attn_npu_3_models={module.FlashAttentionNpu3Model.QWEN}),
    )
    validate_dmtp_engine_config(config)
    config.scheduler_config.async_scheduling = True
    with pytest.raises(ValueError, match="--no-async-scheduling"):
        validate_dmtp_engine_config(config)
