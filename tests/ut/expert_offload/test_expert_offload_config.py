# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.expert_offload.config import ExpertOffloadConfig


@pytest.fixture(autouse=True)
def a5_hardware(monkeypatch):
    profile = SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED)
    monkeypatch.setattr("vllm_ascend.expert_offload.config.get_current_hardware_profile", lambda: profile)


def make_configs():
    vc = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            model="/models/target",
            enforce_eager=True,
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(model_type="other_moe"),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=4),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0), prefetch=SimpleNamespace(offload_group_size=0)
        ),
        load_config=SimpleNamespace(load_format="auto"),
    )
    ac = SimpleNamespace(
        mix_placement=False,
        enable_force_eplb=False,
        enable_fused_mc2=0,
        multistream_overlap_shared_expert=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        rl_config=SimpleNamespace(enabled=False),
    )
    return vc, ac


def test_default_off_and_explicit_enable():
    disabled = ExpertOffloadConfig()
    assert not disabled.enabled and disabled.num_device_experts is None
    disabled.validate(None, None)
    config = TypeAdapter(ExpertOffloadConfig).validate_python({"enabled": True, "num_device_experts": 4})
    config.validate(*make_configs())
    config.validate_layer(8, 2)


@pytest.mark.parametrize(
    "method,async_scheduling", [(None, True), ("dspark", True), ("mtp", True), ("deepseek_mtp", False)]
)
def test_async_multi_request_and_checkpoint_drafting(method, async_scheduling):
    vc, ac = make_configs()
    vc.scheduler_config.async_scheduling = async_scheduling
    if method is not None:
        dspark = method == "dspark"
        vc.model_config.hf_text_config.model_type = "deepseek_v4" if dspark else "glm5_next_text"
        vc.speculative_config = SimpleNamespace(
            method=method,
            enforce_eager=True,
            enable_adaptive_verification=False,
            draft_model_config=SimpleNamespace(
                model=vc.model_config.model,
                hf_config=SimpleNamespace(
                    model_type="deepseek_v4" if dspark else "glm5_next_mtp",
                    architectures=["DSparkDraftModel" if dspark else "Glm5NextMTPModel"],
                ),
            ),
        )
    ExpertOffloadConfig(enabled=True, num_device_experts=68).validate(vc, ac)


def test_full_decode_graph_configuration():
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    vc, ac = make_configs()
    vc.model_config.enforce_eager = False
    vc.compilation_config = SimpleNamespace(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    ExpertOffloadConfig(enabled=True, num_device_experts=68).validate(vc, ac)
