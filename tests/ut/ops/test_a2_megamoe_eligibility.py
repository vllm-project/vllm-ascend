# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only eligibility and routing tests for A2 (910B3) MegaMoe.

The A2 capability is permanently advertised by the hardware profile and the
feature reuses the A3/A5 ``enable_fused_mc2`` enablement chain. A dedicated
predicate guards the only validated envelope (DeepSeek-V4.1-Flash W4A8
prefill, DP1/TP1/PCP8/EP8, eager, BF16); every other topology downgrades to
the ordinary capacity-and-expert-density MoE route. Hardware profiles are
real objects, runtime gates are monkeypatched at module level.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    is_draft_model,
    use_cann_megamoe,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    MoECommPolicy,
    get_hardware_profile,
)

A2_PROFILE = get_hardware_profile(AscendDeviceType.A2)
A3_PROFILE = get_hardware_profile(AscendDeviceType.A3)
A5_PROFILE = get_hardware_profile(AscendDeviceType.A5)


def _a2_target_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["DeepseekV41ForCausalLM"],
            runner_type="generate",
            enforce_eager=True,
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                hidden_size=5120,
                moe_intermediate_size=2304,
                num_experts_per_tok=6,
            ),
            get_num_experts=lambda: 384,
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            data_parallel_size=1,
            tensor_parallel_size=1,
            prefill_context_parallel_size=8,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
            use_ubatching=False,
            expert_placement_strategy="linear",
            world_size_across_dp=8,
        ),
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
        use_v2_model_runner=True,
        lora_config=None,
    )


@pytest.fixture
def a2_hardware(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_current_hardware_profile",
        lambda: A2_PROFILE,
    )


def test_a2_profile_advertises_megamoe_with_capacity_density_policy():
    assert A2_PROFILE.supports(HardwareCapability.CANN_MEGAMOE)
    assert A2_PROFILE.moe_comm_policy is MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY
    # Routing dispatch needs both signals: A3/A5 share the capability but own
    # different MoE communication policies.
    for profile in (A3_PROFILE, A5_PROFILE):
        assert profile.supports(HardwareCapability.CANN_MEGAMOE)
        assert profile.moe_comm_policy is not MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY


def test_a2_target_envelope_passes(a2_hardware):
    assert AscendConfig._is_megamoe_supported_by_config(_a2_target_config())


def test_a2_static_predicate_requires_dsv41_architecture():
    cfg = _a2_target_config()
    assert AscendConfig._is_a2_megamoe_supported_by_config(cfg, ["DeepseekV41ForCausalLM"])
    assert not AscendConfig._is_a2_megamoe_supported_by_config(cfg, ["Qwen4MoeForCausalLM"])


def test_none_model_config_returns_false(a2_hardware):
    cfg = SimpleNamespace(model_config=None, parallel_config=SimpleNamespace())
    assert not AscendConfig._is_megamoe_supported_by_config(cfg)


def test_non_dsv41_architecture_rejected_under_a2_policy(a2_hardware):
    cfg = _a2_target_config()
    cfg.model_config.architectures = ["Qwen4MoeForCausalLM"]
    assert not AscendConfig._is_megamoe_supported_by_config(cfg)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: setattr(cfg.parallel_config, "enable_expert_parallel", False),
        lambda cfg: setattr(cfg.parallel_config, "data_parallel_size", 2),
        lambda cfg: setattr(cfg.parallel_config, "tensor_parallel_size", 2),
        lambda cfg: setattr(cfg.parallel_config, "prefill_context_parallel_size", 4),
        lambda cfg: setattr(cfg.parallel_config, "pipeline_parallel_size", 2),
        lambda cfg: setattr(cfg.parallel_config, "decode_context_parallel_size", 2),
        lambda cfg: setattr(cfg.parallel_config, "use_ubatching", True),
        lambda cfg: setattr(cfg.parallel_config, "expert_placement_strategy", "expert"),
        lambda cfg: setattr(cfg.parallel_config, "enable_eplb", True),
        lambda cfg: setattr(cfg.model_config, "enforce_eager", False),
        lambda cfg: setattr(cfg.model_config, "dtype", torch.float16),
        lambda cfg: setattr(cfg.model_config.hf_text_config, "hidden_size", 2560),
        lambda cfg: setattr(cfg.model_config.hf_text_config, "moe_intermediate_size", 640),
        lambda cfg: setattr(cfg.model_config.hf_text_config, "num_experts_per_tok", 10),
        lambda cfg: setattr(cfg.model_config, "get_num_experts", lambda: 256),
        lambda cfg: setattr(cfg, "kv_transfer_config", SimpleNamespace(kv_role="kv_consumer")),
        lambda cfg: setattr(cfg, "kv_transfer_config", None),
        lambda cfg: setattr(cfg, "use_v2_model_runner", False),
        lambda cfg: setattr(cfg, "lora_config", SimpleNamespace()),
    ],
)
def test_a2_predicate_rejects_envelope_violations(a2_hardware, mutate):
    cfg = _a2_target_config()
    mutate(cfg)
    assert not AscendConfig._is_megamoe_supported_by_config(cfg)


def test_megamoe_dispatch_requires_both_architecture_and_a2_policy(a2_hardware):
    cfg = _a2_target_config()
    cfg.parallel_config.prefill_context_parallel_size = 4
    assert not AscendConfig._is_megamoe_supported_by_config(cfg)
    with patch(
        "vllm_ascend.ascend_config.get_current_hardware_profile",
        return_value=A3_PROFILE,
    ):
        assert AscendConfig._is_megamoe_supported_by_config(cfg)


@pytest.fixture
def a2_gating(monkeypatch):
    def _patch(
        *,
        enable_fused_mc2=1,
        mega_supported=True,
        ep_world_size=8,
        is_moe=True,
        profile=A2_PROFILE,
    ):
        monkeypatch.setattr(afc, "is_moe_model", lambda _: is_moe)
        monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: mega_supported)
        monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: profile)
        monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
        monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 128)
        monkeypatch.setattr(
            afc,
            "get_ascend_config",
            lambda: SimpleNamespace(enable_fused_mc2=enable_fused_mc2),
        )

    return _patch


@pytest.mark.parametrize("num_tokens", [None, 0, 1, 128, 129, 4096])
def test_a2_target_always_short_circuits_to_fused_mc2(a2_gating, num_tokens):
    a2_gating()
    cfg = _a2_target_config()
    assert use_cann_megamoe(cfg)
    assert afc.select_moe_comm_method(num_tokens, cfg) is MoECommType.FUSED_MC2


def test_a2_without_fused_mc2_keeps_capacity_density_route(a2_gating):
    a2_gating(enable_fused_mc2=0)
    cfg = _a2_target_config()
    assert not use_cann_megamoe(cfg)
    # 384 experts / EP8 = 48 experts per device never satisfies the <=24
    # MC2 density rule, so every batch width falls back to ALLGATHER.
    for num_tokens in (0, 128, 4096):
        assert afc.select_moe_comm_method(num_tokens, cfg) is MoECommType.ALLGATHER


def test_a2_density_route_still_prefers_mc2_for_dense_expert_layout(a2_gating):
    a2_gating(enable_fused_mc2=0, ep_world_size=16)
    cfg = _a2_target_config()
    cfg.parallel_config.world_size_across_dp = 16
    cfg.model_config.get_num_experts = lambda: 128
    assert afc.select_moe_comm_method(128, cfg) is MoECommType.MC2
    assert afc.select_moe_comm_method(129, cfg) is MoECommType.ALLGATHER


def test_draft_identities_detected_while_target_is_not(a2_gating):
    a2_gating()
    assert not is_draft_model(_a2_target_config())

    by_runner = _a2_target_config()
    by_runner.model_config.runner_type = "draft"
    by_runner.parallel_config.prefill_context_parallel_size = 1
    assert is_draft_model(by_runner)

    by_arch = _a2_target_config()
    by_arch.model_config.architectures = ["DeepseekV41DSparkModel"]
    by_arch.parallel_config.prefill_context_parallel_size = 1
    assert is_draft_model(by_arch)

    # The nested DSpark draft is rebuilt with PCP=1 before its architecture is
    # rewritten, so the PCP signature plus an active MegaMoe chain identifies it.
    by_pcp = _a2_target_config()
    by_pcp.parallel_config.prefill_context_parallel_size = 1
    assert is_draft_model(by_pcp)


def test_draft_keeps_ordinary_route_under_a2_policy(a2_gating):
    a2_gating()
    draft = _a2_target_config()
    draft.model_config.runner_type = "draft"
    draft.parallel_config.prefill_context_parallel_size = 1
    assert is_draft_model(draft)
    assert not use_cann_megamoe(draft)
    assert afc.select_moe_comm_method(4096, draft) is MoECommType.ALLGATHER


def test_draft_predicate_tolerates_partial_namespaces(a2_gating):
    a2_gating()
    sparse_parallel = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["DeepseekV41ForCausalLM"],
            runner_type="generate",
        ),
        parallel_config=SimpleNamespace(),
    )
    # Missing PCP attribute defaults to the PCP1 draft signature.
    assert is_draft_model(sparse_parallel)
    missing_parallel = SimpleNamespace(model_config=sparse_parallel.model_config)
    assert not is_draft_model(missing_parallel)


def test_megamoe_disabled_when_operator_unavailable(a2_gating):
    a2_gating(mega_supported=False)
    assert not use_cann_megamoe(_a2_target_config())


def test_megamoe_requires_ep_size_above_one(a2_gating):
    a2_gating(ep_world_size=1)
    assert not use_cann_megamoe(_a2_target_config())


def test_lora_config_disables_megamoe(a2_gating):
    a2_gating()
    cfg = _a2_target_config()
    cfg.lora_config = SimpleNamespace()
    assert not use_cann_megamoe(cfg)


def test_draft_exclusion_is_scoped_to_a2_policy(a2_gating):
    a2_gating(profile=A3_PROFILE)
    draft = _a2_target_config()
    draft.model_config.runner_type = "draft"
    draft.parallel_config.prefill_context_parallel_size = 1
    # A3 owns a different MoE policy, so the A2 draft exclusion never fires.
    assert use_cann_megamoe(draft)
