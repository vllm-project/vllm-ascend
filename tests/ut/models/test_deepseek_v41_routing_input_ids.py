# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import ascend_forward_context as context_module
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware_profile import MoECommPolicy
from vllm_ascend.models.common.ops import sequence_parallel as sp_module
from vllm_ascend.models.deepseek_v41 import model as model_module


@pytest.mark.parametrize(
    "sp,dp,pcp,microbatch,comm,expected",
    [
        (True, 1, 1, False, MoECommType.ALLGATHER, True),
        (False, 1, 1, False, MoECommType.ALLGATHER, False),
        (True, 2, 1, False, MoECommType.ALLGATHER, False),
        (True, 1, 2, False, MoECommType.ALLGATHER, False),
        (True, 1, 1, True, MoECommType.ALLGATHER, False),
        (True, 1, 1, False, MoECommType.MC2, False),
        (True, 1, 1, False, MoECommType.ALLTOALL, False),
        (True, 1, 1, False, MoECommType.FUSED_MC2, False),
    ],
)
def test_replicated_routing_ids_require_fixed_allgather_layout(monkeypatch, sp, dp, pcp, microbatch, comm, expected):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=sp,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            use_ubatching=microbatch,
        )
    )
    monkeypatch.setattr(model_module, "select_moe_comm_method", lambda num_tokens, config: comm)
    assert model_module._can_reuse_routing_input_ids(config) is expected


@pytest.mark.parametrize(
    "ep_size,experts,expected", [(8, 384, True), (16, 384, False), (16, 512, True), (32, 384, False)]
)
def test_a2_reuse_policy_is_constant_across_prefill_and_decode(monkeypatch, ep_size, experts, expected):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=True,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
            enable_expert_parallel=True,
            world_size_across_dp=ep_size,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(get_num_experts=lambda: experts),
        lora_config=None,
    )
    monkeypatch.setattr(context_module, "is_moe_model", lambda config: True)
    monkeypatch.setattr(context_module, "get_mc2_tokens_capacity", lambda: 64)
    monkeypatch.setattr(context_module, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))
    monkeypatch.setattr(
        context_module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(moe_comm_policy=MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY),
    )
    assert model_module._can_reuse_routing_input_ids(config) is expected
    if expected:
        for num_tokens in (1, 12, 50, 4096):
            assert context_module.select_moe_comm_method(num_tokens, config) == MoECommType.ALLGATHER


@pytest.mark.parametrize("reuse_ids,skip_padding", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("num_valid_tokens", [0, 3, 12])
@pytest.mark.parametrize("shard_padding_mask", [False, True])
def test_backbone_prepares_routing_ids_once_for_all_layers(
    monkeypatch, reuse_ids, skip_padding, tp_size, num_valid_tokens, shard_padding_mask
):
    monkeypatch.setattr(sp_module, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(sp_module, "get_tensor_model_parallel_rank", lambda: tp_size - 1)
    monkeypatch.setattr(model_module, "sp_shard", sp_module._ascend_sp_shard_impl)
    monkeypatch.setattr(model_module, "sp_all_gather", lambda x: torch.cat([x] * tp_size))
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", shard_padding_mask)
    context = SimpleNamespace(is_padding=torch.arange(12) >= num_valid_tokens)
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        model_module, "sp_padding_mask", lambda mask, hidden: sp_module._ascend_sp_padding_mask_impl(mask)
    )
    prepare = MagicMock(side_effect=sp_module._prepare_sp_routing_input_ids_impl)
    monkeypatch.setattr(model_module, "prepare_sp_routing_input_ids", prepare)
    seen_ids = []

    class Layer:
        engram = None

        def __init__(self, index):
            self.layer_idx = index

        def __call__(self, positions, hidden, pre_mix, scaling, input_ids):
            seen_ids.append(input_ids)
            return hidden, pre_mix

        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden[:, 0]

    model = SimpleNamespace(
        use_sequence_parallel=True,
        reuse_routing_input_ids=reuse_ids,
        skip_moe_padding=skip_padding,
        hc_mult=4,
        _mtp_hidden_buffer=None,
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        decoder_replay_layers=None,
        aux_hidden_state_layers=(),
        layers=[Layer(i) for i in range(40)],
        norm=lambda x: x,
    )
    input_ids = torch.arange(12, dtype=torch.int32) + 10
    input_ids[0] = -1
    input_ids[1] = 0
    original_ids = input_ids.clone()
    model_module.DeepseekV41Model.forward(
        model,
        input_ids,
        torch.arange(12),
        None,
        inputs_embeds=torch.zeros(12, 4),
        engram_lookups={},
        engram_mask=torch.empty(0, dtype=torch.bool),
    )

    assert len(seen_ids) == 40
    assert all(ids is seen_ids[0] for ids in seen_ids)
    if reuse_ids:
        prepare.assert_called_once()
        assert prepare.call_args.kwargs == {"padding_value": -2 if skip_padding else 0}
        padded_size = (12 + tp_size - 1) // tp_size * tp_size
        expected = torch.full((padded_size,), -2 if skip_padding else 0, dtype=torch.int64)
        expected[:12] = input_ids
        expected[expected == -1] = 0
        if skip_padding:
            expected[num_valid_tokens:] = -2
    else:
        prepare.assert_not_called()
        expected = sp_module._ascend_sp_shard_impl(input_ids)
    torch.testing.assert_close(seen_ids[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(input_ids, original_ids, rtol=0, atol=0)


@pytest.mark.parametrize(
    "section,option,value",
    [
        (None, None, None),
        ("runner", "v2", False),
        ("parallel", "use_sequence_parallel_moe", False),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "use_ubatching", True),
        ("parallel", "enable_expert_parallel", False),
        ("parallel", "tensor_parallel_size", 1),
        ("parallel", "expert_placement_strategy", "round_robin"),
        ("parallel", "enable_eplb", True),
        ("ascend", "mix_placement", True),
        ("ascend", "enable_force_eplb", True),
        ("speculative", "enable_adaptive_verification", True),
        ("communication", "method", MoECommType.MC2),
        ("communication", "method", MoECommType.FUSED_MC2),
        ("communication", "method", MoECommType.ALLTOALL),
    ],
)
def test_skip_moe_padding_requires_supported_layout(monkeypatch, section, option, value):
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=True,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        use_ubatching=False,
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        expert_placement_strategy="linear",
        enable_eplb=False,
    )
    ascend = SimpleNamespace(mix_placement=False, enable_force_eplb=False)
    speculative = SimpleNamespace(enable_adaptive_verification=False)
    if section in ("parallel", "ascend", "speculative"):
        setattr({"parallel": parallel, "ascend": ascend, "speculative": speculative}[section], option, value)
    config = SimpleNamespace(parallel_config=parallel, speculative_config=speculative)
    monkeypatch.setattr(model_module, "use_v2_model_runner", lambda config: section != "runner")
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(
        model_module,
        "select_moe_comm_method",
        lambda num_tokens, config: value if section == "communication" else MoECommType.ALLGATHER,
    )
    assert model_module._can_skip_moe_padding(config) is (section is None)
    if section is None:
        config.speculative_config = None
        assert model_module._can_skip_moe_padding(config)
