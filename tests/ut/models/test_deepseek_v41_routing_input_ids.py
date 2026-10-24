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
    "sp,dp,pcp,microbatch,dsa_cp,comm,expected",
    [
        (True, 1, 1, False, False, MoECommType.ALLGATHER, True),
        (False, 1, 1, False, False, MoECommType.ALLGATHER, False),
        (True, 2, 1, False, False, MoECommType.ALLGATHER, False),
        (True, 1, 2, False, False, MoECommType.ALLGATHER, False),
        (True, 1, 1, True, False, MoECommType.ALLGATHER, False),
        (True, 1, 1, False, True, MoECommType.ALLGATHER, False),
        (True, 1, 1, False, False, MoECommType.MC2, False),
        (True, 1, 1, False, False, MoECommType.ALLTOALL, False),
        (True, 1, 1, False, False, MoECommType.FUSED_MC2, False),
    ],
)
def test_replicated_routing_ids_require_fixed_allgather_layout(
    monkeypatch, sp, dp, pcp, microbatch, dsa_cp, comm, expected
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=sp,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            use_ubatching=microbatch,
        )
    )
    monkeypatch.setattr(model_module, "enable_dsa_cp", lambda: dsa_cp)
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
    monkeypatch.setattr(model_module, "enable_dsa_cp", lambda: False)
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


@pytest.mark.parametrize("reuse_ids", [False, True])
def test_backbone_prepares_routing_ids_once_for_all_layers(monkeypatch, reuse_ids):
    monkeypatch.setattr(sp_module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(sp_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_module, "sp_shard", sp_module._ascend_sp_shard_impl)
    monkeypatch.setattr(model_module, "sp_all_gather", lambda x: torch.cat([x] * 8))
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", False)
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
        prepare.assert_called_once_with(input_ids)
        expected = torch.tensor([0, *range(11, 22), 0, 0, 0, 0], dtype=torch.int64)
    else:
        prepare.assert_not_called()
        expected = input_ids[:2]
    torch.testing.assert_close(seen_ids[0], expected, rtol=0, atol=0)
