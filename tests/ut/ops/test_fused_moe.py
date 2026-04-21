#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from pytest_mock import MockerFixture

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.fused_moe.moe_mlp import cumsum_group_list, unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEPrepareOutput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, adapt_patch

adapt_patch(True)


def mock_ep_and_mc2_group(mocker):
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.rank = 0
    mock_group.world_size = 4
    mock_group.device_group = "mock_group_ep"
    mock_group.all_to_all = MagicMock(return_value=torch.randn(8, 8))
    return mock_group


def mock_dp_and_tp_group(mocker):
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 2
    mock_group.device_group = "mock_group"
    mock_group.all_gather = MagicMock(return_value=torch.randn(10, 32))
    return mock_group


def mock_npu_format_cast(weight_data, format):
    return weight_data


def build_mlp_compute_input_fixture(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    group_list: torch.Tensor,
    with_quant: bool,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor | None = None,
    topk_scales: torch.Tensor | None = None,
    w1_scale: torch.Tensor | list[torch.Tensor] | None = None,
    w2_scale: torch.Tensor | list[torch.Tensor] | None = None,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    activation: str = "silu",
    need_trans: bool = True,
    dynamic_eplb: bool = False,
) -> MoEMlpComputeInput:
    return MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=group_list_type,
        dynamic_scale=dynamic_scale,
        topk_scales=topk_scales,
        weights=MoEWeights(
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8 if with_quant else QuantType.NONE),
        fusion=fusion,
        activation=activation,
        need_trans=need_trans,
        dynamic_eplb=dynamic_eplb,
    )


@pytest.fixture(autouse=True)
def setup_vllm_config_mock(mocker: MockerFixture):
    mock_hf_config = MagicMock()
    mock_hf_config.model_type = "llama"

    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config

    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_vllm_config.parallel_config = MagicMock(tensor_parallel_size=2)
    mock_vllm_config.scheduler_config = MagicMock(max_num_seqs=4)
    mock_vllm_config.model_config.max_model_len = 2048

    mocker.patch("vllm_ascend.ops.fused_moe.fused_moe.get_current_vllm_config", return_value=mock_vllm_config)


@pytest.fixture
def mock_dist_env(mocker: MockerFixture):
    mock_moe_comm_method = MagicMock()

    def mock_prepare(hidden_states, router_logits, **kwargs):
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=kwargs.get("mc2_mask"),
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    mock_moe_comm_method.prepare.side_effect = mock_prepare

    mock_fused_experts_result = torch.randn(16, 2)
    mock_moe_comm_method.fused_experts.return_value = mock_fused_experts_result

    def mock_finalize(hidden_states, **kwargs):
        return hidden_states

    mock_moe_comm_method.finalize.side_effect = mock_finalize
    dp_metadata = MagicMock(num_tokens_across_dp_cpu=[5, 5])
    mock_weight_prefetch_method = MagicMock()
    mock_forward_context_obj = MagicMock(
        moe_comm_method=mock_moe_comm_method,
        moe_comm_type=MoECommType.MC2,
        max_tokens_across_dp=10,
        dp_metadata=dp_metadata,
        mc2_mask=torch.zeros(16, dtype=torch.bool),
        padded_num_tokens=16,
        with_quant=False,
    )

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_ep_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_mc2_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_tp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.model_executor.layers.fused_moe.layer.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.model_executor.layers.fused_moe.config.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config",
            return_value=MagicMock(enable_multistream_moe=False, expert_map_path=None),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.init_eplb_config",
            return_value=(torch.tensor([0, 1, 2, -1, -1, -1, -1, -1]), None, 0),
        ),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_forward_context", return_value=mock_forward_context_obj),
        patch("vllm_ascend.ascend_forward_context.get_forward_context", return_value=mock_forward_context_obj),
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.MC2CommImpl._get_token_dispatcher", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.AlltoAllCommImpl._get_token_dispatcher", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.AllGatherCommImpl._get_token_dispatcher", return_value=None),
        patch(
            "vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method",
            return_value=mock_weight_prefetch_method,
        ),
    ):
        yield {
            "mock_forward_context_obj": mock_forward_context_obj,
            "mock_moe_comm_method": mock_moe_comm_method,
        }


@pytest.fixture
def mock_moe_env(mocker: MockerFixture):
    with (
        patch("torch_npu.npu_moe_gating_top_k", return_value=(torch.randn(8, 2), torch.randint(0, 8, (8, 2)), None)),
        patch(
            "torch_npu.npu_moe_init_routing",
            return_value=(torch.randn(8, 2), torch.randint(0, 8, (8, 2)), torch.tensor([0, 1, 2, 4, 6, 2, 7, 1])),
        ),
        patch("torch_npu.npu_moe_compute_expert_tokens", return_value=(torch.randn(8, 2))),
        patch("torch_npu.npu_moe_distribute_dispatch", return_value=(torch.randn(16, 2))),
        patch("torch_npu.npu_moe_distribute_combine", return_value=(torch.randn(16, 2))),
        patch("torch_npu.npu_grouped_matmul", return_value=([torch.randn(16, 2)])),
        patch("torch_npu.npu_swiglu", return_value=(torch.randn(16, 2))),
        patch(
            "torch_npu.npu_moe_gating_top_k_softmax",
            return_value=(torch.randn(8, 2), torch.randint(0, 8, (8, 2)), torch.tensor([0, 1, 2, 4, 6, 2, 7, 1])),
        ),
        patch("torch_npu.npu_moe_finalize_routing", return_value=(torch.randn(16, 2))),
    ):
        if hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            with (
                patch("torch_npu.npu_moe_distribute_dispatch_v2", return_value=(torch.randn(16, 2))),
                patch("torch_npu.npu_moe_distribute_combine_v2", return_value=(torch.randn(16, 2))),
            ):
                yield
        else:
            yield


@pytest.fixture
def default_moe_config():
    return {"num_experts": 8, "top_k": 2, "hidden_size": 512, "intermediate_size": 1024}


@pytest.fixture
def moe_method(mock_dist_env):
    moe = MagicMock()
    moe.moe_parallel_config.return_value = MagicMock(ep_size=4)
    moe.moe_parallel_config.use_ep = False
    moe.moe_parallel_config.dp_size = 1
    return AscendUnquantizedFusedMoEMethod(moe)


class Device(TypedDict):
    device_id: int
    device_expert: list[int]


class Layer(TypedDict):
    layer_id: int
    device_count: int
    device_list: list[Device]


class MockData(TypedDict):
    moe_layer_count: int
    layer_list: list[Layer]


class MockQuantMethod(nn.Module):
    def __init__(self, shared_experts, num_tokens):
        super().__init__()
        if shared_experts:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32), torch.randn(num_tokens, 10)))
        else:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32)))


class TestExpertsSelector:
    @pytest.mark.parametrize("global_num_experts", [256, 128])
    def test_select_experts(self, mock_dist_env, mock_moe_env, global_num_experts):
        x = torch.randn(8, 2)
        router_logits = torch.randn(8, 2)
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            global_num_experts=global_num_experts,
        )

        assert topk_weights.shape == (8, 2)
        assert topk_ids.shape == (8, 2)

    @pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("renormalize", [True, False])
    def test_select_experts_with_different_scoring_func(
            self, mock_dist_env, mock_moe_env, scoring_func, renormalize):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def simple_custom_routing(hidden_states, gating_output, topk,
                                  renormalize, global_num_experts):
            if scoring_func == "softmax":
                weights = gating_output.softmax(dim=-1)
            else:
                weights = gating_output.sigmoid()
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                               keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=renormalize,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=simple_custom_routing,
            scoring_func=scoring_func,
            e_score_correction_bias=None,
            global_num_experts=num_experts)

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)
        assert topk_weights.dtype == hidden_states.dtype
        assert topk_ids.dtype == torch.int32

        if renormalize:
            weight_sum = topk_weights.sum(dim=-1)
            torch.testing.assert_close(weight_sum,
                                       torch.ones_like(weight_sum),
                                       rtol=1e-4,
                                       atol=1e-4)

    def test_select_experts_with_grouped_topk(self, mock_dist_env, mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32
        num_expert_group = 4
        topk_group = 2

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=True,
            renormalize=True,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            global_num_experts=num_experts)

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)

    def test_select_experts_with_e_score_correction_bias(self, mock_dist_env,
                                                         mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)
        e_score_correction_bias = torch.randn(num_experts)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=num_experts)

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)

    def test_select_experts_with_custom_routing_function(self, mock_dist_env,
                                                         mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def custom_routing(hidden_states, gating_output, topk, renormalize,
                           global_num_experts):
            weights = gating_output.softmax(dim=-1)
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                               keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=custom_routing,
            scoring_func="softmax",
            e_score_correction_bias=None,
            global_num_experts=num_experts)

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)

    def test_select_experts_weight_sum_range(self, mock_dist_env, mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def simple_routing(hidden_states, gating_output, topk, renormalize,
                           global_num_experts):
            weights = gating_output.softmax(dim=-1)
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                               keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=simple_routing,
            scoring_func="softmax",
            e_score_correction_bias=None,
            global_num_experts=num_experts)

        assert (topk_weights >= 0).all()
        assert (topk_weights <= 1).all()

        weight_sum = topk_weights.sum(dim=-1)
        torch.testing.assert_close(weight_sum,
                                   torch.ones_like(weight_sum),
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_select_experts_expert_id_range(self, mock_dist_env, mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def simple_routing(hidden_states, gating_output, topk, renormalize,
                           global_num_experts):
            weights = gating_output.softmax(dim=-1)
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                               keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=simple_routing,
            scoring_func="softmax",
            e_score_correction_bias=None,
            global_num_experts=num_experts)

        assert (topk_ids >= 0).all()
        assert (topk_ids < num_experts).all()

    @patch("vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method")
    @patch("vllm_ascend.ops.fused_moe.experts_selector.check_npu_moe_gating_top_k",
           return_value=False)
    def test_select_experts_native_softmax_matches_expected(
            self, _, mock_get_weight_prefetch_method):
        hidden_states = torch.tensor([[1.0, 0.0, -1.0, 2.0],
                                      [0.5, 1.5, -0.5, 1.0]],
                                     dtype=torch.float32)
        router_logits = torch.tensor([[3.0, 1.0, 0.0, 2.0],
                                      [0.0, 4.0, 1.0, 2.0]],
                                     dtype=torch.float32)

        prefetch_method = MagicMock()
        mock_get_weight_prefetch_method.return_value = prefetch_method

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            global_num_experts=4)

        expected_probs = router_logits.softmax(dim=-1)
        expected_weights, expected_ids = expected_probs.topk(2, dim=-1)
        expected_weights = expected_weights / expected_weights.sum(
            dim=-1, keepdim=True)

        prefetch_method.maybe_prefetch_moe_weight_preprocess.assert_called_once_with(
            hidden_states, "gate_up")
        torch.testing.assert_close(topk_weights,
                                   expected_weights.to(hidden_states.dtype))
        assert torch.equal(topk_ids, expected_ids.to(torch.int32))

    @patch("vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method")
    @patch("vllm_ascend.ops.fused_moe.experts_selector.check_npu_moe_gating_top_k",
           return_value=False)
    def test_select_experts_grouped_topk_bias_uses_original_weights(
            self, _, mock_get_weight_prefetch_method):
        hidden_states = torch.tensor([[1.0, 0.0, -1.0, 2.0]],
                                     dtype=torch.float32)
        router_logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]],
                                     dtype=torch.float32)
        e_score_correction_bias = torch.tensor([-10.0, -10.0, 5.0, 5.0],
                                               dtype=torch.float32)

        mock_get_weight_prefetch_method.return_value = MagicMock()

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=1,
            num_expert_group=2,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=4)

        original_weights = router_logits.softmax(dim=-1)
        sorted_ids, order = torch.sort(topk_ids, dim=-1)
        sorted_weights = topk_weights.gather(1, order)

        expected_ids = torch.tensor([[2, 3]], dtype=torch.int32)
        expected_weights = original_weights[:, 2:4].to(hidden_states.dtype)

        assert torch.equal(sorted_ids, expected_ids)
        torch.testing.assert_close(sorted_weights, expected_weights)

    @patch("vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method",
           return_value=MagicMock())
    @patch("vllm_ascend.ops.fused_moe.experts_selector.check_npu_moe_gating_top_k",
           return_value=False)
    def test_select_experts_invalid_scoring_func_raises(self, _, __):
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 4)

        with pytest.raises(ValueError, match="Unsupported scoring function"):
            select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=2,
                use_grouped_topk=False,
                renormalize=True,
                topk_group=None,
                num_expert_group=None,
                custom_routing_function=None,
                scoring_func="unsupported",
                e_score_correction_bias=None,
                global_num_experts=4)


class TestCumsumGroupList(TestBase):
    def setUp(self):
        self.active_num = 8
        self.expert_num = 128
        self.experts = torch.zeros((self.expert_num,), dtype=torch.int64)
        self.experts[: self.active_num] = 1
        self.experts = self.experts[torch.randperm(self.expert_num)]
        self.group_list = self.experts.cumsum(dim=0)

    def test_cumsum_group_list_with_type_0(self):
        group_list = self.experts.cumsum(dim=0)
        group_list_type = 0
        result = cumsum_group_list(group_list, group_list_type, 0)
        self.assertTrue(torch.equal(result, self.group_list))

    def test_cumsum_group_list_with_type_1(self):
        group_list = self.experts
        group_list_type = 1
        result = cumsum_group_list(group_list, group_list_type, 0)
        self.assertTrue(torch.equal(result, self.group_list))

    def test_cumsum_group_list_with_type_2(self):
        tokens = torch.arange(self.expert_num, dtype=torch.int64)
        group_list = torch.cat([tokens.reshape(self.expert_num, 1), self.experts.reshape(self.expert_num, 1)], dim=1)
        group_list_type = 2
        result = cumsum_group_list(
            group_list, group_list_type, 0, active_num=self.active_num, expert_num=self.expert_num
        )
        self.assertTrue(torch.equal(result, self.group_list))


class TestUnifiedApplyMLP(TestBase):
    @patch("vllm_ascend.ops.fused_moe.moe_mlp.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch_npu.npu_dequant_swiglu_quant")
    def test_unified_apply_mlp_with_quantization_mc2(
        self,
        mock_npu_dequant,
        mock_npu_dynamic_quant,
        mock_npu_grouped_matmul,
        mock_soc_version,
        mock_get_forward_context,
        mock_get_weight_prefetch_method,
    ):
        mock_forward_context = MagicMock()
        mock_forward_context.moe_comm_type = MoECommType.MC2
        mock_get_forward_context.return_value = mock_forward_context

        mock_npu_dynamic_quant.return_value = (
            torch.randint(-128, 127, (10, 20), dtype=torch.int8),
            torch.rand(10, 1, dtype=torch.float32),
        )

        mock_npu_grouped_matmul.side_effect = [
            [torch.randint(-2147483648, 2147483647, (10, 40), dtype=torch.int32)],
            [torch.randn(10, 20, dtype=torch.bfloat16)],
        ]

        mock_npu_dequant.return_value = (
            torch.randn(10, 40, dtype=torch.bfloat16),
            torch.randn(10, 1, dtype=torch.float32),
        )

        hidden_states = torch.randn(10, 20, dtype=torch.bfloat16)
        w1 = torch.randint(-128, 127, (5, 20, 40), dtype=torch.int8)
        w1_scale = torch.randn(5, 40, dtype=torch.float32)
        w2 = torch.randint(-128, 127, (5, 40, 20), dtype=torch.int8)
        w2_scale = torch.randn(5, 20, dtype=torch.bfloat16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        )

        mock_get_forward_context.assert_called()

        mock_npu_dynamic_quant.assert_called()

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)

        mock_npu_dequant.assert_called_once()

        self.assertEqual(result.dtype, torch.bfloat16)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_dynamic_quant")
    def test_unified_apply_mlp_without_quantization(
        self, mock_npu_dynamic_quant, mock_npu_swiglu, mock_npu_grouped_matmul, mock_soc_version
    ):
        mock_npu_grouped_matmul.side_effect = [
            [torch.randn(10, 40, dtype=torch.float16)],
            [torch.randn(10, 20, dtype=torch.float16)],
        ]
        mock_npu_swiglu.return_value = torch.randn(10, 40, dtype=torch.float16)
        mock_npu_dynamic_quant.return_value = (MagicMock(), MagicMock())

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        topk_scales = torch.randn(10, 1, dtype=torch.float16)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=False,
                topk_scales=topk_scales,
            )
        )

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_swiglu.assert_called_once()

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)

    @patch("vllm_ascend.ops.fused_moe.moe_mlp.HAS_TRITON", False)
    @patch("vllm_ascend.ops.fused_moe.moe_mlp.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_dynamic_quant")
    def test_unified_apply_mlp_with_quantization_and_dynamic_scale(
        self,
        mock_npu_dynamic_quant,
        mock_npu_swiglu,
        mock_npu_grouped_matmul,
        mock_get_forward_context,
        mock_get_weight_prefetch_method,
    ):
        mock_forward_context = MagicMock()
        mock_forward_context.with_quant = True
        mock_forward_context.fused_moe_state = "NOT_MC2"
        mock_get_forward_context.return_value = mock_forward_context

        mock_npu_grouped_matmul.side_effect = [
            [torch.randn(10, 40, dtype=torch.bfloat16)],
            [torch.randn(10, 20, dtype=torch.bfloat16)],
        ]

        mock_npu_swiglu.return_value = torch.randn(10, 40, dtype=torch.bfloat16)

        mock_npu_dynamic_quant.return_value = (
            torch.randint(-128, 127, (10, 40), dtype=torch.int8),
            torch.rand(10, 1, dtype=torch.float32),
        )

        hidden_states = torch.randn(10, 20, dtype=torch.bfloat16)
        hidden_states_shape = hidden_states.shape
        w1 = torch.randn(5, 20, 40, dtype=torch.bfloat16)
        w1_scale = torch.randn(5, 40, dtype=torch.bfloat16)
        w2 = torch.randn(5, 40, 20, dtype=torch.bfloat16)
        w2_scale = torch.randn(5, 20, dtype=torch.bfloat16)
        w1_scale_bias = torch.randn(5, 40, dtype=torch.bfloat16)
        w2_scale_bias = torch.randn(5, 20, dtype=torch.bfloat16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        provided_dynamic_scale = torch.rand(10, 1, dtype=torch.float32)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=True,
                dynamic_scale=provided_dynamic_scale,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
            )
        )

        mock_get_forward_context.assert_called()

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_swiglu.assert_called_once()
        mock_npu_dynamic_quant.assert_called_once()

        self.assertEqual(result.shape, hidden_states_shape)
        self.assertEqual(result.dtype, torch.bfloat16)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType._310P)
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_dynamic_quant")
    def test_unified_apply_mlp_without_quantization_310p(
        self, mock_npu_dynamic_quant, mock_npu_swiglu, mock_npu_grouped_matmul, mock_soc_version
    ):
        mock_gmm1_out = torch.randn(10, 40, dtype=torch.float16)
        mock_gmm2_out = torch.randn(10, 20, dtype=torch.float16)
        mock_npu_grouped_matmul.side_effect = [[mock_gmm1_out], [mock_gmm2_out]]

        mock_npu_swiglu.return_value = torch.randn(10, 40, dtype=torch.float16)

        mock_npu_dynamic_quant.return_value = (MagicMock(), MagicMock())

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        topk_scales = torch.randn(10, 1, dtype=torch.float16)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=False,
                topk_scales=topk_scales,
            )
        )

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_swiglu.assert_called_once()

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)

    @patch("vllm_ascend.ops.fused_moe.moe_mlp.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_grouped_matmul_swiglu_quant")
    @patch("torch_npu.npu_dynamic_quant")
    def test_unified_apply_mlp_with_quantization_and_fusion_mlp(
        self,
        mock_npu_dynamic_quant,
        mock_npu_grouped_matmul_swiglu_quant,
        mock_npu_swiglu,
        mock_npu_grouped_matmul,
        mock_get_forward_context,
        mock_get_weight_prefetch_method,
    ):
        mock_forward_context = MagicMock()
        mock_forward_context.with_quant = True
        mock_forward_context.fused_moe_state = "NOT_MC2"
        mock_get_forward_context.return_value = mock_forward_context

        mock_npu_grouped_matmul_swiglu_quant.return_value = (
            torch.randint(-128, 127, (10, 40), dtype=torch.int8),
            torch.rand(10, 1, dtype=torch.float32),
            torch.rand(10, 1, dtype=torch.float32),
        )
        mock_npu_grouped_matmul.side_effect = [[torch.randn(10, 20, dtype=torch.bfloat16)]]
        mock_npu_swiglu.return_value = torch.randn(10, 40, dtype=torch.bfloat16)
        mock_npu_dynamic_quant.return_value = (
            torch.randint(-128, 127, (10, 40), dtype=torch.int8),
            torch.rand(10, 1, dtype=torch.float32),
        )

        hidden_states = torch.randn(10, 20, dtype=torch.bfloat16)
        hidden_states_shape = hidden_states.shape
        w1 = torch.randn(5, 20, 40, dtype=torch.bfloat16)
        w1_scale = torch.randn(5, 40, dtype=torch.bfloat16)
        w2 = torch.randn(5, 40, 20, dtype=torch.bfloat16)
        w2_scale = torch.randn(5, 20, dtype=torch.bfloat16)
        w1_scale_bias = torch.randn(5, 40, dtype=torch.bfloat16)
        w2_scale_bias = torch.randn(5, 20, dtype=torch.bfloat16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        provided_dynamic_scale = torch.rand(10, 1, dtype=torch.float32)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=True,
                dynamic_scale=provided_dynamic_scale,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                fusion=True,
            )
        )

        mock_get_forward_context.assert_called()
        mock_npu_grouped_matmul.assert_called_once()
        mock_npu_grouped_matmul_swiglu_quant.assert_called_once()

        self.assertTrue(mock_forward_context.with_quant)
        self.assertEqual(result.shape, hidden_states_shape)
        self.assertEqual(result.dtype, torch.bfloat16)

    @patch('torch_npu.npu_swiglu')
    @patch('torch_npu.npu_grouped_matmul')
    def test_unified_apply_mlp_without_quantization_matches_expected_values(
            self, mock_npu_grouped_matmul, mock_npu_swiglu):
        def fake_grouped_matmul(*, x, weight, bias=None, **kwargs):
            weight_tensor = weight[0]
            if weight_tensor.dim() == 3 and weight_tensor.shape[0] == 1:
                weight_tensor = weight_tensor[0]

            result = x[0] @ weight_tensor
            if bias is not None:
                bias_tensor = bias[0]
                if bias_tensor.dim() == 2 and bias_tensor.shape[0] == 1:
                    bias_tensor = bias_tensor[0]
                result = result + bias_tensor.to(result.dtype)
            return [result]

        def fake_swiglu(x):
            left, right = x.chunk(2, dim=-1)
            return F.silu(left) * right

        mock_npu_grouped_matmul.side_effect = fake_grouped_matmul
        mock_npu_swiglu.side_effect = fake_swiglu

        hidden_states = torch.tensor([[1.0, 2.0], [0.5, -1.0]],
                                     dtype=torch.float32)
        w1 = torch.tensor([[[1.0, 0.0, 1.0, -1.0],
                            [0.0, 1.0, 2.0, 1.0]]],
                          dtype=torch.float32)
        w2 = torch.tensor([[[1.0, 2.0], [-1.0, 1.0]]], dtype=torch.float32)
        topk_scales = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
        group_list = torch.tensor([2], dtype=torch.int64)

        result = unified_apply_mlp(
            mlp_compute_input=build_mlp_compute_input_fixture(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=group_list,
                with_quant=False,
                topk_scales=topk_scales,
                need_trans=False,
            ))

        gate_up = hidden_states @ w1[0]
        activated = F.silu(gate_up[:, :2]) * gate_up[:, 2:]
        activated = activated * topk_scales
        expected = activated @ w2[0]

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_swiglu.assert_called_once()
        torch.testing.assert_close(result, expected)


class TestZeroExpertsCompute(TestBase):
    def test_zero_experts_compute_identity_type(self):
        from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

        num_experts = 8
        num_tokens = 4
        top_k = 2
        hidden_size = 16

        expert_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32)
        expert_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_indices.shape == expert_indices.shape
        assert result_scales.shape == expert_scales.shape
        assert result_hidden.shape == hidden_states.shape

    def test_zero_experts_compute_with_zero_experts(self):
        from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

        num_experts = 4
        num_tokens = 3
        top_k = 2
        hidden_size = 8

        expert_indices = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]], dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_indices.shape == expert_indices.shape
        assert result_scales.shape == expert_scales.shape
        assert result_hidden.shape == hidden_states.shape

    def test_zero_experts_compute_normal_experts_masked(self):
        from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

        num_experts = 4
        num_tokens = 2
        top_k = 2
        hidden_size = 8

        expert_indices = torch.tensor([[0, 5], [6, 7]], dtype=torch.int32)
        expert_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, _ = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        normal_expert_mask = expert_indices >= num_experts
        for i in range(num_tokens):
            for j in range(top_k):
                if normal_expert_mask[i, j]:
                    assert result_scales[i, j] == 0.0
                    assert result_indices[i, j] == 0

    def test_zero_experts_compute_output_sum(self):
        from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

        num_experts = 2
        num_tokens = 2
        top_k = 2
        hidden_size = 4

        expert_indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        _, _, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_hidden.shape == (num_tokens, hidden_size)

    def test_zero_experts_compute_all_zero_experts(self):
        from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

        num_experts = 4
        num_tokens = 2
        top_k = 2
        hidden_size = 8

        expert_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        expert_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert torch.equal(result_indices, expert_indices)
        assert torch.equal(result_scales, expert_scales)
        assert result_hidden.shape == hidden_states.shape

    def test_zero_experts_compute_mixed_experts(self):
        from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

        num_experts = 3
        num_tokens = 2
        top_k = 3
        hidden_size = 8

        expert_indices = torch.tensor([[0, 1, 4], [2, 5, 6]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]], dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, _ = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_indices.shape == expert_indices.shape
        assert result_scales.shape == expert_scales.shape

    def test_zero_experts_compute_identity_values_match_expected(self):
        from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute

        expert_indices = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.25, 0.75], [0.60, 0.40]],
                                     dtype=torch.float32)
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                                     dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=2,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        expected_indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
        expected_scales = torch.tensor([[0.25, 0.0], [0.0, 0.40]],
                                       dtype=torch.float32)
        expected_hidden = torch.tensor([[0.75, 1.50], [1.80, 2.40]],
                                       dtype=torch.float32)

        assert torch.equal(result_indices, expected_indices)
        torch.testing.assert_close(result_scales, expected_scales)
        torch.testing.assert_close(result_hidden, expected_hidden)
