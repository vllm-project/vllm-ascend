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
from typing import List, TypedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch_npu
from pytest_mock import MockerFixture
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

import vllm_ascend.ops.moe.token_dispatcher as token_dispatcher_module
from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import (FusedMoEState,
                                                _get_fused_moe_state)
from vllm_ascend.ops.fused_moe import (AscendFusedMoE,
                                       AscendUnquantizedFusedMoEMethod)
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_mlp import cumsum_group_list, unified_apply_mlp
from vllm_ascend.utils import AscendSocVersion, adapt_patch

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


@pytest.fixture
def mock_dist_env(mocker: MockerFixture):
    mock_setup_token_dispatchers = MagicMock()
    mock_token_dispatcher_with_allgather = MagicMock()
    mock_token_dispatcher_with_all2allv = MagicMock()
    mock_token_dispatcher_with_mc2 = MagicMock()

    mock_dispatch_result_allgather = {
        "hidden_states": torch.randn(16, 2),
        "group_list": torch.tensor([8, 16], dtype=torch.int64),
        "group_list_type": 0,
    }
    mock_combine_result_allgather = torch.randn(16, 2)

    mock_token_dispatcher_with_allgather.token_dispatch.return_value = mock_dispatch_result_allgather
    mock_token_dispatcher_with_allgather.token_combine.return_value = mock_combine_result_allgather

    mock_dispatch_result_all2allv = {
        "hidden_states": torch.randn(16, 2),
        "group_list": torch.tensor([4, 8, 12, 16], dtype=torch.int64),
        "group_list_type": 1,
        "dynamic_scale": None,
    }
    mock_combine_result_all2allv = torch.randn(16, 2)
    mock_token_dispatcher_with_all2allv.token_dispatch.return_value = mock_dispatch_result_all2allv
    mock_token_dispatcher_with_all2allv.token_combine.return_value = mock_combine_result_all2allv

    mock_dispatch_result_mc2 = {
        "hidden_states": torch.randn(16, 2),
        "group_list": torch.tensor([5, 10, 15, 16], dtype=torch.int64),
        "group_list_type": 1,
        "dynamic_scale": None,
        "assist_info_for_combine": torch.randn(16, 2),
        "ep_recv_counts": torch.tensor([4, 4, 4, 4], dtype=torch.int32),
    }
    mock_combine_result_mc2 = torch.randn(16, 2)
    mock_token_dispatcher_with_mc2.token_dispatch.return_value = mock_dispatch_result_mc2
    mock_token_dispatcher_with_mc2.token_combine.return_value = mock_combine_result_mc2

    captured_dispatchers = {}

    def capture_register(dispatcher_instance):
        key = dispatcher_instance.__class__.__name__
        captured_dispatchers[key] = dispatcher_instance
        if key == 'TokenDispatcherWithAllGather':
            captured_dispatchers[key] = mock_token_dispatcher_with_allgather
        elif key == 'TokenDispatcherWithAll2AllV':
            captured_dispatchers[key] = mock_token_dispatcher_with_all2allv
        elif key == 'TokenDispatcherWithMC2':
            captured_dispatchers[key] = mock_token_dispatcher_with_mc2

    mock_register_token_dispatcher_patcher = patch(
        'vllm_ascend.ops.moe.token_dispatcher._register_token_dispatcher',
        side_effect=capture_register)

    mock_get_token_dispatcher_patcher = patch(
        'vllm_ascend.ops.moe.token_dispatcher.get_token_dispatcher',
        side_effect=lambda name: captured_dispatchers.get(name))

    default_mock_token_dispatcher = mock_token_dispatcher_with_allgather

    mock_forward_context_obj = MagicMock(
        fused_moe_state=FusedMoEState.AllGather,
        token_dispatcher=default_mock_token_dispatcher,
        max_tokens_across_dp=10,
        dp_metadata=MagicMock(cu_tokens_across_dp_cpu=[5, 10]),
        mc2_mask=torch.zeros(16, dtype=torch.bool),
        padded_num_tokens=16,
        with_quant=False)

    with patch('torch.distributed.get_rank', return_value=0), \
        patch('torch.distributed.get_world_size', return_value=4), \
        patch('vllm_ascend.ops.fused_moe.get_ep_group', return_value=mock_ep_and_mc2_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.get_mc2_group', return_value=mock_ep_and_mc2_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.get_tp_group', return_value=mock_dp_and_tp_group(mocker)), \
        patch('vllm.distributed.parallel_state.get_tp_group', return_value=mock_dp_and_tp_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.get_dp_group', return_value=mock_dp_and_tp_group(mocker)), \
        patch('vllm.model_executor.layers.fused_moe.layer.get_dp_group', return_value=mock_dp_and_tp_group(mocker)), \
        patch('torch.distributed.all_gather'), \
        patch('torch.distributed.all_to_all_single'), \
        patch('vllm_ascend.ops.fused_moe.tensor_model_parallel_all_reduce'), \
        patch('vllm.model_executor.layers.fused_moe.config.get_dp_group',
            return_value=mock_dp_and_tp_group(mocker)), \
        patch('vllm_ascend.ops.fused_moe.get_ascend_config',
            return_value=MagicMock(
                torchair_graph_config=MagicMock(enabled=False, enable_multistream_moe=False),
                expert_map_path=None
            )), \
        patch('vllm_ascend.ops.fused_moe.determine_expert_map',
            return_value=(3, torch.tensor([0, 1, 2, -1, -1, -1, -1, -1]))), \
        patch('vllm_ascend.ops.fused_moe.get_forward_context',
            return_value=mock_forward_context_obj), \
        patch('vllm_ascend.ops.fused_moe.get_current_vllm_config',
                return_value=MagicMock(
                    parallel_config=MagicMock(tensor_parallel_size=2),
                    scheduler_config=MagicMock(max_num_seqs=4),
                    model_config=MagicMock(max_model_len=2048)
                )), \
        patch("vllm_ascend.utils.get_ascend_soc_version", return_value=AscendSocVersion.A3), \
        patch.object(token_dispatcher_module, 'setup_token_dispatchers', mock_setup_token_dispatchers), \
        patch('vllm_ascend.ops.moe.moe_mlp.get_forward_context',
                return_value=mock_forward_context_obj):

        yield {
            'mock_forward_context_obj': mock_forward_context_obj,
            'mock_token_dispatcher_with_allgather':
            mock_token_dispatcher_with_allgather,
            'mock_token_dispatcher_with_all2allv':
            mock_token_dispatcher_with_all2allv,
            'mock_token_dispatcher_with_mc2': mock_token_dispatcher_with_mc2,
        }

    mock_register_token_dispatcher_patcher.stop()
    mock_get_token_dispatcher_patcher.stop()


@pytest.fixture
def mock_moe_env(mocker: MockerFixture):

    with patch('torch_npu.npu_moe_gating_top_k', return_value=(
            torch.randn(8, 2),
            torch.randint(0, 8, (8, 2)),
            None
        )), \
        patch('torch_npu.npu_moe_init_routing', return_value=(
                torch.randn(8, 2),
                torch.randint(0, 8, (8, 2)),
                torch.tensor([0, 1, 2, 4, 6, 2, 7, 1])
        )), \
        patch("torch_npu.npu_moe_compute_expert_tokens", return_value=(
                torch.randn(8, 2)
        )), \
        patch("torch_npu.npu_moe_distribute_dispatch", return_value=(
                torch.randn(16, 2)
        )), \
        patch("torch_npu.npu_moe_distribute_combine", return_value=(
                torch.randn(16, 2)
        )), \
        patch("torch_npu.npu_grouped_matmul", return_value=(
                [torch.randn(16, 2)]
        )), \
        patch("torch_npu.npu_swiglu", return_value=(
                torch.randn(16, 2)
        )), \
        patch("torch_npu.npu_moe_gating_top_k_softmax", return_value=(
                torch.randn(8, 2),
                torch.randint(0, 8, (8, 2)),
                torch.tensor([0, 1, 2, 4, 6, 2, 7, 1])
        )), \
        patch("torch_npu.npu_moe_finalize_routing", return_value=(
                torch.randn(16, 2)
        )):
        if hasattr(torch_npu, 'npu_moe_distribute_dispatch_v2'):
            with patch("torch_npu.npu_moe_distribute_dispatch_v2", return_value=(
                torch.randn(16, 2))), \
                patch("torch_npu.npu_moe_distribute_combine_v2", return_value=(
                torch.randn(16, 2))):
                yield
        else:
            yield


@pytest.fixture
def default_moe_config():
    return {
        'num_experts': 8,
        'top_k': 2,
        'hidden_size': 512,
        'intermediate_size': 1024
    }


@pytest.fixture
def moe_method(mock_dist_env):
    moe = MagicMock()
    moe.moe_parallel_config.return_value = MagicMock(ep_size=4)
    return AscendUnquantizedFusedMoEMethod(moe)


class Device(TypedDict):
    device_id: int
    device_expert: List[int]


class Layer(TypedDict):
    layer_id: int
    device_count: int
    device_list: List[Device]


class MockData(TypedDict):
    moe_layer_count: int
    layer_list: List[Layer]


class MockQuantMethod(nn.Module):

    def __init__(self, shared_experts, num_tokens):
        super().__init__()
        if shared_experts:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32),
                                                 torch.randn(num_tokens, 10)))
        else:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32)))


class MockFusedMoEMethod(FusedMoEMethodBase):
    moe = MagicMock()

    def __init__(self):
        super().__init__(self.moe)

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        pass

    def apply(self, hidden_states: torch.Tensor,
              expert_weights: torch.Tensor) -> torch.Tensor:
        pass


class TestAscendFusedMoe:

    def test_init_no_quant(self, mock_dist_env, default_moe_config):
        layer = AscendFusedMoE(**default_moe_config)

        layer.w13_weight = nn.Parameter(
            torch.randn(default_moe_config['num_experts'],
                        default_moe_config['intermediate_size'] * 2,
                        default_moe_config['hidden_size']))
        layer.w2_weight = nn.Parameter(
            torch.randn(default_moe_config['num_experts'],
                        default_moe_config['hidden_size'],
                        default_moe_config['intermediate_size']))

        assert layer.num_experts == default_moe_config['num_experts']
        assert layer.top_k == default_moe_config['top_k']
        assert hasattr(layer, 'w13_weight')
        assert hasattr(layer, 'w2_weight')

        with pytest.raises(AssertionError):
            error_config = default_moe_config.copy()
            error_config['use_grouped_topk'] = True
            layer = AscendFusedMoE(**error_config)

        with pytest.raises(ValueError):
            error_config = default_moe_config.copy()
            error_config['scoring_func'] = "random"
            layer = AscendFusedMoE(**error_config)

    def test_init_with_quant(self, mock_dist_env, default_moe_config):
        mock_quant_config = MagicMock()
        mock_quant_method = MockFusedMoEMethod()
        mock_quant_config.get_quant_method.return_value = mock_quant_method

        moe = AscendFusedMoE(**default_moe_config,
                             quant_config=mock_quant_config)

        assert moe.quant_method is not None
        assert moe.quant_method == mock_quant_method

    @pytest.mark.parametrize(
        "others_param",
        [[None,
          MagicMock(return_value=torch.randn(5, 32)), False, 5, None],
         [2, None, False, 5, None], [None, None, True, 5, None],
         [None, None, False, 1, None], [None, None, True, 5, 1],
         [None, None, False, 5, 1]])
    def test_forward(self, mock_dist_env, default_moe_config, others_param):

        top_k, shared_experts, is_prefill, num_tokens, ep_size = others_param
        inputs = torch.randn(num_tokens, 32)
        router_logits = torch.randn(num_tokens, 8)
        moe = AscendFusedMoE(**default_moe_config)

        if ep_size == 1:
            moe.moe_parallel_config.ep_size = 1

        moe.quant_method = MockQuantMethod(shared_experts, num_tokens)
        forward_context = MagicMock(mc2_mask=torch.zeros(num_tokens,
                                                         dtype=torch.bool),
                                    padded_num_tokens=num_tokens)
        with patch("vllm_ascend.ops.fused_moe.get_forward_context",
                   return_value=forward_context):
            output = moe.forward(inputs,
                                 router_logits,
                                 is_prefill=is_prefill,
                                 top_k=top_k,
                                 shared_experts=shared_experts)

        moe.quant_method.apply.assert_called_once()

        if shared_experts:
            assert output[0].shape == (num_tokens, 32)
            assert output[1].shape == (num_tokens, 10)
        else:
            assert output.shape == (num_tokens, 32)

    def test_forward_ms_fused_moe_comp(self, mock_dist_env,
                                       default_moe_config):
        inputs = torch.randn(5, 32)
        router_logits = torch.randn(5, 8)
        moe = AscendFusedMoE(**default_moe_config)

        moe.quant_method = MockQuantMethod(None, 5)
        output = moe._forward_ms_fused_moe_comp(inputs,
                                                router_logits,
                                                is_prefill=False,
                                                real_top_k=1)

        moe.quant_method.apply.assert_called_once()

        assert output.shape == (5, 32)


class TestAscendUnquantizedFusedMoEMethod:

    def test_process_weights_after_loading(self, moe_method, mock_dist_env):
        layer = MagicMock()
        layer.w13_weight.data = torch.randn(16, 32)
        layer.w2_weight.data = torch.randn(16, 32)

        with patch('torch_npu.npu_format_cast', mock_npu_format_cast), \
                patch('vllm_ascend.utils.is_310p', return_value=False):
            moe_method.process_weights_after_loading(layer)

            assert isinstance(layer.w13_weight, torch.nn.Parameter)
            assert isinstance(layer.w2_weight, torch.nn.Parameter)
            assert not layer.w13_weight.requires_grad
            assert not layer.w2_weight.requires_grad

    @pytest.mark.parametrize("others_param",
                             [[256, 4], [128, 1], [128, 1], [128, 4]])
    def test_apply_without_expert_map(self, moe_method, mock_dist_env,
                                      mock_moe_env, others_param):

        global_num_experts, ep_size = others_param
        is_prefill = False
        is_deepseek_v3_r1 = global_num_experts == 256

        if ep_size == 1:
            selected_token_dispatcher = mock_dist_env[
                'mock_token_dispatcher_with_allgather']
        elif ep_size < 16:
            selected_token_dispatcher = mock_dist_env[
                'mock_token_dispatcher_with_all2allv']
        else:
            selected_token_dispatcher = mock_dist_env[
                'mock_token_dispatcher_with_mc2']

        forward_context = MagicMock(fused_moe_state=_get_fused_moe_state(
            ep_size, is_prefill, is_deepseek_v3_r1),
                                    with_quant=False,
                                    token_dispatcher=selected_token_dispatcher)

        with patch("vllm_ascend.ops.fused_moe.get_forward_context",
                   return_value=forward_context):
            moe_method.ep_size = ep_size
            x = torch.randn(8, 2, 2)
            router_logits = torch.randn(8, 8)
            layer = MagicMock()
            local_num_experts = 2
            hidden_size = 2
            intermediate_size_per_partition = 4

            layer.w13_weight = torch.randn(local_num_experts,
                                           intermediate_size_per_partition * 2,
                                           hidden_size)
            layer.w2_weight = torch.randn(local_num_experts, hidden_size,
                                          intermediate_size_per_partition)

            result = moe_method.apply(layer=layer,
                                      x=x,
                                      router_logits=router_logits,
                                      top_k=2,
                                      renormalize=True,
                                      global_num_experts=global_num_experts,
                                      is_prefill=is_prefill)

            expected_shape = (16, 2)

            assert result.shape == expected_shape

    @pytest.mark.parametrize("others_param", [16, 1, 4])
    def test_apply_with_expert_map(self, moe_method, mock_dist_env,
                                   mock_moe_env, others_param):

        ep_size = others_param
        is_prefill = False

        if ep_size == 1:
            selected_token_dispatcher = mock_dist_env[
                'mock_token_dispatcher_with_allgather']
        elif ep_size < 16:
            selected_token_dispatcher = mock_dist_env[
                'mock_token_dispatcher_with_all2allv']
        else:
            selected_token_dispatcher = mock_dist_env[
                'mock_token_dispatcher_with_mc2']

        forward_context = MagicMock(fused_moe_state=_get_fused_moe_state(
            ep_size, is_prefill, True),
                                    with_quant=False,
                                    token_dispatcher=selected_token_dispatcher)

        with patch("vllm_ascend.ops.fused_moe.get_forward_context", return_value=forward_context), \
             patch("vllm_ascend.utils.get_ascend_soc_version", return_value=AscendSocVersion.A3):

            expert_map = torch.tensor([0, 1, 2, -1, -1, -1, -1, -1])
            moe_method.ep_size = ep_size
            x = torch.randn(8, 2, 2)
            if ep_size == 1:
                x = x.view(-1, 2)
            router_logits = torch.randn(8, 8)
            layer = MagicMock()

            local_num_experts = 2
            hidden_size = 2
            intermediate_size_per_partition = 4
            layer.w13_weight = torch.randn(local_num_experts,
                                           intermediate_size_per_partition * 2,
                                           hidden_size)
            layer.w2_weight = torch.randn(local_num_experts, hidden_size,
                                          intermediate_size_per_partition)

            result = moe_method.apply(layer=layer,
                                      x=x,
                                      router_logits=router_logits,
                                      top_k=2,
                                      renormalize=True,
                                      global_num_experts=128,
                                      expert_map=expert_map,
                                      is_prefill=is_prefill)

            expected_shape = (16, 2)

            assert result.shape == expected_shape


class TestExpertsSelector:

    @pytest.mark.parametrize("global_num_experts", [[256], [128]])
    def test_select_experts(self, mock_dist_env, mock_moe_env,
                            global_num_experts):

        x = torch.randn(8, 2)
        router_logits = torch.randn(8, 2)
        topk_weights, topk_ids, _ = select_experts(
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
            global_num_experts=global_num_experts)

        assert topk_weights.shape == (8, 2)
        assert topk_ids.shape == (8, 2)


class TestCumsumGroupList(TestBase):

    def setUp(self):
        self.active_num = 8
        self.expert_num = 128
        self.experts = torch.zeros((self.expert_num, ), dtype=torch.int64)
        self.experts[:self.active_num] = 1
        self.experts = self.experts[torch.randperm(self.expert_num)]
        self.group_list = self.experts.cumsum(dim=0)

    def test_cumsum_group_list_with_type_0(self):
        group_list = self.experts.cumsum(dim=0)
        group_list_type = 0
        result = cumsum_group_list(group_list, group_list_type)
        self.assertTrue(torch.equal(result, self.group_list))

    def test_cumsum_group_list_with_type_1(self):
        group_list = self.experts
        group_list_type = 1
        result = cumsum_group_list(group_list, group_list_type)
        self.assertTrue(torch.equal(result, self.group_list))

    def test_cumsum_group_list_with_type_2(self):
        tokens = torch.arange(self.expert_num, dtype=torch.int64)
        group_list = torch.cat([
            tokens.reshape(self.expert_num, 1),
            self.experts.reshape(self.expert_num, 1)
        ],
                               dim=1)
        group_list_type = 2
        result = cumsum_group_list(group_list,
                                   group_list_type,
                                   active_num=self.active_num,
                                   expert_num=self.expert_num)
        self.assertTrue(torch.equal(result, self.group_list))


class TestUnifiedApplyMLP(TestBase):

    @patch('vllm_ascend.ops.moe.moe_mlp.get_forward_context')
    @patch('vllm_ascend.ops.moe.moe_mlp.is_310p')
    @patch('torch_npu.npu_grouped_matmul')
    @patch('torch_npu.npu_dynamic_quant')
    @patch('torch_npu.npu_dequant_swiglu_quant')
    def test_unified_apply_mlp_with_quantization_mc2(self, mock_npu_dequant,
                                                     mock_npu_dynamic_quant,
                                                     mock_npu_grouped_matmul,
                                                     mock_is_310p,
                                                     mock_get_forward_context):

        mock_forward_context = MagicMock()
        mock_forward_context.fused_moe_state = FusedMoEState.MC2
        mock_get_forward_context.return_value = mock_forward_context

        mock_is_310p.return_value = False

        mock_npu_dynamic_quant.return_value = (torch.randint(-128,
                                                             127, (10, 20),
                                                             dtype=torch.int8),
                                               torch.rand(10,
                                                          1,
                                                          dtype=torch.float32))

        mock_npu_grouped_matmul.side_effect = [[
            torch.randint(-2147483648, 2147483647, (10, 40), dtype=torch.int32)
        ], [torch.randn(10, 20, dtype=torch.bfloat16)]]

        mock_npu_dequant.return_value = (torch.randn(10,
                                                     40,
                                                     dtype=torch.bfloat16),
                                         torch.randn(10,
                                                     1,
                                                     dtype=torch.float32))

        hidden_states = torch.randn(10, 20, dtype=torch.bfloat16)
        w1 = torch.randint(-128, 127, (5, 20, 40), dtype=torch.int8)
        w1_scale = torch.randn(5, 40, dtype=torch.float32)
        w2 = torch.randint(-128, 127, (5, 40, 20), dtype=torch.int8)
        w2_scale = torch.randn(5, 20, dtype=torch.bfloat16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)

        result = unified_apply_mlp(hidden_states=hidden_states,
                                   w1=w1,
                                   w1_scale=w1_scale,
                                   w2=w2,
                                   w2_scale=w2_scale,
                                   group_list=group_list,
                                   dynamic_scale=None,
                                   group_list_type=1,
                                   w1_scale_bias=None,
                                   w2_scale_bias=None,
                                   topk_scales=None,
                                   with_quant=True)

        mock_get_forward_context.assert_called()
        self.assertEqual(mock_forward_context.fused_moe_state,
                         FusedMoEState.MC2)

        mock_npu_dynamic_quant.assert_called()

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)

        mock_npu_dequant.assert_called_once()

        self.assertEqual(result.dtype, torch.bfloat16)

    @patch('vllm_ascend.ops.moe.moe_mlp.is_310p')
    @patch('torch_npu.npu_grouped_matmul')
    @patch('torch_npu.npu_swiglu')
    @patch('torch_npu.npu_dynamic_quant')
    def test_unified_apply_mlp_without_quantization(self,
                                                    mock_npu_dynamic_quant,
                                                    mock_npu_swiglu,
                                                    mock_npu_grouped_matmul,
                                                    mock_is_310p):
        mock_is_310p.return_value = False

        mock_npu_grouped_matmul.side_effect = [[
            torch.randn(10, 40, dtype=torch.float16)
        ], [torch.randn(10, 20, dtype=torch.float16)]]
        mock_npu_swiglu.return_value = torch.randn(10, 40, dtype=torch.float16)
        mock_npu_dynamic_quant.return_value = (MagicMock(), MagicMock())

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        topk_scales = torch.randn(10, 1, dtype=torch.float16)

        result = unified_apply_mlp(hidden_states=hidden_states,
                                   w1=w1,
                                   w1_scale=None,
                                   w2=w2,
                                   w2_scale=None,
                                   group_list=group_list,
                                   dynamic_scale=None,
                                   group_list_type=1,
                                   w1_scale_bias=None,
                                   w2_scale_bias=None,
                                   topk_scales=topk_scales,
                                   with_quant=False)

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_swiglu.assert_called_once()

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)

    @patch('vllm_ascend.ops.moe.moe_mlp.get_forward_context')
    @patch('torch_npu.npu_grouped_matmul')
    @patch('torch_npu.npu_swiglu')
    @patch('torch_npu.npu_dynamic_quant')
    def test_unified_apply_mlp_with_quantization_and_dynamic_scale(
            self, mock_npu_dynamic_quant, mock_npu_swiglu,
            mock_npu_grouped_matmul, mock_get_forward_context):

        mock_forward_context = MagicMock()
        mock_forward_context.with_quant = True
        mock_forward_context.fused_moe_state = "NOT_MC2"
        mock_get_forward_context.return_value = mock_forward_context

        mock_npu_grouped_matmul.side_effect = [[
            torch.randn(10, 40, dtype=torch.bfloat16)
        ], [torch.randn(10, 20, dtype=torch.bfloat16)]]

        mock_npu_swiglu.return_value = torch.randn(10,
                                                   40,
                                                   dtype=torch.bfloat16)

        mock_npu_dynamic_quant.return_value = (torch.randint(-128,
                                                             127, (10, 40),
                                                             dtype=torch.int8),
                                               torch.rand(10,
                                                          1,
                                                          dtype=torch.float32))

        hidden_states = torch.randn(10, 20, dtype=torch.bfloat16)
        w1 = torch.randn(5, 20, 40, dtype=torch.bfloat16)
        w1_scale = torch.randn(5, 40, dtype=torch.bfloat16)
        w2 = torch.randn(5, 40, 20, dtype=torch.bfloat16)
        w2_scale = torch.randn(5, 20, dtype=torch.bfloat16)
        w1_scale_bias = torch.randn(5, 40, dtype=torch.bfloat16)
        w2_scale_bias = torch.randn(5, 20, dtype=torch.bfloat16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        provided_dynamic_scale = torch.rand(10, 1, dtype=torch.float32)

        result = unified_apply_mlp(hidden_states=hidden_states,
                                   w1=w1,
                                   w1_scale=w1_scale,
                                   w2=w2,
                                   w2_scale=w2_scale,
                                   group_list=group_list,
                                   dynamic_scale=provided_dynamic_scale,
                                   group_list_type=1,
                                   w1_scale_bias=w1_scale_bias,
                                   w2_scale_bias=w2_scale_bias,
                                   topk_scales=None,
                                   with_quant=True)

        mock_get_forward_context.assert_called()

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_swiglu.assert_called_once()
        mock_npu_dynamic_quant.assert_called_once()

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.bfloat16)

    @patch('vllm_ascend.ops.moe.moe_mlp.is_310p')
    @patch('torch_npu.npu_grouped_matmul')
    @patch('torch_npu.npu_swiglu')
    @patch('torch_npu.npu_dynamic_quant')
    def test_unified_apply_mlp_without_quantization_310p(
            self, mock_npu_dynamic_quant, mock_npu_swiglu,
            mock_npu_grouped_matmul, mock_is_310p):
        mock_is_310p.return_value = True

        mock_gmm1_out = torch.randn(10, 40, dtype=torch.float16)
        mock_gmm2_out = torch.randn(10, 20, dtype=torch.float16)
        mock_npu_grouped_matmul.side_effect = [[mock_gmm1_out],
                                               [mock_gmm2_out]]

        mock_npu_swiglu.return_value = torch.randn(10, 40, dtype=torch.float16)

        mock_npu_dynamic_quant.return_value = (MagicMock(), MagicMock())

        hidden_states = torch.randn(10, 20, dtype=torch.float16)
        w1 = torch.randn(5, 20, 40, dtype=torch.float16)
        w2 = torch.randn(5, 40, 20, dtype=torch.float16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        topk_scales = torch.randn(10, 1, dtype=torch.float16)

        result = unified_apply_mlp(hidden_states=hidden_states,
                                   w1=w1,
                                   w1_scale=None,
                                   w2=w2,
                                   w2_scale=None,
                                   group_list=group_list,
                                   dynamic_scale=None,
                                   group_list_type=1,
                                   w1_scale_bias=None,
                                   w2_scale_bias=None,
                                   topk_scales=topk_scales,
                                   with_quant=False)

        mock_is_310p.assert_called_once()

        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_swiglu.assert_called_once()

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.float16)

    @patch("vllm_ascend.ops.moe.moe_mlp.get_forward_context")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    @patch("torch_npu.npu_grouped_matmul_swiglu_quant")
    @patch("torch_npu.npu_dynamic_quant")
    def test_unified_apply_mlp_with_quantization_and_fusion_mlp(
            self, mock_npu_dynamic_quant, mock_npu_grouped_matmul_swiglu_quant,
            mock_npu_swiglu, mock_npu_grouped_matmul,
            mock_get_forward_context):

        mock_forward_context = MagicMock()
        mock_forward_context.with_quant = True
        mock_forward_context.fused_moe_state = "NOT_MC2"
        mock_get_forward_context.return_value = mock_forward_context

        mock_npu_grouped_matmul_swiglu_quant.return_value = (torch.randint(
            -128, 127, (10, 40),
            dtype=torch.int8), torch.rand(
                10, 1,
                dtype=torch.float32), torch.rand(10, 1, dtype=torch.float32))
        mock_npu_grouped_matmul.side_effect = [[
            torch.randn(10, 20, dtype=torch.bfloat16)
        ]]
        mock_npu_swiglu.return_value = torch.randn(10,
                                                   40,
                                                   dtype=torch.bfloat16)
        mock_npu_dynamic_quant.return_value = (torch.randint(-128,
                                                             127, (10, 40),
                                                             dtype=torch.int8),
                                               torch.rand(10,
                                                          1,
                                                          dtype=torch.float32))

        hidden_states = torch.randn(10, 20, dtype=torch.bfloat16)
        w1 = torch.randn(5, 20, 40, dtype=torch.bfloat16)
        w1_scale = torch.randn(5, 40, dtype=torch.bfloat16)
        w2 = torch.randn(5, 40, 20, dtype=torch.bfloat16)
        w2_scale = torch.randn(5, 20, dtype=torch.bfloat16)
        w1_scale_bias = torch.randn(5, 40, dtype=torch.bfloat16)
        w2_scale_bias = torch.randn(5, 20, dtype=torch.bfloat16)
        group_list = torch.tensor([2, 4, 6, 8, 10], dtype=torch.int64)
        provided_dynamic_scale = torch.rand(10, 1, dtype=torch.float32)

        result = unified_apply_mlp(hidden_states=hidden_states,
                                   w1=w1,
                                   w1_scale=w1_scale,
                                   w2=w2,
                                   w2_scale=w2_scale,
                                   group_list=group_list,
                                   dynamic_scale=provided_dynamic_scale,
                                   group_list_type=1,
                                   w1_scale_bias=w1_scale_bias,
                                   w2_scale_bias=w2_scale_bias,
                                   topk_scales=None,
                                   with_quant=True,
                                   fusion=True)

        mock_get_forward_context.assert_called()
        mock_npu_grouped_matmul.assert_called_once()
        mock_npu_grouped_matmul_swiglu_quant.assert_called_once()

        self.assertTrue(mock_forward_context.with_quant)
        self.assertEqual(result.shape, hidden_states.shape)
        self.assertEqual(result.dtype, torch.bfloat16)
