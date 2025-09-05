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

from vllm_ascend.ascend_forward_context import _get_fused_moe_state
from vllm_ascend.quantization.quant_config import AscendFusedMoEMethod
from vllm_ascend.torchair.ops.torchair_fused_moe import (
    TorchairAscendFusedMoE, TorchairAscendUnquantizedFusedMoEMethod)
from vllm_ascend.utils import AscendSocVersion, adapt_patch  # noqa E402

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


@pytest.fixture
def mock_dist_env(mocker: MockerFixture):
    # init dist env patch

    with patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.get_world_size', return_value=4), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.get_ep_group', return_value=mock_ep_and_mc2_group(mocker)), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.get_mc2_group', return_value=mock_ep_and_mc2_group(mocker)), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.get_tp_group', return_value=mock_dp_and_tp_group(mocker)), \
         patch('vllm.distributed.parallel_state.get_tp_group', return_value=mock_dp_and_tp_group(mocker)), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.get_dp_group', return_value=mock_dp_and_tp_group(mocker)), \
         patch('vllm.model_executor.layers.fused_moe.layer.get_dp_group', return_value=mock_dp_and_tp_group(mocker)), \
         patch('torch.distributed.all_gather', return_value=MagicMock(return_value=torch.randn(10,32))), \
         patch('torch.distributed.all_to_all_single', return_value=torch.randn(8, 32)), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.tensor_model_parallel_all_reduce',
               return_value=torch.randn(5, 32)), \
         patch('vllm.model_executor.layers.fused_moe.config.get_dp_group',
               return_value=mock_dp_and_tp_group(mocker)), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.get_ascend_config',
               return_value=MagicMock(
                   torchair_graph_config=MagicMock(enabled=False, enable_multistream_moe=False),
                   expert_map_path=None
               )), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.determine_expert_map',
               return_value=(3, torch.tensor([0, 1, 2, -1, -1, -1, -1, -1]))), \
         patch('vllm_ascend.torchair.ops.torchair_fused_moe.get_forward_context',
               return_value=MagicMock(
                   max_tokens_across_dp=10,
                   dp_metadata=MagicMock(cu_tokens_across_dp_cpu=[5, 10])
               )), \
        patch('vllm_ascend.torchair.ops.torchair_fused_moe.get_current_vllm_config',
               return_value=MagicMock(
                   parallel_config=MagicMock(tensor_parallel_size=2),
                   scheduler_config=MagicMock(max_num_seqs=4),
                   model_config=MagicMock(max_model_len=2048)
               )):
        yield


@pytest.fixture
def mock_moe_env(mocker: MockerFixture):
    # init moe env patch

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
    """default moe config"""
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
    return TorchairAscendUnquantizedFusedMoEMethod(moe)


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


class TestTorchairAscendFusedMoe:

    def test_init_no_quant(self, mock_dist_env, default_moe_config):
        layer = TorchairAscendFusedMoE(**default_moe_config)

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

        # check group_topk
        with pytest.raises(AssertionError):
            error_config = default_moe_config.copy()
            error_config['use_grouped_topk'] = True
            layer = TorchairAscendFusedMoE(**error_config)

        # check scoring_func
        with pytest.raises(ValueError):
            error_config = default_moe_config.copy()
            error_config['scoring_func'] = "random"
            layer = TorchairAscendFusedMoE(**error_config)

    def test_init_with_quant(self, mock_dist_env, default_moe_config):
        mock_quant_config = MagicMock()
        mock_quant_method = MockFusedMoEMethod()
        mock_quant_config.get_quant_method.return_value = mock_quant_method
        mock_quant_config.is_layer_skipped_ascend.return_value = False
        with patch("vllm_ascend.quantization.quant_config.get_quant_method"):
            moe = TorchairAscendFusedMoE(**default_moe_config,
                                         quant_config=mock_quant_config)
            assert moe.quant_method is not None
            assert isinstance(moe.quant_method, AscendFusedMoEMethod)

    def test_init_with_mixed_quant(self, mock_dist_env, default_moe_config):
        mock_quant_config = MagicMock()
        mock_quant_method = MockFusedMoEMethod()
        mock_quant_config.get_quant_method.return_value = mock_quant_method
        mock_quant_config.is_layer_skipped_ascend.return_value = True

        moe = TorchairAscendFusedMoE(**default_moe_config,
                                     quant_config=mock_quant_config)

        assert moe.quant_method is not None
        assert isinstance(moe.quant_method,
                          TorchairAscendUnquantizedFusedMoEMethod)

    @pytest.mark.parametrize(
        "others_param",
        [[None,
          MagicMock(return_value=torch.randn(5, 32)), False, 5, None],
         [2, None, False, 5, None], [None, None, True, 5, None],
         [None, None, False, 1, None], [None, None, True, 5, 1],
         [None, None, False, 5, 1]])
    def test_forward(self, mock_dist_env, default_moe_config, others_param):
        """
        1 test has shared_experts
        2 test has top_k
        3 test is_prefill is true
        4 test single num_tokens(decode)
        5 test ep_size is 1 and is_prefill is true
        6 test ep_size is 1 and is_prefill is False
        """
        top_k, shared_experts, is_prefill, num_tokens, ep_size = others_param
        inputs = torch.randn(num_tokens, 32)
        router_logits = torch.randn(num_tokens, 8)
        moe = TorchairAscendFusedMoE(**default_moe_config)

        if ep_size == 1:
            moe.moe_parallel_config.ep_size = 1

        moe.quant_method = MockQuantMethod(shared_experts, num_tokens)
        forward_context = MagicMock(mc2_mask=torch.zeros(num_tokens,
                                                         dtype=torch.bool),
                                    padded_num_tokens=num_tokens)
        with patch(
                "vllm_ascend.torchair.ops.torchair_fused_moe.get_forward_context",
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
        moe = TorchairAscendFusedMoE(**default_moe_config)

        moe.quant_method = MockQuantMethod(None, 5)
        output = moe._forward_ms_fused_moe_comp(inputs,
                                                router_logits,
                                                is_prefill=False,
                                                real_top_k=1)

        moe.quant_method.apply.assert_called_once()

        assert output.shape == (5, 32)


class TestTorchairAscendUnquantizedFusedMoEMethod:

    def test_process_weights_after_loading(self, moe_method, mock_dist_env):
        layer = MagicMock()
        layer.w13_weight.data = torch.randn(16, 32)
        layer.w2_weight.data = torch.randn(16, 32)

        moe_method.process_weights_after_loading(layer)

        assert isinstance(layer.w13_weight, torch.nn.Parameter)
        assert isinstance(layer.w2_weight, torch.nn.Parameter)
        assert not layer.w13_weight.requires_grad
        assert not layer.w2_weight.requires_grad

    @pytest.mark.parametrize("others_param",
                             [[256, 4], [128, 1], [128, 1], [128, 4]])
    def test_apply_without_expert_map(self, moe_method, mock_dist_env,
                                      mock_moe_env, others_param):
        """
        1 test is_deepseek_v3_r1=true and use fused_experts_with_all2all
        2 test use_select_experts and fused_experts
        3 test use select_gating_topk_softmax_experts and fused_experts
        4 test use select_experts and fused_experts_with_all2all_buffer
        """
        global_num_experts, ep_size = others_param
        is_prefill = False
        is_deepseek_v3_r1 = global_num_experts == 256
        forward_context = MagicMock(fused_moe_state=_get_fused_moe_state(
            ep_size, is_prefill, is_deepseek_v3_r1))
        with patch(
                "vllm_ascend.torchair.ops.torchair_fused_moe.get_forward_context",
                return_value=forward_context):
            moe_method.ep_size = ep_size
            x = torch.randn(8, 2, 2)
            router_logits = torch.randn(8, 8)
            layer = MagicMock()
            layer.w13_weight = torch.randn(8, 16, 1)
            layer.w2_weight = torch.randn(16, 8, 1)
            result = moe_method.apply(layer=layer,
                                      x=x,
                                      router_logits=router_logits,
                                      top_k=2,
                                      renormalize=True,
                                      global_num_experts=global_num_experts,
                                      is_prefill=is_prefill)

            if ep_size == 1:
                assert result.shape == (16, 2)
            else:
                assert result.shape == x.shape

    @pytest.mark.parametrize("others_param", [16, 1, 4])
    def test_apply_with_expert_map(self, moe_method, mock_dist_env,
                                   mock_moe_env, others_param):
        """
        1 test use_select_experts and use fused_expters_with_mc2
        2 test use_select_experts and fused_experts_with_all2all_buffer
        3 test use_select_experts and fused_experts_with_all2all
        4 test use_select_experts and fused_experts
        """
        ep_size = others_param
        is_prefill = False
        forward_context = MagicMock(
            fused_moe_state=_get_fused_moe_state(ep_size, is_prefill, True))
        with patch("vllm_ascend.torchair.ops.torchair_fused_moe.get_forward_context", return_value=forward_context), \
             patch("vllm_ascend.torchair.ops.torchair_fused_moe.get_ascend_soc_version", return_value=AscendSocVersion.A3):
            expert_map = torch.tensor([0, 1, 2, -1, -1, -1, -1, -1])
            moe_method.ep_size = ep_size
            x = torch.randn(8, 2, 2)
            if ep_size == 1:
                x = x.view(-1, 2)
            router_logits = torch.randn(8, 8)
            layer = MagicMock()
            layer.w13_weight = torch.randn(8, 16, 1)
            layer.w2_weight = torch.randn(16, 8, 1)
            result = moe_method.apply(layer=layer,
                                      x=x,
                                      router_logits=router_logits,
                                      top_k=2,
                                      renormalize=True,
                                      global_num_experts=128,
                                      expert_map=expert_map,
                                      is_prefill=is_prefill)

            if ep_size == 16 or ep_size == 1:
                assert result.shape == (16, 2)
            else:
                assert result.shape == x.shape
