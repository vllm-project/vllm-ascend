from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import PytestBase
from vllm_ascend.torchair.models.qwen3_moe import CustomSparseMoeBlock


class TestCustomSparseMoeBlock(PytestBase):

    @pytest.fixture
    def setup_csmb(self, mocker: MockerFixture):
        config = PretrainedConfig(num_experts=64,
                                  hidden_size=2048,
                                  num_experts_per_tok=2,
                                  moe_intermediate_size=1408,
                                  norm_topk_prob=True)
        mocker.patch(
            'vllm_ascend.torchair.models.qwen3_moe.get_tensor_model_parallel_world_size',
            return_value=10)
        mocker.patch(
            'vllm.model_executor.layers.linear.ReplicatedLinear.__init__',
            return_value=None)
        mocker.patch(
            'vllm_ascend.torchair.ops.torchair_fused_moe.TorchairAscendFusedMoE.__init__',
            return_value=None)

        tp_group = Mock(spec=GroupCoordinator)
        tp_group.rank_in_group = 0
        tp_group.world_size = 1
        tp_group.device_group = Mock()

        dp_group = Mock(spec=GroupCoordinator)
        dp_group.rank_in_group = 0
        dp_group.world_size = 1

        ep_group = Mock(spec=GroupCoordinator)
        ep_group.rank_in_group = 0
        ep_group.world_size = 1

        mocker.patch('vllm_ascend.torchair.models.qwen3_moe.get_tp_group',
                     return_value=tp_group)
        mocker.patch('vllm_ascend.torchair.models.qwen3_moe.get_dp_group',
                     return_value=dp_group)
        mocker.patch('vllm_ascend.torchair.models.qwen3_moe.get_ep_group',
                     return_value=ep_group)
        ascend_config = mocker.MagicMock()
        ascend_config.max_num_batched_tokens = 2048
        ascend_config.max_model_len = 1024
        mocker.patch("vllm_ascend.utils.get_ascend_config",
                     return_value=ascend_config)

        custom_moe_block = CustomSparseMoeBlock(config, None, "")
        return custom_moe_block

    def test_init(self, mocker: MockerFixture, setup_csmb):
        custom_moe_block = setup_csmb
        assert isinstance(custom_moe_block, CustomSparseMoeBlock)
