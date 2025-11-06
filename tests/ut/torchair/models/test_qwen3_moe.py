import pytest
from pytest_mock import MockerFixture
from transformers import PretrainedConfig

from tests.ut.base import PytestBase
from vllm_ascend.torchair.models.qwen3_moe import CustomSparseMoeBlock


class TestCustomSparseMoeBlock(PytestBase):

    @pytest.fixture
    def setup_csmb(self, mocker: MockerFixture):
        config = PretrainedConfig(vocab_size=1000,
                                  hidden_size=768,
                                  rms_norm_eps=1e-5)
        mocker.patch(
            'vllm_ascend.torchair.models.qwen3_moe.get_tensor_model_parallel_world_size',
            return_value=1)
        mocker.patch(
            'vllm.model_executor.layers.linear.ReplicatedLinear.__init__',
            return_value=None)
        mocker.patch(
            'vllm_ascend.torchair.ops.torchair_fused_moe.TorchairAscendFusedMoE.__init__',
            return_value=None)
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
