from unittest.mock import MagicMock, Mock

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.config import CacheConfig, VllmConfig

from tests.ut.base import PytestBase
from vllm_ascend.torchair.torchair_mtp_proposer import TorchairMtpProposer


class TestTorchairMtpProposer(PytestBase):

    @pytest.fixture
    def setup_torchair_mtp_proposer(self, mocker: MockerFixture):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.device_config = MagicMock()
        vllm_config.device_config.device = torch.device("cpu")
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.draft_model_config = MagicMock()
        vllm_config.speculative_config.draft_model_config.dtype = torch.float16
        vllm_config.speculative_config.method = "deepseek_mtp"
        vllm_config.speculative_config.num_speculative_tokens = 5
        vllm_config.load_config = MagicMock()
        cache_config = CacheConfig(block_size=16)
        vllm_config.cache_config = cache_config
        vllm_config.scheduler_config = MagicMock(max_num_batched_tokens=1024,
                                                 max_num_seqs=64)

        device = torch.device("cpu")
        runner = MagicMock()
        runner.pcp_size = 1
        runner.dcp_size = 1
        runner.pcp_rank = 0
        runner.max_num_tokens = 1024
        runner.max_num_reqs = 10
        runner._use_aclgraph.return_value = True

        mocker.patch(
            "vllm_ascend.torchair.torchair_mtp_proposer.MtpProposer.__init__",
            return_value=None)
        mock_set_default_dtype = mocker.patch(
            'vllm.utils.torch_utils.set_default_torch_dtype')
        mock_set_default_dtype.return_value.__enter__.return_value = None

        mock_model_loader = MagicMock()
        mocker.patch("vllm.model_executor.model_loader.get_model_loader",
                     return_value=mock_model_loader)
        mock_layers = {
            "target_attn_layer_1": Mock(),
            "draft_attn_layer_2": Mock()
        }
        mocker.patch("vllm.config.get_layers_from_vllm_config",
                     return_value=mock_layers)
        mock_set_current = mocker.patch("vllm.config.set_current_vllm_config")
        mock_set_current.return_value.__enter__.return_value = None
        mock_torchair_deepseek_mtp = MagicMock()
        mock_torchair_deepseek_mtp.to.return_value = mock_torchair_deepseek_mtp
        mocker.patch(
            "vllm_ascend.torchair.models.torchair_deepseek_mtp.TorchairDeepSeekMTP",
            return_value=mock_torchair_deepseek_mtp)
        mocker.patch(
            "vllm.model_executor.model_loader.utils.process_weights_after_loading"
        )

        proposer = TorchairMtpProposer(vllm_config, device, runner)
        proposer.vllm_config = vllm_config
        proposer.device = device
        proposer.runner = runner
        proposer.speculative_config = vllm_config.speculative_config
        proposer.draft_model_config = vllm_config.speculative_config.draft_model_config
        proposer.method = vllm_config.speculative_config.method

        return proposer, mock_model_loader, mock_torchair_deepseek_mtp

    def test_init(self, setup_torchair_mtp_proposer):
        proposer, _, _, = setup_torchair_mtp_proposer
        assert isinstance(proposer, TorchairMtpProposer)
