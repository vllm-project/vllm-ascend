import pytest
from unittest.mock import Mock, MagicMock, patch
from vllm_ascend import torchair_mtp_proposer
from vllm.config import VllmConfig

import torch

def vllm_version_is(version):
    return version == "0.11.0"

import sys
sys.modules[__name__].vllm_version_is = vllm_version_is


class TestTorchairMtpProposer(PytestBase):

    @pytest.fixture
    def setup_torchair_mtp_proposer(self, mocker: pytest.MockerFixture):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.device_config = MagicMock()
        vllm_config.device_config.device = torch.device("npu:0")
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.draft_model_config = MagicMock()
        vllm_config.speculative_config.draft_model_config.dtype = torch.float16
        vllm_config.load_config = MagicMock()

        device = torch.device("npu:0")
        runner = MagicMock()

        mocker.patch("vllm_ascend.torchair_mtp_proposer.__init__",
                     return_value=None)
        
        if vllm_version_is("0.11.0"):
            mock_set_default_dtype = mocker.patch(
                'vllm.model_executor.model_loader.utils.set_default_torch_dtype'
            )
        else:
            mock_set_default_dtype = mocker.patch(
                'vllm.utls.torch_utils.set_default_torch_dtype'
            )
        mock_set_default_dtype.return_value.__enter__.return_value = None

        mock_model_loader = MagicMock()
        mocker.patch("vllm.model_executor.model_loader.get_model_loader",
                     return_value=mock_model_loader)
        mock_layers = {"target_attn_layer_1": Mock(), "draft_attn_layer_2": Mock()}
        mocker.patch("vllm.config.get_layers_from_vllm_config",
                     return_value=mock_layers)
        mock_set_current = mocker.patch("vllm.config.set_current_vllm_config")
        mock_set_current.return_value.__enter__.return_value = None
        mock_torchair_deepseek_mtp = MagicMock()
        mock_torchair_deepseek_mtp.to.return_value = mock_torchair_deepseek_mtp
        mocker.patch("vllm_ascend.torchair.models.torchair_deepseek_mtp.TorchairDeepSeekMTP",
                     return_value=mock_torchair_deepseek_mtp)
        mocker.patch("vllm.model_executor.model_loader.utils.process_weights_after_loading")

        proposer = TorchairMtpProposer(vllm_config, device, runner)
        proposer.vllm_config = vllm_config
        proposer.device = device
        proposer.runner = runner

        return proposer, mock_model_loader, mock_torchair_deepseek_mtp
    
    def test_init(self, setup_torchair_mtp_proposer):
        proposer, _, _, = setup_torchair_mtp_proposer

        assert isinstance(proposer, setup_torchair_mtp_proposer)
        assert proposer.torchair_compiled_model is None
        assert proposer.torchair_compiled_models = {}
        Mock.assert_called_once_with(
            proposer.__class__.__bases__[0],
            proposer.vllm_config,
            proposer.device,
            proposer.runner
        )
    
    def test_load_model(self, setup_torchair_mtp_proposer, mocker: pytest.MockerFixture):
        proposer, mock_model_loader, mock_torchair_deepseek_mtp = setup_torchair_mtp_proposer
        dummpy_model = Mock()

        proposer.load_model(dummpy_model)

        mocker.patch("vllm.model_executor.model_loader.get_model_loader").assert_called_once_with(
            proposer.vllm_config.load_config
        )

        mock_get_layers = mocker.patch("vllm.config.get_layers_from_vllm_config")
        assert mock_get_layers.call_count = 2
        mock_get_layers.assert_called_with(
            proposer.vllm_config,
            mocker.patch("vllm.model_executor.layers.attention_layer_base.AttentionLayerBase")
        )

        mocker.patch("vllm_ascend.torchair.models.torchair_deepseek_mtp.TorchairDeepSeekMTP").assert_called_once_with(
            vllm_config=proposer.vllm_config
        )
        mock_torchair_deepseek_mtp.to.assert_called_once(
            proposer.vllm_config.device_config.device
        )

        assert len(proposer.attn_layer_name) == 1
        mocker_layers_keys = mock_get_layers.return_value.keys()
        assert proposer.attn_layer_name[0] in mocker_layers_keys

        mock_model_loader.get_all_weights.assert_called_once_with(
            proposer.vllm_config.speculative_config.draft_model_config,
            mock_torchair_deepseek_mtp
        )
        mock_torchair_deepseek_mtp.load_weights.assert_called_once_with(
            mock_model_loader.get_all_weights.return_value
        )

        mock_process_weights = mocker.patch("vllm.model_executor.model_loader.utils.process_weights_after_loading")
        mock_process_weights.assert_called_once_with(
            mock_torchair_deepseek_mtp,
            proposer.vllm_config.speculative_config.draft_model_config,
            proposer.vllm_config.device_config.device
        )
