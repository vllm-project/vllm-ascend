import torch
import pytest
from pytest_mock import MockFixture

from transformers import PreTrainedConfig

from docs.source.conf import release
from examples.disaggregated_prefill_v1.gen_ranktable import local_device_list
from tests.ut.base import PytestBase
from vllm.config import CacheConfig, VllmConfig, ModelConfig
from vllm_ascend.models.deepseek_mtp import (CustomDeepSeekMTP, CustomDeepSeekMultiTokenPredictor, CustomDeepSeekMultiTokenPredictorLayer)

class TestCustomDeepSeekMultiTokenPredictorLayer(PytestBase):

    @pytest.fixture()
    def setup_mtp_layer(self, mocker: MockFixture):
        config = PreTrainedConfig(vocab_size=1000, hidden_size=768, rms_norm_eps=1e-5)
        mocker.patch(
            "vllm.model_executor.layers.vocab_parallel_embedding_layer.VocabParallelEmbedding.__init__",
            return_value=None
        )
        mocker.patch(
            "vllm.model_executor.layers.layernorm.RMSNorm.__init__",
            return_value=None
        )
        mocker.patch(
            "vllm.model_executor.models.deepseek_mtp.SharedHead.__init__",
            return_value=None
        )
        mocker_deepseek_v2_decode_layer = mocker.patch(
            "vllm_ascend.models.deepseek_mtp.CustomDeepSeekV2DecoderLayer.__init__",
            return_value=None
        )

        mtp_layer=CustomDeepSeekMultiTokenPredictorLayer(config, "", None)
        mocker_deepseek_v2_decode_layer.assert_called_once()
        return mtp_layer

    def test_init(self, mocker: MockFixture, setup_mtp_layer):
        mtp_layer = setup_mtp_layer
        assert isinstance(mtp_layer, CustomDeepSeekMultiTokenPredictorLayer)

    def test_forward(self, mocker: MockFixture, setup_mtp_layer):
        mtp_layer = setup_mtp_layer
        mocker.patch("torch.nn.Module.__setattr__", return_value=None)
        mocker.patch("torch.nn.Module.__getattr__", return_value=None)
        mocker.patch("torch.nn.Module.__delattr__", return_value=None)
        mocker.patch("torch.cat", return_value=torch.randn(2, 3, 768))

        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
        kv_cache = torch.randn(2, 3, 768)
        previous_hidden_states = torch.randn(2, 3, 768)
        inputs_embeds = torch.tensor([[1, 2, 3] ])

        output = mtp_layer(
            input_ids,
            positions,
            kv_cache,
            None,
            previous_hidden_states,
            inputs_embeds,
            0
        )
        assert output.shape == (2, 3, 768)


class TestCustomDeepSeekMultiTokenPredictor(PytestBase):

    @pytest.fixture()
    def setup_predictor(self, mocker: MockFixture):
        mock_vllm_config = mocker.MagicMock()
        mock_model_config = mocker.MagicMock()
        mock_hf_config = mocker.MagicMock()
        mock_hf_config.num_hidden_layers = 12
        mock_hf_config.num_nextn_predict_layers = 3
        mock_hf_config.vocab_size = 30000
        mock_model_config.hf_config = mock_hf_config
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.cache_config = CacheConfig()
        mock_vllm_config.quant_config = mocker.MagicMock()
        mocker_mtp_predictor_layer = mocker.patch(
            "vllm_ascend.models.deepseek_mtp.CustomDeepSeekMultiTokenPredictorLayer.__init__",
            return_value=None
        )

        predictor = CustomDeepSeekMultiTokenPredictor(vllm_config=mock_vllm_config)
        mocker_mtp_predictor_layer.assert_called_once()
        return predictor

    def test_init(self, mocker: MockFixture, setup_predictor):
        predictor = setup_predictor
        assert isinstance(predictor, CustomDeepSeekMultiTokenPredictor)

    def test_forward(self, mocker: MockFixture, setup_predictor):
        predictor = setup_predictor
        mock_layer = mocker.MagicMock()
        mock_layer.return_value = torch.randn(2, 3, 768)
