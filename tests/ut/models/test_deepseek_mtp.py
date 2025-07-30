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

    @pytest.mark.parametrize('kv_caches, inputs_embeds', [
        (torch.tensor([[[0.1, 0.2, 0.3]]])), torch.tensor([[0.1, 0.2, 0.3]]),
        (None, None),
    ])
    def test_forward(self, mocker: MockFixture, setup_predictor):
        predictor = setup_predictor
        mock_layer = mocker.MagicMock()
        mock_layer.return_value = torch.tensor([1.0, 2.0, 3.0])
        predictor.layers_list = [mock_layer]

        # todo: need or not?
        # predictor.num_mtp_layers = 1
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
        mocker.patch(
            "vllm_ascend.models.deepseek_mtp.CustomDeepSeekMultiTokenPredictorLayer.__call__",
            return_value=None
        )
        output = predictor.forward(
            input_ids,
            positions,
            kv_caches,
            None,
            None,
            inputs_embeds,
            0
        )
        mock_layer.assert_called_once()
        assert torch.allclose(output, torch.tensor([1.0, 2.0, 3.0]))


    def test_compute_logits(self, mocker: MockFixture, setup_predictor):
        hidden_states = torch.tensor([[1, 2, 3], [4, 5, 6]])
        predictor = setup_predictor

        mock_layer = mocker.MagicMock()
        mock_layer.return_value = torch.tensor([1.0, 2.0, 3.0])
        predictor.layers_list = [mock_layer]
        mocker.patch("torch.nn.Module.__setattr__", return_value=None)
        mocker.patch("torch.nn.Module.__getattr__", return_value=None)
        mocker.patch("torch.nn.Module.__delattr__", return_value=None)
        mocker.patch(
            "vllm.model_executor.layers.logits_processor.LogitsProcessor.__init__",
            return_value=None
        )
        predictor.logits_processor.return_value = torch.tensor([1.0, 2.0, 3.0])

        result_logits = predictor.compute_logits(hidden_states, None)
        predictor.logits_processor.assert_called_once()
        assert torch.allclose(result_logits, torch.tensor([1.0, 2.0, 3.0]))


class TestCustomDeepSeekMTP(PytestBase):

    @pytest.fixture
    def setup_mtp(self, mocker: MockFixture):
        vllm_config = mocker.MagicMock()
        vllm_config.model_config.hf_config.num_nextn_predict_layers = 12
        vllm_config.model_config.hf_config.num_hidden_layers = 3
        vllm_config.cache_config = mocker.MagicMock()
        vllm_config.quant_config = mocker.MagicMock()

        mocker.patch("torch.nn.Module.__setattr__", return_value=None)
        mocker.patch("torch.nn.Module.__getattr__", return_value=None)
        mocker.patch("torch.nn.Module.__delattr__", return_value=None)
        mocker.patch("vllm.model_executor.layers.sampler.get_sampler", return_value=None)
        mocker_deepseek_mtp_predictor = mocker.patch(
            "vllm_ascend.models.deepseek_mtp.CustomDeepSeekMultiTokenPredictorLayer.__call__",
            return_value=None
        )

        mtp = CustomDeepSeekMTP(vllm_config=vllm_config)
        mocker_deepseek_mtp_predictor.assert_called_once()
        return mtp

    def test_init(self, mocker: MockFixture, setup_mtp):
        mtp = setup_mtp
        assert isinstance(mtp, CustomDeepSeekMTP)

    def test_forward(self, mocker: MockFixture, setup_mtp):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
        kv_caches = [torch.tensor([[0.1, 0.2, 0.3]])]
        previous_hidden_states = torch.tensor([[1.0, 2.0, 3.0]])
        inputs_embeds = torch.tensor([[0.1, 0.2, 0.3]])
        spec_step_idx = 0

        mtp = setup_mtp
        mtp.model.return_value = torch.tensor([1.0, 2.0, 3.0])
        output = setup_mtp.forward(
            input_ids,
            positions,
            kv_caches,
            None,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx
        )
        assert torch.allclose(output, torch.tensor([1.0, 2.0, 3.0]))