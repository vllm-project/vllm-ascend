from unittest.mock import MagicMock, patch

import torch
from vllm.config import VllmConfig
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from tests.ut.base import TestBase
from vllm_ascend.distributed.kv_transfer.simple_buffer import SimpleBuffer
from vllm_ascend.distributed.kv_transfer.simple_connector import \
    SimpleConnector
from vllm_ascend.distributed.kv_transfer.simple_pipe import SimplePipe


class TestSimpleConnector(TestBase):

    def setUp(self):
        self.mock_pipe = MagicMock(spec=SimplePipe)
        self.mock_buffer = MagicMock(spec=SimpleBuffer)

        patcher = patch(
            'vllm_ascend.distributed.kv_transfer.simple_buffer.SimpleBuffer')
        self.addCleanup(patcher.stop)
        self.MockSimpleBuffer = patcher.start()
        self.MockSimpleBuffer.return_value = self.mock_buffer

    def _create_mock_config(self, kv_role):
        mock_config = MagicMock()
        mock_config.kv_role = "kv_producer"
        mock_config.kv_connector_extra_config = {
            "prefill_device_ips": ["127.0.0.1"],
            "decode_device_ips": ["127.0.0.1"],
            "llmdatadist_comm_port": 26000,
            "http_port": 8000,
            "proxy_ip": "127.0.0.1",
            "proxy_port": "8000",
            "port": 5500
        }
        mock_config.kv_port = 5500
        self.mock_config = MagicMock(spec=VllmConfig)
        self.mock_config.kv_transfer_config.is_kv_producer = True
        self.mock_config.model_config.hf_config.hidden_size = 128
        self.mock_config.model_config.hf_config.num_attention_heads = 8
        self.mock_config.model_config.hf_config.num_key_value_heads = 8
        self.mock_config.model_config.hf_config.qk_rope_head_dim = 16
        self.mock_config.model_config.hf_config.kv_lora_rank = 16
        self.mock_config.model_config.is_deepseek_mla = True
        # 模拟 parallel_config
        self.mock_config.parallel_config = MagicMock()
        self.mock_config.parallel_config.tensor_parallel_size = 1
        self.mock_config.parallel_config.get_num_layers.return_value = 4

        if kv_role == "kv_producer":
            self.mock_config.kv_transfer_config.kv_role = "kv_producer"
        else:
            self.mock_config.kv_transfer_config.kv_role = "kv_consumer"
        return mock_config

    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimplePipe')
    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimpleBuffer')
    @patch('llm_datadist.LLMDataDist')
    def test_select_init(self, mock_pipe, mock_buffer, MockLLMDataDist):
        """Test select method when buffer retrieval succeeds."""
        connector = SimpleConnector(
            rank=0,
            local_rank=0,
            config=self._create_mock_config("kv_producer"))
        assert connector.producer_data_pipe is not None
        assert connector.producer_buffer is not None
        mock_data_dist = MockLLMDataDist.return_value
        mock_data_dist.init.return_value = None

    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimplePipe')
    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimpleBuffer')
    @patch('llm_datadist.LLMDataDist')
    def test_select_select(self, mock_pipe, mock_buffer, MockLLMDataDist):

        connector = SimpleConnector(
            rank=0,
            local_rank=0,
            config=self._create_mock_config("kv_consumer"))
        connector.consumer_data_pipe = mock_pipe
        connector.consumer_buffer = mock_buffer
        assert connector.consumer_data_pipe is not None
        assert connector.consumer_buffer is not None
        input_tokens = torch.tensor([1, 2, 3])
        roi = torch.tensor([True, True, True])
        req_id = "test_req"
        connector.select(input_tokens, roi, req_id)

    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimplePipe')
    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimpleBuffer')
    @patch('llm_datadist.LLMDataDist')
    def test_insert(self, mock_pipe, mock_buffer, MockLLMDataDist):
        """Test insert operation"""
        connector = SimpleConnector(
            rank=0,
            local_rank=0,
            config=self._create_mock_config("kv_producer"))

        connector.producer_buffer = mock_buffer

        input_tokens = torch.randint(0, 1000, (5, ))
        roi = torch.ones_like(input_tokens, dtype=torch.bool)
        keys = torch.randn(3, 5, 1, 96)
        values = torch.randn(3, 5, 1, 96)
        hidden = torch.randn(5, 768)
        req_id = "test_req"

        connector.insert(input_tokens, roi, keys, values, hidden, req_id)

        mock_buffer.insert.assert_called_once_with(input_tokens, roi, keys,
                                                   values, hidden, req_id)

    @patch.object(SimpleConnector, 'insert')
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimplePipe')
    @patch('vllm_ascend.distributed.kv_transfer.simple_connector.SimpleBuffer')
    @patch('llm_datadist.LLMDataDist')
    def test_send_kv_caches_and_hidden_states(self, mock_pipe, mock_buffer,
                                              MockLLMDataDist, mock_insert,
                                              mock_rank):
        """Test sending KV caches and hidden states"""
        connector = SimpleConnector(
            rank=0,
            local_rank=0,
            config=self._create_mock_config("kv_producer"))

        mock_model_executable = MagicMock()
        mock_model_executable.model.start_layer = 0
        mock_model_executable.model.end_layer = 3

        mock_model_input = MagicMock(spec=ModelInputForGPUWithSamplingMetadata)
        mock_model_input.input_tokens = torch.randint(0, 1000, (10, ))
        mock_model_input.attn_metadata.seq_lens = [5, 5]
        mock_model_input.attn_metadata.slot_mapping = torch.randint(
            0, 100, (10, ))
        mock_model_input.attn_metadata.num_prefill_tokens = 10
        mock_model_input.request_ids_to_seq_ids = {"req1": [0], "req2": [1]}

        kv_caches = [torch.randn(2, 100, 1, 96) for _ in range(3)]

        hidden_states = torch.randn(10, 768)

        connector.send_kv_caches_and_hidden_states(mock_model_executable,
                                                   mock_model_input, kv_caches,
                                                   hidden_states)
