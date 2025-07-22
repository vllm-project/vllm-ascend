from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.distributed.kv_transfer.simple_pipe import SimplePipe


class TestSimplePipe(TestBase):

    @classmethod
    def _create_mock_config(self):
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
        return mock_config

    @patch('threading.Thread')
    @patch('llm_datadist.LLMDataDist')
    def test_init_success(self, mock_thread, MockLLMDataDist):

        mock_config = self._create_mock_config()

        self.pipe = SimplePipe(rank=5,
                               local_rank=0,
                               kv_transfer_config=mock_config,
                               hostname="127.0.0.1",
                               port_offset=0)

        self.pipe.router_socket.close()

    @patch('threading.Thread')
    @patch('llm_datadist.LLMDataDist')
    def test_prepare_data_dist(self, mock_thread, MockLLMDataDist):
        self.pipe = SimplePipe(rank=5,
                               local_rank=0,
                               kv_transfer_config=self._create_mock_config(),
                               hostname="127.0.0.1",
                               port_offset=0)
        mock_data_dist = MockLLMDataDist.return_value
        mock_data_dist.init.return_value = None
        self.pipe.router_socket.close()

    def test_init_with_invalid_kv_role(self):
        with self.assertRaises(NotImplementedError):
            mock_config = MagicMock()
            mock_config.kv_role = "err_role"
            mock_config.kv_connector_extra_config = {
                "prefill_device_ips": ["127.0.0.1"],
                "decode_device_ips": ["127.0.0.1"],
                "llmdatadist_comm_port": 26000,
                "http_port": 8000,
                "proxy_ip": "127.0.0.1",
                "proxy_port": "8000",
                "port": 5500
            }
            pipe = SimplePipe(rank=5,
                              local_rank=0,
                              kv_transfer_config=mock_config,
                              hostname="127.0.0.1",
                              port_offset=0)
            pipe.router_socket.close()

    def test_init_with_missing_device_ips(self):
        with self.assertRaises(ValueError):
            mock_config = MagicMock()
            mock_config.kv_role = "kv_producer"
            mock_config.kv_connector_extra_config = {
                "llmdatadist_comm_port": 26000,
                "http_port": 8000,
                "proxy_ip": "127.0.0.1",
                "proxy_port": "8000",
                "port": 5500
            }
            pipe = SimplePipe(rank=0,
                              local_rank=0,
                              kv_transfer_config=mock_config,
                              hostname="127.0.0.1",
                              port_offset=0)
            pipe.router_socket.close()

    @patch('threading.Thread')
    @patch('llm_datadist.LLMDataDist')
    def test_create_register_thread_address_is_empty(self, MockThread,
                                                     MockLLMDataDist):

        mock_config = self._create_mock_config()
        pipe = SimplePipe(rank=5,
                          local_rank=0,
                          kv_transfer_config=mock_config,
                          hostname="127.0.0.1",
                          port_offset=0)
        self.assertIsNotNone(pipe._register_thread)
        mock_data_dist = MockLLMDataDist.return_value
        mock_data_dist.init.return_value = None
        pipe.router_socket.close()

    @patch('threading.Thread')
    @patch('llm_datadist.LLMDataDist')
    def test_create_register_thread_address_is_not_empty(
            self, MockThread, MockLLMDataDist):
        mock_config = MagicMock()
        mock_config.kv_role = "kv_producer"
        mock_config.kv_connector_extra_config = {
            "prefill_device_ips": [""],
            "decode_device_ips": [""],
            "llmdatadist_comm_port": 26000,
            "http_port": 8000,
            "proxy_ip": "127.0.0.1",
            "proxy_port": "8000",
            "port": 5500
        }
        pipe = SimplePipe(rank=5,
                          local_rank=0,
                          kv_transfer_config=mock_config,
                          hostname="127.0.0.1",
                          port_offset=0)
        self.assertIsNotNone(pipe._register_thread)
        mock_data_dist = MockLLMDataDist.return_value
        mock_data_dist.init.return_value = None
        pipe.router_socket.close()

    @patch('vllm_ascend.distributed.kv_transfer.simple_pipe.SimplePipe')
    @patch('llm_datadist.LLMDataDist')
    def test_should_send_tensor_when_valid_input(self, MockSimplePipe,
                                                 MockLLMDataDist):
        pipe = MockSimplePipe()
        tensor = torch.randn(3, 3)
        tensor_desc = MockLLMDataDist.CacheDesc(
            num_tensors=1,
            shape=(3, 3),
            data_type=MockLLMDataDist.DataType.DT_FLOAT,
            seq_len_dim_index=1)
        tensor_key = MockLLMDataDist.CacheKey(1, 0, 1)
        result = pipe.send_tensor(tensor, tensor_desc, tensor_key)
        self.assertIsNotNone(result)
