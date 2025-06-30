import unittest
import torch
import llm_datadist 
import zmq  
from unittest.mock import MagicMock, patch
from vllm_ascend.distributed.kv_transfer.simple_pipe import SimplePipe
class TestSimplePipe(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        with patch.dict('os.environ', {'LLMDATADIST_SYNC_CACHE_WAIT_TIME': '5000'}):
            self.mock_data_dist = MagicMock()
            self.mock_data_dist.init.return_value = None
            self.mock_data_dist.link_clusters.return_value = (True, 0)
            self.mock_data_dist.unlink_clusters.return_value = None
            
            patcher = patch('llm_datadist.LLMDataDist', return_value=self.mock_data_dist)
            self.patches = []
            self.patches.append(patcher)
            patcher.start()
            
            patcher = patch('threading.Thread')
            self.patches.append(patcher)
            mock_thread = patcher.start()
            mock_thread.return_value.start.return_value = None
            

            self.context = zmq.Context()
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
    def test_init_success(self,mock_thread):
        
        with patch('llm_datadist.LLMDataDist') as MockLLMDataDist:
            mock_config = self._create_mock_config()
        
            self.pipe = SimplePipe(
                rank=5,
                local_rank=0,
                kv_transfer_config=mock_config,
                hostname="127.0.0.1",
                port_offset=0
            )
        
            self.pipe.router_socket.close()
            mock_data_dist = MockLLMDataDist.return_value
            mock_data_dist.init.return_value = None

    @patch('threading.Thread')
    def test_prepare_data_dist(self,mock_thread):
        with patch('llm_datadist.LLMDataDist') as MockLLMDataDist:
            self.context = zmq.Context()
            self.router_socket = self.context.socket(zmq.ROUTER)
            self.pipe = SimplePipe(
                rank=5,
                local_rank=0,
                kv_transfer_config=self._create_mock_config(),
                hostname="127.0.0.1",
                port_offset=0
            )
            mock_data_dist = MockLLMDataDist.return_value
            mock_data_dist.init.return_value = None
            mock_data_dist.init.assert_called_once()
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
            pipe = SimplePipe(
                rank=5,
                local_rank=0,
                kv_transfer_config=mock_config,
                hostname="127.0.0.1",
                port_offset=0
            )
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
            pipe = SimplePipe(
                rank=0,
                local_rank=0,
                kv_transfer_config=mock_config,
                hostname="127.0.0.1",
                port_offset=0
            )     
            pipe.router_socket.close()

    @patch('threading.Thread')
    def test_create_register_thread_address_is_empty(self, MockThread):
        with patch('llm_datadist.LLMDataDist') as MockLLMDataDist:
            mock_config = self._create_mock_config()
            pipe = SimplePipe(
                rank=5,
                local_rank=0,
                kv_transfer_config=mock_config,
                hostname="127.0.0.1",
                port_offset=0
            )
            self.assertIsNotNone(pipe._register_thread)
            mock_data_dist = MockLLMDataDist.return_value
            mock_data_dist.init.return_value = None
            mock_data_dist.init.assert_called_once()
            pipe.router_socket.close()

    @patch('threading.Thread')
    def test_create_register_thread_address_is_not_empty(self, MockThread):
        with patch('llm_datadist.LLMDataDist') as MockLLMDataDist:
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
            pipe = SimplePipe(
                rank=5,
                local_rank=0,
                kv_transfer_config=mock_config,
                hostname="127.0.0.1",
                port_offset=0
            )
            self.assertIsNotNone(pipe._register_thread)
            mock_data_dist = MockLLMDataDist.return_value
            mock_data_dist.init.return_value = None
            # Assert
            mock_data_dist.init.assert_called_once()
            pipe.router_socket.close()

    @patch('vllm_ascend.distributed.kv_transfer.simple_pipe.SimplePipe')
    def test_should_send_tensor_when_valid_input(self, MockSimplePipe):
        pipe = MockSimplePipe()
        tensor = torch.randn(3, 3)
        tensor_desc = llm_datadist.CacheDesc(num_tensors=1, shape=(3, 3),  data_type=llm_datadist.DataType.DT_FLOAT,seq_len_dim_index=1)
        tensor_key = llm_datadist.CacheKey(1, 0, 1)
        result = pipe.send_tensor(tensor, tensor_desc, tensor_key)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
