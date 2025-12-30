#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import io
import json
import logging
import socket
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.logger

from vllm_ascend.model_loader.netloader.interaction.elastic import (
    ElasticClient, ElasticServer)


# Simulate server's normal response
def mock_server_response(data):
    return json.dumps({
        "label": "JOIN_ACK",
        "content": {
            "name": "mocked_name"
        }
    }).encode("utf-8")


# Simulate server's error response
def mock_server_error_response(data):
    return json.dumps({"label": "JOIN_ACK", "content": None}).encode("utf-8")


# Simulated server's abnormal response
def mock_server_exception_response(data):
    raise Exception("Mocked server exception")


# Test the initialization of ElasticClient
def test_elastic_client_init():
    sources = ["127.0.0.1:12345"]
    device_id = 0
    model_path = "mocked_model_path"
    tp = 1
    pp = 1

    with patch('socket.socket') as mock_socket:
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.recv.return_value = mock_server_response(None)

        mock_socket_instance.getsockname.return_value = ('127.0.0.1', 12346)
        mock_socket_instance.__enter__.return_value = mock_socket_instance

        with ElasticClient(sources, device_id, model_path, tp, pp) as client:
            assert client.server_addr == "127.0.0.1"
            assert client.server_port == 12345
            assert client.ack == ("mocked_name", 12346)
        mock_socket_instance.close.assert_called_once()


# Test the register method of ElasticClient
def test_elastic_client_register():
    sources = ["127.0.0.1:12345"]
    device_id = 0
    model_path = "mocked_model_path"
    tp = 1
    pp = 1

    with patch('socket.socket') as mock_socket:
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.connect.return_value = None
        mock_socket_instance.recv.return_value = mock_server_response(None)

        mock_socket_instance.getsockname.return_value = ('127.0.0.1', 12346)
        mock_socket_instance.__enter__.return_value = mock_socket_instance

        client = ElasticClient(sources, device_id, model_path, tp, pp)
        assert client.register(device_id, model_path, tp,
                               pp) == ("mocked_name", 12346)


# Test the behavior of the `register` method of ElasticClient when the server returns an error response.
def test_elastic_client_register_error_response():
    sources = ["127.0.0.1:12345"]
    device_id = 0
    model_path = "mocked_model_path"
    tp = 1
    pp = 1

    with patch('socket.socket') as mock_socket:
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.connect.return_value = None
        mock_socket_instance.recv.return_value = mock_server_error_response(
            None)

        with ElasticClient(sources, device_id, model_path, tp, pp) as client:
            with pytest.raises(RuntimeError):
                client.register(device_id, model_path, tp, pp)
        mock_socket_instance.close.assert_called_once()


# Test the behavior of the `register` method of ElasticClient when an exception is thrown on the server.
def test_elastic_client_register_exception():
    sources = ["127.0.0.1:12345"]
    device_id = 0
    model_path = "mocked_model_path"
    tp = 1
    pp = 1

    with patch('socket.socket') as mock_socket:
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        mock_socket_instance.connect.return_value = None
        mock_socket_instance.recv.side_effect = mock_server_exception_response
        mock_socket_instance.__enter__.return_value = mock_socket_instance
        mock_socket_instance.__exit__.return_value = None

        with ElasticClient(sources, device_id, model_path, tp, pp) as client:
            with pytest.raises(RuntimeError):
                client.register(device_id, model_path, tp, pp)
        mock_socket_instance.close.assert_called_once()


class FakeInt8Param:

    def __init__(self, name="param", device="npu", dtype=torch.int8):
        self.dtype = dtype
        self.device = torch.device(device)

    @property
    def data(self):
        return self  # Simulate .data returning self so .cpu() etc. can be chained

    def clone(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        self.device = torch.device("cpu")
        return self


class FakeModel:

    def __init__(self):
        self.params = {
            "param1": MagicMock(dtype=torch.float32),  # This will be ignored
            "param2": FakeInt8Param(),  # This simulates a real int8 param
        }

    def named_parameters(self):
        return self.params.items()


@pytest.fixture
def mock_model():
    return FakeModel()


@pytest.fixture
def server_config():
    return {
        "addr": "127.0.0.1",
        "port": 8080,
        "model": MagicMock(),
        "device_id": 0,
        "model_path": "/test/model",
        "tp": 1,
        "pp": 1,
        "int8_cache": "dram",
        'int8_cache_name': None
    }


# Test server initialization
def test_server_initialization(server_config, mock_model):
    server_config["model"] = mock_model
    with patch("socket.socket") as mock_socket:
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.DEBUG)
        vllm.logger.logger.addHandler(ch)

        server = ElasticServer(**server_config)

        # Check the socket configuration
        mock_socket.assert_called_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket.return_value.setsockopt.assert_called_with(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mock_socket.return_value.bind.assert_called_with(("127.0.0.1", 8080))
        mock_socket.return_value.listen.assert_called_with(256)

        # Check int8 cache
        assert "param2" in server.original_int8
        assert server.original_int8[
            "param2"].device.type == "cpu"  # Verifying DRAM Cache

        assert server.addr == server_config['addr']
        assert server.port == server_config['port']
        assert server.device_id == server_config['device_id']
        assert server.model_path == server_config['model_path']
        assert server.tp == server_config['tp']
        assert server.pp == server_config['pp']

        # Get captured logs
        log_output = log_capture_string.getvalue()
        vllm.logger.logger.removeHandler(ch)
        log_capture_string.close()

        # Check output
        assert "Server 127.0.0.1:8080 starts" in log_output


# Test the int8 cache option
@pytest.mark.parametrize("cache_option,expected_device", [("dram", "cpu"),
                                                          ("no", None),
                                                          ("invalid", None)])
def test_int8_cache_handling(server_config, mock_model, cache_option,
                             expected_device, caplog):
    server_config["int8_cache"] = cache_option
    server_config["model"] = mock_model

    with patch("socket.socket"):
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.DEBUG)
        vllm.logger.logger.addHandler(ch)

        server = ElasticServer(**server_config)

        log_output = log_capture_string.getvalue()
        vllm.logger.logger.removeHandler(ch)
        log_capture_string.close()

        if cache_option == "invalid":
            assert "int8_cache should be selected in [HBM, DRAM]" in log_output

        if expected_device is None:
            assert len(server.original_int8) == 0
        else:
            assert server.original_int8[
                "param2"].device.type == expected_device


# Test client processing
def test_client_handler_valid_join(server_config, mock_model):
    server_config["model"] = mock_model
    with patch("vllm_ascend.model_loader.netloader.interaction.elastic.P2PSend"
               ) as mock_p2p_send:

        # Create a simulated connection
        mock_conn = MagicMock()
        mock_addr = ("192.168.1.1", 12345)

        # Configuring Client Data
        valid_data = {
            "label": "JOIN",
            "content": {
                "device_id": 0,
                "model_path": "/test/model",
                "tp": 1,
                "pp": 1,
                "port": 9090
            }
        }
        mock_conn.recv.return_value = json.dumps(valid_data).encode("utf-8")

        # Start the server
        server = ElasticServer(**server_config)
        server.register_handler(mock_conn, mock_addr)

        # Verify response
        expected_ack = {
            "label": "JOIN_ACK",
            "content": {
                "name": "192.168.1.1:12345"
            }
        }
        mock_conn.send.assert_called_once_with(
            json.dumps(expected_ack).encode("utf-8"))
        mock_p2p_send.assert_called_once_with("127.0.0.1", 9090,
                                              "192.168.1.1:12345")
        mock_conn.close.assert_called_once()


# Test mismatched JOIN requests
def test_client_handler_mismatch(server_config):
    with patch("socket.socket"):
        server = ElasticServer(**server_config)
        mock_conn = MagicMock()
        mock_addr = ("192.168.1.1", 12345)

        # Send mismatched data
        mismatch_data = {
            "label": "JOIN",
            "content": {
                "device_id": 1,  # 不匹配的ID
                "model_path": "/wrong/model",
                "tp": 2,
                "pp": 2,
                "port": 9090
            }
        }
        mock_conn.recv.return_value = json.dumps(mismatch_data).encode("utf-8")

        server.register_handler(mock_conn, mock_addr)

        assert isinstance(mismatch_data["content"], dict)

        # Verify response
        expected_ack = {
            "label":
            "JOIN_NACK",
            "content":
            f"Received data {(mismatch_data['content']['device_id'], mismatch_data['content']['model_path'], mismatch_data['content']['tp'], mismatch_data['content']['pp'])} does not consist with this server {(server_config['device_id'], server_config['model_path'], server_config['tp'], server_config['pp'])}"
        }
        mock_conn.send.assert_called_once_with(
            json.dumps(expected_ack).encode("utf-8"))
        mock_conn.close.assert_called_once()


# Test Invalid Request
@pytest.mark.parametrize(
    "invalid_data,should_send",
    [
        (
            {
                "label": "WRONG_LABEL"
            }, True
        ),  # Incorrect label, can be decoded as JSON, but the content is invalid. 
        (
            {
                "content": {
                    "missing_fields": True
                }
            }, True
        ),  # Missing field, can be decoded as JSON, but the content is invalid. 
        ("plain text", False),  # Non-JSON data, json.loads failed 
        (b"invalid_bytes", False)  # Invalid byte, decode or json.loads failed 
    ])
def test_client_handler_invalid_requests(server_config, invalid_data,
                                         should_send):
    with patch("socket.socket"):
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.DEBUG)
        vllm.logger.logger.addHandler(ch)

        with patch("socket.socket"):
            server = ElasticServer(**server_config)
            mock_conn = MagicMock()
            mock_addr = ("192.168.1.1", 12345)

            if isinstance(invalid_data, (str, bytes)):
                mock_conn.recv.return_value = invalid_data if isinstance(
                    invalid_data, bytes) else invalid_data.encode()
            else:
                mock_conn.recv.return_value = json.dumps(invalid_data).encode(
                    "utf-8")

            server.register_handler(mock_conn, mock_addr)

            if should_send:
                expected_ack = {
                    "label":
                    "JOIN_NACK",
                    "content":
                    f"Received data does not contain required fields: {invalid_data}"
                }
                mock_conn.send.assert_called_once_with(
                    json.dumps(expected_ack).encode("utf-8"))
            else:
                mock_conn.send.assert_not_called()

            log_output = log_capture_string.getvalue()
            vllm.logger.logger.removeHandler(ch)
            log_capture_string.close()

            # Any warning in the log is acceptable
            assert "Failed to load" in log_output or "does not contain" in log_output
            mock_conn.close.assert_called_once()


# Test the thread startup.
def test_server_start(server_config):
    with patch("socket.socket"), \
         patch("threading.Thread") as mock_thread:

        handler_thread_instance = mock_thread.return_value

        server = ElasticServer(**server_config)
        server.start()

        # Assert that the correct target parameter was passed when instantiating the Thread instance.
        mock_thread.assert_called_once()
        args, kwargs = mock_thread.call_args
        assert kwargs['target'] == server.elastic_client_handler

        # Check that the daemon attribute is set to True (the attribute value will be recorded after MagicMock assignment).
        assert handler_thread_instance.daemon is True

        # Check if the start() method is called.
        handler_thread_instance.start.assert_called_once()


# Test resource clearing
def test_server_cleanup(server_config):
    with patch("socket.socket") as mock_socket:
        server = ElasticServer(**server_config)
        del server
        mock_socket.return_value.close.assert_called_once()


if __name__ == "__main__":
    pytest.main()
