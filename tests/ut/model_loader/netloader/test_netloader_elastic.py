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
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.model_loader.netloader.executor import elastic_load as p2p_elastic_load
from vllm_ascend.model_loader.netloader.executor.elastic_load import (
    P2PSend,
    _cast_packed_int32_via_int8,
    _cast_tensor_to_fractal_nd,
    _cast_tensor_to_fractal_nz,
    _collect_processed_layout_tensors,
    _finalize_hccl_recv_buffer,
    _get_recv_transfer_items,
    _get_send_transfer_items,
    _prepare_hccl_recv_buffer,
    cache_processed_layout_transfer_manifest,
    get_cached_processed_layout_transfer_items,
    register_processed_layout_transfer_items,
    reshape_tensor_to_manifest_shape,
    reshape_transfer_items_to_manifest,
)
from vllm_ascend.model_loader.netloader.interaction import elastic
from vllm_ascend.model_loader.netloader.interaction.elastic import (
    ElasticClient,
    ElasticServer,
    _recv_json_message,
)


# Simulate server's normal response
def mock_server_response(data):
    return json.dumps({"label": "JOIN_ACK", "content": {"name": "mocked_name"}}).encode("utf-8")


# Simulate server's error response
def mock_server_error_response(data):
    return json.dumps({"label": "JOIN_ACK", "content": None}).encode("utf-8")


# Simulated server's abnormal response
def mock_server_exception_response(data):
    raise Exception("Mocked server exception")


@contextmanager
def capture_elastic_logs(level=logging.DEBUG):
    log_capture_string = io.StringIO()
    handler = logging.StreamHandler(log_capture_string)
    handler.setLevel(level)
    original_level = elastic.logger.level
    elastic.logger.setLevel(level)
    elastic.logger.addHandler(handler)
    try:
        yield log_capture_string
    finally:
        elastic.logger.removeHandler(handler)
        elastic.logger.setLevel(original_level)
        log_capture_string.close()


def _mock_client_socket(mock_socket, recv_return=None, recv_side_effect=None, capture_send=None):
    mock_socket_instance = MagicMock()
    mock_socket.return_value = mock_socket_instance
    mock_socket_instance.connect.return_value = None
    if recv_side_effect is not None:
        mock_socket_instance.recv.side_effect = recv_side_effect
    else:
        mock_socket_instance.recv.return_value = recv_return
    if capture_send is not None:
        mock_socket_instance.send.side_effect = lambda data: capture_send.append(json.loads(data.decode()))
    mock_socket_instance.getsockname.return_value = ("127.0.0.1", 12346)
    mock_socket_instance.__enter__.return_value = mock_socket_instance
    mock_socket_instance.__exit__.return_value = None
    return mock_socket_instance


def test_elastic_client_register():
    sources = ["127.0.0.1:12345"]
    sent_payloads: list[dict] = []

    with patch("socket.socket") as mock_socket:
        mock_socket_instance = _mock_client_socket(
            mock_socket,
            recv_return=mock_server_response(None),
            capture_send=sent_payloads,
        )

        with ElasticClient(sources, 0, "mocked_model_path", 1, 1) as client:
            assert client.server_addr == "127.0.0.1"
            assert client.server_port == 12345
            assert client.ack == ("mocked_name", 12346)
            assert client.register(0, "mocked_model_path", 1, 1) == ("mocked_name", 12346)
            assert sent_payloads[-1]["content"]["int8_cache"] == "no"
        mock_socket_instance.close.assert_called_once()


@pytest.mark.parametrize(
    "recv_return,recv_side_effect",
    [
        (mock_server_error_response(None), None),
        (None, mock_server_exception_response),
    ],
)
def test_elastic_client_register_failure(recv_return, recv_side_effect):
    with patch("socket.socket") as mock_socket:
        mock_socket_instance = _mock_client_socket(
            mock_socket,
            recv_return=recv_return,
            recv_side_effect=recv_side_effect,
        )

        with ElasticClient(["127.0.0.1:12345"], 0, "mocked_model_path", 1, 1) as client, pytest.raises(RuntimeError):
            client.register(0, "mocked_model_path", 1, 1)
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


class FakeP2PParam:
    def __init__(self, name):
        self.name = name
        self.device = torch.device("cpu")
        self.shape = (1,)

    def is_contiguous(self):
        return True

    def contiguous(self):
        return f"{self.name}:contiguous"

    def to(self, device):
        return f"{self.name}:to:{device}"


class FakeP2PModel:
    def __init__(self):
        self.params = {
            "weight": FakeP2PParam("weight"),
            "aclnn_input_scale": FakeP2PParam("aclnn_input_scale"),
        }

    def parameters(self):
        return iter(self.params.values())

    def named_parameters(self):
        return self.params.items()


class FakeProcessedLayoutLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.aclnn_input_scale_reciprocal = torch.ones(2)


class FakeProcessedLayoutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = FakeProcessedLayoutLayer()


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
        "int8_cache_name": None,
    }


# Test server initialization
def test_server_initialization(server_config, mock_model):
    server_config["model"] = mock_model
    with patch("socket.socket") as mock_socket, capture_elastic_logs() as log_capture_string:
        server = ElasticServer(**server_config)

        # Check the socket configuration
        mock_socket.assert_called_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket.return_value.setsockopt.assert_called_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mock_socket.return_value.bind.assert_called_with(("127.0.0.1", 8080))
        mock_socket.return_value.listen.assert_called_with(256)

        # Check int8 cache
        assert "param2" in server.original_int8
        assert server.original_int8["param2"].device.type == "cpu"  # Verifying DRAM Cache

        assert server.addr == server_config["addr"]
        assert server.port == server_config["port"]
        assert server.device_id == server_config["device_id"]
        assert server.model_path == server_config["model_path"]
        assert server.tp == server_config["tp"]
        assert server.pp == server_config["pp"]

        # Get captured logs
        log_output = log_capture_string.getvalue()

        # Check output
        assert "Server 127.0.0.1:8080 starts" in log_output


# Test the int8 cache option
@pytest.mark.parametrize("cache_option,expected_device", [("dram", "cpu"), ("no", None), ("invalid", None)])
def test_int8_cache_handling(server_config, mock_model, cache_option, expected_device):
    server_config["int8_cache"] = cache_option
    server_config["model"] = mock_model

    with patch("socket.socket"), capture_elastic_logs() as log_capture_string:
        server = ElasticServer(**server_config)

        log_output = log_capture_string.getvalue()

        if cache_option == "invalid":
            assert "int8_cache should be selected in [HBM, DRAM]" in log_output

        if expected_device is None:
            assert len(server.original_int8) == 0
        else:
            assert server.original_int8["param2"].device.type == expected_device


def test_p2p_send_skips_aclnn_params_only_for_raw_weights():
    sender_pg = MagicMock()
    sender_pg.send.return_value.wait.return_value = None
    model = FakeP2PModel()

    with (
        patch.object(p2p_elastic_load.torch, "npu", MagicMock()),
        patch.object(p2p_elastic_load.torch_npu.npu, "Stream", return_value=MagicMock()),
        patch.object(p2p_elastic_load.torch_npu.npu, "stream", return_value=nullcontext()),
        patch.object(p2p_elastic_load.torch_npu.npu, "synchronize"),
        patch.object(p2p_elastic_load.torch.distributed, "barrier"),
        patch.object(p2p_elastic_load, "stateless_init_process_group", return_value=sender_pg),
        patch.object(p2p_elastic_load, "destroy_stateless_process_group"),
    ):
        P2PSend(
            "127.0.0.1",
            9090,
            "127.0.0.1:12345",
            send_processed_weights=False,
        ).send(model, {})

    payloads = [call.args[0][0] for call in sender_pg.send.call_args_list]
    assert len(payloads) == 1
    assert payloads[0] is model.params["weight"]


def test_processed_layout_transfer_items_include_module_attribute_tensors(monkeypatch):
    monkeypatch.setattr(
        p2p_elastic_load,
        "_is_transferable_tensor",
        lambda tensor: not tensor.is_meta and tensor.numel() > 0,
    )
    model = FakeProcessedLayoutModel()
    send_names = [name for name, _ in _collect_processed_layout_tensors(model)]
    recv_names = [name for name, _ in _get_recv_transfer_items(model, transfer_processed_layout=True)]

    assert "layer.weight" in send_names
    assert "layer.aclnn_input_scale_reciprocal" in send_names
    assert send_names == recv_names


def test_dram_transfer_items_still_use_named_parameters_only():
    model = FakeP2PModel()
    transfer_names = [name for name, _ in _get_send_transfer_items(model, send_processed_weights=False)]

    assert transfer_names == ["weight"]


def test_send_uses_registered_items_and_skips_rescan(monkeypatch):
    monkeypatch.setattr(
        p2p_elastic_load,
        "_is_transferable_tensor",
        lambda tensor: not tensor.is_meta and tensor.numel() > 0,
    )
    model = FakeProcessedLayoutModel()
    registered_transfer_items = register_processed_layout_transfer_items(model)
    model.layer.kv_cache = torch.ones(3)
    model.layer.runtime_only = torch.ones(4)

    with patch.object(p2p_elastic_load, "_collect_processed_layout_tensors") as mock_collect:
        send_items = _get_send_transfer_items(
            model,
            send_processed_weights=True,
            registered_transfer_items=registered_transfer_items,
        )
        mock_collect.assert_not_called()

    live_items = _collect_processed_layout_tensors(model)

    assert len(send_items) == len(registered_transfer_items)
    assert len(live_items) > len(send_items)
    assert "layer.kv_cache" not in [name for name, _ in send_items]

    sender_pg = MagicMock()
    sender_pg.send.return_value.wait.return_value = None
    with (
        patch.object(p2p_elastic_load.torch, "npu", MagicMock()),
        patch.object(p2p_elastic_load.torch_npu.npu, "Stream", return_value=MagicMock()),
        patch.object(p2p_elastic_load.torch_npu.npu, "stream", return_value=nullcontext()),
        patch.object(p2p_elastic_load.torch_npu.npu, "synchronize"),
        patch.object(p2p_elastic_load.torch.distributed, "barrier"),
        patch.object(p2p_elastic_load, "stateless_init_process_group", return_value=sender_pg),
        patch.object(p2p_elastic_load, "destroy_stateless_process_group"),
        patch.object(p2p_elastic_load, "_collect_processed_layout_tensors") as mock_collect,
    ):
        P2PSend(
            "127.0.0.1",
            9090,
            "127.0.0.1:12345",
            send_processed_weights=True,
        ).send(model, {}, registered_transfer_items=registered_transfer_items)
        mock_collect.assert_not_called()

    sent_tensors = [call.args[0][0] for call in sender_pg.send.call_args_list]
    assert len(sent_tensors) == 2
    assert sent_tensors[0].shape == (2, 2)
    assert sent_tensors[1].shape == (2,)


def test_processed_layout_send_requires_registered_manifest():
    model = FakeProcessedLayoutModel()
    sender_pg = MagicMock()
    sender_pg.send.return_value.wait.return_value = None

    with (
        patch.object(p2p_elastic_load.torch, "npu", MagicMock()),
        patch.object(p2p_elastic_load.torch_npu.npu, "Stream", return_value=MagicMock()),
        patch.object(p2p_elastic_load.torch_npu.npu, "stream", return_value=nullcontext()),
        patch.object(p2p_elastic_load.torch_npu.npu, "synchronize"),
        patch.object(p2p_elastic_load.torch.distributed, "barrier"),
        patch.object(p2p_elastic_load, "stateless_init_process_group", return_value=sender_pg),
        patch.object(p2p_elastic_load, "destroy_stateless_process_group"),
        pytest.raises(RuntimeError, match="registered transfer items"),
    ):
        P2PSend(
            "127.0.0.1",
            9090,
            "127.0.0.1:12345",
            send_processed_weights=True,
        ).send(model, {})


def test_cache_and_register_processed_layout_transfer_manifest():
    model = MagicMock()
    transfer_items = [("layer.weight", torch.empty(2, 3))]

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._collect_processed_layout_tensors",
        return_value=transfer_items,
    ) as mock_collect:
        assert cache_processed_layout_transfer_manifest(model) == 1
        assert get_cached_processed_layout_transfer_items(model) == transfer_items
        assert register_processed_layout_transfer_items(model) == transfer_items
        mock_collect.assert_called_once()


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor", return_value=True)
@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._cast_tensor_to_fractal_nd")
def test_prepare_hccl_recv_buffer_allocates_fresh_nd_for_fractal_nz(mock_cast_nd, _mock_is_nz):
    tensor = torch.empty(2, 3)

    recv_buffer, restore_fractal_nz = _prepare_hccl_recv_buffer(tensor)

    assert restore_fractal_nz is True
    assert recv_buffer is not tensor
    mock_cast_nd.assert_not_called()


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load._cast_tensor_to_fractal_nz")
def test_finalize_hccl_recv_buffer_restores_fractal_nz(mock_cast_nz):
    tensor = torch.empty(2, 3)
    recv_buffer = torch.empty(2, 3)
    nz_tensor = torch.ones(2, 3)
    mock_cast_nz.return_value = nz_tensor

    _finalize_hccl_recv_buffer(tensor, recv_buffer, restore_fractal_nz=True)

    mock_cast_nz.assert_called_once_with(recv_buffer)
    assert torch.equal(tensor, nz_tensor)


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_packed_int32_nz_via_int8_view(mock_format_cast):
    # W4A8: int8 NZ packed as int32 via view; TransData must see int8.
    int8_nz = torch.arange(2 * 4 * 8, dtype=torch.int8).reshape(2, 4, 8)
    packed_int32 = int8_nz.view(torch.int32)
    mock_format_cast.side_effect = lambda t, _fmt: t.clone()

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor",
        return_value=True,
    ):
        result = _cast_tensor_to_fractal_nd(packed_int32)

    assert result.dtype == torch.int32
    assert tuple(result.shape) == (2, 4, 2)
    assert mock_format_cast.call_args.args[0].dtype == torch.int8
    assert mock_format_cast.call_args.args[1] == 2


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_packed_int32_nd_to_nz_via_int8_view(mock_format_cast):
    packed_int32_nd = torch.arange(2 * 4 * 2, dtype=torch.int32).reshape(2, 4, 2)
    mock_format_cast.side_effect = lambda t, _fmt: t.clone()

    with patch(
        "vllm_ascend.model_loader.netloader.executor.elastic_load._is_fractal_nz_tensor",
        return_value=False,
    ):
        result = _cast_tensor_to_fractal_nz(packed_int32_nd)

    cast_input = mock_format_cast.call_args.args[0]
    assert result.dtype == torch.int32
    assert cast_input.dtype == torch.int8
    assert cast_input.data_ptr() != packed_int32_nd.view(torch.int8).data_ptr()
    assert mock_format_cast.call_args.args[1] == 29


@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.torch_npu.npu_format_cast")
def test_cast_packed_int32_via_int8_skips_contiguous_after_nz_cast(mock_format_cast):
    packed_int32_nd = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    nz_int8 = torch.arange(32, dtype=torch.int8).reshape(2, 16)
    mock_format_cast.return_value = nz_int8

    result = _cast_packed_int32_via_int8(packed_int32_nd, 29)

    assert result.dtype == torch.int32
    assert result.data_ptr() == nz_int8.data_ptr()


def test_reshape_transfer_items_to_manifest():
    weight = torch.empty(24)
    scale = torch.empty(8)
    items = [("layer.weight", weight), ("layer.scale", scale)]
    assert reshape_transfer_items_to_manifest(
        items,
        {"layer.weight": (2, 3, 4), "layer.scale": (8,)},
    )
    assert tuple(weight.shape) == (2, 3, 4)
    assert tuple(scale.shape) == (8,)
    assert not reshape_tensor_to_manifest_shape("t", torch.empty(24), (2, 3, 5))


def test_recv_json_message_reads_large_payload():
    payload = {
        "label": "JOIN_ACK",
        "content": {
            "name": "127.0.0.1:1234",
            "transfer_shapes": {"a": [2, 3], "b": [4, 5]},
        },
    }
    encoded = json.dumps(payload).encode("utf-8")

    sender, receiver = socket.socketpair()
    try:
        sender.sendall(encoded)
        sender.close()
        received = _recv_json_message(receiver)
    finally:
        receiver.close()

    assert received == payload


@pytest.mark.parametrize(
    "server_int8_cache,join_content,send_processed_weights",
    [
        (
            "dram",
            {
                "device_id": 0,
                "model_path": "/test/model",
                "tp": 1,
                "pp": 1,
                "port": 9090,
                "int8_cache": "dram",
            },
            False,
        ),
        (
            "dram",
            {
                "device_id": 0,
                "model_path": "/test/model",
                "tp": 1,
                "pp": 1,
                "port": 9090,
            },
            False,
        ),
        (
            "no",
            {
                "device_id": 0,
                "model_path": "/test/model",
                "tp": 1,
                "pp": 1,
                "port": 9090,
                "int8_cache": "no",
            },
            True,
        ),
    ],
)
def test_client_handler_valid_join(server_config, mock_model, server_int8_cache, join_content, send_processed_weights):
    server_config["model"] = mock_model
    server_config["int8_cache"] = server_int8_cache
    with (
        patch("vllm_ascend.model_loader.netloader.interaction.elastic.socket.socket"),
        patch("vllm_ascend.model_loader.netloader.interaction.elastic.P2PSend") as mock_p2p_send,
    ):
        mock_conn = MagicMock()
        mock_addr = ("192.168.1.1", 12345)
        mock_conn.recv.return_value = json.dumps({"label": "JOIN", "content": join_content}).encode("utf-8")

        server = ElasticServer(**server_config)
        registered_items = [("weight", MagicMock())] if send_processed_weights else None
        if registered_items is not None:
            server._registered_transfer_items = registered_items
        server.register_handler(mock_conn, mock_addr)

        expected_ack = {"label": "JOIN_ACK", "content": {"name": "192.168.1.1:12345"}}
        mock_conn.sendall.assert_called_once_with(json.dumps(expected_ack).encode("utf-8"))
        mock_p2p_send.assert_called_once_with(
            "127.0.0.1",
            9090,
            "192.168.1.1:12345",
            "netloader",
            send_processed_weights=send_processed_weights,
        )
        if send_processed_weights:
            mock_p2p_send.return_value.send.assert_called_once_with(
                mock_model,
                {},
                registered_transfer_items=registered_items,
            )
        mock_conn.close.assert_called_once()


def test_client_handler_rejects_int8_cache_mismatch(server_config, mock_model):
    server_config["model"] = mock_model
    with (
        patch("vllm_ascend.model_loader.netloader.interaction.elastic.socket.socket"),
        patch("vllm_ascend.model_loader.netloader.interaction.elastic.P2PSend") as mock_p2p_send,
    ):
        mock_conn = MagicMock()
        mock_addr = ("192.168.1.1", 12345)
        mismatch_data = {
            "label": "JOIN",
            "content": {
                "device_id": 0,
                "model_path": "/test/model",
                "tp": 1,
                "pp": 1,
                "port": 9090,
                "int8_cache": "no",
            },
        }
        mock_conn.recv.return_value = json.dumps(mismatch_data).encode("utf-8")

        ElasticServer(**server_config).register_handler(mock_conn, mock_addr)

        expected_ack = {
            "label": "JOIN_NACK",
            "content": "Received int8_cache no does not consist with this server dram",
        }
        mock_conn.sendall.assert_called_once_with(json.dumps(expected_ack).encode("utf-8"))
        mock_p2p_send.assert_not_called()
        mock_conn.close.assert_called_once()


# Test mismatched JOIN requests
def test_client_handler_mismatch(server_config):
    with patch("vllm_ascend.model_loader.netloader.interaction.elastic.socket.socket"):
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
                "port": 9090,
            },
        }
        mock_conn.recv.return_value = json.dumps(mismatch_data).encode("utf-8")

        server.register_handler(mock_conn, mock_addr)

        assert isinstance(mismatch_data["content"], dict)

        # Verify response
        mismatch_tuple = (
            mismatch_data["content"]["device_id"],
            mismatch_data["content"]["model_path"],
            mismatch_data["content"]["tp"],
            mismatch_data["content"]["pp"],
        )

        server_tuple = (
            server_config["device_id"],
            server_config["model_path"],
            server_config["tp"],
            server_config["pp"],
        )

        expected_ack = {
            "label": "JOIN_NACK",
            "content": (f"Received data {mismatch_tuple} does not consist with this server {server_tuple}"),
        }
        mock_conn.sendall.assert_called_once_with(json.dumps(expected_ack).encode("utf-8"))
        mock_conn.close.assert_called_once()


# Test Invalid Request
@pytest.mark.parametrize(
    "invalid_data,should_send",
    [
        ({"label": "WRONG_LABEL"}, True),  # Incorrect label, can be decoded as JSON, but the content is invalid.
        (
            {"content": {"missing_fields": True}},
            True,
        ),  # Missing field, can be decoded as JSON, but the content is invalid.
        ("plain text", False),  # Non-JSON data, json.loads failed
        (b"invalid_bytes", False),  # Invalid byte, decode or json.loads failed
    ],
)
def test_client_handler_invalid_requests(server_config, invalid_data, should_send):
    with (
        patch("vllm_ascend.model_loader.netloader.interaction.elastic.socket.socket"),
        capture_elastic_logs() as log_capture_string,
    ):
        server = ElasticServer(**server_config)
        mock_conn = MagicMock()
        mock_addr = ("192.168.1.1", 12345)

        if isinstance(invalid_data, (str, bytes)):
            mock_conn.recv.return_value = invalid_data if isinstance(invalid_data, bytes) else invalid_data.encode()
        else:
            mock_conn.recv.return_value = json.dumps(invalid_data).encode("utf-8")

        server.register_handler(mock_conn, mock_addr)

        if should_send:
            expected_ack = {
                "label": "JOIN_NACK",
                "content": f"Received data does not contain required fields: {invalid_data}",
            }
            mock_conn.sendall.assert_called_once_with(json.dumps(expected_ack).encode("utf-8"))
        else:
            mock_conn.sendall.assert_not_called()

        log_output = log_capture_string.getvalue()

        # Any warning in the log is acceptable
        assert "Failed to load" in log_output or "does not contain" in log_output
        mock_conn.close.assert_called_once()


# Test the thread startup.
def test_server_start(server_config):
    with patch("socket.socket"), patch("threading.Thread") as mock_thread:
        handler_thread_instance = mock_thread.return_value

        server = ElasticServer(**server_config)
        server.start()

        # Assert that the correct target parameter was passed when instantiating the Thread instance.
        mock_thread.assert_called_once()
        args, kwargs = mock_thread.call_args
        assert kwargs["target"] == server.elastic_client_handler

        # Verify the daemon attribute is set to True (the attribute value will be recorded after MagicMock assignment).
        assert handler_thread_instance.daemon is True

        # Check if the start() method is called.
        handler_thread_instance.start.assert_called_once()


if __name__ == "__main__":
    pytest.main()
