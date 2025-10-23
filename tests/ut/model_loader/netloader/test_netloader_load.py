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

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.model_loader.netloader.load import elastic_load


@pytest.fixture
def mock_sources():
    return [
        {
            "device_id": 0,
            "sources": ["a", "b"]
        },
        {
            "device_id": 1,
            "sources": ["c"]
        },
    ]


@patch("vllm_ascend.model_loader.netloader.interaction.elastic.ElasticClient")
@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.P2PLoad")
def test_sources_this_device_empty(mock_p2p, mock_client):
    sources = [{"device_id": 1, "sources": ["c"]}]
    result = elastic_load("model", 0, "model_path", sources, 1, 1)
    assert result is None
    mock_client.assert_not_called()
    mock_p2p.assert_not_called()


@patch("vllm_ascend.model_loader.netloader.interaction.elastic.ElasticClient")
@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.P2PLoad")
def test_client_s_none(mock_p2p, mock_client, mock_sources):
    # Simulate ElasticClient.s as None
    mock_instance = MagicMock()
    mock_instance.s = None
    mock_client.return_value = mock_instance
    result = elastic_load("model", 0, "model_path", mock_sources, 1, 1)
    assert result is None


@patch("vllm_ascend.model_loader.netloader.interaction.elastic.ElasticClient")
@patch("vllm_ascend.model_loader.netloader.executor.elastic_load.P2PLoad")
def test_client_ack_none(mock_p2p, mock_client, mock_sources):
    # Simulate ElasticClient.ack as None
    mock_instance = MagicMock()
    mock_instance.s = True
    mock_instance.ack = None
    mock_client.return_value = mock_instance
    result = elastic_load("model", 0, "model_path", mock_sources, 1, 1)
    assert result is None


@patch("vllm_ascend.model_loader.netloader.load.P2PLoad")
@patch("vllm_ascend.model_loader.netloader.load.logger")
def test_model_load_fail(mock_logger, mock_p2p):
    mock_client = MagicMock()
    mock_client.s = True
    mock_client.ack = ["foo", "bar"]
    mock_client.server_addr = "addr"

    with patch("vllm_ascend.model_loader.netloader.load.ElasticClient",
               return_value=mock_client):
        # P2PLoad.load returns None
        mock_p2p_instance = MagicMock()
        mock_p2p_instance.load.return_value = None
        mock_p2p.return_value = mock_p2p_instance

        sources = [{"device_id": 0, "sources": ["whatever"]}]
        result = elastic_load("model", 0, "model_path", sources, 1, 1)
        assert result is None
        mock_logger.error.assert_called_once()


@patch("vllm_ascend.model_loader.netloader.load.P2PLoad")
@patch("vllm_ascend.model_loader.netloader.load.logger")
def test_model_load_success(mock_logger, mock_p2p):
    mock_client = MagicMock()
    mock_client.s = True
    mock_client.ack = ["foo", "bar"]
    mock_client.server_addr = "addr"

    with patch("vllm_ascend.model_loader.netloader.load.ElasticClient",
               return_value=mock_client):
        expected_model = object()
        mock_p2p_instance = MagicMock()
        mock_p2p_instance.load.return_value = expected_model
        mock_p2p.return_value = mock_p2p_instance

        sources = [{"device_id": 0, "sources": ["whatever"]}]
        result = elastic_load("model", 0, "model_path", sources, 1, 1)
        assert result is expected_model
        mock_logger.info.assert_called_once()


if __name__ == "__main__":
    pytest.main()
