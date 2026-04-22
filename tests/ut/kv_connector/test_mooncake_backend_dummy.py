"""Unit tests for MooncakeBackend dummy client mode."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def dummy_config_file():
    """Create a temporary config file with dummy client enabled."""
    config = {
        "metadata_server": "127.0.0.1:8080",
        "global_segment_size": 1073741824,
        "local_buffer_size": 1073741824,
        "protocol": "ascend",
        "device_name": "",
        "master_server_address": "127.0.0.1:8081",
        "use_dummy_client": True,
        "dummy_server_address": "127.0.0.1:50052",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def embedded_config_file():
    """Create a temporary config file with embedded mode (default)."""
    config = {
        "metadata_server": "127.0.0.1:8080",
        "global_segment_size": 1073741824,
        "local_buffer_size": 1073741824,
        "protocol": "ascend",
        "device_name": "",
        "master_server_address": "127.0.0.1:8081",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestMooncakeStoreConfig:
    """Test config loading for dummy client fields."""

    def test_dummy_client_from_json(self, dummy_config_file):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeStoreConfig,
        )

        config = MooncakeStoreConfig.from_file(dummy_config_file)
        assert config.use_dummy_client is True
        assert config.dummy_server_address == "127.0.0.1:50052"

    def test_dummy_client_defaults_false(self, embedded_config_file):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeStoreConfig,
        )

        config = MooncakeStoreConfig.from_file(embedded_config_file)
        assert config.use_dummy_client is False
        assert config.dummy_server_address == ""

    def test_dummy_client_env_override(self, embedded_config_file):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeStoreConfig,
        )

        with patch.dict(os.environ, {
            "MOONCAKE_USE_DUMMY_CLIENT": "1",
            "MOONCAKE_DUMMY_SERVER_ADDRESS": "10.0.0.1:50052",
        }):
            config = MooncakeStoreConfig.from_file(embedded_config_file)
            assert config.use_dummy_client is True
            assert config.dummy_server_address == "10.0.0.1:50052"


class TestMooncakeBackendDummy:
    """Test MooncakeBackend initialization with dummy client."""

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip")
    def test_setup_dummy_called(self, mock_get_ip, dummy_config_file):
        mock_get_ip.return_value = "127.0.0.1"
        mock_store = MagicMock()
        mock_store.setup_dummy.return_value = 0

        with (
            patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": dummy_config_file}),
            patch("mooncake.store.MooncakeDistributedStore", return_value=mock_store),
            patch("mooncake.store.ReplicateConfig"),
        ):
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
                MooncakeBackend,
            )

            MooncakeBackend(parallel_config=MagicMock())

        mock_store.setup_dummy.assert_called_once_with(
            mem_pool_size=0,
            local_buffer_size=0,
            server_address="127.0.0.1:50052",
        )
        mock_store.setup.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip")
    def test_register_buffer_forwards_to_store_in_dummy_mode(self, mock_get_ip, dummy_config_file):
        mock_get_ip.return_value = "127.0.0.1"
        mock_store = MagicMock()
        mock_store.setup_dummy.return_value = 0

        with (
            patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": dummy_config_file}),
            patch("mooncake.store.MooncakeDistributedStore", return_value=mock_store),
            patch("mooncake.store.ReplicateConfig"),
        ):
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
                MooncakeBackend,
            )

            backend = MooncakeBackend(parallel_config=MagicMock())

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
        ) as mock_te:
            backend.register_buffer([0x1000, 0x2000], [4096, 8192])
            # In dummy mode, register_buffer forwards to store, not global_te
            mock_te.register_buffer.assert_not_called()
            assert mock_store.register_buffer.call_count == 2
            mock_store.register_buffer.assert_any_call(0x1000, 4096)
            mock_store.register_buffer.assert_any_call(0x2000, 8192)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip")
    def test_embedded_mode_calls_setup(self, mock_get_ip, mock_global_te, embedded_config_file):
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = MagicMock()
        mock_engine.get_rpc_port.return_value = 12345
        mock_engine.get_engine.return_value = MagicMock()
        mock_global_te.get_transfer_engine.return_value = mock_engine

        mock_store = MagicMock()
        mock_store.setup.return_value = 0

        with (
            patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": embedded_config_file}),
            patch("mooncake.store.MooncakeDistributedStore", return_value=mock_store),
            patch("mooncake.store.ReplicateConfig"),
        ):
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
                MooncakeBackend,
            )

            MooncakeBackend(parallel_config=MagicMock())

        mock_store.setup.assert_called_once()
        mock_store.setup_dummy.assert_not_called()
