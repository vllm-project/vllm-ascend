"""Unit tests for MooncakeConnectorWorker dummy client mode."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def dummy_config_file():
    """Create a temporary config file with dummy client enabled."""
    config = {
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "ascend",
        "device_name": "",
        "global_segment_size": 1073741824,
        "master_server_address": "127.0.0.1:50088",
        "use_dummy_client": True,
        "dummy_server_address": "127.0.0.1:53000",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def no_dummy_config_file():
    """Create a temporary config file without dummy client."""
    config = {
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "ascend",
        "device_name": "",
        "global_segment_size": 1073741824,
        "master_server_address": "127.0.0.1:50088",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestInitDummyClient:
    """Test _init_dummy_client() method on MooncakeConnectorWorker."""

    def test_returns_none_without_config_path(self):
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
            MooncakeConnectorWorker,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MOONCAKE_CONFIG_PATH", None)
            result = MooncakeConnectorWorker._init_dummy_client(MagicMock())
        assert result is None

    def test_returns_none_when_dummy_disabled(self, no_dummy_config_file):
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
            MooncakeConnectorWorker,
        )

        with patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": no_dummy_config_file}):
            os.environ.pop("MOONCAKE_USE_DUMMY_CLIENT", None)
            result = MooncakeConnectorWorker._init_dummy_client(MagicMock())
        assert result is None

    def test_returns_store_when_enabled(self, dummy_config_file):
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
            MooncakeConnectorWorker,
        )

        mock_store = MagicMock()
        mock_store.setup_dummy.return_value = 0

        with (
            patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": dummy_config_file}),
            patch("mooncake.store.MooncakeDistributedStore", return_value=mock_store),
        ):
            os.environ.pop("MOONCAKE_USE_DUMMY_CLIENT", None)
            result = MooncakeConnectorWorker._init_dummy_client(MagicMock())

        assert result is mock_store
        mock_store.setup_dummy.assert_called_once_with(
            mem_pool_size=0,
            local_buffer_size=0,
            server_address="127.0.0.1:53000",
        )

    def test_env_override(self, no_dummy_config_file):
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
            MooncakeConnectorWorker,
        )

        mock_store = MagicMock()
        mock_store.setup_dummy.return_value = 0

        with (
            patch.dict(
                os.environ,
                {
                    "MOONCAKE_CONFIG_PATH": no_dummy_config_file,
                    "MOONCAKE_USE_DUMMY_CLIENT": "1",
                    "MOONCAKE_DUMMY_SERVER_ADDRESS": "10.0.0.1:53000",
                },
            ),
            patch("mooncake.store.MooncakeDistributedStore", return_value=mock_store),
        ):
            result = MooncakeConnectorWorker._init_dummy_client(MagicMock())

        assert result is mock_store
        mock_store.setup_dummy.assert_called_once_with(
            mem_pool_size=0,
            local_buffer_size=0,
            server_address="10.0.0.1:53000",
        )

    def test_raises_on_setup_failure(self, dummy_config_file):
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
            MooncakeConnectorWorker,
        )

        mock_store = MagicMock()
        mock_store.setup_dummy.return_value = -1

        with (
            patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": dummy_config_file}),
            patch("mooncake.store.MooncakeDistributedStore", return_value=mock_store),
            pytest.raises(RuntimeError, match="setup_dummy.*failed"),
        ):
            MooncakeConnectorWorker._init_dummy_client(MagicMock())
