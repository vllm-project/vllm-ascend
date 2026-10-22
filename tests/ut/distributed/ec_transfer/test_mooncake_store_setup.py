import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import store_setup as store_setup_module


def _make_store_config():
    return MagicMock(
        metadata_server="P2PHANDSHAKE",
        global_segment_size=1024,
        local_buffer_size=512,
        protocol="ascend",
        device_name="",
        master_server_address="127.0.0.1:50051",
        tenant_id=store_setup_module.DEFAULT_TENANT_ID,
        enable_offload=False,
        mode="embedded",
    )


def _make_mooncake_modules(store):
    mooncake = ModuleType("mooncake")
    mooncake.__path__ = []
    mooncake_store = ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = MagicMock(return_value=store)
    mooncake_store.ObjectDataType = SimpleNamespace(TENSOR="tensor")
    mooncake_store.ReplicateConfig = MagicMock(return_value=MagicMock())
    mooncake.store = mooncake_store
    return mooncake, mooncake_store


def test_factory_rejects_non_ascend_store_protocol():
    bounce_arena = MagicMock()
    registration_transfer = MagicMock()

    with (
        patch.object(
            store_setup_module.MooncakeStoreConfig,
            "load_from_config",
            return_value=MagicMock(protocol="rdma"),
        ),
        pytest.raises(ValueError, match="protocol='ascend'"),
    ):
        store_setup_module.create_ascend_mooncake_embedding_store_client(
            bounce_arena,
            registration_transfer,
        )

    registration_transfer._ensure_engine.assert_not_called()


@pytest.mark.parametrize("value", ["not-json", "[]"])
def test_independent_store_requires_global_config_json_object(value):
    with (
        patch.dict(
            store_setup_module.os.environ,
            {
                "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
                "ASCEND_GLOBAL_RESOURCE_CONFIG": value,
            },
            clear=True,
        ),
        pytest.raises(ValueError, match="must be a JSON object"),
    ):
        store_setup_module.use_store_independent_te()


def test_shared_roce_engine_enables_roce_before_engine_initialization():
    with patch.dict(
        store_setup_module.os.environ,
        {
            "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
            "HCCL_INTRA_PCIE_ENABLE": "1",
            "HCCL_INTRA_ROCE_ENABLE": "0",
        },
        clear=True,
    ):
        store_setup_module._configure_shared_roce()
        assert "ASCEND_ENABLE_USE_FABRIC_MEM" not in store_setup_module.os.environ
        assert store_setup_module.os.environ["HCCL_INTRA_PCIE_ENABLE"] == "0"
        assert store_setup_module.os.environ["HCCL_INTRA_ROCE_ENABLE"] == "1"


@pytest.mark.parametrize(
    ("ec_config", "expected_selectors"),
    [
        ({"protocol_desc": ["roce:device"]}, ("0", "1")),
        ({"protocol_desc": ["hccs:device"]}, (None, None)),
        ({"protocol_desc": ["hccs:device", "roce:device"]}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
    ],
    ids=["explicit-roce", "hccs", "multiple-protocols", "implicit", "store-only"],
)
def test_independent_ec_transport_keeps_only_explicit_roce_selectors(ec_config, expected_selectors):
    resource_config = json.dumps(
        {
            **({"comm_resource_config": ec_config} if ec_config is not None else {}),
            "store": {
                "comm_resource_config": {"protocol_desc": ["roce:device"]},
            },
        }
    )
    with patch.dict(
        store_setup_module.os.environ,
        {
            "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
            "ASCEND_GLOBAL_RESOURCE_CONFIG": resource_config,
            "HCCL_INTRA_PCIE_ENABLE": "1",
            "HCCL_INTRA_ROCE_ENABLE": "0",
        },
        clear=True,
    ):
        store_setup_module.configure_non_fabric_transport()

        assert "ASCEND_ENABLE_USE_FABRIC_MEM" not in store_setup_module.os.environ
        assert (
            tuple(
                store_setup_module.os.environ.get(key) for key in ("HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE")
            )
            == expected_selectors
        )
        assert store_setup_module.os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] == resource_config


def test_shared_roce_engine_rejects_internal_buffer_pool():
    with (
        patch.dict(
            store_setup_module.os.environ,
            {
                "HCCL_INTRA_ROCE_ENABLE": "1",
                "ASCEND_BUFFER_POOL": "4:8",
            },
            clear=True,
        ),
        pytest.raises(ValueError, match="ASCEND_BUFFER_POOL to be unset"),
    ):
        store_setup_module._configure_shared_roce()


def test_factory_adopts_ec_engine_for_roce_store():
    store = MagicMock()
    store.setup.return_value = 0
    registration_transfer = MagicMock()
    transfer_engine = registration_transfer._ensure_engine.return_value
    transfer_engine.get_rpc_port.return_value = 12345
    native_engine = transfer_engine.get_engine.return_value
    bounce_arena = MagicMock()
    config = _make_store_config()
    mooncake, mooncake_store = _make_mooncake_modules(store)

    with (
        patch.dict(
            store_setup_module.os.environ,
            {"ASCEND_ENABLE_USE_FABRIC_MEM": "0"},
        ),
        patch.dict(
            sys.modules,
            {"mooncake": mooncake, "mooncake.store": mooncake_store},
        ),
        patch.object(
            store_setup_module.MooncakeStoreConfig,
            "load_from_config",
            return_value=config,
        ),
        patch.object(store_setup_module, "get_ip", return_value="127.0.0.1"),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.rdma_utils.get_requester_local_hostname",
            return_value="local-host",
        ),
        patch.object(
            store_setup_module,
            "_configure_shared_roce",
            return_value=True,
        ),
    ):
        client = store_setup_module.create_ascend_mooncake_embedding_store_client(
            bounce_arena,
            registration_transfer,
        )

    registration_transfer._ensure_engine.assert_called_once_with()
    store.setup.assert_called_once_with(
        "local-host:12345",
        "P2PHANDSHAKE",
        1024,
        512,
        "ascend",
        "",
        "127.0.0.1:50051",
        engine=native_engine,
    )
    store.register_buffer.assert_not_called()
    assert client._staging_pool is bounce_arena
    assert client._memory.transfer is registration_transfer
    assert client._use_fabric_mem is False


@pytest.mark.parametrize(
    ("environment", "use_fabric_mem", "use_independent_te"),
    [
        (
            {"ASCEND_ENABLE_USE_FABRIC_MEM": "1"},
            True,
            False,
        ),
        (
            {
                "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
                "ASCEND_GLOBAL_RESOURCE_CONFIG": "{}",
            },
            False,
            True,
        ),
    ],
    ids=["fabric", "global-resource-config"],
)
def test_factory_lets_store_create_owned_engine(
    environment,
    use_fabric_mem,
    use_independent_te,
):
    store = MagicMock()
    store.setup.return_value = 0
    registration_transfer = MagicMock()
    config = _make_store_config()
    mooncake, mooncake_store = _make_mooncake_modules(store)

    with (
        patch.dict(
            store_setup_module.os.environ,
            environment,
            clear=True,
        ),
        patch.dict(
            sys.modules,
            {"mooncake": mooncake, "mooncake.store": mooncake_store},
        ),
        patch.object(
            store_setup_module.MooncakeStoreConfig,
            "load_from_config",
            return_value=config,
        ),
        patch.object(store_setup_module, "get_ip", return_value="127.0.0.1"),
    ):
        client = store_setup_module.create_ascend_mooncake_embedding_store_client(
            MagicMock(),
            registration_transfer,
        )

        if use_independent_te:
            assert "ASCEND_ENABLE_USE_FABRIC_MEM" not in store_setup_module.os.environ
            assert store_setup_module.os.environ["HCCL_INTRA_PCIE_ENABLE"] == "0"
            assert store_setup_module.os.environ["HCCL_INTRA_ROCE_ENABLE"] == "1"

    registration_transfer._ensure_engine.assert_not_called()
    store.setup.assert_called_once_with(
        "127.0.0.1",
        "P2PHANDSHAKE",
        1024,
        0,
        "ascend",
        "",
        "127.0.0.1:50051",
    )
    assert client._memory.transfer is registration_transfer
    assert client._use_fabric_mem is use_fabric_mem
    assert client.uses_store_independent_te is use_independent_te


@pytest.mark.parametrize("has_store", [False, True], ids=["consumer", "producer"])
def test_roce_selectors_survive_until_lazy_channel_creation_with_or_without_store(has_store):
    resource_config = json.dumps(
        {
            "comm_resource_config": {"protocol_desc": ["roce:device"]},
            "store": {"comm_resource_config": {"protocol_desc": ["roce:device"]}},
        }
    )
    store = MagicMock()
    store.setup.return_value = 0
    mooncake, mooncake_store = _make_mooncake_modules(store)

    def selectors():
        return tuple(
            store_setup_module.os.environ.get(key) for key in ("HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE")
        )

    with (
        patch.dict(store_setup_module.os.environ, {"ASCEND_GLOBAL_RESOURCE_CONFIG": resource_config}, clear=True),
        patch.dict(sys.modules, {"mooncake": mooncake, "mooncake.store": mooncake_store}),
        patch.object(store_setup_module.MooncakeStoreConfig, "load_from_config", return_value=_make_store_config()),
        patch.object(store_setup_module, "get_ip", return_value="127.0.0.1"),
    ):
        # Both roles execute this before creating their EC TE. Only producers
        # subsequently create a Store. HCCL may read the env much later.
        store_setup_module.configure_non_fabric_transport()
        assert selectors() == ("0", "1")
        if has_store:
            store_setup_module.create_ascend_mooncake_embedding_store_client(MagicMock(), MagicMock())
            store.setup.assert_called_once()
        else:
            store.setup.assert_not_called()
        assert selectors() == ("0", "1")
        assert store_setup_module.os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] == resource_config
