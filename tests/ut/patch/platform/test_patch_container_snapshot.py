# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import regex as re

from vllm_ascend.patch.platform import patch_container_snapshot as snapshot_patch


@pytest.mark.parametrize(
    ("engine_id", "dp_suffix"),
    [
        pytest.param(
            "instance-0123456789abcdef0123456789abcdef",
            "",
            id="standard-format",
        ),
        pytest.param(
            "instance-0123456789abcdef0123456789abcdef_dp2",
            "_dp2",
            id="with-dp-suffix",
        ),
    ],
)
def test_rotate_snapshot_engine_id_replaces_uuid(engine_id, dp_suffix):
    prefix = "instance"
    old_uuid = "0123456789abcdef0123456789abcdef"
    old_id = f"{prefix}-{old_uuid}{dp_suffix}"

    new_id = snapshot_patch._rotate_snapshot_engine_id(old_id)

    assert new_id.startswith(f"{prefix}-")
    assert new_id.endswith(dp_suffix)
    assert new_id != old_id
    assert re.fullmatch(rf"{prefix}-[0-9a-f]{{32}}{re.escape(dp_suffix)}", new_id)


def test_rotate_snapshot_engine_id_fallback_for_unexpected_format():
    mock_uuid = MagicMock()
    mock_uuid.hex = "newuuid"
    with patch.object(snapshot_patch, "uuid4", return_value=mock_uuid):
        new_id = snapshot_patch._rotate_snapshot_engine_id("unexpected-format")

    assert new_id == "unexpected-format-newuuid"


def test_refresh_scheduler_after_resume_updates_host_and_engine_id():
    kv_cfg = SimpleNamespace(
        kv_connector="MooncakeLayerwiseConnector",
        is_kv_producer=True,
        is_kv_consumer=False,
        engine_id="instance-0123456789abcdef0123456789abcdef",
    )
    connector_scheduler = SimpleNamespace(
        side_channel_host="1.1.1.1",
        engine_id=kv_cfg.engine_id,
    )
    connector = SimpleNamespace(
        connector_scheduler=connector_scheduler,
        engine_id=kv_cfg.engine_id,
    )
    engine_core = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=kv_cfg),
        scheduler=SimpleNamespace(connector=connector),
    )

    snapshot_patch._refresh_scheduler_after_resume(engine_core, "10.0.0.8")

    assert connector_scheduler.side_channel_host == "10.0.0.8"
    assert connector.engine_id == connector_scheduler.engine_id
    assert kv_cfg.engine_id == connector_scheduler.engine_id
    assert connector_scheduler.engine_id != "instance-0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize(
    "kv_connector",
    [
        pytest.param("MooncakeHybridConnector", id="hybrid-connector"),
        pytest.param("AscendStoreConnector", id="non-pd-connector"),
    ],
)
def test_refresh_scheduler_after_resume_skips_unsupported_connectors(kv_connector):
    kv_cfg = SimpleNamespace(
        kv_connector=kv_connector,
        is_kv_producer=False,
        is_kv_consumer=False,
        engine_id="instance-0123456789abcdef0123456789abcdef",
    )
    connector_scheduler = SimpleNamespace(
        side_channel_host="1.1.1.1",
        engine_id=kv_cfg.engine_id,
    )
    connector = SimpleNamespace(
        connector_scheduler=connector_scheduler,
        engine_id=kv_cfg.engine_id,
    )
    engine_core = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=kv_cfg),
        scheduler=SimpleNamespace(connector=connector),
    )

    snapshot_patch._refresh_scheduler_after_resume(engine_core, "10.0.0.8")

    assert connector_scheduler.side_channel_host == "1.1.1.1"
    assert connector_scheduler.engine_id == kv_cfg.engine_id


def test_advertise_zmq_endpoint_replaces_host():
    from vllm.v1.engine.coordinator import DPCoordinatorProc

    snapshot_patch._patch_coordinator()
    endpoint = DPCoordinatorProc._advertise_zmq_endpoint("tcp://127.0.0.1:1234", "10.0.0.5")
    assert endpoint == "tcp://10.0.0.5:1234"


def test_advertise_zmq_endpoint_returns_original_when_host_missing():
    from vllm.v1.engine.coordinator import DPCoordinatorProc

    snapshot_patch._patch_coordinator()
    endpoint = "inproc://snapshot-pipe"
    assert DPCoordinatorProc._advertise_zmq_endpoint(endpoint, "10.0.0.5") == endpoint


def test_patch_mp_client_empty_client_addresses_uses_launch_capture():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import vllm.v1.engine.core_client as core_client_mod

    expected_output_address = "tcp://127.0.0.1:40000"

    def fake_original_init(self, asyncio_mode, vllm_config, executor_class, log_stats, client_addresses=None):
        if client_addresses:
            return
        with core_client_mod.launch_core_engines(None, None, None, None):
            pass

    @contextmanager
    def fake_launch(*args, **kwargs):
        yield MagicMock(), MagicMock(), SimpleNamespace(outputs=[expected_output_address]), None

    core_client_mod.MPClient._container_snapshot_output_address_patched = False  # type: ignore[attr-defined]
    core_client_mod.MPClient.__init__ = fake_original_init  # type: ignore[method-assign]

    with patch("vllm.v1.engine.utils.launch_core_engines", fake_launch):
        snapshot_patch._patch_mp_client()

    client = core_client_mod.MPClient.__new__(core_client_mod.MPClient)
    core_client_mod.MPClient.__init__(client, False, MagicMock(), MagicMock(), False, {})

    assert client.output_address == expected_output_address
