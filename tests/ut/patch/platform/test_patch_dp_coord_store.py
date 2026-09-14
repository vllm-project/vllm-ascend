from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.patch.platform import patch_dp_coord_store


def _make_vllm_config(*, dp_size=8, coord_store_port=0):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_master_ip="192.0.2.10",
            _coord_store_port=coord_store_port,
        )
    )


def test_create_coord_store_for_local_dp_engines():
    vllm_config = _make_vllm_config()
    store = SimpleNamespace(port=43210)

    with patch.object(patch_dp_coord_store, "create_tcp_store", return_value=store) as create_store:
        result = patch_dp_coord_store._create_local_dp_coord_store(8, vllm_config)

    assert result is store
    assert vllm_config.parallel_config._coord_store_port == 43210
    create_store.assert_called_once_with(
        "192.0.2.10",
        0,
        is_master=True,
        world_size=-1,
        wait_for_workers=False,
    )


def test_coord_store_is_not_created_when_not_all_dp_engines_are_local():
    vllm_config = _make_vllm_config()

    with patch.object(patch_dp_coord_store, "create_tcp_store") as create_store:
        result = patch_dp_coord_store._create_local_dp_coord_store(4, vllm_config)

    assert result is None
    assert vllm_config.parallel_config._coord_store_port == 0
    create_store.assert_not_called()


def test_coord_store_is_not_replaced_when_already_configured():
    vllm_config = _make_vllm_config(coord_store_port=12345)

    with patch.object(patch_dp_coord_store, "create_tcp_store") as create_store:
        result = patch_dp_coord_store._create_local_dp_coord_store(8, vllm_config)

    assert result is None
    assert vllm_config.parallel_config._coord_store_port == 12345
    create_store.assert_not_called()


def test_manager_keeps_coord_store_alive_before_starting_engines():
    vllm_config = _make_vllm_config()
    manager = SimpleNamespace()
    store = SimpleNamespace(port=43210)
    original_init = MagicMock()

    with (
        patch.object(patch_dp_coord_store, "_create_local_dp_coord_store", return_value=store),
        patch.object(patch_dp_coord_store, "_ORIGINAL_CORE_ENGINE_PROC_MANAGER_INIT", original_init),
    ):
        patch_dp_coord_store._patched_core_engine_proc_manager_init(
            manager,
            8,
            0,
            0,
            vllm_config,
            True,
            "ipc:///tmp/handshake",
            object,
            False,
        )

    assert manager._ascend_dp_coord_store is store
    original_init.assert_called_once_with(
        manager,
        8,
        0,
        0,
        vllm_config,
        True,
        "ipc:///tmp/handshake",
        object,
        False,
        None,
        None,
    )
