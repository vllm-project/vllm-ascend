from unittest.mock import MagicMock, call

import pytest

pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (  # noqa: E402
    GlobalTE,
)


def _manager_with_engine(engine: object) -> GlobalTE:
    manager = GlobalTE()
    manager.transfer_engine = engine
    return manager


def test_register_buffer_uses_wildcard_location_by_default():
    engine = MagicMock()
    engine.register_memory.return_value = 0
    manager = _manager_with_engine(engine)

    manager.register_buffer([100, 200], [10, 20])

    assert engine.register_memory.call_args_list == [call(100, 10, "*"), call(200, 20, "*")]


def test_register_buffer_uses_explicit_locations():
    engine = MagicMock()
    engine.register_memory.return_value = 0
    manager = _manager_with_engine(engine)
    manager.register_buffer([100, 200], [10, 20], ["*", "npu:0"])

    assert engine.register_memory.call_args_list == [call(100, 10, "*"), call(200, 20, "npu:0")]

def test_register_buffer_only_registers_once():
    engine = MagicMock()
    engine.register_memory.return_value = 0
    manager = _manager_with_engine(engine)

    manager.register_buffer([100], [10])
    manager.register_buffer([200], [20])

    engine.register_memory.assert_called_once_with(100, 10, "*")


def test_register_buffer_raises_when_registration_fails():
    engine = MagicMock()
    engine.register_memory.return_value = 7
    manager = _manager_with_engine(engine)

    with pytest.raises(RuntimeError, match="Mooncake memory registration failed"):
        manager.register_buffer([100], [10])

    assert not manager.is_register_buffer


def test_unregister_buffer_reverses_registration_and_allows_reregistration():
    engine = MagicMock()
    engine.register_memory.return_value = 0
    engine.unregister_memory.return_value = 0
    manager = _manager_with_engine(engine)

    manager.register_buffer([100, 200], [10, 20])
    manager.unregister_buffer()
    manager.unregister_buffer()

    assert engine.unregister_memory.call_args_list == [call(200), call(100)]
    assert manager._registered_regions == []
    assert not manager.is_register_buffer

    manager.register_buffer([300], [30])
    assert engine.register_memory.call_args_list[-1] == call(300, 30, "*")


def test_unregister_buffer_retains_failed_regions_for_retry():
    engine = MagicMock()
    engine.register_memory.return_value = 0
    engine.unregister_memory.side_effect = [9, 0, 0]
    manager = _manager_with_engine(engine)
    manager.register_buffer([100, 200], [10, 20])

    with pytest.raises(RuntimeError, match="ptr=200: ret_value=9"):
        manager.unregister_buffer()

    assert manager._registered_regions == [(200, 20, "*")]
    assert manager.is_register_buffer

    manager.unregister_buffer()
    assert engine.unregister_memory.call_args_list == [call(200), call(100), call(200)]
    assert not manager.is_register_buffer
