from unittest.mock import MagicMock, call

import pytest

pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (  # noqa: E402
    GlobalTE,
)

_GLM_LAYER_BYTES = 1_300_234_240
_GLM_FIRST_REGION_LAYER_COUNT = 52
_GLM_SECOND_REGION_LAYER_COUNT = 27
_TE_REGISTRATION_LIMIT_BYTES = 64 * 1024 * 1024 * 1024


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


def test_glm_split_regions_register_and_unregister_in_order():
    engine = MagicMock()
    engine.register_memory.return_value = 0
    engine.unregister_memory.return_value = 0
    manager = _manager_with_engine(engine)
    base_ptr = 0x400000
    first_size = _GLM_FIRST_REGION_LAYER_COUNT * _GLM_LAYER_BYTES
    second_size = _GLM_SECOND_REGION_LAYER_COUNT * _GLM_LAYER_BYTES

    assert first_size < _TE_REGISTRATION_LIMIT_BYTES
    assert first_size + _GLM_LAYER_BYTES >= _TE_REGISTRATION_LIMIT_BYTES
    assert second_size < _TE_REGISTRATION_LIMIT_BYTES

    manager.register_buffer(
        [base_ptr, base_ptr + first_size],
        [first_size, second_size],
        ["npu:0", "npu:0"],
    )
    manager.unregister_buffer()

    assert engine.register_memory.call_args_list == [
        call(base_ptr, first_size, "npu:0"),
        call(base_ptr + first_size, second_size, "npu:0"),
    ]
    assert engine.unregister_memory.call_args_list == [
        call(base_ptr + first_size),
        call(base_ptr),
    ]


def test_split_registration_failure_rolls_back_prior_regions():
    engine = MagicMock()
    engine.register_memory.side_effect = [0, -1]
    engine.unregister_memory.return_value = 0
    manager = _manager_with_engine(engine)
    first_ptr = 0x400000
    first_size = _GLM_FIRST_REGION_LAYER_COUNT * _GLM_LAYER_BYTES
    second_ptr = first_ptr + first_size
    second_size = _GLM_SECOND_REGION_LAYER_COUNT * _GLM_LAYER_BYTES

    with pytest.raises(RuntimeError, match="Mooncake memory registration failed"):
        manager.register_buffer(
            [first_ptr, second_ptr],
            [first_size, second_size],
            ["npu:0", "npu:0"],
        )

    assert engine.register_memory.call_args_list == [
        call(first_ptr, first_size, "npu:0"),
        call(second_ptr, second_size, "npu:0"),
    ]
    engine.unregister_memory.assert_called_once_with(first_ptr)
    assert manager._registered_regions == []
    assert not manager.is_register_buffer


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
