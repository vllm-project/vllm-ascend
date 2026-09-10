from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    transfer as transfer_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
    _plan_registration_ranges,
    _plan_source,
    _plan_transfer_waves,
)

_MIB = 1024 * 1024


def _make_source(
    *,
    storage_start: int,
    source_start: int,
    source_nbytes: int,
) -> MagicMock:
    storage = MagicMock()
    storage.data_ptr.return_value = storage_start

    tensor = MagicMock()
    tensor.untyped_storage.return_value = storage
    tensor.data_ptr.return_value = source_start
    tensor.nbytes = source_nbytes
    return tensor


def test_initialize_engine_binds_npu_and_uses_ascend_protocol():
    transfer = AscendMooncakeTransfer("producer-host", 3)
    engine = MagicMock()
    calls = []

    def initialize(*args):
        calls.append(("initialize", args))
        return 0

    engine.initialize.side_effect = initialize

    with patch.object(
        transfer_module.torch.npu,
        "set_device",
        side_effect=lambda device: calls.append(("set_device", device)),
    ) as set_device:
        result = transfer._initialize_engine(engine)

    assert result == 0

    set_device.assert_called_once_with(3)
    engine.initialize.assert_called_once_with(
        "producer-host",
        "P2PHANDSHAKE",
        "ascend",
        "",
    )
    assert calls == [
        ("set_device", 3),
        (
            "initialize",
            ("producer-host", "P2PHANDSHAKE", "ascend", ""),
        ),
    ]


def test_plan_source_whole_direct():
    tensor = _make_source(
        storage_start=2 * _MIB,
        source_start=5 * _MIB,
        source_nbytes=2 * _MIB,
    )

    plan = _plan_source(tensor)

    assert plan.owner is tensor
    assert plan.prefix_nbytes == 0
    assert plan.direct_address == 5 * _MIB
    assert plan.direct_nbytes == 2 * _MIB
    assert plan.registration_address == 4 * _MIB
    assert plan.registration_nbytes == 3 * _MIB


def test_plan_source_prefix_and_suffix():
    tensor = _make_source(
        storage_start=5 * _MIB,
        source_start=11 * _MIB // 2,
        source_nbytes=5 * _MIB // 2,
    )

    plan = _plan_source(tensor)

    assert plan.owner is tensor
    assert plan.prefix_nbytes == _MIB // 2
    assert plan.direct_address == 6 * _MIB
    assert plan.direct_nbytes == 2 * _MIB
    assert plan.registration_address == 6 * _MIB
    assert plan.registration_nbytes == 2 * _MIB


def test_plan_source_all_bounce():
    tensor = _make_source(
        storage_start=5 * _MIB,
        source_start=11 * _MIB // 2,
        source_nbytes=_MIB // 4,
    )

    plan = _plan_source(tensor)

    assert plan.owner is tensor
    assert plan.prefix_nbytes == _MIB // 4
    assert plan.direct_address is None
    assert plan.direct_nbytes == 0
    assert plan.registration_address is None
    assert plan.registration_nbytes == 0


def test_plan_registration_ranges_merges_chained_ranges_in_same_storage():
    source_a = _make_source(
        storage_start=2 * _MIB,
        source_start=5 * _MIB,
        source_nbytes=_MIB,
    )
    source_b = _make_source(
        storage_start=2 * _MIB,
        source_start=6 * _MIB,
        source_nbytes=5 * _MIB // 2,
    )
    source_c = _make_source(
        storage_start=2 * _MIB,
        source_start=17 * _MIB // 2,
        source_nbytes=_MIB // 2,
    )
    storage = source_a.untyped_storage()
    source_b.untyped_storage.return_value = storage
    source_c.untyped_storage.return_value = storage

    registrations = _plan_registration_ranges([_plan_source(source) for source in (source_a, source_b, source_c)])

    assert len(registrations) == 1
    registration = registrations[0]
    assert registration.address == 4 * _MIB
    assert registration.nbytes == 5 * _MIB
    assert registration.owners == (source_a, source_b, source_c)


def test_plan_registration_ranges_keeps_gap_in_same_storage():
    source_a = _make_source(
        storage_start=2 * _MIB,
        source_start=5 * _MIB,
        source_nbytes=_MIB,
    )
    source_b = _make_source(
        storage_start=2 * _MIB,
        source_start=10 * _MIB,
        source_nbytes=_MIB,
    )
    source_b.untyped_storage.return_value = source_a.untyped_storage()

    registrations = _plan_registration_ranges([_plan_source(source_a), _plan_source(source_b)])

    assert [(item.address, item.nbytes) for item in registrations] == [
        (4 * _MIB, 2 * _MIB),
        (10 * _MIB, _MIB),
    ]
    assert registrations[0].owners == (source_a,)
    assert registrations[1].owners == (source_b,)


def test_plan_registration_ranges_keeps_adjacent_different_storages_separate():
    source_a = _make_source(
        storage_start=2 * _MIB,
        source_start=5 * _MIB,
        source_nbytes=_MIB,
    )
    source_b = _make_source(
        storage_start=6 * _MIB,
        source_start=6 * _MIB,
        source_nbytes=2 * _MIB,
    )

    registrations = _plan_registration_ranges([_plan_source(source_a), _plan_source(source_b)])

    assert [(item.address, item.nbytes) for item in registrations] == [
        (4 * _MIB, 2 * _MIB),
        (6 * _MIB, 2 * _MIB),
    ]
    assert registrations[0].owners == (source_a,)
    assert registrations[1].owners == (source_b,)


def test_plan_registration_ranges_skips_all_bounce_source():
    source = _make_source(
        storage_start=5 * _MIB,
        source_start=11 * _MIB // 2,
        source_nbytes=_MIB // 4,
    )

    registrations = _plan_registration_ranges([_plan_source(source)])

    assert registrations == []


def test_plan_transfer_waves_returns_no_waves_for_empty_batch():
    assert _plan_transfer_waves([], _MIB) == []


def test_plan_transfer_waves_unifies_sources_in_input_order():
    direct = _make_source(
        storage_start=2 * _MIB,
        source_start=5 * _MIB,
        source_nbytes=_MIB,
    )
    mixed = _make_source(
        storage_start=7 * _MIB,
        source_start=15 * _MIB // 2,
        source_nbytes=_MIB,
    )
    all_bounce = _make_source(
        storage_start=9 * _MIB,
        source_start=19 * _MIB // 2,
        source_nbytes=_MIB // 4,
    )

    waves = _plan_transfer_waves([direct, mixed, all_bounce], _MIB)

    assert len(waves) == 1
    wave = waves[0]
    assert [item.source.owner for item in wave.sources] == [
        direct,
        mixed,
        all_bounce,
    ]
    assert [item.bounce_offset for item in wave.sources] == [
        None,
        0,
        _MIB // 2,
    ]
    assert wave.bounce_nbytes == 3 * _MIB // 4

    mixed_plan = wave.sources[1].source
    assert mixed_plan.prefix_nbytes == _MIB // 2
    assert mixed_plan.direct_address == 8 * _MIB
    assert mixed_plan.direct_nbytes == _MIB // 2

    assert [(item.address, item.nbytes) for item in wave.registration_ranges] == [
        (4 * _MIB, 2 * _MIB),
        (8 * _MIB, _MIB // 2),
    ]
    assert wave.registration_ranges[0].owners == (direct,)
    assert wave.registration_ranges[1].owners == (mixed,)


def test_plan_transfer_waves_moves_complete_source_across_capacity_boundary():
    direct = _make_source(
        storage_start=2 * _MIB,
        source_start=5 * _MIB,
        source_nbytes=_MIB,
    )
    bounced = [
        _make_source(
            storage_start=storage_start,
            source_start=storage_start + _MIB // 2,
            source_nbytes=_MIB,
        )
        for storage_start in (7 * _MIB, 9 * _MIB, 11 * _MIB)
    ]

    waves = _plan_transfer_waves(
        [bounced[0], direct, bounced[1], bounced[2]],
        _MIB,
    )

    assert len(waves) == 2
    assert [item.source.owner for item in waves[0].sources] == [
        bounced[0],
        direct,
        bounced[1],
    ]
    assert [item.bounce_offset for item in waves[0].sources] == [
        0,
        None,
        _MIB // 2,
    ]
    assert waves[0].bounce_nbytes == _MIB
    assert [(item.address, item.nbytes) for item in waves[0].registration_ranges] == [
        (8 * _MIB, _MIB // 2),
        (4 * _MIB, 2 * _MIB),
        (10 * _MIB, _MIB // 2),
    ]

    assert [item.source.owner for item in waves[1].sources] == [bounced[2]]
    assert [item.bounce_offset for item in waves[1].sources] == [0]
    assert waves[1].bounce_nbytes == _MIB // 2
    assert [(item.address, item.nbytes) for item in waves[1].registration_ranges] == [
        (12 * _MIB, _MIB // 2),
    ]


def test_plan_transfer_waves_packs_prefixes_without_internal_padding():
    source_a = _make_source(
        storage_start=5 * _MIB,
        source_start=6 * _MIB - 300,
        source_nbytes=600,
    )
    source_b = _make_source(
        storage_start=7 * _MIB,
        source_start=8 * _MIB - 500,
        source_nbytes=1000,
    )

    waves = _plan_transfer_waves([source_a, source_b], 800)

    assert len(waves) == 1
    assert [item.bounce_offset for item in waves[0].sources] == [0, 300]
    assert waves[0].bounce_nbytes == 800


def test_plan_transfer_waves_all_bounce_has_no_registration_ranges():
    source = _make_source(
        storage_start=5 * _MIB,
        source_start=11 * _MIB // 2,
        source_nbytes=_MIB // 4,
    )

    waves = _plan_transfer_waves([source], _MIB)

    assert len(waves) == 1
    assert waves[0].bounce_nbytes == _MIB // 4
    assert waves[0].registration_ranges == ()
    assert waves[0].sources[0].bounce_offset == 0


def test_acquire_sources_merges_views_into_aligned_storage_region():
    alignment = 2 * 1024 * 1024
    raw = torch.empty(alignment + 1024, dtype=torch.uint8)
    offset = (-raw.data_ptr()) % alignment
    aligned = raw.narrow(0, offset, 512)
    source_a = aligned.narrow(0, 128, 64)
    source_b = aligned.narrow(0, 256, 64)

    transfer = AscendMooncakeTransfer("producer-host", 0)
    engine = MagicMock()
    engine.batch_register_memory.return_value = 0

    with patch.object(transfer, "_ensure_engine", return_value=engine):
        addresses = transfer.acquire_sources([source_a, source_b])

    assert addresses == [aligned.data_ptr()]
    engine.batch_register_memory.assert_called_once_with(
        [aligned.data_ptr()],
        [raw.untyped_storage().nbytes() - offset],
    )
