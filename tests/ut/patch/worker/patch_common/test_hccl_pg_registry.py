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
# This file is a part of the vllm-ascend project.
#
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_MODULE_PATH = Path(__file__).resolve().parents[5] / "vllm_ascend/patch/worker/_hccl_pg_registry.py"
_MODULE_NAME = "vllm_ascend.patch.worker._hccl_pg_registry"
_SPEC = spec_from_file_location(_MODULE_NAME, str(_MODULE_PATH))
if _SPEC is None:
    raise RuntimeError("Failed to load _hccl_pg_registry module spec")

_MODULE: Any = module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)  # type: ignore[union-attr]

RegistryEntry = _MODULE.RegistryEntry
HcclPgRegistry = _MODULE.HcclPgRegistry
make_hccl_pg_key = _MODULE.make_hccl_pg_key


@contextmanager
def _patch_destroy_process_group(destroy_fn):
    previous = _MODULE._destroy_process_group
    _MODULE._destroy_process_group = destroy_fn
    try:
        yield
    finally:
        _MODULE._destroy_process_group = previous


@contextmanager
def _set_non_group_member_sentinel(sentinel: object):
    previous = (_MODULE._NON_GROUP_MEMBER, _MODULE._NON_GROUP_MEMBER_SET)
    _MODULE._NON_GROUP_MEMBER = sentinel
    _MODULE._NON_GROUP_MEMBER_SET = True
    try:
        yield
    finally:
        _MODULE._NON_GROUP_MEMBER, _MODULE._NON_GROUP_MEMBER_SET = previous


@dataclass
class FakeOptions:
    hccl_config: dict[str, int] | None = None


@dataclass
class FakeOptionsWithUnknown:
    hccl_config: dict[str, int] | None = None
    non_default_option: int = 7


@dataclass
class RealisticFakeHcclOptions:
    backend: str = "hccl"
    global_ranks_in_group: list[int] | tuple[int, ...] = ()
    group_id: str = ""
    group_name: str = ""
    hccl_config: dict[str, int] | None = None
    is_high_priority_stream: bool = False
    op_timeout: object = timedelta(seconds=10)


def test_make_hccl_pg_key_respects_rank_order_and_reuse_domain():
    opts = FakeOptions(hccl_config={"hccl_buffer_size": 200})
    key_a = make_hccl_pg_key([0, 1], "hccl", opts, reuse_domain="shared")
    key_b = make_hccl_pg_key([1, 0], "hccl", opts, reuse_domain="shared")
    key_c = make_hccl_pg_key([0, 1], "hccl", opts, reuse_domain="eplb")

    assert key_a != key_b
    assert key_a != key_c


def test_make_hccl_pg_key_mapping_hccl_config_affects_distinct_keys():
    key_a = make_hccl_pg_key(
        [0, 1],
        "hccl",
        {"hccl_config": {"hccl_buffer_size": 200}},
        reuse_domain="shared",
    )
    key_b = make_hccl_pg_key(
        [0, 1],
        "hccl",
        {"hccl_config": {"hccl_buffer_size": 400}},
        reuse_domain="shared",
    )

    assert key_a != key_b


def test_make_hccl_pg_key_accepts_realistic_options_object_defaults():
    key_a = make_hccl_pg_key(
        [0, 1],
        "hccl",
        RealisticFakeHcclOptions(hccl_config={"hccl_buffer_size": 200}),
        reuse_domain="shared",
    )
    key_b = make_hccl_pg_key(
        [0, 1],
        "hccl",
        RealisticFakeHcclOptions(hccl_config={"hccl_buffer_size": 400}),
        reuse_domain="shared",
    )

    assert key_a is not None
    assert key_b is not None
    assert key_a != key_b


def test_make_hccl_pg_key_accepts_matching_global_ranks_in_group():
    key = make_hccl_pg_key(
        [0, 1],
        "hccl",
        RealisticFakeHcclOptions(
            global_ranks_in_group=[0, 1],
            hccl_config={"hccl_buffer_size": 200},
        ),
        reuse_domain="shared",
    )

    assert key is not None


def test_make_hccl_pg_key_fails_closed_on_mismatched_global_ranks_in_group():
    key = make_hccl_pg_key(
        [0, 1],
        "hccl",
        RealisticFakeHcclOptions(
            global_ranks_in_group=[1, 2],
            hccl_config={"hccl_buffer_size": 200},
        ),
        reuse_domain="shared",
    )

    assert key is None


def test_make_hccl_pg_key_ignores_runtime_populated_group_identity_fields():
    key_a = make_hccl_pg_key(
        [0, 1],
        "hccl",
        RealisticFakeHcclOptions(
            global_ranks_in_group=[0, 1],
            group_id="hccl_pg_1",
            group_name="tp_auto",
            hccl_config={"hccl_buffer_size": 200},
        ),
        reuse_domain="shared",
    )
    key_b = make_hccl_pg_key(
        [0, 1],
        "hccl",
        RealisticFakeHcclOptions(
            global_ranks_in_group=[0, 1],
            group_id="hccl_pg_2",
            group_name="world_auto",
            hccl_config={"hccl_buffer_size": 200},
        ),
        reuse_domain="shared",
    )

    assert key_a is not None
    assert key_a == key_b


def test_make_hccl_pg_key_fails_closed_for_unknown_mapping_fields():
    assert (
        make_hccl_pg_key(
            [0, 1],
            "hccl",
            {"hccl_config": {"hccl_buffer_size": 200}, "non_default_field": 7},
            reuse_domain="shared",
        )
        is None
    )


def test_registry_release_only_destroys_real_pg_at_zero_refcount():
    destroy_fn = MagicMock()
    with _patch_destroy_process_group(destroy_fn):
        registry = HcclPgRegistry()
        pg = object()
        key = make_hccl_pg_key([0, 1], "hccl", FakeOptions(), reuse_domain="shared")
        registry._entries[key] = RegistryEntry(handle=pg, refcount=1)

        assert registry.release(key) == pg
        assert key not in registry._entries
        destroy_fn.assert_called_once_with(pg)


def test_release_of_non_group_member_only_drops_registry_entry():
    destroy_fn = MagicMock()
    sentinel = _MODULE._load_non_group_member_sentinel()
    with _patch_destroy_process_group(destroy_fn), _set_non_group_member_sentinel(sentinel):
        registry = HcclPgRegistry()
        key = make_hccl_pg_key([0, 1], "hccl", FakeOptions(), reuse_domain="shared")
        registry._entries[key] = RegistryEntry(handle=sentinel, refcount=1)

        assert registry.release(key) is None
        assert key not in registry._entries
        destroy_fn.assert_not_called()


def test_acquire_reuses_cached_handle_and_refcount():
    registry = HcclPgRegistry()
    create_fn = MagicMock(side_effect=[MagicMock(name="first")])
    destroy_fn = MagicMock()
    key = make_hccl_pg_key(
        [0, 1],
        "hccl",
        FakeOptions(hccl_config={"hccl_buffer_size": 200}),
        reuse_domain="shared",
    )

    with _patch_destroy_process_group(destroy_fn):
        first = registry.acquire(
            ranks=[0, 1],
            backend="hccl",
            pg_options=FakeOptions(hccl_config={"hccl_buffer_size": 200}),
            reuse_domain="shared",
            create_fn=create_fn,
        )
        second = registry.acquire(
            ranks=[0, 1],
            backend="hccl",
            pg_options=FakeOptions(hccl_config={"hccl_buffer_size": 200}),
            reuse_domain="shared",
            create_fn=create_fn,
        )

        assert first is second
        assert create_fn.call_count == 1
        assert registry._entries[key].refcount == 2

        assert registry.release(key) is None
        assert registry._entries[key].refcount == 1
        assert registry.release(key) == first
        assert key not in registry._entries
        destroy_fn.assert_called_once_with(first)


def test_acquire_duplicate_non_group_member_handle_is_not_destroyed():
    sentinel = _MODULE._load_non_group_member_sentinel()
    destroy_fn = MagicMock()
    registry = HcclPgRegistry()
    key = make_hccl_pg_key([0, 1], "hccl", FakeOptions(), reuse_domain="shared")

    existing_handle = MagicMock(name="existing_handle")

    def create_fn():
        registry._entries[key] = RegistryEntry(handle=existing_handle, refcount=1)
        return sentinel

    with _patch_destroy_process_group(destroy_fn), _set_non_group_member_sentinel(sentinel):
        merged = registry.acquire(
            ranks=[0, 1],
            backend="hccl",
            pg_options=FakeOptions(),
            reuse_domain="shared",
            create_fn=create_fn,
        )

    assert merged is existing_handle
    assert registry._entries[key].refcount == 2
    destroy_fn.assert_not_called()


def test_clear_removes_entries_without_destroying_handles():
    destroy_fn = MagicMock()
    with _patch_destroy_process_group(destroy_fn):
        registry = HcclPgRegistry()
        key = make_hccl_pg_key(
            [0, 1],
            "hccl",
            FakeOptions(hccl_config={"hccl_buffer_size": 200}),
            reuse_domain="shared",
        )
        registry._entries[key] = RegistryEntry(handle=MagicMock(name="pg"), refcount=1)
        registry.clear()

    assert key not in registry._entries
    destroy_fn.assert_not_called()


def test_release_non_group_member_uses_actual_sentinel():
    destroy_fn = MagicMock()
    sentinel = _MODULE._load_non_group_member_sentinel()
    with _patch_destroy_process_group(destroy_fn), _set_non_group_member_sentinel(sentinel):
        registry = HcclPgRegistry()
        key = make_hccl_pg_key([0, 1], "hccl", FakeOptions(), reuse_domain="shared")
        registry._entries[key] = RegistryEntry(handle=sentinel, refcount=1)

        assert registry.release(key) is None
        assert key not in registry._entries
        destroy_fn.assert_not_called()


def test_acquire_fails_closed_when_unknown_non_default_option_is_present():
    registry = HcclPgRegistry()
    create_fn = MagicMock(side_effect=[MagicMock(name="first"), MagicMock(name="second")])

    first = registry.acquire(
        ranks=[0, 1],
        backend="hccl",
        pg_options=FakeOptionsWithUnknown(non_default_option=7),
        reuse_domain="shared",
        create_fn=create_fn,
    )
    second = registry.acquire(
        ranks=[0, 1],
        backend="hccl",
        pg_options=FakeOptionsWithUnknown(non_default_option=7),
        reuse_domain="shared",
        create_fn=create_fn,
    )

    assert first is not second
    assert create_fn.call_count == 2
    assert not registry._entries


def test_acquire_fails_closed_for_unknown_mapping_fields():
    registry = HcclPgRegistry()
    create_fn = MagicMock(side_effect=[MagicMock(name="first"), MagicMock(name="second")])

    options = {"hccl_config": {"hccl_buffer_size": 200}, "non_default_field": 7}

    first = registry.acquire(
        ranks=[0, 1],
        backend="hccl",
        pg_options=options,
        reuse_domain="shared",
        create_fn=create_fn,
    )
    second = registry.acquire(
        ranks=[0, 1],
        backend="hccl",
        pg_options=options,
        reuse_domain="shared",
        create_fn=create_fn,
    )

    assert first is not second
    assert create_fn.call_count == 2
    assert not registry._entries
