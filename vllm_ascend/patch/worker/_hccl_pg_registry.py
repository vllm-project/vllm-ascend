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

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from threading import Lock
from typing import cast

logger = logging.getLogger(__name__)


_AUDITED_PG_OPTION_FIELDS = ("hccl_config",)
# These fields are populated by torch_npu/new_group at runtime and are either
# already represented elsewhere in the reuse key or intentionally excluded.
_REDUNDANT_PG_OPTION_FIELDS = (
    "global_ranks_in_group",
    "group_id",
    "group_name",
)
_KNOWN_PG_OPTION_DEFAULTS = {
    "backend": "hccl",
    "global_ranks_in_group": (),
    "group_id": "",
    "group_name": "",
    "hccl_config": {},
    "is_high_priority_stream": False,
    "op_timeout": timedelta(seconds=10),
}
_OPTION_DEFAULT_NON_AUDITED = (None, False, 0, 0.0)

_NON_GROUP_MEMBER = object()
_NON_GROUP_MEMBER_SET = False


@dataclass(frozen=True)
class HcclPgKey:
    backend: str
    ranks: tuple[int, ...]
    options_key: tuple[tuple[str, object], ...]
    reuse_domain: str


@dataclass
class RegistryEntry:
    handle: object
    refcount: int


def make_hccl_pg_key(
    ranks: list[int] | tuple[int, ...],
    backend: str,
    pg_options: object,
    reuse_domain: str,
) -> HcclPgKey | None:
    """
    Return a hashable key that identifies a shared HCCL process group.

    Unknown non-default pg option fields cause fail-closed behavior (returns None),
    which disables process-group reuse for this configuration.
    """
    if backend != "hccl":
        return None

    normalized_options = _normalize_hccl_pg_options(pg_options)
    if normalized_options is None:
        return None
    if not _global_ranks_match_requested_ranks(ranks, pg_options):
        return None

    return HcclPgKey(
        backend=backend,
        ranks=tuple(ranks),
        options_key=normalized_options,
        reuse_domain=reuse_domain,
    )


class HcclPgRegistry:
    """
    HCCL process-group reuse registry.

    Cross-key process-group creation is intentionally not a full concurrent factory:
    callers still need to serialize creation by design, and this helper keeps lock
    scope to registry lookup/refcount mutation only.
    """

    def __init__(self):
        self._entries: dict[HcclPgKey, RegistryEntry] = {}
        self._registry_lock = Lock()

    def acquire(
        self,
        *,
        ranks,
        backend,
        pg_options,
        reuse_domain,
        create_fn,
    ) -> object:
        key = make_hccl_pg_key(ranks, backend, pg_options, reuse_domain)
        if key is None:
            return create_fn()

        with self._registry_lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.refcount += 1
                return entry.handle

        handle = create_fn()

        with self._registry_lock:
            existing = self._entries.get(key)
            if existing is None:
                self._entries[key] = RegistryEntry(handle=handle, refcount=1)
                return handle
            existing.refcount += 1
            if not _is_non_group_member(handle):
                _destroy_process_group(handle)
            return existing.handle

    def release(self, key: HcclPgKey) -> object | None:
        with self._registry_lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.refcount > 1:
                entry.refcount -= 1
                return None
            del self._entries[key]

        if _is_non_group_member(entry.handle):
            return None

        _destroy_process_group(entry.handle)
        return entry.handle

    def clear(self):
        with self._registry_lock:
            self._entries.clear()
            # Full reinitialization path already destroys process groups; clear
            # only removes stale registry metadata.


def _normalize_hccl_pg_options(
    pg_options: object,
) -> tuple[tuple[str, object], ...] | None:
    if pg_options is None:
        return ()
    options_dict = dict(pg_options) if isinstance(pg_options, Mapping) else None
    if _has_unknown_non_default_fields(pg_options):
        return None

    normalized_items: list[tuple[str, object]] = []
    for field_name in _AUDITED_PG_OPTION_FIELDS:
        default_value = _KNOWN_PG_OPTION_DEFAULTS[field_name]
        if options_dict is not None:
            actual_value = options_dict.get(field_name, default_value)
        else:
            actual_value = getattr(pg_options, field_name, default_value)
        if _is_default_option_value(field_name, actual_value):
            continue
        normalized_items.append((field_name, _freeze_for_key(actual_value)))
    return tuple(sorted(normalized_items))


def _has_unknown_non_default_fields(pg_options: object) -> bool:
    options_dict = None
    if isinstance(pg_options, Mapping):
        options_dict = dict(pg_options)
    else:
        options_dict = vars(pg_options) if hasattr(pg_options, "__dict__") else None

    if options_dict is not None:
        field_names: list[str] = list(options_dict.keys())
    else:
        field_names = [name for name in dir(pg_options) if not name.startswith("_")]

    for name in field_names:
        if name in _AUDITED_PG_OPTION_FIELDS:
            continue
        if name in _REDUNDANT_PG_OPTION_FIELDS:
            continue
        try:
            if options_dict is not None:
                value = options_dict[name]
            else:
                value = getattr(pg_options, name)
        except Exception:
            continue
        if callable(value):
            continue
        if _is_default_option_value(name, value):
            continue
        logger.warning(
            "Disabling HCCL process-group reuse because pg_options has non-default field '%s'",
            name,
        )
        return True
    return False


def _global_ranks_match_requested_ranks(
    ranks: list[int] | tuple[int, ...],
    pg_options: object,
) -> bool:
    if isinstance(pg_options, Mapping):
        value = pg_options.get("global_ranks_in_group", ())
    else:
        value = getattr(pg_options, "global_ranks_in_group", ())
    if value is None:
        return True

    value_tuple = tuple(value)
    if not value_tuple:
        return True
    ranks_tuple = tuple(ranks)
    if value_tuple == ranks_tuple:
        return True

    logger.warning(
        "Disabling HCCL process-group reuse because pg_options.global_ranks_in_group=%s "
        "does not match requested ranks=%s",
        value_tuple,
        ranks_tuple,
    )
    return False


def _freeze_for_key(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_for_key(val)) for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_key(item) for item in value)
    if isinstance(value, set):
        return tuple(_freeze_for_key(item) for item in sorted(value, key=lambda item: str(item)))
    return value


def _is_default_option_value(name: str, value: object) -> bool:
    if name in _KNOWN_PG_OPTION_DEFAULTS:
        default_value = _KNOWN_PG_OPTION_DEFAULTS[name]
        if name in ("global_ranks_in_group",):
            default_ranks = cast(tuple[object, ...], default_value)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                return tuple(value) == default_ranks
            return value == default_value
        if name == "hccl_config":
            return value in (None, {}, default_value)
        return value == default_value
    if name in ("_rank", "_backend"):
        return True
    return value in _OPTION_DEFAULT_NON_AUDITED


def _is_non_group_member(handle: object) -> bool:
    global _NON_GROUP_MEMBER
    global _NON_GROUP_MEMBER_SET
    if not _NON_GROUP_MEMBER_SET:
        _NON_GROUP_MEMBER = _load_non_group_member_sentinel()
        _NON_GROUP_MEMBER_SET = True
    return handle is _NON_GROUP_MEMBER


def _load_non_group_member_sentinel() -> object:
    try:
        from torch.distributed.distributed_c10d import GroupMember

        return GroupMember.NON_GROUP_MEMBER
    except Exception:
        return object()


def _destroy_process_group(handle: object):
    from torch.distributed import destroy_process_group

    destroy_process_group(handle)
