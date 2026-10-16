# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Backport the local SWA replay admission protocol to vLLM 0.30.

The installed vLLM files and extensions are left intact. This patch supplies
configuration and scheduling interfaces missing before upstream #56227. The
Ascend worker already implements replay slot masking and SWA read clamping.
Remove this module when the release lane carries the upstream protocol.
"""

import argparse
import dataclasses
import functools
import inspect
import textwrap
from typing import Any

import pydantic.dataclasses
from vllm.config import CacheConfig
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs
from vllm.v1.request import Request


def _declare_bool(cls: Any, name: str) -> None:
    """Add a real dataclass field, including its serialization metadata."""
    if name in cls.__dataclass_fields__:
        return
    prototype: Any = dataclasses.make_dataclass("ReplayConfigField", [(name, bool, False)])
    cls.__annotations__[name] = bool
    cls.__dataclass_fields__ = dict(cls.__dataclass_fields__)
    cls.__dataclass_fields__[name] = prototype.__dataclass_fields__[name]
    setattr(cls, name, False)


def _patch_config() -> None:
    _declare_bool(CacheConfig, "swa_bounded_replay")
    pydantic.dataclasses.rebuild_dataclass(CacheConfig, force=True, raise_errors=True)
    for cls in (EngineArgs, AsyncEngineArgs):
        _declare_bool(cls, "swa_bounded_replay")
        original = cls.__init__

        def make_init(original):
            @functools.wraps(original)
            def init(self, *args, swa_bounded_replay=False, **kwargs):
                original(self, *args, **kwargs)
                self.swa_bounded_replay = swa_bounded_replay

            return init

        cls.__init__ = make_init(original)
    _rewrite_method(
        EngineArgs,
        "create_engine_config",
        (
            (
                "cache_config = CacheConfig(\n",
                "cache_config = CacheConfig(\n        swa_bounded_replay=self.swa_bounded_replay,\n",
            ),
        ),
    )


def register_legacy_replay_argument(parser) -> None:
    if (
        "swa_bounded_replay" in EngineArgs.__dataclass_fields__
        and "--swa-bounded-replay" not in parser._option_string_actions
    ):
        parser.add_argument(
            "--swa-bounded-replay",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Rebuild the DeepSeek V4.1 SWA window after a local prefix-cache hit.",
        )


def _rewrite_method(cls, name: str, replacements: tuple[tuple[str, str], ...], *, install: bool = True):
    """Retain the pinned scheduler body, inserting only the replay protocol.

    Every insertion must match once. A changed upstream body fails explicitly
    rather than installing a partially patched scheduler. No source file is
    rewritten and no newer scheduler implementation replaces the pinned one.
    """
    original = getattr(cls, name)
    source = textwrap.dedent(inspect.getsource(original))
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f"Cannot backport SWA replay: unexpected {cls.__name__}.{name} body")
        source = source.replace(before, after)
    namespace: dict[str, Any] = {}
    exec(compile(source, inspect.getfile(original), "exec"), original.__globals__, namespace)
    patched = namespace[name]
    functools.update_wrapper(patched, original)
    if install:
        setattr(cls, name, patched)
    return patched


def _mark_prefix_replay(self, request, num_hit_tokens: int) -> int:
    request.replay_start = 0
    if not self.prefix_replay_tokens:
        return 0
    num_replay_tokens = min(self.prefix_replay_tokens, num_hit_tokens)
    request.replay_start = num_hit_tokens - num_replay_tokens
    if num_replay_tokens:
        logger.info_once("SWA bounded replay is active: a local prefix hit is rebuilding its window.")
        logger.debug(
            "SWA bounded replay: hit=%d start=%d tokens=%d", num_hit_tokens, request.replay_start, num_replay_tokens
        )
    return num_replay_tokens


def _replay_window(spec) -> int:
    nested = getattr(spec, "kv_cache_specs", None)
    if nested is not None:
        return max((_replay_window(s) for s in nested.values()), default=0)
    return getattr(spec, "prefix_replay_tokens", 0)


def _initialize_replay_window(self) -> None:
    enabled = getattr(self.cache_config, "swa_bounded_replay", False)
    windows = (
        {_replay_window(g.kv_cache_spec) for g in self.kv_cache_manager.kv_cache_config.kv_cache_groups}
        if enabled
        else set()
    )
    windows.discard(0)
    if len(windows) > 1:
        raise ValueError("SWA bounded replay requires matching window sizes")
    prefix_replay_tokens = next(iter(windows), 0)
    if prefix_replay_tokens and self.connector is not None:
        raise ValueError("The vLLM 0.30 SWA replay backport supports local prefix caching only")
    self.prefix_replay_tokens = prefix_replay_tokens
    if prefix_replay_tokens:
        logger.info("SWA bounded replay enabled on vLLM 0.30: window=%d", prefix_replay_tokens)


def _patch_scheduler() -> None:
    """Validate the pinned bodies, then install the opt-in replay behavior."""
    original_init = Scheduler.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _initialize_replay_window(self)

    original_schedule = Scheduler.schedule
    replay_schedule = _rewrite_method(
        Scheduler,
        "schedule",
        (
            (
                "assert num_computed_tokens <= request.num_tokens",
                "if 0 < num_computed_tokens <= self.prefix_replay_tokens:\n"
                "                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks\n"
                "                    num_new_local_computed_tokens = 0\n"
                "                    num_external_computed_tokens = 0\n"
                "                    num_computed_tokens = 0\n"
                "                assert num_computed_tokens <= request.num_tokens",
            ),
            (
                "pad_spec_decode = False\n\n            if load_kv_async:",
                "pad_spec_decode = False\n            num_replay_tokens = 0\n\n            if load_kv_async:",
            ),
            (
                "request_token_budget = min(token_budget, input_budget - draft_slots)\n"
                "                # Number of tokens",
                "request_token_budget = min(token_budget, input_budget - draft_slots)\n"
                "                if did_prefix_cache_lookup:\n"
                "                    num_replay_tokens = self._mark_prefix_replay(request, num_computed_tokens)\n"
                "                    num_computed_tokens -= num_replay_tokens\n"
                "                # Number of tokens",
            ),
            (
                "new_blocks = self.kv_cache_manager.allocate_slots(\n"
                "                request,\n                num_new_tokens,",
                "new_blocks = self.kv_cache_manager.allocate_slots(\n"
                "                request,\n                max(num_new_tokens - num_replay_tokens, 0),",
            ),
        ),
        install=False,
    )
    replay_allocate_slots = _rewrite_method(
        KVCacheManager,
        "allocate_slots",
        (
            (
                "if num_new_tokens == 0 and num_external_computed_tokens == 0:",
                "if num_new_tokens == 0 and num_external_computed_tokens == 0 and num_new_computed_tokens == 0:",
            ),
        ),
        install=False,
    )

    @functools.wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        if not getattr(self.cache_config, "swa_bounded_replay", False):
            return original_schedule(self, *args, **kwargs)
        # Config revalidation can install this patch after a scheduler has
        # already entered its original constructor (including AsyncScheduler).
        if not hasattr(self, "prefix_replay_tokens"):
            _initialize_replay_window(self)
        return replay_schedule(self, *args, **kwargs)

    # Validate both source bodies before installing any scheduler behavior.
    Scheduler.__init__ = init
    Scheduler._mark_prefix_replay = _mark_prefix_replay
    Scheduler.schedule = schedule
    KVCacheManager.allocate_slots = replay_allocate_slots


def _patch_payload() -> None:
    # Declare the field before msgspec first inspects NewRequestData.
    prototype: Any = dataclasses.make_dataclass("ReplayPayloadField", [("replay_start", int, 0)])
    NewRequestData.__annotations__["replay_start"] = int
    NewRequestData.__dataclass_fields__["replay_start"] = prototype.__dataclass_fields__["replay_start"]
    NewRequestData.replay_start = 0
    del NewRequestData.__init__
    dataclasses.dataclass(NewRequestData)
    original_from_request = NewRequestData.from_request.__func__

    @functools.wraps(original_from_request)
    def from_request(cls, request, *args, **kwargs):
        result = original_from_request(cls, request, *args, **kwargs)
        result.replay_start = request.replay_start
        return result

    NewRequestData.from_request = classmethod(from_request)
    # Do not import the resume patch here: it must capture the scheduler after
    # BalanceScheduler and PP/MTP have installed their wrappers.
    if "replay_start" not in CachedRequestData.__dataclass_fields__:
        CachedRequestData.__annotations__["replay_start"] = dict[str, int]
        CachedRequestData.replay_start = dataclasses.field(default_factory=dict)
        del CachedRequestData.__init__
        dataclasses.dataclass(CachedRequestData)


def _patch_cache() -> None:
    original_cache_blocks = SingleTypeKVCacheManager.cache_blocks

    @functools.wraps(original_cache_blocks)
    def cache_blocks(self, *args, **kwargs):
        if not self.kv_cache_spec.prefix_cacheable:
            return
        return original_cache_blocks(self, *args, **kwargs)

    SingleTypeKVCacheManager.cache_blocks = cache_blocks


def apply_legacy_replay_patch(vllm_config) -> None:
    """Install the admission backport only after replay is explicitly enabled."""
    if not getattr(vllm_config.cache_config, "swa_bounded_replay", False):
        return
    if hasattr(Scheduler, "_mark_prefix_replay"):
        return
    _patch_scheduler()
    _patch_cache()
    # The resume module was imported before the scheduler backport existed.
    from vllm_ascend.patch.platform.patch_swa_bounded_replay import _patch_make_cached_request_data

    _patch_make_cached_request_data()


def _register_legacy_replay_interfaces() -> None:
    """Register CLI and wire fields before parsing or msgspec caches the schema.

    This does not replace scheduler or cache methods. Both sides of the worker
    boundary need the same payload schema even before the engine config exists.
    Native replay implementations retain their own interfaces and behavior.
    """
    if hasattr(Scheduler, "_mark_prefix_replay"):
        return
    # Workers also read the window from group specs, including uniform groups.
    KVCacheSpec.prefix_replay_tokens = property(lambda self: 0)
    UniformTypeKVCacheSpecs.prefix_replay_tokens = property(_replay_window)
    _patch_config()
    Request.replay_start = 0
    _patch_payload()


_register_legacy_replay_interfaces()
