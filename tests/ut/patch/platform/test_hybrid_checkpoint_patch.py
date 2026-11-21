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
"""Unit tests for the flashstate monkey patches.

The module is inert unless ``additional_config["flashstate"]["enable_frontier"]``
is set, so these tests cover the gate and the installed hooks only.
"""

import importlib
from types import SimpleNamespace

MODULE = "vllm_ascend.patch.platform.patch_hybrid_checkpoint"


def _module():
    return importlib.import_module(MODULE)


def test_settings_accepts_only_a_mapping():
    module = _module()

    assert module._settings(SimpleNamespace(additional_config={"flashstate": {"enable_frontier": True}})) == {
        "enable_frontier": True
    }
    assert module._settings(SimpleNamespace(additional_config=None)) == {}
    assert module._settings(SimpleNamespace(additional_config=["not", "a", "dict"])) == {}
    assert module._settings(SimpleNamespace(additional_config={"flashstate": "on"})) == {}
    assert module._settings(SimpleNamespace()) == {}


def test_disabled_by_default():
    module = _module()

    assert module._enabled(SimpleNamespace(additional_config=None)) is False
    assert module._enabled(SimpleNamespace(additional_config={"flashstate": {}})) is False
    assert module._enabled(SimpleNamespace(additional_config={"flashstate": {"enable_frontier": False}})) is False


def test_enabled_by_the_frontier_switch():
    module = _module()

    assert module._enabled(SimpleNamespace(additional_config={"flashstate": {"enable_frontier": True}})) is True


def test_hooks_are_installed():
    module = _module()

    # Re-running the installer is safe: every hook delegates to the handle
    # captured at import time, so a second install does not nest.
    module._install()

    # Assert against the classes the patch itself holds. Other platform patches
    # may swap the module-level ``Scheduler`` symbol for a subclass, so reading
    # it afresh here would compare against the wrong object.
    assert module.Scheduler.__init__ is module._scheduler_init
    assert module.Scheduler._free_request is module._free_request
    assert module.Scheduler._preempt_request is module._preempt_request
    assert module.Scheduler.reset_prefix_cache is module._reset_prefix_cache
    assert module.KVCacheManager.get_computed_blocks is module._get_computed_blocks
    assert module.KVCacheManager.allocate_slots is module._allocate_slots
    assert module.BlockPool.__init__ is module._block_pool_init
    assert module.BlockPool._remove_cached_block_hashes is module._remove_cached_block_hashes
    assert module.MambaManager.__init__ is module._mamba_init
    assert module.MambaManager.cache_blocks is module._mamba_cache_blocks


def test_install_is_idempotent():
    # Installing twice must not wrap the hooks again: every shim delegates to a
    # handle captured at import time.
    module = _module()

    module._install()
    first = module.Scheduler.__init__
    module._install()

    assert module.Scheduler.__init__ is first


def test_no_mutable_module_counters():
    # Regression guard: diagnostics must not be module-level mutable state.
    module = _module()

    assert not hasattr(module, "_DIAG")
    assert not hasattr(module, "_TIME")
    assert not hasattr(module, "_diag_bump")


def _bounded_request(num_tokens, prefix_cache_hit_limit=None):
    class _Request:
        def __init__(self):
            self.num_tokens = num_tokens
            self.prefix_cache_hit_limit = prefix_cache_hit_limit
            self.request_id = "req"

        def get_max_cache_hit_length(self, bound=None):
            # Same rule as ``vllm.v1.request.Request``: the last token is always
            # recomputed and an explicit bound may only tighten, never widen.
            limits = (self.num_tokens - 1, self.prefix_cache_hit_limit, bound)
            return min(limit for limit in limits if limit is not None)

    return _Request()


def test_frontier_extension_respects_the_caller_bound(monkeypatch):
    # A frontier longer than the caller's bound must not be resumed into: the
    # core lookup tightened ``max_cache_hit_length`` and the wrapper has to use
    # the same value, otherwise tokens the caller asked to recompute come back
    # as cache hits.
    module = _module()

    class _Blocks:
        blocks = ("aligned",)

    class _Checkpoints:
        def __init__(self):
            self.max_lengths = []

        def extend_hit(self, request, blocks, hit, max_length):
            self.max_lengths.append(max_length)
            return blocks, hit

        def admitted(self, request, num_computed_tokens, blocks, num_external_tokens):
            pass

    checkpoints = _Checkpoints()
    manager = SimpleNamespace(checkpoints=checkpoints)
    monkeypatch.setattr(
        module,
        "_original_get_computed_blocks",
        lambda self, request, max_cache_hit_length=None: (_Blocks(), 8, 0),
    )

    module._get_computed_blocks(manager, _bounded_request(num_tokens=100), 16)
    module._get_computed_blocks(manager, _bounded_request(num_tokens=100, prefix_cache_hit_limit=32))
    module._get_computed_blocks(manager, _bounded_request(num_tokens=100))

    assert checkpoints.max_lengths == [16, 32, 99]
