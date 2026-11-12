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
