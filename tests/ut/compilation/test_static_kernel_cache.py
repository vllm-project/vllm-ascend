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
#
"""Tests for the static kernel run-package cache (issue #22725).

With ``enable_static_kernel`` enabled, vllm-ascend compiles every npugraph_ex
piece with ``force_eager=True``. torch_npu skips its run-package cache for
that combination, so every restart recompiled all static kernels and left a
new ``ts<timestamp>_pid<pid>_outputs`` directory behind.

The tests below drive the real ``npugraph_ex`` classes and the real torch_npu
decision function, so they fail on the pre-fix code even though the fix lives
in vllm-ascend.
"""

import os
from unittest.mock import Mock, patch

import pytest
import torch
from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StubConfig:
    """Minimal stand-in for ``npugraph_ex``'s CompilerConfig."""

    def __init__(self, static_kernel_compile: bool):
        self.static_kernel_compile = static_kernel_compile

    def as_dict(self):
        return {"static_kernel_compile": self.static_kernel_compile}, {}


def _enable_cache():
    """Apply the vllm-ascend static kernel cache patch."""
    from vllm_ascend.compilation.compiler_interface import _enable_static_kernel_run_package_cache

    _enable_static_kernel_run_package_cache()


def _cache_dir():
    from vllm_ascend.compilation.compiler_interface import _static_kernel_cache_dir

    return _static_kernel_cache_dir()


def _cache_options():
    from vllm_ascend.compilation.compiler_interface import _static_kernel_cache_options

    return _static_kernel_cache_options()


def test_cache_dir_follows_vllm_cache_root(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm_cache")
    assert _cache_dir() == os.path.join("/data/vllm_cache", "ascend_static_kernel")


def test_cache_dir_falls_back_to_the_user_cache(monkeypatch):
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    monkeypatch.setenv("HOME", "/home/tester")
    assert _cache_dir() == os.path.join("/home/tester", ".cache", "vllm", "ascend_static_kernel")


def test_cache_options_carry_the_torch_npu_cache_keys(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm_cache")
    options = _cache_options()

    assert options["static_kernel_compile.use_cache_compile"] is True
    assert options["static_kernel_compile.cann_version"]
    assert options["static_kernel_compile.compile_cache_dir"] == _cache_dir()
    assert options["static_kernel_compile.deterministic"] == torch.are_deterministic_algorithms_enabled()


def test_cache_options_track_the_deterministic_setting(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm_cache")
    with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
        assert _cache_options()["static_kernel_compile.deterministic"] is True


def test_cache_keys_reach_the_runtime_config():
    """The keys must land in the config dict the static kernel compiler reads.

    ``AclConcreteGraph.normalize_config`` builds that dict; without the patch
    the cache keys never reach ``compile_static_kernel``.
    """
    from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

    original_normalize_config = AclConcreteGraph.normalize_config
    _enable_cache()
    try:
        converter = AclConcreteGraph.__new__(AclConcreteGraph)
        converter._config = _StubConfig(static_kernel_compile=True)
        converter._aclgraph_cache_info = Mock(unupdated_sym_input_index=[])
        configs = AclConcreteGraph.normalize_config(converter)
    finally:
        AclConcreteGraph.normalize_config = original_normalize_config

    assert configs["static_kernel_compile.use_cache_compile"] is True
    assert configs["static_kernel_compile.compile_cache_dir"] == _cache_dir()
    assert "static_kernel_compile.cann_version" in configs
    assert "static_kernel_compile.deterministic" in configs


def test_cache_keys_are_absent_without_static_kernel():
    """Without static kernel the keys must not be added."""
    from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

    original_normalize_config = AclConcreteGraph.normalize_config
    _enable_cache()
    try:
        converter = AclConcreteGraph.__new__(AclConcreteGraph)
        converter._config = _StubConfig(static_kernel_compile=False)
        converter._aclgraph_cache_info = Mock(unupdated_sym_input_index=[])
        configs = AclConcreteGraph.normalize_config(converter)
    finally:
        AclConcreteGraph.normalize_config = original_normalize_config

    assert "static_kernel_compile.use_cache_compile" not in configs


def test_static_kernel_compiler_takes_the_cache_path_under_force_eager():
    """The whole point of the fix: force_eager must not disable the cache.

    torch_npu's own decision skips the cache for
    ``force_eager and static_kernel_compile``. vllm-ascend uses exactly that
    combination, so the patched compiler has to take the cache path anyway.
    """
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    try:
        calls = []
        config = {
            "force_eager": True,
            "static_kernel_compile": True,
            "static_kernel_compile.use_cache_compile": True,
            "static_kernel_compile.cann_version": "9.1.0",
            "static_kernel_compile.compile_cache_dir": "/tmp/ascend_static_kernel",
            "static_kernel_compile.deterministic": False,
        }
        with (
            patch.object(static_kernel, "_install_or_compile_static_kernel", lambda *a, **k: calls.append("cache")),
            patch.object(static_kernel, "_compile_static_kernel", lambda *a, **k: calls.append("compile")),
        ):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert calls == ["cache"]


def test_static_kernel_compiler_without_cache_keys_still_compiles():
    """Without the cache keys the compiler must keep the upstream behaviour."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    try:
        calls = []
        config = {"force_eager": True, "static_kernel_compile": True}
        with (
            patch.object(static_kernel, "_install_or_compile_static_kernel", lambda *a, **k: calls.append("cache")),
            patch.object(static_kernel, "_compile_static_kernel", lambda *a, **k: calls.append("compile")),
        ):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert calls == ["compile"]


def test_the_patch_is_applied_only_once():
    """initialize_cache() can run more than once; the wrappers must not stack."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    original_normalize_config = AclConcreteGraph.normalize_config
    try:
        _enable_cache()
        first_compiler = static_kernel.compile_static_kernel
        first_normalize = AclConcreteGraph.normalize_config

        _enable_cache()
        _enable_cache()

        assert static_kernel.compile_static_kernel is first_compiler
        assert AclConcreteGraph.normalize_config is first_normalize
        assert acl_graph.compile_static_kernel is first_compiler
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel
        AclConcreteGraph.normalize_config = original_normalize_config


def test_a_missing_config_object_does_not_add_the_cache_keys():
    """The config object belongs to npugraph_ex, so read it defensively.

    The patched ``normalize_config`` must not raise when the config object is
    absent or does not carry the flag, and it must not add the cache keys in
    that case. The upstream ``normalize_config`` reads the config object
    itself, so it is stubbed out here to isolate the patched part.
    """
    from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

    original_normalize_config = AclConcreteGraph.normalize_config
    try:
        with patch.object(AclConcreteGraph, "normalize_config", return_value={}):
            _enable_cache()
            converter = AclConcreteGraph.__new__(AclConcreteGraph)
            converter._config = None
            configs = AclConcreteGraph.normalize_config(converter)
    finally:
        AclConcreteGraph.normalize_config = original_normalize_config

    assert "static_kernel_compile.use_cache_compile" not in configs


def test_a_config_object_without_the_flag_does_not_add_the_cache_keys():
    """A config object without the flag must not get the cache keys."""
    from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

    original_normalize_config = AclConcreteGraph.normalize_config
    try:
        with patch.object(AclConcreteGraph, "normalize_config", return_value={}):
            _enable_cache()
            converter = AclConcreteGraph.__new__(AclConcreteGraph)
            converter._config = object()
            configs = AclConcreteGraph.normalize_config(converter)
    finally:
        AclConcreteGraph.normalize_config = original_normalize_config

    assert "static_kernel_compile.use_cache_compile" not in configs


def test_the_compiler_falls_back_when_the_install_path_fails():
    """A failure in the private install API must not fail the compilation."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    try:
        calls = []
        config = {
            "force_eager": True,
            "static_kernel_compile": True,
            "static_kernel_compile.use_cache_compile": True,
            "static_kernel_compile.cann_version": "9.1.0",
            "static_kernel_compile.compile_cache_dir": "/tmp/ascend_static_kernel",
            "static_kernel_compile.deterministic": False,
        }

        def failing_install(*args, **kwargs):
            raise RuntimeError("the private install API changed")

        with (
            patch.object(static_kernel, "_install_or_compile_static_kernel", failing_install),
            patch.object(
                static_kernel,
                "_compile_static_kernel",
                lambda *a, **k: calls.append("compile"),
            ),
        ):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert calls == ["compile"]
