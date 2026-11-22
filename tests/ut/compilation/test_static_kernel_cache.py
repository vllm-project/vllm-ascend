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

``npugraph_ex`` is vendored in torch_npu and is only present on the NPU
images, so the tests that need it are skipped on the CPU runners. The cache
directory and cache option helpers are pure Python, so they are tested
everywhere: ``compiler_interface`` imports ``npugraph_ex`` at module level, so
it is stubbed out when the package is missing, the same way
``test_compiler_interface.py`` does it.
"""

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

try:
    import npugraph_ex._acl_concrete_graph.fx2acl_converter  # noqa: F401

    npugraph_ex_available = True
except ImportError:
    npugraph_ex_available = False

needs_npugraph_ex = pytest.mark.skipif(
    not npugraph_ex_available,
    reason="npugraph_ex is only available on the NPU images",
)

_installed_stub_names: tuple[str, ...] = ()
if not npugraph_ex_available:
    # CPU CI has no npugraph_ex (it is vendored in torch_npu). Stub the
    # symbols compiler_interface imports at module level so the pure-Python
    # helpers stay testable on the CPU runners.

    class _StaticKernelStub(ModuleType):
        _set_static_kernel_blacklist = MagicMock()

    _installed_stub_names = (
        "npugraph_ex",
        "npugraph_ex._acl_concrete_graph",
        "npugraph_ex._acl_concrete_graph.static_kernel",
    )
    sys.modules.setdefault("npugraph_ex", ModuleType("npugraph_ex"))
    sys.modules.setdefault("npugraph_ex._acl_concrete_graph", ModuleType("npugraph_ex._acl_concrete_graph"))
    sys.modules["npugraph_ex._acl_concrete_graph.static_kernel"] = _StaticKernelStub(
        "npugraph_ex._acl_concrete_graph.static_kernel"
    )

from vllm_ascend.compilation import compiler_interface  # noqa: E402

for _name in _installed_stub_names:
    # Drop the temporary stubs again so importlib.util.find_spec() keeps
    # reporting npugraph_ex as absent for later collectors (a spec-less
    # sys.modules entry makes find_spec raise ValueError).
    sys.modules.pop(_name, None)


@pytest.fixture(autouse=True)
def _reset_static_kernel_cache_patch():
    """Keep the tests independent of each other and of their order.

    The patch replaces torch_npu's compiler functions, so they have to be
    restored after every test.
    """
    if npugraph_ex_available:
        from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

        original_compile_static_kernel = static_kernel.compile_static_kernel
        original_normalize_config = AclConcreteGraph.normalize_config
    try:
        yield
    finally:
        if npugraph_ex_available:
            static_kernel.compile_static_kernel = original_compile_static_kernel
            acl_graph.compile_static_kernel = original_compile_static_kernel
            AclConcreteGraph.normalize_config = original_normalize_config


class _StubConfig:
    """Minimal stand-in for ``npugraph_ex``'s CompilerConfig."""

    def __init__(self, static_kernel_compile: bool):
        self.static_kernel_compile = static_kernel_compile

    def as_dict(self):
        return {"static_kernel_compile": self.static_kernel_compile}, {}


def _enable_cache():
    """Apply the vllm-ascend static kernel cache patch."""
    compiler_interface._enable_static_kernel_run_package_cache()


def _cache_root():
    return compiler_interface._static_kernel_cache_root()


def _cache_options():
    return compiler_interface._static_kernel_cache_options()


def _cache_config(**overrides):
    """Return a runtime config that carries the torch_npu cache keys."""
    config = {
        "force_eager": True,
        "static_kernel_compile": True,
        "static_kernel_compile.use_cache_compile": True,
        "static_kernel_compile.cann_version": "9.1.0",
        "static_kernel_compile.compile_cache_dir": "/tmp/ascend_static_kernel",
        "static_kernel_compile.deterministic": False,
        "static_kernel_compile.deterministic_level": 0,
    }
    config.update(overrides)
    return config


def test_cache_root_follows_vllm_cache_root(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm_cache")
    assert _cache_root() == os.path.join("/data/vllm_cache", "ascend_static_kernel")


def test_cache_root_falls_back_to_the_user_cache(monkeypatch):
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("USERPROFILE", "/home/tester")
    assert _cache_root() == os.path.join("/home/tester", ".cache", "vllm", "ascend_static_kernel")


def test_cache_options_carry_the_torch_npu_cache_keys(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm_cache")
    options = _cache_options()

    assert options["static_kernel_compile.use_cache_compile"] is True
    assert options["static_kernel_compile.cann_version"]
    assert options["static_kernel_compile.compile_cache_dir"] == _cache_root()
    assert options["static_kernel_compile.deterministic"] == torch.are_deterministic_algorithms_enabled()
    assert "static_kernel_compile.deterministic_level" in options


def test_cache_options_track_the_deterministic_setting(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm_cache")
    with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
        assert _cache_options()["static_kernel_compile.deterministic"] is True


def test_the_cache_dir_is_isolated_per_graph_and_shape(tmp_path):
    """Every compiled piece needs its own cache directory.

    torch_npu installs the first run package it finds under the cache
    directory without checking which graph it belongs to, so a shared
    directory would let the first compiled piece short-circuit all the
    others.
    """
    graph_a = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
    graph_a["static_kernel_compile.graph_id"] = "graph_a"
    graph_b = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
    graph_b["static_kernel_compile.graph_id"] = "graph_b"

    args_a = (torch.zeros(1, 8),)
    args_b = (torch.zeros(1, 16),)

    dir_a = compiler_interface._static_kernel_piece_cache_dir(graph_a, None, args_a)
    dir_b = compiler_interface._static_kernel_piece_cache_dir(graph_b, None, args_a)
    dir_c = compiler_interface._static_kernel_piece_cache_dir(graph_a, None, args_b)

    assert dir_a != dir_b, "different graphs must not share a cache directory"
    assert dir_a != dir_c, "different shapes must not share a cache directory"
    assert os.path.dirname(dir_a) == str(tmp_path)
    assert dir_a == compiler_interface._static_kernel_piece_cache_dir(graph_a, None, args_a)


def test_the_cache_dir_is_isolated_per_runtime_version(tmp_path):
    """A CANN upgrade must not reuse the run packages of the old version."""
    config = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
    args = (torch.zeros(1, 8),)

    old = compiler_interface._static_kernel_piece_cache_dir(config, None, args)
    new_properties = dict(compiler_interface._static_kernel_runtime_properties())
    new_properties["cann_version"] = "9.2.1"
    with patch.object(compiler_interface, "_static_kernel_runtime_properties", return_value=new_properties):
        new = compiler_interface._static_kernel_piece_cache_dir(config, None, args)

    assert old != new


def test_the_cache_dir_is_isolated_per_deterministic_setting(tmp_path):
    config = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
    args = (torch.zeros(1, 8),)

    default = compiler_interface._static_kernel_piece_cache_dir(config, None, args)
    new_properties = dict(compiler_interface._static_kernel_runtime_properties())
    new_properties["deterministic"] = True
    with patch.object(compiler_interface, "_static_kernel_runtime_properties", return_value=new_properties):
        deterministic = compiler_interface._static_kernel_piece_cache_dir(config, None, args)

    assert default != deterministic


def test_the_graph_id_identifies_the_graph():
    """The cache keys carry the identity of the graph they belong to."""
    graph_a = SimpleNamespace(code="def forward(self):\n    return 1\n")
    graph_b = SimpleNamespace(code="def forward(self):\n    return 2\n")

    id_a = compiler_interface._static_kernel_graph_id(SimpleNamespace(_fx_graph=graph_a))
    id_b = compiler_interface._static_kernel_graph_id(SimpleNamespace(_fx_graph=graph_b))

    assert id_a
    assert id_a != id_b
    assert id_a == compiler_interface._static_kernel_graph_id(SimpleNamespace(_fx_graph=graph_a))

    # A graph that is not available must not produce an identifier.
    assert compiler_interface._static_kernel_graph_id(SimpleNamespace()) == ""

    options = compiler_interface._static_kernel_cache_options(SimpleNamespace(_fx_graph=graph_a))
    assert options["static_kernel_compile.graph_id"] == id_a


def test_the_cache_metadata_is_persisted_and_read_back(tmp_path):
    """The runtime properties of a cached package are read from the disk.

    torch_npu compares the properties the cached run packages were built
    with against the current runtime, so they have to come from the metadata
    that was written when the packages were compiled.
    """
    cache_dir = str(tmp_path / "piece")
    meta = {"cann_version": "9.1.0", "deterministic": False, "deterministic_level": 0}
    compiler_interface._write_static_kernel_cache_meta(cache_dir, meta)

    assert compiler_interface._read_static_kernel_cache_meta(cache_dir) == meta

    # A truncated metadata file must not be used.
    with open(compiler_interface._static_kernel_cache_meta_path(cache_dir), "w") as f:
        f.write("{not json")
    assert compiler_interface._read_static_kernel_cache_meta(cache_dir) is None


@needs_npugraph_ex
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
    assert configs["static_kernel_compile.compile_cache_dir"] == _cache_root()
    assert "static_kernel_compile.cann_version" in configs
    assert "static_kernel_compile.deterministic" in configs
    assert "static_kernel_compile.deterministic_level" in configs


@needs_npugraph_ex
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


@needs_npugraph_ex
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
        config = _cache_config()
        with (
            patch.object(static_kernel, "_install_or_compile_static_kernel", lambda *a, **k: calls.append("cache")),
            patch.object(static_kernel, "_compile_static_kernel", lambda *a, **k: calls.append("compile")),
        ):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert calls == ["cache"]


@needs_npugraph_ex
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


@needs_npugraph_ex
def test_the_cache_directory_is_created_before_it_is_used(tmp_path):
    """A missing cache directory makes torch_npu skip the compilation.

    ``_check_for_load_run_pkg`` refuses a directory that does not exist and
    the caller then returns without compiling anything, so the directory has
    to be created before torch_npu sees it.
    """
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    seen = {}

    def fake_install(*args, **kwargs):
        seen["compile_cache_dir"] = kwargs["compile_cache_dir"]
        seen["exists"] = os.path.isdir(kwargs["compile_cache_dir"])

    try:
        config = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
        with patch.object(static_kernel, "_install_or_compile_static_kernel", fake_install):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert seen["exists"], "the cache directory must exist before torch_npu uses it"
    assert seen["compile_cache_dir"].startswith(str(tmp_path))


@needs_npugraph_ex
def test_the_cached_properties_are_read_from_the_metadata(tmp_path):
    """The properties handed to torch_npu come from the persisted metadata.

    Comparing the live runtime values against themselves never invalidates
    anything, so the values recorded when the run packages were compiled are
    the ones that are checked.
    """
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    seen = {}

    def fake_install(*args, **kwargs):
        seen.update(kwargs)

    try:
        config = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
        with patch.object(static_kernel, "_install_or_compile_static_kernel", fake_install):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")

            cache_dir = seen["compile_cache_dir"]
            # Pretend the packages were built with another CANN version.
            meta = {
                "cann_version": "8.0.0",
                "deterministic": True,
                "deterministic_level": 3,
            }
            compiler_interface._write_static_kernel_cache_meta(cache_dir, meta)

            seen.clear()
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert seen["cached_cann_version"] == "8.0.0"
    assert seen["cached_deterministic"] is True
    assert seen["cached_deterministic_level"] == 3


@needs_npugraph_ex
def test_the_deterministic_level_is_handed_over(tmp_path):
    """torch_npu checks the CANN deterministic level, so it has to be passed."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    seen = {}

    def fake_install(*args, **kwargs):
        seen.update(kwargs)

    try:
        config = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
        with patch.object(static_kernel, "_install_or_compile_static_kernel", fake_install):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert seen["cached_deterministic_level"] == 0


@needs_npugraph_ex
def test_an_invalid_multicard_environment_falls_back_to_compiling(tmp_path):
    """torch_npu disables the static kernel feature in that case."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    try:
        calls = []
        config = _cache_config(**{"static_kernel_compile.compile_cache_dir": str(tmp_path)})
        with (
            patch.object(static_kernel, "_is_multicard_env_valid", return_value=False),
            patch.object(static_kernel, "_install_or_compile_static_kernel", lambda *a, **k: calls.append("cache")),
            patch.object(static_kernel, "_compile_static_kernel", lambda *a, **k: calls.append("compile")),
        ):
            acl_graph.compile_static_kernel(lambda *a, **k: None, config=config, build_dir="/tmp")
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert calls == ["compile"]


@needs_npugraph_ex
def test_the_patch_is_applied_only_once():
    """initialize_cache() can run more than once; the wrappers must not stack."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel
    from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

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


@needs_npugraph_ex
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


@needs_npugraph_ex
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


@needs_npugraph_ex
def test_the_compiler_falls_back_when_the_install_path_fails():
    """A failure in the private install API must not fail the compilation."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    _enable_cache()
    try:
        calls = []
        config = _cache_config()

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


@needs_npugraph_ex
def test_the_run_package_cache_is_not_enabled_when_the_cache_is_disabled(tmp_path):
    """``disable_cache`` must disable the run-package cache as well."""
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel

    original_compile_static_kernel = static_kernel.compile_static_kernel
    original_normalize_config = static_kernel.compile_static_kernel
    ascend_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=True)
    )
    compiler = compiler_interface.AscendCompiler()
    compiler.vllm_config = SimpleNamespace()
    try:
        with (
            patch.object(compiler_interface, "get_ascend_config", return_value=ascend_config),
            patch.object(compiler_interface, "_initialize_static_kernel_runtime"),
        ):
            compiler.initialize_cache(str(tmp_path), disable_cache=True)
    finally:
        static_kernel.compile_static_kernel = original_compile_static_kernel
        acl_graph.compile_static_kernel = original_compile_static_kernel

    assert static_kernel.compile_static_kernel is original_normalize_config
