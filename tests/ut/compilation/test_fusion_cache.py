# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import patch

import pytest
import torch
from torch._dynamo.utils import counters

from vllm_ascend.compilation.fusion_cache import compile_cached_fusion


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    torch._dynamo.reset()
    yield tmp_path
    torch._dynamo.reset()


def test_cache_preserves_mutation_aliases_and_dynamic_shapes(isolated_cache):
    compilations = []

    def inner(graph, inputs):
        compilations.append(graph)
        return graph

    def backend(graph, inputs):
        return compile_cached_fusion(graph, inputs, inner)

    def model(x, y):
        x.add_(y)
        return x, x.view(-1), x.sin()

    hits_before = counters["aot_autograd"]["autograd_cache_hit"]
    for _ in range(2):
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
        for rows in (2, 5):
            x, y = torch.randn(rows, 3), torch.randn(rows, 3)
            expected = x + y
            with torch.no_grad():
                mutated, view, sine = compiled(x, y)
            torch.testing.assert_close(x, expected)
            torch.testing.assert_close(mutated, expected)
            torch.testing.assert_close(view, expected.reshape(-1))
            torch.testing.assert_close(sine, expected.sin())
            assert mutated.data_ptr() == x.data_ptr()
            assert view.data_ptr() == x.data_ptr()
    assert len(compilations) == 1
    assert counters["aot_autograd"]["autograd_cache_hit"] > hits_before


def test_serialization_failure_falls_back_to_uncached_graph(isolated_cache):
    def model(x):
        return x.cos() + 1

    def backend(graph, inputs):
        return compile_cached_fusion(graph, inputs, lambda graph, inputs: graph)

    with patch("vllm_ascend.compilation.fusion_cache.GraphPickler.dumps", side_effect=TypeError("unsupported")):
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        x = torch.randn(3, 4)
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), model(x))


def test_corrupted_cache_recompiles(isolated_cache):
    compilations = []

    def inner(graph, inputs):
        compilations.append(graph)
        return graph

    def backend(graph, inputs):
        return compile_cached_fusion(graph, inputs, inner)

    def model(x):
        return x.cos() + 2

    x = torch.randn(3, 4)
    with torch.no_grad():
        torch.testing.assert_close(torch.compile(model, backend=backend)(x), model(x))
    entries = list((isolated_cache / "aotautograd").glob("*/*"))
    assert entries
    for entry in entries:
        if entry.is_file():
            entry.write_bytes(b"invalid cache entry")
    torch._dynamo.reset()
    with torch.no_grad():
        torch.testing.assert_close(torch.compile(model, backend=backend)(x), model(x))
    assert len(compilations) == 2


@pytest.mark.parametrize("disable_cache,sequence_parallel", [(True, False), (False, True), (False, False)])
def test_cache_respects_disable_and_sequence_parallel(isolated_cache, disable_cache, sequence_parallel):
    from types import SimpleNamespace

    from vllm.config.utils import Range

    from vllm_ascend.compilation.compiler_interface import AscendCompiler
    from vllm_ascend.utils import COMPILATION_PASS_KEY

    graph = torch.fx.symbolic_trace(lambda x: (x.sin(),))
    compiler = AscendCompiler()
    # No legacy fuse_gemm_comms attribute: follow enable_sp.
    compiler.vllm_config = SimpleNamespace()
    config = SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=False))
    with (
        patch("vllm_ascend.compilation.compiler_interface.get_ascend_config", return_value=config),
        patch("vllm_ascend.compilation.compiler_interface.enable_sp", return_value=sequence_parallel),
        patch(
            "vllm_ascend.compilation.compiler_interface.compile_cached_fusion", wraps=compile_cached_fusion
        ) as cached,
    ):
        compiler.initialize_cache(str(isolated_cache), disable_cache=disable_cache)
        x = torch.randn(2, 3)
        with torch.no_grad():
            result, handle = compiler.compile(graph, [x], {COMPILATION_PASS_KEY: lambda gm: gm}, Range(1, 8))
            torch.testing.assert_close(result(x), graph(x))
    assert handle is None
    assert cached.call_count == int(not disable_cache and not sequence_parallel)
    if disable_cache or sequence_parallel:
        assert not list(isolated_cache.rglob("*"))


@pytest.mark.parametrize(
    "relative",
    [
        "graph_fusion_pass_manager.py",
        "passes/base_pattern.py",
        "passes/norm_quant_fusion_pass.py",
        "passes/qknorm_rope_fusion_pass.py",
        "passes/muls_add_pass.py",
        "passes/mm_reduce_scatter_fusion_pass.py",
        "passes/utils/npugraph_ex_utils_check.py",
    ],
)
def test_fusion_dependency_changes_invalidate_compiler_hash(tmp_path, monkeypatch, relative):
    import shutil
    from pathlib import Path
    from types import SimpleNamespace

    import vllm_ascend.compilation.compiler_interface as interface
    from vllm_ascend.ascend_config import AscendCompilationConfig

    source = Path(interface.__file__).parent
    copied = tmp_path / "compilation"
    copied.mkdir()
    for name in interface.FUSION_CACHE_SOURCE_FILES:
        target = copied / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    # Copy the exercised dependency even if a regressed allowlist omits it.
    dependency = copied / relative
    dependency.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / relative, dependency)
    monkeypatch.setattr(interface, "__file__", str(copied / "compiler_interface.py"))
    monkeypatch.setattr(
        interface,
        "get_ascend_config",
        lambda: SimpleNamespace(ascend_compilation_config=AscendCompilationConfig(enable_npugraph_ex=False)),
    )
    monkeypatch.setattr(interface, "get_current_hardware_profile", lambda: SimpleNamespace(capabilities=[]))
    compiler = interface.AscendCompiler()
    before = compiler.compute_hash(None)
    (copied / "unrelated.py").write_text("UNRELATED = True\n")
    assert compiler.compute_hash(None) == before
    path = copied / relative
    path.write_text(path.read_text() + "\n# Changed fusion implementation\n")
    assert compiler.compute_hash(None) != before


def test_npugraph_hash_does_not_read_fusion_sources(monkeypatch):
    from types import SimpleNamespace

    import vllm_ascend.compilation.compiler_interface as interface

    config = SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=False, enable_super_kernel=False)
    monkeypatch.setattr(interface, "get_ascend_config", lambda: SimpleNamespace(ascend_compilation_config=config))
    with patch.object(interface.Path, "read_bytes", side_effect=AssertionError("unrelated fusion source")):
        assert interface.AscendCompiler().compute_hash(None)
