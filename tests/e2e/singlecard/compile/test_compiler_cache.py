#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""
Tests for AscendCompiler cache protocol (compile / load).

Design notes
------------
The save path relies on patched_init inside npugraph_ex_compile, which patches
_CompiledFxGraph.__init__ to eagerly call get_code() at construction time.
Tests mock _CompiledFxGraph.get_code to return a fake py_code string so that
the caching mechanism can be exercised without depending on torchair's internal
codegen producing a specific output.

The load path reads the py_code text file written by compile(), builds a
_CompiledFxArtifacts object, and calls _CompiledFxGraph.load_artifacts().
load_artifacts() calls _compile_py_code() which exec()s the py_code string and
extracts the `kernel` symbol.  FAKE_PY_CODE therefore defines a minimal kernel
function that satisfies this contract.

What is validated
-----------------
1. compile() save path: when patched_init fires and get_code() returns a string,
   the cache file is written as a text file and a non-None handle is returned.
2. load() roundtrip: compile() writes the cache; load() restores it and returns
   a callable _CompiledFxGraph instance.
3. Model-structure isolation: two distinct graph keys produce separate cache
   files; each is independently loadable.
4. disable_cache suppression: compile() returns None handle when cache is off.
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from vllm.config import ModelConfig, VllmConfig
from vllm.config.utils import Range

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.compilation.compiler_interface import AscendCompiler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN = 64
BATCH = 4
DTYPE = torch.bfloat16

# Minimal valid py_code: _compile_py_code exec()s this and looks up `kernel`.
FAKE_PY_CODE = "def kernel(*args, **kwargs):\n    return args\n"


# ---------------------------------------------------------------------------
# Minimal models
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeeperModel(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs() -> list[torch.Tensor]:
    return [torch.randn(BATCH, HIDDEN, device="npu", dtype=DTYPE)]


def _capture_graph(model: nn.Module, example_inputs: list[torch.Tensor]):
    """Capture an FX GraphModule via a no-op dynamo backend.

    Dynamo passes FakeTensors as *inputs* – same convention as vllm uses.
    """
    captured: dict = {}

    def _noop(gm, inputs):
        captured["graph"] = gm
        captured["inputs"] = inputs
        return gm.forward

    with torch.no_grad():
        torch.compile(model, backend=_noop, fullgraph=True)(*example_inputs)

    assert "graph" in captured
    return captured["graph"], captured["inputs"]


def _make_vllm_config(enable_npugraph_ex: bool = True) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(dtype=DTYPE),
        additional_config={
            "ascend_compilation_config": {"enable_npugraph_ex": enable_npugraph_ex},
            "refresh": True,
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compiler_with_cache(tmp_path):
    cfg = _make_vllm_config(enable_npugraph_ex=True)
    init_ascend_config(cfg)
    c = AscendCompiler()
    c.compute_hash(cfg)
    c.initialize_cache(cache_dir=str(tmp_path), disable_cache=False)
    return c


@pytest.fixture
def compiler_no_cache(tmp_path):
    cfg = _make_vllm_config(enable_npugraph_ex=True)
    init_ascend_config(cfg)
    c = AscendCompiler()
    c.compute_hash(cfg)
    c.initialize_cache(cache_dir=str(tmp_path), disable_cache=True)
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_compile_produces_cache_file(compiler_with_cache):
    """compile() must write a text cache file and return a non-None handle.

    _CompiledFxGraph.get_code is mocked to return FAKE_PY_CODE so the test
    does not depend on torchair codegen producing a specific output.
    patched_init (inside npugraph_ex_compile) captures this value into
    py_code_holder, which is then written to the cache file.
    """
    from torchair.npu_fx_compiler import _CompiledFxGraph

    model = SimpleModel().to("npu").to(DTYPE)
    graph, fake_inputs = _capture_graph(model, _make_inputs())

    with patch.object(_CompiledFxGraph, "get_code", return_value=FAKE_PY_CODE):
        compiled_fn, handle = compiler_with_cache.compile(
            graph=graph,
            example_inputs=fake_inputs,
            compiler_config={},
            compile_range=Range(BATCH, BATCH),
            key="simple_model_v1",
        )

    assert handle is not None, "compile() must return a handle when cache is enabled"
    key_name, cache_path = handle
    assert key_name == "simple_model_v1"
    assert os.path.isfile(cache_path), f"Cache file not found at {cache_path}"
    assert os.path.getsize(cache_path) > 0, "Cache file must not be empty"
    with open(cache_path) as f:
        assert f.read() == FAKE_PY_CODE, "Cache file content must match py_code"
    assert compiled_fn is not None


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_load_restores_compiled_function(compiler_with_cache):
    """load() must restore a callable _CompiledFxGraph from the cache file.

    Full roundtrip: compile() writes FAKE_PY_CODE to the cache file; load()
    reads it, calls _CompiledFxGraph.load_artifacts(), which exec()s the code
    and wraps the resulting `kernel` symbol in a _CompiledFxGraph instance.
    """
    from torchair.npu_fx_compiler import _CompiledFxGraph

    model = SimpleModel().to("npu").to(DTYPE)
    graph, fake_inputs = _capture_graph(model, _make_inputs())
    compile_range = Range(BATCH, BATCH)

    # Step 1: compile with mocked get_code → writes FAKE_PY_CODE to cache.
    with patch.object(_CompiledFxGraph, "get_code", return_value=FAKE_PY_CODE):
        _, handle = compiler_with_cache.compile(
            graph=graph,
            example_inputs=fake_inputs,
            compiler_config={},
            compile_range=compile_range,
            key="load_roundtrip",
        )

    assert handle is not None
    _key, cache_path = handle
    assert os.path.isfile(cache_path)

    # Step 2: load() reads the cache file and returns a callable.
    loaded_fn = compiler_with_cache.load(
        handle=handle,
        graph=graph,
        example_inputs=fake_inputs,
        graph_index=0,
        compile_range=compile_range,
    )

    assert loaded_fn is not None
    assert callable(loaded_fn)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_model_structure_change_triggers_recompile(compiler_with_cache):
    """Different model keys must produce separate, independently loadable cache files.

    In vllm's real flow the key encodes the graph hash; a structural change
    produces a different key and forces a fresh compile() call that writes a
    new cache file.  This test simulates that by using two distinct keys.
    """
    from torchair.npu_fx_compiler import _CompiledFxGraph

    compile_range = Range(BATCH, BATCH)

    model_a = SimpleModel().to("npu").to(DTYPE)
    graph_a, inputs_a = _capture_graph(model_a, _make_inputs())

    model_b = DeeperModel().to("npu").to(DTYPE)
    graph_b, inputs_b = _capture_graph(model_b, _make_inputs())

    with patch.object(_CompiledFxGraph, "get_code", return_value=FAKE_PY_CODE):
        _, handle_a = compiler_with_cache.compile(
            graph=graph_a,
            example_inputs=inputs_a,
            compiler_config={},
            compile_range=compile_range,
            key="model_a",
        )
        _, handle_b = compiler_with_cache.compile(
            graph=graph_b,
            example_inputs=inputs_b,
            compiler_config={},
            compile_range=compile_range,
            key="model_b",
        )

    assert handle_a is not None, "model_a should have been cached"
    assert handle_b is not None, "model_b should have been cached"

    _, path_a = handle_a
    _, path_b = handle_b
    assert path_a != path_b, "Different keys must use separate cache files"
    assert os.path.isfile(path_a)
    assert os.path.isfile(path_b)

    # Both caches must be independently loadable.
    loaded_a = compiler_with_cache.load(handle_a, graph_a, inputs_a, 0, compile_range)
    loaded_b = compiler_with_cache.load(handle_b, graph_b, inputs_b, 0, compile_range)
    assert callable(loaded_a)
    assert callable(loaded_b)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_disable_cache_returns_none_handle(compiler_no_cache):
    """When disable_cache=True compile() returns None handle regardless of key.

    cache_dir is set to None inside compile() when disable_cache is True, so
    npugraph_ex_compile never writes a cache file and always returns handle=None.
    """
    from torchair.npu_fx_compiler import _CompiledFxGraph

    model = SimpleModel().to("npu").to(DTYPE)
    graph, fake_inputs = _capture_graph(model, _make_inputs())

    with patch.object(_CompiledFxGraph, "get_code", return_value=FAKE_PY_CODE):
        compiled_fn, handle = compiler_no_cache.compile(
            graph=graph,
            example_inputs=fake_inputs,
            compiler_config={},
            compile_range=Range(BATCH, BATCH),
            key="disabled_key",
        )

    assert compiled_fn is not None, "compile() must still return a callable"
    assert handle is None, "handle must be None when cache is disabled"
