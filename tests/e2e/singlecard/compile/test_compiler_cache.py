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
The save path relies on AclConcreteGraph.optimize_graph_without_runtime being
called by torchair during compilation.  In the installed torchair version this
hook is not triggered for simple test graphs, so the *save* path is exercised
via a mock of _capture_acl_codegen.

_capture_acl_codegen is mocked with the actual dynamo-captured FX graph so that
the cache file represents the real model graph.  The *load* path is then tested
as a full roundtrip: compile() writes the cache, load() reads it, and the output
of the loaded function is compared against direct model execution.

What is validated
-----------------
1. compile() save path: when the AclConcreteGraph hook fires (simulated via
   mock), the cache file is written and a non-None handle is returned.
2. load() roundtrip: compile() writes the cache with the real graph; load()
   restores it and produces output matching direct model execution.
3. Model-structure isolation: two distinct graph keys produce separate cache
   files; each is independently loadable.
4. disable_cache suppression: compile() returns None handle when cache is off.
"""

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.fx as fx
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
# Mock for _capture_acl_codegen
#
# The real context manager patches optimize_graph_without_runtime to populate
# `captured`, then torch.save's captured[0] after the with-block.
# Our mock pre-populates `captured` with a real GraphModule and performs the
# same torch.save, so npugraph_ex_compile sees a non-empty captured list and
# sets handle = (key, cache_path).
# ---------------------------------------------------------------------------


def _make_mock_capture_acl_codegen(fx_graph: fx.GraphModule):
    """Return a mock _capture_acl_codegen that injects fx_graph into captured."""

    @contextmanager
    def _mock(cache_path: str):
        captured = [fx_graph]  # Simulate optimize_graph_without_runtime firing
        yield captured
        # Replicate the real post-yield save logic
        if captured and cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(captured[0], cache_path)

    return _mock


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
    """compile() must write a .acl_fx.pt file and return a non-None handle.

    _capture_acl_codegen is mocked to simulate AclConcreteGraph firing the
    optimize_graph_without_runtime hook, which is how the save path triggers
    in a real vllm inference run.  The actual dynamo-captured graph is injected
    as the mock's fx_graph so the saved file contains the real model graph.
    """
    model = SimpleModel().to("npu").to(DTYPE)
    graph, fake_inputs = _capture_graph(model, _make_inputs())

    mock_ctx = _make_mock_capture_acl_codegen(graph)

    with patch("vllm_ascend.compilation.compiler_interface._capture_acl_codegen", mock_ctx):
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
    assert cache_path.endswith(".acl_fx.pt"), "Cache file must use .acl_fx.pt extension"
    assert os.path.isfile(cache_path), f"Cache file not found at {cache_path}"
    assert os.path.getsize(cache_path) > 0, "Cache file must not be empty"
    assert compiled_fn is not None


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_load_restores_compiled_function(compiler_with_cache):
    """load() must restore a callable whose output matches direct model execution.

    Full roundtrip: compile() writes the cache (with mocked _capture_acl_codegen
    that injects the real dynamo-captured graph), then load() restores it.
    The loaded function's output is compared against model(x) to verify correctness.
    """
    model = SimpleModel().to("npu").to(DTYPE)
    real_inputs = _make_inputs()
    graph, fake_inputs = _capture_graph(model, real_inputs)

    compile_range = Range(BATCH, BATCH)
    mock_ctx = _make_mock_capture_acl_codegen(graph)

    # Step 1: compile with mock → writes the real dynamo graph to the cache file.
    with patch("vllm_ascend.compilation.compiler_interface._capture_acl_codegen", mock_ctx):
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

    # Step 2: load() reads the cache file written by compile().
    loaded_fn = compiler_with_cache.load(
        handle=handle,
        graph=graph,
        example_inputs=fake_inputs,
        graph_index=0,
        compile_range=compile_range,
    )

    assert loaded_fn is not None
    assert callable(loaded_fn)

    # Step 3: output must match direct model execution.
    # Dynamo lifts model parameters as the first graph inputs (before activations).
    # Construct the full input list: [lifted_params..., activations...].
    n_params = len(fake_inputs) - len(real_inputs)
    full_graph_inputs = list(model.parameters())[:n_params] + real_inputs

    with torch.no_grad():
        expected = model(real_inputs[0])
        raw = loaded_fn(*full_graph_inputs)
    # make_graph_return_tuple may wrap the graph output in a tuple; unwrap if needed.
    actual = raw[0] if isinstance(raw, (list, tuple)) else raw
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_model_structure_change_triggers_recompile(compiler_with_cache):
    """Different model structures (different keys) must produce separate cache files.

    In vllm's real flow, the key is derived from the graph hash: changing the
    model structure changes the hash and therefore the key, forcing a fresh
    compile call that writes a new cache file.  This test simulates that by
    using two distinct keys and verifies independent caching and loading.
    """
    compile_range = Range(BATCH, BATCH)

    # --- model A ---
    model_a = SimpleModel().to("npu").to(DTYPE)
    graph_a, inputs_a = _capture_graph(model_a, _make_inputs())

    with patch(
        "vllm_ascend.compilation.compiler_interface._capture_acl_codegen",
        _make_mock_capture_acl_codegen(graph_a),
    ):
        _, handle_a = compiler_with_cache.compile(
            graph=graph_a,
            example_inputs=inputs_a,
            compiler_config={},
            compile_range=compile_range,
            key="model_a",
        )

    # --- model B (different structure → different key) ---
    model_b = DeeperModel().to("npu").to(DTYPE)
    graph_b, inputs_b = _capture_graph(model_b, _make_inputs())

    with patch(
        "vllm_ascend.compilation.compiler_interface._capture_acl_codegen",
        _make_mock_capture_acl_codegen(graph_b),
    ):
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
    assert path_a != path_b, "Different models must use separate cache files"
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

    cache_dir is None so _capture_acl_codegen is never reached; the else-branch
    in npugraph_ex_compile goes through real torchair compilation and returns
    handle=None unconditionally.
    """
    model = SimpleModel().to("npu").to(DTYPE)
    graph, fake_inputs = _capture_graph(model, _make_inputs())

    compiled_fn, handle = compiler_no_cache.compile(
        graph=graph,
        example_inputs=fake_inputs,
        compiler_config={},
        compile_range=Range(BATCH, BATCH),
        key="disabled_key",
    )

    assert compiled_fn is not None, "compile() must still return a callable"
    assert handle is None, "handle must be None when cache is disabled"
