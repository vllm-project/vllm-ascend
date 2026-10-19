# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare runtime NUM_SLOTS against the previous constexpr specialization.

Run with pytest -s to retain the JSON timing records, or --junitxml to collect
properties. No absolute latency threshold is imposed across different NPUs.
"""

import ast
import importlib.util
import inspect
import json
import statistics

import pytest
import torch

from tests.e2e.pull_request.one_card.test_minimax_qknorm_rope_cache import make_compile_case
from vllm_ascend.ops.triton.msa import minimax_qknorm_rope_cache as fused_module

WARMUP_ITERATIONS = 5
CAPTURE_ITERATIONS = 20
REPLAY_ITERATIONS = 10
TIMING_SAMPLES = 5


@pytest.fixture(scope="module")
def constexpr_reference(tmp_path_factory):
    # Generate the previous specialization from exactly the current kernel body,
    # avoiding a second hand-maintained copy of the production implementation.
    tree = ast.parse(inspect.getsource(fused_module._minimax_qknorm_rope_cache_kernel.fn))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    fn.decorator_list = [ast.parse("triton.jit", mode="eval").body]
    argument = next(arg for arg in fn.args.args if arg.arg == "NUM_SLOTS")
    argument.annotation = ast.parse("tl.constexpr", mode="eval").body
    path = tmp_path_factory.mktemp("minimax_constexpr") / "reference.py"
    path.write_text(
        "from vllm.triton_utils import tl, triton\n"
        "from vllm_ascend.ops.triton.triton_utils import extract_slice, insert_slice\n"
        "from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import _norm_rope, FP8_E4M3_MAX\n"
        + ast.unparse(tree),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("minimax_constexpr_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._minimax_qknorm_rope_cache_kernel


def captured_call(fn):
    for _ in range(WARMUP_ITERATIONS):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(CAPTURE_ITERATIONS):
            fn()
    graph.replay()
    torch.npu.synchronize()
    return graph


def replay_latency_us(graph):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAY_ITERATIONS):
        graph.replay()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000 / (CAPTURE_ITERATIONS * REPLAY_ITERATIONS)


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 64, 128, 192, 256, 512])
@pytest.mark.parametrize("q_heads", [8, 16])
@torch.inference_mode()
def test_num_slots_specialization_performance(tokens, q_heads, constexpr_reference, monkeypatch, record_property):
    # Same tensors, strides and kernel body for both variants; only the bound's
    # specialization differs. Compilation is excluded from steady-state timing.
    args, kwargs = make_compile_case(tokens=tokens, actual=tokens, blocks=5392, q_heads=q_heads)
    runtime_kernel = fused_module._minimax_qknorm_rope_cache_kernel
    assert "NUM_SLOTS" in runtime_kernel.do_not_specialize
    assert next(p for p in runtime_kernel.params if p.name == "NUM_SLOTS").is_constexpr is False
    assert next(p for p in constexpr_reference.params if p.name == "NUM_SLOTS").is_constexpr is True
    graphs, outputs = {}, {}
    for name, kernel in (("constexpr", constexpr_reference), ("runtime", runtime_kernel)):
        with monkeypatch.context() as context:
            context.setattr(fused_module, "_minimax_qknorm_rope_cache_kernel", kernel)
            fn = lambda: fused_module.minimax_qknorm_rope_cache(*args, **kwargs)
            q, iq = fn()
            outputs[name] = (q.clone(), iq.clone(), args[7].clone(), args[8].clone(), args[9].clone())
            graphs[name] = captured_call(fn)
    for actual, expected in zip(outputs["runtime"], outputs["constexpr"]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    # Alternate order to reduce systematic clock/thermal drift between variants.
    for sample in range(TIMING_SAMPLES):
        order = ("constexpr", "runtime") if sample % 2 == 0 else ("runtime", "constexpr")
        for name in order:
            samples[name].append(replay_latency_us(graphs[name]))
    result = {"tokens": tokens, "q_heads": q_heads, "cache_blocks": 5392}
    result.update({f"{name}_us": statistics.median(values) for name, values in samples.items()})
    result["change_percent"] = (result["runtime_us"] / result["constexpr_us"] - 1) * 100
    result["samples_us"] = samples
    for name, value in result.items():
        record_property(name, json.dumps(value))
    print(json.dumps(result), flush=True)
