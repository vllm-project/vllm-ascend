# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.ut.models.test_engram_elastic_executor import executor_env as executor_env
from vllm_ascend.models.deepseek_v41.engram import ops
from vllm_ascend.models.deepseek_v41.engram.layer import AscendEngram


def make_layer(layer_id, executor):
    layer = AscendEngram.__new__(AscendEngram)
    torch.nn.Module.__init__(layer)
    layer.layer_id, layer.layer_hash_index = layer_id, (1, 14).index(layer_id)
    layer.layer_name = f"elastic-test.layers.{layer_id}"
    layer.dim, layer.hc_mult, layer.eps, layer.rotation = 4, 2, 1e-6, None
    layer.executor = executor
    layer.wkv = torch.nn.Linear(2, 12, bias=False, dtype=torch.bfloat16)
    layer.wkv.weight.data.fill_(1 / 16)
    layer.q_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
    layer.k_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
    layer.start_prefetch = lambda ids: ids.clone()
    layer.finish_prefetch = lambda handle: handle.bfloat16()
    return layer


@pytest.fixture
def native_ops(executor_env, monkeypatch):
    executor, calls = executor_env
    layers = {layer_id: make_layer(layer_id, executor) for layer_id in (1, 14)}
    executor.engram_layers = layers

    def hash_inputs(ids, *metadata, **kwargs):
        return ids[:, None, None].expand(-1, 2, 2).int().clone(), ids >= 0

    executor.hash_fn = SimpleNamespace(forward_packed=hash_inputs)
    context = SimpleNamespace(
        cudagraph_runtime_mode=ops.CUDAGraphMode.NONE,
        no_compile_layers={layer.layer_name: layer for layer in layers.values()},
    )
    monkeypatch.setattr(ops, "get_forward_context", lambda: context)
    return executor, layers, context, calls


def run_ops(ids, layers):
    query = torch.tensor([0, ids.numel()], dtype=torch.int32)
    history = torch.zeros((1, 2), dtype=torch.int32)
    hashes, mask = torch.ops.vllm.engram_elastic_begin(
        ids,
        torch.arange(ids.numel()),
        query,
        history,
        torch.zeros_like(ids),
        history,
        torch.tensor([ids.numel()]),
        ids < 0,
        history.bool(),
        layers[1].layer_name,
        2,
        2,
    )
    hidden = ids[:, None, None].bfloat16().expand(-1, 2, 4).clone()
    for layer in layers.values():
        kv = torch.ops.vllm.engram_elastic_project(hashes, hidden, layer.layer_name, 2, 4)
        hidden = torch.ops.vllm.engram_elastic_consume(hidden, *kv, mask, layer.layer_name)
    return hidden


def register_cpu_ops():
    # Production registers PrivateUse1 kernels. CPU UTs run the very same
    # opaque bodies so real fake/meta and dispatcher contracts are exercised.
    library = torch.library.Library("vllm", "IMPL", "CPU")
    for name in ("engram_elastic_begin", "engram_elastic_project", "engram_elastic_consume"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(f"vllm::{name}", "CPU"):
            library.impl(name, getattr(ops, name))
    return library


def test_registered_ops_match_serial_gate_and_preserve_fake_layout(native_ops):
    executor, layers, _, _ = native_ops
    library = register_cpu_ops()
    for ids in (torch.tensor([1, -1, 3]), torch.tensor([7, 0, -1])):
        actual = run_ops(ids, layers)
        expected = ids[:, None, None].bfloat16().expand(-1, 2, 4).clone()
        rows = ids[:, None].expand(-1, 2).bfloat16()
        for layer in layers.values():
            expected = layer.apply_precomputed(expected, *layer.precompute_kv(rows), ids >= 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        executor.release()
    source = torch.zeros((3, 2, 2), dtype=torch.int32)
    real = layers[1].precompute_kv(torch.ones((3, 2), dtype=torch.bfloat16))
    fake = ops.engram_elastic_project_fake(source, source, "unused", 2, 4)
    assert [(t.shape, t.dtype, t.stride()) for t in real] == [(t.shape, t.dtype, t.stride()) for t in fake]
    del library


def test_piecewise_is_rejected_before_fetch(native_ops):
    executor, layers, context, calls = native_ops
    context.cudagraph_runtime_mode = ops.CUDAGraphMode.PIECEWISE
    before = len(calls)
    library = register_cpu_ops()
    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        run_ops(torch.tensor([1, 2]), layers)
    assert len(calls) == before and not executor._pending_fetches
    del library


def test_opaque_ops_compile_as_one_full_graph(native_ops):
    executor, layers, _, _ = native_ops
    library = register_cpu_ops()
    compiled = torch.compile(lambda ids: run_ops(ids, layers), backend="eager", fullgraph=True)
    for ids in (torch.tensor([2, -1, 4]), torch.tensor([8, 3, -1])):
        expected = run_ops(ids, layers)
        actual = compiled(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        executor.release()
    del library
