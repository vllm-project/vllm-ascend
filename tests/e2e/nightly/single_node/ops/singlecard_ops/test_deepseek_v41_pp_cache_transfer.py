# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.cache_config import uses_a5_packed_cache
from vllm_ascend.utils import bootstrap_custom_op_env
from vllm_ascend.worker.v2.pp_transport import gather_pp_cache_rows as gather_rows

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: E402, F401

PLANES = [
    (torch.bfloat16, 512, 0),
    (torch.int8, 128, 131072),
    (torch.float16, 1, 147456),
]


def _cache(dtype, width, offset_bytes, rows=128):
    if uses_a5_packed_cache():
        pytest.skip("These are A3 cache-plane strides")
    blocks, page_bytes = 32, 147712
    backing = torch.full((blocks * page_bytes,), 0x35, dtype=torch.uint8)
    shape = (blocks, rows, width)
    strides = (page_bytes // dtype.itemsize, width, 1)
    cpu_cache = torch.as_strided(backing.view(dtype), shape, strides, offset_bytes // dtype.itemsize)
    cpu_cache.copy_(torch.arange(blocks * rows * width).reshape(shape).remainder(97).sub(48).to(dtype))
    device_backing = backing.npu()
    cache = torch.as_strided(device_backing.view(dtype), shape, strides, offset_bytes // dtype.itemsize)
    return backing, cpu_cache, device_backing, cache


def _reference(cache, slots):
    expected = torch.zeros((slots.numel(), cache.shape[-1]), dtype=cache.dtype)
    valid = slots >= 0
    expected[valid] = cache[slots[valid] // 128, slots[valid] % 128]
    return expected


@pytest.mark.parametrize("dtype,width,offset_bytes", PLANES, ids=["long-kv", "index-k", "index-scale"])
@pytest.mark.parametrize("tokens", [0, 3, 257, 1331])
@torch.inference_mode()
def test_v41_pp_gather_keeps_plane_bits_and_does_not_modify_paged_cache(dtype, width, offset_bytes, tokens):
    backing, cpu_cache, device_backing, cache = _cache(dtype, width, offset_bytes)
    # A nonzero slot-buffer offset also occurs for graph-padded batches.
    slots_buffer = (torch.arange(tokens + 1, dtype=torch.int64) * 3 + 37).remainder(32 * 128)
    slots = slots_buffer[1:]
    slots[::37] = -1
    output = torch.full((tokens, width), 23, dtype=dtype, device="npu")
    gather_rows(cache, slots_buffer.npu()[1:], output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), _reference(cpu_cache, slots), rtol=0, atol=0)
    torch.testing.assert_close(device_backing.cpu(), backing, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,width,offset_bytes", PLANES, ids=["long-kv", "index-k", "index-scale"])
@torch.inference_mode()
def test_v41_pp_gather_graph_replay_reads_new_slots(dtype, width, offset_bytes):
    _, cpu_cache, _, cache = _cache(dtype, width, offset_bytes)
    slots = torch.arange(24, dtype=torch.int64, device="npu")
    output = torch.empty((24, width), dtype=dtype, device="npu")
    gather_rows(cache, slots, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        gather_rows(cache, slots, output)
    address = output.data_ptr()
    for base in (37, 257, 3001):
        current = torch.arange(24, dtype=torch.int64) + base
        current[::7] = -1
        slots.copy_(current.npu())
        graph.replay()
        torch.npu.synchronize()
        assert output.data_ptr() == address
        torch.testing.assert_close(output.cpu(), _reference(cpu_cache, current), rtol=0, atol=0)


@pytest.mark.parametrize("ratio,source,boundary", [(2, 2, 3), (1, 20, 25)])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("pcp_size", [1, 2])
@torch.inference_mode()
def test_pp_dependency_transport_forwards_native_cache_and_selection_states(
    monkeypatch, ratio, source, boundary, graph_mode, pcp_size
):
    from types import SimpleNamespace

    from vllm.sequence import IntermediateTensors

    from vllm_ascend.models.deepseek_v41 import model as v41_model
    from vllm_ascend.models.deepseek_v41.model import V41PPBoundaryPlan
    from vllm_ascend.worker.v2 import pp_transport

    if uses_a5_packed_cache():
        pytest.skip("These are A3 cache-plane strides")
    tokens = 24
    cache_tokens = tokens * pcp_size
    live = V41PPBoundaryPlan(boundary, (source,), topk=True, candidates=ratio == 1)
    empty = V41PPBoundaryPlan(0)
    config = SimpleNamespace(
        compress_ratios=[ratio] * 40,
        head_dim=512,
        index_head_dim=128,
        candidate_source_layer_id=20,
    )
    context = SimpleNamespace(attn_metadata=None, no_compile_layers={})
    monkeypatch.setattr(v41_model, "get_forward_context", lambda: context)
    monkeypatch.setattr(pp_transport, "get_forward_context", lambda: context)
    monkeypatch.setattr(pp_transport, "is_forward_context_available", lambda: True)
    stages = []
    for rank in range(3):
        runtime = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        incoming, outgoing = (empty if rank == 0 else live), (empty if rank == 2 else live)
        handler = v41_model.V41PPCacheHandler(runtime, config, "model", incoming, outgoing)
        layers = runtime.compilation_config.static_forward_context
        long_prefix, index_prefix = handler.cache_prefixes(source)
        planes = [_cache(*plane, rows=128 // ratio)[3] for plane in PLANES]
        if rank:
            layers[long_prefix].bind_kv_cache(planes[0])
            layers[index_prefix].bind_kv_cache(tuple(planes[1:]))
            for plane in planes:
                plane.fill_(-7)
        else:
            layers[long_prefix] = SimpleNamespace(kv_cache=[planes[0]])
            layers[index_prefix] = SimpleNamespace(kv_cache=[tuple(planes[1:])])
        states = {
            "pp_test_topk": torch.zeros((tokens, 512), dtype=torch.int32, device="npu"),
            "pp_test_topk_lengths": torch.zeros((tokens, 1), dtype=torch.int32, device="npu"),
        }
        if live.candidates:
            states["pp_test_candidates"] = torch.zeros((tokens, 1, 8), dtype=torch.int32, device="npu")
            states["pp_test_candidate_lengths"] = torch.zeros((tokens, 1), dtype=torch.int32, device="npu")
        transport = pp_transport.PPDependencyTransport(
            runtime,
            f"transport_{rank}",
            handler,
            handler.wire_specs(incoming),
            handler.wire_specs(outgoing),
            states if rank else {},
            states if rank < 2 else {},
        )
        slots = torch.zeros(cache_tokens, dtype=torch.int64, device="npu")
        coordinates = torch.zeros((cache_tokens, 2), dtype=torch.int32, device="npu")
        metadata = SimpleNamespace(flat_slot_mapping=slots, slot_mapping=coordinates)
        if pcp_size > 1:
            metadata = SimpleNamespace(
                flat_slot_mapping=slots[::pcp_size],
                slot_mapping=coordinates[::pcp_size],
                global_metadata=metadata,
            )
        stages.append(
            SimpleNamespace(
                transport=transport,
                layers=layers,
                planes=planes,
                states=states,
                slots=slots,
                coordinates=coordinates,
                metadata={long_prefix: metadata, index_prefix: metadata},
            )
        )
    hidden = torch.zeros((tokens, 4), dtype=torch.bfloat16, device="npu")

    def forward():
        payload = IntermediateTensors({"hidden_states": hidden})
        for rank, stage in enumerate(stages):
            context.no_compile_layers, context.attn_metadata = stage.layers, stage.metadata
            if rank:
                stage.transport.receive(payload, tokens)
            payload = stage.transport.send(IntermediateTensors({"hidden_states": hidden}), tokens)

    def update(step):
        rows = 128 // ratio
        slots = (torch.arange(cache_tokens, dtype=torch.int64) + 37 + step * 131).remainder(32 * rows)
        # C2 publishes only completed pairs; padding is skipped for C1 too.
        slots[:: ratio + 1] = -1
        coordinates = torch.stack((slots // rows, slots % rows), dim=-1).int()
        coordinates[slots < 0] = -1
        for stage in stages:
            stage.slots.copy_(slots.npu())
            stage.coordinates.copy_(coordinates.npu())
        for tensor in stages[0].states.values():
            tensor.fill_(step + 19)
        return slots

    update(0)
    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph() if graph_mode else None
    if graph is not None:
        with torch.npu.graph(graph):
            forward()
    for step in (1, 2):
        slots = update(step)
        if graph is not None:
            graph.replay()
        else:
            forward()
        torch.npu.synchronize()
        valid = slots[slots >= 0]
        for original, mirror in zip(stages[0].planes, stages[-1].planes):
            torch.testing.assert_close(
                mirror.cpu()[valid // mirror.shape[1], valid % mirror.shape[1]],
                original.cpu()[valid // original.shape[1], valid % original.shape[1]],
                rtol=0,
                atol=0,
            )
        for name, tensor in stages[-1].states.items():
            torch.testing.assert_close(tensor.cpu(), stages[0].states[name].cpu(), rtol=0, atol=0)
