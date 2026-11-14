# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from time import perf_counter
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention import dsa_v1
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.triton_utils import (
    get_aicore_num,
    get_vectorcore_num,
    init_device_properties_triton,
)
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

enable_custom_op()

# SAS and QLI ABI: 36 FA/LI rows, then 72 FD/LD rows, eight int32s per row.
_METADATA_AIC_CAPACITY = 36
_METADATA_ROW_WIDTH = 8
_BENCHMARK_ITERATIONS = 100


def _assert_metadata_equal(actual, expected):
    # Disabled SAS rows and FD column 7 are not initialized by the kernel.
    # Compare enable flags and every field actually consumed by active cores.
    for offset, cores, fields in (
        (0, get_aicore_num(), _METADATA_ROW_WIDTH),
        (_METADATA_AIC_CAPACITY * _METADATA_ROW_WIDTH, get_vectorcore_num(), _METADATA_ROW_WIDTH - 1),
    ):
        rows = slice(offset, offset + cores * _METADATA_ROW_WIDTH)
        lhs, rhs = (value[rows].reshape(cores, _METADATA_ROW_WIDTH) for value in (actual, expected))
        torch.testing.assert_close(lhs[:, 0], rhs[:, 0], rtol=0, atol=0)
        active = rhs[:, 0] != 0
        torch.testing.assert_close(lhs[active, :fields], rhs[active, :fields], rtol=0, atol=0)


def _builder(kind, num_reqs):
    cls = dsa_v1.AscendDSAIndexerMetadataBuilder if kind == "qli" else dsa_v1.AscendDSAAttentionMetadataBuilder
    builder = cls.__new__(cls)
    builder._device_metadata_enabled = False
    builder._device_metadata_tasks = ()
    builder._sas_variant_buffers = {}
    builder.compressor_ratio = 4 if kind == "qli" else kind
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_attention_heads=64, index_topk=512, index_n_heads=64, index_head_dim=128),
        get_head_size=lambda: 512,
    )
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))
    builder.seqused_q = torch.empty(0, dtype=torch.int32, device="npu")
    builder.sas_metadata_buffer = torch.empty(dsa_v1.DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device="npu")
    builder.qli_metadata_buffer = torch.empty_like(builder.sas_metadata_buffer)
    builder.qli_seqused_k = torch.empty(num_reqs, dtype=torch.int32, device="npu")
    builder.qli_cmp_residual_k = torch.empty_like(builder.qli_seqused_k)
    return builder


@pytest.mark.parametrize("num_reqs", [1, 4, 16, 32])
@pytest.mark.parametrize("kind", [0, 4, 128, "qli"])
@torch.inference_mode()
def test_dsa_metadata_graph_dynamic_lengths(kind, num_reqs, monkeypatch, record_property):
    if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
        pytest.skip("requires the custom PA_ND SAS/QLI metadata backend")
    torch.npu.set_device(0)
    init_device_properties_triton()
    monkeypatch.setattr(dsa_v1, "get_tensor_model_parallel_world_size", lambda: 1)
    builder = _builder(kind, num_reqs)
    eager_builder = _builder(kind, num_reqs)
    query_offsets = torch.arange(num_reqs + 1, dtype=torch.int32, device="npu")
    lengths = torch.full((num_reqs,), 128, dtype=torch.int32, device="npu")
    executor = DeviceMetadataExecutor(capture_producers=True)
    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()

    def build(in_graph, max_query, max_kv):
        active_builder = builder if in_graph else eager_builder
        with active_builder.defer_device_metadata(in_graph=True) if in_graph else nullcontext():
            if kind == "qli":
                return active_builder._build_qli_metadata({}, query_offsets, lengths, max_query, max_kv)
            return active_builder._build_sas_metadata(
                {}, f"c{kind}", query_offsets, lengths, max_query, max_kv, None, None, 127, 0, False
            )

    metadata = build(True, 0, 0)
    tasks = builder.take_device_metadata_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    consumed = torch.empty_like(metadata)

    def forward():
        executor.submit(tasks)
        executor.wait(task.stage, task.group_id)
        consumed.copy_(metadata)
        executor.release()

    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        for _ in range(3):
            forward()
        stream.synchronize()
        with torch.npu.graph(graph, stream=stream):
            forward()
        snapshots = []
        # Cross compressed-length boundaries, grow/shrink, pad requests, then
        # reactivate an entirely empty DP rank. Shape/address stay unchanged.
        cases = [
            (1, 128, num_reqs),
            (6, 131071, num_reqs),
            (6, 131072, num_reqs),
            (1, 7, 1),
            (1, 0, 0),
            (6, 131073, num_reqs),
            (1, 128, num_reqs),
        ]
        for query_len, kv_len, active_reqs in cases:
            q_cpu = torch.tensor([min(i, active_reqs) * query_len for i in range(num_reqs + 1)], dtype=torch.int32)
            kv_cpu = torch.tensor([kv_len if i < active_reqs else 0 for i in range(num_reqs)], dtype=torch.int32)
            query_offsets.copy_(q_cpu)
            lengths.copy_(kv_cpu)
            expected = build(False, query_len, kv_len).clone()
            # The reference must not write the captured output/length buffers:
            # otherwise a graph that failed to run its producer could pass.
            metadata.fill_(-1)
            if kind == "qli":
                builder.qli_seqused_k.fill_(-1)
                builder.qli_cmp_residual_k.fill_(-1)
            graph.replay()
            qli_lengths = (builder.qli_seqused_k.clone(), builder.qli_cmp_residual_k.clone()) if kind == "qli" else None
            snapshots.append((consumed.clone(), expected, qli_lengths, kv_cpu))
        stream.synchronize()
        for actual, expected, qli_lengths, expected_lengths in snapshots:
            _assert_metadata_equal(actual.cpu(), expected.cpu())
            if qli_lengths is not None:
                torch.testing.assert_close(qli_lengths[0].cpu(), expected_lengths // 4, rtol=0, atol=0)
                torch.testing.assert_close(qli_lengths[1].cpu(), expected_lengths % 4, rtol=0, atol=0)

        # Diagnostic benchmark, without a hardware-dependent CI speed assertion.
        # Synchronization is outside host-submission timing for both paths.
        for name, submit in (("eager", lambda: build(False, 1, 128)), ("graph", graph.replay)):
            stream.synchronize()
            started = perf_counter()
            for _ in range(_BENCHMARK_ITERATIONS):
                submit()
            elapsed = perf_counter() - started
            stream.synchronize()
            record_property(f"{name}_submission_us", elapsed * 1e6 / _BENCHMARK_ITERATIONS)
