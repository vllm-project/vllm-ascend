# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.forward_context import BatchDescriptor

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("consumer_order", [(0, 1), (1, 0)])
def test_metadata_order_and_buffer_reuse(full_graph, consumer_order):
    torch.npu.set_device(0)
    executor = DeviceMetadataExecutor()
    stream = torch.npu.Stream()
    source = torch.zeros(16, device="npu")
    prepared = torch.empty_like(source)
    metadata = [torch.empty_like(source) for _ in range(3)]
    outputs = [torch.empty_like(source) for _ in range(3)]
    calls = []
    snapshots = []
    descriptor = BatchDescriptor(num_tokens=16, num_reqs=16) if full_graph else None
    graph = torch.npu.NPUGraph() if full_graph else None

    def prepare():
        calls.append("prepare")
        prepared.copy_(source)

    def build(index):
        calls.append(index)
        torch.add(prepared, index + 1, out=metadata[index])

    tasks = (
        DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: build(2), 2),
        DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: build(0), 0),
        DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: build(1), 1),
        DeviceMetadataTask(DeviceMetadataStage.COMPRESSOR, prepare, 0),
    )

    def consume():
        for index in consumer_order:
            executor.wait(DeviceMetadataStage.ATTENTION, index)
            outputs[index].copy_(metadata[index])
        executor.wait(DeviceMetadataStage.INDEXER, 2)
        outputs[2].copy_(metadata[2])

    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        executor.submit(tasks, descriptor)
        assert calls == ["prepare", 0, 1, 2]
        # Capture without a recording warmup: the next submission must be able
        # to reorder producers while retaining the captured per-group events.
        if graph is not None:
            with torch.npu.graph(graph, stream=stream):
                consume()
        else:
            consume()
        executor.release()

        for step in range(20):
            calls.clear()
            source.fill_(step)
            executor.submit(tasks, descriptor)
            assert calls == ["prepare", *consumer_order, 2]
            if graph is not None:
                graph.replay()
            else:
                consume()
            snapshots.append([output.clone() for output in outputs])
            executor.release()

    torch.npu.synchronize()
    for step, snapshot in enumerate(snapshots):
        for index, actual in enumerate(snapshot):
            torch.testing.assert_close(actual.cpu(), torch.full((16,), float(step + index + 1)), rtol=0, atol=0)


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("query_len", [1, 6])
def test_a3_captured_smla_and_qli_metadata(full_graph, query_len):
    """Replay real A3 producers after an empty capture and changing KV lengths."""
    if not get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE):
        pytest.skip("Requires native INT8 sparse metadata operators")
    torch.npu.set_device(0)
    executor = DeviceMetadataExecutor(capture_producers=True)
    stream = torch.npu.Stream()
    query_start = torch.zeros(3, dtype=torch.int32, device="npu")
    seq_lens = torch.zeros(2, dtype=torch.int32, device="npu")
    cache_lens = torch.zeros_like(seq_lens)
    residual = torch.zeros_like(seq_lens)
    buffers = [torch.empty(1024, dtype=torch.int32, device="npu") for _ in range(2)]
    outputs = [torch.empty_like(buffer) for buffer in buffers]

    def smla_metadata():
        return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
            64,
            1,
            512,
            cu_seqlens_q=query_start,
            seqused_ori_kv=seq_lens,
            seqused_cmp_kv=cache_lens,
            cmp_residual_kv=residual,
            batch_size=2,
            max_seqlen_q=-1,
            max_seqlen_ori_kv=-1,
            max_seqlen_cmp_kv=-1,
            ori_topk=0,
            cmp_topk=512,
            cmp_ratio=2,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=True,
        )

    def qli_metadata():
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            64,
            1,
            128,
            512,
            2,
            cu_seqlens_q=query_start,
            seqused_k=cache_lens,
            cmp_residual_k=residual,
            batch_size=2,
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=2,
        )

    stages = (DeviceMetadataStage.ATTENTION, DeviceMetadataStage.INDEXER)
    tasks = (
        DeviceMetadataTask(stages[0], lambda: buffers[0].copy_(smla_metadata()), id(buffers[0])),
        DeviceMetadataTask(stages[1], lambda: buffers[1].copy_(qli_metadata()), id(buffers[1])),
    )

    def execute():
        executor.submit(tasks)
        for stage, buffer, output in zip(stages, buffers, outputs):
            executor.wait(stage, id(buffer))
            output.copy_(buffer)
        executor.release()

    addresses = [buffer.data_ptr() for buffer in buffers]
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        execute()
        stream.synchronize()
        graph = torch.npu.NPUGraph() if full_graph else None
        if graph is not None:
            with torch.npu.graph(graph, stream=stream):
                execute()
        for length in (32, 256, 2048, 64):
            query_start.copy_(torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32, device="npu"))
            seq_lens.fill_(length)
            cache_lens.fill_(length // 2)
            if graph is None:
                execute()
            else:
                graph.replay()
            expected = (smla_metadata(), qli_metadata())
            stream.synchronize()
            # The C++ ABI initializes 36 AIC rows and 72 AIV rows; the rest
            # of the 1024-word allocation is unspecified padding.
            for actual, reference, words in zip(outputs, expected, (36 * 9 + 72 * 8, 36 * 8 + 72 * 8)):
                torch.testing.assert_close(actual[:words].cpu(), reference[:words].cpu(), rtol=0, atol=0)
    assert [buffer.data_ptr() for buffer in buffers] == addresses
