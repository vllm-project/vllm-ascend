# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.forward_context import BatchDescriptor

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
