# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.forward_context import BatchDescriptor

from vllm_ascend.attention.indexer import QLIMetadataBuffers, QLIMetadataConfig
from vllm_ascend.device.device_op import A5DeviceAdaptor, DeviceOperator
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


@pytest.mark.parametrize("query_len", [1, 8])
@pytest.mark.parametrize("quant_mode", [1])
@pytest.mark.parametrize("full_graph", [False, True])
@torch.inference_mode()
def test_qli_metadata_rebuild_matches_inline_selection(quant_mode, full_graph, query_len):
    if DeviceOperator is not A5DeviceAdaptor:
        pytest.skip("requires A5 CANN V2 indexer")
    pytest.importorskip("cann_ops_transformer.ops")
    torch.npu.set_device(0)
    metadata_op, select_op = A5DeviceAdaptor._load_cann_quant_lightning_indexer_ops()
    config = QLIMetadataConfig(32, 128, quant_mode)
    buffers = QLIMetadataBuffers(config, 2)
    dtype = torch.float8_e4m3fn
    scale_dtype = torch.float32
    torch.manual_seed(7)
    num_layers = 4
    num_tokens = 2 * query_len
    query = [torch.randn(num_tokens, 32, 128, device="npu").mul_(4).to(dtype) for _ in range(num_layers)]
    key = torch.randn(64, 128, 1, 128, device="npu").mul_(4).to(dtype)
    weights = [torch.rand(num_tokens, 32, dtype=scale_dtype, device="npu") for _ in range(num_layers)]
    query_scale = torch.ones(num_tokens, 32, dtype=scale_dtype, device="npu")
    key_scale = torch.ones(64, 128, 1, dtype=scale_dtype, device="npu")
    block_table = torch.arange(64, dtype=torch.int32, device="npu").view(2, 32)
    queries = torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32, device="npu")
    lengths = torch.tensor([128, 256], dtype=torch.int32, device="npu")
    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph() if full_graph else None

    def select(inputs, layer):
        return select_op(
            query=query[layer],
            key=key,
            weights=weights[layer],
            query_dequant_scale=query_scale,
            key_dequant_scale=key_scale,
            topk=config.topk,
            quant_mode=quant_mode,
            cu_seqlens_q=inputs.query_start_loc,
            seqused_k=inputs.seq_lens,
            block_table=block_table,
            metadata=inputs.metadata,
            max_seqlen_q=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            return_value=0,
        )[0]

    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        captured = buffers.build("target", queries, lengths, metadata_op)
        for _ in range(2):
            for layer in range(num_layers):
                select(captured, layer)
    stream.synchronize()
    with torch.npu.stream(stream):
        if graph is not None:
            with torch.npu.graph(graph, stream=stream):
                graph_outputs = [select(captured, layer) for layer in range(num_layers)]
        for first, second in [(129, 257), (2049, 3073), (4096, 4096), (64, 96)]:
            lengths.copy_(torch.tensor([first, second], dtype=torch.int32))
            prepared = buffers.build("target", queries, lengths, metadata_op)
            assert prepared.metadata.data_ptr() == captured.metadata.data_ptr()
            if graph is not None:
                graph.replay()
                actual = [output.clone() for output in graph_outputs]
            else:
                actual = [select(prepared, layer) for layer in range(num_layers)]
            # The kernel updates execution state within metadata. Compare the
            # selections after several consumers, rather than mutable scratch
            # bytes, against a fresh producer for every reference layer.
            expected = []
            for layer in range(num_layers):
                fresh = QLIMetadataBuffers(config, 2).build("reference", queries, lengths, metadata_op)
                expected.append(select(fresh, layer))
            stream.synchronize()
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=0, atol=0)
