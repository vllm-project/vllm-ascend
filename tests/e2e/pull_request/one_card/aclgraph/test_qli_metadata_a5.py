# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu

from vllm_ascend.attention.indexer import QLIMetadataBuffers, QLIMetadataConfig
from vllm_ascend.device.device_op import A5DeviceAdaptor, DeviceOperator


@pytest.mark.parametrize("query_len", [1, 8])
@pytest.mark.parametrize("quant_mode", [1, 5])
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
    torch.manual_seed(7)
    num_layers = 4
    num_tokens = 2 * query_len
    if quant_mode == 5:
        query, query_scale = [], []
        for _ in range(num_layers):
            packed, scale = torch_npu.npu_dynamic_mx_quant(
                torch.randn(num_tokens * 32, 128, dtype=torch.bfloat16, device="npu"),
                dst_type=torch_npu.float4_e2m1fn_x2,
            )
            query.append(packed.view(num_tokens, 32, 64))
            query_scale.append(scale.view(torch.float8_e8m0fnu).view(num_tokens, 32, 2, 2))
        key, key_scale = torch_npu.npu_dynamic_mx_quant(
            torch.randn(64 * 128, 128, dtype=torch.bfloat16, device="npu"),
            dst_type=torch_npu.float4_e2m1fn_x2,
        )
        key = key.view(64, 128, 1, 64)
        key_scale = key_scale.view(torch.float8_e8m0fnu).view(64, 128, 1, 2, 2)
    else:
        query = [
            torch.randn(num_tokens, 32, 128, device="npu").mul_(4).to(torch.float8_e4m3fn) for _ in range(num_layers)
        ]
        key = torch.randn(64, 128, 1, 128, device="npu").mul_(4).to(torch.float8_e4m3fn)
        query_scale = [torch.ones(num_tokens, 32, dtype=torch.float32, device="npu") for _ in range(num_layers)]
        key_scale = torch.ones(64, 128, 1, dtype=torch.float32, device="npu")
    # Both FP8 C8 and MXFP4 C4 require FP32 weights, independent of scale dtype.
    weights = [torch.rand(num_tokens, 32, dtype=torch.float32, device="npu") for _ in range(num_layers)]
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
            query_dequant_scale=query_scale[layer],
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
