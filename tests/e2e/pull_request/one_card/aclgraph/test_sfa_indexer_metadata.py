# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)
from vllm_ascend.device.device_op import A5DeviceAdaptor, DeviceOperator
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.device_metadata import TargetDeviceMetadata

# QliV2Metadata has int32 task arrays [36][8] and [72][8]. The binding
# allocates 1024 ints; its remaining tail is not part of the defined payload.
DEFINED_METADATA_INTS = (36 + 72) * 8


@pytest.mark.skipif(DeviceOperator is not A5DeviceAdaptor, reason="Owner-local FP8 QLI requires A5")
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("num_reqs", [4, 8])
def test_deferred_qli_metadata_refresh_and_cross_stream_consumption(full_graph, num_reqs):
    torch.npu.set_device(0)
    # The worker's physical-device lookup lazily loads the native extension,
    # independently of the generic fused-custom-op flag on A5.
    assert NPUPlatform.visible_device_id_to_physical_device_id(0) >= 0
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device="npu")
    local_lengths = torch.full((num_reqs,), 128, dtype=torch.int32, device="npu")
    empty = torch.empty(0, device="npu")
    builder = AscendSFAIndexerMetadataBuilder.__new__(AscendSFAIndexerMetadataBuilder)
    builder.use_sharded_dcp = True
    builder._device_metadata_enabled = False
    builder._device_metadata_tasks = ()
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_n_heads=32, index_head_dim=128, index_topk=2048)
        )
    )
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]
    target = TargetDeviceMetadata()
    layers = []
    for layer_index in (3, 7, 11):
        layer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        torch.nn.Module.__init__(layer)
        layer.n_head, layer.head_dim, layer.topk_tokens = 32, 128, 2048
        layer.k_cache = SimpleNamespace(prefix=f"layer{layer_index}.indexer.k_cache")
        layers.append(layer)

    def build(**kwargs):
        metadata = AscendSFAIndexerMetadata(
            num_actual_tokens=num_reqs,
            slot_mapping=empty,
            seq_lens=local_lengths,
            cum_query_lens=query_start_loc[1:],
            block_table=empty,
            sin=empty,
            cos=empty,
            query_start_loc=query_start_loc,
            actual_seq_lengths_key=local_lengths,
        )
        assert builder._device_metadata_enabled
        builder._defer_local_qli_metadata(metadata)
        return {layer.k_cache.prefix: metadata for layer in layers}

    def prepare():
        result = target.run_build(build, attn_groups=groups, full_graph_mode=full_graph)
        return result, ForwardContext(no_compile_layers={}, attn_metadata=result, slot_mapping={})

    outputs = [torch.empty(DEFINED_METADATA_INTS, dtype=torch.int32, device="npu") for _ in layers]
    reference = torch.empty_like(outputs[0])

    def consume(metadata):
        for layer, output in zip(layers, outputs):
            plan = layer._get_local_qli_inputs(metadata[layer.k_cache.prefix])[2]
            output.copy_(plan[:DEFINED_METADATA_INTS])

    def build_reference():
        reference.copy_(
            DeviceOperator.build_indexer_local_metadata(
                query_start_loc, local_lengths, num_heads_q=32, head_dim=128, topk=2048
            )[:DEFINED_METADATA_INTS]
        )

    main = torch.npu.Stream()
    main.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph() if full_graph else None
    snapshots = []
    with torch.npu.stream(main), torch.inference_mode():
        metadata, context = prepare()
        with override_forward_context(context):
            target.begin_forward()
            consume(metadata)
            target.finish()
            build_reference()
        main.synchronize()
        if full_graph:
            metadata, context = prepare()
            with override_forward_context(context), torch.npu.graph(graph, stream=main):
                target.begin_forward()
                consume(metadata)
                target.finish()
                build_reference()

        for step in range(8):
            # Reuse input addresses with different values, including empty
            # local histories and graph padding (one query has zero tokens).
            query_start_loc.copy_(torch.arange(num_reqs + 1, dtype=torch.int32))
            query_start_loc[1] = step % 2
            local_lengths.copy_(torch.arange(num_reqs, dtype=torch.int32) * 128 + step * 257)
            local_lengths[0] = 0
            metadata, context = prepare()
            if full_graph:
                graph.replay()
                target.finish_replay()
            else:
                with override_forward_context(context):
                    consume(metadata)
                    target.finish()
                    build_reference()
            snapshots.append(([output.clone() for output in outputs], reference.clone()))
    main.synchronize()
    for actuals, expected in snapshots:
        for actual in actuals:
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert not target.executor.submission_in_flight
    assert len(target.executor._stage_ready) == 1
