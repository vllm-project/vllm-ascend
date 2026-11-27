# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-NPU graph dataflow coverage; the collective is simulated, not HCCL."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.router import fused_topk_router as router_module


def test_routing_ids_refresh_across_separate_graphs(monkeypatch):
    bucket_sizes = (3, 7)
    remote_ids = {size: torch.arange(size, dtype=torch.int32, device="npu") + 100 for size in bucket_sizes}
    gather = MagicMock(side_effect=lambda ids: torch.cat((ids, remote_ids[ids.numel()])))
    monkeypatch.setattr(
        router_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(all_gather_input_ids=gather)),
        ),
    )
    producer = router_module.AscendFusedTopKRouter(top_k=2, global_num_experts=4)
    consumer = router_module.AscendFusedTopKRouter(top_k=2, global_num_experts=4)
    buffer = torch.empty(2 * max(bucket_sizes), dtype=torch.int64, device="npu")
    producer.routing_input_ids_buffer = buffer
    producer.is_routing_input_ids_producer = True
    consumer.routing_input_ids_buffer = buffer

    captures = []
    for size in bucket_sizes:
        input_ids = torch.arange(size, dtype=torch.int32, device="npu")
        num_rows = 2 * size
        for _ in range(3):
            producer._prepare_routing_input_ids(input_ids, num_rows)
            calls_after_producer = gather.call_count
            consumer._prepare_routing_input_ids(input_ids, num_rows).clone()
            assert gather.call_count == calls_after_producer
        torch.npu.synchronize()

        producer_graph = torch.npu.NPUGraph()
        calls_before_capture = gather.call_count
        with torch.npu.graph(producer_graph):
            producer._prepare_routing_input_ids(input_ids, num_rows)
        assert gather.call_count == calls_before_capture + 1

        consumer_graph = torch.npu.NPUGraph()
        with torch.npu.graph(consumer_graph):
            # A device-side consumer is required: returning only a view would
            # not exercise the address recorded by the consumer graph.
            output = consumer._prepare_routing_input_ids(input_ids, num_rows).clone()
        assert gather.call_count == calls_before_capture + 1
        captures.append((input_ids, producer_graph, consumer_graph, output))

    # Alternate graph buckets after both are captured. Also write the shared
    # buffer from an eager producer while retaining the captured consumer.
    for step, (bucket, eager_producer) in enumerate(((0, False), (1, False), (0, True), (1, True), (0, False))):
        input_ids, producer_graph, consumer_graph, output = captures[bucket]
        size = input_ids.numel()
        local_cpu = torch.arange(size, dtype=torch.int32) + 1000 * (step + 1)
        remote_cpu = torch.arange(size, dtype=torch.int32) + 2000 * (step + 1)
        local_cpu[0] = -1
        local_cpu[1] = 0
        remote_cpu[-1] = -1
        input_ids.copy_(local_cpu)
        remote_ids[size].copy_(remote_cpu)
        buffer.fill_(-999)
        calls_before_replay = gather.call_count
        if eager_producer:
            producer._prepare_routing_input_ids(input_ids, 2 * size)
        else:
            producer_graph.replay()
        consumer_graph.replay()
        torch.npu.synchronize()

        expected = torch.cat((local_cpu, remote_cpu))
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        # Python callbacks execute during eager/capture, never graph replay.
        # Each eager producer calls gather once; consumers never call it.
        assert gather.call_count == calls_before_replay + int(eager_producer)
