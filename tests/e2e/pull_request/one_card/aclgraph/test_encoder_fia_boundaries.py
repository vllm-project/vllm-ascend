# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Device regression for one captured FIA task, not a model/HTTP benchmark."""

import math
from itertools import accumulate

import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendMetadata, FIAParamProvider
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph, register_task

SEQ_LENGTHS = (20,) * 100 + (61,) * 8 + (62,) * 92
NUM_HEADS = 12
HEAD_SIZE = 64


def test_encoder_fia_replays_changed_boundaries_in_the_same_8192_graph():
    torch.npu.set_device(0)
    ends_a = list(accumulate(SEQ_LENGTHS))
    ends_b = list(accumulate(reversed(SEQ_LENGTHS)))
    assert len(SEQ_LENGTHS) == 200 and ends_a[-1] == ends_b[-1] == 8192
    assert all(a != b for a, b in zip(ends_a[:-1], ends_b[:-1], strict=True))
    offsets = [0, *ends_a]
    generator = torch.Generator().manual_seed(731)
    original = [
        torch.randn((8192, NUM_HEADS, HEAD_SIZE), generator=generator).to(device="npu", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    query, key, value = [tensor.clone() for tensor in original]
    output = torch.empty_like(query)
    lse = torch.empty(1, dtype=query.dtype, device=query.device)
    operation = torch_npu.npu_fused_infer_attention_score.out
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        atten_mask=None,
        block_table=None,
        input_layout="TND",
        block_size=128,
        actual_seq_lengths=ends_a,
        actual_seq_lengths_kv=ends_a,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        scale=1 / math.sqrt(HEAD_SIZE),
        sparse_mode=0,
    )
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
    kwargs.update(workspace=workspace, out=[output, lse])
    provider = FIAParamProvider("encoder", None, is_non_pa=True, is_encoder_only=True)
    capture_stream = torch.npu.Stream()
    update_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    graph = UpdatableGraph()
    snapshots = []
    with torch.npu.stream(capture_stream):
        operation(**kwargs)
        capture_stream.synchronize()
        with torch.npu.graph(graph, stream=capture_stream):
            register_task(operation, kwargs, provider)
        assert len(graph.tasks) == 1

        try:
            for order in (list(range(200)), list(reversed(range(200))), list(range(200))):
                lengths = [SEQ_LENGTHS[index] for index in order]
                ends = list(accumulate(lengths))
                for target, source in zip((query, key, value), original, strict=True):
                    target.copy_(torch.cat([source[offsets[index] : offsets[index + 1]] for index in order]))
                metadata = AscendMetadata(
                    model_runner_type="pooling",
                    causal=False,
                    num_actual_tokens=8192,
                    seq_lens_list=lengths,
                    actual_seq_lengths_q=ends,
                )
                tasks = graph.resolve_tasks(ContextSource({"encoder": metadata}))
                assert len(tasks) == 1
                assert tasks[0].kwargs["actual_seq_lengths"] == ends
                assert tasks[0].kwargs["actual_seq_lengths_kv"] == ends

                eager_output = torch.empty_like(output)
                eager_lse = torch.empty_like(lse)
                operation(
                    **dict(kwargs, actual_seq_lengths=ends, actual_seq_lengths_kv=ends, out=[eager_output, eager_lse])
                )
                # The eager reference and update task share scratch workspace.
                capture_stream.synchronize()
                output.fill_(float("nan"))
                # Match the non-ENPU production order: replay waits for the update event.
                graph.replay()
                graph.update(update_stream, tasks)
                capture_stream.synchronize()
                actual = output.cpu()
                torch.testing.assert_close(actual, eager_output.cpu(), rtol=0, atol=0)

                restored = torch.empty_like(actual)
                start = 0
                for index, length in zip(order, lengths, strict=True):
                    restored[offsets[index] : offsets[index + 1]].copy_(actual[start : start + length])
                    start += length
                snapshots.append(restored)
        finally:
            graph.reset()

    assert len(snapshots) == 3
    torch.testing.assert_close(snapshots[0], snapshots[1], rtol=0, atol=0)
    torch.testing.assert_close(snapshots[0], snapshots[2], rtol=0, atol=0)
