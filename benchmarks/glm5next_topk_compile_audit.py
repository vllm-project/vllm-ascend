# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count actual Triton compilation separately from NPU graph capture/replay.

Run in a fresh process with a new --cache-dir. This audits the operator's
Triton specializations, not full-model torch.compile or unrelated CANN kernels.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton


def _capture_graph(fn, graph):
    with torch.npu.graph(graph):
        return fn()


@torch.inference_mode()
def main():
    """Audit cold kernels, device-value changes and revisited shape buckets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.cache_dir.exists() and any(options.cache_dir.iterdir()):
        parser.error("use an empty cache directory to distinguish cold compilation from disk cache hits")
    options.cache_dir.mkdir(parents=True, exist_ok=True)
    previous_cache, previous_listener = triton.knobs.cache.dir, triton.knobs.compilation.listener
    triton.knobs.cache.dir = str(options.cache_dir.resolve())
    torch.npu.set_device(0)
    capacity, pool, selected, head_dim = 262208, 4, 512, 128
    pages, block = (capacity + 15) // 16, 16
    logical = torch.arange(pages * block)
    keys = torch.zeros(pages * block, head_dim, dtype=torch.bfloat16)
    keys[:, 0], keys[:, 1], keys[:, 2] = logical // 16384, logical // 128 % 128, logical % 128
    cache = keys.view(pages, block, 1, head_dim).npu()
    events, phases = [], []
    current_phase = "setup"

    def listener(*, src, metadata, cache_hit, **kwargs):
        event = dict(
            phase=current_phase,
            kernel=src.name,
            cache_hit=cache_hit,
            hash=metadata.get("hash"),
            constexprs=str(getattr(src, "constexprs", {})),
        )
        events.append(event)
        print(json.dumps(dict(event="triton_compilation", **event)), flush=True)

    def phase(name, fn, *, captures=0, replays=0):
        nonlocal current_phase
        current_phase = name
        start = len(events)
        result = fn()
        torch.npu.synchronize()
        observed = events[start:]
        record = dict(
            phase=name,
            actual_compilations=sum(not event["cache_hit"] for event in observed),
            disk_cache_hits=sum(event["cache_hit"] for event in observed),
            graph_captures=captures,
            graph_replays=replays,
        )
        phases.append(record)
        print(json.dumps(record), flush=True)
        return result

    def assert_result(result, tokens, visible):
        expected = torch.full((tokens, selected * pool + pool - 1), -1, dtype=torch.int32)
        count = min(selected, visible)
        history = (torch.arange(visible - count, visible)[:, None] * pool + torch.arange(pool)).flatten()
        expected[:, : history.numel()] = history.to(torch.int32)
        if visible == 0:
            expected[:, selected * pool :] = torch.arange(pool - 1)
        torch.testing.assert_close(result.cpu()[:, 0].sort(1).values, expected.sort(1).values, rtol=0, atol=0)

    triton.knobs.compilation.listener = listener
    try:
        for iteration, tokens in enumerate((1, 60, 1, 60)):
            query = torch.zeros(tokens, 32, head_dim, dtype=torch.bfloat16, device="npu")
            query[:, 0, :3] = torch.tensor([16384, 128, 1], dtype=torch.bfloat16, device="npu")
            weights = torch.zeros(tokens, 32, dtype=torch.bfloat16, device="npu")
            weights[:, 0] = 1
            ends = torch.arange(1, tokens + 1, dtype=torch.int32, device="npu")
            lengths = torch.full((tokens,), 1024, dtype=torch.int32, device="npu")
            positions = torch.full((tokens,), 4095, dtype=torch.int64, device="npu")
            table = torch.arange(pages, dtype=torch.int32, device="npu").repeat(tokens, 1)

            run = partial(
                glm5_next_lightning_indexer_triton,
                query,
                cache,
                weights,
                ends,
                lengths,
                table,
                positions,
                index_topk=selected * pool,
                index_kpool=pool,
                max_pool_seq_len=capacity,
                allow_cache_packing=False,
                use_streaming_topk=True,
            )

            prefix = f"shape_{iteration}_tokens_{tokens}"
            result = phase(prefix + "_warmup", run)
            assert_result(result, tokens, 1024)
            for visible in (32768, 262144, 0, 1024):
                lengths.fill_(visible)
                positions.fill_(visible * pool - 1 if visible else 2)
                result = phase(prefix + f"_eager_visible_{visible}", run)
                assert_result(result, tokens, visible)
                assert phases[-1]["actual_compilations"] == 0
            graph = torch.npu.NPUGraph()

            result = phase(prefix + "_capture", partial(_capture_graph, run, graph), captures=1)
            assert phases[-1]["actual_compilations"] == 0
            for visible in (1024, 32768, 262144, 0, 1024):
                lengths.fill_(visible)
                positions.fill_(visible * pool - 1 if visible else 2)
                phase(prefix + f"_replay_visible_{visible}", graph.replay, replays=1)
                assert_result(result, tokens, visible)
                assert phases[-1]["actual_compilations"] == 0
            del graph, result
        report = dict(
            capacity_pools=capacity,
            selected_pools=selected,
            sequence_tokens=[1, 60, 1, 60],
            phases=phases,
            compiler_events=events,
            correctness=True,
            scope="Triton operator compilation; NPU captures/replays counted separately",
        )
        options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        triton.knobs.compilation.listener = previous_listener
        triton.knobs.cache.dir = previous_cache


if __name__ == "__main__":
    main()
