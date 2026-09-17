# SPDX-License-Identifier: Apache-2.0
"""A5 TP8/DCP8 FlashMLA stream A/B, using synthetic K3-shaped BF16 weights.

Run after sourcing the service environment:
  torchrun --standalone --nproc-per-node=8 benchmarks/flash_mla_dcp_overlap.py

This measures the real FlashMLA forward composition, not model throughput.
It excludes model loading, MoE, quantized projections and the output TP reduction.
"""

import argparse
import gc
import inspect
import json
import os
import statistics
import textwrap
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.distributed.parallel_state import init_distributed_environment, init_model_parallel_group

from vllm_ascend.utils import bootstrap_custom_op_env


def make_case(group, batch, query_len, history_len):
    from vllm_ascend.attention.attention_v1 import AscendFlashAttentionMetadata, _flash_attention_schedule
    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl

    device = torch.device(f"npu:{group.rank_in_group}")
    dtype = torch.bfloat16
    active_batch = batch
    batch = max(batch, (8 + query_len - 1) // query_len)
    tokens, heads, hidden, qrank, block = batch * query_len, 12, 7168, 1536, 128

    def weight(shape):
        return (torch.randn(shape, device=device) * 0.015).to(dtype)

    torch.manual_seed(16362)
    w_a = weight((qrank + 576, hidden))
    x = weight((tokens, hidden))
    torch.manual_seed(16362 + group.rank_in_group)
    w_q = weight((heads * 192, qrank))
    w_o = weight((hidden, heads * 128))
    w_g = weight((heads * 128, hidden))
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.num_heads, impl.num_kv_heads, impl.kv_lora_rank = heads, 1, 512
    impl.qk_head_dim, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 192, 128, 64
    impl.v_head_dim, impl.q_lora_rank = 128, qrank
    impl.dcp_group, impl.dcp_size, impl.dcp_rank = group, group.world_size, group.rank_in_group
    q_gamma = torch.ones(qrank, device=device, dtype=dtype)
    kv_gamma = torch.ones(512, device=device, dtype=dtype)
    impl.fused_qkv_a_proj = lambda x: (F.linear(x, w_a),)
    impl.q_a_layernorm = lambda x: torch_npu.npu_rms_norm(x, q_gamma, 1e-5)[0]
    impl.kv_a_layernorm = lambda x: torch_npu.npu_rms_norm(x, kv_gamma, 1e-5)[0]
    impl.q_proj = lambda x: (F.linear(x, w_q),)
    impl.W_UK_T, impl.W_UV = weight((heads, 128, 512)), weight((heads, 512, 128))
    impl.o_proj = lambda x, **kwargs: (F.linear(x, w_o),)
    impl.g_proj = lambda x: (F.linear(x, w_g),)
    impl.use_mla_rope, impl.use_output_gate, impl.scale = False, True, 192**-0.5

    history_pages = (history_len + block - 1) // block
    current_pages = (query_len + block - 1) // block
    pages_per_request = history_pages + current_pages
    # The extra layer slot makes the kernel page axis noncontiguous.
    backing = weight((batch * pages_per_request, 2, block, 576))
    cache = backing[:, 0]
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * query_len
    used = torch.full((batch,), query_len, dtype=torch.int32, device=device)
    lengths = torch.full((batch,), history_len, dtype=torch.int32, device=device)
    used[active_batch:].zero_()
    lengths[active_batch:].zero_()
    table = (
        torch.arange(batch, dtype=torch.int32, device=device)[:, None] * pages_per_request
        + torch.arange(history_pages, dtype=torch.int32, device=device)[None, :]
    )
    current_table = torch.arange(batch * current_pages, dtype=torch.int32, device=device).reshape(batch, current_pages)
    request = torch.arange(batch, dtype=torch.int64, device=device).repeat_interleave(query_len)
    offset = torch.arange(query_len, dtype=torch.int64, device=device).repeat(batch)
    slots = request * pages_per_request * block + history_len + offset
    if group.rank_in_group != 0:
        slots.fill_(-1)
    slots[active_batch * query_len :].fill_(-1)
    token_live = torch.arange(tokens, device=device) < active_batch * query_len
    current_slots = request * current_pages * block + offset
    current_slots.masked_fill_(~token_live, -1)
    flash = AscendFlashAttentionMetadata(
        query=torch.empty((tokens, heads * group.world_size, 576), dtype=dtype, device=device),
        schedule=None,
        cu=cu,
        used_q=used,
        cache_lens=lengths,
        block_table=table,
        slots=slots,
        live_boundaries=torch.empty(0, dtype=torch.int32, device=device),
        token_live=token_live,
        positions=torch.arange(tokens, dtype=torch.int64, device=device),
        attn_mask=torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1),
        max_query_len=query_len,
        max_seq_len=history_len,
        is_prefill=query_len > 4,
        dcp_size=group.world_size,
        split_kv=True,
        current_cache=torch.empty((batch * current_pages, 1, block, 576), dtype=dtype, device=device),
        current_block_table=current_table,
        current_slots=current_slots,
    )
    builder = SimpleNamespace(flash_num_heads=heads, flash_num_kv_heads=1)
    flash.schedule = _flash_attention_schedule(builder, flash, is_mla=True)
    flash.current_schedule = _flash_attention_schedule(builder, flash, is_mla=True, current=True)
    meta = SimpleNamespace(flash=flash, causal=True)
    output = torch.empty_like(x)

    def forward():
        return impl._forward_flash("dcp_overlap_benchmark", x, cache, meta, output)

    def check_tail():
        larger = torch.full((tokens + 3, hidden), float("nan"), device=device, dtype=dtype)
        impl._forward_flash("dcp_overlap_benchmark", x, cache, meta, larger)
        torch.testing.assert_close(larger[:tokens], output, atol=1e-6, rtol=1e-2)
        assert not torch.count_nonzero(larger[tokens:]).item()
        assert not torch.count_nonzero(larger[:tokens][~token_live]).item()

    return impl, forward, output, check_tail


def graph_time(graph, group, iterations, trials, layers):
    timings = []
    for _ in range(trials):
        dist.barrier(group=group.device_group)
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        ms = torch.tensor(start.elapsed_time(end) / (iterations * layers), dtype=torch.float32, device="npu")
        dist.all_reduce(ms, op=dist.ReduceOp.MAX, group=group.device_group)
        timings.append(ms.item())
    return timings


def original_projection(self, x):
    x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
    return torch.bmm(x, self.W_UV).transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)


def original_writeback_forward(method):
    """Recreate the pre-change output writes for the measured baseline only."""
    current = "    return flash_attention_output(result, b.token_live, output)"
    original = """    output.zero_()
    output[:t].copy_(result)
    output[:t].masked_fill_(~b.token_live.unsqueeze(1), 0)
    return output"""
    source = textwrap.dedent(inspect.getsource(method))
    assert current in source, "The baseline must be updated when the forward output contract changes."
    namespace = dict(method.__globals__)
    exec(compile(source.replace(current, original), "<baseline-writeback>", "exec"), namespace)
    return namespace[method.__name__]


def check_output(output, reference):
    # Raw-bit LSE preserves more information than the original BF16 pack.
    # Report rounding differences explicitly instead of claiming bit identity.
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-2)
    difference = (output.float() - reference.float()).abs()
    return dict(exact=torch.equal(output, reference), max_abs_error=difference.max().item())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="flash-mla-dcp-overlap.json")
    parser.add_argument("--cases", default="1x4,4x4")
    parser.add_argument("--history-per-rank", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--layers-per-graph", type=int, default=8)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    assert size == 8, "Use one A5 node with TP8/DCP8."
    torch.npu.set_device(rank)
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.ops.triton import sfa_cp

    raw_guard = sfa_cp.can_use_raw_dcp_exchange

    init_distributed_environment(size, rank, "env://", rank, "hccl", timeout=timedelta(minutes=5))
    group = init_model_parallel_group([list(range(size))], rank, "hccl", group_name="dcp")
    records = []
    for case in args.cases.split(","):
        batch, query_len = map(int, case.split("x"))
        impl, forward, output, check_tail = make_case(group, batch, query_len, args.history_per_rank)
        fused_projection = impl._v_up_proj_batch_major
        current_forward = impl._forward_flash
        baseline_forward = MethodType(original_writeback_forward(current_forward), impl)

        def configure(
            writeback,
            projection,
            overlap,
            query_prep,
            raw,
            impl=impl,
            fused_projection=fused_projection,
            current_forward=current_forward,
            baseline_forward=baseline_forward,
        ):
            impl._forward_flash = current_forward if writeback else baseline_forward
            impl._v_up_proj_batch_major = fused_projection if projection else MethodType(original_projection, impl)
            impl.flash_dcp_overlap = impl.flash_dcp_preprocess_overlap = overlap
            impl.flash_dcp_query_prep = query_prep
            sfa_cp.can_use_raw_dcp_exchange = raw_guard if raw else lambda *args, **kwargs: False

        configure(False, False, False, False, False)
        for _ in range(3):
            forward()
        torch.npu.synchronize()
        reference = output.clone()
        modes = []
        variants = [
            ("serial", False, False, False, False, False),
            ("writeback", True, False, False, False, False),
            ("projection", True, True, False, False, False),
            ("overlap", True, True, True, False, False),
            ("query_layout", True, True, True, True, False),
            ("raw_exchange", True, True, True, True, True),
        ]
        if args.reverse:
            variants.reverse()
        for name, writeback, projection, overlap, query_prep, raw in variants:
            configure(writeback, projection, overlap, query_prep, raw)
            for _ in range(3):
                forward()
            torch.npu.synchronize()
            check_output(output, reference)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(args.layers_per_graph):
                    forward()
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            check_output(output, reference)
            timings = graph_time(graph, group, args.iterations, args.trials, args.layers_per_graph)
            error = check_output(output, reference)
            error_stats = torch.tensor(
                [float(not error["exact"]), error["max_abs_error"]], dtype=torch.float32, device="npu"
            )
            dist.all_reduce(error_stats, op=dist.ReduceOp.MAX, group=group.device_group)
            error = dict(exact=not bool(error_stats[0].item()), max_abs_error=error_stats[1].item())
            modes.append(dict(mode=name, max_rank_ms=timings, median_ms=statistics.median(timings), **error))
            if rank == 0:
                print(json.dumps(dict(case=case, **modes[-1])), flush=True)
            del graph
        configure(True, True, True, True, True)
        forward()
        check_tail()
        medians = {row["mode"]: row["median_ms"] for row in modes}
        record = dict(
            case=case,
            active_tokens=batch * query_len,
            captured_tokens=max(8, batch * query_len),
            layers_per_graph=args.layers_per_graph,
            history_per_rank=args.history_per_rank,
            modes=modes,
        )
        record["total_speedup"] = medians["serial"] / medians["raw_exchange"]
        records.append(record)
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(scope=__doc__, cases=records), indent=2))
        del impl, forward, output, reference, check_tail
        gc.collect()
        torch.npu.empty_cache()
    dist.barrier(group=group.device_group)
    sfa_cp.can_use_raw_dcp_exchange = raw_guard
    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
