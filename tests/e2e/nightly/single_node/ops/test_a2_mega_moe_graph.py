# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2 native-I640 MegaMoE capture/replay regression, requiring eight idle 910B NPUs.

Run each shape/precision in a fresh process with a compatible CANN operator:
    torchrun --nproc-per-node=8 test_a2_mega_moe_graph.py --tokens 32 --precision bf16
    torchrun --nproc-per-node=8 test_a2_mega_moe_graph.py --tokens 32 --precision w8a8

Checks updated inputs, routes, padding and expert counts against eager calls.
This isolates graph integration; it does not establish end-to-end model throughput.
"""

import argparse
import json
import os
import statistics
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.a2_mega_moe import A2MegaMoEPrefillConfig
from vllm_ascend.ops.fused_moe import moe_comm_method, prepare_finalize
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_utils import _get_cann_mega_moe_quant_settings, load_cann_mega_moe_ops
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import scale_from_float_to_int64
from vllm_ascend.quantization.quant_type import QuantType

HIDDEN = 2560
INTERMEDIATE = 640
EXPERTS = 512
TOPK = 10
EP_SIZE = 8


def make_weights(precision):
    weights, scales = [], []
    for rows, cols in ((HIDDEN, 2 * INTERMEDIATE), (INTERMEDIATE, HIDDEN)):
        shape = (EXPERTS // EP_SIZE, rows, cols)
        if precision == "bf16":
            weight = (torch.randn(shape, device="npu") / rows**0.5).bfloat16()
            scale = None
        else:
            weight = torch.randint(-16, 17, shape, dtype=torch.int8, device="npu")
            weight = torch_npu.npu_format_cast(weight, 29)
            scale = scale_from_float_to_int64(torch.full((shape[0], cols), 0.01, device="npu"))
            scale = list(scale.unbind(0))
        weights.append(list(weight.unbind(0)))
        scales.append(scale)
    return MoEWeights(w1=weights[0], w2=weights[1], w1_scale=scales[0], w2_scale=scales[1])


def measure(fn, repeats):
    samples = []
    for _ in range(repeats):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    # Report the slowest rank's median, rather than rank zero alone.
    elapsed = torch.tensor(statistics.median(samples), device="npu")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def run(args, rank, group):
    allocate, mega = load_cann_mega_moe_ops()
    weights = make_weights(args.precision)
    quant = QuantType.NONE if args.precision == "bf16" else QuantType.W8A8
    quant_mode, quant_dtype, _ = _get_cann_mega_moe_quant_settings(quant)
    context = SimpleNamespace(mc2_mask=None, padded_num_tokens=args.tokens, in_profile_run=False)
    config = SimpleNamespace(a2_mega_moe_prefill=A2MegaMoEPrefillConfig(enabled=True))
    comm = object.__new__(moe_comm_method.FusedMC2CommImpl)
    comm.mega_moe = mega
    comm.mega_moe_activation_kwargs = {"activation": "swiglu"}
    comm.mega_moe_symm_buffer = allocate(
        group,
        EXPERTS,
        args.tokens,
        TOPK,
        hidden=HIDDEN,
        intermediate_hidden=INTERMEDIATE,
        max_recv_token_num=args.tokens * EP_SIZE * TOPK,
        dispatch_quant_mode=quant_mode,
        dispatch_quant_out_dtype=quant_dtype,
    )
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = args.tokens
    comm.token_dispatcher.global_bs = 0
    prepare = object.__new__(prepare_finalize.PrepareAndFinalizeWithMC2)
    prepare.tp_size, prepare.tp_rank = 1, 0
    layer = SimpleNamespace(a2_mega_moe_intermediate_size=INTERMEDIATE)
    x = torch.zeros(args.tokens, HIDDEN, dtype=torch.bfloat16, device="npu")
    ids = torch.arange(TOPK, dtype=torch.int32, device="npu").expand(args.tokens, -1).clone()
    probs = torch.full((args.tokens, TOPK), 1 / TOPK, device="npu")
    padding = torch.ones(args.tokens, dtype=torch.bool, device="npu")

    def fused():
        prepared = prepare.prepare(x, ids)
        inputs = build_fused_experts_input(
            hidden_states=prepared.hidden_states,
            topk_ids=prepared.router_logits,
            topk_weights=probs,
            quant_type=quant,
            dynamic_eplb=False,
            mc2_mask=prepared.mc2_mask,
            layer=layer,
        )
        return comm._apply_cann_mega_moe(inputs, weights, is_decode_only_node=False)

    # Mimic MRv2 capture: the mask is derived inside the graph from a persistent
    # input. Capture with all rows inactive; change those same tensors on replay.
    with (
        patch.object(moe_comm_method, "get_ascend_config", return_value=config),
        patch.object(moe_comm_method, "_EXTRA_CTX", context),
        patch.object(prepare_finalize, "_EXTRA_CTX", context),
        patch.object(prepare_finalize, "get_forward_context", return_value=SimpleNamespace(is_padding=padding)),
    ):
        for _ in range(3):
            fused()
        torch.npu.synchronize()
        dist.barrier()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = fused()
        for valid, skew in ((args.tokens, False), (1, True), (args.tokens - 1, False), (0, True), (args.tokens, True)):
            x.copy_(torch.randn_like(x))
            ids.copy_(torch.rand(args.tokens, EXPERTS, device="npu").topk(TOPK, dim=-1).indices.to(torch.int32))
            if skew:
                ids.copy_(torch.arange(TOPK, device="npu", dtype=torch.int32).expand_as(ids))
            probs.copy_(torch.softmax(torch.randn_like(probs), dim=-1))
            padding.fill_(True)
            padding[:valid] = False
            expected, expected_counts = fused()
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured[0][:valid], expected[:valid], rtol=0, atol=0)
            torch.testing.assert_close(captured[1], expected_counts, rtol=0, atol=0)
            counts = torch.bincount(ids[:valid].long().flatten(), minlength=EXPERTS)
            dist.all_reduce(counts)
            local = counts.chunk(EP_SIZE)[rank]
            torch.testing.assert_close(captured[1].flatten().long(), local, rtol=0, atol=0)
        eager_us = measure(fused, args.repeats)
        graph_us = measure(graph.replay, args.repeats)
    if rank == 0:
        print(json.dumps({**vars(args), "eager_us": eager_us, "graph_us": graph_us, "passed": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32, help="Padded rows per rank; run a fresh process per shape")
    parser.add_argument("--precision", choices=("bf16", "w8a8"), default="bf16")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 1:
        parser.error("tokens and repeats must be positive")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    if dist.get_world_size() != EP_SIZE or not torch.npu.get_device_name().startswith("Ascend910B"):
        raise RuntimeError("This regression requires exactly eight 910B devices")
    group = dist.new_group(backend="hccl")
    rank = dist.get_rank()
    torch.manual_seed(17 + rank)
    torch.npu.manual_seed_all(17 + rank)
    with torch.inference_mode():
        run(args, rank, group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
