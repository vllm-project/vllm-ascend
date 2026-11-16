# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone A2 EP16 W4A8 prefill regression, requiring 16 idle NPUs.

torchrun --nproc-per-node=16 test_a2_deepseek_prefill.py --tokens 32

Compares the production TensorList/MegaMoE endpoint against independent
all-gather routing and ordinary stacked GMM. Tests changed routes, skew and
active masks. This is an operator test, not full PCP/model/PD qualification.
"""

import argparse
import importlib
import os
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_utils import load_cann_mega_moe_ops
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.w4a8 import w4a8
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, ASCEND_QUANTIZATION_METHOD, bootstrap_custom_op_env

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 5120, 2304, 384, 6
BF16_TOLERANCE = 2**-6
REQUIRED_MATCHED_RATIO = 0.99
MAX_ABS_ERROR = 1.0


def assert_bf16_close(actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), "Non-finite MoE output"
    if actual.numel() == 0:
        return
    error = (actual - expected).abs()
    ratio = (error <= BF16_TOLERANCE * (1 + expected.abs())).float().mean().item()
    maximum = error.max().item()
    assert ratio >= REQUIRED_MATCHED_RATIO and maximum <= MAX_ABS_ERROR, (ratio, maximum)


def make_layer(local_experts):
    layer = torch.nn.Module()
    for name, k, n in (("w13", HIDDEN, 2 * INTERMEDIATE), ("w2", INTERMEDIATE, HIDDEN)):
        # Native ModelSlim packing: output pairs share a byte. Build the MSD
        # compensation independently from the signed INT4 values.
        packed = torch.randint(-128, 128, (local_experts, k, n // 2), device="npu", dtype=torch.int8)
        low = (packed.to(torch.int16) & 15).to(torch.int8)
        high = packed >> 4
        low = torch.where(low >= 8, low - 16, low)
        unpacked = torch.stack((low, high), dim=-1).flatten(-2)
        scale = torch.full((local_experts, n, 1), 0.005, device="npu", dtype=torch.float32)
        bias = 8 * unpacked.float().sum(1) * scale.squeeze(-1)
        partitions = 16 if name == "w2" else 1
        tensors = {
            name + "_weight": packed.transpose(1, 2).contiguous(),
            name + "_weight_scale": scale,
            name + "_scale_bias": bias.unsqueeze(-1).expand(-1, -1, partitions).contiguous() / partitions,
        }
        for key, value in tensors.items():
            layer.register_parameter(key, torch.nn.Parameter(value, requires_grad=False))
    return layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("tokens must be positive")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 16, "Use the target P node's 16-rank EP group"
    torch.manual_seed(42 + rank)
    torch.npu.manual_seed_all(42 + rank)
    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    allocate, mega = load_cann_mega_moe_ops()

    # Exercise both production weight layouts from the same checkpoint payload.
    # The stacked layout remains the ordinary GMM numerical reference.
    layer = make_layer(EXPERTS // world)
    baseline = torch.nn.Module()
    for name, value in layer.named_parameters():
        baseline.register_parameter(name, torch.nn.Parameter(value.clone(), requires_grad=False))
    method = object.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.quant_method = ASCEND_QUANTIZATION_METHOD
    method.use_expert_weight_list = False
    with (
        patch.object(w4a8, "get_current_vllm_config", return_value=None),
        patch.object(w4a8, "use_cann_megamoe", return_value=False),
        patch.object(w4a8, "maybe_trans_nz", side_effect=lambda w: torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ)),
    ):
        method.process_weights_after_loading(baseline)
    method.use_expert_weight_list = True
    with (
        patch.object(w4a8, "get_current_vllm_config", return_value=None),
        patch.object(w4a8, "use_cann_megamoe", return_value=True),
        patch.object(w4a8, "maybe_trans_nz", side_effect=lambda w: torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ)),
    ):
        method.process_weights_after_loading(layer)
    with patch.object(w4a8, "_EXTRA_CTX", NS(use_mega_moe=True)):
        weights = method.get_fused_mc2_weights(layer)
    layer.mega_moe_activation_kwargs = {"activation_clamp": 10.0}
    group = dist.new_group(backend="hccl")
    buffer = allocate(
        group,
        EXPERTS,
        args.tokens,
        TOPK,
        hidden=HIDDEN,
        intermediate_hidden=INTERMEDIATE,
        max_recv_token_num=args.tokens * world * TOPK,
        dispatch_quant_mode=2,
        dispatch_quant_out_dtype=258,
    )
    comm = object.__new__(FusedMC2CommImpl)
    comm.mega_moe = mega
    comm.mega_moe_symm_buffer = buffer
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = args.tokens
    comm.token_dispatcher.global_bs = 0

    def gather(x):
        chunks = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(chunks, x)
        return torch.cat(chunks)

    def reference(x, ids, probs, active, *, use_lists=False):
        # Route gathered tokens independently of MegaMoE's communication and
        # scatter the local experts' weighted outputs before the EP reduction.
        x, ids, probs, active = map(gather, (x, ids, probs, active))
        start = rank * (EXPERTS // world)
        rows, routes = torch.where((ids >= start) & (ids < start + EXPERTS // world) & active[:, None])
        local_ids = ids[rows, routes] - start
        order = local_ids.argsort(stable=True)
        rows, routes, local_ids = rows[order], routes[order], local_ids[order]
        counts = torch.bincount(local_ids.long(), minlength=EXPERTS // world).to(torch.int64)
        output = torch.zeros_like(x, dtype=torch.float32)
        if use_lists:
            w1, s1, w2, s2 = method._get_mlp_weights(layer)
            b1, b2 = method._get_mlp_scale_bias(layer)
        else:
            w1, s1 = [baseline.w13_weight], [baseline.w13_weight_scale]
            w2, s2 = [baseline.w2_weight], [baseline.w2_weight_scale]
            b1, b2 = [baseline.w13_scale_bias], [baseline.w2_scale_bias]
        if rows.numel():
            expanded, scales = torch_npu.npu_dynamic_quant(x[rows])
            hidden, hidden_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
                x=expanded,
                weight=w1,
                weight_scale=s1,
                x_scale=scales,
                group_list=counts,
                weight_assist_matrix=b1,
                dequant_mode=0,
                group_list_type=1,
                swiglu_limit=10.0,
            )
            expert_output = torch_npu.npu_grouped_matmul(
                x=[hidden],
                weight=w2,
                scale=s2,
                bias=b2,
                per_token_scale=[hidden_scale],
                group_list=counts,
                group_list_type=1,
                split_item=2,
                group_type=0,
                output_dtype=torch.bfloat16,
            )[0]
            output.index_add_(0, rows, expert_output.float() * probs[rows, routes, None])
        dist.all_reduce(output)
        return output[rank * args.tokens : (rank + 1) * args.tokens].bfloat16(), counts

    for case in ("uniform", "skew", "padding", "idle_dp_peer", "empty"):
        x = torch.randn(args.tokens, HIDDEN, device="npu", dtype=torch.bfloat16)
        ids = torch.rand(args.tokens, EXPERTS, device="npu").topk(TOPK).indices.to(torch.int32)
        if case == "skew":
            ids[:] = torch.arange(TOPK, device="npu", dtype=torch.int32)
        probs = torch.softmax(torch.randn(args.tokens, TOPK, device="npu"), dim=-1)
        active = torch.ones(args.tokens, device="npu", dtype=torch.bool)
        if case == "padding":
            active[(rank % 3) :: 3] = False
        elif case == "idle_dp_peer" and rank >= world // 2 or case == "empty":
            active.zero_()
        inp = build_fused_experts_input(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=probs,
            quant_type=QuantType.W4A8,
            layer=layer,
            dynamic_eplb=False,
            mc2_mask=active,
        )
        actual, counts = comm._apply_cann_mega_moe(inp, weights, is_decode_only_node=False)
        expected, expected_counts = reference(x, ids, probs, active)
        fallback, fallback_counts = reference(x, ids, probs, active, use_lists=True)
        torch.testing.assert_close(counts.flatten().long(), expected_counts, rtol=0, atol=0)
        torch.testing.assert_close(fallback_counts, expected_counts, rtol=0, atol=0)
        assert_bf16_close(fallback[active], expected[active])
        assert_bf16_close(actual[active], expected[active])
        if rank == 0:
            print(f"PASS {case}: {args.tokens} tokens/rank, EP16 W4A8", flush=True)
    torch.npu.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
