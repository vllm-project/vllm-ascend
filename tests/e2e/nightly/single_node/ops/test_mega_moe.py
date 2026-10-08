"""A5 MegaMoe weight/activation integration and latency comparison.

Requires eight idle A5 NPUs and a CANN package with SiTU MegaMoe support.
Source the CANN environment and set the HCCL/GLOO interface before running:

    torchrun --nproc_per_node=8 test_mega_moe.py --tokens 4 32 256 512
    torchrun --nproc_per_node=8 test_mega_moe.py --tokens 256 --graph
    torchrun --nproc_per_node=8 test_mega_moe.py --tokens 256 --max-recv-tokens 65536
    torchrun --nproc_per_node=8 test_mega_moe.py --tokens 2048 --input-scale 10

Use a fresh process for each graph shape to isolate CANN/HCCL graph lifetime.
Defaults retain 28 local experts and the routed FFN dimensions of an EP32
Kimi K3 deployment, scaled to EP8 for a single-node operator test. Latency is
reported, not asserted: eager and graph replay have different launch costs.
"""

import argparse
import importlib
import json
import os
import statistics
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_utils import load_cann_mega_moe_ops, select_mega_moe_activation_kwargs
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import bootstrap_custom_op_env

MC2_REFERENCE_TOKENS = 256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=int, default=224)
    p.add_argument("--hidden", type=int, default=3584)
    p.add_argument("--intermediate", type=int, default=3072)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--tokens", type=int, nargs="+", default=[4, 32, 256, 512])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--activation", choices=["situ", "silu"], default="situ")
    p.add_argument("--graph", action="store_true")
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--max-recv-tokens", type=int, default=0, help="0 uses CANN automatic receive capacity")
    a = p.parse_args()
    if a.max_recv_tokens < 0:
        p.error("--max-recv-tokens must be non-negative")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    group = dist.new_group(backend="hccl")
    mega_group = dist.new_group(backend="hccl")
    name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    mega_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    torch.manual_seed(20260928 + rank)
    torch.npu.manual_seed_all(20260928 + rank)
    # The standalone reference needs SiTU even when the platform disables the
    # generic custom-op capability on A5. Import after selecting the device.
    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    allocate, mega = load_cann_mega_moe_ops()
    act_kwargs = select_mega_moe_activation_kwargs(
        mega,
        activation=a.activation,
        activation_clamp=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    def make_weights(out_dim, in_dim):
        weights, scales = [], []
        for _ in range(a.experts // world):
            source = torch.randn((out_dim, in_dim), device="npu", dtype=torch.bfloat16) / in_dim**0.5
            w, s = torch_npu.npu_dynamic_mx_quant(source, dst_type=torch_npu.float4_e2m1fn_x2)
            weights.append(w.view(torch.uint8))
            scales.append(s.view(torch.uint8).reshape(out_dim, in_dim // 64, 2))
        return torch.stack(weights), torch.stack(scales).flatten(-2)

    def prepare_weights():
        # Exercise the production weight producer for both layouts. Only backend
        # selection is isolated from the global model configuration in this test.
        layer = torch.nn.Module()
        for name, dims in (("w13", (2 * a.intermediate, a.hidden)), ("w2", (a.hidden, a.intermediate))):
            w, scale = make_weights(*dims)
            layer.register_parameter(name + "_weight", torch.nn.Parameter(w, requires_grad=False))
            layer.register_parameter(name + "_weight_scale", torch.nn.Parameter(scale, requires_grad=False))
        baseline = torch.nn.Module()
        for name, value in layer.named_parameters():
            baseline.register_parameter(name, torch.nn.Parameter(value.clone(), requires_grad=False))
        method = object.__new__(w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod)
        with patch.object(w4a8_mxfp4, "get_current_vllm_config", return_value=None):
            with patch.object(w4a8_mxfp4, "use_cann_megamoe", return_value=False):
                method.process_weights_after_loading(baseline)
            with patch.object(w4a8_mxfp4, "use_cann_megamoe", return_value=True):
                method.process_weights_after_loading(layer)
        with patch.object(w4a8_mxfp4, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True)):
            payload = method.get_fused_mc2_weights(layer)
        return baseline, layer, payload

    baseline_layer, mega_layer, weights = prepare_weights()
    mega_layer.mega_moe_activation_kwargs = act_kwargs
    w1, s1 = baseline_layer.w13_weight, baseline_layer.w13_weight_scale
    w2, s2 = baseline_layer.w2_weight, baseline_layer.w2_weight_scale
    buffer = allocate(
        mega_group,
        a.experts,
        max(a.tokens),
        a.topk,
        a.hidden,
        2 * a.intermediate,
        max_recv_token_num=min(a.max_recv_tokens, max(a.tokens) * world * min(a.topk, a.experts // world)),
        dispatch_quant_mode=4,
        dispatch_quant_out_dtype=24,
    )
    # Instantiate only the integration endpoint: no model runner or checkpoint is
    # required. The unit tests separately cover FusedMC2CommImpl.__init__ binding.
    comm = object.__new__(FusedMC2CommImpl)
    comm.mega_moe = mega
    comm.mega_moe_symm_buffer = buffer
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = max(a.tokens)
    comm.token_dispatcher.global_bs = 0
    print(json.dumps({"rank": rank, "stage": "initialized", "config": vars(a)}), flush=True)

    def fused(x, ids, probs):
        inp = build_fused_experts_input(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=probs,
            quant_type=QuantType.W4A8MXFP,
            layer=mega_layer,
            mxfp_act_quant_type=torch.float8_e4m3fn,
            mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
            mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
            mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            mxfp_use_bf16=True,
            dynamic_eplb=False,
            activation=a.activation,
        )
        return comm._apply_cann_mega_moe(inp, weights, is_decode_only_node=False)

    def decomposed_chunk(x, ids, probs):
        expanded, scale, assist, counts, ep_counts, tp_counts, expanded_scales = (
            torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=ids,
                expert_shard_type=0,
                shared_expert_rank_num=0,
                moe_expert_num=a.experts,
                global_bs=0,
                expert_token_nums_type=1,
                scales=None,
                quant_mode=4,
                group_ep=name,
                ep_world_size=world,
                ep_rank_id=rank,
                comm_alg="",
                tp_world_size=1,
                tp_rank_id=0,
                y_dtype=torch.float8_e4m3fn,
                expert_scales=probs,
            )[:7]
        )
        if scale.ndim == 2:
            scale = scale.reshape(scale.shape[0], -1, 2)
        mm1 = torch_npu.npu_grouped_matmul(
            x=[expanded],
            weight=[w1],
            antiquant_scale=[s1],
            per_token_scale=[scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=counts,
            group_list_type=1,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        if a.activation == "situ":
            activated, act_scale = torch.ops._C_ascend.situ_mx_quant(
                x=mm1, beta=4.0, linear_beta=25.0, activate_left=True, dst_type=36
            )
        else:
            activated, act_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                mm1, topk_weight=None, group_index=counts, dst_type=torch.float8_e4m3fn, quant_mode=2, clamp_value=0.0
            )
        if act_scale.ndim == 2:
            act_scale = act_scale.reshape(act_scale.shape[0], -1, 2)
        mm2 = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[w2],
            antiquant_scale=[s2],
            per_token_scale=[act_scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=counts,
            group_list_type=1,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        y = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=mm2,
            expert_ids=ids,
            expert_scales=probs,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=a.experts,
            global_bs=0,
            ep_send_counts=ep_counts,
            group_ep=name,
            ep_world_size=world,
            ep_rank_id=rank,
            expand_scales=expanded_scales,
            comm_quant_mode=0,
            comm_alg="",
            assist_info_for_combine=assist,
        )
        return y, counts

    def decomposed(x, ids, probs):
        if x.shape[0] <= MC2_REFERENCE_TOKENS:
            return decomposed_chunk(x, ids, probs)
        # Use bounded MC2 batches for the numerical reference. Tokens are
        # independent, so chunking also covers a larger single MegaMoe call.
        # Do not use its latency as an AllToAll performance baseline.
        outputs, counts = [], []
        for start in range(0, x.shape[0], MC2_REFERENCE_TOKENS):
            end = start + MC2_REFERENCE_TOKENS
            output, count = decomposed_chunk(x[start:end], ids[start:end], probs[start:end])
            outputs.append(output)
            counts.append(count)
        return torch.cat(outputs), torch.stack(counts).sum(dim=0)

    captured_graphs = []

    def measure(fn, x, ids, probs):
        for _ in range(3):
            result = fn(x, ids, probs)
        torch.npu.synchronize()
        dist.barrier()
        if a.graph:
            expected = result[0].clone()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = fn(x, ids, probs)
            captured_graphs.append((graph, captured))
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
            fn = lambda *_: graph.replay()
            torch.npu.synchronize()
        samples = []
        for _ in range(a.repeats):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            fn(x, ids, probs)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        return statistics.median(samples)

    with torch.inference_mode():
        for tokens in a.tokens:
            x = torch.randn((tokens, a.hidden), device="npu", dtype=torch.bfloat16) * a.input_scale
            logits = torch.rand((tokens, a.experts), device="npu")
            ids = logits.topk(a.topk, dim=-1).indices.to(torch.int32)
            probs = torch.softmax(torch.randn((tokens, a.topk), device="npu"), dim=-1)
            reference, reference_counts = decomposed(x, ids, probs)
            output, counts = fused(x, ids, probs)
            torch.npu.synchronize()
            ref, out = reference.cpu().double(), output.cpu().double()
            relative_rmse = ((out - ref).square().mean() / ref.square().mean()).sqrt().item()
            cosine = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
            counts_equal = torch.equal(counts.cpu().to(torch.int64), reference_counts.cpu().to(torch.int64))
            assert torch.isfinite(out).all(), "nonfinite MegaMoe output"
            assert counts_equal, (counts.cpu(), reference_counts.cpu())
            assert relative_rmse < 0.03 and cosine > 0.999, (relative_rmse, cosine)
            baseline_us = measure(decomposed, x, ids, probs) if tokens <= MC2_REFERENCE_TOKENS else None
            fused_us = measure(fused, x, ids, probs)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "tokens": tokens,
                        "relative_rmse": relative_rmse,
                        "cosine": cosine,
                        "max_abs_error": (out - ref).abs().max().item(),
                        "counts_equal": counts_equal,
                        "decomposed_us": baseline_us,
                        "mega_us": fused_us,
                        "speedup": baseline_us / fused_us if baseline_us is not None else None,
                    }
                ),
                flush=True,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
