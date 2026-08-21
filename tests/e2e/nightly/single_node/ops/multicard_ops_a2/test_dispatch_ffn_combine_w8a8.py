# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.quantization.methods.w8a8_dynamic import scale_from_float_to_int64
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _get_hcomm_name(rank: int) -> str:
    default_group = _get_default_group()
    backend = default_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def _expert_value(global_expert: torch.Tensor) -> torch.Tensor:
    # Keep the encoded GMM2 coefficient representable by signed INT8 while
    # still making every route except the wrap point directly observable.
    return global_expert.remainder(127) + 1


def _routes(tokens: int, top_k: int, global_experts: int, source_rank: int, generation: int) -> torch.Tensor:
    routes = torch.arange(tokens * top_k, dtype=torch.int32)
    routes = (routes * (2 * generation + 9) + source_rank * 23 + generation * 7) % global_experts
    return routes.reshape(tokens, top_k)


def _run_rank(rank: int, world_size: int, port: int) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )

    try:
        # EP * local_experts sits exactly on the old 128-entry alignment
        # boundary, so the masked-row sentinel must receive its own stride.
        local_experts = 64
        tokens = 32
        top_k = 4
        hidden_size = 256
        ffn_size = 256
        gate_up_size = 2 * ffn_size
        global_experts = world_size * local_experts

        torch_npu.npu.config.allow_internal_format = True

        x = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
        x[:, 0] = 1

        # Dynamic quantization maps the unit input to INT8 and its per-token
        # scale. GMM1 reconstructs unit gate/up values, and GMM2 writes an
        # expert-specific coefficient into output column zero.
        weight1 = torch.zeros((local_experts, hidden_size, gate_up_size), dtype=torch.int8)
        weight1[:, 0, 0] = 1
        weight1[:, 0, ffn_size] = 1
        weight2 = torch.zeros((local_experts, ffn_size, hidden_size), dtype=torch.int8)
        for local_expert in range(local_experts):
            global_expert = rank * local_experts + local_expert
            weight2[local_expert, 0, 0] = global_expert % 127 + 1

        weight1_nz = [torch_npu.npu_format_cast(weight1.npu(), 29)]
        weight2_nz = [torch_npu.npu_format_cast(weight2.npu(), 29)]
        scale1 = [scale_from_float_to_int64(torch.ones((local_experts, gate_up_size))).npu()]
        scale2 = [scale_from_float_to_int64(torch.ones((local_experts, hidden_size))).npu()]
        empty_bias = [torch.empty(0, dtype=torch.float32)]

        probs_cpu = torch.arange(1, top_k + 1, dtype=torch.float32).repeat(tokens, 1)
        probs_cpu /= probs_cpu.sum(dim=-1, keepdim=True)
        probs = probs_cpu.npu()
        x = x.npu()
        out = torch.empty_like(x)
        expert_token_nums = torch.zeros(local_experts, dtype=torch.int32).npu()
        silu_one = F.silu(torch.tensor(1, dtype=torch.bfloat16))

        for generation in range(1, 5):
            routes_by_rank = [
                _routes(tokens, top_k, global_experts, source_rank, generation) for source_rank in range(world_size)
            ]
            expert_idx = routes_by_rank[rank]
            expected = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
            expected[:, 0] = (_expert_value(expert_idx) * probs_cpu).sum(dim=-1).to(torch.bfloat16) * silu_one
            expected_counts = torch.bincount(
                torch.cat([routes.reshape(-1) for routes in routes_by_rank]),
                minlength=global_experts,
            ).to(torch.int32)
            expected_counts = expected_counts[rank * local_experts : (rank + 1) * local_experts]

            out.fill_(torch.nan)
            expert_token_nums.fill_(-1)
            torch.ops._C_ascend.dispatch_ffn_combine(
                x=x,
                weight1=weight1_nz,
                weight2=weight2_nz,
                expert_idx=expert_idx.npu(),
                scale1=scale1,
                scale2=scale2,
                bias1=empty_bias,
                bias2=empty_bias,
                probs=probs,
                group=_get_hcomm_name(rank),
                max_output_size=512,
                out=out,
                expert_token_nums=expert_token_nums,
            )
            torch_npu.npu.synchronize()

            torch.testing.assert_close(out.cpu(), expected, rtol=0.04, atol=0.04)
            torch.testing.assert_close(expert_token_nums.cpu(), expected_counts)

        # Publish a real all-zero count row, then reuse the same HCCL windows
        # again. A stale plain-zero row must not satisfy the tagged ready
        # predicate for the following wave.
        x_active_mask = torch.zeros(tokens, dtype=torch.bool)
        expert_idx = _routes(tokens, top_k, global_experts, rank, generation=5)
        out.fill_(torch.nan)
        expert_token_nums.fill_(-1)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz,
            weight2=weight2_nz,
            expert_idx=expert_idx.npu(),
            scale1=scale1,
            scale2=scale2,
            bias1=empty_bias,
            bias2=empty_bias,
            probs=probs,
            group=_get_hcomm_name(rank),
            max_output_size=512,
            x_active_mask=x_active_mask.npu(),
            out=out,
            expert_token_nums=expert_token_nums,
        )
        torch_npu.npu.synchronize()
        torch.testing.assert_close(expert_token_nums.cpu(), torch.zeros(local_experts, dtype=torch.int32))

        # Reuse the same HCCL windows with only one real token in a graph-sized
        # buffer. Inactive rows must not contribute expert work.
        active_tokens = 1
        x_active_mask = torch.zeros(tokens, dtype=torch.bool)
        x_active_mask[:active_tokens] = True
        routes_by_rank = [
            _routes(tokens, top_k, global_experts, source_rank, generation=6) for source_rank in range(world_size)
        ]
        expert_idx = routes_by_rank[rank]
        expected = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
        expected[:active_tokens, 0] = (_expert_value(expert_idx[:active_tokens]) * probs_cpu[:active_tokens]).sum(
            dim=-1
        ).to(torch.bfloat16) * silu_one
        expected_counts = torch.bincount(
            torch.cat([routes[:active_tokens].reshape(-1) for routes in routes_by_rank]),
            minlength=global_experts,
        ).to(torch.int32)
        expected_counts = expected_counts[rank * local_experts : (rank + 1) * local_experts]

        out.fill_(torch.nan)
        expert_token_nums.fill_(-1)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz,
            weight2=weight2_nz,
            expert_idx=expert_idx.npu(),
            scale1=scale1,
            scale2=scale2,
            bias1=empty_bias,
            bias2=empty_bias,
            probs=probs,
            group=_get_hcomm_name(rank),
            max_output_size=512,
            x_active_mask=x_active_mask.npu(),
            out=out,
            expert_token_nums=expert_token_nums,
        )
        torch_npu.npu.synchronize()

        torch.testing.assert_close(out[:active_tokens].cpu(), expected[:active_tokens], rtol=0.04, atol=0.04)
        torch.testing.assert_close(expert_token_nums.cpu(), expected_counts)
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
def test_dispatch_ffn_combine_w8a8_two_ranks():
    world_size = 2
    port = 29501 + random.randint(0, 10000)
    mp.spawn(_run_rank, args=(world_size, port), nprocs=world_size, join=True)
