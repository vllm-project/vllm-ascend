# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared helpers for MC2 MoE dispatch/combine e2e tests (ZB + PTA baseline)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def mc2_test_mode(env_prefix: str, default: str = "correctness") -> str:
    return os.environ.get(f"{env_prefix}_MODE", default).strip().lower()


def mc2_hccl_port(env_prefix: str = "VLLM_ASCEND_MOE_MC2_TEST") -> int:
    zb_port = os.environ.get("VLLM_ASCEND_ZB_TEST_HCCL_PORT")
    if zb_port:
        return int(zb_port)
    return int(os.environ.get(f"{env_prefix}_HCCL_PORT", "29500"))


def mc2_int_env(name: str, zb_fallback: str, default: str) -> int:
    raw = os.environ.get(name)
    if raw is not None:
        return int(raw)
    zb_raw = os.environ.get(zb_fallback)
    if zb_raw is not None:
        return int(zb_raw)
    return int(default)


def mc2_world_size() -> int:
    return mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE",
        "VLLM_ASCEND_ZB_TEST_WORLD_SIZE",
        "8",
    )


def mc2_bench_iters() -> tuple[int, int]:
    warmups = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_WARMUPS",
        "VLLM_ASCEND_ZB_TEST_NUM_WARMUPS",
        "50",
    )
    tests = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS",
        "VLLM_ASCEND_ZB_TEST_NUM_TESTS",
        "100",
    )
    return warmups, tests


def mc2_profile_iters() -> int:
    tests_default = str(
        mc2_int_env(
            "VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS",
            "VLLM_ASCEND_ZB_TEST_NUM_TESTS",
            "100",
        )
    )
    return mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_PROFILE_TESTS",
        "VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS",
        tests_default,
    )


def mc2_trace_dir(default: str) -> str:
    return os.environ.get(
        "VLLM_ASCEND_MOE_MC2_TEST_TRACE_DIR",
        os.environ.get("VLLM_ASCEND_ZB_TEST_TRACE_DIR", default),
    )


def mc2_shape_config(world_size: int) -> dict:
    """Read tensor shapes; falls back to VLLM_ASCEND_ZB_TEST_* for cross-test parity."""
    num_tokens = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS",
        "VLLM_ASCEND_ZB_TEST_NUM_TOKENS",
        "32",
    )
    hidden = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_HIDDEN",
        "VLLM_ASCEND_ZB_TEST_HIDDEN",
        "2048",
    )
    num_topk = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK",
        "VLLM_ASCEND_ZB_TEST_NUM_TOPK",
        "8",
    )
    num_experts = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS",
        "VLLM_ASCEND_ZB_TEST_NUM_EXPERTS",
        str(max(world_size * 2, 16)),
    )
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size"
    assert num_topk <= num_experts, (
        f"num_topk={num_topk} must be <= num_experts={num_experts} "
        "(build_fixed_inputs uses random.sample without replacement)"
    )
    num_local_experts = num_experts // world_size
    global_bs = num_tokens * world_size
    return {
        "num_tokens": num_tokens,
        "hidden": hidden,
        "num_topk": num_topk,
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "global_bs": global_bs,
    }


def get_group_ep(rank: int) -> str:
    group = dist.group.WORLD
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def normalize_topk_weights(topk_weights: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    valid_mask = (topk_idx >= 0).to(topk_weights.dtype)
    masked_weights = topk_weights * valid_mask
    denom = masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return masked_weights / denom


def build_fixed_inputs(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (rank + 1)

    topk_idx_cpu = torch.empty((num_tokens, num_topk), dtype=torch.int32)
    expert_range = range(num_experts)
    for token_id in range(num_tokens):
        topk_idx_cpu[token_id] = torch.tensor(random.sample(expert_range, num_topk), dtype=torch.int32)
    topk_idx = topk_idx_cpu.to(device="npu")

    denom = float(num_topk * (num_topk + 1)) / 2.0
    topk_weights = (torch.arange(1, num_topk + 1, dtype=torch.float32, device="npu") / denom).repeat(num_tokens, 1)
    return x, topk_idx, topk_weights


def verify_combine_local(
    combined_x: torch.Tensor,
    original_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    rank: int,
    atol: float = 5e-5,
    rtol: float = 5e-5,
) -> None:
    normalized_weights = normalize_topk_weights(topk_weights.float(), topk_idx)
    weight_sum = normalized_weights.sum(dim=1).view(-1, 1)
    expected_x = original_x.float() * weight_sum

    actual_np = combined_x.float().cpu().numpy()
    expected_np = expected_x.cpu().numpy()
    passed = np.allclose(actual_np, expected_np, atol=atol, rtol=rtol)

    abs_diff = float(np.max(np.abs(actual_np - expected_np)))
    rel_diff = float(np.max(np.abs(actual_np - expected_np) / (np.abs(expected_np) + 1e-12)))
    assert passed, (
        f"rank {rank}: combine mismatch max_abs={abs_diff:.3e} max_rel={rel_diff:.3e} (atol={atol}, rtol={rtol})"
    )


def seed_worker(rank: int) -> None:
    random.seed(rank + 42)
    np.random.seed(rank + 42)
    torch.manual_seed(rank + 42)
