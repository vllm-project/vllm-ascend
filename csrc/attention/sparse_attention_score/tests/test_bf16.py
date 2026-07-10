# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

import importlib
import importlib.util
import math
import os
import sys
from math import ceil
from pathlib import Path

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
INNER_PRECISE_FP8 = 4

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
sys.path.insert(0, str(_TORCH_EXT))


def _per_block_scale(fp32_block: torch.Tensor) -> float:
    amax = fp32_block.abs().max().item()
    if amax < 1e-12:
        return 1.0
    return amax / FP8_MAX


def _quantize_fp8(fp32_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return (fp32_tensor / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def _register_sparse_attention_score_op():
    spec = importlib.util.spec_from_file_location(
        "sparse_attention_score",
        _TORCH_EXT / "cann_ops_transformer" / "ops" / "sparse_attention_score.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cache_base = Path.home() / ".cache" / "torch_extensions"
    candidate_dirs = [cache_base / "py311_npu" / "npu_sparse_attention_score"]
    if cache_base.is_dir():
        candidate_dirs.extend(
            sub / "npu_sparse_attention_score"
            for sub in cache_base.iterdir()
            if "npu" in sub.name.lower()
        )

    so_file = None
    for so_dir in candidate_dirs:
        if so_dir.is_dir():
            for f in so_dir.iterdir():
                if f.suffix == ".so":
                    so_file = f
                    break
        if so_file is not None:
            break

    if so_file is None:
        print("[INFO] Pre-compiled npu_sparse_attention_score .so not found; "
              "OpBuilder will compile it on first NPU call.")
        return

    ext_spec = importlib.util.spec_from_file_location("npu_sparse_attention_score", so_file)
    ext_mod = importlib.util.module_from_spec(ext_spec)
    ext_spec.loader.exec_module(ext_mod)

    from cann_ops_transformer.op_builder.builder import OpBuilder

    OpBuilder._loaded_ops["npu_sparse_attention_score"] = ext_mod
    print(f"[INFO] Loaded pre-compiled extension: {so_file}")


_register_sparse_attention_score_op()


def generate_block_index_with_causal(query_fp32, key_fp32, q_seqlen, kv_seqlen,
                                     kv_heads, group_size, block_size=128, top_k=16):
    his_seq_len = kv_seqlen - q_seqlen
    total_blocks = ceil(kv_seqlen / block_size)
    head_dim = query_fp32.shape[-1]

    select_idx = torch.full((kv_heads, q_seqlen, top_k), -1, dtype=torch.int32)
    select_num_idx = torch.zeros((kv_heads, q_seqlen), dtype=torch.int32)

    for kv_head in range(kv_heads):
        representative_q_head = kv_head * group_size
        k_head = key_fp32[:, kv_head, :]

        for q_token in range(q_seqlen):
            q_vec = query_fp32[q_token, representative_q_head, :]
            causal_bound = his_seq_len + q_token

            scores = torch.matmul(q_vec, k_head[:kv_seqlen, :].transpose(0, 1))

            pooled = torch.full((total_blocks,), -float("inf"), dtype=torch.float32)
            q_block = causal_bound // block_size

            for block_idx in range(total_blocks):
                block_begin = block_idx * block_size
                block_end = min(block_begin + block_size, kv_seqlen)

                if block_idx > q_block:
                    pooled[block_idx] = -float("inf")
                elif block_idx == q_block:
                    pooled[block_idx] = float("inf")
                else:
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end > block_begin:
                        pooled[block_idx] = torch.max(scores[block_begin:effective_end]).item()

            visible_blocks = min(total_blocks, q_block + 1)
            valid_k = min(top_k, visible_blocks)
            select_num_idx[kv_head, q_token] = valid_k
            if valid_k > 0:
                topk_indices = torch.topk(pooled, k=valid_k, largest=True).indices.to(torch.int32)
                select_idx[kv_head, q_token, :valid_k] = topk_indices

    return select_idx, select_num_idx


def generate_block_table(batch, max_blocks_per_batch, shuffle=True):
    """Generate block table mapping logical -> physical block IDs.

    When shuffle=True, the mapping is randomized so logical_id != physical_id,
    which tests the paged KV cache address translation in the kernel.
    """
    total_physical = batch * max_blocks_per_batch
    all_physical_ids = list(range(total_physical))
    if shuffle:
        import random
        rng = random.Random(137)
        rng.shuffle(all_physical_ids)
    block_table = torch.zeros(batch, max_blocks_per_batch, dtype=torch.int32)
    for b in range(batch):
        for i in range(max_blocks_per_batch):
            block_table[b, i] = all_physical_ids[b * max_blocks_per_batch + i]
    return block_table


def build_fp8_tensors_and_scales(query_fp32, key_fp32, value_fp32, block_table,
                                 actual_seq_lengths, actual_seq_lengths_kv,
                                 block_size):
    batch = len(actual_seq_lengths)
    q_heads = query_fp32.shape[1]
    kv_heads = key_fp32.shape[2]
    head_dim = query_fp32.shape[2]
    max_q_seqlen = max(actual_seq_lengths)
    max_kv_blocks = block_table.shape[1]
    max_q_blocks = ceil(max_q_seqlen / block_size)

    q_scales = torch.ones(batch, q_heads, max_q_blocks, 1, dtype=torch.float32)
    k_scales = torch.ones(batch, kv_heads, max_kv_blocks, 1, dtype=torch.float32)
    v_scales = torch.ones(batch, kv_heads, max_kv_blocks, 1, dtype=torch.float32)

    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = actual_seq_lengths_kv[batch_idx]
        history_len = kv_seqlen - q_seqlen
        for q_token in range(q_seqlen):
            global_q = q_offset + q_token
            logical_q_block = (history_len + q_token) // block_size
            for head in range(q_heads):
                vec = query_fp32[global_q, head, :]
                q_scales[batch_idx, head, logical_q_block, 0] = _per_block_scale(vec)
        q_offset += q_seqlen

        for logical_id in range(max_kv_blocks):
            physical_id = int(block_table[batch_idx, logical_id].item())
            for kv_h in range(kv_heads):
                k_block = key_fp32[physical_id, :, kv_h, :]
                v_block = value_fp32[physical_id, :, kv_h, :]
                k_scales[batch_idx, kv_h, logical_id, 0] = _per_block_scale(k_block)
                v_scales[batch_idx, kv_h, logical_id, 0] = _per_block_scale(v_block)

    query_fp8 = torch.empty(query_fp32.shape, dtype=FP8_DTYPE)
    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = actual_seq_lengths_kv[batch_idx]
        history_len = kv_seqlen - q_seqlen
        for q_token in range(q_seqlen):
            global_q = q_offset + q_token
            logical_q_block = (history_len + q_token) // block_size
            for head in range(q_heads):
                scale = q_scales[batch_idx, head, logical_q_block, 0].item()
                query_fp8[global_q, head, :] = _quantize_fp8(
                    query_fp32[global_q, head, :], scale)
        q_offset += q_seqlen

    key_fp8 = key_fp32.to(FP8_DTYPE)
    value_fp8 = value_fp32.to(FP8_DTYPE)
    for batch_idx in range(batch):
        for logical_id in range(max_kv_blocks):
            physical_id = int(block_table[batch_idx, logical_id].item())
            for kv_h in range(kv_heads):
                ks = k_scales[batch_idx, kv_h, logical_id, 0].item()
                vs = v_scales[batch_idx, kv_h, logical_id, 0].item()
                key_fp8[physical_id, :, kv_h, :] = _quantize_fp8(
                    key_fp32[physical_id, :, kv_h, :], ks)
                value_fp8[physical_id, :, kv_h, :] = _quantize_fp8(
                    value_fp32[physical_id, :, kv_h, :], vs)

    return query_fp8, key_fp8, value_fp8, q_scales, k_scales, v_scales


def cpu_sparse_attention_score_fp32(
    query, key, value, select_idx, block_table,
    actual_seq_lengths, actual_seq_lengths_kv,
    num_key_value_heads, select_num_idx=None,
    block_size=128, scale_value=1.0,
):
    """FP32 golden: pure float32 online-softmax without any truncation."""
    select_idx_cpu = select_idx.to(torch.int64)
    block_table_cpu = block_table.to(torch.int64)

    total_q_tokens, q_heads, head_dim = query.shape
    kv_heads = num_key_value_heads
    group_size = q_heads // kv_heads
    top_k = select_idx.shape[2]
    batch = len(actual_seq_lengths)

    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.float32)
    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = actual_seq_lengths_kv[batch_idx]
        history_len = kv_seqlen - q_seqlen

        for q_token_in_batch in range(q_seqlen):
            global_q_token = q_offset + q_token_in_batch
            causal_bound = history_len + q_token_in_batch

            for kv_head in range(kv_heads):
                valid_top_k = top_k
                if select_num_idx is not None:
                    valid_top_k = int(select_num_idx[kv_head, global_q_token].item())
                    valid_top_k = min(valid_top_k, top_k)
                if valid_top_k == 0:
                    continue

                q_start = kv_head * group_size
                q_group = query[global_q_token, q_start:q_start + group_size, :].float()

                last_max = torch.full((group_size,), -float("inf"), dtype=torch.float32)
                last_sum = torch.zeros(group_size, dtype=torch.float32)
                o_acc = torch.zeros(group_size, head_dim, dtype=torch.float32)
                is_first = True

                for topk_idx in range(valid_top_k):
                    logical_id = int(select_idx_cpu[kv_head, global_q_token, topk_idx].item())
                    if logical_id < 0:
                        continue
                    block_begin = logical_id * block_size
                    block_end = min(block_begin + block_size, kv_seqlen)
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end <= block_begin:
                        continue

                    physical_id = int(block_table_cpu[batch_idx, logical_id].item())
                    valid_len = effective_end - block_begin
                    k_block = key[physical_id, :valid_len, kv_head, :].float()
                    v_block = value[physical_id, :valid_len, kv_head, :].float()

                    s = torch.matmul(q_group, k_block.t())
                    s_scaled = s * scale_value

                    now_max = s_scaled.max(dim=1).values
                    if not is_first:
                        now_max = torch.max(now_max, last_max)

                    p = torch.exp(s_scaled - now_max.unsqueeze(1))
                    now_sum = p.sum(dim=1)

                    if is_first:
                        last_sum = now_sum
                        last_max = now_max
                    else:
                        correction = torch.exp(last_max - now_max)
                        last_sum = correction * last_sum + now_sum
                        last_max = now_max

                    pv = torch.matmul(p, v_block)
                    if is_first:
                        o_acc = pv
                    else:
                        o_acc = o_acc * correction.unsqueeze(1) + pv

                    is_first = False

                if last_sum.max() > 0:
                    result = o_acc / last_sum.unsqueeze(1)
                    output[global_q_token, q_start:q_start + group_size, :] = result

        q_offset += q_seqlen

    return output


def dual_golden_l1norm(actual, golden_high, golden_low, label):
    """Compute l1norm relative error using dual-golden method."""
    actual_f = actual.float().flatten()
    high_f = golden_high.float().flatten()
    low_f = golden_low.float().flatten()

    valid_mask = ~(torch.isnan(actual_f) | torch.isnan(high_f) | torch.isnan(low_f))
    if valid_mask.sum() == 0:
        print(f"  [{label}] All elements are NaN, cannot compare.")
        return

    actual_valid = actual_f[valid_mask]
    high_valid = high_f[valid_mask]
    low_valid = low_f[valid_mask]
    low_norm = low_valid.abs().sum().item()
    high_norm = high_valid.abs().sum().item()
    l1_actual_vs_low = (actual_valid - low_valid).abs().sum().item() / (low_norm + 1e-12)
    l1_high_vs_low = (high_valid - low_valid).abs().sum().item() / (low_norm + 1e-12)
    l1_actual_vs_high = (actual_valid - high_valid).abs().sum().item() / (high_norm + 1e-12)

    if l1_high_vs_low > 1e-12:
        l1_ratio = l1_actual_vs_low / l1_high_vs_low
    else:
        l1_ratio = float('inf') if l1_actual_vs_low > 1e-12 else 0.0

    cos_sim = torch.nn.functional.cosine_similarity(
        actual_valid.unsqueeze(0), high_valid.unsqueeze(0)
    ).item()

    print(f"  [{label}]")
    print(f"    L1norm(actual - golden_low)/L1norm(golden_low)  = {l1_actual_vs_low:.6f}")
    print(f"    L1norm(golden_high - golden_low)/L1norm(golden_low) = {l1_high_vs_low:.6f}")
    print(f"    L1norm(actual - golden_high)/L1norm(golden_high) = {l1_actual_vs_high:.6f}")
    print(f"    L1 ratio (actual-low)/(high-low) = {l1_ratio:.6f}")
    print(f"    cosine_similarity(actual, golden_high) = {cos_sim:.8f}")
    if l1_ratio <= 1.0:
        print(f"    PASS")
    else:
        print(f"    WARNING: ratio > 1.0")


def cpu_sparse_attention_score_bf16(
    query, key, value, select_idx, block_table,
    actual_seq_lengths, actual_seq_lengths_kv,
    num_key_value_heads, select_num_idx=None,
    block_size=128, scale_value=1.0,
):
    """BF16 golden: vectorized simulation of kernel online-softmax pipeline.

    Processes all heads in a KV group together for efficiency.
    Uses bf16 truncation at key points to approximate hardware behavior.
    """
    select_idx_cpu = select_idx.to(torch.int64)
    block_table_cpu = block_table.to(torch.int64)

    total_q_tokens, q_heads, head_dim = query.shape
    kv_heads = num_key_value_heads
    group_size = q_heads // kv_heads
    top_k = select_idx.shape[2]
    batch = len(actual_seq_lengths)
    scale_bf16 = torch.tensor(scale_value, dtype=torch.bfloat16).item()

    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.bfloat16)
    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = actual_seq_lengths_kv[batch_idx]
        history_len = kv_seqlen - q_seqlen

        for q_token_in_batch in range(q_seqlen):
            global_q_token = q_offset + q_token_in_batch
            causal_bound = history_len + q_token_in_batch

            for kv_head in range(kv_heads):
                valid_top_k = top_k
                if select_num_idx is not None:
                    valid_top_k = int(select_num_idx[kv_head, global_q_token].item())
                    valid_top_k = min(valid_top_k, top_k)
                if valid_top_k == 0:
                    continue

                # Q vectors for all heads in this group: [groupSize, D]
                q_start = kv_head * group_size
                q_group_bf16 = query[global_q_token, q_start:q_start + group_size, :]

                last_max_fp32 = torch.full((group_size,), -float("inf"), dtype=torch.float32)
                last_sum_fp32 = torch.zeros(group_size, dtype=torch.float32)
                o_acc_fp32 = torch.zeros(group_size, head_dim, dtype=torch.float32)
                is_first = True

                for topk_idx in range(valid_top_k):
                    logical_id = int(select_idx_cpu[kv_head, global_q_token, topk_idx].item())
                    if logical_id < 0:
                        continue
                    block_begin = logical_id * block_size
                    block_end = min(block_begin + block_size, kv_seqlen)
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end <= block_begin:
                        continue

                    physical_id = int(block_table_cpu[batch_idx, logical_id].item())
                    valid_len = effective_end - block_begin
                    k_bf16 = key[physical_id, :valid_len, kv_head, :]    # [valid_len, D]
                    v_bf16 = value[physical_id, :valid_len, kv_head, :]  # [valid_len, D]

                    # Step 1: QK: [groupSize, D] x [D, valid_len] -> [groupSize, valid_len]
                    s_fp32 = torch.matmul(q_group_bf16.float(), k_bf16.float().t())
                    s_bf16 = s_fp32.to(torch.bfloat16)

                    # Step 2a: scale
                    s_scaled_bf16 = (s_bf16.float() * scale_bf16).to(torch.bfloat16)

                    # Step 2b: row max -> [groupSize]
                    now_max_bf16 = s_scaled_bf16.max(dim=1).values.to(torch.bfloat16)

                    # Step 2c: update max
                    if not is_first:
                        last_max_bf16 = last_max_fp32.to(torch.bfloat16)
                        now_max_bf16 = torch.max(now_max_bf16, last_max_bf16)

                    # Step 2d: exp(S - max) -> P [groupSize, valid_len]
                    diff_bf16 = (s_scaled_bf16.float() - now_max_bf16.float().unsqueeze(1)).to(torch.bfloat16)
                    p_bf16 = torch.exp(diff_bf16.float()).to(torch.bfloat16)

                    # Step 2e: row sum -> [groupSize]
                    now_sum_bf16 = p_bf16.float().sum(dim=1).to(torch.bfloat16)

                    # Step 2f: update lastMax/lastSum in fp32
                    now_max_fp32 = now_max_bf16.float()
                    if is_first:
                        last_sum_fp32 = now_sum_bf16.float()
                        last_max_fp32 = now_max_fp32
                    else:
                        correction_fp32 = torch.exp(last_max_fp32 - now_max_fp32)
                        last_sum_fp32 = correction_fp32 * last_sum_fp32 + now_sum_bf16.float()
                        last_max_fp32 = now_max_fp32

                    # Step 3: PV: [groupSize, valid_len] x [valid_len, D] -> [groupSize, D]
                    pv_fp32 = torch.matmul(p_bf16.float(), v_bf16.float())

                    # Step 4: rescale O
                    if is_first:
                        o_acc_fp32 = pv_fp32
                    else:
                        o_acc_fp32 = o_acc_fp32 * correction_fp32.unsqueeze(1) + pv_fp32

                    is_first = False

                # Step 5: final divide + cast
                if last_sum_fp32.max() > 0:
                    result_fp32 = o_acc_fp32 / last_sum_fp32.unsqueeze(1)
                    output[global_q_token, q_start:q_start + group_size, :] = result_fp32.to(torch.bfloat16)

        q_offset += q_seqlen

    return output


def cpu_sparse_attention_score_fp8(
    query_fp8, key_fp8, value_fp8, select_idx, block_table,
    q_scales, k_scales, v_scales,
    actual_seq_lengths, actual_seq_lengths_kv,
    num_key_value_heads, select_num_idx=None,
    block_size=128, scale_value=1.0,
):
    """FP8 golden: per-block dequant aligned with full-quant kernel indexing."""
    select_idx_cpu = select_idx.to(torch.int64)
    block_table_cpu = block_table.to(torch.int64)

    total_q_tokens, q_heads, head_dim = query_fp8.shape
    kv_heads = num_key_value_heads
    group_size = q_heads // kv_heads
    top_k = select_idx.shape[2]
    batch = len(actual_seq_lengths)

    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.float32)

    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = actual_seq_lengths_kv[batch_idx]
        history_len = kv_seqlen - q_seqlen

        for q_token_in_batch in range(q_seqlen):
            global_q_token = q_offset + q_token_in_batch
            causal_bound = history_len + q_token_in_batch
            logical_q_block = (history_len + q_token_in_batch) // block_size

            for q_head in range(q_heads):
                kv_head = q_head // group_size

                valid_top_k = top_k
                if select_num_idx is not None:
                    valid_top_k = int(select_num_idx[kv_head, global_q_token].item())
                    valid_top_k = min(valid_top_k, top_k)

                q_scale = q_scales[batch_idx, q_head, logical_q_block, 0].item()
                q_deq = query_fp8[global_q_token, q_head, :].float() * q_scale

                max_score = -float("inf")
                sum_exp = 0.0
                o_acc = torch.zeros(head_dim, dtype=torch.float32)

                for topk_idx in range(valid_top_k):
                    logical_id = int(select_idx_cpu[kv_head, global_q_token, topk_idx].item())
                    if logical_id < 0:
                        continue
                    block_begin = logical_id * block_size
                    block_end = min(block_begin + block_size, kv_seqlen)
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end <= block_begin:
                        continue

                    physical_id = int(block_table_cpu[batch_idx, logical_id].item())
                    valid_len = effective_end - block_begin

                    k_scale = k_scales[batch_idx, kv_head, logical_id, 0].item()
                    v_scale = v_scales[batch_idx, kv_head, logical_id, 0].item()

                    k_deq = key_fp8[physical_id, :valid_len, kv_head, :].float() * k_scale
                    v_deq = value_fp8[physical_id, :valid_len, kv_head, :].float() * v_scale

                    raw_score = torch.matmul(q_deq, k_deq.transpose(0, 1))
                    score = raw_score * scale_value
                    tile_max = score.max().item()
                    new_max = max(max_score, tile_max)
                    correction = math.exp(max_score - new_max) if max_score > -float("inf") else 0.0
                    if max_score > -float("inf"):
                        sum_exp = sum_exp * correction
                        o_acc = o_acc * correction
                    exp_score = torch.exp(score - new_max)
                    sum_exp = sum_exp + exp_score.sum().item()

                    p_fp8 = (exp_score * FP8_MAX).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
                    p_deq = p_fp8.float()
                    pv_acc_tile = torch.matmul(p_deq, v_deq)
                    o_acc = o_acc + pv_acc_tile
                    max_score = new_max

                if sum_exp > 0:
                    output[global_q_token, q_head, :] = o_acc / (sum_exp * FP8_MAX)

        q_offset += q_seqlen

    return output


# ---------------------------------------------------------------------------
# BF16 Test Class
# ---------------------------------------------------------------------------
class TestNpuSparseAttentionScoreBf16(TestCase):
    def make_case(self, q_seqlen=1, kv_seqlen=128, q_heads=1, kv_heads=1,
                  head_dim=128, block_size=128, top_k=1, seed=42):
        batch = 1
        group_size = q_heads // kv_heads
        total_blocks = ceil(kv_seqlen / block_size)
        max_blocks_per_batch = total_blocks
        actual_seq_lengths = torch.tensor([q_seqlen] * batch, dtype=torch.int32)
        actual_seq_lengths_kv = torch.tensor([kv_seqlen] * batch, dtype=torch.int32)

        torch.manual_seed(seed)
        query_fp32 = torch.rand(q_seqlen, q_heads, head_dim, dtype=torch.float32) * 2 - 1
        total_physical_blocks = total_blocks * batch
        key_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
        value_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1

        block_table = generate_block_table(batch, max_blocks_per_batch)
        key_logical = torch.zeros(total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
        for logical_id in range(total_blocks):
            physical_id = int(block_table[0, logical_id].item())
            key_logical[logical_id * block_size:(logical_id + 1) * block_size] = key_fp32[physical_id]
        key_flat = key_logical[:kv_seqlen, :, :]
        select_idx, select_num_idx = generate_block_index_with_causal(
            query_fp32, key_flat, q_seqlen, kv_seqlen,
            kv_heads, group_size, block_size, top_k)
        scale_value = 1.0 / math.sqrt(head_dim)

        return (
            query_fp32.to(torch.bfloat16),
            key_fp32.to(torch.bfloat16),
            value_fp32.to(torch.bfloat16),
            select_idx,
            block_table,
            select_num_idx,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            kv_heads,
            block_size,
            top_k,
            scale_value,
        )

    def make_case_multi_batch(self, batch=2, q_seqlens=None, kv_seqlens=None,
                              q_heads=8, kv_heads=2, head_dim=128, block_size=128,
                              top_k=4, seed=42):
        """Generate multi-batch test case.

        Each batch can have different q_seqlen and kv_seqlen.
        selectIdx shape: [kvHeads, totalQTokens, topK] (global token indexing).
        selectNumIdx shape: [kvHeads, totalQTokens].
        """
        if q_seqlens is None:
            q_seqlens = [1] * batch
        if kv_seqlens is None:
            kv_seqlens = [256] * batch
        assert len(q_seqlens) == batch and len(kv_seqlens) == batch

        group_size = q_heads // kv_heads
        max_kv_seqlen = max(kv_seqlens)
        max_blocks_per_batch = ceil(max_kv_seqlen / block_size)
        total_q_tokens = sum(q_seqlens)
        total_physical_blocks = max_blocks_per_batch * batch

        torch.manual_seed(seed)
        query_fp32 = torch.rand(total_q_tokens, q_heads, head_dim, dtype=torch.float32) * 2 - 1
        key_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
        value_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1

        block_table = generate_block_table(batch, max_blocks_per_batch)

        # Generate selectIdx per batch, then concatenate along token dim
        select_idx = torch.full((kv_heads, total_q_tokens, top_k), -1, dtype=torch.int32)
        select_num_idx = torch.zeros((kv_heads, total_q_tokens), dtype=torch.int32)

        q_offset = 0
        for b in range(batch):
            q_seqlen_b = q_seqlens[b]
            kv_seqlen_b = kv_seqlens[b]
            total_blocks_b = ceil(kv_seqlen_b / block_size)

            # Reconstruct logical key for this batch
            key_logical_b = torch.zeros(
                total_blocks_b * block_size, kv_heads, head_dim, dtype=torch.float32)
            for logical_id in range(total_blocks_b):
                physical_id = int(block_table[b, logical_id].item())
                key_logical_b[logical_id * block_size:(logical_id + 1) * block_size] = key_fp32[physical_id]
            key_flat_b = key_logical_b[:kv_seqlen_b, :, :]

            q_for_batch = query_fp32[q_offset:q_offset + q_seqlen_b, :, :]
            batch_select_idx, batch_select_num = generate_block_index_with_causal(
                q_for_batch, key_flat_b, q_seqlen_b, kv_seqlen_b,
                kv_heads, group_size, block_size, top_k)

            select_idx[:, q_offset:q_offset + q_seqlen_b, :] = batch_select_idx
            select_num_idx[:, q_offset:q_offset + q_seqlen_b] = batch_select_num
            q_offset += q_seqlen_b

        scale_value = 1.0 / math.sqrt(head_dim)

        actual_seq_lengths = torch.tensor(q_seqlens, dtype=torch.int32)
        actual_seq_lengths_kv = torch.tensor(kv_seqlens, dtype=torch.int32)

        return (
            query_fp32.to(torch.bfloat16),
            key_fp32.to(torch.bfloat16),
            value_fp32.to(torch.bfloat16),
            select_idx,
            block_table,
            select_num_idx,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            kv_heads,
            block_size,
            top_k,
            scale_value,
        )

    def _run_bf16_multi_batch_case(self, **kwargs):
        torch.npu.synchronize()
        torch.npu.empty_cache()
        case_data = self.make_case_multi_batch(**kwargs)
        (
            query, key, value, select_idx, block_table, select_num_idx,
            actual_seq_lengths, actual_seq_lengths_kv,
            kv_heads, block_size, top_k, scale_value,
        ) = case_data

        cpu_out = cpu_sparse_attention_score_bf16(
            query, key, value, select_idx, block_table,
            actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
            num_key_value_heads=kv_heads,
            select_num_idx=select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        cpu_out_fp32 = cpu_sparse_attention_score_fp32(
            query, key, value, select_idx, block_table,
            actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
            num_key_value_heads=kv_heads,
            select_num_idx=select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        npu_out = torch_npu.npu_sparse_attention_score(
            query.npu(), key.npu(), value.npu(), select_idx.npu(), block_table.npu(),
            select_num_idx=select_num_idx.npu(),
            actual_seq_lengths=actual_seq_lengths.npu(),
            actual_seq_lengths_kv=actual_seq_lengths_kv.npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=4,
        )

        npu_out_cpu = npu_out.cpu()
        diff = (npu_out_cpu.float() - cpu_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[bf16-multi-batch] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        dual_golden_l1norm(npu_out_cpu, cpu_out_fp32, cpu_out, "NPU vs dual-golden")

        self.assertRtolEqual(cpu_out.float().numpy(), npu_out_cpu.float().numpy(), prec=7e-3)

    # --- Multi-batch tests ---
    def test_bf16_mb_2batch_same_seqlen(self):
        self._run_bf16_multi_batch_case(
            batch=2, q_seqlens=[1, 1], kv_seqlens=[256, 256],
            q_heads=8, kv_heads=2, top_k=2, seed=42)

    def test_bf16_mb_2batch_diff_kvseqlen(self):
        self._run_bf16_multi_batch_case(
            batch=2, q_seqlens=[1, 1], kv_seqlens=[256, 512],
            q_heads=8, kv_heads=2, top_k=2, seed=42)

    def test_bf16_mb_2batch_diff_qseqlen(self):
        self._run_bf16_multi_batch_case(
            batch=2, q_seqlens=[1, 4], kv_seqlens=[512, 512],
            q_heads=8, kv_heads=2, top_k=4, seed=100)

    def test_bf16_mb_4batch_decode(self):
        self._run_bf16_multi_batch_case(
            batch=4, q_seqlens=[1, 1, 1, 1], kv_seqlens=[256, 512, 1024, 300],
            q_heads=16, kv_heads=4, top_k=3, seed=42)

    def test_bf16_mb_4batch_mixed(self):
        self._run_bf16_multi_batch_case(
            batch=4, q_seqlens=[1, 2, 4, 1], kv_seqlens=[300, 500, 1000, 700],
            q_heads=8, kv_heads=2, top_k=4, seed=7)

    def test_bf16_mb_8batch_decode_large_kv(self):
        self._run_bf16_multi_batch_case(
            batch=8, q_seqlens=[1]*8,
            kv_seqlens=[256, 512, 1024, 2048, 333, 555, 777, 1500],
            q_heads=16, kv_heads=4, top_k=5, seed=999)

    def test_bf16_mb_2batch_mha(self):
        self._run_bf16_multi_batch_case(
            batch=2, q_seqlens=[1, 1], kv_seqlens=[256, 384],
            q_heads=4, kv_heads=4, top_k=2, seed=42)

    def test_bf16_mb_4batch_gqa_large_group(self):
        self._run_bf16_multi_batch_case(
            batch=4, q_seqlens=[1, 1, 1, 1], kv_seqlens=[512, 700, 900, 1024],
            q_heads=32, kv_heads=4, top_k=4, seed=13)

    def test_bf16_prefill_then_decode_q64_kv4(self):
        """Simulate P-D sequential execution: Prefill (qseqlen=132, kvseqlen=132)
        then Decode (qseqlen=1, kvseqlen=133) using shared KV cache."""
        q_heads = 64
        kv_heads = 4
        head_dim = 128
        block_size = 128
        top_k = 16
        seed = 42
        batch = 1
        group_size = q_heads // kv_heads

        # Shared KV cache: kvseqlen=133 needs ceil(133/128)=2 blocks
        kv_seqlen_final = 133
        total_blocks = ceil(kv_seqlen_final / block_size)
        total_physical_blocks = total_blocks + 5

        torch.manual_seed(seed)
        # Generate full KV cache (2 blocks worth of data)
        key_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim,
            dtype=torch.float32) * 2 - 1
        value_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim,
            dtype=torch.float32) * 2 - 1

        import random
        rng = random.Random(seed)
        physical_ids = rng.sample(range(total_physical_blocks), total_blocks)
        block_table = torch.tensor([physical_ids], dtype=torch.int32)

        key = key_fp32.to(torch.bfloat16)
        value = value_fp32.to(torch.bfloat16)
        scale_value = 1.0 / math.sqrt(head_dim)

        # --- Phase 1: Prefill (qseqlen=132, kvseqlen=132) ---
        print("\n" + "=" * 60)
        print("[Phase 1] Prefill: qseqlen=132, kvseqlen=132")
        print("=" * 60)
        p_q_seqlen = 132
        p_kv_seqlen = 132

        torch.manual_seed(seed + 1)
        p_query_fp32 = torch.rand(p_q_seqlen, q_heads, head_dim,
                                  dtype=torch.float32) * 2 - 1
        p_query = p_query_fp32.to(torch.bfloat16)

        key_logical = torch.zeros(
            total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
        for lid in range(total_blocks):
            pid = int(block_table[0, lid].item())
            key_logical[lid * block_size:(lid + 1) * block_size] = key_fp32[pid]
        key_flat_p = key_logical[:p_kv_seqlen, :, :]

        p_select_idx, p_select_num_idx = generate_block_index_with_causal(
            p_query_fp32, key_flat_p, p_q_seqlen, p_kv_seqlen,
            kv_heads, group_size, block_size, top_k)

        p_cpu_bf16 = cpu_sparse_attention_score_bf16(
            p_query, key, value, p_select_idx, block_table,
            [p_q_seqlen], [p_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=p_select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )
        p_cpu_fp32 = cpu_sparse_attention_score_fp32(
            p_query, key, value, p_select_idx, block_table,
            [p_q_seqlen], [p_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=p_select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        p_npu_out = torch_npu.npu_sparse_attention_score(
            p_query.npu(), key.npu(), value.npu(),
            p_select_idx.npu(), block_table.npu(),
            select_num_idx=p_select_num_idx.npu(),
            actual_seq_lengths=torch.tensor([p_q_seqlen], dtype=torch.int32).npu(),
            actual_seq_lengths_kv=torch.tensor([p_kv_seqlen], dtype=torch.int32).npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=4,
        )
        p_npu_cpu = p_npu_out.cpu()

        p_diff = (p_npu_cpu.float() - p_cpu_bf16.float()).abs()
        print(f"  [Prefill] max_diff={p_diff.max().item():.6f}, "
              f"mean_diff={p_diff.mean().item():.6f}")
        dual_golden_l1norm(p_npu_cpu, p_cpu_fp32, p_cpu_bf16,
                           "Prefill NPU vs dual-golden")

        # --- Phase 2: Decode (qseqlen=1, kvseqlen=133) ---
        print("\n" + "=" * 60)
        print("[Phase 2] Decode: qseqlen=1, kvseqlen=133")
        print("=" * 60)
        d_q_seqlen = 1
        d_kv_seqlen = 133

        torch.manual_seed(seed + 2)
        d_query_fp32 = torch.rand(d_q_seqlen, q_heads, head_dim,
                                  dtype=torch.float32) * 2 - 1
        d_query = d_query_fp32.to(torch.bfloat16)

        key_flat_d = key_logical[:d_kv_seqlen, :, :]
        d_select_idx, d_select_num_idx = generate_block_index_with_causal(
            d_query_fp32, key_flat_d, d_q_seqlen, d_kv_seqlen,
            kv_heads, group_size, block_size, top_k)

        d_cpu_bf16 = cpu_sparse_attention_score_bf16(
            d_query, key, value, d_select_idx, block_table,
            [d_q_seqlen], [d_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=d_select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )
        d_cpu_fp32 = cpu_sparse_attention_score_fp32(
            d_query, key, value, d_select_idx, block_table,
            [d_q_seqlen], [d_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=d_select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        d_npu_out = torch_npu.npu_sparse_attention_score(
            d_query.npu(), key.npu(), value.npu(),
            d_select_idx.npu(), block_table.npu(),
            select_num_idx=d_select_num_idx.npu(),
            actual_seq_lengths=torch.tensor([d_q_seqlen], dtype=torch.int32).npu(),
            actual_seq_lengths_kv=torch.tensor([d_kv_seqlen], dtype=torch.int32).npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=4,
        )
        d_npu_cpu = d_npu_out.cpu()

        d_diff = (d_npu_cpu.float() - d_cpu_bf16.float()).abs()
        print(f"  [Decode] max_diff={d_diff.max().item():.6f}, "
              f"mean_diff={d_diff.mean().item():.6f}")
        dual_golden_l1norm(d_npu_cpu, d_cpu_fp32, d_cpu_bf16,
                           "Decode NPU vs dual-golden")

        # Assert both phases pass
        self.assertRtolEqual(p_cpu_bf16.float().numpy(),
                             p_npu_cpu.float().numpy(), prec=7e-3)
        self.assertRtolEqual(d_cpu_bf16.float().numpy(),
                             d_npu_cpu.float().numpy(), prec=7e-3)

    def _run_prefill_decode_case(self, p_q_seqlen, kv_seqlen, q_heads=64,
                                 kv_heads=4, head_dim=128, block_size=128,
                                 top_k=16, seed=42):
        """Run Prefill then Decode with shared KV cache."""
        group_size = q_heads // kv_heads
        d_kv_seqlen = kv_seqlen + p_q_seqlen
        total_blocks = ceil(d_kv_seqlen / block_size)
        total_physical_blocks = total_blocks + 5

        torch.manual_seed(seed)
        key_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim,
            dtype=torch.float32) * 2 - 1
        value_fp32 = torch.rand(
            total_physical_blocks, block_size, kv_heads, head_dim,
            dtype=torch.float32) * 2 - 1

        import random
        rng = random.Random(seed)
        physical_ids = rng.sample(range(total_physical_blocks), total_blocks)
        block_table = torch.tensor([physical_ids], dtype=torch.int32)

        key = key_fp32.to(torch.bfloat16)
        value = value_fp32.to(torch.bfloat16)
        scale_value = 1.0 / math.sqrt(head_dim)

        key_logical = torch.zeros(
            total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
        for lid in range(total_blocks):
            pid = int(block_table[0, lid].item())
            key_logical[lid * block_size:(lid + 1) * block_size] = key_fp32[pid]

        # --- Prefill ---
        p_kv_seqlen = kv_seqlen + p_q_seqlen  # After prefill, all tokens visible
        torch.manual_seed(seed + 1)
        p_query_fp32 = torch.rand(p_q_seqlen, q_heads, head_dim,
                                  dtype=torch.float32) * 2 - 1
        p_query = p_query_fp32.to(torch.bfloat16)

        key_flat_p = key_logical[:p_kv_seqlen, :, :]
        p_select_idx, p_select_num_idx = generate_block_index_with_causal(
            p_query_fp32, key_flat_p, p_q_seqlen, p_kv_seqlen,
            kv_heads, group_size, block_size, top_k)

        p_cpu_bf16 = cpu_sparse_attention_score_bf16(
            p_query, key, value, p_select_idx, block_table,
            [p_q_seqlen], [p_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=p_select_num_idx,
            block_size=block_size, scale_value=scale_value,
        )
        p_cpu_fp32 = cpu_sparse_attention_score_fp32(
            p_query, key, value, p_select_idx, block_table,
            [p_q_seqlen], [p_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=p_select_num_idx,
            block_size=block_size, scale_value=scale_value,
        )

        p_npu_out = torch_npu.npu_sparse_attention_score(
            p_query.npu(), key.npu(), value.npu(),
            p_select_idx.npu(), block_table.npu(),
            select_num_idx=p_select_num_idx.npu(),
            actual_seq_lengths=torch.tensor([p_q_seqlen], dtype=torch.int32).npu(),
            actual_seq_lengths_kv=torch.tensor([p_kv_seqlen], dtype=torch.int32).npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size, top_k=top_k, inner_precise=4,
        )
        p_npu_cpu = p_npu_out.cpu()

        # --- Decode ---
        d_q_seqlen = 1
        torch.manual_seed(seed + 2)
        d_query_fp32 = torch.rand(d_q_seqlen, q_heads, head_dim,
                                  dtype=torch.float32) * 2 - 1
        d_query = d_query_fp32.to(torch.bfloat16)

        key_flat_d = key_logical[:d_kv_seqlen, :, :]
        d_select_idx, d_select_num_idx = generate_block_index_with_causal(
            d_query_fp32, key_flat_d, d_q_seqlen, d_kv_seqlen,
            kv_heads, group_size, block_size, top_k)

        d_cpu_bf16 = cpu_sparse_attention_score_bf16(
            d_query, key, value, d_select_idx, block_table,
            [d_q_seqlen], [d_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=d_select_num_idx,
            block_size=block_size, scale_value=scale_value,
        )
        d_cpu_fp32 = cpu_sparse_attention_score_fp32(
            d_query, key, value, d_select_idx, block_table,
            [d_q_seqlen], [d_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=d_select_num_idx,
            block_size=block_size, scale_value=scale_value,
        )

        d_npu_out = torch_npu.npu_sparse_attention_score(
            d_query.npu(), key.npu(), value.npu(),
            d_select_idx.npu(), block_table.npu(),
            select_num_idx=d_select_num_idx.npu(),
            actual_seq_lengths=torch.tensor([d_q_seqlen], dtype=torch.int32).npu(),
            actual_seq_lengths_kv=torch.tensor([d_kv_seqlen], dtype=torch.int32).npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size, top_k=top_k, inner_precise=4,
        )
        d_npu_cpu = d_npu_out.cpu()

        # Report
        p_diff = (p_npu_cpu.float() - p_cpu_bf16.float()).abs()
        d_diff = (d_npu_cpu.float() - d_cpu_bf16.float()).abs()
        print(f"  [P] max_diff={p_diff.max().item():.6f}, mean={p_diff.mean().item():.6f}")
        print(f"  [D] max_diff={d_diff.max().item():.6f}, mean={d_diff.mean().item():.6f}")
        dual_golden_l1norm(p_npu_cpu, p_cpu_fp32, p_cpu_bf16, "Prefill NPU vs dual-golden")
        dual_golden_l1norm(d_npu_cpu, d_cpu_fp32, d_cpu_bf16, "Decode NPU vs dual-golden")

        self.assertRtolEqual(p_cpu_bf16.float().numpy(),
                             p_npu_cpu.float().numpy(), prec=7e-3)
        self.assertRtolEqual(d_cpu_bf16.float().numpy(),
                             d_npu_cpu.float().numpy(), prec=7e-3)

    def _run_bf16_case(self, **kwargs):
        torch.npu.synchronize()
        # torch.npu.empty_cache()
        case_data = self.make_case(**kwargs)
        (
            query, key, value, select_idx, block_table, select_num_idx,
            actual_seq_lengths, actual_seq_lengths_kv,
            kv_heads, block_size, top_k, scale_value,
        ) = case_data

        print("=" * 60)
        print("[SparseAttentionScore] Input shapes and params:")
        print(f"  query:              shape={list(query.shape)}, dtype={query.dtype}")
        print(f"  key:                shape={list(key.shape)}, dtype={key.dtype}")
        print(f"  value:              shape={list(value.shape)}, dtype={value.dtype}")
        print(f"  select_idx:         shape={list(select_idx.shape)}, dtype={select_idx.dtype}")
        print(f"    value={select_idx}")
        print(f"  block_table:        shape={list(block_table.shape)}, dtype={block_table.dtype}")
        print(f"    value={block_table}")
        print(f"  select_num_idx:     shape={list(select_num_idx.shape)}, dtype={select_num_idx.dtype}")
        print(f"    value={select_num_idx}")
        print(f"  actual_seq_lengths:     {actual_seq_lengths}")
        print(f"  actual_seq_lengths_kv:  {actual_seq_lengths_kv}")
        print(f"  num_key_value_heads:    {kv_heads}")
        print(f"  scale_value:            {scale_value}")
        print(f"  block_size:             {block_size}")
        print(f"  top_k:                  {top_k}")
        print(f"  inner_precise:          4")
        print("=" * 60)

        cpu_out = cpu_sparse_attention_score_bf16(
            query, key, value, select_idx, block_table,
            actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
            num_key_value_heads=kv_heads,
            select_num_idx=select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        cpu_out_fp32 = cpu_sparse_attention_score_fp32(
            query, key, value, select_idx, block_table,
            actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
            num_key_value_heads=kv_heads,
            select_num_idx=select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        npu_out = torch_npu.npu_sparse_attention_score(
            query.npu(), key.npu(), value.npu(), select_idx.npu(), block_table.npu(),
            select_num_idx=select_num_idx.npu(),
            actual_seq_lengths=actual_seq_lengths.npu(),
            actual_seq_lengths_kv=actual_seq_lengths_kv.npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=4,
        )

        npu_out_cpu = npu_out.cpu()
        diff = (npu_out_cpu.float() - cpu_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[bf16] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        dual_golden_l1norm(npu_out_cpu, cpu_out_fp32, cpu_out, "NPU vs dual-golden")

        self.assertRtolEqual(cpu_out.float().numpy(), npu_out_cpu.float().numpy(), prec=7e-3)


# Parametrized 100-case matrix for broad BF16 coverage
_BF16_CASES = [
    # (q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed)
    # --- topk sweep (aligned) ---
    (1, 128, 1, 1, 1, 42),
    (1, 256, 1, 1, 2, 42),
    (1, 384, 1, 1, 3, 42),
    (1, 512, 1, 1, 4, 42),
    (1, 640, 1, 1, 5, 42),
    (1, 768, 1, 1, 6, 42),
    (1, 1024, 1, 1, 8, 42),
    (1, 1280, 1, 1, 10, 42),
    (1, 1536, 1, 1, 12, 42),
    (1, 2048, 1, 1, 16, 42),
    # --- partial last block (various remainders) ---
    (1, 65, 1, 1, 1, 42),
    (1, 129, 1, 1, 2, 42),
    (1, 130, 1, 1, 2, 42),
    (1, 191, 1, 1, 2, 42),
    (1, 200, 1, 1, 2, 42),
    (1, 255, 1, 1, 2, 42),
    (1, 257, 1, 1, 3, 42),
    (1, 300, 1, 1, 3, 42),
    (1, 350, 1, 1, 3, 42),
    (1, 500, 1, 1, 4, 42),
    (1, 600, 1, 1, 5, 42),
    (1, 700, 1, 1, 6, 42),
    (1, 900, 1, 1, 7, 42),
    (1, 1000, 1, 1, 8, 42),
    (1, 1500, 1, 1, 12, 42),
    # --- GQA configs ---
    (1, 256, 4, 1, 2, 42),
    (1, 384, 4, 1, 3, 42),
    (1, 512, 4, 1, 4, 42),
    (1, 256, 8, 1, 2, 42),
    (1, 384, 8, 1, 3, 42),
    (1, 512, 8, 1, 4, 42),
    (1, 1024, 8, 1, 8, 42),
    (1, 256, 8, 2, 2, 42),
    (1, 384, 8, 2, 3, 42),
    (1, 512, 8, 2, 4, 42),
    (1, 1024, 8, 2, 8, 42),
    (1, 256, 16, 2, 2, 42),
    (1, 512, 16, 2, 4, 42),
    (1, 1024, 16, 2, 8, 42),
    (1, 256, 16, 4, 2, 42),
    (1, 512, 16, 4, 4, 42),
    (1, 1024, 16, 4, 8, 42),
    (1, 2048, 16, 4, 16, 42),
    # --- MHA configs ---
    (1, 256, 2, 2, 2, 42),
    (1, 384, 2, 2, 3, 42),
    (1, 512, 4, 4, 4, 42),
    (1, 1024, 4, 4, 8, 42),
    (1, 512, 8, 8, 4, 42),
    (1, 1024, 8, 8, 8, 42),
    # --- GQA + partial block ---
    (1, 200, 4, 1, 2, 42),
    (1, 300, 8, 2, 3, 42),
    (1, 500, 8, 1, 4, 42),
    (1, 700, 16, 4, 5, 42),
    (1, 900, 16, 2, 7, 42),
    # --- sparse select (topk << total_blocks) ---
    (1, 512, 1, 1, 2, 42),
    (1, 512, 1, 1, 3, 42),
    (1, 1024, 1, 1, 2, 42),
    (1, 1024, 1, 1, 3, 42),
    (1, 1024, 1, 1, 4, 42),
    (1, 2048, 1, 1, 2, 42),
    (1, 2048, 1, 1, 4, 42),
    (1, 2048, 1, 1, 8, 42),
    # --- seed sweep (topk=2 baseline with varied randomness) ---
    (1, 256, 1, 1, 2, 1),
    (1, 256, 1, 1, 2, 7),
    (1, 256, 1, 1, 2, 13),
    (1, 256, 1, 1, 2, 100),
    (1, 256, 1, 1, 2, 200),
    (1, 256, 1, 1, 2, 333),
    (1, 256, 1, 1, 2, 555),
    (1, 256, 1, 1, 2, 999),
    (1, 256, 1, 1, 2, 2024),
    (1, 256, 1, 1, 2, 12345),
    # --- seed sweep (topk=4) ---
    (1, 512, 1, 1, 4, 1),
    (1, 512, 1, 1, 4, 7),
    (1, 512, 1, 1, 4, 100),
    (1, 512, 1, 1, 4, 999),
    (1, 512, 1, 1, 4, 12345),
    # --- seed sweep (topk=8) ---
    (1, 1024, 1, 1, 8, 1),
    (1, 1024, 1, 1, 8, 7),
    (1, 1024, 1, 1, 8, 100),
    (1, 1024, 1, 1, 8, 999),
    (1, 1024, 1, 1, 8, 12345),
    # --- GQA + seed ---
    (1, 256, 4, 1, 2, 100),
    (1, 256, 4, 1, 2, 999),
    (1, 512, 8, 2, 4, 100),
    (1, 512, 8, 2, 4, 999),
    (1, 1024, 16, 4, 8, 100),
    (1, 1024, 16, 4, 8, 999),
    # --- partial + seed ---
    (1, 200, 1, 1, 2, 100),
    (1, 200, 1, 1, 2, 999),
    (1, 300, 1, 1, 3, 100),
    (1, 300, 1, 1, 3, 999),
    (1, 500, 1, 1, 4, 100),
    (1, 500, 1, 1, 4, 999),
    # --- extreme: topk=1 various kv lengths ---
    (1, 128, 1, 1, 1, 100),
    (1, 256, 1, 1, 1, 100),
    (1, 512, 1, 1, 1, 100),
    (1, 1024, 1, 1, 1, 100),
    (1, 2048, 1, 1, 1, 100),
]


def _make_bf16_test(q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed):
    def test_fn(self):
        self._run_bf16_case(q_seqlen=q_seqlen, kv_seqlen=kv_seqlen,
                            q_heads=q_heads, kv_heads=kv_heads,
                            top_k=top_k, seed=seed)
    return test_fn


for _i, (_qs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_BF16_CASES):
    _name = (f"test_bf16_{_i:03d}_qs{_qs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_"
             f"top{_tk}_seed{_sd}")
    setattr(TestNpuSparseAttentionScoreBf16, _name,
            _make_bf16_test(_qs, _kvs, _qh, _kvh, _tk, _sd))


# Large q_seqlen / kv_seqlen scenarios (1000 cases, programmatically generated)
def _generate_stress_cases(num_cases=1000):
    """Generate diverse cases: q<1k, kv<1k, multi-batch focus, GQA heavy.

    Coverage dimensions:
      - q_seqlen: 1 ~ 64
      - kv_seqlen: 128 ~ 1000 (aligned and unaligned)
      - q_heads / kv_heads: GQA-heavy (group=2,4,8) + some MHA
      - top_k: 1 ~ 8
      - seed: varied
    """
    import random
    rng = random.Random(20260609)

    q_seqlen_pool = [1, 1, 1, 1, 1, 2, 2, 4, 4, 8, 16, 32, 64]
    kv_seqlen_pool = [
        128, 200, 256, 300, 333, 384, 400, 450, 500, 512,
        555, 600, 640, 700, 750, 768, 800, 850, 900, 950, 1000,
    ]

    head_configs = [
        # (q_heads, kv_heads) — groupSize — GQA focused
        (8, 2),    # GQA group=4
        (8, 4),    # GQA group=2
        (16, 2),   # GQA group=8
        (16, 4),   # GQA group=4
        (32, 4),   # GQA group=8
        (32, 8),   # GQA group=4
        (4, 1),    # GQA group=4
        (8, 1),    # GQA group=8
        (4, 2),    # GQA group=2
        # MHA
        (4, 4),
        (8, 8),
    ]

    cases = []
    for i in range(num_cases):
        q_seqlen = rng.choice(q_seqlen_pool)
        kv_seqlen = rng.choice(kv_seqlen_pool)
        while kv_seqlen < q_seqlen:
            kv_seqlen = rng.choice(kv_seqlen_pool)
        q_heads, kv_heads = rng.choice(head_configs)
        from math import ceil
        total_blocks = ceil(kv_seqlen / 128)
        max_top_k = min(total_blocks, 8)
        if max_top_k < 2:
            continue
        else:
            top_k = rng.randint(2, max_top_k) #修改，暂时不支持等于1
        seed = rng.randint(1, 99999)
        cases.append((q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed))
    return cases


_BF16_LARGE_CASES = _generate_stress_cases(1000)


def _make_bf16_large_test(q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed):
    def test_fn(self):
        self._run_bf16_case(q_seqlen=q_seqlen, kv_seqlen=kv_seqlen,
                            q_heads=q_heads, kv_heads=kv_heads,
                            top_k=top_k, seed=seed)
    return test_fn


for _i, (_qs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_BF16_LARGE_CASES):
    _name = (f"test_bf16_large_{_i:03d}_qs{_qs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_"
             f"top{_tk}_seed{_sd}")
    setattr(TestNpuSparseAttentionScoreBf16, _name,
            _make_bf16_large_test(_qs, _kvs, _qh, _kvh, _tk, _sd))


# Long-sequence stress cases (q_seqlen up to 128, kv_seqlen 1k~64k)
def _generate_long_seq_cases(num_cases=200):
    """Generate cases with long kv_seqlen (1k~64k) and varied q_seqlen."""
    import random
    rng = random.Random(20260610)

    q_seqlen_pool = [1, 1, 1, 2, 4, 8, 16, 32, 64, 128]
    kv_seqlen_pool = [
        1024, 1500, 2000, 2048, 2500, 3000, 3333, 4000, 4096,
        5000, 5555, 6000, 7000, 7777, 8000, 8192, 9000, 10000,
        12000, 14000, 16000, 16384, 20000, 24000, 30000, 32000,
        32768, 40000, 50000, 60000, 65536,
    ]

    head_configs = [
        (8, 2),    # GQA group=4
        (16, 4),   # GQA group=4
        (32, 4),   # GQA group=8
        (16, 2),   # GQA group=8
        (8, 1),    # GQA group=8
        (4, 1),    # GQA group=4
        (8, 4),    # GQA group=2
        (8, 8),    # MHA
        (4, 4),    # MHA
    ]

    cases = []
    for i in range(num_cases):
        q_seqlen = rng.choice(q_seqlen_pool)
        kv_seqlen = rng.choice(kv_seqlen_pool)
        while kv_seqlen < q_seqlen:
            kv_seqlen = rng.choice(kv_seqlen_pool)
        q_heads, kv_heads = rng.choice(head_configs)
        from math import ceil
        total_blocks = ceil(kv_seqlen / 128)
        max_top_k = min(total_blocks, 16)
        if max_top_k < 2:
            continue
        else:
            top_k = rng.randint(1, max_top_k) #修改，暂时不支持等于1
        seed = rng.randint(1, 99999)
        cases.append((q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed))
    return cases


_BF16_LONGSEQ_CASES = _generate_long_seq_cases(200)


for _i, (_qs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_BF16_LONGSEQ_CASES):
    _name = (f"test_bf16_longseq_{_i:03d}_qs{_qs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_"
             f"top{_tk}_seed{_sd}")
    setattr(TestNpuSparseAttentionScoreBf16, _name,
            _make_bf16_large_test(_qs, _kvs, _qh, _kvh, _tk, _sd))


# 100-case Prefill+Decode sequential test matrix
_PD_CASES = [
    # (p_q_seqlen, kv_seqlen_before_prefill, q_heads, kv_heads, top_k, seed)
    # kv_seqlen during prefill = kv_seqlen + p_q_seqlen
    # kv_seqlen during decode = kv_seqlen + p_q_seqlen + 1
    # --- Basic: small prefill, various kv lengths ---
    (1, 127, 64, 4, 2, 42),
    (1, 128, 64, 4, 2, 42),
    (1, 132, 64, 4, 2, 42),
    (1, 200, 64, 4, 2, 42),
    (1, 255, 64, 4, 2, 42),
    (1, 256, 64, 4, 2, 42),
    (1, 300, 64, 4, 3, 42),
    (1, 384, 64, 4, 3, 42),
    (1, 500, 64, 4, 4, 42),
    (1, 512, 64, 4, 4, 42),
    # --- Prefill crossing block boundary ---
    (132, 0, 64, 4, 2, 42),
    (132, 1, 64, 4, 2, 42),
    (128, 0, 64, 4, 2, 42),
    (128, 5, 64, 4, 2, 42),
    (64, 64, 64, 4, 2, 42),
    (64, 128, 64, 4, 2, 42),
    (32, 100, 64, 4, 2, 42),
    (16, 120, 64, 4, 2, 42),
    (4, 128, 64, 4, 2, 42),
    (2, 130, 64, 4, 2, 42),
    # --- Larger prefill ---
    (256, 0, 64, 4, 4, 42),
    (256, 128, 64, 4, 4, 42),
    (256, 256, 64, 4, 4, 42),
    (512, 0, 64, 4, 8, 42),
    (512, 128, 64, 4, 8, 42),
    (512, 512, 64, 4, 8, 42),
    (1024, 0, 64, 4, 16, 42),
    (1024, 128, 64, 4, 16, 42),
    (1024, 512, 64, 4, 16, 42),
    (1024, 1024, 64, 4, 16, 42),
    # --- Different head configs ---
    (132, 0, 32, 4, 2, 42),
    (132, 0, 16, 4, 2, 42),
    (132, 0, 8, 2, 2, 42),
    (132, 0, 8, 1, 2, 42),
    (132, 0, 4, 1, 2, 42),
    (132, 0, 4, 4, 2, 42),
    (256, 0, 32, 4, 4, 42),
    (256, 0, 16, 2, 4, 42),
    (256, 0, 8, 8, 4, 42),
    (512, 0, 32, 4, 8, 42),
    # --- TopK sweep ---
    (132, 0, 64, 4, 1, 42),
    (132, 0, 64, 4, 2, 42),
    (132, 0, 64, 4, 4, 42),
    (132, 0, 64, 4, 8, 42),
    (132, 0, 64, 4, 16, 42),
    # (256, 128, 64, 4, 1, 42),
    (256, 128, 64, 4, 2, 42),
    (256, 128, 64, 4, 4, 42),
    (256, 128, 64, 4, 8, 42),
    (256, 128, 64, 4, 16, 42),
    # --- Partial last block (decode hits partial) ---
    (1, 126, 64, 4, 2, 42),
    (1, 127, 64, 4, 2, 42),
    (1, 129, 64, 4, 2, 42),
    (1, 130, 64, 4, 2, 42),
    (1, 131, 64, 4, 2, 42),
    (1, 254, 64, 4, 2, 42),
    (1, 255, 64, 4, 2, 42),
    (1, 257, 64, 4, 3, 42),
    (1, 383, 64, 4, 3, 42),
    (1, 511, 64, 4, 4, 42),
    # --- Seed sweep ---
    (132, 0, 64, 4, 2, 1),
    (132, 0, 64, 4, 2, 7),
    (132, 0, 64, 4, 2, 13),
    (132, 0, 64, 4, 2, 100),
    (132, 0, 64, 4, 2, 200),
    (132, 0, 64, 4, 2, 333),
    (132, 0, 64, 4, 2, 555),
    (132, 0, 64, 4, 2, 999),
    (132, 0, 64, 4, 2, 2024),
    (132, 0, 64, 4, 2, 12345),
    # --- Larger seq + seed ---
    (256, 256, 64, 4, 4, 1),
    (256, 256, 64, 4, 4, 100),
    (256, 256, 64, 4, 4, 999),
    (512, 512, 64, 4, 8, 1),
    (512, 512, 64, 4, 8, 100),
    (512, 512, 64, 4, 8, 999),
    (1024, 0, 64, 4, 16, 1),
    (1024, 0, 64, 4, 16, 100),
    (1024, 0, 64, 4, 16, 999),
    (1024, 1024, 64, 4, 16, 999),
    # --- Edge cases ---
    (1, 0, 64, 4, 1, 42),
    (2, 0, 64, 4, 1, 42),
    (128, 128, 64, 4, 2, 42),
    (127, 1, 64, 4, 2, 42),
    (129, 0, 64, 4, 2, 42),
    (63, 65, 64, 4, 2, 42),
    (33, 95, 64, 4, 2, 42),
    (7, 121, 64, 4, 2, 42),
    (3, 253, 64, 4, 2, 42),
    (1, 1023, 64, 4, 8, 42),
]


def _make_pd_test(p_q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed):
    def test_fn(self):
        self._run_prefill_decode_case(
            p_q_seqlen=p_q_seqlen, kv_seqlen=kv_seqlen,
            q_heads=q_heads, kv_heads=kv_heads,
            top_k=top_k, seed=seed)
    return test_fn


for _i, (_pqs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_PD_CASES):
    _name = (f"test_pd_{_i:03d}_pq{_pqs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_"
             f"top{_tk}_seed{_sd}")
    setattr(TestNpuSparseAttentionScoreBf16, _name,
            _make_pd_test(_pqs, _kvs, _qh, _kvh, _tk, _sd))


# --- Custom standalone test: q_heads=64, kv_heads=4, qseqlen=132, kvseqlen=132, topk=16 ---
def _test_bf16_q64_kv4_seqlen132_topk16(self):
    q_seqlen = 1
    kv_seqlen = 133
    q_heads = 64
    kv_heads = 4
    head_dim = 128
    block_size = 128
    top_k = 16
    seed = 42
    batch = 1
    group_size = q_heads // kv_heads
    total_blocks = ceil(kv_seqlen / block_size)  # = 100
    # KV cache physical blocks >= total_blocks
    total_physical_blocks = total_blocks + 10  # extra headroom
    max_blocks_per_batch = total_blocks
    actual_seq_lengths = torch.tensor([q_seqlen], dtype=torch.int32)
    actual_seq_lengths_kv = torch.tensor([kv_seqlen], dtype=torch.int32)

    torch.manual_seed(seed)
    query_fp32 = torch.rand(q_seqlen, q_heads, head_dim, dtype=torch.float32) * 2 - 1
    key_fp32 = torch.rand(
        total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
    value_fp32 = torch.rand(
        total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1

    # block_table shape [batch, max_blocks_per_batch], random mapping into physical blocks
    import random
    rng = random.Random(seed)
    physical_ids = rng.sample(range(total_physical_blocks), total_blocks)
    block_table = torch.tensor([physical_ids], dtype=torch.int32)

    key_logical = torch.zeros(total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
    for logical_id in range(total_blocks):
        physical_id = int(block_table[0, logical_id].item())
        key_logical[logical_id * block_size:(logical_id + 1) * block_size] = key_fp32[physical_id]
    key_flat = key_logical[:kv_seqlen, :, :]
    select_idx, select_num_idx = generate_block_index_with_causal(
        query_fp32, key_flat, q_seqlen, kv_seqlen,
        kv_heads, group_size, block_size, top_k)
    scale_value = 1.0 / math.sqrt(head_dim)

    query = query_fp32.to(torch.bfloat16)
    key = key_fp32.to(torch.bfloat16)
    value = value_fp32.to(torch.bfloat16)

    print("=" * 60)
    print("[SparseAttentionScore] Custom case - Input shapes and params:")
    print(f"  query:              shape={list(query.shape)}, dtype={query.dtype}")
    print(f"  key:                shape={list(key.shape)}, dtype={key.dtype}")
    print(f"  value:              shape={list(value.shape)}, dtype={value.dtype}")
    print(f"  select_idx:         shape={list(select_idx.shape)}, dtype={select_idx.dtype}")
    print(f"    value={select_idx}")
    print(f"  block_table:        shape={list(block_table.shape)}, dtype={block_table.dtype}")
    print(f"    value={block_table}")
    print(f"  select_num_idx:     shape={list(select_num_idx.shape)}, dtype={select_num_idx.dtype}")
    print(f"    value={select_num_idx}")
    print(f"  actual_seq_lengths:     {actual_seq_lengths}")
    print(f"  actual_seq_lengths_kv:  {actual_seq_lengths_kv}")
    print(f"  num_key_value_heads:    {kv_heads}")
    print(f"  scale_value:            {scale_value}")
    print(f"  block_size:             {block_size}")
    print(f"  top_k:                  {top_k}")
    print(f"  inner_precise:          4")
    print("=" * 60)

    cpu_out = cpu_sparse_attention_score_bf16(
        query, key, value, select_idx, block_table,
        actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
        num_key_value_heads=kv_heads,
        select_num_idx=select_num_idx,
        block_size=block_size,
        scale_value=scale_value,
    )

    cpu_out_fp32 = cpu_sparse_attention_score_fp32(
        query, key, value, select_idx, block_table,
        actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
        num_key_value_heads=kv_heads,
        select_num_idx=select_num_idx,
        block_size=block_size,
        scale_value=scale_value,
    )

    npu_out = torch_npu.npu_sparse_attention_score(
        query.npu(), key.npu(), value.npu(), select_idx.npu(), block_table.npu(),
        select_num_idx=select_num_idx.npu(),
        actual_seq_lengths=actual_seq_lengths.npu(),
        actual_seq_lengths_kv=actual_seq_lengths_kv.npu(),
        num_key_value_heads=kv_heads,
        scale_value=scale_value,
        block_size=block_size,
        top_k=top_k,
        inner_precise=4,
    )

    npu_out_cpu = npu_out.cpu()
    diff = (npu_out_cpu.float() - cpu_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[bf16] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    dual_golden_l1norm(npu_out_cpu, cpu_out_fp32, cpu_out, "NPU vs dual-golden")

    self.assertRtolEqual(cpu_out.float().numpy(), npu_out_cpu.float().numpy(), prec=7e-3)

setattr(TestNpuSparseAttentionScoreBf16, "test_bf16_q64_kv4_seqlen132_topk16",
        _test_bf16_q64_kv4_seqlen132_topk16)


# ---------------------------------------------------------------------------
# FP8 Test Class
# ---------------------------------------------------------------------------
# class TestNpuSparseAttentionScoreFp8(TestCase):
#     def make_case(self, q_seqlen=1, kv_seqlen=128, q_heads=1, kv_heads=1,
#                   head_dim=128, block_size=128, top_k=1, seed=42):
#         batch = 1
#         group_size = q_heads // kv_heads
#         total_blocks = ceil(kv_seqlen / block_size)
#         max_blocks_per_batch = total_blocks
#         actual_seq_lengths = torch.tensor([q_seqlen] * batch, dtype=torch.int32)
#         actual_seq_lengths_kv = torch.tensor([kv_seqlen] * batch, dtype=torch.int32)

#         torch.manual_seed(seed)
#         query_fp32 = torch.randn(q_seqlen, q_heads, head_dim, dtype=torch.float32)
#         total_physical_blocks = total_blocks * batch
#         key_fp32 = torch.randn(
#             total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32)
#         value_fp32 = torch.randn(
#             total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32)

#         block_table = generate_block_table(batch, max_blocks_per_batch)
#         key_logical = torch.zeros(total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
#         for logical_id in range(total_blocks):
#             physical_id = int(block_table[0, logical_id].item())
#             key_logical[logical_id * block_size:(logical_id + 1) * block_size] = key_fp32[physical_id]
#         key_flat = key_logical[:kv_seqlen, :, :]
#         select_idx, select_num_idx = generate_block_index_with_causal(
#             query_fp32, key_flat, q_seqlen, kv_seqlen,
#             kv_heads, group_size, block_size, top_k)
#         scale_value = 1.0 / math.sqrt(head_dim)

#         query_fp8, key_fp8, value_fp8, q_scales, k_scales, v_scales = (
#             build_fp8_tensors_and_scales(
#                 query_fp32, key_fp32, value_fp32, block_table,
#                 actual_seq_lengths, actual_seq_lengths_kv, block_size))

#         return (
#             query_fp8, key_fp8, value_fp8, select_idx, block_table, select_num_idx,
#             q_scales, k_scales, v_scales,
#             actual_seq_lengths, actual_seq_lengths_kv,
#             kv_heads, block_size, top_k, scale_value,
#         )

#     def _run_fp8_case(self, **kwargs):
#         (
#             query, key, value, select_idx, block_table, select_num_idx,
#             q_scales, k_scales, v_scales,
#             actual_seq_lengths, actual_seq_lengths_kv,
#             kv_heads, block_size, top_k, scale_value,
#         ) = self.make_case(**kwargs)

#         cpu_out = cpu_sparse_attention_score_fp8(
#             query, key, value, select_idx, block_table,
#             q_scales, k_scales, v_scales,
#             actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
#             num_key_value_heads=kv_heads,
#             select_num_idx=select_num_idx,
#             block_size=block_size,
#             scale_value=scale_value,
#         )

#         npu_out = torch_npu.npu_sparse_attention_score(
#             query.npu(), key.npu(), value.npu(), select_idx.npu(), block_table.npu(),
#             select_num_idx=select_num_idx.npu(),
#             q_dequant_scale=q_scales.npu(),
#             k_dequant_scale=k_scales.npu(),
#             v_dequant_scale=v_scales.npu(),
#             actual_seq_lengths=actual_seq_lengths.npu(),
#             actual_seq_lengths_kv=actual_seq_lengths_kv.npu(),
#             num_key_value_heads=kv_heads,
#             scale_value=scale_value,
#             block_size=block_size,
#             top_k=top_k,
#             inner_precise=INNER_PRECISE_FP8,
#         )

#         npu_out_cpu = npu_out.cpu()
#         cpu_out_fp16 = cpu_out.to(torch.float16)
#         diff = (npu_out_cpu.float() - cpu_out_fp16.float()).abs()
#         max_diff = diff.max().item()
#         print(f"[fp8] max_diff={max_diff:.6f}, mean_diff={diff.mean().item():.6f}")
#         self.assertRtolEqual(cpu_out_fp16.numpy(), npu_out_cpu.numpy(), prec=2e-2)

#     # --- Single block ---
#     def test_fp8_topk1_basic(self):
#         self._run_fp8_case(q_seqlen=1, kv_seqlen=128, top_k=1)

#     # --- Two blocks ---
#     def test_fp8_topk2(self):
#         self._run_fp8_case(q_seqlen=1, kv_seqlen=256, top_k=2)

#     # --- Three blocks ---
#     def test_fp8_topk3(self):
#         self._run_fp8_case(q_seqlen=1, kv_seqlen=384, top_k=3)

#     # --- Four blocks ---
#     def test_fp8_topk4(self):
#         self._run_fp8_case(q_seqlen=1, kv_seqlen=512, top_k=4)

#     # --- GQA ---
#     def test_fp8_gqa_4q_1kv_topk2(self):
#         self._run_fp8_case(q_seqlen=1, kv_seqlen=256, q_heads=4, kv_heads=1, top_k=2)

#     # --- Partial last block ---
#     def test_fp8_topk2_partial_last_block(self):
#         self._run_fp8_case(q_seqlen=1, kv_seqlen=200, top_k=2)


if __name__ == "__main__":
    run_tests()
