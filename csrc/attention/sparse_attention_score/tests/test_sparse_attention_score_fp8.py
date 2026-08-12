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

ATTENTION_OUT_DTYPE = torch.bfloat16


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
            sub / "npu_sparse_attention_score" for sub in cache_base.iterdir() if "npu" in sub.name.lower()
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
        print(
            "[INFO] Pre-compiled npu_sparse_attention_score .so not found; OpBuilder will compile it on first NPU call."
        )
        return

    ext_spec = importlib.util.spec_from_file_location("npu_sparse_attention_score", so_file)
    ext_mod = importlib.util.module_from_spec(ext_spec)
    ext_spec.loader.exec_module(ext_mod)

    from cann_ops_transformer.op_builder.builder import OpBuilder

    OpBuilder._loaded_ops["npu_sparse_attention_score"] = ext_mod
    print(f"[INFO] Loaded pre-compiled extension: {so_file}")


_register_sparse_attention_score_op()


def generate_block_index_with_causal(
    query_fp32, key_fp32, q_seqlen, kv_seqlen, kv_heads, group_size, block_size=128, top_k=16
):
    his_seq_len = kv_seqlen - q_seqlen
    total_blocks = ceil(kv_seqlen / block_size)
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


def build_fp8_tensors_and_scales(
    query_fp32, key_fp32, value_fp32, block_table, actual_seq_lengths, actual_seq_lengths_kv, block_size
):
    query_fp8 = query_fp32.to(FP8_DTYPE)
    key_fp8 = key_fp32.to(FP8_DTYPE)
    value_fp8 = value_fp32.to(FP8_DTYPE)

    batch = len(actual_seq_lengths)
    q_heads = query_fp32.shape[1]
    kv_heads = key_fp32.shape[2]
    max_q_seqlen = max(actual_seq_lengths)
    max_kv_blocks = block_table.shape[1]
    max_q_blocks = ceil(max_q_seqlen / block_size)

    q_scales = torch.ones(batch, q_heads, max_q_blocks, 1, dtype=torch.float32)
    k_scales = torch.ones(batch, kv_heads, max_kv_blocks, 1, dtype=torch.float32)
    v_scales = torch.ones(batch, kv_heads, max_kv_blocks, 1, dtype=torch.float32)

    return query_fp8, key_fp8, value_fp8, q_scales, k_scales, v_scales


def cpu_sparse_attention_score_fp32(
    query_fp8,
    key_fp8,
    value_fp8,
    select_idx,
    block_table,
    q_scales,
    k_scales,
    v_scales,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    num_key_value_heads,
    select_num_idx=None,
    block_size=128,
    scale_value=1.0,
):
    select_idx_cpu = select_idx.to(torch.int64)
    block_table_cpu = block_table.to(torch.int64)

    total_q_tokens, q_heads, head_dim = query_fp8.shape
    kv_heads = num_key_value_heads
    group_size = q_heads // kv_heads
    top_k = select_idx.shape[2]
    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.float32)

    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = actual_seq_lengths_kv[batch_idx]
        history_len = kv_seqlen - q_seqlen

        for q_token_in_batch in range(q_seqlen):
            global_q_token = q_offset + q_token_in_batch
            causal_bound = history_len + q_token_in_batch

            for q_head in range(q_heads):
                kv_head = q_head // group_size

                valid_top_k = top_k
                if select_num_idx is not None:
                    valid_top_k = int(select_num_idx[kv_head, global_q_token].item())
                    valid_top_k = min(valid_top_k, top_k)

                q_fp8 = query_fp8[global_q_token, q_head, :]
                q_fp32 = q_fp8.float()

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

                    k_fp8 = key_fp8[physical_id, :valid_len, kv_head, :]
                    v_fp8 = value_fp8[physical_id, :valid_len, kv_head, :]
                    k_fp32 = k_fp8.float()
                    v_fp32 = v_fp8.float()

                    raw_score = torch.matmul(q_fp32, k_fp32.transpose(0, 1))
                    score = raw_score * scale_value
                    tile_max = score.max().item()
                    new_max = max(max_score, tile_max)
                    correction = math.exp(max_score - new_max) if max_score > -float("inf") else 0.0
                    if max_score > -float("inf"):
                        sum_exp = sum_exp * correction
                        o_acc = o_acc * correction
                    exp_score = torch.exp(score - new_max)
                    sum_exp = sum_exp + exp_score.sum().item()

                    p_fp32 = exp_score
                    pv_acc_tile = torch.matmul(p_fp32, v_fp32)
                    o_acc = o_acc + pv_acc_tile
                    max_score = new_max

                if sum_exp > 0:
                    output[global_q_token, q_head, :] = o_acc / sum_exp

        q_offset += q_seqlen

    return output


_FP8_CASES = [
    # --- basic aligned ---
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
    # --- partial last block ---
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
    # --- GQA ---
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
    # --- MHA ---
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
    # --- seed sweep ---
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
    (1, 512, 1, 1, 4, 1),
    (1, 512, 1, 1, 4, 7),
    (1, 512, 1, 1, 4, 100),
    (1, 512, 1, 1, 4, 999),
    (1, 512, 1, 1, 4, 12345),
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
    # --- topk=1 various kv ---
    (1, 128, 1, 1, 1, 100),
    (1, 256, 1, 1, 1, 100),
    (1, 512, 1, 1, 1, 100),
    (1, 1024, 1, 1, 1, 100),
    (1, 2048, 1, 1, 1, 100),
]

_FP8_MULTI_BATCH_CASES = [
    (2, [1, 1], [256, 256], 8, 2, 2, 42),
    (2, [1, 1], [256, 512], 8, 2, 2, 42),
    (2, [1, 4], [512, 512], 8, 2, 4, 100),
    (4, [1, 1, 1, 1], [256, 512, 1024, 300], 16, 4, 3, 42),
    (4, [1, 2, 4, 1], [300, 500, 1000, 700], 8, 2, 4, 7),
    (8, [1] * 8, [256, 512, 1024, 2048, 333, 555, 777, 1500], 16, 4, 5, 999),
    (2, [1, 1], [256, 384], 4, 4, 2, 42),
    (4, [1, 1, 1, 1], [512, 700, 900, 1024], 32, 4, 4, 13),
    (3, [1, 1, 1], [256, 384, 512], 8, 2, 3, 42),
    (2, [1, 1], [512, 1024], 16, 4, 4, 42),
    (4, [1, 2, 2, 1], [256, 256, 512, 512], 8, 1, 4, 7),
    (6, [1] * 6, [256] * 6, 12, 3, 3, 42),
]


def _generate_stress_cases(num_cases=200):
    import random

    rng = random.Random(20260718)

    q_seqlen_pool = [1, 1, 1, 1, 1, 2, 2, 4, 4, 8]
    kv_seqlen_pool = [
        128,
        200,
        256,
        300,
        333,
        384,
        400,
        450,
        500,
        512,
        555,
        600,
        640,
        700,
        750,
        768,
        800,
        850,
        900,
        950,
        1000,
    ]
    head_configs = [
        (8, 2),
        (8, 4),
        (16, 2),
        (16, 4),
        (32, 4),
        (32, 8),
        (4, 1),
        (8, 1),
        (4, 2),
        (4, 4),
        (8, 8),
    ]
    cases = []
    for _ in range(num_cases):
        q_seqlen = rng.choice(q_seqlen_pool)
        kv_seqlen = rng.choice(kv_seqlen_pool)
        while kv_seqlen < q_seqlen:
            kv_seqlen = rng.choice(kv_seqlen_pool)
        q_heads, kv_heads = rng.choice(head_configs)
        total_blocks = ceil(kv_seqlen / 128)
        max_top_k = min(total_blocks, 8)
        top_k = rng.randint(1, max_top_k)
        seed = rng.randint(1, 99999)
        cases.append((q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed))
    return cases


def _generate_long_seq_cases(num_cases=200):
    import random

    rng = random.Random(20260719)

    q_seqlen_pool = [1, 1, 1, 2, 4, 8, 16, 32, 64, 128]
    kv_seqlen_pool = [
        1024,
        1500,
        2000,
        2048,
        2500,
        3000,
        3333,
        4000,
        4096,
        5000,
        5555,
        6000,
        7000,
        7777,
        8000,
        8192,
        9000,
        10000,
        12000,
        14000,
        16000,
        16384,
        20000,
        24000,
        30000,
        32000,
        32768,
        40000,
        50000,
        60000,
        65536,
    ]
    head_configs = [
        (8, 2),
        (16, 4),
        (32, 4),
        (16, 2),
        (8, 1),
        (4, 1),
        (8, 4),
        (8, 8),
        (4, 4),
    ]
    cases = []
    for _ in range(num_cases):
        q_seqlen = rng.choice(q_seqlen_pool)
        kv_seqlen = rng.choice(kv_seqlen_pool)
        while kv_seqlen < q_seqlen:
            kv_seqlen = rng.choice(kv_seqlen_pool)
        q_heads, kv_heads = rng.choice(head_configs)
        total_blocks = ceil(kv_seqlen / 128)
        max_top_k = min(total_blocks, 16)
        top_k = rng.randint(1, max_top_k)
        top_k = 16
        seed = rng.randint(1, 99999)
        cases.append((q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed))
    return cases


_FP8_STRESS_CASES = _generate_stress_cases(200)
_FP8_LONGSEQ_CASES = _generate_long_seq_cases(200)


class TestNpuSparseAttentionScoreFp8(TestCase):
    def make_case(
        self, q_seqlen=1, kv_seqlen=128, q_heads=1, kv_heads=1, head_dim=128, block_size=128, top_k=1, seed=42
    ):
        batch = 1
        group_size = q_heads // kv_heads
        total_blocks = ceil(kv_seqlen / block_size)
        max_blocks_per_batch = total_blocks
        actual_seq_lengths = torch.tensor([q_seqlen] * batch, dtype=torch.int32)
        actual_seq_lengths_kv = torch.tensor([kv_seqlen] * batch, dtype=torch.int32)

        torch.manual_seed(seed)
        query_fp32 = torch.randn(q_seqlen, q_heads, head_dim, dtype=torch.float32)
        total_physical_blocks = total_blocks * batch
        key_fp32 = torch.randn(total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32)
        value_fp32 = torch.randn(total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32)

        block_table = generate_block_table(batch, max_blocks_per_batch)
        key_logical = torch.zeros(total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
        for logical_id in range(total_blocks):
            physical_id = int(block_table[0, logical_id].item())
            key_logical[logical_id * block_size : (logical_id + 1) * block_size] = key_fp32[physical_id]
        key_flat = key_logical[:kv_seqlen, :, :]
        select_idx, select_num_idx = generate_block_index_with_causal(
            query_fp32, key_flat, q_seqlen, kv_seqlen, kv_heads, group_size, block_size, top_k
        )
        scale_value = 1.0 / math.sqrt(head_dim)

        query_fp8, key_fp8, value_fp8, q_scales, k_scales, v_scales = build_fp8_tensors_and_scales(
            query_fp32, key_fp32, value_fp32, block_table, actual_seq_lengths, actual_seq_lengths_kv, block_size
        )

        return (
            query_fp8,
            key_fp8,
            value_fp8,
            select_idx,
            block_table,
            select_num_idx,
            q_scales,
            k_scales,
            v_scales,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            kv_heads,
            block_size,
            top_k,
            scale_value,
        )

    def _run_fp8_case(self, **kwargs):
        torch.npu.synchronize()
        (
            query,
            key,
            value,
            select_idx,
            block_table,
            select_num_idx,
            q_scales,
            k_scales,
            v_scales,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            kv_heads,
            block_size,
            top_k,
            scale_value,
        ) = self.make_case(**kwargs)

        print("=" * 60)
        print("[SparseAttentionScore FP8] Input shapes and params:")
        print(f"  query:    shape={list(query.shape)}, dtype={query.dtype}")
        print(f"  key:      shape={list(key.shape)}, dtype={key.dtype}")
        print(f"  value:    shape={list(value.shape)}, dtype={value.dtype}")
        print(f"  select_idx: shape={list(select_idx.shape)}")
        print(f"  block_table: shape={list(block_table.shape)}")
        print(f"  select_num_idx: shape={list(select_num_idx.shape)}")
        print(f"  kv_heads: {kv_heads}, scale_value: {scale_value:.6f}")
        print(f"  block_size: {block_size}, top_k: {top_k}")
        print("=" * 60)

        cpu_out = cpu_sparse_attention_score_fp32(
            query,
            key,
            value,
            select_idx,
            block_table,
            q_scales,
            k_scales,
            v_scales,
            actual_seq_lengths.tolist(),
            actual_seq_lengths_kv.tolist(),
            num_key_value_heads=kv_heads,
            select_num_idx=select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        npu_out = torch_npu.npu_sparse_attention_score(
            query.npu(),
            key.npu(),
            value.npu(),
            select_idx.npu(),
            block_table.npu(),
            select_num_idx=select_num_idx.npu(),
            actual_seq_lengths=actual_seq_lengths.npu(),
            actual_seq_lengths_kv=actual_seq_lengths_kv.npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=INNER_PRECISE_FP8,
            # attention_out_dtype=ATTENTION_OUT_DTYPE,
        )

        npu_out_cpu = npu_out.cpu()
        print(f"[dtype] npu_out dype: {npu_out_cpu.dtype}")
        npu_out_fp32 = npu_out_cpu.float()
        cpu_out_fp32 = cpu_out.float()
        diff = (npu_out_fp32 - cpu_out_fp32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            npu_out_fp32.flatten().unsqueeze(0), cpu_out_fp32.flatten().unsqueeze(0)
        ).item()
        print(f"[fp8] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f}")
        self.assertRtolEqual(cpu_out_fp32.numpy(), npu_out_fp32.numpy(), prec=2e-2)

    def make_case_multi_batch(
        self,
        batch=2,
        q_seqlens=None,
        kv_seqlens=None,
        q_heads=8,
        kv_heads=2,
        head_dim=128,
        block_size=128,
        top_k=4,
        seed=42,
    ):
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
        key_fp32 = torch.rand(total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
        value_fp32 = torch.rand(total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1

        block_table = generate_block_table(batch, max_blocks_per_batch)

        select_idx = torch.full((kv_heads, total_q_tokens, top_k), -1, dtype=torch.int32)
        select_num_idx = torch.zeros((kv_heads, total_q_tokens), dtype=torch.int32)

        q_offset = 0
        for b in range(batch):
            q_seqlen_b = q_seqlens[b]
            kv_seqlen_b = kv_seqlens[b]
            total_blocks_b = ceil(kv_seqlen_b / block_size)

            key_logical_b = torch.zeros(total_blocks_b * block_size, kv_heads, head_dim, dtype=torch.float32)
            for logical_id in range(total_blocks_b):
                physical_id = int(block_table[b, logical_id].item())
                key_logical_b[logical_id * block_size : (logical_id + 1) * block_size] = key_fp32[physical_id]
            key_flat_b = key_logical_b[:kv_seqlen_b, :, :]

            q_for_batch = query_fp32[q_offset : q_offset + q_seqlen_b, :, :]
            batch_select_idx, batch_select_num = generate_block_index_with_causal(
                q_for_batch, key_flat_b, q_seqlen_b, kv_seqlen_b, kv_heads, group_size, block_size, top_k
            )

            select_idx[:, q_offset : q_offset + q_seqlen_b, :] = batch_select_idx
            select_num_idx[:, q_offset : q_offset + q_seqlen_b] = batch_select_num
            q_offset += q_seqlen_b

        scale_value = 1.0 / math.sqrt(head_dim)
        actual_seq_lengths = torch.tensor(q_seqlens, dtype=torch.int32)
        actual_seq_lengths_kv = torch.tensor(kv_seqlens, dtype=torch.int32)

        query_fp8, key_fp8, value_fp8, q_scales, k_scales, v_scales = build_fp8_tensors_and_scales(
            query_fp32, key_fp32, value_fp32, block_table, actual_seq_lengths, actual_seq_lengths_kv, block_size
        )

        return (
            query_fp8,
            key_fp8,
            value_fp8,
            select_idx,
            block_table,
            select_num_idx,
            q_scales,
            k_scales,
            v_scales,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            kv_heads,
            block_size,
            top_k,
            scale_value,
        )

    def _run_fp8_multi_batch_case(self, **kwargs):
        torch.npu.synchronize()
        torch.npu.empty_cache()
        case_data = self.make_case_multi_batch(**kwargs)
        (
            query,
            key,
            value,
            select_idx,
            block_table,
            select_num_idx,
            q_scales,
            k_scales,
            v_scales,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            kv_heads,
            block_size,
            top_k,
            scale_value,
        ) = case_data

        cpu_out = cpu_sparse_attention_score_fp32(
            query,
            key,
            value,
            select_idx,
            block_table,
            q_scales,
            k_scales,
            v_scales,
            actual_seq_lengths.tolist(),
            actual_seq_lengths_kv.tolist(),
            num_key_value_heads=kv_heads,
            select_num_idx=select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        npu_out = torch_npu.npu_sparse_attention_score(
            query.npu(),
            key.npu(),
            value.npu(),
            select_idx.npu(),
            block_table.npu(),
            select_num_idx=select_num_idx.npu(),
            actual_seq_lengths=actual_seq_lengths.npu(),
            actual_seq_lengths_kv=actual_seq_lengths_kv.npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=INNER_PRECISE_FP8,
            # attention_out_dtype=ATTENTION_OUT_DTYPE,
        )

        npu_out_cpu = npu_out.cpu()
        print(f"[dtype] npu_out dype: {npu_out_cpu.dtype}")
        cpu_out_fp32 = cpu_out.float()
        npu_out_cpu_fp32 = npu_out_cpu.float()
        diff = (npu_out_cpu_fp32 - cpu_out_fp32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            npu_out_cpu_fp32.flatten().unsqueeze(0), cpu_out_fp32.flatten().unsqueeze(0)
        ).item()
        print(f"[fp8-multi-batch] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f}")
        self.assertRtolEqual(cpu_out_fp32.numpy(), npu_out_cpu_fp32.numpy(), prec=2e-2)

    def _run_fp8_prefill_decode_case(
        self, p_q_seqlen=132, kv_seqlen=0, q_heads=64, kv_heads=4, head_dim=128, block_size=128, top_k=16, seed=42
    ):
        group_size = q_heads // kv_heads
        d_kv_seqlen = kv_seqlen + p_q_seqlen + 1
        total_blocks = ceil(d_kv_seqlen / block_size)
        total_physical_blocks = total_blocks + 5

        torch.manual_seed(seed)
        key_fp32 = torch.rand(total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
        value_fp32 = torch.rand(total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1

        import random

        rng = random.Random(seed)
        physical_ids = rng.sample(range(total_physical_blocks), total_blocks)
        block_table = torch.tensor([physical_ids], dtype=torch.int32)

        key_logical = torch.zeros(total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
        for lid in range(total_blocks):
            pid = int(block_table[0, lid].item())
            key_logical[lid * block_size : (lid + 1) * block_size] = key_fp32[pid]

        scale_value = 1.0 / math.sqrt(head_dim)

        # Prefill
        p_kv_seqlen = kv_seqlen + p_q_seqlen
        torch.manual_seed(seed + 1)
        p_query_fp32 = torch.rand(p_q_seqlen, q_heads, head_dim, dtype=torch.float32) * 2 - 1
        key_flat_p = key_logical[:p_kv_seqlen, :, :]
        p_select_idx, p_select_num_idx = generate_block_index_with_causal(
            p_query_fp32, key_flat_p, p_q_seqlen, p_kv_seqlen, kv_heads, group_size, block_size, top_k
        )

        p_query_fp8, p_key_fp8, p_value_fp8, _, _, _ = build_fp8_tensors_and_scales(
            p_query_fp32,
            key_fp32,
            value_fp32,
            block_table,
            torch.tensor([p_q_seqlen], dtype=torch.int32),
            torch.tensor([p_kv_seqlen], dtype=torch.int32),
            block_size,
        )

        p_cpu = cpu_sparse_attention_score_fp32(
            p_query_fp8,
            p_key_fp8,
            p_value_fp8,
            p_select_idx,
            block_table,
            torch.ones(1, q_heads, ceil(p_q_seqlen / block_size), 1, dtype=torch.float32),
            torch.ones(1, kv_heads, total_blocks, 1, dtype=torch.float32),
            torch.ones(1, kv_heads, total_blocks, 1, dtype=torch.float32),
            [p_q_seqlen],
            [p_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=p_select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        p_npu = torch_npu.npu_sparse_attention_score(
            p_query_fp8.npu(),
            p_key_fp8.npu(),
            p_value_fp8.npu(),
            p_select_idx.npu(),
            block_table.npu(),
            select_num_idx=p_select_num_idx.npu(),
            actual_seq_lengths=torch.tensor([p_q_seqlen], dtype=torch.int32).npu(),
            actual_seq_lengths_kv=torch.tensor([p_kv_seqlen], dtype=torch.int32).npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=INNER_PRECISE_FP8,
            # attention_out_dtype=ATTENTION_OUT_DTYPE,
        )
        p_npu_cpu = p_npu.cpu()
        print(f"[dtype] npu_out dype: {p_npu_cpu.dtype}")
        # Decode
        d_q_seqlen = 1
        torch.manual_seed(seed + 2)
        d_query_fp32 = torch.rand(d_q_seqlen, q_heads, head_dim, dtype=torch.float32) * 2 - 1
        key_flat_d = key_logical[:d_kv_seqlen, :, :]
        d_select_idx, d_select_num_idx = generate_block_index_with_causal(
            d_query_fp32, key_flat_d, d_q_seqlen, d_kv_seqlen, kv_heads, group_size, block_size, top_k
        )

        d_query_fp8, d_key_fp8, d_value_fp8, _, _, _ = build_fp8_tensors_and_scales(
            d_query_fp32,
            key_fp32,
            value_fp32,
            block_table,
            torch.tensor([d_q_seqlen], dtype=torch.int32),
            torch.tensor([d_kv_seqlen], dtype=torch.int32),
            block_size,
        )

        d_cpu = cpu_sparse_attention_score_fp32(
            d_query_fp8,
            d_key_fp8,
            d_value_fp8,
            d_select_idx,
            block_table,
            torch.ones(1, q_heads, 1, 1, dtype=torch.float32),
            torch.ones(1, kv_heads, total_blocks, 1, dtype=torch.float32),
            torch.ones(1, kv_heads, total_blocks, 1, dtype=torch.float32),
            [d_q_seqlen],
            [d_kv_seqlen],
            num_key_value_heads=kv_heads,
            select_num_idx=d_select_num_idx,
            block_size=block_size,
            scale_value=scale_value,
        )

        d_npu = torch_npu.npu_sparse_attention_score(
            d_query_fp8.npu(),
            d_key_fp8.npu(),
            d_value_fp8.npu(),
            d_select_idx.npu(),
            block_table.npu(),
            select_num_idx=d_select_num_idx.npu(),
            actual_seq_lengths=torch.tensor([d_q_seqlen], dtype=torch.int32).npu(),
            actual_seq_lengths_kv=torch.tensor([d_kv_seqlen], dtype=torch.int32).npu(),
            num_key_value_heads=kv_heads,
            scale_value=scale_value,
            block_size=block_size,
            top_k=top_k,
            inner_precise=INNER_PRECISE_FP8,
            # attention_out_dtype=ATTENTION_OUT_DTYPE,
        )
        d_npu_cpu = d_npu.cpu()
        print(f"[dtype] npu_out dype: {d_npu_cpu.dtype}")
        p_diff = (p_npu_cpu.float() - p_cpu.float()).abs()
        d_diff = (d_npu_cpu.float() - d_cpu.float()).abs()
        p_cos = torch.nn.functional.cosine_similarity(
            p_npu_cpu.float().flatten().unsqueeze(0), p_cpu.float().flatten().unsqueeze(0)
        ).item()
        d_cos = torch.nn.functional.cosine_similarity(
            d_npu_cpu.float().flatten().unsqueeze(0), d_cpu.float().flatten().unsqueeze(0)
        ).item()
        print(f"  [P] max_diff={p_diff.max().item():.6f}, mean={p_diff.mean().item():.6f}, cos={p_cos:.8f}")
        print(f"  [D] max_diff={d_diff.max().item():.6f}, mean={d_diff.mean().item():.6f}, cos={d_cos:.8f}")
        self.assertRtolEqual(p_cpu.float().numpy(), p_npu_cpu.float().numpy(), prec=2e-2)
        self.assertRtolEqual(d_cpu.float().numpy(), d_npu_cpu.float().numpy(), prec=2e-2)


_FP8_PD_CASES = [
    # (p_q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed)
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


def _make_fp8_test(q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed):
    def test_fn(self):
        self._run_fp8_case(
            q_seqlen=q_seqlen, kv_seqlen=kv_seqlen, q_heads=q_heads, kv_heads=kv_heads, top_k=top_k, seed=seed
        )

    return test_fn


def _make_fp8_mb_test(batch, q_seqlens, kv_seqlens, q_heads, kv_heads, top_k, seed):
    def test_fn(self):
        self._run_fp8_multi_batch_case(
            batch=batch,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            q_heads=q_heads,
            kv_heads=kv_heads,
            top_k=top_k,
            seed=seed,
        )

    return test_fn


for _i, (_qs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_FP8_CASES):
    _name = f"test_fp8_{_i:03d}_qs{_qs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_top{_tk}_seed{_sd}"
    setattr(TestNpuSparseAttentionScoreFp8, _name, _make_fp8_test(_qs, _kvs, _qh, _kvh, _tk, _sd))

for _i, (_b, _qsl, _kvsl, _qh, _kvh, _tk, _sd) in enumerate(_FP8_MULTI_BATCH_CASES):
    _qsl_str = "_".join(str(x) for x in _qsl)
    _kvsl_str = "_".join(str(x) for x in _kvsl)
    _name = f"test_fp8_mb_{_i:03d}_b{_b}_q{_qsl_str}_kv{_kvsl_str}_qh{_qh}_kvh{_kvh}_top{_tk}_seed{_sd}"
    setattr(TestNpuSparseAttentionScoreFp8, _name, _make_fp8_mb_test(_b, _qsl, _kvsl, _qh, _kvh, _tk, _sd))


def _make_fp8_pd_test(p_q_seqlen, kv_seqlen, q_heads, kv_heads, top_k, seed):
    def test_fn(self):
        self._run_fp8_prefill_decode_case(
            p_q_seqlen=p_q_seqlen, kv_seqlen=kv_seqlen, q_heads=q_heads, kv_heads=kv_heads, top_k=top_k, seed=seed
        )

    return test_fn


for _i, (_pqs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_FP8_PD_CASES):
    _name = f"test_fp8_pd_{_i:03d}_pq{_pqs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_top{_tk}_seed{_sd}"
    setattr(TestNpuSparseAttentionScoreFp8, _name, _make_fp8_pd_test(_pqs, _kvs, _qh, _kvh, _tk, _sd))

for _i, (_qs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_FP8_STRESS_CASES):
    _name = f"test_fp8_stress_{_i:03d}_qs{_qs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_top{_tk}_seed{_sd}"
    setattr(TestNpuSparseAttentionScoreFp8, _name, _make_fp8_test(_qs, _kvs, _qh, _kvh, _tk, _sd))

for _i, (_qs, _kvs, _qh, _kvh, _tk, _sd) in enumerate(_FP8_LONGSEQ_CASES):
    _name = f"test_fp8_longseq_{_i:03d}_qs{_qs}_kv{_kvs}_qh{_qh}_kvh{_kvh}_top{_tk}_seed{_sd}"
    setattr(TestNpuSparseAttentionScoreFp8, _name, _make_fp8_test(_qs, _kvs, _qh, _kvh, _tk, _sd))

if __name__ == "__main__":
    run_tests()
