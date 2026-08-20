# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""DenseAttentionScore FP8 precision test (Q=NTD, KV=BNSD, output=NTD).

Reference: test_dense_attention_score.py (repo root, run-once example).
Golden: causal dense attention computed in fp32 on dequantized fp8 inputs
(unit dequant scales), TND layout internally, compared after layout restore.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch_npu

FP8_DTYPE = torch.float8_e4m3fn
INNER_PRECISE_FP8 = 4
BLOCK_SIZE = 128
HEAD_DIM = 128

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"

# (name, batch, q_seqlen, kv_seqlen, q_heads, kv_heads, seed)
_DENSE_NTD_CASES = [
    # decode, short kv (non-FD path)
    ("decode_gqa16_kv256", 1, 1, 256, 16, 1, 42),
    ("decode_gqa16_kv1024", 1, 1, 1024, 16, 1, 43),
    ("decode_batch2_kv512", 2, 1, 512, 8, 2, 44),
    # decode, long kv (FlashDecoding path)
    ("decode_fd_kv4096", 1, 1, 4096, 16, 1, 45),
    ("decode_fd_batch4_kv2048", 4, 1, 2048, 16, 1, 46),
    # prefill (non-FD: totalTaskNum > SASA_FD_MAX_BASE_TASK or balanced cores)
    ("prefill_q128_kv128", 1, 128, 128, 8, 1, 47),
    ("prefill_q200_kv500_partial", 1, 200, 500, 8, 2, 48),
    ("prefill_q129_kv257_partial", 1, 129, 257, 16, 2, 49),
    ("prefill_batch2_q128_kv384", 2, 128, 384, 8, 2, 50),
    # MHA
    ("decode_mha_kv768", 1, 1, 768, 4, 4, 51),
    ("prefill_mha_q128_kv640", 1, 128, 640, 4, 4, 52),
]

# Hybrid layout (Q=TND, KV=BNSD, out=TND): must match Q=NTD+KV=BNSD output.
_HYBRID_CASES = [
    ("hybrid_decode_kv256", 1, 1, 256, 16, 1, 42),
    ("hybrid_decode_fd_kv4096", 1, 1, 4096, 16, 1, 45),
    ("hybrid_prefill_q128_kv128", 1, 128, 128, 8, 1, 47),
    ("hybrid_prefill_q200_kv500_partial", 1, 200, 500, 8, 2, 48),
    ("hybrid_decode_batch4_kv2048", 4, 1, 2048, 16, 1, 46),
]


def register_dense_attention_score_op() -> None:
    """Load DenseAttentionScore torch extension wrapper (whl first, then source)."""
    try:
        import cann_ops_transformer.ops.dense_attention_score  # noqa: F401

        print("[INFO] loaded cann_ops_transformer.ops.dense_attention_score from whl")
        return
    except Exception as err:  # noqa: BLE001
        print(f"[INFO] whl import failed ({err}); fall back to source wrapper")

    wrapper = _TORCH_EXT / "cann_ops_transformer" / "ops" / "dense_attention_score.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"Operator wrapper does not exist: {wrapper}")
    ext_path = str(_TORCH_EXT)
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)
    spec = importlib.util.spec_from_file_location("dense_attention_score_test_reg", wrapper)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load operator wrapper: {wrapper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("[INFO] loaded DenseAttentionScore wrapper from source tree")


def generate_case(batch, q_seqlen, kv_seqlen, q_heads, kv_heads, seed):
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    if kv_seqlen < q_seqlen:
        raise ValueError("require kv_seqlen >= q_seqlen")

    total_q_tokens = batch * q_seqlen
    max_blocks_per_batch = math.ceil(kv_seqlen / BLOCK_SIZE)
    total_physical_blocks = batch * max_blocks_per_batch

    gen = torch.Generator().manual_seed(seed)
    query = (torch.rand(total_q_tokens, q_heads, HEAD_DIM, generator=gen, dtype=torch.float32) * 2 - 1).to(FP8_DTYPE)
    key = (
        torch.rand(total_physical_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM, generator=gen, dtype=torch.float32) * 2 - 1
    ).to(FP8_DTYPE)
    value = (
        torch.rand(total_physical_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM, generator=gen, dtype=torch.float32) * 2 - 1
    ).to(FP8_DTYPE)

    # shuffled paged layout: logical order != physical storage order
    layout_gen = torch.Generator().manual_seed(seed + 1000)
    block_table = (
        torch.randperm(total_physical_blocks, generator=layout_gen, dtype=torch.int64)
        .reshape(batch, max_blocks_per_batch)
        .to(torch.int32)
    )

    q_seqlens = [q_seqlen] * batch
    kv_seqlens = [kv_seqlen] * batch

    # unit dequant scales
    max_q_blocks = math.ceil(q_seqlen / BLOCK_SIZE)
    q_dequant_scale = torch.ones(batch, q_heads, max_q_blocks, 1, dtype=torch.float32)
    k_dequant_scale = torch.ones(batch, kv_heads, max_blocks_per_batch, 1, dtype=torch.float32)
    v_dequant_scale = torch.ones(batch, kv_heads, max_blocks_per_batch, 1, dtype=torch.float32)

    return {
        "query": query,  # TND [T, N, D] fp8 (golden layout)
        "key": key,  # BSND [blocks, bs, kvh, D] fp8
        "value": value,
        "block_table": block_table,
        "q_seqlens": q_seqlens,
        "kv_seqlens": kv_seqlens,
        "actual_seq_lengths": torch.tensor(q_seqlens, dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor(kv_seqlens, dtype=torch.int32),
        "q_dequant_scale": q_dequant_scale,
        "k_dequant_scale": k_dequant_scale,
        "v_dequant_scale": v_dequant_scale,
        "scale_value": 1.0 / math.sqrt(HEAD_DIM),
    }


def cpu_dense_attention_score_fp32(case):
    """Causal dense attention golden in fp32 on dequantized fp8 inputs (TND output)."""
    query = case["query"]  # [T, N, D]
    key = case["key"]  # [blocks, bs, kvh, D]
    value = case["value"]
    block_table = case["block_table"].to(torch.int64)
    q_seqlens = case["q_seqlens"]
    kv_seqlens = case["kv_seqlens"]

    total_q_tokens, q_heads, head_dim = query.shape
    kv_heads = key.shape[2]
    group_size = q_heads // kv_heads

    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.float32)
    scale_value = case["scale_value"]

    q_offset = 0
    for batch_idx, q_seqlen in enumerate(q_seqlens):
        kv_seqlen = kv_seqlens[batch_idx]
        history_len = kv_seqlen - q_seqlen
        for q_token_in_batch in range(q_seqlen):
            global_q_token = q_offset + q_token_in_batch
            causal_bound = history_len + q_token_in_batch  # visible kv = bound + 1
            last_logical_block = causal_bound // BLOCK_SIZE

            for q_head in range(q_heads):
                kv_head = q_head // group_size
                q_fp32 = query[global_q_token, q_head, :].float()

                max_score = -float("inf")
                sum_exp = 0.0
                o_acc = torch.zeros(head_dim, dtype=torch.float32)

                for logical_id in range(last_logical_block + 1):
                    block_begin = logical_id * BLOCK_SIZE
                    effective_end = min(block_begin + BLOCK_SIZE, causal_bound + 1)
                    if effective_end <= block_begin:
                        continue
                    physical_id = int(block_table[batch_idx, logical_id].item())
                    valid_len = effective_end - block_begin
                    k_fp32 = key[physical_id, :valid_len, kv_head, :].float()
                    v_fp32 = value[physical_id, :valid_len, kv_head, :].float()

                    score = torch.matmul(q_fp32, k_fp32.transpose(0, 1)) * scale_value
                    tile_max = score.max().item()
                    new_max = max(max_score, tile_max)
                    correction = math.exp(max_score - new_max) if max_score > -float("inf") else 0.0
                    if max_score > -float("inf"):
                        sum_exp *= correction
                        o_acc = o_acc * correction
                    exp_score = torch.exp(score - new_max)
                    sum_exp += exp_score.sum().item()
                    o_acc = o_acc + torch.matmul(exp_score, v_fp32)
                    max_score = new_max

                if sum_exp > 0:
                    output[global_q_token, q_head, :] = o_acc / sum_exp
        q_offset += q_seqlen
    return output


def run_npu_dense(case, device, attention_out_dtype):
    """Call npu_dense_attention_score with Q=NTD, KV=BNSD; returns TND fp32 cpu tensor."""
    query_ntd = case["query"].permute(1, 0, 2).contiguous()  # [N, T, D]
    key_bnsd = case["key"].permute(0, 2, 1, 3).contiguous()  # [blocks, kvh, bs, D]
    value_bnsd = case["value"].permute(0, 2, 1, 3).contiguous()

    torch_npu.npu.set_device(device)
    out = torch_npu.npu_dense_attention_score(
        query_ntd.npu(),
        key_bnsd.npu(),
        value_bnsd.npu(),
        case["block_table"].npu(),
        q_dequant_scale=case["q_dequant_scale"].npu(),
        k_dequant_scale=case["k_dequant_scale"].npu(),
        v_dequant_scale=case["v_dequant_scale"].npu(),
        actual_seq_lengths=case["actual_seq_lengths"].npu(),
        actual_seq_lengths_kv=case["actual_seq_lengths_kv"].npu(),
        q_input_layout="NTD",
        kv_input_layout="BNSD",
        num_key_value_heads=int(key_bnsd.shape[1]),
        scale_value=case["scale_value"],
        block_size=BLOCK_SIZE,
        inner_precise=INNER_PRECISE_FP8,
        attention_out_dtype=attention_out_dtype,
    )
    torch_npu.npu.synchronize()
    # NTD [N, T, D] -> TND [T, N, D]
    return out.cpu().permute(1, 0, 2).float()


def run_npu_dense_hybrid(case, device, attention_out_dtype):
    """Call npu_dense_attention_score with Q=TND, KV=BNSD; returns TND fp32 cpu tensor."""
    query_tnd = case["query"]  # [T, N, D]
    key_bnsd = case["key"].permute(0, 2, 1, 3).contiguous()  # [blocks, kvh, bs, D]
    value_bnsd = case["value"].permute(0, 2, 1, 3).contiguous()

    torch_npu.npu.set_device(device)
    out = torch_npu.npu_dense_attention_score(
        query_tnd.npu(),
        key_bnsd.npu(),
        value_bnsd.npu(),
        case["block_table"].npu(),
        q_dequant_scale=case["q_dequant_scale"].npu(),
        k_dequant_scale=case["k_dequant_scale"].npu(),
        v_dequant_scale=case["v_dequant_scale"].npu(),
        actual_seq_lengths=case["actual_seq_lengths"].npu(),
        actual_seq_lengths_kv=case["actual_seq_lengths_kv"].npu(),
        q_input_layout="TND",
        kv_input_layout="BNSD",
        num_key_value_heads=int(key_bnsd.shape[1]),
        scale_value=case["scale_value"],
        block_size=BLOCK_SIZE,
        inner_precise=INNER_PRECISE_FP8,
        attention_out_dtype=attention_out_dtype,
    )
    torch_npu.npu.synchronize()
    # Q=TND -> output already TND [T, N, D]
    return out.cpu().float()


def check_precision(name, ref, npu_out, prec=2e-2):
    """Repo FP8 convention (assertRtolEqual): element passes if abs diff <= prec
    OR relative diff <= prec; allow a small outlier fraction (like size*prec).
    Note: full-quant kernel re-quantizes P to fp8 before the PV matmul, so
    tokens with a tiny causal window (large P weights) carry ~prec abs error;
    cos_sim gate guards against structural errors."""
    diff = (npu_out - ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(npu_out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    deno = torch.maximum(ref.abs(), npu_out.abs())
    elem_ok = (diff <= prec) | (diff / (deno + 1e-6) <= prec)
    pass_rate = elem_ok.float().mean().item()
    passed = pass_rate >= 1.0 - 1e-3 and cos_sim > 0.999
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
        f"cos_sim={cos_sim:.8f}, elem_pass_rate={pass_rate:.6f}"
    )
    return passed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--attention-out-dtype", default="bfloat16", choices=("bfloat16", "float16"))
    return parser.parse_args()


def main():
    args = parse_args()
    register_dense_attention_score_op()
    out_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.attention_out_dtype]

    failed = []
    for name, batch, q_seqlen, kv_seqlen, q_heads, kv_heads, seed in _DENSE_NTD_CASES:
        print("=" * 70)
        print(
            f"[CASE] {name}: batch={batch}, q_seqlen={q_seqlen}, kv_seqlen={kv_seqlen}, "
            f"q_heads={q_heads}, kv_heads={kv_heads}, seed={seed}"
        )
        case = generate_case(batch, q_seqlen, kv_seqlen, q_heads, kv_heads, seed)
        ref = cpu_dense_attention_score_fp32(case)
        npu_out = run_npu_dense(case, args.device, out_dtype)
        if not check_precision(name, ref, npu_out):
            failed.append(name)

    print("=" * 70)
    for name, batch, q_seqlen, kv_seqlen, q_heads, kv_heads, seed in _HYBRID_CASES:
        print(
            f"[CASE] {name}: batch={batch}, q_seqlen={q_seqlen}, kv_seqlen={kv_seqlen}, "
            f"q_heads={q_heads}, kv_heads={kv_heads}, seed={seed}"
        )
        case = generate_case(batch, q_seqlen, kv_seqlen, q_heads, kv_heads, seed)
        ref = cpu_dense_attention_score_fp32(case)
        out_ntd = run_npu_dense(case, args.device, out_dtype)  # baseline TND view
        out_hybrid = run_npu_dense_hybrid(case, args.device, out_dtype)

        exact = torch.equal(out_hybrid, out_ntd)
        max_diff = (out_hybrid - out_ntd).abs().max().item()
        golden_ok = check_precision(f"{name}/golden", ref, out_hybrid)
        passed = exact and golden_ok
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}/match_ntd: exact_equal={exact}, max_diff_vs_ntd={max_diff:.8f}")
        if not passed:
            failed.append(name)

    print("=" * 70)
    total = len(_DENSE_NTD_CASES) + len(_HYBRID_CASES)
    if failed:
        print(f"[RESULT] FAILED: {len(failed)}/{total} cases failed: {failed}")
        sys.exit(1)
    print(
        f"[RESULT] ALL {total} cases PASSED (11 NTD + {len(_HYBRID_CASES)} hybrid Q=TND/KV=BNSD, out dtype={out_dtype})"
    )


if __name__ == "__main__":
    main()
