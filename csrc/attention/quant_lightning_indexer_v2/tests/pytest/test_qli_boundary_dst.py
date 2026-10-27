#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""boundaryDst > 0 regression test for QLI mode4 tail-block fixup.

The original 6 equivalence cases never exercise the path where the mixed
boundary block sits at a nonzero rank inside the candidate list (boundaryDst
> 0), which is exactly where the two fixed bugs (eScr progression start
value + clamp off-by-one) corrupt the invalid-entry mask. This test injects
hand-built candidate lists so the boundary block lands at rank > 0, then
compares mode2 vs mode4 output multisets per (T, N2) row.
"""

import sys

import numpy as np
import torch
import torch_npu

PYTEST_DIR = "/vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_v2/tests/pytest"
sys.path.insert(0, PYTEST_DIR)

import quant_lightning_indexer_v2_golden as golden  # noqa: E402
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

CAND_TOPK_BLOCKS = 2048
CAND_BLOCK_SIZE = 8
SPARSE_COUNT = 512


def build_params(batch_size, q_seq, seqused_k_list, cmp_residual_list, block_num):
    cu_q = [0]
    for _ in range(batch_size):
        cu_q.append(cu_q[-1] + q_seq)
    return (
        batch_size,
        q_seq,
        max(seqused_k_list),
        batch_size * q_seq,
        0,
        64,
        1,
        128,
        128,
        block_num,
        "INT8",
        "FP16",
        "FLOAT16",
        "INT32",
        cu_q,
        None,
        [q_seq] * batch_size,
        seqused_k_list,
        cmp_residual_list,
        -1,
        2,
        "TND",
        "PA_BBND",
        SPARSE_COUNT,
        3,
        (-2, 2),
        (-2, 2),
        (-2, 2),
        (0.5, 1.5),
        (0.5, 1.5),
        2,
        0,
        None,
    )


def call_v3(data, metadata, mode, candidates):
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        data["query"].npu(),
        data["key"].npu(),
        data["weights"].npu(),
        data["query_dequant_scale"].npu(),
        data["key_dequant_scale"].npu(),
        data["sparse_count"],
        data["quant_mode"],
        candidate_topk_index=candidates,
        cu_seqlens_q=data["cu_seqlens_query"].npu() if data["cu_seqlens_query"] is not None else None,
        cu_seqlens_k=None,
        seqused_q=data["seqused_q"].npu() if data["seqused_q"] is not None else None,
        seqused_k=data["seqused_k"].npu() if data["seqused_k"] is not None else None,
        cmp_residual_k=data["cmp_residual_k_for_npu"].npu() if data["cmp_residual_k_for_npu"] is not None else None,
        block_table=data["block_table"].npu(),
        metadata=metadata,
        max_seqlen_q=data["max_seqlen_q_meta"],
        layout_q=data["layout_query"],
        layout_k=data["layout_key"],
        mask_mode=data["sparse_mode"],
        cmp_ratio=data["cmp_ratio"],
        candidate_mode=mode,
        candidate_topk_blocks=CAND_TOPK_BLOCKS,
        candidate_block_size=CAND_BLOCK_SIZE,
    )


def make_candidates(seqused_k, boundary_rank, total_blocks, t_rows):
    """Boundary block at the given rank, same list for every T row."""
    boundary_blk = (seqused_k + CAND_BLOCK_SIZE - 1) // CAND_BLOCK_SIZE - 1
    blocks = [b for b in range(total_blocks) if b != boundary_blk]
    order = (blocks[:boundary_rank] + [boundary_blk] + blocks[boundary_rank:])[:CAND_TOPK_BLOCKS]
    cand = torch.full((t_rows, 1, CAND_TOPK_BLOCKS), -1, dtype=torch.int32)
    for i, b in enumerate(order):
        cand[:, 0, i] = b
    return cand, boundary_blk


def run_case(name, seqused_k, boundary_rank, total_blocks, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    block_num = (seqused_k + 127) // 128 + 4
    params = build_params(1, 6, [seqused_k], [0], block_num)
    data = golden.generate_qliv2_test_data(params, generate_golden=False)
    md = data["metadata_input"]
    metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        cu_seqlens_q=md["cu_seqlens_q"].npu() if md["cu_seqlens_q"] is not None else None,
        cu_seqlens_k=None,
        seqused_q=md["seqused_q"].npu() if md["seqused_q"] is not None else None,
        seqused_k=md["seqused_k"].npu() if md["seqused_k"] is not None else None,
        cmp_residual_k=md["cmp_residual_k"].npu() if md["cmp_residual_k"] is not None else None,
        batch_size=md["batch_size"],
        max_seqlen_q=md["max_seqlen_q"],
        max_seqlen_k=md["max_seqlen_k"],
        num_heads_q=md["num_heads_q"],
        num_heads_k=md["num_heads_k"],
        head_dim=md["head_dim"],
        topk=md["topk"],
        quant_mode=md["quant_mode"],
        mask_mode=md["mask_mode"],
        layout_q=md["layout_q"],
        layout_k=md["layout_k"],
        cmp_ratio=md["cmp_ratio"],
    ).npu()

    cand, boundary_blk = make_candidates(seqused_k, boundary_rank, total_blocks, t_rows=6)
    cand = cand.npu()
    valid_tail = seqused_k - boundary_blk * CAND_BLOCK_SIZE

    sel2 = call_v3(data, metadata, 2, cand)[0].cpu()
    sel4 = call_v3(data, metadata, 4, cand)[0].cpu()
    torch.npu.synchronize()

    bad_rows = 0
    for t in range(sel2.shape[0]):
        for n in range(sel2.shape[1]):
            a = np.sort(sel2[t, n].numpy())
            b = np.sort(sel4[t, n].numpy())
            if not np.array_equal(a, b):
                bad_rows += 1
                if bad_rows <= 2:
                    diff_idx = np.where(a != b)[0][:5]
                    print(f"[{name}] row=({t},{n}) mismatch @{diff_idx}: m2={a[diff_idx]} m4={b[diff_idx]}")
    print(
        f"[{name}] {'PASS' if bad_rows == 0 else 'FAIL'} "
        f"(boundary_blk={boundary_blk}, valid_tail={valid_tail}, "
        f"boundary_rank={boundary_rank}, "
        f"boundary_dst={boundary_rank * CAND_BLOCK_SIZE}, bad={bad_rows})"
    )
    return bad_rows == 0


def main():
    torch_npu.npu.set_device(0)
    ok = True
    # seqused_k=100 -> 13 blocks, boundary blk 12, valid_tail=4
    ok &= run_case("bdst_rank3", 100, 3, 13, seed=2048)
    # boundary blk at rank 9 -> boundary_dst=72, window w=64, eScr spans two 8-blocks
    ok &= run_case("bdst_rank9", 100, 9, 13, seed=2049)
    # odd valid_tail (seqused_k=101 -> valid_tail=5)
    ok &= run_case("bdst_odd_tail", 101, 5, 13, seed=2050)
    # full-tail block boundary (valid_tail=8, no fixup expected but rank>0)
    ok &= run_case("bdst_full_tail", 104, 4, 13, seed=2051)
    # larger seq: 3200 blocks -> filtered path, boundary blk at rank 2047
    ok &= run_case("bdst_filtered_tail", 25600, 2047, 3200, seed=2052)
    print("BOUNDARY-RESULT:", "ALL-PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
