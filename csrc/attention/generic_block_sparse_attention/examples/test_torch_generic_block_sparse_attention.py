# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

"""GBSA torch example: Metadata -> Attention.

Flow:
  1) npu_generic_block_sparse_attention_metadata(...)  -> metadata[1024]
  2) npu_generic_block_sparse_attention(..., metadata=metadata)

Supported smoke config (regular path):
  layout_q=TND, layout_kv=PAGED_BBND, mask_type=1, block_shape=[1,128], D=128

Runtime env (Ascend950):
  export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/custom_transformer
  # do NOT point ASCEND_CUSTOM_OPP_PATH at $ASCEND_TOOLKIT_HOME/opp (custom AICPU json will not load)
  softmax_precision must be 1 on chip 950
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch_npu  # noqa: F401

# Allow running from repo root without installing the torch_extension package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_TORCH_EXT = os.path.join(_REPO_ROOT, "torch_extension")
if _TORCH_EXT not in sys.path:
    sys.path.insert(0, _TORCH_EXT)

from cann_ops_transformer.ops.generic_block_sparse_attention import (  # noqa: E402
    npu_generic_block_sparse_attention,
)
from cann_ops_transformer.ops.generic_block_sparse_attention_metadata import (  # noqa: E402
    npu_generic_block_sparse_attention_metadata,
)

# Keep aligned with sparse_attention_score_metadata.h
SA_TOTAL_TASK_NUM_INDEX = 4
METADATA_SIZE = 1024

HEAD_DIM = 128
BLOCK_SHAPE = [1, 128]
PAGED_BLOCK_SIZE = 128


def build_tnd_paged_inputs(
    batch: int = 1,
    q_seqlen: int = 4,
    kv_seqlen: int = 256,
    kv_heads: int = 1,
    group_size: int = 4,
    top_k: int = 2,
    dtype: torch.dtype = torch.float16,
):
    num_heads = kv_heads * group_size
    max_blocks = (kv_seqlen + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
    total_q_tokens = batch * q_seqlen
    # blockShapeX == 1 => one Q-block per token
    total_q_blocks = total_q_tokens

    query = torch.randn(total_q_tokens, num_heads, HEAD_DIM, dtype=dtype, device="npu") * 0.02
    key = torch.randn(max_blocks, PAGED_BLOCK_SIZE, kv_heads, HEAD_DIM, dtype=dtype, device="npu") * 0.02
    value = torch.randn(max_blocks, PAGED_BLOCK_SIZE, kv_heads, HEAD_DIM, dtype=dtype, device="npu") * 0.02

    sparse_block_idx = torch.full((kv_heads, total_q_blocks, top_k), -1, dtype=torch.int32, device="npu")
    sparse_block_count = torch.zeros((kv_heads, total_q_blocks), dtype=torch.int32, device="npu")
    for q_block in range(total_q_blocks):
        # Rough causal window: keep up to top_k trailing KV blocks for this Q token.
        history = kv_seqlen - q_seqlen
        visible = history + q_block + 1
        last_kv_block = min(max_blocks - 1, (visible - 1) // PAGED_BLOCK_SIZE)
        count = min(top_k, last_kv_block + 1)
        start = last_kv_block - count + 1
        sparse_block_idx[0, q_block, :count] = torch.arange(start, last_kv_block + 1, dtype=torch.int32, device="npu")
        sparse_block_count[0, q_block] = count

    block_table = torch.arange(max_blocks, dtype=torch.int32, device="npu").view(batch, max_blocks)
    # Equal-batch packed TND: only one batch length here.
    cu_seq_lengths_q = torch.tensor([0, q_seqlen], dtype=torch.int64, device="npu")
    cu_seq_lengths_kv = torch.tensor([0, kv_seqlen], dtype=torch.int64, device="npu")

    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_block_idx": sparse_block_idx,
        "sparse_block_count": sparse_block_count,
        "block_table": block_table,
        "cu_seq_lengths_q": cu_seq_lengths_q,
        "cu_seq_lengths_kv": cu_seq_lengths_kv,
        "num_heads": num_heads,
        "kv_heads": kv_heads,
        "q_seqlen": q_seqlen,
        "kv_seqlen": kv_seqlen,
        "total_q_tokens": total_q_tokens,
    }


def run_metadata(inputs: dict) -> torch.Tensor:
    """Step 1: AICPU metadata -> INT32[1024] schedule table."""
    metadata = npu_generic_block_sparse_attention_metadata(
        inputs["sparse_block_idx"],
        inputs["sparse_block_count"],
        inputs["q_seqlen"],
        inputs["kv_seqlen"],
        inputs["num_heads"],
        inputs["kv_heads"],
        HEAD_DIM,
        BLOCK_SHAPE,
        cu_seq_lengths=inputs["cu_seq_lengths_q"],
        cu_seq_lengths_kv=inputs["cu_seq_lengths_kv"],
        is_packed_gqa=1,
        q_input_layout="TND",
        kv_input_layout="PAGED_BBND",
        mask_type=1,
        quant_type=0,
        softmax_precision=1,
        window_size_left=-1,
        window_size_right=-1,
    )
    torch.npu.synchronize()

    assert metadata.shape == (METADATA_SIZE,), metadata.shape
    assert metadata.dtype == torch.int32
    sa_total_task_num = int(metadata[SA_TOTAL_TASK_NUM_INDEX].item())
    expected_task_num = inputs["total_q_tokens"] * inputs["kv_heads"]
    print(
        f"[metadata] shape={tuple(metadata.shape)}, saTotalTaskNum={sa_total_task_num}, "
        f"expected(no-pad)={expected_task_num}"
    )
    assert sa_total_task_num > 0
    return metadata


def run_attention(inputs: dict, metadata: torch.Tensor):
    """Step 2: main GBSA kernel, consuming metadata (incl. total task num)."""
    attention_out, softmax_lse = npu_generic_block_sparse_attention(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["sparse_block_idx"],
        inputs["sparse_block_count"],
        BLOCK_SHAPE,
        metadata=metadata,
        cu_seq_lengths_q=inputs["cu_seq_lengths_q"],
        cu_seq_lengths_kv=inputs["cu_seq_lengths_kv"],
        block_table=inputs["block_table"],
        is_packed_gqa=1,
        layout_q="TND",
        layout_kv="PAGED_BBND",
        scale_value=1.0 / math.sqrt(HEAD_DIM),
        mask_type=1,
        quant_type=0,
        dst_type_max=0.0,
        softmax_precision=1,
        win_left=-1,
        win_right=-1,
        return_softmax_lse=1,
    )
    torch.npu.synchronize()

    assert attention_out.shape == inputs["query"].shape
    assert attention_out.dtype == inputs["query"].dtype
    assert torch.isfinite(attention_out.float()).all()
    assert softmax_lse.shape == (inputs["query"].shape[0], inputs["query"].shape[1], 1)
    assert softmax_lse.dtype == torch.float32
    print(
        f"[attention] out={tuple(attention_out.shape)}, lse={tuple(softmax_lse.shape)}, "
        f"out_mean={attention_out.float().mean().item():.6f}"
    )
    return attention_out, softmax_lse


def main():
    torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))
    torch.manual_seed(0)

    inputs = build_tnd_paged_inputs()
    print(
        "[case] TND + PAGED_BBND, "
        f"T={inputs['total_q_tokens']}, Nq={inputs['num_heads']}, Nkv={inputs['kv_heads']}, "
        f"q_seqlen={inputs['q_seqlen']}, kv_seqlen={inputs['kv_seqlen']}"
    )

    # 1) metadata
    metadata = run_metadata(inputs)
    # 2) main op
    run_attention(inputs, metadata)

    print("GenericBlockSparseAttention metadata->attention torch example passed.")


if __name__ == "__main__":
    main()
