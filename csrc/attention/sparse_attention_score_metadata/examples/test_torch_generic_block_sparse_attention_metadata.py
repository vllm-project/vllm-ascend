# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

import torch
import torch_npu

from cann_ops_transformer.ops.generic_block_sparse_attention_metadata import (
    npu_generic_block_sparse_attention_metadata,
)


def run_bsnd_case():
    block_counts = torch.tensor([[[12, 3]]], dtype=torch.int32, device="npu")
    sparse_block_idx = torch.full((1, 1, 2, 12), -1, dtype=torch.int32, device="npu")
    sparse_block_idx[0, 0, 0, :12] = torch.arange(12, dtype=torch.int32, device="npu")
    sparse_block_idx[0, 0, 1, :3] = torch.arange(3, dtype=torch.int32, device="npu")
    seq_used_q = torch.tensor([2], dtype=torch.int32, device="npu")

    metadata = npu_generic_block_sparse_attention_metadata(
        sparse_block_idx,
        block_counts,
        2,
        2048,
        4,
        1,
        128,
        [1, 128],
        seq_used_q=seq_used_q,
        q_input_layout="BSND",
        kv_input_layout="BSND",
    )
    torch_npu.npu.synchronize()
    assert metadata.shape == (1024,)
    assert metadata.dtype == torch.int32


def run_tnd_case():
    block_counts = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32, device="npu"
    )
    sparse_block_idx = torch.full((1, 8, 8), -1, dtype=torch.int32, device="npu")
    for q_block, block_count in enumerate(block_counts[0].cpu().tolist()):
        sparse_block_idx[0, q_block, :block_count] = torch.arange(
            block_count, dtype=torch.int32, device="npu"
        )
    cu_seq_lengths = torch.tensor([0, 4, 8], dtype=torch.int64, device="npu")
    seq_used_q = torch.tensor([2, 3], dtype=torch.int32, device="npu")

    metadata = torch_npu.npu_generic_block_sparse_attention_metadata(
        sparse_block_idx,
        block_counts,
        4,
        2048,
        4,
        1,
        128,
        [1, 128],
        cu_seq_lengths=cu_seq_lengths,
        seq_used_q=seq_used_q,
        q_input_layout="TND",
        kv_input_layout="BSND",
    )
    torch_npu.npu.synchronize()
    assert metadata.shape == (1024,)
    assert metadata.dtype == torch.int32


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    run_bsnd_case()
    run_tnd_case()
    torch_npu.npu.synchronize()
    print("GenericBlockSparseAttentionMetadata torch_npu examples passed.")
