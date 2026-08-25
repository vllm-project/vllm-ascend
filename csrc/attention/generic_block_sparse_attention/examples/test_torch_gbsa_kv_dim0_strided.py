# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Contiguous vs fused PAGED_BBND KV cache (k0 v0 k1 v1 ...) for GBSA.

from __future__ import annotations

import math
import os
import sys

import torch
import torch_npu

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
from test_torch_generic_block_sparse_attention import (  # noqa: E402
    BLOCK_SHAPE,
    HEAD_DIM,
    build_tnd_paged_inputs,
)


def make_interleaved_kv_cache(key_c: torch.Tensor, value_c: torch.Tensor):
    """Fused page cache: k0 v0 k1 v1 ...  K/V views share storage, dim0 stride = 2 pages.

    Logical shape stays [P, Bs, N, D]. Page interior is contiguous; consecutive K (or V)
    pages skip the other tensor's page, so only dim0 is non-contiguous.
    """
    assert key_c.shape == value_c.shape
    p, bs, n, d = key_c.shape
    page = bs * n * d
    stride0 = 2 * page
    storage = torch.empty(p * stride0, dtype=key_c.dtype, device=key_c.device)
    k_flat = key_c.reshape(p, page)
    v_flat = value_c.reshape(p, page)
    for i in range(p):
        storage[i * stride0 : i * stride0 + page].copy_(k_flat[i])
        storage[i * stride0 + page : i * stride0 + 2 * page].copy_(v_flat[i])

    inner = (n * d, d, 1)
    key_v = storage.as_strided(size=(p, bs, n, d), stride=(stride0,) + inner)
    value_v = storage[page:].as_strided(size=(p, bs, n, d), stride=(stride0,) + inner)
    assert not key_v.is_contiguous() and not value_v.is_contiguous()
    assert key_v.stride() == (stride0,) + inner
    assert value_v.stride() == (stride0,) + inner
    assert torch.equal(key_v, key_c)
    assert torch.equal(value_v, value_c)
    return key_v, value_v


def run_once(inputs: dict, key: torch.Tensor, value: torch.Tensor, metadata: torch.Tensor):
    out, lse = npu_generic_block_sparse_attention(
        inputs["query"],
        key,
        value,
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
    return out, lse


def main():
    torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", "0")))
    torch.manual_seed(0)

    inputs = build_tnd_paged_inputs()
    key_c = inputs["key"].contiguous()
    value_c = inputs["value"].contiguous()
    print(f"[contig] key.stride={key_c.stride()} contiguous={key_c.is_contiguous()}")

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

    out_c, lse_c = run_once(inputs, key_c, value_c, metadata)
    print(f"[contig] out_mean={out_c.float().mean().item():.6f}")

    key_s, value_s = make_interleaved_kv_cache(key_c, value_c)
    print(f"[interleaved k0v0k1v1] key.stride={key_s.stride()} value.stride={value_s.stride()}")

    out_s, lse_s = run_once(inputs, key_s, value_s, metadata)
    print(f"[interleaved] out_mean={out_s.float().mean().item():.6f}")

    torch.testing.assert_close(out_s, out_c, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse_s, lse_c, atol=1e-3, rtol=1e-3)
    print("PASS: interleaved fused KV cache matches contiguous K/V.")


if __name__ == "__main__":
    main()
