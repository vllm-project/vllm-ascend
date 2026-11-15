/*
 * Copyright (c) 2026 SeaGaruda
 * Licensed under the Apache License, Version 2.0.
 *
 * Torch bindings for TopKPerRow, mirroring the two entry points that vLLM's
 * CUDA backend exposes (`torch.ops._C.top_k_per_row_prefill/_decode`) so that
 * vllm-ascend can register them as drop-in replacements.
 */
#ifndef VLLM_ASCEND_TOP_K_PER_ROW_TORCH_ADPT_H
#define VLLM_ASCEND_TOP_K_PER_ROW_TORCH_ADPT_H

namespace vllm_ascend {

namespace detail {
inline at::Tensor ensure_row_major(const at::Tensor& t, const char* what)
{
    if (t.stride(-1) == 1 && t.is_contiguous()) {
        return t;
    }
    TORCH_CHECK(t.scalar_type() != at::kFloat || t.is_contiguous(),
                what, " must be contiguous for the Ascend TopKPerRow kernel");
    return t.contiguous();
}

inline void check_common(const at::Tensor& logits, const at::Tensor& indices, int64_t topk)
{
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D (num_rows, width)");
    TORCH_CHECK(logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kHalf ||
                    logits.scalar_type() == at::kBFloat16,
                "logits must be float32/float16/bfloat16");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == logits.size(0) && indices.size(1) == topk,
                "indices must be (num_rows, topk)");
    TORCH_CHECK(topk > 0 && topk <= 2048, "topk must be in (0, 2048]");
}
}  // namespace detail

/**
 * Prefill variant: row r selects within [cu_seqlen_ks[r], cu_seqlen_ke[r]).
 * `stride0`/`stride1` are accepted for signature compatibility with the CUDA
 * kernel; the Ascend kernel indexes rows by their logical width.
 */
inline void top_k_per_row_prefill(const at::Tensor& logits, const at::Tensor& cu_seqlen_ks,
                                  const at::Tensor& cu_seqlen_ke, at::Tensor& indices,
                                  int64_t num_rows, int64_t stride0, int64_t stride1, int64_t topk)
{
    detail::check_common(logits, indices, topk);
    TORCH_CHECK(num_rows == logits.size(0), "num_rows mismatch");
    TORCH_CHECK(cu_seqlen_ks.scalar_type() == at::kInt && cu_seqlen_ke.scalar_type() == at::kInt,
                "cu_seqlen_ks/cu_seqlen_ke must be int32");
    TORCH_CHECK(cu_seqlen_ks.numel() == num_rows && cu_seqlen_ke.numel() == num_rows,
                "cu_seqlen_ks/cu_seqlen_ke must have num_rows elements");
    (void)stride0;
    (void)stride1;
    const at::Tensor l = detail::ensure_row_major(logits, "logits");
    EXEC_NPU_CMD(aclnnTopKPerRow, l, cu_seqlen_ks, cu_seqlen_ke, topk, indices);
}

/**
 * Decode variant: every row selects within [0, row_end) where
 * row_end = clamp(seq_lens - next_n + 1 + arange(next_n), min=0), flattened the
 * same way as vLLM's `_row_ends` helper.
 */
inline void top_k_per_row_decode(const at::Tensor& logits, int64_t next_n, const at::Tensor& seq_lens,
                                 at::Tensor& indices, int64_t num_rows, int64_t stride0,
                                 int64_t stride1, int64_t topk)
{
    detail::check_common(logits, indices, topk);
    TORCH_CHECK(num_rows == logits.size(0), "num_rows mismatch");
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
    (void)stride0;
    (void)stride1;

    const at::Tensor l = detail::ensure_row_major(logits, "logits");
    at::Tensor row_starts;
    at::Tensor row_ends;
    if (seq_lens.numel() == num_rows) {
        row_starts = at::zeros({num_rows}, seq_lens.options());
        row_ends = seq_lens.reshape({num_rows});
    } else {
        const int64_t batch = seq_lens.numel() / next_n;
        auto offsets = at::arange(next_n, seq_lens.options());
        row_ends = (seq_lens.reshape({batch, 1}) - next_n + 1 + offsets.reshape({1, next_n}))
                       .clamp_min(0)
                       .reshape({batch * next_n});
        row_starts = at::zeros({batch * next_n}, seq_lens.options());
        TORCH_CHECK(batch * next_n == num_rows, "seq_lens/next_n do not match num_rows");
    }
    EXEC_NPU_CMD(aclnnTopKPerRow, l, row_starts, row_ends, topk, indices);
}

}  // namespace vllm_ascend
#endif  // VLLM_ASCEND_TOP_K_PER_ROW_TORCH_ADPT_H
