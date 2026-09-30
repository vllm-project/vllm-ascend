// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <optional>

namespace vllm_ascend {

inline bool MhcExpandSupported(const at::Tensor& x, int64_t mult)
{
    constexpr int64_t alignment = 16;  // 32-byte DMA blocks for 16-bit dtypes.
    constexpr int64_t streams = 4;  // GLM stream count covered by benchmarks.
    return mult == streams && x.dim() == 2 && !x.requires_grad() &&
        (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
        x.is_contiguous() && x.sym_numel() != 0 && x.sym_size(1) % alignment == 0;
}

}  // namespace vllm_ascend
