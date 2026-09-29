// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
namespace vllm_ascend {
at::Tensor npu_mhc_expand(const at::Tensor& x, int64_t mult)
{
    TORCH_CHECK(x.dim() == 2, "npu_mhc_expand expects a rank-2 tensor");
    TORCH_CHECK(x.is_contiguous(), "npu_mhc_expand expects contiguous input");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "npu_mhc_expand supports float16 and bfloat16");
    TORCH_CHECK(mult > 0, "npu_mhc_expand mult must be positive");
    TORCH_CHECK(x.numel() <= std::numeric_limits<int64_t>::max() / 2 / mult,
                "npu_mhc_expand output byte size overflows int64");
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto y = at::empty({x.size(0), mult, x.size(1)}, x.options());
    if (x.numel() != 0) {
        EXEC_NPU_CMD(aclnnVllmMhcExpand, x, mult, y);
    }
    return y;
}
}  // namespace vllm_ascend
