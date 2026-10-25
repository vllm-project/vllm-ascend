// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#pragma once
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace vllm_ascend {
void moe_counts_cumsum_impl(void* stream, void* input, void* output, uint32_t count);

at::Tensor moe_counts_cumsum(const at::Tensor& counts)
{
    TORCH_CHECK(counts.device().type() == c10::DeviceType::PrivateUse1,
                "moe_counts_cumsum expects an NPU tensor");
    TORCH_CHECK(counts.scalar_type() == at::kLong && counts.dim() == 1 && counts.is_contiguous(),
                "moe_counts_cumsum expects contiguous one-dimensional INT64 counts");
    TORCH_CHECK(counts.numel() > 0 && counts.numel() <= 4096,
                "moe_counts_cumsum supports 1..4096 experts");
    const c10_npu::NPUGuard guard(counts.device());
    at::Tensor output = at::empty_like(counts);
    const uint32_t count = static_cast<uint32_t>(counts.numel());
    // Launch through the existing task queue without draining pending work.
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand command;
    command.Name("moe_counts_cumsum");
    // Capture launch values before enqueueing, as the other direct-kernel
    // adapters do. Tensor destruction must stay outside the queue callback.
    void* input_ptr = counts.data_ptr();
    void* output_ptr = output.data_ptr();
    command.SetCustomHandler([input_ptr, output_ptr, count, stream]() -> int {
        moe_counts_cumsum_impl(stream, input_ptr, output_ptr, count);
        return 0;
    });
    command.Run();
    return output;
}

at::Tensor moe_counts_cumsum_meta(const at::Tensor& counts)
{
    return at::empty_like(counts);
}
}  // namespace vllm_ascend
