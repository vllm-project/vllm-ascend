// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "aclrtlaunch_flash_mla_a3_grouped_kernel.h"
namespace {
void run(const at::Tensor &q, const at::Tensor &cache, const at::Tensor &table, const at::Tensor &lengths,
         at::Tensor score, at::Tensor weights, at::Tensor value, at::Tensor stats, int64_t queryLength,
         double scale, bool causal, int64_t group, const c10::optional<at::Tensor> &plan,
         const c10::optional<at::Tensor> &cumulative, const c10::optional<at::Tensor> &used, bool packed,
         const c10::optional<at::Tensor> &mapping) {
    TORCH_CHECK(table.dim() == 2 && table.scalar_type() == at::kInt);
    auto batch = table.size(0), width = table.size(1);
    TORCH_CHECK(batch > 0 && batch < 65536 && width > 0 && width < (1LL << 24) && group >= 1 && group <= 3 &&
                queryLength >= 1 && queryLength <= 4);
    const int64_t groups = (width + group - 1) / group;
    const int64_t slots = mapping ? batch * queryLength : batch;
    TORCH_CHECK(q.is_privateuseone() && q.sizes() == at::IntArrayRef({batch * queryLength, 4, 576}) &&
                q.scalar_type() == at::kBFloat16);
    TORCH_CHECK(cache.dim() == 3 && cache.size(0) > 0 && cache.size(1) == 128 && cache.size(2) == 576 &&
                cache.scalar_type() == at::kBFloat16);
    TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == batch && lengths.scalar_type() == at::kInt);
    TORCH_CHECK(score.sizes() == at::IntArrayRef({batch, width, 16, 128}) &&
                score.scalar_type() == at::kFloat);
    TORCH_CHECK(weights.sizes() == score.sizes() && weights.scalar_type() == at::kBFloat16);
    TORCH_CHECK(value.sizes() == at::IntArrayRef({slots, groups, 16, 512}) &&
                value.scalar_type() == at::kFloat);
    TORCH_CHECK(stats.sizes() == at::IntArrayRef({slots, groups, 2, 16}) &&
                stats.scalar_type() == at::kFloat);
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
                "scale must be finite FP32");
    TORCH_CHECK(cache.size(0) <= std::numeric_limits<int32_t>::max(),
                "physical page count exceeds INT32_MAX");
    for (const auto &tensor : {q, cache, table, lengths, score, weights, value, stats}) {
        TORCH_CHECK(tensor.device() == q.device() && tensor.is_contiguous(),
                    "all grouped buffers must be contiguous on the same NPU");
        const auto format = at_npu::native::custom_ops::get_npu_format(tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    "grouped FlashMLA requires linear storage");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 32 == 0,
                    "grouped buffers require 32-byte aligned addresses");
    }
    const bool dynamic = plan || cumulative || used || packed;
    TORCH_CHECK(!dynamic || batch == 1 || mapping,
                "multi-batch dynamic grouped decode requires a row mapping");
    TORCH_CHECK(!packed || plan || cumulative, "packed grouped decode requires a plan or cumulative offsets");
    auto optionalData = [&](const c10::optional<at::Tensor> &tensor, int64_t count) -> void * {
        if (!tensor)
            return nullptr;
        TORCH_CHECK(tensor->device() == q.device() && tensor->scalar_type() == at::kInt &&
                        tensor->is_contiguous() && tensor->numel() == count,
                    "invalid grouped sequence descriptor");
        auto format = at_npu::native::custom_ops::get_npu_format(*tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW, "expected linear length storage");
        return tensor->data_ptr();
    };
    auto planPtr = optionalData(plan, (batch + 1) * 4);
    auto cumulativePtr = optionalData(cumulative, batch + 1);
    auto usedPtr = optionalData(used, batch);
    auto mapPtr = optionalData(mapping, batch * 8);
    TORCH_CHECK(!mapping || (mapping->sizes() == at::IntArrayRef({batch, 8}) &&
                             reinterpret_cast<uintptr_t>(mapPtr) % 32 == 0));
    const uint32_t flags =
        (plan ? 1u : 0u) | (packed ? 2u : 0u) | (used ? 4u : 0u) | (dynamic ? 8u : 0u) | (mapping ? 16u : 0u);
    c10::DeviceGuard guard(q.device());
    thread_local int previousDevice = -1;
    thread_local int64_t coreCount = 0;
    if (q.get_device() != previousDevice) {
        TORCH_CHECK(aclrtGetDeviceInfo(q.get_device(), ACL_DEV_ATTR_CUBE_CORE_NUM, &coreCount) ==
                        ACL_SUCCESS &&
                    coreCount > 0);
        previousDevice = q.get_device();
    }
    const auto blocks = std::min(batch * groups, coreCount);
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto status = aclrtlaunch_flash_mla_a3_grouped_kernel(
        blocks, stream, q.data_ptr(), cache.data_ptr(), table.data_ptr(), lengths.data_ptr(),
        score.data_ptr(), weights.data_ptr(), value.data_ptr(), stats.data_ptr(), batch, width, cache.size(0),
        queryLength, static_cast<float>(scale), causal, group, blocks, planPtr, cumulativePtr, usedPtr, flags,
        mapPtr);
    TORCH_CHECK(status == ACL_SUCCESS, "A3 grouped MIX launch status ", status);
}
} // namespace
TORCH_LIBRARY(flash_mla_a3_grouped, m) {
    m.def("run(Tensor q, Tensor cache, Tensor table, Tensor lengths, Tensor(a!) scores, Tensor(b!) weights, "
          "Tensor(c!) partial, Tensor(d!) stats, int q_len, float scale, bool causal, int group, Tensor? "
          "plan=None, Tensor? cumulative=None, Tensor? used=None, bool packed=False, Tensor(e!)? "
          "mapping=None) -> ()");
}
TORCH_LIBRARY_IMPL(flash_mla_a3_grouped, PrivateUse1, m) { m.impl("run", TORCH_FN(run)); }
