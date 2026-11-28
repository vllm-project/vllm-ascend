// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "aclrtlaunch_flash_mla_a3_bf16.h"
#include "aclrtlaunch_flash_mla_a3_fp16.h"
#include "aclrtlaunch_mla_a3_bounds.h"
#include "aclrtlaunch_mla_a3_decode_merge.h"
#include "aclrtlaunch_mla_a3_decode_split.h"
#include "aclrtlaunch_mla_a3_plan.h"

namespace {
using Tensor = at::Tensor;
using OptionalTensor = std::optional<Tensor>;
using ForwardResult = std::tuple<Tensor, Tensor>;
constexpr int64_t kIntMax = std::numeric_limits<int32_t>::max();
constexpr int64_t kMetadataTileRows = 16;

void checkInt(int64_t value, int64_t minimum, const char* name)
{
    TORCH_CHECK(value >= minimum && value <= kIntMax, name, " must be in [", minimum, ", ", kIntMax, "]");
}

void checkBatch(int64_t batches)
{
    TORCH_CHECK(batches > 0 && batches < 65536, "batch size must be in [1, 65535]");
}

void checkTensor(const Tensor& tensor, const Tensor& reference, at::ScalarType dtype, const char* name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on ", reference.device());
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " must have dtype ", dtype);
    TORCH_CHECK(tensor.layout() == at::kStrided && tensor.is_contiguous(), name, " must be contiguous");
    if (tensor.is_privateuseone()) {
        auto format = at_npu::native::custom_ops::get_npu_format(tensor);
        TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                    name, " requires base ND/NCHW storage, got format ", format);
    }
}

void checkVector(const Tensor& tensor, const Tensor& reference, int64_t length, const char* name)
{
    checkTensor(tensor, reference, at::kInt, name);
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == length, name, " must have shape [", length, "]");
}

int64_t vectorCores(int device)
{
    thread_local int previousDevice = -1;
    thread_local int64_t available = 0;
    if (device != previousDevice) {
        int64_t count = 0;
        const aclError status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &count);
        TORCH_CHECK(status == ACL_SUCCESS && count > 0,
                    "Cannot query A3 vector core count on device ", device, ", ACL status ", status);
        available = count;
        previousDevice = device;
    }
    return available;
}

uint32_t blockCount(int device, int64_t work)
{
    return work == 1 ? 1 : static_cast<uint32_t>(std::min(vectorCores(device), work));
}

void checkLaunch(uint32_t status, const char* name)
{
    TORCH_CHECK(status == ACL_SUCCESS, "Could not launch ", name, ", ACL status ", status);
}

Tensor makePlan(const Tensor& kv, const OptionalTensor& cu, const OptionalTensor& used,
                int64_t maxQ, int64_t maxKv)
{
    TORCH_CHECK(kv.dim() == 1, "cache_seqlens must have shape [B]");
    const int64_t batches = kv.size(0);
    checkBatch(batches);
    checkVector(kv, kv, batches, "cache_seqlens");
    if (cu) {
        checkVector(*cu, kv, batches + 1, "cu_seqlens_q");
    }
    if (used) {
        checkVector(*used, kv, batches, "seqused_q");
    }
    checkInt(maxQ, -1, "max_q");
    checkInt(maxKv, -1, "max_kv");
    auto plan = at::empty({batches + 1, 4}, kv.options());
    if (!kv.is_meta()) {
        const c10::DeviceGuard guard(kv.device());
        const uint32_t blocks = blockCount(kv.get_device(), (batches + kMetadataTileRows) / kMetadataTileRows);
        auto stream = c10_npu::getCurrentNPUStream().stream(true);
        checkLaunch(aclrtlaunch_mla_a3_plan(blocks, stream, kv.data_ptr(), cu ? cu->data_ptr() : nullptr,
                                            used ? used->data_ptr() : nullptr, plan.data_ptr(), batches,
                                            maxQ, maxKv, cu.has_value(), used.has_value()), "mla_a3_plan");
    }
    return plan;
}

Tensor makeBounds(const Tensor& plan, int64_t tokens, bool packed, int64_t mode,
                  int64_t left, int64_t right, int64_t capacity)
{
    checkTensor(plan, plan, at::kInt, "plan");
    TORCH_CHECK(plan.dim() == 2 && plan.size(0) >= 2 && plan.size(1) == 4, "plan must have shape [B+1, 4]");
    const int64_t batches = plan.size(0) - 1;
    checkBatch(batches);
    checkInt(tokens, 0, "tokens");
    checkInt(capacity, 0, "capacity");
    TORCH_CHECK(mode == 0 || mode == 3 || mode == 4, "mask_mode must be 0, 3 or 4");
    checkInt(left, -1, "win_left");
    checkInt(right, -1, "win_right");
    TORCH_CHECK(mode == 4 || (left == -1 && right == -1), "windows require mask_mode 4");
    TORCH_CHECK(packed || tokens % batches == 0, "dense query tokens must be a multiple of batch size");
    auto bounds = at::empty({tokens, 3}, plan.options());
    if (!plan.is_meta() && tokens > 0) {
        const c10::DeviceGuard guard(plan.device());
        const uint32_t blocks = blockCount(plan.get_device(), (tokens + kMetadataTileRows - 1) / kMetadataTileRows);
        auto stream = c10_npu::getCurrentNPUStream().stream(true);
        checkLaunch(aclrtlaunch_mla_a3_bounds(blocks, stream, plan.data_ptr(),
                                              bounds.data_ptr(), batches, tokens, packed, mode, left, right,
                                              capacity), "mla_a3_bounds");
    }
    return bounds;
}

ForwardResult forward(const Tensor& q, const Tensor& cache, const Tensor& table, const Tensor& bounds,
                      const OptionalTensor& sinks, int64_t blockSize, double scale, bool writeLse)
{
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 576, "q must have shape [T,H,576]");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "q must be FP16 or BF16");
    checkTensor(q, q, q.scalar_type(), "q");
    checkTensor(cache, q, q.scalar_type(), "cache");
    checkTensor(table, q, at::kInt, "block_table");
    checkTensor(bounds, q, at::kInt, "row_bounds");
    const int64_t rows = q.size(0);
    const int64_t heads = q.size(1);
    checkInt(rows, 0, "T");
    TORCH_CHECK(heads >= 1 && heads <= 128, "H must be in [1, 128]");
    TORCH_CHECK(bounds.dim() == 2 && bounds.size(0) == rows && bounds.size(1) == 3,
                "row_bounds must have shape [T,3]");
    TORCH_CHECK(table.dim() == 2, "block_table must have shape [B,M]");
    const int64_t batches = table.size(0);
    const int64_t width = table.size(1);
    checkBatch(batches);
    checkInt(width, 1, "M");
    TORCH_CHECK(blockSize >= 16 && blockSize <= 1024 && blockSize % 16 == 0,
                "block_size must be divisible by 16 and in [16, 1024]");
    checkInt(width * blockSize, 1, "per-sequence cache capacity");
    TORCH_CHECK((cache.dim() == 2 || cache.dim() == 3) && cache.size(-1) == 576,
                "cache must have shape [P,S,576] or [P*S,576]");
    TORCH_CHECK(cache.dim() == 3 ? cache.size(1) == blockSize : cache.size(0) % blockSize == 0,
                "cache shape is incompatible with block_size");
    const int64_t pages = cache.dim() == 3 ? cache.size(0) : cache.size(0) / blockSize;
    checkInt(pages, 1, "P");
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
                "scale must be a finite FP32 value");
    if (sinks) {
        checkTensor(*sinks, q, at::kFloat, "sinks");
        TORCH_CHECK(sinks->dim() == 1 && sinks->size(0) == heads, "sinks must have shape [H]");
    }
    auto out = at::empty({rows, heads, 512}, q.options());
    auto fp32 = q.options().dtype(at::kFloat);
    auto lse = writeLse ? at::empty({rows, heads}, fp32) : at::empty({0}, fp32);
    if (!q.is_meta() && rows > 0) {
        const c10::DeviceGuard guard(q.device());
        TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 32 == 0 &&
                        reinterpret_cast<uintptr_t>(cache.data_ptr()) % 32 == 0,
                    "q and cache must be 32-byte aligned");
        const uint32_t blocks = blockCount(q.get_device(), rows * heads);
        auto stream = c10_npu::getCurrentNPUStream().stream(true);
        auto launch = q.scalar_type() == at::kHalf ? aclrtlaunch_flash_mla_a3_fp16 : aclrtlaunch_flash_mla_a3_bf16;
        checkLaunch(launch(blocks, stream, q.data_ptr(), cache.data_ptr(), table.data_ptr(), bounds.data_ptr(),
                           sinks ? sinks->data_ptr() : nullptr, out.data_ptr(), writeLse ? lse.data_ptr() : nullptr,
                           rows, heads, batches, pages, blockSize, width, static_cast<float>(scale),
                           sinks.has_value(), writeLse), "flash_mla_a3");
    }
    return {out, lse};
}

ForwardResult decode(const Tensor& q, const Tensor& cache, const Tensor& table, const Tensor& lengths,
                     int64_t queryLength, double scale, bool causal, bool writeLse, int64_t requestedSplits)
{
    constexpr int64_t heads = 4;
    constexpr int64_t pageSize = 128;
    constexpr int64_t splitTileSize = 32;
    constexpr int64_t partialElements = 520;  // Per-split accumulator and softmax statistics, padded to 32 bytes.
    checkTensor(q, q, at::kBFloat16, "q");
    checkTensor(cache, q, at::kBFloat16, "cache");
    checkTensor(table, q, at::kInt, "block_table");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == heads && q.size(2) == 576, "q must have shape [B*S,4,576]");
    TORCH_CHECK(cache.dim() == 3 && cache.size(1) == pageSize && cache.size(2) == 576,
                "cache must have shape [P,128,576]");
    TORCH_CHECK(table.dim() == 2, "block_table must have shape [B,M]");
    const int64_t batches = table.size(0);
    const int64_t width = table.size(1);
    const int64_t pages = cache.size(0);
    checkBatch(batches);
    checkVector(lengths, q, batches, "lengths");
    checkInt(width, 1, "M");
    checkInt(width * pageSize, 1, "per-sequence cache capacity");
    checkInt(pages, 1, "P");
    TORCH_CHECK(queryLength >= 1 && queryLength <= 4, "decode query_len must be in [1, 4]");
    TORCH_CHECK(q.size(0) == batches * queryLength, "q.size(0) must equal batch size times query_len");
    const int64_t rows = q.size(0) * heads;
    checkInt(rows, 1, "number of query heads across all rows");
    const int64_t tileCount = width * (pageSize / splitTileSize);
    TORCH_CHECK(requestedSplits >= 0 && requestedSplits <= tileCount, "splits must be between 0 and ", tileCount);
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
                "scale must be a finite FP32 value");
    auto out = at::empty({q.size(0), heads, 512}, q.options());
    const auto fp32 = q.options().dtype(at::kFloat);
    auto lse = writeLse ? at::empty({q.size(0), heads}, fp32) : at::empty({0}, fp32);
    if (q.is_meta()) {
        return {out, lse};
    }

    const c10::DeviceGuard guard(q.device());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 32 == 0 &&
                    reinterpret_cast<uintptr_t>(cache.data_ptr()) % 32 == 0,
                "q and cache must be 32-byte aligned");
    const int64_t cores = vectorCores(q.get_device());
    const int64_t splits = requestedSplits ? requestedSplits : std::min(tileCount, (cores + rows - 1) / rows);
    checkInt(rows * splits, 1, "number of split jobs");
    Tensor partials;
    if (splits > 1) {
        partials = at::empty({rows, splits, partialElements}, fp32);
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    void* scratch = partials.defined() ? partials.data_ptr() : nullptr;
    void* lsePtr = writeLse ? lse.data_ptr() : nullptr;
    checkLaunch(aclrtlaunch_mla_a3_decode_split(std::min(cores, rows * splits), stream,
                                                q.data_ptr(), cache.data_ptr(), table.data_ptr(), lengths.data_ptr(),
                                                scratch, out.data_ptr(), lsePtr, batches, queryLength, pages, width,
                                                splits, static_cast<float>(scale), causal, writeLse), "mla_a3_decode_split");
    if (splits > 1) {
        checkLaunch(aclrtlaunch_mla_a3_decode_merge(std::min(cores, rows), stream, scratch, out.data_ptr(),
                                                    lsePtr, rows, splits, writeLse), "mla_a3_decode_merge");
    }
    return {out, lse};
}
}  // namespace

TORCH_LIBRARY(flash_mla_a3, m)
{
    m.def("plan(Tensor cache_seqlens, Tensor? cu_seqlens_q, Tensor? seqused_q, int max_q, int max_kv) -> Tensor");
    m.def("bounds(Tensor plan, int tokens, bool packed, int mask_mode, int win_left, int win_right, int capacity) -> Tensor");
    m.def("forward(Tensor q, Tensor cache, Tensor block_table, Tensor row_bounds, Tensor? sinks, int block_size, float scale, bool return_softmax_lse) -> (Tensor, Tensor)");
    m.def("decode(Tensor q, Tensor cache, Tensor table, Tensor lengths, int query_len, float scale, bool causal, bool write_lse, int splits=0) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_mla_a3, PrivateUse1, m)
{
    m.impl("plan", TORCH_FN(makePlan));
    m.impl("bounds", TORCH_FN(makeBounds));
    m.impl("forward", TORCH_FN(forward));
    m.impl("decode", TORCH_FN(decode));
}

TORCH_LIBRARY_IMPL(flash_mla_a3, Meta, m)
{
    m.impl("plan", TORCH_FN(makePlan));
    m.impl("bounds", TORCH_FN(makeBounds));
    m.impl("forward", TORCH_FN(forward));
    m.impl("decode", TORCH_FN(decode));
}
