// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <limits>
#include "decode.h"

namespace {
using at::Tensor;
constexpr int64_t kPage = 128;
constexpr int64_t kFragment = 32;
constexpr int64_t kNope = 512;
constexpr int64_t kRope = 64;
constexpr int64_t kWidth = kNope + kRope;
constexpr int64_t kHeads = 96;
constexpr auto kFp8 = at::ScalarType::Float8_e4m3fn;

uint8_t* bytes(const Tensor& x)
{
    return static_cast<uint8_t*>(x.data_ptr());
}

void checkPages(const Tensor& pages, int64_t width)
{
    TORCH_CHECK(pages.scalar_type() == kFp8 && pages.dim() == 5 && pages.size(1) == 1 &&
                pages.size(2) == width / kFragment && pages.size(3) == kPage && pages.size(4) == kFragment &&
                pages.stride(4) == 1 && pages.stride(3) == kFragment && pages.stride(2) == kPage * kFragment &&
                pages.stride(0) >= kPage * width, "expected FP8 PA_NZ pages with contiguous page contents");
}

void writeCache(const Tensor& latent, const Tensor& rope, const Tensor& latentPages, const Tensor& ropePages,
                const Tensor& slots)
{
    checkPages(latentPages, kNope);
    checkPages(ropePages, kRope);
    const auto device = latentPages.device();
    auto rows = slots.numel();
    TORCH_CHECK(slots.dim() == 1 && slots.is_contiguous() && (slots.scalar_type() == at::kLong ||
                slots.scalar_type() == at::kInt) && rows <= std::numeric_limits<uint32_t>::max(), "invalid KV slots");
    auto checkSource = [&](const Tensor& source, int64_t width) {
        TORCH_CHECK(source.scalar_type() == kFp8 && source.dim() == 3 && source.size(0) == rows &&
                    source.size(1) == 1 && source.size(2) == width && source.stride(2) == 1 &&
                    source.stride(0) >= width && source.device() == device, "invalid FP8 KV source rows");
    };
    checkSource(latent, kNope);
    checkSource(rope, kRope);
    TORCH_CHECK(ropePages.device() == device && slots.device() == device && ropePages.size(0) == latentPages.size(0) &&
                ropePages.stride(0) == latentPages.stride(0), "incompatible KV pages or slots");
    if (rows == 0) {
        return;
    }
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream();
    uint32_t blocks = std::min<int64_t>(64, rows);
    auto k = bytes(latent), r = bytes(rope), kc = bytes(latentPages), rc = bytes(ropePages), indices = bytes(slots);
    auto wide = slots.scalar_type() == at::kLong;
    auto pages = latentPages.size(0), ks = latent.stride(0), rs = rope.stride(0), ps = latentPages.stride(0);
    at_npu::native::OpCommand::RunOpApiV2("QuantFlashMlaNzWrite", [=]() -> int {
        launch_quant_flash_mla_cache_write(stream.stream(false), blocks, k, r, kc, rc, indices, wide,
                                           rows, pages, ks, rs, ps);
        return 0;
    });
}

Tensor decode(const Tensor& q, const Tensor& kv, const Tensor& qScale, const Tensor& kvScale, const Tensor& table,
              const Tensor& lengths, const Tensor& cu, const Tensor& used, const std::optional<Tensor>& mask,
              const Tensor& schedule, double scale)
{
    checkPages(kv, kWidth);
    const auto device = q.device();
    TORCH_CHECK(q.scalar_type() == kFp8 && q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kWidth &&
                q.is_contiguous() && q.size(0) > 0 && q.size(0) <= std::numeric_limits<uint32_t>::max() &&
                kv.device() == device, "expected contiguous FP8 TND queries with 96 heads and PA_NZ KV");
    auto checkInput = [&](const Tensor& t, at::ScalarType dtype, at::IntArrayRef size) {
        TORCH_CHECK(t.scalar_type() == dtype && t.sizes() == size && t.is_contiguous() && t.device() == device,
                    "invalid QuantFlashMLA decode input, expected ", size);
    };
    int64_t rows = lengths.numel();
    TORCH_CHECK(table.dim() == 2 && rows > 0 && rows <= 65535 && table.size(1) > 0 &&
                table.size(1) <= std::numeric_limits<uint32_t>::max() / kPage, "invalid MLA page table");
    checkInput(qScale, at::kFloat, {q.size(0), kHeads});
    checkInput(kvScale, at::kFloat, {1});
    checkInput(table, at::kInt, {rows, table.size(1)});
    checkInput(lengths, at::kInt, {rows});
    checkInput(cu, at::kInt, {rows + 1});
    checkInput(used, at::kInt, {rows});
    if (mask.has_value()) {
        checkInput(*mask, at::kChar, {2048, 2048});
    }
    TORCH_CHECK(schedule.dim() == 2, "expected two QuantFlashMLA metadata planes");
    checkInput(schedule, at::kInt, {2, schedule.size(1)});
    c10_npu::NPUGuard guard(device);
    int64_t cores = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(device.index(), ACL_DEV_ATTR_CUBE_CORE_NUM, &cores) == ACL_SUCCESS &&
                cores > 0 && cores <= 128 && schedule.size(1) >= 16 + cores * 3 * 16, "invalid MLA core schedule");
    QuantMlaShape shape{static_cast<uint32_t>(rows), static_cast<uint32_t>(q.size(0)), kHeads,
                        static_cast<uint32_t>(table.size(1)), static_cast<uint32_t>(cores),
                        static_cast<uint64_t>(kv.stride(0)), static_cast<float>(scale)};
    // On dav-3510 GetUserWorkspace may reserve a 16 MiB system prefix ahead of the FD records.
    constexpr int64_t systemBytes = 16 * 1024 * 1024;
    const int64_t partialBytes = cores * 2 * 64 * (kNope + 2 * 8) * sizeof(float);
    auto workspace = at::empty({systemBytes + partialBytes}, q.options().dtype(at::kByte));
    auto output = at::empty({kHeads, q.size(0), kNope}, q.options().dtype(at::kBFloat16));
    auto stream = c10_npu::getCurrentNPUStream();
    auto qp = bytes(q), kp = bytes(kv), qs = bytes(qScale), ks = bytes(kvScale), bt = bytes(table);
    auto lens = bytes(lengths), offsets = bytes(cu), valid = bytes(used), maskPtr = mask ? bytes(*mask) : nullptr;
    auto meta = bytes(schedule), out = bytes(output), work = bytes(workspace);
    at_npu::native::OpCommand::RunOpApiV2("QuantFlashMlaNz", [=]() -> int {
        launch_quant_flash_mla_nz(stream.stream(false), qp, kp, qs, ks, bt, lens, offsets, valid, maskPtr,
                                  meta, out, work, shape);
        return 0;
    });
    return output;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("quant_flash_mla_nz_write(Tensor latent, Tensor rope, Tensor(a!) latent_pages, Tensor(b!) rope_pages, Tensor slots) -> ()");
    m.def("quant_flash_mla_nz(Tensor q, Tensor kv, Tensor q_scale, Tensor kv_scale, Tensor table, Tensor lengths, Tensor cu, Tensor used, Tensor? mask, Tensor schedule, float scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("quant_flash_mla_nz_write", &writeCache);
    m.impl("quant_flash_mla_nz", &decode);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, m) {
    m.impl("quant_flash_mla_nz_write", [](const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&) {});
    m.impl("quant_flash_mla_nz", [](const Tensor& q, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
             const Tensor&, const Tensor&, const Tensor&, const std::optional<Tensor>&, const Tensor&, double) {
        return at::empty_symint({q.sym_size(1), q.sym_size(0), kNope}, q.options().dtype(at::kBFloat16));
    });
}
