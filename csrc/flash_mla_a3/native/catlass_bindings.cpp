// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <string>

namespace {
using at::Tensor;
using Launch = int (*)(uint32_t, void*, void*, void*, void*, void*, void*, void*,
                       void*, void*, void*, void*, void*, void*, void*);

int64_t cubeCores(const Tensor& deviceTensor)
{
    const int device = deviceTensor.get_device();
    thread_local int previous = -1;
    thread_local int64_t cores = 0;
    if (previous != device) {
        const auto status = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &cores);
        TORCH_CHECK(status == ACL_SUCCESS && cores > 0, "A3 Cube core query failed, ACL status ", status);
        previous = device;
    }
    return cores;
}

void* loadKernel(const char* name)
{
    Dl_info libraryInfo{};
    TORCH_CHECK(dladdr(reinterpret_cast<void*>(&loadKernel), &libraryInfo), "Cannot locate CATLASS adapter");
    const std::string path(libraryInfo.dli_fname);
    const std::string kernel = path.substr(0, path.find_last_of('/') + 1) + "libflash_mla_a3_catlass.so";
    void* library = dlopen(kernel.c_str(), RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(library, "Cannot load ", kernel, ": ", dlerror());
    auto launch = dlsym(library, name);
    TORCH_CHECK(launch, "Missing the A3 device-tiling launch entry in ", kernel);
    return launch;
}

bool validSplitCacheLayout(const Tensor& cache, int64_t channels)
{
    // The tiling header stores page strides in elements. Rows within a page
    // remain dense, and the Cube DMA requires 32-byte alignment for BF16.
    constexpr int64_t alignment = 16;
    return cache.dim() == 3 && cache.size(1) == 128 && cache.size(2) == channels &&
           cache.stride(2) == 1 && cache.stride(1) == channels &&
           cache.stride(0) >= 128 * channels && cache.stride(0) <= std::numeric_limits<int32_t>::max() &&
           cache.stride(0) % alignment == 0 && cache.storage_offset() % alignment == 0;
}

Tensor runImpl(const Tensor& q, const Tensor& cache, const Tensor& table, const Tensor& tiling,
               int64_t splits, int64_t heads, const Tensor* qRope, const Tensor* kRope,
               const Tensor* result = nullptr, int64_t groups = 0)
{
    TORCH_CHECK(q.is_contiguous() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 &&
                q.size(2) == (qRope ? 512 : 576) && q.size(1) == (result ? 4 : heads) &&
                (heads == 4 || heads == 16 || (qRope && (heads == 8 || heads == 32))),
                "invalid contiguous BF16 query dimensions for A3 CATLASS");
    const int64_t batches = result ? groups : q.size(0);
    TORCH_CHECK(batches > 0 && batches < 65536, "invalid CATLASS virtual batch");
    TORCH_CHECK(cache.device() == q.device() && cache.scalar_type() == at::kBFloat16 &&
                cache.dim() == 3 && cache.size(0) > 0 && cache.size(1) == 128 && cache.size(2) == (kRope ? 512 : 576),
                "expected BF16 cache[P,128,512 or 576] on the query device");
    if (qRope) {
        TORCH_CHECK(validSplitCacheLayout(cache, 512), "invalid split latent cache page layout");
        TORCH_CHECK(qRope->device() == q.device() && qRope->is_contiguous() && qRope->scalar_type() == at::kBFloat16 &&
                    qRope->dim() == 3 && qRope->size(0) == batches && qRope->size(1) == heads && qRope->size(2) == 64,
                    "invalid split query RoPE tensor");
        TORCH_CHECK(kRope->device() == cache.device() && kRope->scalar_type() == at::kBFloat16 &&
                    validSplitCacheLayout(*kRope, 64) && kRope->size(0) == cache.size(0),
                    "invalid split RoPE cache page layout");
    } else {
        TORCH_CHECK(cache.is_contiguous(), "expected contiguous fused cache");
    }
    TORCH_CHECK(table.device() == q.device() && table.scalar_type() == at::kInt && table.is_contiguous() &&
                table.dim() == 2 && (result ? table.size(0) > 0 : table.size(0) == batches) && table.size(1) > 0,
                "expected contiguous int32 page table[B,M]");
    TORCH_CHECK(tiling.device() == q.device() && tiling.scalar_type() == at::kInt && tiling.is_contiguous() &&
                tiling.numel() >= 24 + 17*batches, "CATLASS tiling must contain a header and one record per batch");
    TORCH_CHECK(splits > 0 && splits <= 64, "invalid split count");
    if (result) {
        TORCH_CHECK(splits >= 2 && heads == 16 && q.size(0) >= 4 && q.size(0) <= 4 * table.size(0) &&
                    result->device() == q.device() && result->is_contiguous() &&
                    result->scalar_type() == at::kBFloat16 && result->dim() == 3 && result->size(2) == 512 &&
                    result->numel() == q.size(0) * 4 * 512 && tiling.numel() == (groups + 1) * 32,
                    "invalid packed CATLASS output or tiling size");
    }
    auto out = result ? *result : at::empty({batches, heads, 512}, q.options());
    if (q.is_meta()) {
        return out;
    }
    const c10::DeviceGuard guard(q.device());
    static const auto launch = reinterpret_cast<Launch>(loadKernel("flash_mla_a3_catlass_launch"));
    const int64_t cores = cubeCores(q);
    constexpr int64_t tile = 65536;
    const auto fp32 = q.options().dtype(at::kFloat);
    auto scores = at::empty({cores, 2, tile}, fp32);
    auto weights = at::empty({cores, 2, tile}, q.options());
    auto product = at::empty({cores, 2, tile}, fp32);
    auto update = at::empty({cores, tile}, fp32);
    // Each split/head owns one 512-channel partial. Allow one final DMA block
    // for vector copies when the number of local attention heads is below eight.
    auto partials = at::empty({batches*heads*splits*512}, fp32);
    auto lse = at::empty({batches*heads*splits + 8}, fp32);
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto* query = static_cast<uint16_t*>(q.data_ptr());
    auto* key = static_cast<uint16_t*>(cache.data_ptr());
    const int status = launch(static_cast<uint32_t>(cores), stream, query,
                              qRope ? qRope->data_ptr() : query + 512, key,
                              kRope ? kRope->data_ptr() : key + 512, table.data_ptr(), out.data_ptr(), scores.data_ptr(),
                              weights.data_ptr(), product.data_ptr(), update.data_ptr(), partials.data_ptr(),
                              lse.data_ptr(), tiling.data_ptr());
    TORCH_CHECK(status == 0, "A3 CATLASS launch failed, status ", status);
    return out;
}
Tensor run(const Tensor& q, const Tensor& cache, const Tensor& table, const Tensor& tiling,
           int64_t splits, int64_t heads)
{
    return runImpl(q, cache, table, tiling, splits, heads, nullptr, nullptr);
}

Tensor runSplit(const Tensor& q, const Tensor& qRope, const Tensor& cache, const Tensor& kRope,
                const Tensor& table, const Tensor& tiling, int64_t splits, int64_t heads)
{
    return runImpl(q, cache, table, tiling, splits, heads, &qRope, &kRope);
}
// Offsets in bytes for the per-core GM buffers used in mla_kernel.cpp. Keep
// the per-request partial buffers proportional to actual capture capacity.
struct GraphScratch {
    int64_t weights, product, update, partials, lse, bytes;
    GraphScratch(int64_t cores, int64_t batches, int64_t heads, int64_t splits, bool balanced = false) {
        TORCH_CHECK(batches > 0 && batches < 65536 && (heads == 4 || heads == 8 || heads == 16 || heads == 32)
                    && splits >= 2 && splits <= 64, "invalid graph decode geometry");
        weights = cores * 32768 * sizeof(float);
        product = weights + cores * 65536 * sizeof(uint16_t);
        update = product + cores * 2 * 65536 * sizeof(float);
        partials = update + cores * 65536 * sizeof(float);
        const int64_t rows = (balanced ? batches + cores - 1 : batches * splits) * heads;
        lse = partials + rows * 512 * sizeof(float);
        bytes = lse + (rows + 8) * sizeof(float);
    }
};

int64_t graphWorkspaceBytes(const Tensor& device, int64_t batches, int64_t heads, int64_t splits)
{
    return GraphScratch(cubeCores(device), batches, heads, splits).bytes;
}

void graphOutImpl(const Tensor& q, const Tensor& qr, const Tensor& k, const Tensor& kr, const Tensor& table,
              const Tensor& lengths, const Tensor& tiling, Tensor workspace, Tensor out, int64_t splits, int64_t heads, const Tensor* schedule)
{
    TORCH_CHECK(q.dim() == 3 && q.size(1) == heads && q.size(2) == 512 && qr.dim() == 3 &&
                qr.size(0) == q.size(0) && qr.size(1) == heads && qr.size(2) == 64 &&
                q.is_contiguous() && qr.is_contiguous(), "invalid split graph queries");
    const auto b = q.size(0);
    const auto h = (heads == 8 || heads == 32) ? 8 : 4;
    for (auto* tensor : {&qr, &k, &kr, &table, &lengths, &tiling}) {
        TORCH_CHECK(tensor->device() == q.device(), "graph input device mismatch");
    }
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && qr.scalar_type() == at::kBFloat16 &&
                k.scalar_type() == at::kBFloat16 && kr.scalar_type() == at::kBFloat16 &&
                validSplitCacheLayout(k, 512) && validSplitCacheLayout(kr, 64) && k.size(0) == kr.size(0) && k.size(0) > 0,
                "invalid graph KV cache layout or dtype");
    TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 && table.size(0) == b && table.size(1) > 0 &&
                table.stride(1) == 1 && table.stride(0) >= table.size(1) && lengths.scalar_type() == at::kInt &&
                lengths.dim() == 1 && lengths.numel() == b && lengths.is_contiguous() &&
                tiling.scalar_type() == at::kInt && tiling.numel() == 32 * (b + 1) && tiling.is_contiguous(),
                "invalid graph decode page table, device lengths or static tiling");
    TORCH_CHECK(out.device() == q.device() && out.scalar_type() == at::kBFloat16 && out.dim() == 3 &&
                out.size(0) == h && out.size(1) == b * (heads / h) && out.size(2) == 512 && out.is_contiguous(),
                "expected contiguous NTD graph output");
    TORCH_CHECK(workspace.device() == q.device() && workspace.scalar_type() == at::kByte &&
                workspace.is_contiguous() && workspace.storage_offset() % 32 == 0, "invalid graph workspace");
    if (schedule) {
        TORCH_CHECK(schedule->device() == q.device() && schedule->scalar_type() == at::kInt &&
                    schedule->dim() == 2 && schedule->size(0) == b + 1 && schedule->size(1) == 4 &&
                    schedule->is_contiguous() && b <= 256, "invalid balanced graph schedule");
    }
    if (q.is_meta()) return;
    const c10::DeviceGuard guard(q.device());
    const auto cores = cubeCores(q);
    const GraphScratch scratch(cores, b, heads, schedule ? cores : splits, schedule != nullptr);
    TORCH_CHECK(workspace.numel() >= scratch.bytes, "insufficient graph workspace");
    using GraphLaunch = int (*)(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                void*, void*, void*, void*);
    auto* base = workspace.data_ptr<uint8_t>();
    const auto stream = c10_npu::getCurrentNPUStream().stream(true);
    int status;
    if (schedule) {
        using BalancedLaunch = int (*)(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                      void*, void*, void*, void*, void*, void*, void*);
        static const auto launch = reinterpret_cast<BalancedLaunch>(loadKernel("flash_mla_a3_catlass_balanced_graph_launch"));
        status = launch(cores, stream, q.data_ptr(), qr.data_ptr(), k.data_ptr(), kr.data_ptr(),
                        table.data_ptr(), out.data_ptr(), base, base + scratch.weights, base + scratch.product,
                        base + scratch.update, base + scratch.partials, base + scratch.lse, tiling.data_ptr(),
                        lengths.data_ptr(), schedule->data_ptr());
    } else {
        static const auto launch = reinterpret_cast<GraphLaunch>(loadKernel("flash_mla_a3_catlass_graph_launch"));
        status = launch(cores, stream, q.data_ptr(), qr.data_ptr(), k.data_ptr(), kr.data_ptr(),
                        table.data_ptr(), out.data_ptr(), base, base + scratch.weights, base + scratch.product,
                        base + scratch.update, base + scratch.partials, base + scratch.lse, tiling.data_ptr(),
                        lengths.data_ptr());
    }
    TORCH_CHECK(status == 0, "A3 graph decode launch failed, status ", status);
}

void graphOut(const Tensor& q, const Tensor& qr, const Tensor& k, const Tensor& kr, const Tensor& table,
              const Tensor& lengths, const Tensor& tiling, Tensor workspace, Tensor out, int64_t splits, int64_t heads)
{
    graphOutImpl(q, qr, k, kr, table, lengths, tiling, workspace, out, splits, heads, nullptr);
}

void balancedGraphOut(const Tensor& q, const Tensor& qr, const Tensor& k, const Tensor& kr, const Tensor& table,
                      const Tensor& lengths, const Tensor& tiling, const Tensor& schedule,
                      Tensor workspace, Tensor out, int64_t heads)
{
    graphOutImpl(q, qr, k, kr, table, lengths, tiling, workspace, out, 0, heads, &schedule);
}

int64_t balancedWorkspaceBytes(const Tensor& table, int64_t batches, int64_t heads)
{
    TORCH_CHECK(batches <= 256, "balanced schedule capacity is 256 requests");
    auto cores = cubeCores(table);
    return GraphScratch(cores, batches, heads, cores, true).bytes;
}

void runDirect(const Tensor& q, const Tensor& cache, const Tensor& table, const Tensor& tiling,
               Tensor out, int64_t splits, int64_t groups)
{
    runImpl(q, cache, table, tiling, splits, 16, nullptr, nullptr, &out, groups);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(flash_mla_a3, m)
{
    m.def("catlass_run(Tensor q, Tensor cache, Tensor table, Tensor tiling, int splits, int heads) -> Tensor");
    m.def("catlass_split(Tensor q, Tensor q_rope, Tensor cache, Tensor k_rope, Tensor table, Tensor tiling, int splits, int heads) -> Tensor");
    m.def("catlass_split_wide(Tensor q, Tensor q_rope, Tensor cache, Tensor k_rope, Tensor table, Tensor tiling, int splits, int heads) -> Tensor");
    m.def("catlass_direct(Tensor q, Tensor cache, Tensor table, Tensor tiling, Tensor(a!) out, int splits, int groups) -> ()");
    m.def("catlass_cores(Tensor q) -> int");
    m.def("catlass_balanced_graph_workspace_bytes(Tensor table, int batches, int heads) -> int");
    m.def("catlass_balanced_graph_out(Tensor q, Tensor qr, Tensor k, Tensor kr, Tensor table, Tensor lengths, Tensor tiling, Tensor schedule, Tensor(a!) workspace, Tensor(b!) out, int heads) -> ()");
    m.def("catlass_graph_out(Tensor q, Tensor qr, Tensor k, Tensor kr, Tensor table, Tensor lengths, Tensor tiling, Tensor(a!) workspace, Tensor(b!) out, int splits, int heads) -> ()");
    m.def("catlass_graph_workspace_bytes(Tensor q, int batches, int heads, int splits) -> int");
}
TORCH_LIBRARY_IMPL(flash_mla_a3, PrivateUse1, m)
{
    m.impl("catlass_balanced_graph_out", TORCH_FN(balancedGraphOut));
    m.impl("catlass_graph_out", TORCH_FN(graphOut));
    m.impl("catlass_direct", TORCH_FN(runDirect));
    m.impl("catlass_run", TORCH_FN(run));
    m.impl("catlass_split", TORCH_FN(runSplit));
    m.impl("catlass_split_wide", TORCH_FN(runSplit));
    m.impl("catlass_cores", TORCH_FN(cubeCores));
    m.impl("catlass_balanced_graph_workspace_bytes", TORCH_FN(balancedWorkspaceBytes));
    m.impl("catlass_graph_workspace_bytes", TORCH_FN(graphWorkspaceBytes));
}
TORCH_LIBRARY_IMPL(flash_mla_a3, Meta, m)
{
    m.impl("catlass_balanced_graph_out", TORCH_FN(balancedGraphOut));
    m.impl("catlass_graph_out", TORCH_FN(graphOut));
    m.impl("catlass_direct", TORCH_FN(runDirect));
    m.impl("catlass_run", TORCH_FN(run));
    m.impl("catlass_split", TORCH_FN(runSplit));
    m.impl("catlass_split_wide", TORCH_FN(runSplit));
}
