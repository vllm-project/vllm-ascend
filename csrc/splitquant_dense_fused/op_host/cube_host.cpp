#include "torch_kernel_helper.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"
#include "aclrtlaunch_splitquant_grouped_cube.h"
#include <algorithm>
#include <map>
#include <tuple>
#include <cstdlib>

namespace splitquant {
at::Tensor sparse_cube(const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,int64_t);
at::Tensor splitquant_grouped_cube(const at::Tensor& a, const at::Tensor& w)
{
    if(a.dim()==4)return sparse_cube(a,w,1,w.size(0),0,0);
    TORCH_CHECK(a.dim() == 3 && a.size(2) == 64 && w.dim() == 2,
                "Cube adapter expects packed A=[G,M,64], W=[N,K/2]");
    const int64_t g = a.size(0), m = a.size(1), n = w.size(0), k = w.size(1) * 2;
    TORCH_CHECK(g * 128 == k && m > 0 && n > 0 && n % 16 == 0,
                "Cube adapter requires K=128G and N aligned to 16");
    TORCH_CHECK(a.scalar_type() == at::kByte &&
                (w.scalar_type() == at::kByte || w.scalar_type() == at::kChar) &&
                a.is_contiguous() && w.is_contiguous() && a.device() == w.device(),
                "Cube adapter expects contiguous packed INT4 tensors on one device");
    if (m==1 && !std::getenv("SPLITQUANT_DISABLE_SPARSE")) return sparse_cube(a,w,1,n,0,0);
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "cannot query Ascend platform");
    using Key = std::tuple<int64_t, int64_t, int64_t, int, c10::StreamId>;
    static thread_local std::map<Key, at::Tensor> tiling_cache;
    Key key{m, n, k, a.device().index(),
            c10_npu::getCurrentNPUStream(a.device().index()).id()};
    auto it = tiling_cache.find(key);
    if (it == tiling_cache.end()) {
    matmul_tiling::MatmulApiTiling api(*platform);
    TORCH_CHECK(api.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                             matmul_tiling::DataType::DT_INT4, false) == 0 &&
                api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                             matmul_tiling::DataType::DT_INT4, true) == 0 &&
                api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                             matmul_tiling::DataType::DT_INT32) == 0 &&
                api.SetShape(16, 1024, 128) == 0 && api.SetOrgShape(m, n, k) == 0 &&
                api.SetFixSplit(16, 1024, 128) == 0,
                "Cube adapter failed to configure MatmulApiTiling");
    optiling::TCubeTiling cube{};
    TORCH_CHECK(api.GetTiling(cube) == 0, "Cube adapter failed to generate Cube tiling");
    cube.set_M(16); cube.set_N(1024); cube.set_Ka(128); cube.set_Kb(128);
    auto cpu = at::empty({static_cast<int64_t>(cube.GetDataSize())},
                         at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    cube.SaveToBuffer(cpu.data_ptr(), cube.GetDataSize());
    it = tiling_cache.emplace(key, cpu.to(a.device(), false, true)).first;
    }
    auto tiling = it->second;
    // MoE routing produces many distinct token counts per expert. Caching a
    // [G,M,N] result for every count retains gigabytes over a long PIQA or
    // decode run. The NPU allocator reuses these temporaries safely after the
    // downstream epilogue completes on the same stream, so cache tiling only.
    using WKey=std::tuple<const void*,uint32_t>;
    struct WEntry {at::Tensor source,value;};
    static thread_local std::map<WKey,WEntry> packed_cache;
    WKey wk{w.unsafeGetTensorImpl(),w.is_inference()?0:w.unsafeGetTensorImpl()->version_counter().current_version()};
    auto wi=packed_cache.find(wk);
    if(wi==packed_cache.end())wi=packed_cache.emplace(wk,WEntry{w,w.view({n,g,64}).permute({1,0,2}).contiguous()}).first;
    auto packed=wi->second.value;
    auto partial = at::empty({g, m, n}, a.options().dtype(at::kInt));
    auto workspace = at::empty({static_cast<int64_t>(platform->GetLibApiWorkSpaceSize()) + 4096},
                               a.options().dtype(at::kByte));
    const uint32_t blocks = std::min<int64_t>(platform->GetCoreNumAic(),
        g * ((m + 15) / 16) * ((n + 1023) / 1024));
    uint32_t mu = m, nu = n, ku = k;
    EXEC_KERNEL_CMD(splitquant_grouped_cube, blocks, a, packed, partial,
                    mu, nu, ku, workspace, tiling);
    return partial;
}
} // namespace splitquant
