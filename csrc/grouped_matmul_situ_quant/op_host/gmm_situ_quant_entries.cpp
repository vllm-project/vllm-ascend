/*
 * gmm_situ_quant_v2 — host for the V2-aligned dual entries (P0 SE round).
 *
 * Mirrors the official aclnnGroupedMatmulSwigluQuantWeightNzV2 API layer:
 * TWO entry names sharing ONE implementation (host tiling + the verified
 * gmm_situ_vcv_dev fused kernel), differing only in the weight format
 * contract enforced at the entry layer:
 *
 *   torch.ops._C_ascend.grouped_matmul_situ_quant(x, weight(ND), weightScale,
 *       weightAssistMatrix?, bias?, xScale, smoothScale?, groupList,
 *       dequantMode, dequantDtype, quantMode, groupListType,
 *       tuningConfigOptional?, beta, linearBeta) -> (output, outputScale)
 *
 *   torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz(...)  # NZ weight
 *
 * plus `.list` overloads of both names accepting per-expert TensorLists.
 *
 * Weight dispatch (four forms, §3.2 of the P0 task brief):
 *   ND  stacked : (E, N, K/2) fp4x2 packed, contiguous — converted to the
 *                 NZ byte stream at the entry layer via the production
 *                 aclnn format cast (npu_format_cast 29, the same transform
 *                 the official V2 framework applies internally for its ND
 *                 FP4 weight entry; conversion tax is honest and reported).
 *   NZ  stacked : format-29 storage (any logical view: (E,N,K/2), the
 *                 transposed (E,K/2,N) view, or the canonical 5D
 *                 [E,K/32,N/16,16,32]) — fed to the kernel byte-direct,
 *                 zero conversion (the fused kernel's native form).
 *   ND/NZ list  : per-expert tensors composed with torch-native device ops
 *                 (at::cat of byte views; NZ elements are format-cast back to
 *                 ND first, cat, then a single stacked ND->NZ cast — byte
 *                 identity guaranteed by the proven cast round-trip).
 *   weightScale : ND always (official contract); stacked Tensor used
 *                 directly, TensorList gathered by the same stage kernel.
 *
 * Parameter behavior matrix (§3.3):
 *   bias / smoothScale        : None/undefined -> skipped; real value ->
 *                               TORCH_CHECK unsupported (explicit, never a
 *                               silent wrong result).
 *   weightAssistMatrix        : accepted and ignored (one-shot warning);
 *                               the vendored vf_nz addressing path does not
 *                               consume the NZ assist matrix.
 *   tuningConfigOptional      : accepted and ignored (own tiling policy:
 *                               baseM=128 hits the kbL1Size=512 fast path).
 *   dequantMode/dequantDtype/quantMode : only the MX A8W4 combo is
 *                               implemented (= the Kimi w4a8 / A5 verified
 *                               scenario); other values error out.
 *                               Accepted: dequantMode=1 (MX joint data+scale
 *                               antiquant), dequantDtype=0 (BF16
 *                               intermediate), quantMode=1 (dynamic MX
 *                               quant, FP8-E4M3 out + E8M0 scale).
 *   groupList                 : device int64 (E,); type0 cumsum / type1
 *                               counts, decoded by the in-kernel preamble
 *                               (gmm_situ_vcv_dev mechanism, graph-safe).
 *
 * Outputs are typed (output: float8_e4m3fn (M, N/2); outputScale:
 * float8_e8m0fnu (M, ceil((N/2)/64), 2)) — shapes aligned with the
 * golden split chain, so downstream sees the V2 naming/typing contract.
 */
#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include <map>
#include <mutex>
#include <tuple>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "aclrtlaunch_gmm_situ_vcv_dev.h"

namespace ascend_kernel {

namespace {

// ACL format / dtype constants (torch_npu int enums, verified on-device)
constexpr int64_t kAclFormatNd = 2;
constexpr int64_t kAclCastTargetFractalNz = 29;      // aclnn format-cast target id
constexpr int64_t kNpuFormatFractalNz = 29;          // torch_npu Format::FRACTAL_NZ tag
constexpr int64_t kNpuFormatFractalNzC0_16 = 50;     // torch_npu Format::FRACTAL_NZ_C0_16 tag (cast result)
constexpr int64_t kAclDtFloat8E4m3Fn = 292;
constexpr int64_t kAclDtFloat4E2m1FnX2 = 296;

// Accepted enum matrix (see file header)
constexpr int64_t kDequantModeMx = 1;
constexpr int64_t kDequantDtypeBf16 = 0;
constexpr int64_t kQuantModeMx = 1;

constexpr uint32_t SITU_BASE_M_DEV = 128;
constexpr uint32_t SITU_MAIN_BLOCK_N2_DEV = 64;

// ---- tiling blob layout (must match op_kernel/gmsq_vcv_controller.h) ----
struct SituTilingHeaderHostDev { // 64B
    uint32_t coreNum;
    uint32_t activeCount; // dev entry: E (full expert count)
    uint32_t kSize;
    uint32_t nSize;
    uint32_t baseM;
    uint32_t mainBlockSize;
    uint32_t firstTailBlockSize;
    uint32_t reserved; // dev entry: groupListType
    uint64_t mainBlockCount;
    uint64_t firstTailBlockCount;
    float beta;
    float invBeta;
    float linearBeta;
    float invLinearBeta;
};

using TilingKey = std::tuple<int64_t, int64_t, int64_t, double, double, int64_t>; // k, n, e, beta, lbeta, glType

at::Tensor GetStaticTilingDev(int64_t k, int64_t n, int64_t e, double beta, double linearBeta, int64_t glType,
                              uint32_t coreNum)
{
    static std::mutex mu;
    static std::map<TilingKey, at::Tensor> cache;
    TilingKey key{k, n, e, beta, linearBeta, glType};
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }
    const int64_t n2 = n / 2;
    SituTilingHeaderHostDev hdr = {};
    hdr.coreNum = coreNum;
    hdr.activeCount = static_cast<uint32_t>(e);
    hdr.kSize = static_cast<uint32_t>(k);
    hdr.nSize = static_cast<uint32_t>(n);
    hdr.baseM = SITU_BASE_M_DEV;
    hdr.mainBlockSize = SITU_MAIN_BLOCK_N2_DEV;
    hdr.firstTailBlockSize = 0;
    hdr.reserved = static_cast<uint32_t>(glType);
    hdr.mainBlockCount = static_cast<uint64_t>(n2 / SITU_MAIN_BLOCK_N2_DEV);
    hdr.firstTailBlockCount = 0;
    hdr.beta = static_cast<float>(beta);
    hdr.invBeta = 1.0f / static_cast<float>(beta);
    hdr.linearBeta = static_cast<float>(linearBeta);
    hdr.invLinearBeta = 1.0f / static_cast<float>(linearBeta);

    auto blobCpu = at::empty({static_cast<int64_t>(sizeof(hdr))},
                             at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
    memcpy(blobCpu.data_ptr<uint8_t>(), &hdr, sizeof(hdr));
    auto tilingNpu = blobCpu.to(at::device(at::kPrivateUse1), /*non_blocking=*/false, /*copy=*/true);
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    cache.emplace(key, tilingNpu);
    return tilingNpu;
}

bool IsNullOrUndefined(const std::optional<at::Tensor> &t)
{
    return !t.has_value() || !t->defined() || t->numel() == 0;
}

// §3.3 parameter behavior matrix — shared by all four entry functions.
void ValidateV2Params(const std::optional<at::Tensor> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
                      const std::optional<at::Tensor> &smoothScale, int64_t dequantMode, int64_t dequantDtype,
                      int64_t quantMode, int64_t groupListType)
{
    TORCH_CHECK(IsNullOrUndefined(bias),
                "gmm_situ_quant: bias is not supported by this SiTU variant (pass None); non-empty bias would be "
                "silently wrong, so it is rejected explicitly");
    TORCH_CHECK(IsNullOrUndefined(smoothScale),
                "gmm_situ_quant: smoothScale is not supported (pass None); rejected explicitly");
    if (!IsNullOrUndefined(weightAssistMatrix)) {
        static std::once_flag once;
        std::call_once(once, [] {
            fprintf(stderr,
                    "[gmm_situ_quant] note: weightAssistMatrix accepted but ignored (vendored vf_nz addressing does "
                    "not consume the NZ assist matrix)\n");
        });
    }
    TORCH_CHECK(dequantMode == kDequantModeMx,
                "gmm_situ_quant: only dequantMode=1 (MX joint data+scale antiquant, A8W4) is implemented, got ",
                dequantMode);
    TORCH_CHECK(dequantDtype == kDequantDtypeBf16,
                "gmm_situ_quant: only dequantDtype=0 (BF16 intermediate GEMM result) is implemented, got ",
                dequantDtype);
    TORCH_CHECK(quantMode == kQuantModeMx,
                "gmm_situ_quant: only quantMode=1 (dynamic MX quant, FP8-E4M3 out + E8M0 scale) is implemented, got ",
                quantMode);
    TORCH_CHECK(groupListType == 0 || groupListType == 1, "gmm_situ_quant: groupListType must be 0 or 1, got ",
                groupListType);
    // tuningConfigOptional: accepted and ignored (own tiling policy).
}

// Compose per-expert plain (non-internal-format) tensors into one stacked
// byte buffer with a single at::cat device op.
at::Tensor CatTensorList(const std::vector<at::Tensor> &list)
{
    const int64_t e = static_cast<int64_t>(list.size());
    TORCH_CHECK(e > 0, "tensor list must not be empty");
    std::vector<at::Tensor> byteViews;
    byteViews.reserve(static_cast<size_t>(e));
    for (const auto &t : list) {
        TORCH_CHECK(t.is_privateuseone(), "tensor list element must be a NPU device tensor");
        TORCH_CHECK(t.is_contiguous(), "tensor list element must be contiguous");
        byteViews.push_back(t.view(at::kByte).reshape({-1}));
    }
    return at::cat(byteViews, 0);
}

// ND stacked -> NZ byte stream via the production aclnn format cast.
at::Tensor CastNdToNz(const at::Tensor &w)
{
    return at_npu::native::custom_ops::npu_format_cast(w, kAclCastTargetFractalNz,
                                                                kAclDtFloat8E4m3Fn, kAclDtFloat4E2m1FnX2);
}

// NZ byte stream -> ND logical bytes (inverse of the production cast).
at::Tensor CastNzToNd(const at::Tensor &w)
{
    return at_npu::native::custom_ops::npu_format_cast(w, kAclFormatNd, kAclDtFloat8E4m3Fn,
                                                       kAclDtFloat4E2m1FnX2);
}

// Shared core: run the verified fused kernel on an op-ready stacked NZ byte
// stream + stacked weight scale, device group_list. Returns typed outputs.
std::tuple<at::Tensor, at::Tensor> RunV2Core(const at::Tensor &x, const at::Tensor &xScale, const at::Tensor &wNzStacked,
                                  const at::Tensor &wScaleStacked, const at::Tensor &groupList, double beta,
                                  double linearBeta, int64_t groupListType)
{
    const int64_t mCap = x.sizes()[0];
    const int64_t k = x.sizes()[1];
    const int64_t e = wNzStacked.numel() == 0 ? 0 : wNzStacked.sizes()[0];
    TORCH_CHECK(wNzStacked.dim() >= 1 && e > 0, "weight must be non-empty");
    const int64_t n = wNzStacked.numel() / (e * (k / 2));
    const int64_t n2 = n / 2;

    TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64 (MX scale pairing)");
    TORCH_CHECK(n2 % SITU_MAIN_BLOCK_N2_DEV == 0, "N/2 must be a multiple of 64");
    TORCH_CHECK(groupList.is_privateuseone(), "groupList must be a NPU device tensor (graph-capturable contract)");
    TORCH_CHECK(groupList.scalar_type() == at::kLong, "groupList must be int64");
    TORCH_CHECK(groupList.numel() == e, "groupList length must equal E");

    const int64_t scaleRowBytes = (n2 + 63) / 64 * 2;
    auto dev = at::device(at::kPrivateUse1);
    at::Tensor y = at::empty({mCap, n2}, dev.dtype(at::kFloat8_e4m3fn));
    at::Tensor yScale = at::empty({mCap, (n2 + 63) / 64, 2}, dev.dtype(at::kFloat8_e8m0fnu)); // = (M, ceil(N/128), 2), golden 同形

    // An EP rank can legitimately receive no routed tokens. Preserve the
    // inferred empty output shapes and avoid launching a kernel with empty
    // storage; all metadata validation above still applies to this path.
    if (mCap == 0) {
        return std::make_tuple(y, yScale);
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int32_t aicNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAic());
    const int64_t sysWs = static_cast<int64_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    at::Tensor workspace = at::empty({sysWs > 0 ? sysWs : 32}, dev.dtype(at::kByte));

    auto tilingNpu = GetStaticTilingDev(k, n, e, beta, linearBeta, groupListType,
                                        static_cast<uint32_t>(aicNum > 0 ? aicNum : 1));
    uint32_t blockDim = static_cast<uint32_t>(aicNum > 0 ? aicNum : 1);

    EXEC_KERNEL_CMD(gmm_situ_vcv_dev, blockDim, x, xScale, wNzStacked, wScaleStacked, groupList, y, yScale,
                    workspace, tilingNpu);
    (void)scaleRowBytes;
    return std::make_tuple(y, yScale);
}

void ValidateXScales(const at::Tensor &x, const at::Tensor &xScale)
{
    TORCH_CHECK(x.is_privateuseone(), "x must be a NPU device tensor");
    TORCH_CHECK(xScale.is_privateuseone(), "xScale must be a NPU device tensor");
    const int64_t mCap = x.sizes()[0];
    const int64_t k = x.sizes()[1];
    TORCH_CHECK(xScale.numel() >= mCap * (k / 32), "xScale numel inconsistent with (M_cap, K)");
}

void ValidateWScaleNumel(const at::Tensor &wScale, int64_t e, int64_t k, int64_t n)
{
    TORCH_CHECK(wScale.is_privateuseone(), "weightScale must be a NPU device tensor");
    const int64_t kb = (k + 63) / 64;
    TORCH_CHECK(wScale.numel() == e * kb * n * 2, "weightScale numel must be E*ceil(K/64)*N*2, got ", wScale.numel(),
                ", expected ", e * kb * n * 2);
}

} // namespace

// ---- entry 1: gmm_situ_quant (ND weight, stacked) ----
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nd(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weightScale,
    const std::optional<at::Tensor> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta)
{
    ValidateV2Params(weightAssistMatrix, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);

    TORCH_CHECK(weight.is_privateuseone(), "weight must be a NPU device tensor");
    const int64_t e = weight.sizes()[0];
    const int64_t k = x.sizes()[1];
    const int64_t n = weight.numel() / (e * (k / 2));
    TORCH_CHECK(weight.numel() == e * n * (k / 2), "weight must be (E, N, K/2) ND packed");
    TORCH_CHECK(at_npu::native::custom_ops::get_npu_format(weight) == kAclFormatNd,
                "gmm_situ_quant expects ND-format weight (use gmm_situ_quant_weight_nz for FRACTAL_NZ weight)");
    TORCH_CHECK(weight.is_contiguous(), "ND weight must be contiguous");
    ValidateWScaleNumel(weightScale, e, k, n);

    at::Tensor wNz = CastNdToNz(weight);
    return RunV2Core(x, xScale, wNz, weightScale, groupList, beta, linearBeta, groupListType);
}

// ---- entry 2: gmm_situ_quant_weight_nz (NZ weight, stacked) ----
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nz(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weightScale,
    const std::optional<at::Tensor> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta)
{
    ValidateV2Params(weightAssistMatrix, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);

    TORCH_CHECK(weight.is_privateuseone(), "weight must be a NPU device tensor");
    const int64_t e = weight.sizes()[0];
    const int64_t k = x.sizes()[1];
    const int64_t n = weight.numel() / (e * (k / 2));
    TORCH_CHECK(weight.numel() == e * n * (k / 2), "weight numel inconsistent with (E, N, K/2)");
    const int64_t wFmt = at_npu::native::custom_ops::get_npu_format(weight);
    TORCH_CHECK(wFmt == kNpuFormatFractalNzC0_16 || wFmt == kNpuFormatFractalNz,
                "gmm_situ_quant_weight_nz expects FRACTAL_NZ weight (format tag FRACTAL_NZ_C0_16/FRACTAL_NZ; use "
                "gmm_situ_quant for ND weight), got tag ", wFmt);
    ValidateWScaleNumel(weightScale, e, k, n);

    return RunV2Core(x, xScale, weight, weightScale, groupList, beta, linearBeta, groupListType);
}

// ---- entry 1.list: gmm_situ_quant.list (ND weight TensorList) ----
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nd_list(
    const at::Tensor &x, const std::vector<at::Tensor> &weight, const std::vector<at::Tensor> &weightScale,
    const std::optional<std::vector<at::Tensor>> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta)
{
    std::optional<at::Tensor> assist =
        weightAssistMatrix.has_value() && !weightAssistMatrix->empty() ? std::optional<at::Tensor>((*weightAssistMatrix)[0])
                                                                      : std::nullopt;
    ValidateV2Params(assist, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);

    const int64_t e = static_cast<int64_t>(weight.size());
    const int64_t k = x.sizes()[1];
    const int64_t perExpert = weight[0].numel();
    TORCH_CHECK(perExpert % (k / 2) == 0, "each weight element must be (N, K/2) ND packed");
    const int64_t n = perExpert / (k / 2);
    for (const auto &w : weight) {
        TORCH_CHECK(at_npu::native::custom_ops::get_npu_format(w) == kAclFormatNd,
                    "gmm_situ_quant.list expects ND-format weight elements");
    }

    at::Tensor wStackedNd = CatTensorList(weight).view({e, n, k / 2});
    at::Tensor wScaleStacked = CatTensorList(weightScale);
    at::Tensor wNz = CastNdToNz(wStackedNd);
    return RunV2Core(x, xScale, wNz, wScaleStacked, groupList, beta, linearBeta, groupListType);
}

// ---- entry 2.list: gmm_situ_quant_weight_nz.list (NZ weight TensorList) ----
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nz_list(
    const at::Tensor &x, const std::vector<at::Tensor> &weight, const std::vector<at::Tensor> &weightScale,
    const std::optional<std::vector<at::Tensor>> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta)
{
    std::optional<at::Tensor> assist =
        weightAssistMatrix.has_value() && !weightAssistMatrix->empty() ? std::optional<at::Tensor>((*weightAssistMatrix)[0])
                                                                      : std::nullopt;
    ValidateV2Params(assist, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);

    const int64_t e = static_cast<int64_t>(weight.size());
    const int64_t k = x.sizes()[1];
    const int64_t perExpert = weight[0].numel();
    TORCH_CHECK(perExpert % (k / 2) == 0, "each weight element numel inconsistent with (N, K/2)");
    const int64_t n = perExpert / (k / 2);
    for (const auto &w : weight) {
        const int64_t wFmt = at_npu::native::custom_ops::get_npu_format(w);
        TORCH_CHECK(wFmt == kNpuFormatFractalNzC0_16 || wFmt == kNpuFormatFractalNz,
                    "gmm_situ_quant_weight_nz.list expects FRACTAL_NZ weight elements");
    }

    // NZ 元素先 format-cast 回 ND（字节恒等：round-trip 已板上验证），cat 成
    // stacked ND 后单次统一 cast 到 NZ 字节流 — 与 stacked NZ 入口同一 kernel 输入。
    std::vector<at::Tensor> ndViews;
    ndViews.reserve(weight.size());
    for (const auto &w : weight) {
        ndViews.push_back(CastNzToNd(w));
    }
    at::Tensor wStackedNd = at::cat(ndViews, 0);
    at::Tensor wScaleStacked = CatTensorList(weightScale);
    at::Tensor wNz = CastNdToNz(wStackedNd);
    return RunV2Core(x, xScale, wNz, wScaleStacked, groupList, beta, linearBeta, groupListType);
}

} // namespace ascend_kernel
