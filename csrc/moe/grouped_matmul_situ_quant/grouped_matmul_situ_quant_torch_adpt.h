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
 *   ND list     : per-expert tensors composed once per call, then converted to
 *                 the kernel's native stacked NZ byte stream.
 *   NZ list     : passed as an ACL dynamic input; the kernel dereferences the
 *                 per-expert address table directly, with no cast or cat.
 *   weightScale : ND always (official contract); stacked and TensorList forms
 *                 are both passed through the ACL dynamic-input address table.
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
#include <mutex>
#include <tuple>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

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

constexpr uint32_t SITU_MAIN_BLOCK_N2_DEV = 64;

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

// Shared core: dynamic ACL inputs encode either one stacked tensor or E
// independent per-expert tensors as an address table. Tensor descriptors and
// the address table are created for the launch; the tensor storage is reused.
std::tuple<at::Tensor, at::Tensor> RunV2Core(const at::Tensor &x, const at::Tensor &xScale,
                                            const std::vector<at::Tensor> &weights,
                                            const std::vector<at::Tensor> &weightScales,
                                            const at::Tensor &groupList, int64_t e, int64_t n, double beta,
                                            double linearBeta, int64_t groupListType)
{
    const int64_t mCap = x.sizes()[0];
    const int64_t k = x.sizes()[1];
    const int64_t n2 = n / 2;

    TORCH_CHECK(e > 0 && !weights.empty(), "weight must be non-empty");
    TORCH_CHECK(weights.size() == 1 || weights.size() == static_cast<size_t>(e),
                "weight must contain one stacked tensor or E per-expert tensors");
    TORCH_CHECK(weightScales.size() == weights.size(), "weightScale must use the same stacked/list form as weight");
    TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64 (MX scale pairing)");
    TORCH_CHECK(n2 % SITU_MAIN_BLOCK_N2_DEV == 0, "N/2 must be a multiple of 64");
    TORCH_CHECK(groupList.is_privateuseone(), "groupList must be a NPU device tensor (graph-capturable contract)");
    TORCH_CHECK(groupList.scalar_type() == at::kLong, "groupList must be int64");
    TORCH_CHECK(groupList.numel() == e, "groupList length must equal E");

    auto dev = at::device(at::kPrivateUse1);
    at::Tensor y = at::empty({mCap, n2}, dev.dtype(at::kFloat8_e4m3fn));
    at::Tensor yScale = at::empty({mCap, (n2 + 63) / 64, 2}, dev.dtype(at::kFloat8_e8m0fnu)); // = (M, ceil(N/128), 2), golden 同形

    // An EP rank can legitimately receive no routed tokens. Preserve the
    // inferred empty output shapes and avoid launching a kernel with empty
    // storage; all metadata validation above still applies to this path.
    if (mCap == 0) {
        return std::make_tuple(y, yScale);
    }

    at::TensorList weightList(weights);
    at::TensorList weightScaleList(weightScales);
    EXEC_NPU_CMD(aclnnGroupedMatmulSituQuant, x, xScale, weightList,
                 weightScaleList, groupList, groupListType, beta, linearBeta,
                 y, yScale);
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
    return RunV2Core(x, xScale, std::vector<at::Tensor>{wNz}, std::vector<at::Tensor>{weightScale}, groupList,
                     e, n, beta, linearBeta, groupListType);
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

    return RunV2Core(x, xScale, std::vector<at::Tensor>{weight}, std::vector<at::Tensor>{weightScale}, groupList,
                     e, n, beta, linearBeta, groupListType);
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
    return RunV2Core(x, xScale, std::vector<at::Tensor>{wNz}, std::vector<at::Tensor>{wScaleStacked}, groupList,
                     e, n, beta, linearBeta, groupListType);
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

    TORCH_CHECK(weightScale.size() == weight.size(), "weightScale list length must equal weight list length");
    for (const auto &scale : weightScale) {
        ValidateWScaleNumel(scale, 1, k, n);
    }
    return RunV2Core(x, xScale, weight, weightScale, groupList, e, n, beta, linearBeta, groupListType);
}

} // namespace ascend_kernel
