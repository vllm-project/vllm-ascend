/*
 * gmm_situ_quant_v2 — host for the V2-aligned dual entries.
 *
 * Mirrors the official aclnnGroupedMatmulSwigluQuantWeightNzV2 API layer:
 * TWO entry names sharing ONE implementation, differing only in the weight
 * format contract enforced at the entry layer:
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
 * Two SoC paths share the same prototypes (see op_host/*_def.cpp):
 *   Ascend950 (arch35)  MX A8W4 : x=FP8, weight=FP4(NZ), weightScale=E8M0,
 *                                 y=FP8-E4M3, yScale=E8M0.
 *   Atlas A2/A3 (arch32) A8W4 MSD: x=INT8, weight=INT4, weightScale=UINT64,
 *                                 y=INT8, yScale=FLOAT(per-token).
 * 4bit 权重在 torch 侧一律按整数打包传入：Ascend950 是 uint8 承载 packed FP4
 * （1 字节 2 个），A2/A3 是 int32 承载 packed INT4（1 个 int32 装 8 个，torch 无
 * uint64 时 scale 用 int64）。A2/A3 的 int32/int64 描述符改写由 op_api 的 V2 入口
 * （op_host/op_api/aclnn_grouped_matmul_situ_quant_v2.cpp）完成，本层不感知。
 * The path is selected here by x's dtype; the kernel entry file is selected by
 * the op prototype's per-SoC opFile.value.
 *
 * Parameter behavior matrix:
 *   bias / smoothScale        : None/undefined -> skipped; real value ->
 *                               TORCH_CHECK unsupported (explicit, never a
 *                               silent wrong result).
 *   weightAssistMatrix        : A2/A3 consumes the real matrix; Ascend950 does
 *                               not read it, but the DYNAMIC prototype slot
 *                               rejects an empty list, so a [16] FLOAT32
 *                               placeholder is supplied for that path.
 *   tuningConfigOptional      : passed through; tiling applies its own policy.
 *   dequantMode/dequantDtype/quantMode : only the A8W4 combo is implemented, and
 *                               the accepted values differ by path:
 *                                 Ascend950 MX : dequantMode=1, quantMode=1;
 *                                 A2/A3 MSD    : dequantMode=0, quantMode=0.
 *                               dequantDtype must be 0 on both paths; any other
 *                               value errors out.
 *   groupList                 : device int64 (E,); type0 cumsum / type1 counts.
 *
 * Ascend950 outputs are (M, N/2) fp8-e4m3fn + (M, ceil((N/2)/64), 2) e8m0;
 * A2/A3 outputs are (M, N/2) int8 + (M,) float — shapes aligned with the golden
 * split chain, so downstream sees the V2 naming/typing contract.
 */
#include <cmath>
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
constexpr int64_t kDequantModeMx = 1;   // Ascend950 MX：data+scale 联合反量化（per-group）
constexpr int64_t kDequantModeMsd = 0;  // A2/A3 MSD：per-channel 反量化
constexpr int64_t kDequantDtypeBf16 = 0;
constexpr int64_t kQuantModeMx = 1;     // Ascend950 MX 量化
constexpr int64_t kQuantModeMsd = 0;    // A2/A3 per-token 量化
constexpr int64_t kQuantDtypeDefault = 0;

constexpr uint32_t SITU_MAIN_BLOCK_N2_DEV = 64;

bool IsNullOrUndefined(const std::optional<at::Tensor> &t)
{
    return !t.has_value() || !t->defined() || t->numel() == 0;
}

// A2/A3 的 A8W4 MSD 输入是 INT8，Ascend950 的 MX 输入是 FP8。
bool IsA8W4Msd(const at::Tensor &x)
{
    return x.scalar_type() == at::kChar;
}

// §3.3 parameter behavior matrix — shared by all four entry functions.
// 反量化 / 量化模式按通路固定（与官方 A8W4 样例、A5 调用方一致）：
//   Ascend950 MX（x=FP8）  : dequantMode=1（per-group）、quantMode=1；
//   A2/A3 MSD（x=INT8）    : dequantMode=0（per-channel）、quantMode=0（per-token）。
void ValidateV2Params(const at::Tensor &x, const std::optional<at::Tensor> &bias,
                      const std::optional<at::Tensor> &smoothScale, int64_t dequantMode, int64_t dequantDtype,
                      int64_t quantMode, int64_t groupListType)
{
    const bool isA8W4Msd = IsA8W4Msd(x);
    TORCH_CHECK(IsNullOrUndefined(bias),
                "gmm_situ_quant: bias is not supported by this SiTU variant (pass None); non-empty bias would be "
                "silently wrong, so it is rejected explicitly");
    TORCH_CHECK(IsNullOrUndefined(smoothScale),
                "gmm_situ_quant: smoothScale is not supported (pass None); rejected explicitly");
    TORCH_CHECK(dequantMode == (isA8W4Msd ? kDequantModeMsd : kDequantModeMx),
                "gmm_situ_quant: A2/A3 A8W4 MSD requires dequantMode=0 (per-channel), Ascend950 MX requires "
                "dequantMode=1 (per-group), got ",
                dequantMode);
    TORCH_CHECK(dequantDtype == kDequantDtypeBf16,
                "gmm_situ_quant: only dequantDtype=0 (BF16 intermediate GEMM result) is implemented, got ",
                dequantDtype);
    TORCH_CHECK(quantMode == (isA8W4Msd ? kQuantModeMsd : kQuantModeMx),
                "gmm_situ_quant: A2/A3 A8W4 MSD requires quantMode=0 (per-token), Ascend950 MX requires quantMode=1, "
                "got ",
                quantMode);
    TORCH_CHECK(groupListType == 0 || groupListType == 1, "gmm_situ_quant: groupListType must be 0 or 1, got ",
                groupListType);
    // tuningConfigOptional: passed through (own tiling policy).
}

// 非 list 入口的原型只能收一个 assist 张量，而 kernel 侧统一按 DYNAMIC 地址表读取：
// 单张量形态对应 [E, N] 堆叠矩阵，这里补成单元素列表即可。
std::optional<std::vector<at::Tensor>> AsTensorList(const std::optional<at::Tensor> &t)
{
    if (IsNullOrUndefined(t)) {
        return std::nullopt;
    }
    return std::vector<at::Tensor>{*t};
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

// 4bit 权重的 N 推断。packed 权重固定是“2 个 4bit 占 1 字节”的密度（Ascend950 的 uint8
// 承载 packed FP4、A2/A3 的 int32 承载 packed INT4 都一样），所以用总 bit 数反推 N 与具体
// 封装方式和 ND/NZ 排布都无关。e 传 1 表示“单个专家的权重”。
int64_t InferPackedWeightN(const at::Tensor &weight, int64_t e, int64_t k, const char *name)
{
    TORCH_CHECK(e > 0 && k > 0, name, ": E and K must be positive");
    const int64_t totalBits = weight.numel() * weight.element_size() * 8;
    TORCH_CHECK(totalBits % (4 * e * k) == 0, name, ": numel is not consistent with an (E, N, K) 4bit weight");
    return totalBits / (4 * e * k);
}

// Shared core: dynamic ACL inputs encode either one stacked tensor or E
// independent per-expert tensors as an address table. Tensor descriptors and
// the address table are created for the launch; the tensor storage is reused.
std::tuple<at::Tensor, at::Tensor> RunV2Core(const at::Tensor &x, const at::Tensor &xScale,
                                            const std::vector<at::Tensor> &weights,
                                            const std::vector<at::Tensor> &weightScales,
                                            const at::Tensor &groupList, int64_t e, int64_t n,
                                            const std::optional<std::vector<at::Tensor>> &weightAssistMatrix,
                                            const std::optional<at::Tensor> &bias,
                                            const std::optional<at::Tensor> &smoothScale, int64_t groupListType,
                                            const std::optional<std::vector<int64_t>> &tuningConfigOptional,
                                            double beta, double linearBeta)
{
    const int64_t mCap = x.sizes()[0];
    const int64_t k = x.sizes()[1];
    const int64_t n2 = n / 2;
    const bool isA8W4Msd = IsA8W4Msd(x);

    TORCH_CHECK(e > 0 && !weights.empty(), "weight must be non-empty");
    TORCH_CHECK(weights.size() == 1 || weights.size() == static_cast<size_t>(e),
                "weight must contain one stacked tensor or E per-expert tensors");
    TORCH_CHECK(weightScales.size() == weights.size(), "weightScale must use the same stacked/list form as weight");
    TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64");
    TORCH_CHECK(n2 % SITU_MAIN_BLOCK_N2_DEV == 0, "N/2 must be a multiple of 64");
    TORCH_CHECK(groupList.is_privateuseone(), "groupList must be a NPU device tensor (graph-capturable contract)");
    TORCH_CHECK(groupList.scalar_type() == at::kLong, "groupList must be int64");
    TORCH_CHECK(groupList.dim() == 1 && groupList.is_contiguous(), "groupList must be a contiguous vector");
    TORCH_CHECK(groupList.numel() == e, "groupList length must equal E");
    TORCH_CHECK(groupList.device() == x.device(), "groupList and x must be on the same device");
    TORCH_CHECK(std::isfinite(beta) && beta > 0.0 && std::isfinite(linearBeta),
                "SiTU requires finite beta > 0 and finite linear_beta");
    if (isA8W4Msd) {
        for (size_t i = 0; i < weights.size(); ++i) {
            TORCH_CHECK(weights[i].scalar_type() == at::kInt && weights[i].is_contiguous(),
                        "MSD weight must be contiguous packed INT32");
            TORCH_CHECK(weights[i].device() == x.device() && weightScales[i].device() == x.device(),
                        "MSD weights, scales and x must be on the same device");
            TORCH_CHECK(weightScales[i].scalar_type() == at::kLong && weightScales[i].is_contiguous(),
                        "MSD weight_scale must be contiguous INT64 carrying UINT64 bits");
            const int64_t localExperts = weights.size() == 1 ? e : 1;
            TORCH_CHECK(InferPackedWeightN(weights[i], localExperts, k, "weight") == n,
                        "MSD weights must have the same logical width");
            TORCH_CHECK(weightScales[i].numel() == localExperts * n,
                        "MSD weight_scale must contain one scale per expert/output column");
        }
        if (weightAssistMatrix.has_value()) {
            TORCH_CHECK(weightAssistMatrix->size() == weights.size(),
                        "MSD assist and weight lists must have equal lengths");
            for (size_t i = 0; i < weights.size(); ++i) {
                const auto &assist = (*weightAssistMatrix)[i];
                TORCH_CHECK(assist.scalar_type() == at::kFloat && assist.is_contiguous() &&
                            assist.device() == x.device() && assist.numel() == weightScales[i].numel(),
                            "MSD assist must be contiguous FP32 with the same device and size as weight_scale");
            }
        }
    }

    auto dev = x.options();
    at::Tensor y;
    at::Tensor yScale;
    if (isA8W4Msd) {
        y = at::empty({mCap, n2}, dev.dtype(at::kChar));
        yScale = at::empty({mCap}, dev.dtype(at::kFloat));
    } else {
        y = at::empty({mCap, n2}, dev.dtype(at::kFloat8_e4m3fn));
        yScale = at::empty({mCap, (n2 + 63) / 64, 2}, dev.dtype(at::kFloat8_e8m0fnu));
    }

    // An EP rank can legitimately receive no routed tokens. Preserve the
    // inferred empty output shapes and avoid launching a kernel with empty
    // storage; all metadata validation above still applies to this path.
    if (mCap == 0) {
        return std::make_tuple(y, yScale);
    }

    at::TensorList weightList(weights);
    at::TensorList weightScaleList(weightScales);

    // weight_assist_matrix 在原型里是 DYNAMIC 槽位，GE 侧不接受空列表。
    // A2/A3 两种形态都支持：不传时 tiling 按 numel == 1 认出占位，改由 kernel 在每个专家块
    // 末尾打一行全 -8 的 int4，让 cube 顺手算出 cAux（省掉一次 O(E*K*N) 的离线展开）；
    // 传了就用真实矩阵（单张量形态是 [E, N]，多张量形态是逐专家 [N]）。
    // Ascend950 不读它，按 op 定义补一份 [16] FLOAT32 占位。
    std::vector<at::Tensor> assistHolder;
    if (weightAssistMatrix.has_value() && !weightAssistMatrix->empty()) {
        assistHolder = *weightAssistMatrix;
        // kernel 侧按 isSingleTensor 决定读法：单张量形态按 [E, N] 偏移，逐专家形态走
        // GetTensorAddr(groupIdx)。isSingleTensor 由 weight 的堆叠/列表形态决定，assist 必须同形。
        TORCH_CHECK(assistHolder.size() == weights.size(),
                    "weightAssistMatrix must use the same stacked/list form as weight, got ", assistHolder.size(),
                    " assist tensors vs ", weights.size(), " weight tensors");
    } else if (isA8W4Msd) {
        // 占位必须是 numel == 1：真正的补偿矩阵是 [N] 或 [E, N]，N 至少 16 对齐，
        // tiling 就靠「只有 1 个元素」判定调用方没给、kernel 自算。
        assistHolder.push_back(at::empty({1}, dev.dtype(at::kFloat)));
    } else {
        assistHolder.push_back(at::empty({16}, dev.dtype(at::kFloat)));
    }
    at::TensorList assistList(assistHolder);

    const std::vector<int64_t> tuning = tuningConfigOptional.has_value() && !tuningConfigOptional->empty()
                                            ? *tuningConfigOptional
                                            : std::vector<int64_t>{0};
    at::IntArrayRef tuningRef(tuning);
    const bool transposeWeight = false;
    const bool activateLeft = true;
    // 反量化 / 量化模式按通路取固定值（调用方传入的同类值已由 ValidateV2Params 校验过）。
    const int64_t dequantMode = isA8W4Msd ? kDequantModeMsd : kDequantModeMx;
    const int64_t quantMode = isA8W4Msd ? kQuantModeMsd : kQuantModeMx;

    // A2/A3 的 int32 打包 int4 权重由 V2 入口先改写成 INT4 描述符（见
    // op_host/op_api/aclnn_grouped_matmul_situ_quant_v2.cpp）；Ascend950 的 FP4 权重
    // 不需要改写，保持原有入口不变。
    if (isA8W4Msd) {
        EXEC_NPU_CMD(aclnnGroupedMatmulSituQuantV2, x, xScale, weightList, weightScaleList, groupList, assistList,
                     bias, smoothScale, groupListType, beta, linearBeta, dequantMode, kDequantDtypeBf16,
                     quantMode, kQuantDtypeDefault, transposeWeight, tuningRef, activateLeft, y, yScale);
    } else {
        EXEC_NPU_CMD(aclnnGroupedMatmulSituQuant, x, xScale, weightList, weightScaleList, groupList, assistList, bias,
                     smoothScale, groupListType, beta, linearBeta, dequantMode, kDequantDtypeBf16, quantMode,
                     kQuantDtypeDefault, transposeWeight, tuningRef, activateLeft, y, yScale);
    }
    return std::make_tuple(y, yScale);
}

void ValidateXScales(const at::Tensor &x, const at::Tensor &xScale)
{
    TORCH_CHECK(x.is_privateuseone(), "x must be a NPU device tensor");
    TORCH_CHECK(xScale.is_privateuseone(), "xScale must be a NPU device tensor");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be a contiguous [M, K] tensor");
    TORCH_CHECK(xScale.device() == x.device() && xScale.is_contiguous(),
                "x_scale must be contiguous and on the same device as x");
    const int64_t mCap = x.sizes()[0];
    const int64_t k = x.sizes()[1];
    if (IsA8W4Msd(x)) {
        // A2/A3 的 A8W4 MSD 是 per-token 反量化，一个 token 一个 x scale。
        TORCH_CHECK(xScale.scalar_type() == at::kFloat && xScale.numel() == mCap,
                    "MSD x_scale must contain one FP32 scale per row");
        return;
    }
    TORCH_CHECK(xScale.numel() >= mCap * (k / 32), "xScale numel inconsistent with (M_cap, K)");
}

void ValidateWScaleNumel(const at::Tensor &wScale, int64_t e, int64_t k, int64_t n)
{
    TORCH_CHECK(wScale.is_privateuseone(), "weightScale must be a NPU device tensor");
    if (wScale.scalar_type() != at::kFloat8_e8m0fnu) {
        // A2/A3 的打包 UINT64 scale 布局由 tiling 侧校验。
        return;
    }
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
    ValidateV2Params(x, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);
    const std::optional<std::vector<at::Tensor>> assistList = AsTensorList(weightAssistMatrix);

    TORCH_CHECK(weight.is_privateuseone(), "weight must be a NPU device tensor");
    TORCH_CHECK(weight.dim() >= 3, "stacked weight must include an expert dimension");
    const int64_t e = weight.sizes()[0];
    const int64_t k = x.sizes()[1];
    const int64_t n = InferPackedWeightN(weight, e, k, "weight");
    TORCH_CHECK(at_npu::native::custom_ops::get_npu_format(weight) == kAclFormatNd,
                "gmm_situ_quant expects ND-format weight (use gmm_situ_quant_weight_nz for FRACTAL_NZ weight)");
    TORCH_CHECK(weight.is_contiguous(), "ND weight must be contiguous");
    ValidateWScaleNumel(weightScale, e, k, n);

    if (IsA8W4Msd(x)) {
        // A2/A3 的 kernel 直接支持 ND weight，不需要格式转换。
        return RunV2Core(x, xScale, std::vector<at::Tensor>{weight}, std::vector<at::Tensor>{weightScale}, groupList,
                         e, n, assistList, bias, smoothScale, groupListType, tuningConfigOptional, beta, linearBeta);
    }
    at::Tensor wNz = CastNdToNz(weight);
    return RunV2Core(x, xScale, std::vector<at::Tensor>{wNz}, std::vector<at::Tensor>{weightScale}, groupList, e, n,
                     assistList, bias, smoothScale, groupListType, tuningConfigOptional, beta, linearBeta);
}

// ---- entry 2: gmm_situ_quant_weight_nz (NZ weight, stacked) ----
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nz(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weightScale,
    const std::optional<at::Tensor> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta)
{
    ValidateV2Params(x, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);
    const std::optional<std::vector<at::Tensor>> assistList = AsTensorList(weightAssistMatrix);

    TORCH_CHECK(weight.is_privateuseone(), "weight must be a NPU device tensor");
    TORCH_CHECK(weight.dim() >= 3, "stacked weight must include an expert dimension");
    const int64_t e = weight.sizes()[0];
    const int64_t k = x.sizes()[1];
    const int64_t n = InferPackedWeightN(weight, e, k, "weight");
    const int64_t wFmt = at_npu::native::custom_ops::get_npu_format(weight);
    TORCH_CHECK(wFmt == kNpuFormatFractalNzC0_16 || wFmt == kNpuFormatFractalNz,
                "gmm_situ_quant_weight_nz expects FRACTAL_NZ weight (format tag FRACTAL_NZ_C0_16/FRACTAL_NZ; use "
                "gmm_situ_quant for ND weight), got tag ", wFmt);
    ValidateWScaleNumel(weightScale, e, k, n);

    return RunV2Core(x, xScale, std::vector<at::Tensor>{weight}, std::vector<at::Tensor>{weightScale}, groupList, e, n,
                     assistList, bias, smoothScale, groupListType, tuningConfigOptional, beta, linearBeta);
}

// ---- entry 1.list: gmm_situ_quant.list (ND weight TensorList) ----
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nd_list(
    const at::Tensor &x, const std::vector<at::Tensor> &weight, const std::vector<at::Tensor> &weightScale,
    const std::optional<std::vector<at::Tensor>> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta)
{
    ValidateV2Params(x, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);

    TORCH_CHECK(!weight.empty() && !weightScale.empty(), "weight and weightScale lists must not be empty");
    const int64_t e = static_cast<int64_t>(weight.size());
    const int64_t k = x.sizes()[1];
    const int64_t n = InferPackedWeightN(weight[0], 1, k, "each weight element");
    for (const auto &w : weight) {
        TORCH_CHECK(at_npu::native::custom_ops::get_npu_format(w) == kAclFormatNd,
                    "gmm_situ_quant.list expects ND-format weight elements");
    }

    if (IsA8W4Msd(x)) {
        // A2/A3 的 kernel 直接按地址表读逐专家的 ND weight，不做 cat/cast；
        // assist 也要按专家逐条传，kernel 用 GetTensorAddr(groupIdx) 取。
        return RunV2Core(x, xScale, weight, weightScale, groupList, e, n, weightAssistMatrix, bias, smoothScale,
                         groupListType, tuningConfigOptional, beta, linearBeta);
    }
    at::Tensor wStackedNd = CatTensorList(weight).view({e, n, k / 2});
    at::Tensor wScaleStacked = CatTensorList(weightScale);
    at::Tensor wNz = CastNdToNz(wStackedNd);
    return RunV2Core(x, xScale, std::vector<at::Tensor>{wNz}, std::vector<at::Tensor>{wScaleStacked}, groupList, e, n,
                     weightAssistMatrix, bias, smoothScale, groupListType, tuningConfigOptional, beta, linearBeta);
}

// ---- entry 2.list: gmm_situ_quant_weight_nz.list (NZ weight TensorList) ----
std::tuple<at::Tensor, at::Tensor> gmm_situ_quant_v2_nz_list(
    const at::Tensor &x, const std::vector<at::Tensor> &weight, const std::vector<at::Tensor> &weightScale,
    const std::optional<std::vector<at::Tensor>> &weightAssistMatrix, const std::optional<at::Tensor> &bias,
    const at::Tensor &xScale, const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfigOptional, double beta, double linearBeta)
{
    ValidateV2Params(x, bias, smoothScale, dequantMode, dequantDtype, quantMode, groupListType);
    ValidateXScales(x, xScale);

    TORCH_CHECK(!weight.empty() && !weightScale.empty(), "weight and weightScale lists must not be empty");
    const int64_t e = static_cast<int64_t>(weight.size());
    const int64_t k = x.sizes()[1];
    const int64_t n = InferPackedWeightN(weight[0], 1, k, "each weight element");
    for (const auto &w : weight) {
        const int64_t wFmt = at_npu::native::custom_ops::get_npu_format(w);
        TORCH_CHECK(wFmt == kNpuFormatFractalNzC0_16 || wFmt == kNpuFormatFractalNz,
                    "gmm_situ_quant_weight_nz.list expects FRACTAL_NZ weight elements");
    }

    TORCH_CHECK(weightScale.size() == weight.size(), "weightScale list length must equal weight list length");
    for (const auto &scale : weightScale) {
        ValidateWScaleNumel(scale, 1, k, n);
    }
    return RunV2Core(x, xScale, weight, weightScale, groupList, e, n, weightAssistMatrix, bias, smoothScale,
                     groupListType, tuningConfigOptional, beta, linearBeta);
}

} // namespace ascend_kernel
