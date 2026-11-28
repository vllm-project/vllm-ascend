#pragma once

#include <cmath>
#include <tuple>
#include <unordered_map>

#include "../../aclnn_torch_adapter/NPUBridge.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

namespace ascend_kernel::gmm1_situ {

namespace {

// ACL format / dtype constants (torch_npu int enums, verified on-device)
constexpr int64_t kAclFormatNd = 2;
constexpr int64_t kNpuFormatFractalNz = 29;          // torch_npu Format::FRACTAL_NZ tag
constexpr int64_t kNpuFormatFractalNzC0_16 = 50;     // torch_npu Format::FRACTAL_NZ_C0_16 tag (cast result)

// Accepted enum matrix (see file header)
constexpr int64_t kDequantModeMsd = 0;  // A3 MSD：per-channel 反量化
constexpr int64_t kDequantDtypeBf16 = 0;
constexpr int64_t kQuantModeMsd = 0;    // A3 per-token 量化
constexpr int64_t kQuantDtypeDefault = 0;

constexpr int64_t kCubeKAlignment = 64;
constexpr int64_t kCubeNAlignment = 128;
constexpr int64_t kInt4PerInt32 = 8;
constexpr int64_t kInt4ValuesPerByte = 2;
constexpr int64_t kInt4Radix = 16;

bool IsNullOrUndefined(const std::optional<at::Tensor> &t)
{
    return !t.has_value() || !t->defined() || t->numel() == 0;
}

void ValidateV2Params(const at::Tensor &x, const std::optional<at::Tensor> &bias,
                      const std::optional<at::Tensor> &smoothScale, int64_t dequantMode, int64_t dequantDtype,
                      int64_t quantMode, int64_t groupListType)
{
    TORCH_CHECK(x.scalar_type() == at::kChar, "A3 SiTU requires INT8 activations");
    TORCH_CHECK(IsNullOrUndefined(bias),
                "gmm_situ_quant: bias is not supported by this SiTU variant (pass None); non-empty bias would be "
                "silently wrong, so it is rejected explicitly");
    TORCH_CHECK(IsNullOrUndefined(smoothScale),
                "gmm_situ_quant: smoothScale is not supported (pass None); rejected explicitly");
    TORCH_CHECK(dequantMode == kDequantModeMsd, "A3 SiTU requires dequantMode=0");
    TORCH_CHECK(dequantDtype == kDequantDtypeBf16,
                "gmm_situ_quant: only dequantDtype=0 (BF16 intermediate GEMM result) is implemented, got ",
                dequantDtype);
    TORCH_CHECK(quantMode == kQuantModeMsd, "A3 SiTU requires quantMode=0");
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

int64_t InferWeightN(const at::Tensor &weight, const at::Tensor &scale, int64_t e, int64_t k)
{
    TORCH_CHECK(e > 0 && k > 0 && scale.numel() % e == 0,
                "MSD weight requires positive E/K and the same scale width for each expert");
    const int64_t n = scale.numel() / e;
    TORCH_CHECK(n > 0 && n % 2 == 0, "MSD GMM output width must be positive and even for SiTU");
    const int64_t packedColumns = (n + kInt4PerInt32 - 1) / kInt4PerInt32;
    TORCH_CHECK(weight.numel() == e * k * packedColumns,
                "MSD packed weight and per-expert scale width disagree");
    return n;
}

at::Tensor PadGateAndUp(const at::Tensor &tensor, int64_t n, int64_t paddedN)
{
    const int64_t half = n / 2;
    const int64_t pad = (paddedN - n) / 2;
    return at::cat({at::constant_pad_nd(tensor.narrow(-1, 0, half), {0, pad}),
                    at::constant_pad_nd(tensor.narrow(-1, half, half), {0, pad})}, -1);
}

at::Tensor ToNdPackedWeight(const at::Tensor &weight,
                           std::unordered_map<const c10::StorageImpl *, at::Tensor> &convertedStorages)
{
    if (at_npu::native::custom_ops::get_npu_format(weight) == kAclFormatNd) {
        return weight;
    }

    const auto bytes = weight.view(at::kChar);
    const auto &desc = vllm_ascend::NPUBridge::GetNpuStorageImplDesc(bytes);
    const auto *storage = bytes.storage().unsafeGetStorageImpl();
    auto [entry, inserted] = convertedStorages.try_emplace(storage);
    if (inserted) {
        // Convert the base allocation once. An expert view's logical offset is
        // only a valid byte offset after removing the per-expert NZ padding.
        auto base = bytes.as_strided(desc.base_sizes_, desc.base_strides_, desc.base_offset_);
        entry->second = at::empty(base.sizes(), base.options());
        EXEC_NPU_CMD(aclnnNpuFormatCast, base, entry->second);
    }
    return entry->second.as_strided(bytes.sizes(), bytes.strides(), bytes.storage_offset() - desc.base_offset_)
        .view(at::kInt);
}

at::Tensor PadWeight(at::Tensor packed, int64_t k, int64_t n, int64_t paddedK, int64_t paddedN)
{
    TORCH_CHECK(packed.dim() >= 2 && packed.size(-2) == k &&
                packed.size(-1) == (n + kInt4PerInt32 - 1) / kInt4PerInt32,
                "MSD tail padding requires packed weights with trailing dimensions [K, ceil(N/8)]");
    if (paddedN != n && n % (2 * kInt4PerInt32) == 0) {
        packed = PadGateAndUp(packed, n / kInt4PerInt32, paddedN / kInt4PerInt32);
    } else if (paddedN != n) {
        // Split at logical N/2 even when the gate/up boundary falls inside a carrier word.
        const auto bytes = packed.view(at::kByte).to(at::kInt);
        const auto high = at::floor_divide(bytes, kInt4Radix);
        const auto lanes = at::stack({bytes - high * kInt4Radix, high}, -1).flatten(-2, -1);
        const auto pairs = PadGateAndUp(lanes, n, paddedN)
                               .unflatten(-1, {paddedN / kInt4ValuesPerByte, kInt4ValuesPerByte});
        packed = (pairs.select(-1, 0) + pairs.select(-1, 1) * kInt4Radix).to(at::kByte).view(at::kInt);
    }
    if (paddedK != k) {
        packed = at::constant_pad_nd(packed, {0, 0, 0, paddedK - k});
    }
    return packed;
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

    TORCH_CHECK(e > 0 && !weights.empty(), "weight must be non-empty");
    TORCH_CHECK(weights.size() == 1 || weights.size() == static_cast<size_t>(e),
                "weight must contain one stacked tensor or E per-expert tensors");
    TORCH_CHECK(weightScales.size() == weights.size(), "weightScale must use the same stacked/list form as weight");
    TORCH_CHECK(groupList.is_privateuseone(), "groupList must be a NPU device tensor (graph-capturable contract)");
    TORCH_CHECK(groupList.scalar_type() == at::kLong, "groupList must be int64");
    TORCH_CHECK(groupList.dim() == 1 && groupList.is_contiguous(), "groupList must be a contiguous vector");
    TORCH_CHECK(groupList.numel() == e, "groupList length must equal E");
    TORCH_CHECK(groupList.device() == x.device(), "groupList and x must be on the same device");
    TORCH_CHECK(std::isfinite(beta) && beta > 0.0 && std::isfinite(linearBeta),
                "SiTU requires finite beta > 0 and finite linear_beta");
    for (size_t i = 0; i < weights.size(); ++i) {
        TORCH_CHECK(weights[i].scalar_type() == at::kInt && weights[i].is_contiguous(),
                    "MSD weight must be contiguous packed INT32");
        TORCH_CHECK(weights[i].device() == x.device() && weightScales[i].device() == x.device(),
                    "MSD weights, scales and x must be on the same device");
        TORCH_CHECK(weightScales[i].scalar_type() == at::kLong && weightScales[i].is_contiguous(),
                    "MSD weight_scale must be contiguous INT64 carrying UINT64 bits");
        const int64_t localExperts = weights.size() == 1 ? e : 1;
        TORCH_CHECK(InferWeightN(weights[i], weightScales[i], localExperts, k) == n,
                    "MSD weights must have the same logical width");
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

    auto dev = x.options();
    const int64_t paddedN = mCap > 0 ? (n + kCubeNAlignment - 1) / kCubeNAlignment * kCubeNAlignment : n;
    auto y = at::empty({mCap, paddedN / 2}, dev.dtype(at::kChar));
    auto yScale = at::empty({mCap}, dev.dtype(at::kFloat));

    // An EP rank can legitimately receive no routed tokens. Preserve the
    // inferred empty output shapes and avoid launching a kernel with empty
    // storage; all metadata validation above still applies to this path.
    if (mCap == 0) {
        return std::make_tuple(y, yScale);
    }

    at::TensorList weightList(weights);
    at::TensorList weightScaleList(weightScales);

    // weight_assist_matrix 在原型里是 DYNAMIC 槽位，GE 侧不接受空列表。
    // A3 两种形态都支持：不传时 tiling 按 numel == 1 认出占位，改由 kernel 在每个专家块
    // 末尾打一行全 -8 的 int4，让 cube 顺手算出 cAux（省掉一次 O(E*K*N) 的离线展开）；
    // 传了就用真实矩阵（单张量形态是 [E, N]，多张量形态是逐专家 [N]）。
    std::vector<at::Tensor> assistHolder;
    if (weightAssistMatrix.has_value() && !weightAssistMatrix->empty()) {
        assistHolder = *weightAssistMatrix;
        // kernel 侧按 isSingleTensor 决定读法：单张量形态按 [E, N] 偏移，逐专家形态走
        // GetTensorAddr(groupIdx)。isSingleTensor 由 weight 的堆叠/列表形态决定，assist 必须同形。
        TORCH_CHECK(assistHolder.size() == weights.size(),
                    "weightAssistMatrix must use the same stacked/list form as weight, got ", assistHolder.size(),
                    " assist tensors vs ", weights.size(), " weight tensors");
    } else {
        // 占位必须是 numel == 1：真正的补偿矩阵是 [N] 或 [E, N]，N 至少 16 对齐，
        // tiling 就靠「只有 1 个元素」判定调用方没给、kernel 自算。
        assistHolder.push_back(at::empty({1}, dev.dtype(at::kFloat)));
    }
    at::TensorList assistList(assistHolder);

    at::Tensor paddedX;
    std::vector<at::Tensor> paddedWeights;
    std::vector<at::Tensor> paddedScales;
    const int64_t paddedK = (k + kCubeKAlignment - 1) / kCubeKAlignment * kCubeKAlignment;
    if (paddedK != k || paddedN != n) {
        std::unordered_map<const c10::StorageImpl *, at::Tensor> convertedStorages;
        paddedWeights.reserve(weights.size());
        if (paddedK != k) {
            paddedX = at::constant_pad_nd(x, {0, paddedK - k});
        }
        for (const auto &weight : weights) {
            paddedWeights.push_back(PadWeight(ToNdPackedWeight(weight, convertedStorages), k, n, paddedK, paddedN));
        }
        weightList = at::TensorList(paddedWeights);
        if (paddedN != n) {
            for (const auto &scale : weightScales) {
                paddedScales.push_back(PadGateAndUp(scale.reshape({-1, n}), n, paddedN));
            }
            weightScaleList = at::TensorList(paddedScales);
            if (weightAssistMatrix.has_value()) {
                for (auto &assist : assistHolder) {
                    assist = PadGateAndUp(assist.reshape({-1, n}), n, paddedN);
                }
            }
        }
    }
    const std::vector<int64_t> tuning = tuningConfigOptional.has_value() && !tuningConfigOptional->empty()
                                            ? *tuningConfigOptional
                                            : std::vector<int64_t>{0};
    at::IntArrayRef tuningRef(tuning);
    const bool transposeWeight = false;
    const bool activateLeft = true;
    const auto &kernelX = paddedX.defined() ? paddedX : x;
    EXEC_NPU_CMD(aclnnGroupedMatmulSituQuantMsdV2, kernelX, xScale, weightList, weightScaleList, groupList, assistList,
                 bias, smoothScale, groupListType, beta, linearBeta, kDequantModeMsd, kDequantDtypeBf16,
                 kQuantModeMsd, kQuantDtypeDefault, transposeWeight, tuningRef, activateLeft, y, yScale);
    if (paddedN != n) {
        y = y.narrow(1, 0, n2).contiguous();
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
    TORCH_CHECK(xScale.scalar_type() == at::kFloat && xScale.numel() == mCap,
                "MSD x_scale must contain one FP32 scale per row");
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
    const int64_t n = InferWeightN(weight, weightScale, e, k);
    TORCH_CHECK(at_npu::native::custom_ops::get_npu_format(weight) == kAclFormatNd,
                "gmm_situ_quant expects ND-format weight (use gmm_situ_quant_weight_nz for FRACTAL_NZ weight)");
    TORCH_CHECK(weight.is_contiguous(), "ND weight must be contiguous");

    return RunV2Core(x, xScale, std::vector<at::Tensor>{weight}, std::vector<at::Tensor>{weightScale}, groupList,
                     e, n, assistList, bias, smoothScale, groupListType, tuningConfigOptional, beta, linearBeta);
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
    const int64_t n = InferWeightN(weight, weightScale, e, k);
    const int64_t wFmt = at_npu::native::custom_ops::get_npu_format(weight);
    TORCH_CHECK(wFmt == kNpuFormatFractalNzC0_16 || wFmt == kNpuFormatFractalNz,
                "gmm_situ_quant_weight_nz expects FRACTAL_NZ weight (format tag FRACTAL_NZ_C0_16/FRACTAL_NZ; use "
                "gmm_situ_quant for ND weight), got tag ", wFmt);

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
    const int64_t n = InferWeightN(weight[0], weightScale[0], 1, k);
    for (const auto &w : weight) {
        TORCH_CHECK(at_npu::native::custom_ops::get_npu_format(w) == kAclFormatNd,
                    "gmm_situ_quant.list expects ND-format weight elements");
    }

    return RunV2Core(x, xScale, weight, weightScale, groupList, e, n, weightAssistMatrix, bias, smoothScale,
                     groupListType, tuningConfigOptional, beta, linearBeta);
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
    const int64_t n = InferWeightN(weight[0], weightScale[0], 1, k);
    for (const auto &w : weight) {
        const int64_t wFmt = at_npu::native::custom_ops::get_npu_format(w);
        TORCH_CHECK(wFmt == kNpuFormatFractalNzC0_16 || wFmt == kNpuFormatFractalNz,
                    "gmm_situ_quant_weight_nz.list expects FRACTAL_NZ weight elements");
    }

    TORCH_CHECK(weightScale.size() == weight.size(), "weightScale list length must equal weight list length");
    return RunV2Core(x, xScale, weight, weightScale, groupList, e, n, weightAssistMatrix, bias, smoothScale,
                     groupListType, tuningConfigOptional, beta, linearBeta);
}

} // namespace ascend_kernel::gmm1_situ
