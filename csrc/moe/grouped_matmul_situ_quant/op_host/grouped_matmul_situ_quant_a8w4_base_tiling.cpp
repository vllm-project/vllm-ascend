/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_situ_quant_base_tiling.cpp
 * \brief
 */
#include <algorithm>
#include "grouped_matmul_situ_quant_a8w4_base_tiling.h"
#include "err/ops_err.h"

using namespace matmul_tiling;

namespace optiling {
namespace GroupedMatmulSituQuantTiling {

constexpr int64_t ND_WEIGHT_MULTI_TENSOR_DIM = 2;
constexpr int64_t NZ_WEIGHT_MULTI_TENSOR_DIM = 4;
constexpr float EFFECTIVE_TASK_RATIO = 0.95f;
constexpr int32_t MIN_BASE_M = 16;

template <typename T>
static inline auto AlignUp(T a, T base) -> T
{
    if (base == 0) {
        return 0;
    }
    return (a + base - 1) / base * base;
}

template <typename T1, typename T2>
auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15U;
    }
    return a & ~15U;
}

int64_t GroupedMatmulSituQuantBaseTiling::CalMaxRowInUbA4W4(const uint64_t ubSize, const uint64_t n) const
{
    // A4W4 vector double-buffer 场景 UB 预算(与 kernel UpdateVecConfig 中 InitBuffer 的实际分配一致):
    //   9 * row * n + 8 * AlignUp(row, 8) + 2 * n + 64 <= ubSize
    // 其中:
    //   9 * row * n  : mmOutQueue(fp32, depth2) = 8*row*n + quantOutQueue(int8, depth2) = row*n
    //   8 * AlignUp(row,8) : quantScaleOutQueue(fp32, depth2) = 8 * AlignUp(row,8)
    //   2 * n + 64   : reduceWorkspace(SwiGLU tmp n/2 *fp32 + ReduceMax res/tmp 各 32B)
    const uint64_t ROW_N_COEF = 9;
    const uint64_t ALIGN_COEF = 8;
    const uint64_t N_COEF = 2;
    const uint64_t CONSTANT_TERM = 64;
    const int64_t MIN_ROW_THRESHOLD = 1;
    auto alignUp8 = [](int64_t v) -> int64_t { return (v + 8 - 1) / 8 * 8; };

    if (n == 0 || ubSize <= CONSTANT_TERM + N_COEF * n) {
        OP_LOGE(context_->GetNodeName(), "GMM_SITU_QUANT TILING A4W4: invalid n or ubSize, n = %lu, ubSize = %lu\n",
                n, ubSize);
        return 0;
    }

    // 先忽略对齐项给出初始估计, 再逐步递减直到满足 double buffer UB 预算
    int64_t maxRowEstimate = static_cast<int64_t>(ubSize - CONSTANT_TERM - N_COEF * n) / (ROW_N_COEF * n);
    if (maxRowEstimate < 0) {
        maxRowEstimate = 0;
    }

    uint64_t totalSize = ROW_N_COEF * maxRowEstimate * n +
                         ALIGN_COEF * static_cast<uint64_t>(alignUp8(maxRowEstimate)) + N_COEF * n + CONSTANT_TERM;
    while (totalSize > ubSize && maxRowEstimate > 0) {
        maxRowEstimate--;
        totalSize = ROW_N_COEF * maxRowEstimate * n + ALIGN_COEF * static_cast<uint64_t>(alignUp8(maxRowEstimate)) +
                    N_COEF * n + CONSTANT_TERM;
    }

    if (maxRowEstimate < MIN_ROW_THRESHOLD) {
        OP_LOGE(context_->GetNodeName(), "GMM_SITU_QUANT TILING A4W4: No valid row found for n = %lu, ubSize = %lu\n",
                n, ubSize);
        return 0;
    }
    return maxRowEstimate;
}

int64_t GroupedMatmulSituQuantBaseTiling::CalMaxRowInUbA8W4(const uint64_t ubSize, const uint64_t n) const
{
    const uint64_t ALIGNMENT = 8;
    const float WEIGHT_FACTOR = 8.5f;
    const uint64_t ALIGNMENT_TERM_FACTOR = 4;
    // 本算子把 reduceWorkspace 由 n/2 个 float 扩到 n 个 float（SiTU 暂存 + Tanh tmp），
    // 线性项因此由 6n 变为 8n。自算补偿矩阵时后处理还要一行 fp16 的 cAux 暂存，再 +2n。
    const uint64_t LINEAR_TERM_FACTOR = assistRowsPerExpert_ != 0 ? 10 : 8;
    const uint64_t CONSTANT_TERM = 64;
    const int64_t MIN_ROW_THRESHOLD = 1;

    // A8W4 表达式：8.5 * row * n + 4 * alignUp(row, 8) + (8 或 10)n + 64 <= ubSize

    // 忽略对齐项的初始估计
    if (n == 0 || ubSize <= CONSTANT_TERM + LINEAR_TERM_FACTOR * n) {
        OP_LOGE(context_->GetNodeName(), "GMM_SITU_QUANT TILING: invalid n or ubSize, n = %lu, ubSize = %lu\n", n,
                ubSize);
        return 0;
    }

    int64_t maxRowEstimate =
        (ubSize - CONSTANT_TERM - LINEAR_TERM_FACTOR * n) / static_cast<int64_t>(WEIGHT_FACTOR * n);

    // 考虑对齐影响
    uint64_t alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    uint64_t totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) +
                         ALIGNMENT_TERM_FACTOR * alignedRow + LINEAR_TERM_FACTOR * n + CONSTANT_TERM;

    // 如果超过UB大小，逐步减少row直到满足条件
    while (totalSize > ubSize && maxRowEstimate > 0) {
        maxRowEstimate--;
        alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) + ALIGNMENT_TERM_FACTOR * alignedRow +
                    LINEAR_TERM_FACTOR * n + CONSTANT_TERM;
    }

    if (maxRowEstimate < MIN_ROW_THRESHOLD) {
        OP_LOGE(context_->GetNodeName(), "GMM_SITU_QUANT TILING: No valid row found for n = %lu, ubSize = %lu\n", n,
                ubSize);
        return 0;
    }
    return maxRowEstimate;
}

int64_t GroupedMatmulSituQuantBaseTiling::CalMaxRowInUb(const uint64_t ubSize, const uint64_t n) const
{
    if (n == 0) {
        OP_LOGE(context_->GetNodeName(), "GMM_SITU_QUANT TILING: n should not be 0\n");
        return 0;
    }
    uint64_t tmpBufSize = (n / SWIGLU_REDUCE_FACTOR) * FP32_DTYPE_SIZE;
    uint64_t perchannleBufSize = n * FP32_DTYPE_SIZE * DOUBLE_BUFFER;
    uint64_t reduceMaxResBufSize = BLOCK_BYTE;
    uint64_t reduceMaxTmpBufSize = BLOCK_BYTE;
    const uint64_t CONSTANT_TERM = 64;
    int64_t remainUbSize = ubSize - tmpBufSize - perchannleBufSize - reduceMaxResBufSize - reduceMaxTmpBufSize;
    int64_t maxRowInUb =
        remainUbSize / (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    int64_t curUb = DOUBLE_BUFFER * (maxRowInUb * (INT32_DTYPE_SIZE * n + n / SWIGLU_REDUCE_FACTOR) +
                                     AlignUp(maxRowInUb, FP32_BLOCK_SIZE) * FP32_DTYPE_SIZE);
    if (curUb > remainUbSize) {
        // 64 : make sure ub does not excceed maxUbSize after align up to 8
        maxRowInUb = (remainUbSize - CONSTANT_TERM) /
                     (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    }
    if (maxRowInUb < 1) {
        // when n > (ubSize - 72) / 19 = 10330, maxRowInUb < 1
        OP_LOGE(context_->GetNodeName(), "GMM_SITU_QUANT TILING: n should not be greater than 10240, now is %lu\n",
                n);
    }
    return maxRowInUb;
}

bool GroupedMatmulSituQuantBaseTiling::IsCapable()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto weightDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    ge::DataType weightDType = weightDesc->GetDataType();
    // 本算子只实现 A8W4 MSD 路径（x=INT8, weight=INT4）；其余组合一律不认领，
    // 避免落到无 kernel 分支的 tiling key 上。
    if (xDesc->GetDataType() != ge::DataType::DT_INT8 || weightDType != ge::DataType::DT_INT4) {
        return false;
    }

    auto wShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShape);
    auto wSShape = wShape->GetStorageShape();
    if (!(wSShape.GetDimNum() == ND_WEIGHT_DIM_LIMIT || wSShape.GetDimNum() == ND_WEIGHT_MULTI_TENSOR_DIM ||
          wSShape.GetDimNum() == NZ_WEIGHT_DIM_LIMIT || wSShape.GetDimNum() == NZ_WEIGHT_MULTI_TENSOR_DIM)) {
        return false;
    }

    return true;
}

ge::graphStatus GroupedMatmulSituQuantBaseTiling::ParseInputAndAttr()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto weightDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    auto wShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShape);
    auto xShape = context_->GetDynamicInputShape(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto wScaleShape = context_->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wScaleShape);
    auto groupListShape = context_->GetDynamicInputShape(GROUPLIST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupListShape);

    auto wDimNum = wShape->GetStorageShape().GetDimNum();
    if (wDimNum == ND_WEIGHT_DIM_LIMIT || wDimNum == NZ_WEIGHT_DIM_LIMIT) {
        isSingleTensor_ = 1;
    } else {
        isSingleTensor_ = 0;
    }

    auto attr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attr); // check attr is not null
    const int64_t* dequantModePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    auto dequantMode = dequantModePtr != nullptr ? *dequantModePtr : 0;
    OP_CHECK_IF(!(dequantMode == 0 || dequantMode == 1),
                OP_LOGE(context_->GetNodeName(), "dequantMode must be 0 or 1, but actual value is %ld.", dequantMode),
                return ge::GRAPH_FAILED);
    const int64_t* groupListTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE);
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
    OP_CHECK_IF(
        !(groupListType_ == 0 || groupListType_ == 1),
        OP_LOGE(context_->GetNodeName(), "GroupListType must be 0 or 1, but actual value is %ld.", groupListType_),
        return ge::GRAPH_FAILED);

    ge::DataType xDType = xDesc->GetDataType();
    ge::DataType weightDType = weightDesc->GetDataType();

    isA8W4MSD_ = (xDType == ge::DataType::DT_INT8 && weightDType == ge::DataType::DT_INT4);
    isA4W4_ = (xDType == ge::DataType::DT_INT4 && weightDType == ge::DataType::DT_INT4);
    if (isA8W4MSD_) {
        // 调用方没给补偿矩阵时，op_api 的 Handler 会下发一个 [1] FLOAT32 占位（GE 侧 GenKey
        // 不接受空列表），这里按 numel 识别：真正的补偿矩阵是 [N] 或 [E, N]，N 至少 16 对齐，
        // 不可能只有 1 个元素。占位即「自算」信号，kernel 在每个专家块末尾自己打一行 -8。
        const gert::StorageShape* assistShape = context_->GetDynamicInputShape(WEIGHT_ASSIST_MATRIX_INDEX, 0);
        const bool isPlaceholder = (assistShape == nullptr) || (assistShape->GetStorageShape().GetShapeSize() == 1);
        assistRowsPerExpert_ = isPlaceholder ? 1 : 0;
    }

    if (isA8W4MSD_) {
        // bias 未实现：DoOpTiling 里固定 SetBias(false)，kernel 的 BiasT 只是占位模板参数。
        // 与 grouped_matmul_swiglu_quant_v2 的做法一致，非空直接拒绝，避免调用方拿到
        // 「少加了一项」的结果却没有任何报错。
        OP_CHECK_IF(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr,
                    OP_LOGE(context_->GetNodeName(),
                            "GMM_SITU_QUANT TILING: when A8W4, bias is not supported, please pass a nullptr."),
                    return ge::GRAPH_FAILED);
    }
    if (isA4W4_) {
        auto smoothScaleTensor = context_->GetDynamicInputTensor(SMOOTH_SCALE_INDEX, 0);
        if (smoothScaleTensor == nullptr) {
            smoothScaleDimNum_ = 0;
        } else {
            smoothScaleDimNum_ = smoothScaleTensor->GetStorageShape().GetDimNum();
        }
    }

    // tiling 只用到 compileInfo 里的 ubSize_ / aicNum_。CANN 9.3 由自动生成的 aclnn 入口走单算子
    // 执行器（individual op）时不会构建 TilingParse 上下文，GetCompileInfo 返回空；此时用
    // platformInfo 现场补齐，取值与 TilingPrepare4GroupedMatmulSituQuant 完全一致。
    auto compileInfoPtr = context_->GetCompileInfo<GMMSituCompileInfo>();
    GMMSituCompileInfo compileInfoOnPlatform;
    if (compileInfoPtr == nullptr) {
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        compileInfoOnPlatform.aicNum_ = ascendcPlatform.GetCoreNumAic();
        compileInfoOnPlatform.aivNum_ = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoOnPlatform.ubSize_);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoOnPlatform.l1Size_);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoOnPlatform.l0CSize_);
        compileInfoPtr = &compileInfoOnPlatform;
    }

    m_ = xShape->GetStorageShape().GetDim(0);
    k_ = xShape->GetStorageShape().GetDim(1);
    auto wScaleDimNum = wScaleShape->GetStorageShape().GetDimNum();
    isWeightTrans_ = false;
    if (wShape->GetStorageShape().GetDimNum() == NZ_WEIGHT_DIM_LIMIT ||
        wShape->GetStorageShape().GetDimNum() == NZ_WEIGHT_MULTI_TENSOR_DIM) {
        isNz_ = true;
    }
    const auto tuningConfigPtr = attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG);
    tuningConfig_ = tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 1 ?
                        (reinterpret_cast<const int64_t*>(tuningConfigPtr->GetData()))[0] :
                        0;

    if (isA4W4_) {
        n_ = wScaleShape->GetStorageShape().GetDim(wScaleDimNum - DIM_1);
    } else {
        auto wSShape = wShape->GetStorageShape();
        if (wSShape.GetDimNum() == ND_WEIGHT_DIM_LIMIT) {
            n_ = wSShape.GetDim(DIM_2);
        } else if (wSShape.GetDimNum() == NZ_WEIGHT_DIM_LIMIT) {
            n_ = wSShape.GetDim(DIM_1) * wSShape.GetDim(DIM_4);
        } else if (wSShape.GetDimNum() == ND_WEIGHT_MULTI_TENSOR_DIM) {
            n_ = wSShape.GetDim(DIM_1);
        } else if (wSShape.GetDimNum() == NZ_WEIGHT_MULTI_TENSOR_DIM) {
            n_ = wSShape.GetDim(DIM_0) * wSShape.GetDim(DIM_3);
        }
    }

    const bool* transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_WEIGHT);
    isWeightTrans_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;

    // ---- A8W4 对齐约束 ----
    // A 矩阵以 INT4 进 Cube，INT4 fractal 在 K 方向按 64 对齐；tiling 不做补齐，
    // K 不满足时 kernel 会算出错误结果（K%16!=0 时还会同步失败）。
    // per-token 量化的 fp32->int8 落盘只覆盖了 32 对齐与 16 对齐两种偏移，
    // N 不是 16 的整数倍时输出整块不会被写回。两条都在这里拦住，避免静默出错。
    OP_CHECK_IF(xDesc->GetDataType() == ge::DataType::DT_INT32,
                OP_LOGE(context_->GetNodeName(),
                        "GMM_SITU_QUANT TILING: when A8W4, x only supports INT8, but got INT32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(k_ % A8W4_ALIGN_K != 0,
                OP_LOGE(context_->GetNodeName(),
                        "GMM_SITU_QUANT TILING: when A8W4, K must be a multiple of %ld, but got %ld.", A8W4_ALIGN_K,
                        k_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(n_ % A8W4_ALIGN_N != 0,
                OP_LOGE(context_->GetNodeName(),
                        "GMM_SITU_QUANT TILING: when A8W4, N must be a multiple of %ld, but got %ld.", A8W4_ALIGN_N,
                        n_),
                return ge::GRAPH_FAILED);
    if (isNz_) {
        // NZ 的 N 方向 tile 是 64：描述符 [E, N/64, K/16, 16, 64] 表达不了非 64 倍数的 N。
        // 此时上面的 n_ = dim1 * 64 已经把 N 反推错（N=80 会得到 64），校验 n_ 是无效的，
        // 必须用 weightScale 上的逻辑 N 来判断。
        const int64_t logicalN = wScaleShape->GetStorageShape().GetDim(wScaleDimNum - DIM_1);
        OP_CHECK_IF(logicalN % A8W4_ALIGN_NZ_N != 0,
                    OP_LOGE(context_->GetNodeName(),
                            "GMM_SITU_QUANT TILING: when A8W4 NZ, N must be a multiple of %ld, but got %ld.",
                            A8W4_ALIGN_NZ_N, logicalN),
                    return ge::GRAPH_FAILED);
    }

    // ---- SiTU 激活参数 ----
    const float* betaPtr = attr->GetAttrPointer<float>(ATTR_INDEX_BETA);
    situBeta_ = betaPtr != nullptr ? *betaPtr : SITU_DEFAULT_BETA;
    OP_CHECK_IF(!(situBeta_ > 0.0f),
                OP_LOGE(context_->GetNodeName(), "GMM_SITU_QUANT TILING: beta must be > 0, but got %f", situBeta_),
                return ge::GRAPH_FAILED);
    const float* linearBetaPtr = attr->GetAttrPointer<float>(ATTR_INDEX_LINEAR_BETA);
    situLinearBeta_ = linearBetaPtr != nullptr ? *linearBetaPtr : SITU_DEFAULT_LINEAR_BETA;
    const bool* activateLeftPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_ACTIVATE_LEFT);
    situActivateLeft_ = (activateLeftPtr != nullptr && !(*activateLeftPtr)) ? 0 : 1;

    if (dequantMode == 1) { // perGroup量化模式：单tensor场景[E, KGroupCount, N]，多tensor场景[KGroupCount, N]
        quantGroupNum_ = wScaleShape->GetStorageShape().GetDim(wScaleDimNum - DIM_2);
    } else { // perChannel量化模式
        quantGroupNum_ = 1;
    }

    groupNum_ = groupListShape->GetStorageShape().GetDim(0);

    if (isA4W4_) {
        maxProcessRowNum_ = CalMaxRowInUbA4W4(compileInfoPtr->ubSize_, n_);
    } else if (isA8W4MSD_) {
        maxProcessRowNum_ = CalMaxRowInUbA8W4(compileInfoPtr->ubSize_, n_);
    } else {
        maxProcessRowNum_ = CalMaxRowInUb(compileInfoPtr->ubSize_, n_);
    }

    OP_CHECK_IF(maxProcessRowNum_ <= 0,
                OP_LOGE(context_->GetNodeName(),
                        "GMM_SITU_QUANT TILING: CalMaxRowInUb returned invalid maxProcessRowNum = %ld, "
                        "ubSize = %lu, n = %ld, isA4W4 = %d, isA8W4MSD = %d\n",
                        maxProcessRowNum_, compileInfoPtr->ubSize_, n_, isA4W4_, isA8W4MSD_),
                return ge::GRAPH_FAILED);

    blockDim_ = compileInfoPtr->aicNum_;
    return ge::GRAPH_SUCCESS;
}

int32_t GroupedMatmulSituQuantBaseTiling::FindBestSingleN(const uint32_t& aicNum, int64_t baseM,
                                                              int64_t baseN) const
{
    uint64_t quantGroupNum = quantGroupNum_;
    if (n_ < baseN || tuningConfig_ <= 0 || !(quantGroupNum == 1)) {
        return baseN;
    }
    int32_t mDim = CeilDiv(tuningConfig_, baseM);
    int32_t nDim = CeilDiv(n_, baseN);
    int64_t taskNum = static_cast<int64_t>(mDim) * nDim * static_cast<int64_t>(groupNum_);
    int64_t taskNumPerCore = CeilDiv(taskNum, static_cast<int64_t>(aicNum));
    // 每个核只需要做1个基本块的时候，任务量太少，无需处理
    if (taskNumPerCore <= 1) {
        return baseN;
    }
    int32_t curNDim = 0;
    int64_t curTaskNum = 0;
    int32_t bestSingleN = baseN;
    float ratio = 0;
    for (uint32_t i = 1; i <= aicNum; ++i) {
        if (isNz_) {
            bestSingleN = CeilDiv(static_cast<int32_t>(n_), i);
            if (bestSingleN != n_ && bestSingleN % baseN != 0) {
                continue;
            }
        } else {
            // 暂时只NZ格式开启动态分块
            return baseN;
        }
        curNDim = CeilDiv(n_, bestSingleN);
        curTaskNum = static_cast<int64_t>(mDim) * curNDim * static_cast<int64_t>(groupNum_);
        ratio = static_cast<float>(curTaskNum) / static_cast<float>(AlignUp(curTaskNum, static_cast<int64_t>(aicNum)));
        if (ratio >= EFFECTIVE_TASK_RATIO) {
            return bestSingleN;
        }
    }
    return baseN;
}

bool GroupedMatmulSituQuantBaseTiling::TryFullLoadA(int32_t baseM, int64_t baseN, int64_t baseK, uint64_t l1Size)
{
    // 暂时只支持A4W4
    float sizeofweightDtype = 0.5f;
    float sizeofxDtype = 0.5f;
    auto matBl1Size = static_cast<int32_t>(tilingData_.mmTilingData.get_depthB1() * baseN * baseK * sizeofweightDtype);
    auto remainL1Size = l1Size - matBl1Size - 8 * baseN;
    int32_t newDepthA1 = CeilDiv(k_, baseK);
    if (static_cast<int32_t>(newDepthA1 * baseM * baseK * sizeofxDtype) < static_cast<int32_t>(remainL1Size)) {
        tilingData_.mmTilingData.set_stepKa(newDepthA1);
        tilingData_.mmTilingData.set_depthA1(newDepthA1);
        return true;
    }
    return false;
}

ge::graphStatus GroupedMatmulSituQuantBaseTiling::DynamicTilingSingleN(const gert::TilingContext* context,
                                                                           const uint32_t& aicNum, int64_t baseM,
                                                                           int64_t baseN, int64_t baseK)
{
    // get info
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t l1Size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    tilingData_.gmmSituQuantBaseParams.set_singleN(0);

    if (n_ < baseN || tuningConfig_ <= 0 || !isA4W4_) {
        return ge::GRAPH_SUCCESS;
    }
    int32_t bestSingleN = FindBestSingleN(aicNum, baseM, baseN);
    if (bestSingleN == baseN) { // 没找到更优的singleN
        return ge::GRAPH_SUCCESS;
    }
    tilingData_.gmmSituQuantBaseParams.set_singleN(bestSingleN);
    // 先不改看看baseM能否全载左矩阵
    if (TryFullLoadA(baseM, baseN, baseK, l1Size)) {
        return ge::GRAPH_SUCCESS;
    }
    // 可以尝试减小baseM来全载左矩阵
    int32_t newBaseM = static_cast<int32_t>(SixteenAlign(tuningConfig_, true));
    // 防止不均匀情况
    newBaseM += MIN_BASE_M;
    // 再看看能否全载左矩阵
    if (newBaseM < baseM && TryFullLoadA(newBaseM, baseN, baseK, l1Size)) {
        tilingData_.mmTilingData.set_baseM(newBaseM);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

// B（权重）在 L1 里的预取深度固定给 8 块，**试过按 L1 剩余空间加深，是负收益，不要再试**。
//
// 起因是 profiling 上 aic_scalar_mte1_stall_time 很高（Cube 在等 MTE1 把 L1 的块搬进
// L0A/L0B），看起来像「预取块数不够、盖不住 GM 延迟」，而 NZ 的 B 块只有
// baseN(128)×baseK(256)/2 = 16 KB、A 占 8×32 KB 之后 L1 还空着一大截，把剩下的都
// 算给 B 应该不亏。
//
// 实测（910B，20 AIC）：
//   * prefill e14 M=2048 ND：融合 622 µs → 930 µs（+49%），A/B 单轮就被打断；
//   * E=2/M=256/NZ 客户 shape：Task 80.8 µs → 84.9 µs。
// 原因是「A 8 块 + B 撑满剩余 L1」把 L1 用到 100%，而 L1 还要容纳 MatmulApi 自己的
// 对齐/中间缓冲，深度一深就互相挤兑，反而更容易 miss。8 块是能稳定跑满带宽的点。

ge::graphStatus GroupedMatmulSituQuantBaseTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Begin Run GMM Swiglu Tiling .");

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // NZ 与 ND 的权重在 K 方向的连续段长度差一个量级，baseN 的取舍方向相反，
    // 所以分块尺寸按排布分开选（常量处的注释给了实测依据）。
    if (isNz_) {
        baseM_ = A8W4_BASEM_NZ;
        baseN_ = A8W4_BASEN_NZ;
        baseK_ = A8W4_BASEK_NZ;
    } else {
        baseM_ = A8W4_BASEM;
        baseN_ = A8W4_BASEN;
        baseK_ = A8W4_BASEK;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    // 主分块和备选分块（A8W4_BASEM_NZ_ALT）除了 baseM/baseN 之外完全一样，所以抽成
    // 一个 lambda 各生成一份 TCubeTiling；kernel 侧按 live 行数二选一。
    auto genMmTiling = [&](auto& out, int64_t baseM, int64_t baseN, int64_t baseK) -> bool {
        MatmulApiTiling tiling(ascendcPlatform);
        tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
        tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT4);
        tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        tiling.SetBias(false);
        tiling.SetShape(baseM, baseN, k_);
        tiling.SetFixSplit(baseM, baseN, baseK);
        tiling.SetOrgShape(m_, n_, k_);
        tiling.SetBufferSpace(-1, -1, -1);
        if (tiling.GetTiling(out) == -1) {
            return false;
        }
        if (isA8W4MSD_ || isA4W4_) {
            out.set_baseM(baseM);
            out.set_baseN(baseN);
            out.set_baseK(baseK);
            out.set_dbL0B(DOUBLE_BUFFER);
            out.set_stepKa(NUM_FOUR);
            out.set_stepKb(NUM_FOUR);
            out.set_depthA1(NUM_EIGHT);
            out.set_depthB1(NUM_EIGHT);
            out.set_stepM(1);
            out.set_stepN(1);
        }
        return true;
    };
    OP_CHECK_IF(!genMmTiling(tilingData_.mmTilingData, baseM_, baseN_, baseK_),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "grouped_matmul_situ_quant_base_tiling, get tiling failed"),
                return ge::GRAPH_FAILED);
    // 备选分块只给 A8W4 MSD 的 NZ 权重用：ND 档的瓶颈是权重 K 方向连续段的长度，
    // 换来换去只会更差（见 A8W4_BASEN 处的注释）。其余情况把阈值置 0，kernel 侧就
    // 不会读 mmTilingDataAlt，这里也就不必浪费一次 GetTiling 去填它。
    if (isA8W4MSD_ && isNz_) {
        OP_CHECK_IF(!genMmTiling(tilingData_.mmTilingDataAlt, A8W4_BASEM_NZ_ALT, A8W4_BASEN_NZ_ALT,
                                 A8W4_BASEK_NZ),
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                                "grouped_matmul_situ_quant_base_tiling, get alt tiling failed"),
                    return ge::GRAPH_FAILED);
        tilingData_.gmmSituQuantBaseParams.set_altRowsPerExpertLimit(A8W4_ALT_ROWS_PER_EXPERT_LIMIT);
    } else {
        tilingData_.gmmSituQuantBaseParams.set_altRowsPerExpertLimit(0);
    }

    usrWorkspaceLimit_ = USER_WORKSPACE_LIMIT;
    mLimit_ = 0;
    if (isA8W4MSD_) {
        // 自算补偿矩阵时每个 split 内每个非空专家块多占一行：A 里是 K/2 字节（一行 int4），
        // C 里是 N*2 字节（一行 fp16）。这是与 m 无关的常数，先扣掉再按行均分。
        const int64_t auxBytesPerSplit =
            assistRowsPerExpert_ * groupNum_ * (k_ / 2 + n_ * static_cast<int64_t>(SIZE_OF_HALF_2));
        mLimit_ = ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) - auxBytesPerSplit) /
                  (k_ * sizeof(int8_t) + DOUBLE_ROW * n_ * SIZE_OF_HALF_2);
    } else if (isA4W4_) {
        mLimit_ = ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / (n_ * SIZE_OF_HALF_2));
    } else {
        mLimit_ = ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n_;
    }

    OP_CHECK_IF(
        mLimit_ <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "mLimit_ is %ld, must be greater than 0.", mLimit_),
        return ge::GRAPH_FAILED);
    tilingData_.gmmSituQuantBaseParams.set_mLimit(mLimit_);

    DynamicTilingSingleN(context_, blockDim_, baseM_, baseN_, baseK_);

    if (isA8W4MSD_) {
        // Captured calls own their workspace. Small decode batches only need
        // space for their capacity, while larger batches reuse bounded chunks.
        int64_t workSpaceMTemp = std::min(m_, mLimit_) * DOUBLE_WORKSPACE_SPLIT;
        // 每个 split 内每个非空专家块多一行（自算补偿矩阵时）：A 区一行 K 个 int4 = K/2 字节，
        // C 区一行 N 个 fp16 = N*2 字节。按专家总数 groupNum_ 上界预留。
        const int64_t auxRowsPerBuffer = assistRowsPerExpert_ * groupNum_;
        const int64_t aBytesTotal =
            workSpaceMTemp * k_ + DOUBLE_WORKSPACE_SPLIT * auxRowsPerBuffer * (k_ / 2);
        const int64_t cBytesTotal =
            DOUBLE_ROW * workSpaceMTemp * n_ * SIZE_OF_HALF_2 +
            DOUBLE_WORKSPACE_SPLIT * auxRowsPerBuffer * n_ * SIZE_OF_HALF_2;
        tilingData_.gmmSituQuantBaseParams.set_workSpaceOffset1(aBytesTotal);
        tilingData_.gmmSituQuantBaseParams.set_workSpaceOffset2(cBytesTotal);
        workspaceSize_ = SYS_WORKSPACE_SIZE + // 系统预留16MB
                         aBytesTotal +       // 第一阶段 预处理左矩阵 (mLimit_, K) * int8 * 2(double WorkSpace)
                         cBytesTotal;        // 第二阶段 矩阵乘结果 (2 * mLimit_, N) * fp16 * 2(double WorkSpace)
    } else if (isA4W4_) {
        int workSpaceMTemp = mLimit_ * DOUBLE_WORKSPACE_SPLIT;
        tilingData_.gmmSituQuantBaseParams.set_workSpaceOffset1(mLimit_ * n_ * SIZE_OF_HALF_2);
        tilingData_.gmmSituQuantBaseParams.set_workSpaceOffset2(0);
        workspaceSize_ = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n_ * SIZE_OF_HALF_2);
    } else {
        int workSpaceMTemp = (mLimit_ * DOUBLE_WORKSPACE_SPLIT > m_ ? m_ : mLimit_ * DOUBLE_WORKSPACE_SPLIT);
        tilingData_.gmmSituQuantBaseParams.set_workSpaceOffset1(0);
        tilingData_.gmmSituQuantBaseParams.set_workSpaceOffset2(0);
        workspaceSize_ = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n_ * sizeof(int32_t));
    }

    isSplitWorkSpace_ = m_ > mLimit_ * DOUBLE_WORKSPACE_SPLIT;
    SetTilingKeyAndScheMode();
    FillTilingData();
    PrintTilingData();
    OP_LOGD(context_->GetNodeName(), "End Run GMM Swiglu Tiling.");
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSituQuantBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

void GroupedMatmulSituQuantBaseTiling::FillTilingData()
{
    tilingData_.gmmSituQuantBaseParams.set_groupNum(groupNum_);
    tilingData_.gmmSituQuantBaseParams.set_coreNum(blockDim_);
    tilingData_.gmmSituQuantBaseParams.set_K(k_);
    tilingData_.gmmSituQuantBaseParams.set_N(n_);
    tilingData_.gmmSituQuantBaseParams.set_M(m_);
    tilingData_.gmmSituQuantBaseParams.set_baseM(baseM_);
    tilingData_.gmmSituQuantBaseParams.set_baseN(baseN_);
    tilingData_.gmmSituQuantBaseParams.set_quantGroupNum(quantGroupNum_);
    tilingData_.gmmSituQuantBaseParams.set_isSingleTensor(isSingleTensor_);
    tilingData_.gmmSituQuantBaseParams.set_groupListType(groupListType_);
    tilingData_.gmmSituQuantBaseParams.set_smoothScaleDimNum(smoothScaleDimNum_);
    tilingData_.gmmSituQuantBaseParams.set_assistRowsPerExpert(assistRowsPerExpert_);
    tilingData_.gmmSituQuant.set_maxProcessRowNum(maxProcessRowNum_);
    tilingData_.gmmSituQuant.set_groupListLen(groupNum_);
    tilingData_.gmmSituQuant.set_tokenLen(n_);
    // SiTU 激活参数
    tilingData_.gmmSituQuant.set_beta(situBeta_);
    tilingData_.gmmSituQuant.set_linearBeta(situLinearBeta_);
    tilingData_.gmmSituQuant.set_activateLeft(situActivateLeft_);
}

void GroupedMatmulSituQuantBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "grouped_matmul_situ_quant_base_tiling.");
    OP_LOGD(context_->GetNodeName(), "groupNum:      %ld", tilingData_.gmmSituQuantBaseParams.get_groupNum());
    OP_LOGD(context_->GetNodeName(), "coreNum:       %ld", tilingData_.gmmSituQuantBaseParams.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "M:             %ld", tilingData_.gmmSituQuantBaseParams.get_M());
    OP_LOGD(context_->GetNodeName(), "K:             %ld", tilingData_.gmmSituQuantBaseParams.get_K());
    OP_LOGD(context_->GetNodeName(), "N:             %ld", tilingData_.gmmSituQuantBaseParams.get_N());
    OP_LOGD(context_->GetNodeName(), "baseM:         %ld", tilingData_.gmmSituQuantBaseParams.get_baseM());
    OP_LOGD(context_->GetNodeName(), "baseN:         %ld", tilingData_.gmmSituQuantBaseParams.get_baseN());
    OP_LOGD(context_->GetNodeName(), "mLimit:        %ld", tilingData_.gmmSituQuantBaseParams.get_mLimit());
    OP_LOGD(context_->GetNodeName(), "quantGroupNum: %ld", tilingData_.gmmSituQuantBaseParams.get_quantGroupNum());
    OP_LOGD(context_->GetNodeName(), "isSingleTensor:%ld", tilingData_.gmmSituQuantBaseParams.get_isSingleTensor());
    OP_LOGD(context_->GetNodeName(), "groupListType: %ld", tilingData_.gmmSituQuantBaseParams.get_groupListType());
    OP_LOGD(context_->GetNodeName(), "smoothScaleDimNum: %ld",
            tilingData_.gmmSituQuantBaseParams.get_smoothScaleDimNum());
    OP_LOGD(context_->GetNodeName(), "maxProcessRowNum:      %ld", tilingData_.gmmSituQuant.get_maxProcessRowNum());
    OP_LOGD(context_->GetNodeName(), "groupListLen:          %ld", tilingData_.gmmSituQuant.get_groupListLen());
    OP_LOGD(context_->GetNodeName(), "tokenLen:              %ld", tilingData_.gmmSituQuant.get_tokenLen());
    OP_LOGD(context_->GetNodeName(), "USER_WORKSPACE_LIMIT:  %ld", usrWorkspaceLimit_);
    OP_LOGD(context_->GetNodeName(), "workspaceSizes:        %lu", workspaceSize_);
    OP_LOGD(context_->GetNodeName(), "isSplitWorkSpace:      %s", isSplitWorkSpace_ ? "true" : "false");
}

void GroupedMatmulSituQuantBaseTiling::SetTilingKeyAndScheMode()
{
    // 本算子只实现 A8W4 MSD 一条路径，IsCapable 已保证进入这里时 isA8W4MSD_ 为真。
    tilingKey_ = A8W4_MSD_TILING_KEY_MODE;
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
}

ge::graphStatus GroupedMatmulSituQuantBaseTiling::PostTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(blockDim_);

    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

} // namespace GroupedMatmulSituQuantTiling
} // namespace optiling
