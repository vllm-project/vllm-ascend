/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_sliding_window_tiling.cc
 * \brief
 */
#include "op_host/tiling_type.h"
#include "quant_batch_matmul_v3_checker.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "mc2_log_compat.h"
#include "adaptive_sliding_window_tiling.h"
#include "mc2_log.h"

using Ops::Transformer::MathUtil;

namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint64_t MX_GROUP_SIZE = 32;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint64_t PER_BLOCK_SIZE = 128;
constexpr uint64_t PER_BLOCK_BASE_SIZE_256 = 256;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_1024 = 1024;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t DOUBLE_CORE_NUM = 2;
constexpr uint32_t DB_SIZE = 2;

constexpr uint32_t SLIDING_WINOW_SHORTER_EDGE = 4;
constexpr uint32_t SLIDING_WINOW_LONGER_EDGE = 8;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint64_t MTE2_MIN_LOAD_SIZE_V100 = 32768;
constexpr uint64_t L2_ALIGN_SIZE = 128;

constexpr uint32_t SCALER_FACTOR_MAX = 127;
constexpr uint32_t SCALER_FACTOR_MIN = 1;
constexpr uint32_t SCALER_FACTOR_DEFAULT = 1;
constexpr uint32_t SCALER_FACTOR_B_BIT = 8;
constexpr uint32_t SCALER_FACTOR_M_BIT = 16;
constexpr uint32_t SCALER_FACTOR_N_BIT = 24;

constexpr uint64_t BASEM_BASEN_RATIO = 2;
constexpr uint64_t BASEK_LIMIT = 4095;

constexpr uint64_t AFULLLOAD_SINGBLE_CORE_A_SCALER = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint32_t VEC_CORE_GROUP_NUM = 2;
} // namespace

namespace optiling {

Mc2AdaptiveSlidingWindowTiling::Mc2AdaptiveSlidingWindowTiling(gert::TilingContext *context)
    : Mc2QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    Reset();
}

Mc2AdaptiveSlidingWindowTiling::Mc2AdaptiveSlidingWindowTiling(gert::TilingContext *context,
                                                               DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *out)
    : Mc2QuantBatchMatmulV3TilingBase(context, true), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

void Mc2AdaptiveSlidingWindowTiling::Reset()
{
    isBf16Opt_ = false;
    isUbQuant_ = false;

    if (!isTilingOut_) {
        tilingData_ = DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

bool Mc2AdaptiveSlidingWindowTiling::CheckDtype() const
{
    Mc2QuantBatchMatmulV3Checker qmmV3Checker(context_, inputParams_);
    OP_TILING_CHECK(!qmmV3Checker.CheckDtype(), OP_LOGE(inputParams_.opName, "CheckDtype fail"), return false);
    return true;
}

bool Mc2AdaptiveSlidingWindowTiling::CheckShape(const std::vector<gert::Shape *> &mandatoryShape,
                                                const gert::StorageShape *biasShape,
                                                const gert::StorageShape *pertokenShape,
                                                const std::vector<int64_t> &dimValueOfMKN) const
{
    Mc2QuantBatchMatmulV3Checker qmmV3Checker(context_, inputParams_);
    OP_TILING_CHECK(!qmmV3Checker.CheckShape(mandatoryShape, biasShape, pertokenShape, dimValueOfMKN),
                    OP_LOGE(inputParams_.opName, "CheckShape fail"), return false);
    return true;
}

ge::graphStatus Mc2AdaptiveSlidingWindowTiling::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfo fail");
    if (aicoreParams_.aicNum == 0UL || aicoreParams_.l1Size == 0UL || aicoreParams_.l0cSize == 0UL) {
        OP_LOGE(inputParams_.opName, "coreNum/L1Size/L0cSize should not be 0. coreNum: %lu, L1Size: %lu, L0cSize: %lu",
                aicoreParams_.aicNum, aicoreParams_.l1Size, aicoreParams_.l0cSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2AdaptiveSlidingWindowTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams);
    return Mc2QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}


ge::graphStatus Mc2AdaptiveSlidingWindowTiling::DoOpTiling()
{
    OP_LOGD(inputParams_.opName, "DoOpTiling of adaptive sliding window tiling strategy.");
    if (!AnalyseSlidingWinInfo()) {
        OP_LOGE(inputParams_.opName, "DoOpTiling fail");
        return ge::GRAPH_FAILED;
    }
    SetBf16Compat();
    CalL1Tiling();
    if (inputParams_.isPertoken) {
        CalcUbTiling();
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.M = inputParams_.mSize;
    tilingData_.matmulTiling.N = inputParams_.nSize;
    tilingData_.matmulTiling.Ka = inputParams_.kSize;
    tilingData_.matmulTiling.Kb = inputParams_.kSize;
    tilingData_.matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_.matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_.matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_.matmulTiling.stepM = basicTiling_.stepM;
    tilingData_.matmulTiling.stepN = basicTiling_.stepN;
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.isMxPerGroup) {
        tilingData_.matmulTiling.mxTypePara =
            (SCALER_FACTOR_MIN << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_MIN << SCALER_FACTOR_M_BIT);
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.matmulTiling.mxTypePara +=
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.matmulTiling.mxTypePara +=
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2AdaptiveSlidingWindowTiling::GetBiasMode() const
{
    uint64_t biasMode = 0UL;
    if (!inputParams_.hasBias) {
        biasMode = static_cast<uint64_t>(Mc2BiasMode::EXCLUEDE_FROM_TEMPLATE);
    } else if (inputParams_.aDtype == ge::DT_INT8) {
        biasMode = static_cast<uint64_t>(Mc2BiasMode::EXCLUEDE_FROM_TEMPLATE);
    } else {
        if (inputParams_.biasDtype == ge::DT_FLOAT) {
            biasMode = static_cast<uint64_t>(Mc2BiasMode::EXCLUEDE_FROM_TEMPLATE);
        } else if (inputParams_.biasDtype == ge::DT_BF16) {
            biasMode = static_cast<uint64_t>(Mc2BiasMode::CUBE_BIAS_BF16_TEMPLATE);
        } else if (inputParams_.biasDtype == ge::DT_FLOAT16) {
            biasMode = static_cast<uint64_t>(Mc2BiasMode::CUBE_BIAS_FP16_TEMPLATE);
        }
    }
    return biasMode;
}

uint64_t Mc2AdaptiveSlidingWindowTiling::GetKernelType() const
{
    uint64_t kernelType = 0UL;
    bool isScaleVecPostProcess = inputParams_.isPerChannel &&
                                 !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64);
    bool isVecPostProcess = (isScaleVecPostProcess || inputParams_.isPertoken || inputParams_.isPerBlock || isBf16Mix_);
    if (!isVecPostProcess && !isAFullLoad_) {
        kernelType = static_cast<uint64_t>(Mc2QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI);
    } else if (!isVecPostProcess && isAFullLoad_) {
        kernelType = static_cast<uint64_t>(Mc2QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
    } else if ((isScaleVecPostProcess || inputParams_.isPertoken || isBf16Mix_) && !isAFullLoad_) {
        kernelType = static_cast<uint64_t>(Mc2QMMKernelType::VEC_EPILOGUE_WITH_MMAPI);
    } else if ((isScaleVecPostProcess || inputParams_.isPertoken || isBf16Mix_) && isAFullLoad_) {
        kernelType = static_cast<uint64_t>(Mc2QMMKernelType::VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
    } else if (inputParams_.isPerBlock) {
        kernelType = static_cast<uint64_t>(Mc2QMMKernelType::VEC_EPILOGUE_WITH_CUSTOM_MM);
    }
    return kernelType;
}

uint64_t Mc2AdaptiveSlidingWindowTiling::GetTilingKey() const
{
    return RecursiveSum(inputParams_.transB, inputParams_.transA, GetBiasMode(), GetKernelType(), false, false, false,
                        false);
}

ge::graphStatus Mc2AdaptiveSlidingWindowTiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2AdaptiveSlidingWindowTiling::PostTiling()
{
    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0UL,
                    OP_LOGE(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.", tilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2AdaptiveSlidingWindowTiling::CalcUbTiling()
{
    uint64_t ubSize = static_cast<uint64_t>(aicoreParams_.ubSize);
    basicTiling_.ubCalcN = basicTiling_.baseN;
    // src(int32) + scale(fp32) + pertoken(fp32) + out(fp16/bf16), in and out need double buffer
    // int16_t reprersent bf16, input src + output dst + veccalc dequant api
    uint64_t ubCalc =
        static_cast<uint64_t>(DOUBLE_BUFFER_NUM * (sizeof(int32_t) + ge::GetSizeByDataType(inputParams_.cDtype))) *
        static_cast<uint64_t>(basicTiling_.ubCalcN);
    // input: scale perchannel
    if (!inputParams_.isPerTensor) {
        ubSize -= ge::GetSizeByDataType(inputParams_.scaleDtype) * basicTiling_.ubCalcN;
    }
    // input: pertokenScale fp32
    ubCalc += DOUBLE_BUFFER_NUM * ge::GetSizeByDataType(inputParams_.perTokenScaleDtype);
    // ub buffer size need 32 align
    ubSize -= DOUBLE_BUFFER_NUM * (UB_ALIGN_SIZE - ge::GetSizeByDataType(inputParams_.perTokenScaleDtype));
    basicTiling_.ubCalcM = static_cast<uint32_t>(ubSize / ubCalc);
    basicTiling_.ubCalcM =
        std::min(std::min(basicTiling_.ubCalcM, ops::CeilDiv(basicTiling_.baseM, VEC_CORE_GROUP_NUM)),
                 ops::CeilDiv(static_cast<uint32_t>(inputParams_.mSize), VEC_CORE_GROUP_NUM));
    return ge::GRAPH_SUCCESS;
}

bool Mc2AdaptiveSlidingWindowTiling::AnalyseSlidingWinInfo()
{
    if (!CalcBasicBlock()) {
        OP_LOGE(inputParams_.opName, "inappropriate basicBlock");
        return false;
    }
    adaptiveWin_.mBlockCnt = ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM);
    adaptiveWin_.nBlockCnt = ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN);
    adaptiveWin_.tatolBlockCnt = adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt;
    adaptiveWin_.mTail = inputParams_.mSize - (adaptiveWin_.mBlockCnt - 1UL) * adaptiveWin_.baseM;
    adaptiveWin_.nTail = inputParams_.nSize - (adaptiveWin_.nBlockCnt - 1UL) * adaptiveWin_.baseN;
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.tatolBlockCnt, aicoreParams_.aicNum);
    adaptiveWin_.tailWinBlockCnt = (adaptiveWin_.tatolBlockCnt) % aicoreParams_.aicNum;
    IsAFullLoad();
    if (adaptiveWin_.useTailWinLogic) {
        if (isAFullLoad_) {
            CalcTailBasicBlockAfullLoad();
        } else {
            CalcTailBasicBlock();
        }
    }
    return true;
}

bool Mc2AdaptiveSlidingWindowTiling::CalcBasicBlock()
{
    if (inputParams_.isPerBlock) {
        // when m or n is not divisible by 128, baseM or baseN is 128,
        // to prevent the division of tail blocks into two vector cores from generating a non-factor of 128.
        // when m <= 128, baseM is 128.
        // only one of baseM and baseN can be 256, with baseM taking priority.
        if (inputParams_.mSize <= PER_BLOCK_SIZE || inputParams_.mSize % PER_BLOCK_SIZE != 0UL) {
            adaptiveWin_.baseM = PER_BLOCK_SIZE;
            if (inputParams_.nSize % PER_BLOCK_SIZE == 0UL) {
                adaptiveWin_.baseN = PER_BLOCK_BASE_SIZE_256;
            } else {
                adaptiveWin_.baseN = PER_BLOCK_SIZE;
            }
        } else {
            adaptiveWin_.baseM = PER_BLOCK_BASE_SIZE_256;
            adaptiveWin_.baseN = PER_BLOCK_SIZE;
        }
        adaptiveWin_.baseK = PER_BLOCK_SIZE;
    } else {
        adaptiveWin_.baseM = std::min(inputParams_.mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
        adaptiveWin_.baseM =
            !inputParams_.transA ?
                ops::CeilAlign(adaptiveWin_.baseM, CUBE_BLOCK) :
                ops::CeilAlign(adaptiveWin_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype));
        adaptiveWin_.baseN = std::min(inputParams_.nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
        adaptiveWin_.baseN =
            inputParams_.transB ?
                ops::CeilAlign(adaptiveWin_.baseN, CUBE_BLOCK) :
                ops::CeilAlign(adaptiveWin_.baseN, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype));
        uint64_t baseKDefaultSize =
            static_cast<uint64_t>(GetShapeWithDataType(BASIC_BLOCK_SIZE_128, inputParams_.aDtype));

        adaptiveWin_.baseK =
            ops::CeilAlign(std::min(baseKDefaultSize, inputParams_.kSize),
                           inputParams_.isMxPerGroup ? MXFP_DIVISOR_SIZE :
                                                       GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype));
    }
    bool isSmallBlock =
        ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM) * ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN) <
        aicoreParams_.aicNum;
    if (isSmallBlock && !inputParams_.isPerBlock) {
        AdjustBasicBlock();
        OP_TILING_CHECK(adaptiveWin_.baseM == 0UL || adaptiveWin_.baseN == 0UL || adaptiveWin_.baseK == 0UL,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "input", std::to_string(adaptiveWin_.baseM).c_str(),
                            "The value of baseM, baseN and baseK must be greater than 0."),
                        return false);
    }
    return true;
}

void Mc2AdaptiveSlidingWindowTiling::AdjustBasicBlock()
{
    uint64_t baseMAlignNum =
        inputParams_.transA ? GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum =
        inputParams_.transB ? CUBE_BLOCK : GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB) ?
                                 GetShapeWithDataType(BASIC_BLOCK_SIZE_32, inputParams_.aDtype) :
                                 GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);
    if (IsMxBackwardTrans()) {
        baseKAlignNum = GetShapeWithDataType(MXFP_DIVISOR_SIZE, inputParams_.aDtype);
    }
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t tempBaseM = adaptiveWin_.baseM;
    uint64_t tempBaseN = adaptiveWin_.baseN;
    if (mMaxtile * nMaxtile >= aicoreParams_.aicNum || (!inputParams_.transA && inputParams_.transB)) {
        uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, adaptiveWin_.baseM);
        uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, adaptiveWin_.baseN);
        if (mMaxtile < nMaxtile || (mMaxtile == nMaxtile && baseNAlignNum == CUBE_BLOCK)) {
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
            mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
            nCore = aicoreParams_.aicNum / mCore;
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        } else {
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
            nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
            mCore = aicoreParams_.aicNum / nCore;
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        }

        while (tempBaseN >= tempBaseM * BASEM_BASEN_RATIO && nCore < aicoreParams_.aicNum / NUM_HALF &&
               tempBaseN != baseNAlignNum) {
            nCore = nCore * DOUBLE_CORE_NUM;
            mCore = aicoreParams_.aicNum / nCore;
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
            mCore = MathUtil::CeilDivision(inputParams_.mSize, static_cast<uint64_t>(tempBaseM));
            nCore = MathUtil::CeilDivision(inputParams_.nSize, static_cast<uint64_t>(tempBaseN));
        }
        while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCore < aicoreParams_.aicNum / NUM_HALF &&
               tempBaseM != baseMAlignNum) {
            mCore = mCore * DOUBLE_CORE_NUM;
            nCore = aicoreParams_.aicNum / mCore;
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
            mCore = MathUtil::CeilDivision(inputParams_.mSize, static_cast<uint64_t>(tempBaseM));
            nCore = MathUtil::CeilDivision(inputParams_.nSize, static_cast<uint64_t>(tempBaseN));
        }
        uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(inputParams_.kSize), baseKAlignNum);
        uint64_t kValueMax =
            GetShapeWithDataType(aicoreParams_.l0aSize / DB_SIZE, inputParams_.aDtype) / std::max(tempBaseM, tempBaseN);
        if (kValueMax >= baseKAlignNum) {
            adaptiveWin_.baseM = tempBaseM;
            adaptiveWin_.baseN = tempBaseN;
            kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
            adaptiveWin_.baseK = std::min(kValueAlign, kValueMax);
            adaptiveWin_.baseK = adaptiveWin_.baseK > BASEK_LIMIT ? adaptiveWin_.baseK / NUM_HALF : adaptiveWin_.baseK;
            adaptiveWin_.useTailWinLogic = false;
        }
    }
}

void Mc2AdaptiveSlidingWindowTiling::SetBf16Compat()
{
    bool isMix = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64 &&
                  inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0) &&
                 inputParams_.isPerChannel;
    bool isCompat = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64 &&
                     inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0) &&
                    inputParams_.isPerTensor && inputParams_.aDtype == ge::DT_INT8 && inputParams_.hasBias &&
                    (inputParams_.biasDtype == ge::DT_FLOAT || inputParams_.biasDtype == ge::DT_BF16);
    isBf16Mix_ = isMix || isCompat;
}

bool Mc2AdaptiveSlidingWindowTiling::IsMxKOdd() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
           ops::CeilDiv(inputParams_.kSize, MX_GROUP_SIZE) % MXFP_MULTI_BASE_SIZE != 0;
}

bool Mc2AdaptiveSlidingWindowTiling::IsMxBackwardTrans() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 && (inputParams_.transA || !inputParams_.transB);
}

void Mc2AdaptiveSlidingWindowTiling::IsAFullLoad()
{
    if (inputParams_.batchA != 1UL || IsMxKOdd() || IsMxBackwardTrans() || isTilingOut_ || inputParams_.isPerBlock) {
        isAFullLoad_ = false;
        return;
    }

    uint64_t singleCoreASize =
        GetSizeWithDataType(static_cast<uint64_t>(adaptiveWin_.baseM) * inputParams_.kSize, inputParams_.aDtype);

    //  mcnt为1 or 2 or 4时，ASW可保证所有核上的计算对M的依赖不换行，可做A(分核/不分核)全载
    isAFullLoad_ = singleCoreASize <= aicoreParams_.l1Size / AFULLLOAD_SINGBLE_CORE_A_SCALER &&
                   adaptiveWin_.mBlockCnt <= adaptiveWin_.nBlockCnt &&
                   (adaptiveWin_.mBlockCnt == 1UL || adaptiveWin_.mBlockCnt == 2UL || adaptiveWin_.mBlockCnt == 4UL ||
                    adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum);
}

bool Mc2AdaptiveSlidingWindowTiling::CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c) const
{
    // bias int32(BT 4096B)对baseN的影响，不超过1024; 开DB不超过512
    // scale uint64(FB 4096B)目前对baseN无影响，api会对超512的scale再做tiling
    uint64_t maxBiasBaseN = dbL0c == 1UL ? BASIC_BLOCK_SIZE_1024 : BASIC_BLOCK_SIZE_512;
    // FB 2KB, api再切分，tbe tiling亦如此
    uint64_t maxScaleBaseN = dbL0c == 1UL ? BASIC_BLOCK_SIZE_512 : BASIC_BLOCK_SIZE_256;
    bool isBiasInvalid = inputParams_.hasBias && inputParams_.biasDtype != ge::DT_BF16 && baseN > maxBiasBaseN;
    bool isScaleInvalid = !isUbQuant_ && baseN > maxScaleBaseN;
    return !(isBiasInvalid || isScaleInvalid);
}

void Mc2AdaptiveSlidingWindowTiling::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(inputParams_.opName, "coreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;
    basicTiling_.baseK = adaptiveWin_.baseK;

    basicTiling_.stepM = 1U;
    basicTiling_.stepN = 1U;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = inputParams_.kSize;

    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t totalL1Size = aicoreParams_.l1Size;

    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) &&
         CheckBiasAndScale(basicTiling_.baseN, DB_SIZE)) ?
            DB_SIZE :
            1U;
    uint64_t singleCoreBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0UL;
    uint64_t singleCoreScaleSize = inputParams_.isPerChannel ? basicTiling_.baseN * scaleDtypeSize : 0UL;
    uint64_t leftL1Size = totalL1Size - singleCoreBiasSize - singleCoreScaleSize;

    if (isAFullLoad_) {
        CalL1TilingDepthAfullload(leftL1Size);
    } else {
        CalL1TilingDepthANotfullload(leftL1Size);
    }
}

void Mc2AdaptiveSlidingWindowTiling::CalL1TilingDepthAfullload(uint64_t leftL1Size)
{
    basicTiling_.stepKa = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    basicTiling_.depthA1 = basicTiling_.stepKa;

    uint64_t singleCoreASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.singleCoreK, inputParams_.aDtype);

    uint64_t leftL1SizeAfterFullA = leftL1Size - ops::CeilAlign(singleCoreASize, L1_ALIGN_SIZE);

    uint64_t basebSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

    if (inputParams_.isMxPerGroup) {
        uint64_t singleCoreScaleASize = GetSizeWithDataType(
            basicTiling_.singleCoreM *
                ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(basicTiling_.singleCoreK), MX_GROUP_SIZE),
                               MXFP_MULTI_BASE_SIZE),
            inputParams_.perTokenScaleDtype);
        basicTiling_.scaleFactorA = 1U;
        leftL1SizeAfterFullA -= ops::CeilAlign(singleCoreScaleASize, L1_ALIGN_SIZE);
        basicTiling_.depthB1 = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
        basicTiling_.stepKb = basicTiling_.depthB1 == 1U ? basicTiling_.depthB1 : basicTiling_.depthB1 / DB_SIZE;
        basicTiling_.scaleFactorB = GetScaleFactorBAfullLoad(
            leftL1SizeAfterFullA - ops::CeilAlign(basicTiling_.depthB1 * basebSize, L1_ALIGN_SIZE));
    } else {
        basicTiling_.depthB1 = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
    }

    basicTiling_.stepKb = basicTiling_.depthB1 == 1U ? basicTiling_.depthB1 : basicTiling_.depthB1 / DB_SIZE;
}

void Mc2AdaptiveSlidingWindowTiling::CalL1TilingDepthANotfullload(uint64_t leftL1Size)
{
    if (inputParams_.isMxPerGroup) {
        uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
        uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

        uint64_t baseScaleASize =
            GetSizeWithDataType(ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE),
                                               MXFP_MULTI_BASE_SIZE) *
                                    basicTiling_.baseM,
                                inputParams_.perTokenScaleDtype);
        uint64_t baseScaleBSize =
            GetSizeWithDataType(ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE),
                                               MXFP_MULTI_BASE_SIZE) *
                                    basicTiling_.baseN,
                                inputParams_.scaleDtype);
        uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
        uint64_t leftL1SizeByDepthInit = leftL1Size - depthInit * (baseL1Size);
        uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
        uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
        basicTiling_.depthA1 = std::max(depthASec, depthBSec);
        basicTiling_.depthB1 = basicTiling_.depthA1;
        if (basicTiling_.depthA1 * baseL1Size > leftL1Size) {
            basicTiling_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
            basicTiling_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
        }
        CalStepKs();
        CalScaleFactors(baseASize, baseBSize, baseScaleASize, baseScaleBSize, leftL1Size);
    } else {
        basicTiling_.depthA1 =
            GetShapeWithDataType(leftL1Size / NUM_HALF / basicTiling_.baseM / basicTiling_.baseK, inputParams_.aDtype);
        basicTiling_.depthB1 =
            GetShapeWithDataType(leftL1Size / NUM_HALF / basicTiling_.baseN / basicTiling_.baseK, inputParams_.bDtype);
        CalStepKs();
    }
}

void Mc2AdaptiveSlidingWindowTiling::CalStepKs()
{
    // depthA,depthB 为1时，stepka, stepkb 只能是1.
    basicTiling_.stepKa = basicTiling_.depthA1 == 1U ? 1U : basicTiling_.depthA1 / DB_SIZE;
    basicTiling_.stepKb = basicTiling_.depthB1 == 1U ? 1U : basicTiling_.depthB1 / DB_SIZE;

    if (static_cast<uint64_t>(basicTiling_.stepKa * basicTiling_.baseK) > inputParams_.kSize) {
        basicTiling_.stepKa = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (static_cast<uint64_t>(basicTiling_.stepKb * basicTiling_.baseK) >= inputParams_.kSize) {
        basicTiling_.stepKb = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (basicTiling_.stepKa >= basicTiling_.stepKb &&
        static_cast<uint64_t>(basicTiling_.stepKa * basicTiling_.baseK) < inputParams_.kSize) {
        basicTiling_.stepKa = basicTiling_.stepKa / basicTiling_.stepKb * basicTiling_.stepKb;
    }
    if (basicTiling_.stepKb > basicTiling_.stepKa &&
        static_cast<uint64_t>(basicTiling_.stepKb * basicTiling_.baseK) < inputParams_.kSize) {
        basicTiling_.stepKb = basicTiling_.stepKb / basicTiling_.stepKa * basicTiling_.stepKa;
    }
    if (inputParams_.isPerBlock) {
        basicTiling_.stepKa =
            std::min(basicTiling_.stepKa, static_cast<uint32_t>(4)); // 限制stepKa最大为4, 防止issue queue阻塞
        basicTiling_.stepKb =
            std::min(basicTiling_.stepKb, static_cast<uint32_t>(4)); // 限制stepKb最大为4, 防止issue queue阻塞
    }
    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

void Mc2AdaptiveSlidingWindowTiling::CalScaleFactors(uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize,
                                                     uint64_t baseScaleBSize, [[maybe_unused]] uint64_t leftL1Size)
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;

    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V100 / baseScaleASize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V100 / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(inputParams_.kSize) / (basicTiling_.stepKa * basicTiling_.baseK);
    uint32_t scaleFactorB = static_cast<uint32_t>(inputParams_.kSize) / (basicTiling_.stepKb * basicTiling_.baseK);
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    // 来自L1 size 的约束
    uint64_t leftL1sie =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(
        leftL1sie / (basicTiling_.depthA1 * baseScaleASize + basicTiling_.depthB1 * baseScaleBSize));
    if (basicTiling_.scaleFactorA <= scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(basicTiling_.scaleFactorA) * basicTiling_.depthA1 * baseScaleASize);
        basicTiling_.scaleFactorB = std::min(static_cast<uint32_t>(leftL1sie / (basicTiling_.depthB1 * baseScaleBSize)),
                                             basicTiling_.scaleFactorB);
    } else if (basicTiling_.scaleFactorB <= scaleInit && basicTiling_.scaleFactorA > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(basicTiling_.scaleFactorB) * basicTiling_.depthB1 * baseScaleBSize);
        basicTiling_.scaleFactorA = std::min(static_cast<uint32_t>(leftL1sie / (basicTiling_.depthA1 * baseScaleASize)),
                                             basicTiling_.scaleFactorA);
    } else if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(scaleInit) * basicTiling_.depthB1 * baseScaleBSize +
                      static_cast<uint64_t>(scaleInit) * basicTiling_.depthA1 * baseScaleASize);
        uint32_t scaleASec = std::min(static_cast<uint32_t>(leftL1sie / (basicTiling_.depthA1 * baseScaleASize)),
                                      basicTiling_.scaleFactorA - scaleInit);
        uint32_t scaleBSec = std::min(static_cast<uint32_t>(leftL1sie / (basicTiling_.depthB1 * baseScaleBSize)),
                                      basicTiling_.scaleFactorB - scaleInit);
        basicTiling_.scaleFactorA = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        basicTiling_.scaleFactorB = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

uint64_t Mc2AdaptiveSlidingWindowTiling::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE_V100) {
        return depthInit;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        uint64_t baseKSize = GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseK), inputParams_.aDtype);
        while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
               (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
            (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
            depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, static_cast<uint64_t>(1));
    } else {
        constexpr uint64_t index = 2; // 2: depth的值是2的幂
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}

uint64_t Mc2AdaptiveSlidingWindowTiling::GetDepthB1AfullLoad(uint64_t leftSize)
{
    // 内轴128B 对齐
    uint64_t stepKbBase = 1UL;
    if (inputParams_.transB) {
        uint64_t singleBaseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.bDtype);
        if (singleBaseKSize < L2_ALIGN_SIZE) {
            stepKbBase = ops::CeilDiv(L2_ALIGN_SIZE, singleBaseKSize);
        }
    }

    // 一次性的总搬运量不少于32KB，不超L1 内存
    uint64_t scaleBBaseK = 0UL;
    if (inputParams_.isMxPerGroup) {
        scaleBBaseK = ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE),
                                     MXFP_MULTI_BASE_SIZE);
    }

    uint64_t basebSize =
        GetSizeWithDataType(basicTiling_.baseN * (basicTiling_.baseK + scaleBBaseK) * stepKbBase, inputParams_.bDtype);
    uint64_t stepKbBaseScale = 1UL;
    if (leftSize >= MTE2_MIN_LOAD_SIZE_V100 * DB_SIZE) {
        stepKbBaseScale = ops::CeilDiv(MTE2_MIN_LOAD_SIZE_V100, basebSize);
    } else {
        stepKbBaseScale = ops::CeilDiv(leftSize / DB_SIZE, basebSize);
    }
    stepKbBase = stepKbBase * stepKbBaseScale;

    uint64_t refinedStepkb = 2UL; // stekpb为1时容易使mte1无法并行，因此在不超出l1 空间的情况下设置stepkb最小为2
    if (stepKbBase == 1UL && inputParams_.kSize > static_cast<uint64_t>(basicTiling_.baseK) &&
        leftSize > basebSize * refinedStepkb) {
        stepKbBase = refinedStepkb;
    }

    return stepKbBase * DB_SIZE;
}

uint64_t Mc2AdaptiveSlidingWindowTiling::GetScaleFactorBAfullLoad(uint64_t leftSize)
{
    uint64_t baseScaleBSize = GetSizeWithDataType(
        basicTiling_.baseN * ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE),
                                            MXFP_MULTI_BASE_SIZE),
        inputParams_.scaleDtype);

    uint64_t scaleFactorbBase = 1UL;
    if (inputParams_.transB) {
        // k 是内轴的情况
        uint64_t singleScalebBasekSize =
            GetSizeWithDataType(ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE),
                                               MXFP_MULTI_BASE_SIZE),
                                inputParams_.scaleDtype);
        if (singleScalebBasekSize < L2_ALIGN_SIZE) {
            scaleFactorbBase = ops::CeilDiv(L2_ALIGN_SIZE, singleScalebBasekSize);
        }
    }

    uint64_t scaleFactorbMaxFromK =
        inputParams_.kSize / static_cast<uint64_t>(basicTiling_.stepKb * basicTiling_.baseK);
    scaleFactorbMaxFromK = std::min(static_cast<uint64_t>(SCALER_FACTOR_MAX), scaleFactorbMaxFromK);
    scaleFactorbMaxFromK = std::max(static_cast<uint64_t>(SCALER_FACTOR_MIN), scaleFactorbMaxFromK);

    uint64_t scaleFactorB = 1;
    if (scaleFactorbBase > scaleFactorbMaxFromK) {
        // 一次性搬运所有的K 方向上所有的scaleB 也无法满足内轴128对齐,按照搬运量来计算scaleFactorb
        uint64_t scaleFactorBMax =
            std::min(MTE2_MIN_LOAD_SIZE_V100 * DB_SIZE, leftSize) / (baseScaleBSize * basicTiling_.depthB1);
        scaleFactorB = std::min(scaleFactorBMax, scaleFactorbMaxFromK);
    } else {
        // 在内轴128对齐的基础上，同时确保搬运量不小于32， 且不超L1 内存
        uint64_t scaleFactor = std::min(MTE2_MIN_LOAD_SIZE_V100 * DB_SIZE, leftSize) /
                               (baseScaleBSize * scaleFactorbBase * basicTiling_.depthB1);
        scaleFactorB = std::min(scaleFactor * scaleFactorbBase, scaleFactorbMaxFromK);
    }

    return scaleFactorB;
}

void Mc2AdaptiveSlidingWindowTiling::SetTilingData()
{
    tilingData_.params.batchA = inputParams_.batchA;
    tilingData_.params.batchB = inputParams_.batchB;
    tilingData_.params.batchC = inputParams_.batchC;
    tilingData_.params.batchA1 = inputParams_.batchA1;
    tilingData_.params.batchA2 = inputParams_.batchA2;
    tilingData_.params.batchA3 = inputParams_.batchA3;
    tilingData_.params.batchA4 = inputParams_.batchA4;
    tilingData_.params.batchB1 = inputParams_.batchB1;
    tilingData_.params.batchB2 = inputParams_.batchB2;
    tilingData_.params.batchB3 = inputParams_.batchB3;
    tilingData_.params.batchB4 = inputParams_.batchB4;
    tilingData_.params.batchC1 = inputParams_.batchC1;
    tilingData_.params.batchC2 = inputParams_.batchC2;
    tilingData_.params.batchC3 = inputParams_.batchC3;
    tilingData_.params.batchC4 = inputParams_.batchC4;
    tilingData_.params.ubCalcN = basicTiling_.ubCalcN;
    tilingData_.params.ubCalcM = basicTiling_.ubCalcM;
    tilingData_.params.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    tilingData_.params.isPerTensor = static_cast<uint32_t>(inputParams_.isPerTensor);
    tilingData_.params.isPertoken = static_cast<uint32_t>(inputParams_.isPertoken);
    tilingData_.params.isDoubleScale = static_cast<uint32_t>(inputParams_.isDoubleScale);
    tilingData_.params.biasThreeDim = static_cast<uint32_t>(inputParams_.batchBias > 1U);
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.params.groupSizeM = static_cast<uint32_t>(inputParams_.groupSizeM);
    tilingData_.params.groupSizeN = static_cast<uint32_t>(inputParams_.groupSizeN);
    tilingData_.params.groupSizeK = static_cast<uint32_t>(inputParams_.groupSizeK);
}

uint32_t Mc2AdaptiveSlidingWindowTiling::CalUsedCoreNum()
{
    if (adaptiveWin_.totalWinCnt > 1UL || adaptiveWin_.tailWinBlockCnt == 0UL) {
        return aicoreParams_.aicNum;
    }

    return static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt * adaptiveWin_.mTailTile * adaptiveWin_.nTailTile);
}

uint32_t Mc2AdaptiveSlidingWindowTiling::CalUsedCoreNum(uint32_t mTile, uint32_t nTile)
{
    return mTile * nTile * static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt);
}

bool Mc2AdaptiveSlidingWindowTiling::IsInValidPerblockTailSplit(uint64_t splitCnt) const
{
    if (!inputParams_.isPerBlock) {
        return false;
    }

    return PER_BLOCK_SIZE % splitCnt != 0UL;
}

bool Mc2AdaptiveSlidingWindowTiling::IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const
{
    if (inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ ||
        (!isAFullLoad_ && ((isPreSplit && adaptiveWin_.mTail >= adaptiveWin_.nTail) ||
                           (!isPreSplit && adaptiveWin_.mTail < adaptiveWin_.nTail)))) {
        return false;
    }

    uint64_t tailN = adaptiveWin_.baseN / splitCnt;
    return tailN % GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype) != 0UL;
}

void Mc2AdaptiveSlidingWindowTiling::CalcTailBasicBlock()
{
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    auto &preSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? mTile : nTile;
    auto &secSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? nTile : mTile;
    while (CalUsedCoreNum(preSplit + 1, secSplit) <= aicoreParams_.aicNum) {
        preSplit += 1UL;
        if (!IsInValidPerblockTailSplit(preSplit) && !IsInValidWeighNzTailSplit(preSplit, true)) {
            preSplitValid = preSplit;
        }

        if (CalUsedCoreNum(preSplit, secSplit + 1) <= aicoreParams_.aicNum) {
            secSplit += 1UL;
            if (!IsInValidPerblockTailSplit(secSplit) && !IsInValidWeighNzTailSplit(secSplit, false)) {
                secSplitValid = secSplit;
            }
        }
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
}

void Mc2AdaptiveSlidingWindowTiling::CalcTailBasicBlockAfullLoad()
{
    adaptiveWin_.mTailTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t nTileValid = 1UL;
    constexpr uint64_t MIN_BASEN_PER_TILE = 16UL; // 尾窗口切分后N方向不少于16
    if (adaptiveWin_.tailWinBlockCnt != 0UL) {
        while (CalUsedCoreNum(adaptiveWin_.mTailTile, (nTile + 1UL)) <= aicoreParams_.aicNum &&
               adaptiveWin_.baseN / (nTile + 1UL) >= MIN_BASEN_PER_TILE) {
            nTile += 1UL;
            if (IsInValidWeighNzTailSplit(nTile, true)) {
                continue;
            }
            nTileValid = nTile;
        }
    }
    adaptiveWin_.nTailTile = nTileValid;
}

} // namespace optiling
