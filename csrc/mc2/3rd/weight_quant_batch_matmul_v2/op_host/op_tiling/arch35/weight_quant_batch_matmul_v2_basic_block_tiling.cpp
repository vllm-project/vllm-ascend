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
 * \file weight_quant_batch_matmul_v2_basic_block_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_basic_block_tiling.h"

namespace {
int64_t DownwardFactor(int64_t value, const int64_t base)
{
    if (value == 0 || base % value == 0) {
        return value;
    }

    while (--value > 1) {
        if (base % value == 0) {
            return value;
        }
    }
    return value; // This will return 1 if no other divisor is found
}
} // namespace

namespace optiling {

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::Init()
{
    opName_ = nullptr;
    aByteSize_ = 0.0;
    bByteSize_ = 0.0;
    biasByteSize_ = 0.0;
    antiquantType_ = Mc2QuantType::PER_CHANNEL;
    hasOffset_ = false;

    mte2BoundResults_.clear();
    cubeBoundResults_.clear();

    basicBlockParam_.mSize = 1;
    basicBlockParam_.nSize = 1;
    basicBlockParam_.kSize = 1;
    basicBlockParam_.kDim = 1;
    basicBlockParam_.singleK = 1;
    basicBlockParam_.groupSize = 0;
    basicBlockParam_.aDtypeBits = 0;
    basicBlockParam_.bDtypeBits = 0;
    basicBlockParam_.biasDtypeBits = 0;
    basicBlockParam_.transA = false;
    basicBlockParam_.transB = true;
    basicBlockParam_.hasBias = false;
    basicBlockParam_.weightNzFlag = false;

    InitBasicBlockParam();
    InitPlatformParam();
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::InitPlatformParam()
{
    platformParam_.blockNum = 0;
    platformParam_.aicNum = 0;
    platformParam_.ubSize = 0;
    platformParam_.l1Size = 0;
    platformParam_.l0aSize = 0;
    platformParam_.l0bSize = 0;
    platformParam_.l0cSize = 0;
    platformParam_.cacheLine = 0;
    platformParam_.minCacheLine = 0;

    platformParam_.frequency = 0;
    platformParam_.hbmBW = 0;
    platformParam_.l2BW = 0;
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::InitL1TilingParam()
{
    basicBlockParam_.l1Param.iterateOrder = 1;
    basicBlockParam_.l1Param.stepM = 1;
    basicBlockParam_.l1Param.stepN = 1;
    basicBlockParam_.l1Param.stepKa = 1;
    basicBlockParam_.l1Param.stepKb = 1;
    basicBlockParam_.l1Param.A1BufferNum = 1;
    basicBlockParam_.l1Param.B1BufferNum = 1;
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::InitBasicBlockParam()
{
    // mSize, nSize, kSize, singleK 采用SetShape传入的结果
    basicBlockParam_.singleM = 1;
    basicBlockParam_.singleN = 1;
    basicBlockParam_.mDim = 1;
    basicBlockParam_.nDim = 1;
    basicBlockParam_.mte2DataSize = 0;
    basicBlockParam_.fixpDataSize = 0;

    basicBlockParam_.basicBlock.baseM = 1;
    basicBlockParam_.basicBlock.baseN = 1;
    basicBlockParam_.basicBlock.baseK = 1;
    basicBlockParam_.basicBlock.mte2BW = 0;
    basicBlockParam_.basicBlock.mte2MinBW = 0;
    basicBlockParam_.basicBlock.mte2BWRatio = 0;
    basicBlockParam_.basicBlock.mte2TailBWRatio = 0;

    InitL1TilingParam();
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::Reset()
{
    InitBasicBlockParam();
    mte2BoundResults_.clear();
    cubeBoundResults_.clear();
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::SetPlatformParam(const PlatformParam &param)
{
    platformParam_ = param;
    OP_LOGI(opName_,
            "Platform info: blockNum: %ld, aicNum: %ld, ubSize: %ld, l1Size: %ld, l0aSize: %ld, l0bSize: %ld, l0cSize: "
            "%ld, cacheLine: %ld, minCacheLine: %ld, frequency: %lf, hbmBW: %lf, l2BW: %lf",
            platformParam_.blockNum, platformParam_.aicNum, platformParam_.ubSize, platformParam_.l1Size,
            platformParam_.l0aSize, platformParam_.l0bSize, platformParam_.l0cSize, platformParam_.cacheLine,
            platformParam_.minCacheLine, platformParam_.frequency, platformParam_.hbmBW, platformParam_.l2BW);
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::SetShape(int64_t mSize, int64_t nSize, int64_t kSize,
                                                           int64_t groupSize)
{
    basicBlockParam_.mSize = mSize;
    basicBlockParam_.nSize = nSize;
    basicBlockParam_.kSize = kSize;
    basicBlockParam_.singleK = kSize;
    basicBlockParam_.groupSize = groupSize;
    OP_LOGI(opName_, "Init shape param, mSize: %ld, nSize: %ld, kSize: %ld, groupSize: %ld", basicBlockParam_.mSize,
            basicBlockParam_.nSize, basicBlockParam_.kSize, basicBlockParam_.groupSize);
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::SetAttr(const char *opName, const WeightQuantBmmAttr &attr)
{
    opName_ = opName;
    basicBlockParam_.transA = attr.transA;
    basicBlockParam_.transB = attr.transB;
    basicBlockParam_.hasBias = attr.hasBias;
    basicBlockParam_.weightNzFlag = attr.weightNzFlag;
    hasOffset_ = attr.hasOffset;
    OP_LOGI(opName_, "Init attr param, transA: %s, transB: %s, hasBias: %s, weightNzFlag: %s, hasOffset: %s",
            basicBlockParam_.transA ? "true" : "false", basicBlockParam_.transB ? "true" : "false",
            basicBlockParam_.hasBias ? "true" : "false", basicBlockParam_.weightNzFlag ? "true" : "false",
            hasOffset_ ? "true" : "false");
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::SetDtypeBits(int64_t aDtypeBits, int64_t bDtypeBits,
                                                               int64_t biasDtypeBits)
{
    basicBlockParam_.aDtypeBits = aDtypeBits;
    basicBlockParam_.bDtypeBits = bDtypeBits;
    basicBlockParam_.biasDtypeBits = biasDtypeBits;
    aByteSize_ = aDtypeBits / BYTE_BITS;
    bByteSize_ = bDtypeBits / BYTE_BITS;
    biasByteSize_ = biasDtypeBits / BYTE_BITS;
    OP_LOGI(opName_, "Init byteSize, aByteSize_: %lf, bByteSize_: %lf, biasByteSize_: %lf", aByteSize_, bByteSize_,
            biasByteSize_);
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::SetQuantType(Mc2QuantType antiquantType)
{
    antiquantType_ = antiquantType;
    OP_LOGI(opName_, "Init antiquantType_: %d", static_cast<int>(antiquantType_));
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::ValidateInputParam() const
{
    OP_TILING_CHECK(basicBlockParam_.mSize <= 0 || basicBlockParam_.nSize <= 0 || basicBlockParam_.kSize <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "shape size",
                                                          (std::to_string(basicBlockParam_.mSize) + ", " +
                                                           std::to_string(basicBlockParam_.nSize) + ", " +
                                                           std::to_string(basicBlockParam_.kSize))
                                                              .c_str(),
                                                          "The value of shape size must be greater than 0."),
                    return false);

    OP_TILING_CHECK(basicBlockParam_.aDtypeBits <= 0 || basicBlockParam_.bDtypeBits <= 0 ||
                        (basicBlockParam_.hasBias && basicBlockParam_.biasDtypeBits <= 0) ||
                        (!basicBlockParam_.hasBias && basicBlockParam_.biasDtypeBits > 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "dtypeBits",
                                                          (std::to_string(basicBlockParam_.aDtypeBits) + ", " +
                                                           std::to_string(basicBlockParam_.bDtypeBits) + ", " +
                                                           std::to_string(basicBlockParam_.biasDtypeBits))
                                                              .c_str(),
                                                          "The value of dtypeBits must be greater than 0."),
                    return false);

    OP_TILING_CHECK(basicBlockParam_.groupSize < 0 || basicBlockParam_.groupSize >= basicBlockParam_.kSize,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "groupSize",
                                                          std::to_string(basicBlockParam_.groupSize).c_str(),
                                                          (std::string("The value of groupSize must be >= 0 and < K(") +
                                                           std::to_string(basicBlockParam_.kSize) + ").")
                                                              .c_str()),
                    return false);

    return true;
}

double Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetMinMte2BW(int64_t baseM, int64_t baseN, int64_t mDim,
                                                                 int64_t nDim) const
{
    if (mDim * nDim * baseM * baseN == 0) {
        return 0;
    }
    // 求V100平台A16W8在给定基本块条件下达到cube bound要求的最低MTE2带宽
    // 0.001含义：将计算结果转换为TB/s
    return BLOCK_CUBE * BLOCK_CUBE * BLOCK_CUBE * platformParam_.frequency * 0.001 * static_cast<double>(mDim) *
           static_cast<double>(nDim) *
           (aByteSize_ / static_cast<double>(baseN) + bByteSize_ / static_cast<double>(baseM));
}

double Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetMte2BW(int64_t baseM, int64_t baseN, int64_t mDim,
                                                              int64_t nDim) const
{
    // 估算V100平台当前分核条件下的综合MTE2带宽
    if (platformParam_.hbmBW <= 0.0 || platformParam_.l2BW <= 0.0 || platformParam_.blockNum <= 0L) {
        OP_LOGE(
            opName_,
            "Invalid platformParam_, hbmBW/l2BW/blockNum must be greater than 0, in fact they are hbmBW: %lf, l2BW: "
            "%lf, blockNum: %ld",
            platformParam_.hbmBW, platformParam_.l2BW, platformParam_.blockNum);
        return 0;
    }
    double denominator =
        (static_cast<double>(mDim * baseM) * aByteSize_ + static_cast<double>(nDim * baseN) * bByteSize_) /
            platformParam_.hbmBW +
        (static_cast<double>(mDim * nDim - mDim) * static_cast<double>(baseM) * aByteSize_ +
         static_cast<double>(baseN) * static_cast<double>(mDim * nDim - nDim) * bByteSize_) /
            platformParam_.l2BW;
    if (denominator <= 0.0) {
        OP_LOGE(opName_, "Invalid denominator, denominator must be greater than 0, in fact it is denominator: %lf",
                denominator);
        return 0;
    }
    double res = (static_cast<double>(mDim * nDim) *
                  (static_cast<double>(baseM) * aByteSize_ + bByteSize_ * static_cast<double>(baseN))) /
                 denominator * (static_cast<double>(mDim * nDim) / static_cast<double>(platformParam_.blockNum));
    return res;
}

double Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetMte2BWRatio(int64_t baseM, int64_t baseN, int64_t mDim,
                                                                   int64_t nDim) const
{
    if (GetMinMte2BW(baseM, baseN, mDim, nDim) > 0) {
        return GetMte2BW(baseM, baseN, mDim, nDim) / GetMinMte2BW(baseM, baseN, mDim, nDim);
    }
    return 0;
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetCachelineAlignFlag(int64_t dtypeBits, int64_t cacheline) const
{
    // 具备cachline对齐的条件：
    // 1）singleK大于cachline；
    // 2）非group场景；
    // 3）group场景，由于要满足kBL1与groupSize的对齐关系，因此需满足groupSize是cacheline的倍数或因子；
    bool cachelineAlignFlag =
        (basicBlockParam_.singleK * dtypeBits > cacheline * BITS_8) &&
        (basicBlockParam_.groupSize == 0 || (basicBlockParam_.groupSize * dtypeBits % (cacheline * BITS_8) == 0 ||
                                             cacheline * BITS_8 % (basicBlockParam_.groupSize * dtypeBits) == 0));
    return cachelineAlignFlag;
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::UpdateMte2DataSize()
{
    // 给定单核L1、L0切分条件，计算MTE2总搬运量
    int64_t singleMLoop =
        CeilDiv(basicBlockParam_.singleM, basicBlockParam_.l1Param.stepM * basicBlockParam_.basicBlock.baseM);
    int64_t singleNLoop =
        CeilDiv(basicBlockParam_.singleN, basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN);
    bool A1KFullLoadFlag =
        basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKa >= basicBlockParam_.singleK;
    bool B1KFullLoadFlag =
        basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKb >= basicBlockParam_.singleK;

    if (A1KFullLoadFlag && B1KFullLoadFlag) {
        int64_t mte2DataSizeOrderM =
            static_cast<int64_t>(basicBlockParam_.singleM * basicBlockParam_.singleK * aByteSize_ +
                                 singleMLoop * basicBlockParam_.singleN * basicBlockParam_.singleK * bByteSize_);
        int64_t mte2DataSizeOrderN =
            static_cast<int64_t>(basicBlockParam_.singleN * basicBlockParam_.singleK * bByteSize_ +
                                 singleNLoop * basicBlockParam_.singleM * basicBlockParam_.singleK * aByteSize_);
        basicBlockParam_.mte2DataSize = std::min(mte2DataSizeOrderN, mte2DataSizeOrderM);
        basicBlockParam_.l1Param.iterateOrder = mte2DataSizeOrderM < mte2DataSizeOrderN ? 0 : 1;
    } else if (A1KFullLoadFlag) {
        basicBlockParam_.mte2DataSize =
            static_cast<int64_t>(basicBlockParam_.singleM * basicBlockParam_.singleK * aByteSize_ +
                                 singleMLoop * basicBlockParam_.singleN * basicBlockParam_.singleK * bByteSize_);
    } else if (B1KFullLoadFlag) {
        basicBlockParam_.mte2DataSize =
            static_cast<int64_t>(basicBlockParam_.singleN * basicBlockParam_.singleK * bByteSize_ +
                                 singleNLoop * basicBlockParam_.singleM * basicBlockParam_.singleK * aByteSize_);
    } else {
        basicBlockParam_.mte2DataSize =
            static_cast<int64_t>(singleMLoop * basicBlockParam_.singleN * basicBlockParam_.singleK * bByteSize_ +
                                 singleNLoop * basicBlockParam_.singleM * basicBlockParam_.singleK * aByteSize_);
    }
    basicBlockParam_.mte2DataSize *= basicBlockParam_.mDim * basicBlockParam_.nDim;
}

int64_t Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetUbLoadSize() const
{
    int64_t ubLoadSize = 0;
    if (!basicBlockParam_.weightNzFlag && basicBlockParam_.groupSize > 0) {
        int64_t innerDimAlignSize = UB_ALIGN_SIZE * BITS_8 / basicBlockParam_.bDtypeBits;
        int64_t weightInnerDim =
            basicBlockParam_.transB ?
                CeilAlign(basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKb, innerDimAlignSize) :
                CeilAlign(basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN, innerDimAlignSize);
        if (basicBlockParam_.bDtypeBits == BITS_4 && basicBlockParam_.transB &&
            basicBlockParam_.groupSize > GROUP_SIZE_64 && basicBlockParam_.groupSize % GROUP_SIZE_64 > 0 &&
            weightInnerDim > basicBlockParam_.groupSize) {
            // 96含义：在W4 NK且gs非64对齐场景，跨gs计算的长度为96，因此至少保证内轴长度大等于gs+96
            weightInnerDim = std::max(weightInnerDim, basicBlockParam_.groupSize + 96);
        }
        int64_t weightOuterDim =
            basicBlockParam_.transB ?
                CeilDiv(basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN, BUFF_NUM_2) :
                CeilDiv(basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKb, BUFF_NUM_2);
        int64_t scaleSize =
            basicBlockParam_.transB ?
                CeilDiv(basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN, BUFF_NUM_2) *
                    CeilAlign(CeilDiv(basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKb,
                                      basicBlockParam_.groupSize),
                              BLOCK_CUBE) :
                CeilDiv(CeilDiv(basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKb, BUFF_NUM_2),
                        basicBlockParam_.groupSize) *
                    basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN;
        // ND场景采用4-buffer方案，包含最多4份weightIn\offset\scale，最多2份weightOut
        ubLoadSize =
            basicBlockParam_.l1Param.B1BufferNum * weightOuterDim * weightInnerDim * basicBlockParam_.bDtypeBits +
            std::min(basicBlockParam_.l1Param.B1BufferNum, BUFF_NUM_2) * (CeilAlign(weightOuterDim, BLOCK_CUBE) + 1) *
                weightInnerDim * basicBlockParam_.aDtypeBits +
            basicBlockParam_.l1Param.B1BufferNum * scaleSize * basicBlockParam_.aDtypeBits *
                (static_cast<int64_t>(hasOffset_) + 1);
    }
    return ubLoadSize;
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetInvalidFlagForBasicBlock() const
{
    // 128: reg base寄存器处理B16最大元素数量
    constexpr int64_t REG_WIDTH = 128;
    // 剪枝A16W8 ND-perGroup KN场景部分baseN不整除VL的情况
    bool invalidFlag =
        basicBlockParam_.bDtypeBits == BITS_8 && basicBlockParam_.groupSize > 0 && !basicBlockParam_.weightNzFlag &&
        !basicBlockParam_.transB &&
        ((basicBlockParam_.singleN > REG_WIDTH && basicBlockParam_.singleN > basicBlockParam_.basicBlock.baseN &&
          basicBlockParam_.basicBlock.baseN % REG_WIDTH > 0) ||
         (basicBlockParam_.singleN <= REG_WIDTH && basicBlockParam_.singleN > basicBlockParam_.basicBlock.baseN));
    invalidFlag = invalidFlag || (basicBlockParam_.basicBlock.baseM <= 0 || basicBlockParam_.basicBlock.baseK <= 0 ||
                                  basicBlockParam_.basicBlock.baseN <= 0);
    return invalidFlag;
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetInvalidFlagA16W4() const
{
    if (basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN <= 0 || basicBlockParam_.singleN <= 0 ||
        basicBlockParam_.singleK <= 0 || basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK <= 0) {
        return true;
    }

    bool invalidFlag;
    if (basicBlockParam_.weightNzFlag) {
        // a16w4场景固定采用4-buffer方案，要求单块BL1空间最多分出两份ub任务（优先在N1轴切分），
        // 根据ub单次最大处理量12KB可反算出单份BL1最大空间
        invalidFlag =
            CeilAlign(basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN, BLOCK_CUBE * BUFF_NUM_2) *
                basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK * aByteSize_ >
            A16W4_MAX_BL1_SIZE_NZ;
        int64_t nBl1Tail =
            basicBlockParam_.singleN % (basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN);
        int64_t nBl1TailBlockTail = basicBlockParam_.nSize % basicBlockParam_.singleN %
                                    (basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN);
        int64_t nBubSize = CeilAlign(
            CeilDiv(basicBlockParam_.basicBlock.baseN * basicBlockParam_.l1Param.stepN, BUFF_NUM_2), BLOCK_CUBE);
        // a16w4场景要求BL1尾块可分给两个vector处理
        invalidFlag = invalidFlag || ((nBl1Tail > 0 && nBl1Tail <= nBubSize) ||
                                      (nBl1TailBlockTail > 0 && nBl1TailBlockTail <= nBubSize));
    } else {
        invalidFlag = GetUbLoadSize() > platformParam_.ubSize * BITS_8;
        // 根据内轴是否minCacheline对齐做初步剪枝
        if (basicBlockParam_.transB) {
            // 当singleK大于minCacheline且groupSize为minCacheline的倍数或因子时，kBL1具备minCacheline对齐的条件，
            // 此时剪枝掉非全载并且非对齐情况
            invalidFlag =
                invalidFlag ||
                (GetCachelineAlignFlag(basicBlockParam_.bDtypeBits, platformParam_.minCacheLine) &&
                 (basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK * basicBlockParam_.bDtypeBits) %
                         (platformParam_.minCacheLine * BITS_8) >
                     0 &&
                 basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK < basicBlockParam_.singleK);
            const int64_t kBl1Size =
                std::min(basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKb, basicBlockParam_.singleK);
            const int64_t kBl1TailSize = basicBlockParam_.singleK % kBl1Size;
            const int64_t groupPairSize = BUFF_NUM_2 * basicBlockParam_.groupSize;
            // 对于groupSize非64对齐场景，VF中以2*groupSize为单位进行处理，需过滤kBL1或kBL1Tail为奇数倍groupSize的场景
            invalidFlag =
                invalidFlag ||
                (basicBlockParam_.groupSize > GROUP_SIZE_64 && basicBlockParam_.groupSize % GROUP_SIZE_64 > 0 &&
                 ((kBl1Size > basicBlockParam_.groupSize && kBl1Size % groupPairSize > 0) ||
                  (kBl1TailSize > basicBlockParam_.groupSize && kBl1TailSize % groupPairSize > 0 &&
                   kBl1TailSize % groupPairSize <= basicBlockParam_.groupSize)));
        } else {
            invalidFlag =
                invalidFlag ||
                ((basicBlockParam_.singleN * basicBlockParam_.bDtypeBits > platformParam_.minCacheLine * BITS_8) &&
                 (basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN * basicBlockParam_.bDtypeBits) %
                         (platformParam_.minCacheLine * BITS_8) >
                     0 &&
                 basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN < basicBlockParam_.singleN);
        }
    }
    return invalidFlag;
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetInvalidFlagA16W8() const
{
    bool invalidFlag = false;
    if (!basicBlockParam_.weightNzFlag && basicBlockParam_.groupSize > 0) {
        invalidFlag = GetUbLoadSize() > platformParam_.ubSize * BITS_8;
        if (basicBlockParam_.transB) {
            invalidFlag =
                invalidFlag ||
                (GetCachelineAlignFlag(basicBlockParam_.bDtypeBits, platformParam_.minCacheLine) &&
                 (basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK * basicBlockParam_.bDtypeBits) %
                         (platformParam_.minCacheLine * BITS_8) >
                     0 &&
                 basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK < basicBlockParam_.singleK);
        } else {
            invalidFlag =
                invalidFlag ||
                ((basicBlockParam_.singleN * basicBlockParam_.bDtypeBits > platformParam_.minCacheLine * BITS_8) &&
                 (basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN * basicBlockParam_.bDtypeBits) %
                         (platformParam_.minCacheLine * BITS_8) >
                     0 &&
                 basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN < basicBlockParam_.singleN);
        }
    } else if (!basicBlockParam_.transB) {
        // A16W8 B非转置场景 需过滤掉nbubsize非32B对齐的解和超ub大小的解
        int64_t kBl1Size =
            std::min(basicBlockParam_.singleK, basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK);
        int64_t nBl1Size =
            std::min(basicBlockParam_.singleN, basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN);
        // 过滤singleN和stepN * baseN非32B对齐的解，确保nBl1 = min(singleN,stepN * baseN)为32B对齐
        invalidFlag = basicBlockParam_.singleN % UB_ALIGN_SIZE != 0 ||
                      (basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN) % UB_ALIGN_SIZE != 0;
        // 比较使用UB空间大小,公式 BufferNum * ((nbub * kbub) * (aByteSize + bByteSize) + Scale + Offset) > ubsize
        int64_t antiquantSize = 0;
        if (antiquantType_ == Mc2QuantType::PER_TENSOR) {
            antiquantSize = 0;
        } else {
            antiquantSize = static_cast<int64_t>(nBl1Size * aByteSize_ + nBl1Size * aByteSize_);
        }
        invalidFlag = invalidFlag || basicBlockParam_.l1Param.B1BufferNum *
                                             (nBl1Size * CeilAlign(CeilDiv(kBl1Size, BUFF_NUM_2), BLOCK_CUBE) *
                                                  (aByteSize_ + bByteSize_) +
                                              antiquantSize) >
                                         platformParam_.ubSize;
    }
    return invalidFlag;
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetInvalidFlag(bool isCubeBoundSolution, int64_t stepKMax) const
{
    bool invalidFlag = GetL1LoadSize(basicBlockParam_.basicBlock, basicBlockParam_.l1Param) > platformParam_.l1Size;
    invalidFlag =
        invalidFlag ||
        (basicBlockParam_.groupSize > 0 &&
         (basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK) % basicBlockParam_.groupSize > 0 &&
         basicBlockParam_.groupSize % (basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK) > 0);
    // group-ND且非转置场景，进一步保证kBL1为groupsize的偶数倍或因子，避免group跨vector
    invalidFlag = invalidFlag ||
                  (basicBlockParam_.groupSize > 0 && !basicBlockParam_.weightNzFlag && !basicBlockParam_.transB &&
                   basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK > basicBlockParam_.groupSize &&
                   basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK %
                           (basicBlockParam_.groupSize * BUFF_NUM_2) >
                       0);
    invalidFlag =
        invalidFlag || (basicBlockParam_.l1Param.stepKa < stepKMax && basicBlockParam_.l1Param.stepKb < stepKMax &&
                        basicBlockParam_.l1Param.stepKa % basicBlockParam_.l1Param.stepKb > 0 &&
                        basicBlockParam_.l1Param.stepKb % basicBlockParam_.l1Param.stepKa > 0);
    invalidFlag =
        invalidFlag ||
        (isCubeBoundSolution && !basicBlockParam_.weightNzFlag && basicBlockParam_.transB &&
         basicBlockParam_.groupSize > GROUP_SIZE_64 && basicBlockParam_.groupSize % GROUP_SIZE_64 > 0 &&
         // NK-per-group且groupSize非64对齐场景，cube-bound选解阶段去除kBL1过小（小于512）的解以保证MTE2效率
         basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.stepKb < std::min(512L, basicBlockParam_.kSize));
    invalidFlag = invalidFlag || (basicBlockParam_.bDtypeBits == BITS_8 && GetInvalidFlagA16W8());
    invalidFlag = invalidFlag || (basicBlockParam_.bDtypeBits == BITS_4 && GetInvalidFlagA16W4());
    return invalidFlag;
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetL1Param(bool isCubeBoundSolution, int64_t stepKMax,
                                                             int64_t stepKaTmp, int64_t stepKbTmp)
{
    int64_t a1BufferNum, b1BufferNum;
    if (basicBlockParam_.groupSize > 0) {
        int64_t a1BufferNumMax =
            (basicBlockParam_.bDtypeBits == BITS_4 && basicBlockParam_.weightNzFlag) ? BUFF_NUM_4 : BUFF_NUM_2;
        a1BufferNum = std::min(CeilDiv(stepKMax, stepKaTmp) *
                                   CeilDiv(basicBlockParam_.singleM, basicBlockParam_.basicBlock.baseM),
                               a1BufferNumMax);
        if (CeilDiv(basicBlockParam_.singleM, basicBlockParam_.basicBlock.baseM) == 1 &&
            CeilDiv(stepKMax, stepKaTmp) <= a1BufferNumMax) {
            a1BufferNum = 1;
            stepKaTmp = stepKMax;
        }
        b1BufferNum = std::min(CeilDiv(stepKMax, stepKbTmp) *
                                   CeilDiv(basicBlockParam_.singleN, basicBlockParam_.basicBlock.baseN),
                               BUFF_NUM_4);
        // 在NK-per-group且K较小场景（K<1024），将cube
        // bound解改为kBL1全载解，以保证scale/offset内轴全载，避免其MTE2性能劣化
        if (isCubeBoundSolution && !basicBlockParam_.weightNzFlag && basicBlockParam_.groupSize > 0 &&
            basicBlockParam_.transB && basicBlockParam_.kSize < 1024L) {
            stepKbTmp = stepKMax;
            b1BufferNum = std::min(CeilDiv(basicBlockParam_.singleN, basicBlockParam_.basicBlock.baseN), BUFF_NUM_4);
        }
        // 当前kernel不支持BL1 3buffer，AL1 3buffer情况极少出现且采用2buffer代替后性能无明显劣化，故直接去掉3-buffer情况
        a1BufferNum = a1BufferNum == BUFF_NUM_3 ? BUFF_NUM_2 : a1BufferNum;
        b1BufferNum = b1BufferNum == BUFF_NUM_3 ? BUFF_NUM_2 : b1BufferNum;
    } else {
        a1BufferNum = stepKaTmp == stepKMax ? 1 : BUFF_NUM_2;
        b1BufferNum = stepKbTmp == stepKMax ? 1 : BUFF_NUM_2;
    }
    // 参数含义分别为iterateOrder，stepM，stepN，stepKa，stepKb，A1BufferNum，B1BufferNum
    basicBlockParam_.l1Param = {1, 1, 1, stepKaTmp, stepKbTmp, a1BufferNum, b1BufferNum};
}

/*
 *  给定分核及基本块，求解L1 tiling。
 *  为优化scalar，当前仅考虑stepM=1且stepN=1场景。遍历stepKa及stepKb组合，根据不同场景筛选出合法组合，并根据mte2Cost最小原则
 *  添加到解集中。特别地，在A16W4 NZ场景，当L1 tiling满足特定载入量时直接添加到解集。
 */
void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::DoL1Tiling(bool isCubeBoundSolution)
{
    int64_t stepKMax = CeilDiv(basicBlockParam_.singleK, basicBlockParam_.basicBlock.baseK);
    int64_t stepKaMax =
        std::min(CeilDiv(stepKMax, BUFF_NUM_2),
                 CeilDiv(platformParam_.l1Size, static_cast<int64_t>(basicBlockParam_.basicBlock.baseM *
                                                                     basicBlockParam_.basicBlock.baseK * aByteSize_)));
    // A非转置且非group场景（group场景约束较多，可能无解），kAL1以cachline为粒度遍历
    int64_t stepKaMin =
        !basicBlockParam_.transA && GetCachelineAlignFlag(basicBlockParam_.aDtypeBits, platformParam_.cacheLine) ?
            std::min(
                CeilDiv(platformParam_.cacheLine, static_cast<int64_t>(basicBlockParam_.basicBlock.baseK * aByteSize_)),
                stepKaMax) :
            1;
    int64_t stepKbMinPruneLimit =
        std::min(CeilDiv(isCubeBoundSolution ? platformParam_.minCacheLine : platformParam_.cacheLine,
                         static_cast<int64_t>(basicBlockParam_.basicBlock.baseK * bByteSize_)),
                 stepKMax);
    stepKbMinPruneLimit = basicBlockParam_.groupSize > 0 && basicBlockParam_.transB ?
                              DownwardFactor(stepKbMinPruneLimit, stepKMax) :
                              stepKbMinPruneLimit;
    const int64_t stepKbMin = basicBlockParam_.transB && !basicBlockParam_.weightNzFlag &&
                                      GetCachelineAlignFlag(basicBlockParam_.bDtypeBits, platformParam_.cacheLine) ?
                                  stepKbMinPruneLimit :
                                  1;
    // 背景：per-group transB k较小场景，cacheline利用率低，ostd使用量高，导致mte2效率极低
    // 解决：提供kBL1全载解，并且根据issueque限制stepKb最大值取6
    int64_t stepKbMaxPruneLimit = basicBlockParam_.groupSize > 0 && basicBlockParam_.transB && stepKMax <= 6L ?
                                      std::min(stepKMax, 6L) :
                                      CeilDiv(stepKMax, BUFF_NUM_2);
    for (int64_t stepKaTmp = stepKaMin; stepKaTmp <= stepKaMax; stepKaTmp += stepKaMin) {
        int64_t stepKbMax =
            std::min(stepKbMaxPruneLimit,
                     CeilDiv(static_cast<int64_t>(platformParam_.l1Size - basicBlockParam_.basicBlock.baseM *
                                                                              basicBlockParam_.basicBlock.baseK *
                                                                              stepKaTmp * aByteSize_),
                             static_cast<int64_t>(basicBlockParam_.basicBlock.baseN *
                                                  basicBlockParam_.basicBlock.baseK * aByteSize_)));
        for (int64_t stepKbTmp = stepKbMin; stepKbTmp <= stepKbMax; stepKbTmp += stepKbMin) {
            GetL1Param(isCubeBoundSolution, stepKMax, stepKaTmp, stepKbTmp);
            if (GetInvalidFlag(isCubeBoundSolution, stepKMax)) {
                continue;
            }
            UpdateMte2DataSize();
            mte2BoundResults_.push_back(basicBlockParam_);
        }
    }
}

int64_t Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetBaseK(int64_t baseM, int64_t baseN) const
{
    // baseK选取能开启L0 DB的最大值(由于baseM/baseN最大取值512，因此baseK最小取值32)
    // baseK满足32对齐，方便MTE2 cache line对齐
    int64_t baseK = std::max(FloorAlign(BASE_MK_LIMIT / std::max(baseM, baseN), UB_ALIGN_SIZE), UB_ALIGN_SIZE);
    // 对于per-group场景，要求baseK为groupSize的倍数或因子
    if (basicBlockParam_.groupSize > 0) {
        if (basicBlockParam_.groupSize > baseK) {
            // baseK满足groupSize的因子，且为32的倍数，便于cacheline对齐
            while (baseK > UB_ALIGN_SIZE && basicBlockParam_.groupSize % baseK > 0) {
                baseK -= UB_ALIGN_SIZE;
            }
        } else {
            // baseK不仅满足groupSize的倍数，同时考虑bubLoadSize需满足4-buffer最大载入量，因此进一步取值最大载入量的因子
            baseK = baseK / basicBlockParam_.groupSize * basicBlockParam_.groupSize;
            int64_t alignSize = basicBlockParam_.bDtypeBits == BITS_4 && basicBlockParam_.weightNzFlag ?
                                    A16W4_MAX_BUB_ELEM_SIZE_NZ :
                                    A16W4_MAX_BUB_ELEM_SIZE_ND;
            while (alignSize % basicBlockParam_.groupSize == 0 && baseK > basicBlockParam_.groupSize &&
                   alignSize % baseK > 0) {
                baseK -= basicBlockParam_.groupSize;
            }
        }
        baseK = std::min(FloorAlign(CeilAlign(basicBlockParam_.kSize, BLOCK_CUBE), basicBlockParam_.groupSize), baseK);
        return baseK;
    }

    // 对于非per-group场景，要求baseK为256的因子，便于kL1 cacheline对齐
    if (baseK >= BASE_K_LIST[0]) {
        baseK = FloorAlign(baseK, BASE_K_LIST[0]);
    } else if (baseK > 0 && BASE_K_LIST[0] % baseK > 0) {
        for (const auto &tmpBaseK : BASE_K_LIST) {
            if (baseK >= tmpBaseK) {
                baseK = tmpBaseK;
                break;
            }
        }
    }
    baseK = std::min(CeilAlign(basicBlockParam_.kSize, BLOCK_CUBE), baseK);
    return baseK;
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetBasicBlockTable()
{
    // 给定singleM、singleN、mDim、nDim，获取可行基本块集
    for (int64_t baseM = BLOCK_CUBE; baseM <= basicBlockParam_.singleM; baseM += BLOCK_CUBE) {
        int64_t baseNMax = std::min(BASE_BLOCK_MAX, FloorAlign(BASE_MN_LIMIT / baseM, BLOCK_CUBE));
        baseNMax = std::min(basicBlockParam_.singleN, baseNMax);
        for (int64_t baseN = baseNMax; baseN >= BLOCK_CUBE; baseN -= BLOCK_CUBE) {
            int64_t baseK = GetBaseK(baseM, baseN);
            double minBW = GetMinMte2BW(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
            double curBW = GetMte2BW(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
            double mteBWRatio = GetMte2BWRatio(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
            basicBlockParam_.basicBlock = {baseM, baseN, baseK, curBW, minBW, mteBWRatio, 0.0};
            if (baseM * baseN > BASE_MN_LIMIT || baseM > BASE_BLOCK_MAX || baseN > BASE_BLOCK_MAX ||
                GetInvalidFlagForBasicBlock()) {
                continue;
            }
            DoL1Tiling(false);
        }
    }
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetHalfSingleShape(const std::vector<BasicBlock> &basicBlockTable,
                                                                     int64_t &halfSingleM, int64_t &halfSingleN)
{
    if (basicBlockTable.empty()) {
        return false;
    }

    // 获取基本块集中baseM、baseN的上下限
    // 备注：要求基本块集中，baseM升序排序、baseN降序排序
    const BasicBlock &frontBasicBlock = basicBlockTable.front();
    const BasicBlock &backBasicBlock = basicBlockTable.back();

    int64_t minBaseM, maxBaseN, maxBaseM, minBaseN;
    minBaseM = frontBasicBlock.baseM;
    maxBaseN = frontBasicBlock.baseN;
    maxBaseM = backBasicBlock.baseM;
    minBaseN = backBasicBlock.baseN;

    // 若单核shape小于可选基本块集下界，则认为此分核情况无解，后续可考虑适当放宽条件
    if (basicBlockParam_.singleM < minBaseM || basicBlockParam_.singleN < minBaseN) {
        return false;
    }

    while (halfSingleM > maxBaseM) {
        // 2的含义：若单核shape大于基本块集上界，则不断折半直到切分大小落在基本块集上、下界内
        halfSingleM = ((halfSingleM / BLOCK_CUBE + 1) / 2) * BLOCK_CUBE;
    }

    while (halfSingleN > maxBaseN) {
        // 2的含义：若单核shape大于基本块集上界，则不断折半直到切分大小落在基本块集上、下界内
        halfSingleN = ((halfSingleN / BLOCK_CUBE + 1) / 2) * BLOCK_CUBE;
    }

    return true;
}

/*
 *  单核切分：遍历给定分核条件下的可行基本块，根据条件添加到cube bound解集或mte2 bound解集。
 *  要求basicBlockParam_中mSize、nSize、kSize、singleM、singleN、mDim、nDim已设置；
 *  要求basicBlockTable_已存放可选基本块集；
 */
void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::SingleShapeTiling(const std::vector<BasicBlock> &basicBlockTable)
{
    InitL1TilingParam();
    int64_t halfSingleM = basicBlockParam_.singleM;
    int64_t halfSingleN = basicBlockParam_.singleN;

    if (!GetHalfSingleShape(basicBlockTable, halfSingleM, halfSingleN)) {
        return;
    }

    for (const BasicBlock &basicBlock : basicBlockTable) {
        basicBlockParam_.basicBlock = basicBlock;
        if (GetInvalidFlagForBasicBlock()) {
            continue;
        }
        // 其中baseK、tailBWRatio需要重新计算
        basicBlockParam_.basicBlock.baseK =
            GetBaseK(basicBlockParam_.basicBlock.baseM, basicBlockParam_.basicBlock.baseN);
        if (basicBlockParam_.singleM % basicBlockParam_.basicBlock.baseM == 0 &&
            basicBlockParam_.singleN % basicBlockParam_.basicBlock.baseN == 0) {
            // 1）特解：存在一组基本块同时整除singleM、singleN，则直接添加到最终解集，此时无尾块
            basicBlockParam_.basicBlock.mte2TailBWRatio = MTE2_TAIL_BW_RATIO_MAX + 1;
            cubeBoundResults_.push_back(basicBlockParam_);
        } else if (basicBlockParam_.singleM % basicBlockParam_.basicBlock.baseM == 0 &&
                   basicBlockParam_.singleN >= basicBlockParam_.basicBlock.baseN) {
            // 2）存在一组基本块，使得singleM整除baseM，则此时M轴尾块被消除，N轴存在尾块，
            //   将该组解添加到可选解集中，后续通过选取尾块bwRato最大值添加到最终解集。
            // 备注：采用BWRatio作为比较指标的依据，即使尾块较大，若能够达到或者接近cube bound状态，其效率也满足要求；
            // 备注：BWRatio越接近1，则表示mte2流水与cube流水越接近平衡，越接近cube bound；
            basicBlockParam_.basicBlock.mte2TailBWRatio = GetMte2BWRatio(
                basicBlockParam_.basicBlock.baseM, basicBlockParam_.singleN % basicBlockParam_.basicBlock.baseN,
                basicBlockParam_.mDim, basicBlockParam_.nDim);
            cubeBoundResults_.push_back(basicBlockParam_);
        } else if (basicBlockParam_.singleN % basicBlockParam_.basicBlock.baseN == 0 &&
                   basicBlockParam_.singleM >= basicBlockParam_.basicBlock.baseM) {
            // 对N轴做2）处理
            basicBlockParam_.basicBlock.mte2TailBWRatio =
                GetMte2BWRatio(basicBlockParam_.singleM % basicBlockParam_.basicBlock.baseM,
                               basicBlockParam_.basicBlock.baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
            cubeBoundResults_.push_back(basicBlockParam_);
        } else if (halfSingleM % basicBlockParam_.basicBlock.baseM == 0 &&
                   basicBlockParam_.singleN >= basicBlockParam_.basicBlock.baseN) {
            // 3）若singleM、singleN均大于基本块上界且无整除解，则通过折半缩小singleM、singleN，直到其落在基本块可选范围内，
            //    此时单核内出现L型尾块。
            // 备注：基本块解集具有连续性，若singleM在基本块解集最小、最大值范围内，则必然存在一组解使得singleM整除baseM，N同理；
            // 备注：由于halfSingleM是singleM多次折半得到，则M轴尾块tailM与baseM较为接近，从而M轴尾块(tailM,
            // baseK)的BWRatio
            //      与主块(baseM， baseN)也较为接近，即主块达到cube bound时，M轴尾块也可达到或接近cube bound；
            // 备注：采用尾块BWRatio作为可选解排序指标，尽量时N轴尾块(baseM，tailN)也达到cube bound；
            basicBlockParam_.basicBlock.mte2TailBWRatio = GetMte2BWRatio(
                basicBlockParam_.basicBlock.baseM, basicBlockParam_.singleN % basicBlockParam_.basicBlock.baseN,
                basicBlockParam_.mDim, basicBlockParam_.nDim);
            cubeBoundResults_.push_back(basicBlockParam_);
        } else if (halfSingleN % basicBlockParam_.basicBlock.baseN == 0 &&
                   basicBlockParam_.singleM >= basicBlockParam_.basicBlock.baseM) {
            // 对N轴做3）处理
            basicBlockParam_.basicBlock.mte2TailBWRatio =
                GetMte2BWRatio(basicBlockParam_.singleM % basicBlockParam_.basicBlock.baseM,
                               basicBlockParam_.basicBlock.baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
            cubeBoundResults_.push_back(basicBlockParam_);
        }
    }
}

int64_t Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetL1LoadSize(const BasicBlock &basicBlock,
                                                                   const L1TilingParam &l1Param) const
{
    int64_t b1BufferNum = l1Param.B1BufferNum;
    // A16W4-ND per-group以及A16W8-ND per-group场景，weightOut及weightL1 buffer数量最大为2
    if (basicBlockParam_.groupSize > 0 && !basicBlockParam_.weightNzFlag && b1BufferNum > 1) {
        b1BufferNum = BUFF_NUM_2;
    }
    int64_t biasBufferNum = std::min(BUFF_NUM_2, CeilDiv(basicBlockParam_.singleN, basicBlock.baseN));
    // 仅在A16W4 NZ场景，采用L1 4-buffer
    if (basicBlockParam_.bDtypeBits == BITS_4 && basicBlockParam_.weightNzFlag && l1Param.A1BufferNum == 1 &&
        l1Param.B1BufferNum == BUFF_NUM_4) {
        return std::max(
                   platformParam_.l1Size / BUFF_NUM_2,
                   static_cast<int64_t>(
                       basicBlock.baseN * l1Param.stepN * basicBlock.baseK * l1Param.stepKb * BUFF_NUM_2 * aByteSize_ +
                       basicBlock.baseM * l1Param.stepM * basicBlock.baseK * l1Param.stepKa * aByteSize_)) +
               static_cast<int64_t>(basicBlock.baseN * l1Param.stepN * basicBlock.baseK * l1Param.stepKb * BUFF_NUM_2 *
                                        aByteSize_ +
                                    basicBlock.baseN * biasBufferNum * biasByteSize_);
    }
    return static_cast<int64_t>(
        basicBlock.baseM * l1Param.stepM * basicBlock.baseK * l1Param.stepKa * l1Param.A1BufferNum * aByteSize_ +
        basicBlock.baseN * l1Param.stepN * basicBlock.baseK * l1Param.stepKb * b1BufferNum * aByteSize_ +
        basicBlock.baseN * biasBufferNum * biasByteSize_);
}

void Mc2WeightQuantBatchMatmulV2BasicBlockTiling::PrintFinalResult(const BasicBlockParam &param, bool enable) const
{
    if (enable) {
        OP_LOGD(opName_,
                "Tiling result: mSize: %ld, nSize: %ld, kSize: %ld, groupSize: %ld, singleM: %ld, singleN: %ld, "
                "singleK: %ld, mDim: %ld, nDim: %ld, kDim: %ld, mte2DataSize: %ld, fixpDataSize: %ld, iterateOrder: "
                "%ld, stepM: %ld, stepN: %ld, stepKa: %ld, stepKb: %ld, A1BufferNum: %ld, B1BufferNum: %ld, baseM: "
                "%ld, baseN: %ld, baseK: %ld, mte2BW: %lf, mte2MinBW: %lf, mte2BWRatio: %lf, mte2TailBWRatio: %lf",
                param.mSize, param.nSize, param.kSize, basicBlockParam_.groupSize, param.singleM, param.singleN,
                param.singleK, param.mDim, param.nDim, param.kDim, param.mte2DataSize, param.fixpDataSize,
                param.l1Param.iterateOrder, param.l1Param.stepM, param.l1Param.stepN, param.l1Param.stepKa,
                param.l1Param.stepKb, param.l1Param.A1BufferNum, param.l1Param.B1BufferNum, param.basicBlock.baseM,
                param.basicBlock.baseN, param.basicBlock.baseK, param.basicBlock.mte2BW, param.basicBlock.mte2MinBW,
                param.basicBlock.mte2BWRatio, param.basicBlock.mte2TailBWRatio);
    }
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::ValidateTilingResult() const
{
    OP_TILING_CHECK(basicBlockParam_.mDim * basicBlockParam_.nDim * basicBlockParam_.kDim > platformParam_.blockNum,
                    OP_LOGE(opName_, "Invalid block dim, mDim: %ld, nDim: %ld, kDim: %ld, maxDimNum: %ld",
                            basicBlockParam_.mDim, basicBlockParam_.nDim, basicBlockParam_.kDim,
                            platformParam_.blockNum),
                    return false);

    OP_TILING_CHECK(GetL1LoadSize(basicBlockParam_.basicBlock, basicBlockParam_.l1Param) > platformParam_.l1Size,
                    OP_LOGE(opName_, "The load size exceeds L1 buffer limit, load size: %ld, L1 buffer size: %ld",
                            GetL1LoadSize(basicBlockParam_.basicBlock, basicBlockParam_.l1Param),
                            platformParam_.l1Size),
                    return false);

    int64_t a2Size = static_cast<int64_t>(basicBlockParam_.basicBlock.baseM * basicBlockParam_.basicBlock.baseK *
                                          aByteSize_ * BUFF_NUM_2);
    int64_t b2Size = static_cast<int64_t>(basicBlockParam_.basicBlock.baseN * basicBlockParam_.basicBlock.baseK *
                                          aByteSize_ * BUFF_NUM_2);

    OP_TILING_CHECK(
        a2Size > platformParam_.l0aSize || b2Size > platformParam_.l0bSize || a2Size == 0 || b2Size == 0,
        OP_LOGE(opName_,
                "The load size may exceed L0 buffer limit, L0A load size: %ld, L0B load size: %ld, L0 buffer size: %ld",
                a2Size, b2Size, platformParam_.l0aSize),
        return false);

    int64_t stepKMax = CeilDiv(basicBlockParam_.singleK, basicBlockParam_.basicBlock.baseK);

    OP_TILING_CHECK((basicBlockParam_.l1Param.stepKa < stepKMax && basicBlockParam_.l1Param.stepKb < stepKMax) &&
                        (basicBlockParam_.l1Param.stepKa % basicBlockParam_.l1Param.stepKb > 0 &&
                         basicBlockParam_.l1Param.stepKb % basicBlockParam_.l1Param.stepKa > 0),
                    OP_LOGE(opName_, "Invalid stepK, stepKa (%ld) should be divisible by stepKb (%ld) or otherwise",
                            basicBlockParam_.l1Param.stepKa, basicBlockParam_.l1Param.stepKb),
                    return false);

    return true;
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::DoL1TilingForCubeBoundResult()
{
    // 遍历排序好的cube bound解集，找到第一个成功求解L1 tiling的解后返回
    for (const BasicBlockParam &cubeBoundRes : cubeBoundResults_) {
        basicBlockParam_ = cubeBoundRes;
        DoL1Tiling(true);
        if (!mte2BoundResults_.empty()) {
            stable_sort(mte2BoundResults_.begin(), mte2BoundResults_.end(), CompareMTE2BoundResult);
            basicBlockParam_ = mte2BoundResults_.front();
            return true;
        }
    }
    return false;
}

bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetFinalResult()
{
    bool ret = true;
    if (cubeBoundResults_.empty() && mte2BoundResults_.empty()) {
        OP_LOGI(opName_, "No solution Found. mSize: %ld, nSize: %ld, kSize: %ld", basicBlockParam_.mSize,
                basicBlockParam_.nSize, basicBlockParam_.kSize);
        ret = false;
        // 分满核且cube bound选解
    } else if (!cubeBoundResults_.empty() && mte2BoundResults_.empty()) {
        OP_LOGI(opName_, "Only find cube-bound solution");
        stable_sort(cubeBoundResults_.begin(), cubeBoundResults_.end(), CompareCubeBoundResult);
        ret = DoL1TilingForCubeBoundResult();
    } else if (cubeBoundResults_.empty() && !mte2BoundResults_.empty()) {
        OP_LOGI(opName_, "Only find mte2-bound solution");
        stable_sort(mte2BoundResults_.begin(), mte2BoundResults_.end(), CompareMTE2BoundResult);
        basicBlockParam_ = mte2BoundResults_.front();
    }
    PrintFinalResult(basicBlockParam_, ret);
    return ret && ValidateTilingResult();
}

/*
 *  该函数用于在绑满核条件下无解时，寻找非满核CUBE BOUND解，或耗时最小的MTE2 bound解。
 */
bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetDefaultBasicBlockTiling()
{
    Reset();
    OP_LOGD(opName_, "Enter GetDefaultBasicBlockTiling");

    int64_t mDimMax = std::min(CeilDiv(basicBlockParam_.mSize, BLOCK_CUBE), platformParam_.blockNum);
    for (int64_t mDim = mDimMax; mDim >= 1; mDim--) {
        int64_t nDimMax = std::min(CeilDiv(basicBlockParam_.nSize, BLOCK_CUBE), platformParam_.blockNum / mDim);
        for (int64_t nDim = nDimMax; nDim >= 1; nDim--) {
            if ((!cubeBoundResults_.empty() || !mte2BoundResults_.empty()) &&
                // 0.8含义：当已有可选解时剪枝分核情况小于0.8倍总核数的解
                mDim * nDim < 0.8 * platformParam_.blockNum) {
                continue;
            }
            basicBlockParam_.mDim = mDim;
            basicBlockParam_.nDim = nDim;
            basicBlockParam_.singleM = CeilAlign(CeilDiv(basicBlockParam_.mSize, basicBlockParam_.mDim), BLOCK_CUBE);
            basicBlockParam_.singleN = CeilAlign(CeilDiv(basicBlockParam_.nSize, basicBlockParam_.nDim), BLOCK_CUBE);
            int64_t mDimFixed = CeilDiv(basicBlockParam_.mSize, basicBlockParam_.singleM);
            int64_t nDimFixed = CeilDiv(basicBlockParam_.nSize, basicBlockParam_.singleN);
            if (mDimFixed != basicBlockParam_.mDim || nDimFixed != basicBlockParam_.nDim) {
                continue;
            }
            GetBasicBlockTable();
        }
    }

    return GetFinalResult();
}

/*
 *  该函数根据给定shape计算能够满足cube bound且绑满核的最佳基本块解。后续还需考虑：
 *  1）核间偏移非128B对齐引入的MTE2拆包、fixpipe写出效率下降；
 *  2）避免fixpipe bound需要考虑kL1；
 */
bool Mc2WeightQuantBatchMatmulV2BasicBlockTiling::GetBasicBlockTiling()
{
    OP_TILING_CHECK(!ValidateInputParam(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "input param",
                                                          (std::to_string(basicBlockParam_.mSize) + ", " +
                                                           std::to_string(basicBlockParam_.nSize) + ", " +
                                                           std::to_string(basicBlockParam_.kSize))
                                                              .c_str(),
                                                          "The value of input param must be valid."),
                    return false);

    Reset();
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2BasicBlockTable> basicBlockTablePtr =
        std::make_unique<Mc2WeightQuantBatchMatmulV2BasicBlockTable>();
    auto basicBlockTable =
        basicBlockTablePtr->GetBasicBlockTable(basicBlockParam_.aDtypeBits, basicBlockParam_.bDtypeBits);

    // 1）遍历分核方式
    for (auto it = basicBlockTable.begin(); it != basicBlockTable.end(); ++it) {
        std::tie(basicBlockParam_.mDim, basicBlockParam_.nDim) = it->first;
        basicBlockParam_.singleM = CeilAlign(CeilDiv(basicBlockParam_.mSize, basicBlockParam_.mDim), BLOCK_CUBE);
        basicBlockParam_.singleN = CeilAlign(CeilDiv(basicBlockParam_.nSize, basicBlockParam_.nDim), BLOCK_CUBE);
        int64_t mDimFixed = CeilDiv(basicBlockParam_.mSize, basicBlockParam_.singleM);
        int64_t nDimFixed = CeilDiv(basicBlockParam_.nSize, basicBlockParam_.singleN);
        if (basicBlockParam_.mDim * basicBlockParam_.nDim != platformParam_.blockNum ||
            mDimFixed != basicBlockParam_.mDim || nDimFixed != basicBlockParam_.nDim) {
            continue;
        }
        // 2）选取合适基本块对singleM、singleN进行切分
        SingleShapeTiling(it->second);
    }

    // 3）挑选最终解，若无解则进一步获取非满核cube bound解或mte2 bound解
    if (!GetFinalResult()) {
        return GetDefaultBasicBlockTiling();
    }

    return true;
}

} // namespace optiling
