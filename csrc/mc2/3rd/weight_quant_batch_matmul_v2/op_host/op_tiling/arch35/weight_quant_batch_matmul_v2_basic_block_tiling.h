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
 * \file weight_quant_batch_matmul_v2_basic_block_tiling.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_BASIC_BLOCK_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_BASIC_BLOCK_TILING_H

#include <algorithm>
#include <cmath>
#include <limits>

#include "weight_quant_batch_matmul_v2_basic_block_table.h"
#include "../weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

constexpr double BYTE_BITS = 8;
constexpr int64_t BITS_8 = 8;
constexpr int64_t BUFF_NUM_2 = 2;
constexpr int64_t BUFF_NUM_3 = 3;
constexpr int64_t BUFF_NUM_4 = 4;
constexpr int64_t BLOCK_CUBE = 16;
constexpr int64_t UB_ALIGN_SIZE = 32;
constexpr int64_t BASE_MN_LIMIT = 256 * 256;
constexpr int64_t BASE_MK_LIMIT = 256 * 64;
constexpr int64_t BASE_BLOCK_MIN = 96;
constexpr int64_t BASE_BLOCK_MAX = 512;
constexpr int64_t A16W4_MAX_BL1_SIZE_NZ = 96 * 1024;
constexpr int64_t A16W4_MAX_BUB_ELEM_SIZE_NZ = 24 * 1024; // A16W4 NZ 4-buffer场景，单buffer最大12KB
constexpr int64_t A16W4_MAX_BUB_ELEM_SIZE_ND = 32 * 1024; // A16W4 ND 4-buffer场景，单buffer最大16KB
constexpr int64_t CACHELINE_SIZE = 256;
constexpr int64_t MIN_CORE_DIM = 28;
constexpr int64_t MIN_A1_MTE2_LOAD_SIZE = 32 * 1024;
constexpr int64_t GROUP_SIZE_64 = 64;
constexpr double MTE2_TAIL_BW_RATIO_MAX = 10000;
const std::vector<int64_t> BASE_K_LIST = {256, 128, 64, 32, 16};

struct PlatformParam {
    int64_t blockNum;
    int64_t aicNum;
    int64_t ubSize;
    int64_t l1Size;
    int64_t l0aSize;
    int64_t l0bSize;
    int64_t l0cSize;
    int64_t cacheLine;
    int64_t minCacheLine;

    double frequency;
    double hbmBW;
    double l2BW;
};

struct WeightQuantBmmAttr {
    bool transA;
    bool transB;
    bool hasBias;
    bool weightNzFlag;
    bool hasOffset;
};

struct L1TilingParam {
    int64_t iterateOrder; // 0: orderM, for m for n; 1: orderN for n for m
    int64_t stepM;
    int64_t stepN;
    int64_t stepKa;
    int64_t stepKb;
    int64_t A1BufferNum;
    int64_t B1BufferNum;
};

struct BasicBlockParam {
    int64_t mSize;
    int64_t nSize;
    int64_t kSize;
    int64_t singleM;
    int64_t singleN;
    int64_t singleK;
    int64_t mDim;
    int64_t nDim;
    int64_t kDim;
    int64_t mte2DataSize;
    int64_t fixpDataSize;
    int64_t groupSize;
    int64_t aDtypeBits;
    int64_t bDtypeBits;
    int64_t biasDtypeBits;
    bool transA;
    bool transB;
    bool hasBias;
    bool weightNzFlag;

    L1TilingParam l1Param;
    BasicBlock basicBlock;
};

class Mc2WeightQuantBatchMatmulV2BasicBlockTiling {
public:
    Mc2WeightQuantBatchMatmulV2BasicBlockTiling()
    {
        Init();
    }
    ~Mc2WeightQuantBatchMatmulV2BasicBlockTiling() = default;

    void Init();
    void Reset();
    void SetPlatformParam(const PlatformParam &param);
    void SetShape(int64_t mSize, int64_t nSize, int64_t kSize, int64_t groupSize);
    void SetAttr(const char *opName, const WeightQuantBmmAttr &attr);
    void SetDtypeBits(int64_t aDtypeBits, int64_t bDtypeBits, int64_t biasDtypeBits);
    void SetQuantType(Mc2QuantType antiquantType);
    double GetMinMte2BW(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim) const;
    double GetMte2BW(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim) const;
    double GetMte2BWRatio(int64_t baseM, int64_t baseN, int64_t mDim, int64_t nDim) const;
    bool GetBasicBlockTiling();
    const BasicBlockParam &GetTilingResult() const
    {
        return basicBlockParam_;
    }

protected:
    void InitBasicBlockParam();
    void InitL1TilingParam();
    void InitPlatformParam();
    void GetBasicBlockTable();
    void SingleShapeTiling(const std::vector<BasicBlock> &basicBlockTable);
    bool ValidateInputParam() const;
    bool GetCachelineAlignFlag(int64_t dtypeBits, int64_t cacheline) const;
    int64_t GetBaseK(int64_t baseM, int64_t baseN) const;
    void UpdateMte2DataSize();
    int64_t GetUbLoadSize() const;
    bool GetInvalidFlagForBasicBlock() const;
    bool GetInvalidFlagA16W4() const;
    bool GetInvalidFlagA16W8() const;
    bool GetInvalidFlag(bool isCubeBoundSolution, int64_t stepKMax) const;
    void GetL1Param(bool isCubeBoundSolution, int64_t stepKMax, int64_t stepKaTmp, int64_t stepKbTmp);
    void DoL1Tiling(bool isCubeBoundSolution);
    bool GetHalfSingleShape(const std::vector<BasicBlock> &basicBlockTable, int64_t &halfSingleM, int64_t &halfSingleN);
    bool GetFinalResult();
    bool DoL1TilingForCubeBoundResult();
    bool ValidateTilingResult() const;
    void PrintFinalResult(const BasicBlockParam &param, bool enable) const;
    int64_t GetL1LoadSize(const BasicBlock &basicBlock, const L1TilingParam &l1Param) const;
    bool GetDefaultBasicBlockTiling();

    /*
     *   cube bound解排序函数，优先级：
     *    1）分核数
     *    2）无尾块 > 有尾块
     *    3）无尾块解中，比主块带宽比mte2BWRatio
     *    4）有尾块解中，比尾块带宽比mte2TailBWRatio
     */
    static bool CompareCubeBoundResult(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        if (param1.mDim * param1.nDim != param2.mDim * param2.nDim) {
            return param1.mDim * param1.nDim > param2.mDim * param2.nDim;
        }

        if (param1.basicBlock.mte2TailBWRatio > MTE2_TAIL_BW_RATIO_MAX &&
            param2.basicBlock.mte2TailBWRatio < MTE2_TAIL_BW_RATIO_MAX) {
            return true;
        }

        if (param1.basicBlock.mte2TailBWRatio < MTE2_TAIL_BW_RATIO_MAX &&
            param2.basicBlock.mte2TailBWRatio > MTE2_TAIL_BW_RATIO_MAX) {
            return false;
        }

        if (param1.basicBlock.mte2TailBWRatio > MTE2_TAIL_BW_RATIO_MAX &&
            param2.basicBlock.mte2TailBWRatio > MTE2_TAIL_BW_RATIO_MAX) {
            return param1.basicBlock.mte2BWRatio > param2.basicBlock.mte2BWRatio;
        }

        return param1.basicBlock.mte2TailBWRatio > param2.basicBlock.mte2TailBWRatio;
    }

    static int64_t GetAInnerSize(const BasicBlockParam &param)
    {
        return param.transA ? std::min(param.l1Param.stepM * param.basicBlock.baseM, param.singleM) :
                              std::min(param.l1Param.stepKa * param.basicBlock.baseK, param.singleK);
    }

    static int64_t GetBInnerSize(const BasicBlockParam &param)
    {
        return param.transB ? std::min(param.l1Param.stepKb * param.basicBlock.baseK, param.singleK) :
                              std::min(param.l1Param.stepN * param.basicBlock.baseN, param.singleN);
    }

    static int64_t GetBFullInnerSize(const BasicBlockParam &param)
    {
        return param.transB ? param.kSize : param.nSize;
    }

    static bool PreferFullloadKInPergroupNKLessCacheline(const BasicBlockParam &param1, const BasicBlockParam &param2,
                                                         bool &result)
    {
        // Early return not weightND pergroup nk
        if (param1.weightNzFlag || param1.groupSize <= 0 || !param1.transB) {
            return false;
        }

        // Calculate groupNum
        const int64_t groupNum = CeilDiv(param1.kSize, param1.groupSize);
        if (groupNum * param1.aDtypeBits >= (CACHELINE_SIZE * BITS_8)) {
            return false;
        }

        // Calculate inner size scales
        const int64_t innerSizeScale1 = CeilDiv(GetBInnerSize(param1), param1.groupSize);
        const int64_t innerSizeScale2 = CeilDiv(GetBInnerSize(param2), param2.groupSize);
        // Determine result based on inner size scales
        if (innerSizeScale1 == groupNum && innerSizeScale2 < groupNum) {
            result = true;
            return true;
        }
        if (innerSizeScale1 < groupNum && innerSizeScale2 == groupNum) {
            result = false;
            return true;
        }

        return false;
    }

    static bool CompareMTE2BoundResultA16W4PriorBND(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        int64_t innerSizeA1 = GetAInnerSize(param1);
        int64_t innerSizeA2 = GetAInnerSize(param2);
        int64_t innerSizeB1 = GetBInnerSize(param1);
        int64_t innerSizeB2 = GetBInnerSize(param2);
        // 1）优先保证B矩阵内轴不全载场景下cacheline对齐
        if (innerSizeB1 * param1.bDtypeBits % (CACHELINE_SIZE * BITS_8) == 0 &&
            (innerSizeB2 * param2.bDtypeBits % (CACHELINE_SIZE * BITS_8) > 0 &&
             innerSizeB2 != GetBFullInnerSize(param2))) {
            return true;
        }
        if ((innerSizeB1 * param1.bDtypeBits % (CACHELINE_SIZE * BITS_8) > 0 &&
             innerSizeB1 != GetBFullInnerSize(param1)) &&
            innerSizeB2 * param2.bDtypeBits % (CACHELINE_SIZE * BITS_8) == 0) {
            return false;
        }

        // 2）weightND nk pergroup场景下scale/offset内轴小于cacheline场景时带宽较差，优先选择全载k的解
        bool result = false;
        if (PreferFullloadKInPergroupNKLessCacheline(param1, param2, result)) {
            return result;
        }

        int64_t nBl1TailSize1 = param1.singleN % std::max(param1.l1Param.stepN * param1.basicBlock.baseN, BLOCK_CUBE);
        nBl1TailSize1 = nBl1TailSize1 == 0 ? param1.l1Param.stepN * param1.basicBlock.baseN : nBl1TailSize1;
        int64_t nBl1TailSize2 = param2.singleN % std::max(param2.l1Param.stepN * param2.basicBlock.baseN, BLOCK_CUBE);
        nBl1TailSize2 = nBl1TailSize2 == 0 ? param2.l1Param.stepN * param2.basicBlock.baseN : nBl1TailSize2;
        int64_t bL1LoadSize1 = nBl1TailSize1 * param1.l1Param.stepKb * param1.basicBlock.baseK;
        int64_t bL1LoadSize2 = nBl1TailSize2 * param2.l1Param.stepKb * param2.basicBlock.baseK;
        // 3）优先选BL1的载入量较大的解，即优先保证B矩阵的MTE2效率
        if (bL1LoadSize1 != bL1LoadSize2) {
            return bL1LoadSize1 > bL1LoadSize2;
        }
        // 4）保证A矩阵cacheline对齐
        if (innerSizeA1 * param1.aDtypeBits % (CACHELINE_SIZE * BITS_8) == 0 &&
            innerSizeA2 * param2.aDtypeBits % (CACHELINE_SIZE * BITS_8) > 0) {
            return true;
        }
        if (innerSizeA1 * param1.aDtypeBits % (CACHELINE_SIZE * BITS_8) > 0 &&
            innerSizeA2 * param2.aDtypeBits % (CACHELINE_SIZE * BITS_8) == 0) {
            return false;
        }

        // 5）保证基本块选取合理，保证mad效率；经测试，mte2BWRatio越大的解，L0切分越好（越接近cube
        // bound基本块）。 另外，测试发现L0切分较好的情况下，mte2重复载入量也较小。
        if (std::fabs(param1.basicBlock.mte2BWRatio - param2.basicBlock.mte2BWRatio) >
            std::numeric_limits<double>::epsilon()) {
            return param1.basicBlock.mte2BWRatio > param2.basicBlock.mte2BWRatio;
        }
        // 6）ND及NZ场景，此处希望单次AL1载入量越大越好
        int64_t al1LoadSize1 =
            param1.l1Param.stepM * param1.basicBlock.baseM * param1.l1Param.stepKa * param1.basicBlock.baseK;
        int64_t al1LoadSize2 =
            param2.l1Param.stepM * param2.basicBlock.baseM * param2.l1Param.stepKa * param2.basicBlock.baseK;
        return al1LoadSize1 > al1LoadSize2;
    }

    static bool CompareMTE2BoundResultA16W4PriorBNZ(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        int64_t nBl1TailSize1 = param1.singleN % std::max(param1.l1Param.stepN * param1.basicBlock.baseN, BLOCK_CUBE);
        nBl1TailSize1 = nBl1TailSize1 == 0 ? param1.l1Param.stepN * param1.basicBlock.baseN : nBl1TailSize1;
        int64_t nBl1TailSize2 = param2.singleN % std::max(param2.l1Param.stepN * param2.basicBlock.baseN, BLOCK_CUBE);
        nBl1TailSize2 = nBl1TailSize2 == 0 ? param2.l1Param.stepN * param2.basicBlock.baseN : nBl1TailSize2;
        int64_t bubLoadSize1 =
            CeilAlign(CeilDiv(nBl1TailSize1, BUFF_NUM_2), BLOCK_CUBE) * param1.l1Param.stepKb * param1.basicBlock.baseK;
        int64_t bubLoadSize2 =
            CeilAlign(CeilDiv(nBl1TailSize2, BUFF_NUM_2), BLOCK_CUBE) * param2.l1Param.stepKb * param2.basicBlock.baseK;
        // 1）优先选Bub的载入量较大的解，即优先保证B矩阵的MTE2效率
        // 此处采用bubLoadSize而非bL1LoadSize，是由于nBL1切分nBub时可能出现非因子切分，因此以nBub计算结果为准
        if (bubLoadSize1 != bubLoadSize2) {
            return bubLoadSize1 > bubLoadSize2;
        }
        // 2）保证基本块选取合理，保证mad效率；经测试，mte2BWRatio越大的解，L0切分越好（越接近cube
        // bound基本块）。 另外，测试发现L0切分较好的情况下，mte2重复载入量也较小。
        if (std::fabs(param1.basicBlock.mte2BWRatio - param2.basicBlock.mte2BWRatio) >
            std::numeric_limits<double>::epsilon()) {
            return param1.basicBlock.mte2BWRatio > param2.basicBlock.mte2BWRatio;
        }
        int64_t al1TailSize1 = param1.singleK % std::max(param1.l1Param.stepKa * param1.basicBlock.baseK, BLOCK_CUBE);
        al1TailSize1 = al1TailSize1 == 0 ? param1.l1Param.stepKa * param1.basicBlock.baseK : al1TailSize1;
        int64_t al1TailSize2 = param2.singleK % std::max(param2.l1Param.stepKa * param2.basicBlock.baseK, BLOCK_CUBE);
        al1TailSize2 = al1TailSize2 == 0 ? param2.l1Param.stepKa * param2.basicBlock.baseK : al1TailSize2;
        // 3）ND及NZ场景，此处希望单次AL1载入量越大越好，尾块越小越好
        if (al1TailSize1 != al1TailSize2) {
            return al1TailSize1 > al1TailSize2;
        } else {
            int64_t al1LoadSize1 =
                param1.l1Param.stepM * param1.basicBlock.baseM * param1.l1Param.stepKa * param1.basicBlock.baseK;
            int64_t al1LoadSize2 =
                param2.l1Param.stepM * param2.basicBlock.baseM * param2.l1Param.stepKa * param2.basicBlock.baseK;
            return al1LoadSize1 > al1LoadSize2;
        }
    }

    static bool CompareMTE2BoundResultA16W4PriorB(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        if (param1.weightNzFlag) {
            return CompareMTE2BoundResultA16W4PriorBNZ(param1, param2);
        } else {
            return CompareMTE2BoundResultA16W4PriorBND(param1, param2);
        }
    }

    static bool CompareMTE2BoundResultA16W4PriorA(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        int64_t innerSizeA1 = GetAInnerSize(param1);
        int64_t innerSizeA2 = GetAInnerSize(param2);
        int64_t innerSizeB1 = GetBInnerSize(param1);
        int64_t innerSizeB2 = GetBInnerSize(param2);
        // 1) 优先保证A矩阵cacheline对齐
        if (innerSizeA1 * param1.aDtypeBits % (CACHELINE_SIZE * BITS_8) == 0 &&
            innerSizeA2 * param2.aDtypeBits % (CACHELINE_SIZE * BITS_8) > 0) {
            return true;
        }
        if (innerSizeA1 * param1.aDtypeBits % (CACHELINE_SIZE * BITS_8) > 0 &&
            innerSizeA2 * param2.aDtypeBits % (CACHELINE_SIZE * BITS_8) == 0) {
            return false;
        }

        // 2）保证A矩阵cacheline对齐的基础上，优先保证B矩阵带宽不处于最差场景
        bool result = false;
        if (PreferFullloadKInPergroupNKLessCacheline(param1, param2, result)) {
            return result;
        }

        // 3）选AL1的载入量较大的解，即优先保证A矩阵的MTE2效率，同时AL1载入量无需过大，否则导致BL1载入量不足
        int64_t al1LoadSize1 = param1.l1Param.stepM * param1.basicBlock.baseM * param1.l1Param.stepKa *
                               param1.basicBlock.baseK * param1.aDtypeBits;
        int64_t al1LoadSize2 = param2.l1Param.stepM * param2.basicBlock.baseM * param2.l1Param.stepKa *
                               param2.basicBlock.baseK * param2.aDtypeBits;

        if (al1LoadSize1 >= MIN_A1_MTE2_LOAD_SIZE * BITS_8 && al1LoadSize2 < MIN_A1_MTE2_LOAD_SIZE * BITS_8) {
            return true;
        }
        if (al1LoadSize1 < MIN_A1_MTE2_LOAD_SIZE * BITS_8 && al1LoadSize2 >= MIN_A1_MTE2_LOAD_SIZE * BITS_8) {
            return false;
        }

        if (!param1.weightNzFlag) {
            // 4) ND场景，优先保证B矩阵cacheline对齐
            if (innerSizeB1 * param1.bDtypeBits % (CACHELINE_SIZE * BITS_8) == 0 &&
                innerSizeB2 * param2.bDtypeBits % (CACHELINE_SIZE * BITS_8) > 0) {
                return true;
            }
            if (innerSizeB1 * param1.bDtypeBits % (CACHELINE_SIZE * BITS_8) > 0 &&
                innerSizeB2 * param2.bDtypeBits % (CACHELINE_SIZE * BITS_8) == 0) {
                return false;
            }
        }

        // 5）保证基本块选取合理，保证mad效率；
        if (std::fabs(param1.basicBlock.mte2BWRatio - param2.basicBlock.mte2BWRatio) >
            std::numeric_limits<double>::epsilon()) {
            return param1.basicBlock.mte2BWRatio > param2.basicBlock.mte2BWRatio;
        }

        // 6) 最后微调BL1，优先选择BL1载入量较大的解
        int64_t bL1LoadSize1 =
            param1.l1Param.stepN * param1.basicBlock.baseN * param1.l1Param.stepKb * param1.basicBlock.baseK;
        int64_t bL1LoadSize2 =
            param2.l1Param.stepN * param2.basicBlock.baseN * param2.l1Param.stepKb * param2.basicBlock.baseK;
        return bL1LoadSize1 > bL1LoadSize2;
    }

    static bool CompareMTE2BoundResultA16W4(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        if (param1.mDim * param1.nDim >= MIN_CORE_DIM && param2.mDim * param2.nDim < MIN_CORE_DIM) {
            return true;
        }
        if (param1.mDim * param1.nDim < MIN_CORE_DIM && param2.mDim * param2.nDim >= MIN_CORE_DIM) {
            return false;
        }
        int64_t aSize = param1.mSize * param1.kSize * param1.aDtypeBits;
        int64_t bSize = param1.nSize * param1.kSize * param1.bDtypeBits;
        if (bSize > aSize) {
            return CompareMTE2BoundResultA16W4PriorB(param1, param2);
        } else {
            return CompareMTE2BoundResultA16W4PriorA(param1, param2);
        }
    }

    static bool CompareMTE2BoundResultA16W8(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        if (param1.mDim * param1.nDim != param2.mDim * param2.nDim) {
            return param1.mDim * param1.nDim > param2.mDim * param2.nDim;
        }

        if (std::fabs(param1.basicBlock.mte2BWRatio - param2.basicBlock.mte2BWRatio) >
            std::numeric_limits<double>::epsilon()) {
            return param1.basicBlock.mte2BWRatio > param2.basicBlock.mte2BWRatio;
        }

        int64_t aSize = param1.mSize * param1.kSize * param1.aDtypeBits;
        int64_t bSize = param1.nSize * param1.kSize * param1.bDtypeBits;
        int64_t bl1TailSize1 = param1.singleK % std::max(param1.l1Param.stepKb * param1.basicBlock.baseK, BLOCK_CUBE);
        bl1TailSize1 = bl1TailSize1 == 0 ? param1.l1Param.stepKb * param1.basicBlock.baseK : bl1TailSize1;
        int64_t bl1TailSize2 = param2.singleK % std::max(param2.l1Param.stepKb * param2.basicBlock.baseK, BLOCK_CUBE);
        bl1TailSize2 = bl1TailSize2 == 0 ? param2.l1Param.stepKb * param2.basicBlock.baseK : bl1TailSize2;
        int64_t al1TailSize1 = param1.singleK % std::max(param1.l1Param.stepKa * param1.basicBlock.baseK, BLOCK_CUBE);
        al1TailSize1 = al1TailSize1 == 0 ? param1.l1Param.stepKa * param1.basicBlock.baseK : al1TailSize1;
        int64_t al1TailSize2 = param2.singleK % std::max(param2.l1Param.stepKa * param2.basicBlock.baseK, BLOCK_CUBE);
        al1TailSize2 = al1TailSize2 == 0 ? param2.l1Param.stepKa * param2.basicBlock.baseK : al1TailSize2;
        if (bSize > aSize) {
            // B载入量优先
            if (bl1TailSize1 != bl1TailSize2) {
                return bl1TailSize1 > bl1TailSize2;
            }
            return al1TailSize1 > al1TailSize2;
        } else {
            // A载入量优先
            if (al1TailSize1 != al1TailSize2) {
                return al1TailSize1 > al1TailSize2;
            }
            return bl1TailSize1 > bl1TailSize2;
        }
    }

    /*
     *  mte2 bound解排序函数，优先级:
     *   1）若mte2Cost相近，则优先选择kL1无尾块的tiling
     *   2）优先选择mte2Cost较小的tiling
     */
    static bool CompareMTE2BoundResult(const BasicBlockParam &param1, const BasicBlockParam &param2)
    {
        if (param1.bDtypeBits == BITS_4) {
            return CompareMTE2BoundResultA16W4(param1, param2);
        } else {
            return CompareMTE2BoundResultA16W8(param1, param2);
        }
    }

    static int64_t CeilDiv(int64_t num1, int64_t num2)
    {
        if (num2 == 0) {
            return 0;
        }
        return (num1 + num2 - 1) / num2;
    }

    static int64_t CeilAlign(int64_t num1, int64_t num2)
    {
        if (num2 == 0) {
            return 0;
        }
        return (num1 + num2 - 1) / num2 * num2;
    }

    static int64_t FloorAlign(int64_t num1, int64_t num2)
    {
        if (num2 == 0) {
            return 0;
        }
        return num1 / num2 * num2;
    }

    const char *opName_;
    bool hasOffset_;
    double aByteSize_;
    double bByteSize_;
    double biasByteSize_;
    Mc2QuantType antiquantType_;

    PlatformParam platformParam_;
    std::vector<BasicBlockParam> mte2BoundResults_;
    std::vector<BasicBlockParam> cubeBoundResults_;
    BasicBlockParam basicBlockParam_;
};

} // namespace optiling

#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_BASIC_BLOCK_TILING_H
