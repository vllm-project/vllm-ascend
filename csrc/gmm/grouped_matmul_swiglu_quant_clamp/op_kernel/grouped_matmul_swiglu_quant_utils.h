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
 * \file grouped_matmul_swiglu_quant_utils.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_UTILS_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_UTILS_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
// A8W4 MSD场景
#if defined(ORIG_DTYPE_X) && defined(DT_INT8) && ORIG_DTYPE_X == DT_INT8 && defined(ORIG_DTYPE_WEIGHT) &&              \
    defined(DT_INT4) && ORIG_DTYPE_WEIGHT == DT_INT4
        #define GMM_SWIGLU_QUANT_A8W4_MSD
        using DTYPE_X_A8W4_MSD = AscendC::int4b_t;
// A8W8 场景
#elif defined(ORIG_DTYPE_X) && defined(DT_INT8) && ORIG_DTYPE_X == DT_INT8 && defined(ORIG_DTYPE_WEIGHT) &&            \
    defined(DT_INT8) && ORIG_DTYPE_WEIGHT == DT_INT8
        #define GMM_SWIGLU_QUANT_A8W8
#endif // 场景分类

#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
    constexpr CubeFormat wFormat = CubeFormat::NZ;
#elif defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_ND
    constexpr CubeFormat wFormat = CubeFormat::ND;
#endif // weight格式分类

#endif // 芯片型号分类

namespace GROUPED_MATMUL_SWIGLU_QUANT {
using namespace AscendC;
constexpr uint32_t INT8_BITS = 8;           // a int8 number has 8 bits
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32; // 32: a block has 32 bytes data
constexpr uint32_t THRESHOLD_BLOCK_NUM = 8;
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE = 64;                   // 64: a block has 64 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2; // 2: a float16 data has two bytes
constexpr uint32_t FLOAT_UB_BLOCK_UNIT_SIZE = 8;                     // 2: a float16 data has two bytes
constexpr uint32_t SINGLE_CORE_M = 128;
constexpr uint32_t SINGLE_CORE_N = 256;
constexpr uint32_t SINGLE_CORE_K = 7168;
constexpr uint32_t BASIC_M = 128;
constexpr uint32_t BASIC_N = 256;
constexpr uint32_t BASIC_K = 128;
constexpr uint32_t STEP_M = 1;
constexpr uint32_t STEP_N = 1;
constexpr uint32_t STEP_Ka = 4;
constexpr uint32_t STEP_Kb = 4;
constexpr uint32_t DEPTH_A1 = 8;
constexpr uint32_t DEPTH_B1 = 8;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_ELE = 64;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_BLOCK = 8;
constexpr uint32_t FP32_LEN_64_REPEAT = 4096;
constexpr uint32_t REPEAT_64 = 64;
constexpr uint32_t REPEAT_8 = 8;
constexpr uint32_t BISECT = 2;
constexpr uint32_t MOD_32_MASK = 0x1F;
constexpr uint32_t MOD_16_MASK = 0x0F;
constexpr uint32_t ALIGN_8_ELE = 8;
constexpr uint32_t ALIGN_16_ELE = 16;
constexpr float QUANT_SCALE_INT8 = 127.0f;
constexpr int64_t SWIGLU_REDUCE_FACTOR = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint8_t NUM_8 = 8;
constexpr bool NO_BIAS = false;
constexpr int64_t DOUBLE_ROW = 2;
constexpr MatmulConfig CUSTOM_CFG_MDL = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig GetMMStaticCFG()
{
    MatmulConfig MM_CFG = CUSTOM_CFG_MDL;
    MM_CFG.singleCoreM = SINGLE_CORE_M;
    MM_CFG.singleCoreN = SINGLE_CORE_N;
    MM_CFG.singleCoreK = SINGLE_CORE_K;
    MM_CFG.basicM = BASIC_M;
    MM_CFG.basicN = BASIC_N;
    MM_CFG.basicK = BASIC_K;
    return MM_CFG;
}

constexpr static MatmulApiStaticTiling GetMMTiling(const MatmulApiStaticTiling &mmTiling)
{
    MatmulApiStaticTiling tiling = mmTiling;
    tiling.stepM = STEP_M;
    tiling.stepN = STEP_N;
    tiling.stepKa = STEP_Ka;
    tiling.stepKb = STEP_Kb;
    tiling.depthA1 = DEPTH_A1;
    tiling.depthB1 = DEPTH_B1;
    tiling.isBias = NO_BIAS;
    return tiling;
}

template <class AT_, class BT_, class CT_>
struct MMImplTypeStatic {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    // bias未被使用但高阶模板参数需要传入
    using BiasT = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;
    static constexpr MatmulConfig cfg = GetMMStaticCFG();
    static constexpr MatmulApiStaticTiling mdl = GetMMTiling(GetMatmulApiTiling<AT, BT, CT, BiasT>(cfg));
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, mdl>;
};

template <class AT_, class BT_, class CT_>
struct MMImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    // bias未被使用但高阶模板参数需要传入
    using BiasT = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, CUSTOM_CFG_MDL>;
};

struct MNConfig {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t mIdx = 0;
    int64_t nIdx = 0;
    int64_t blockDimM = 0;
    int64_t blockDimN = 0;
    int64_t singleM = 0;
    int64_t singleN = 0;
    int64_t wBaseOffset = 0;
    int64_t nAxisBaseOffset = 0;
    int64_t mAxisBaseOffset = 0;
    int64_t xBaseOffset = 0;
    int64_t yBaseOffset = 0;
    int64_t wOutOffset = 0;
    int64_t workSpaceOffset = 0;
};

struct VecConfig {
    int64_t M = 0;
    int64_t usedCoreNum = 0;
    int64_t startOffset = 0;
    int64_t curOffset = 0;
    int64_t startIdx = 0;
    int64_t curIdx = 0;
    int64_t taskNum = 0;
    int64_t curGroupIdx = 0;
    int64_t outLoopNum = 0;
    int64_t innerLoopNum = 0;
    int64_t tailLoopNum = 0;
    int64_t nextUpadteInterVal = 0;
};

struct WorkSpaceSplitConfig {
    int64_t M = 0;
    int64_t loopCount = 0;
    int64_t leftMatrixStartIndex = 0;
    int64_t rightMatrixExpertStartIndex = 0;
    int64_t rightMatrixExpertNextStartIndex = 0;
    int64_t rightMatrixExpertEndIndex = 0;
    int64_t notLastTaskSize = 0;
    int64_t lastLoopTaskSize = 0;
    bool isLastLoop = false;
};

struct GMAddrParams {
    // 输入 GM Tensor
    GM_ADDR xGM;                     // 左矩阵
    GM_ADDR weightGM;                // 右矩阵
    GM_ADDR weightScaleGM;           // 权重scale
    GM_ADDR xScaleGM;                // 激活scale
    GM_ADDR weightAuxiliaryMatrixGM; // 权重辅助矩阵
    GM_ADDR groupListGM;             // 分组矩阵
    // 输出 GM Tensor
    GM_ADDR yGM;      // 输出量化矩阵
    GM_ADDR yScaleGM; // 输出scale矩阵
    // workspace GM Tensor
    GM_ADDR workSpaceGM; // 左矩阵前处理结果矩阵 (double workspace) + 中间处理结果矩阵 (double workspace)
    int64_t workSpaceOffset1;
    int64_t workSpaceOffset2;
    int64_t workSpaceOffset3;
};

template <uint32_t base, typename T = uint32_t>
__aicore__ inline auto AlignUp(T a) -> T
{
    if (unlikely(base == 0)) {
        return a;
    }
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline auto AlignUp(T a, T base) -> T
{
    if (unlikely(base == 0)) {
        return a;
    }
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline auto AlignDown(T a, T base) -> T
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

template <>
__aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a)
{
    // to be Multiple of 4, result should be in a format of b(xxxx,x100).
    // This means last two bits should be zero, requiring that
    // result = num & b(1111,1100) = num & (~3).
    // &(~3) operator may reduces num into the range [num, num - 3].
    // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
    // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
    // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
    // num = a + 3 also satisfies the goal condition.
    return (a + 3) & ~3; // & ~3: set last two bits of (a+3) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<8, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 7) & ~7; // & ~7: set last four bits of (a+7) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 15) & ~15; // & ~15: set last four bits of (a+15) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a)
{
    // refer to the above comments.
    return (a + 31) & ~31; // & ~31: set last five bits of (a+31) to be zero}
}

__aicore__ inline void ReduceMaxSmall(const LocalTensor<float> &dstLocal, const LocalTensor<float> &workLocal,
                                      const LocalTensor<float> &srcLocal, uint32_t count)
{
    /**
     * @brief ReduceMaxSmall 此函数仅支持入参count小于4096。
     */
    uint32_t repeat = count / VEC_LEN_ONCE_REPEAT_ELE;
    uint32_t tailNum = count % VEC_LEN_ONCE_REPEAT_ELE;
    if (likely(repeat > 0)) {
        WholeReduceMax(workLocal, srcLocal, VEC_LEN_ONCE_REPEAT_ELE, repeat, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK,
                       ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailNum != 0)) {
        WholeReduceMax(workLocal[repeat], srcLocal[count - tailNum], tailNum, 1, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK,
                       ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        repeat += 1;
    }
    WholeReduceMax(dstLocal, workLocal, repeat, 1, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK, ReduceOrder::ORDER_ONLY_VALUE);
}

__aicore__ inline void ReduceMaxTemplate(const LocalTensor<float> &dstLocal, const LocalTensor<float> &workLocal,
                                         const LocalTensor<float> &srcLocal, const LocalTensor<float> &resTmpLocal,
                                         uint32_t count)
{
    /**
     * @brief 当前算子仅支持[32, 10240]长度的词向量维度N，对应此函数count入参范围在[16, 5120]。
     * @param [in] count: 本函数支持count范围为[1,8192]。
     */
    if (count <= FP32_LEN_64_REPEAT) {
        ReduceMaxSmall(dstLocal, workLocal, srcLocal, count);
        PipeBarrier<PIPE_V>();
    } else {
        BlockReduceMax(workLocal, srcLocal, REPEAT_64, VEC_LEN_ONCE_REPEAT_ELE, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK);
        PipeBarrier<PIPE_V>();

        BlockReduceMax(workLocal, workLocal, REPEAT_8, VEC_LEN_ONCE_REPEAT_ELE, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK);
        PipeBarrier<PIPE_V>();

        WholeReduceMax(resTmpLocal, workLocal, VEC_LEN_ONCE_REPEAT_ELE, 1, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK,
                       ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();

        ReduceMaxSmall(dstLocal, workLocal, srcLocal[FP32_LEN_64_REPEAT], count - FP32_LEN_64_REPEAT);
        PipeBarrier<PIPE_V>();

        const BinaryRepeatParams repeatParams = {1, 1, 1, NUM_8, NUM_8, NUM_8};
        Max(dstLocal, dstLocal, resTmpLocal, 1, 1, repeatParams);
    }
}

__aicore__ inline void CastFp32ToInt8Template(LocalTensor<int8_t> &dstLocal, LocalTensor<float> &srcLocal,
                                              LocalTensor<int8_t> &oneBlockWorkspace, int32_t dstOffset,
                                              int32_t srcOffset, int32_t count)
{
    Cast(srcLocal[srcOffset].ReinterpretCast<half>(), srcLocal[srcOffset], RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    if ((dstOffset & MOD_32_MASK) == 0) {
        Cast(dstLocal[dstOffset], srcLocal[srcOffset].ReinterpretCast<half>(), RoundMode::CAST_RINT, count);
    } else if ((dstOffset & MOD_16_MASK) == 0) {
        Cast(dstLocal[dstOffset + ALIGN_16_ELE], srcLocal[srcOffset + ALIGN_8_ELE].ReinterpretCast<half>(),
             RoundMode::CAST_RINT, count - ALIGN_16_ELE);
        PipeBarrier<PIPE_V>();
        Cast(oneBlockWorkspace, srcLocal[srcOffset].ReinterpretCast<half>(), RoundMode::CAST_RINT, ALIGN_16_ELE);
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < ALIGN_16_ELE; i++) {
            int8_t temp = oneBlockWorkspace.GetValue(i);
            dstLocal.SetValue(dstOffset + i, temp);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace GROUPED_MATMUL_SWIGLU_QUANT

#endif // ASCENDC_GROUPED_MATMUL_UTILS_H
