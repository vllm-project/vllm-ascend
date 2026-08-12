/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_fp_matmul.h
 * \brief
 */

#ifndef MC2_FP_MATMUL_H
#define MC2_FP_MATMUL_H
#include "matmul_base.h"

namespace MC2KernelTemplate {
// 非量化场景没有额外的数据
struct FpMMAdditionalData {};

// 非量化场景的相关逻辑实现
template <typename MMTilingType, typename MMType>
class MC2FpMatmul {
public:
    __aicore__ inline MC2FpMatmul(AscendC::TPipe *tPipe) : tPipePtr_(tPipe){};
    // 初始化方法
    __aicore__ inline void Init();
    // 获取数据上下文引用
    __aicore__ inline MC2MMContext<FpMMAdditionalData, MMTilingType> *GetContextPtr();
    // 执行一次计算的方法
    __aicore__ inline void Process(uint32_t taskIndex);
    // 结束方法
    __aicore__ inline void End();

protected:
    MC2MMContext<FpMMAdditionalData, MMTilingType> MMcontext_;
    MMType MMImpl_;
    AscendC::TPipe *tPipePtr_;
};

template <typename MMTilingType, typename MMType>
inline __aicore__ void MC2FpMatmul<MMTilingType, MMType>::Init()
{
}

template <typename MMTilingType, typename MMType>
inline __aicore__ MC2MMContext<FpMMAdditionalData, MMTilingType> *MC2FpMatmul<MMTilingType, MMType>::GetContextPtr()
{
    return &MMcontext_;
}

template <typename MMTilingType, typename MMType>
inline __aicore__ void MC2FpMatmul<MMTilingType, MMType>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV {
        return;
    }
    GM_ADDR aGM = MMcontext_.baseData.aGM + taskIndex * MMcontext_.baseData.aOffset;
    GM_ADDR bGM = MMcontext_.baseData.bGM + taskIndex * MMcontext_.baseData.bOffset;
    GM_ADDR cGM = MMcontext_.baseData.cGM + taskIndex * MMcontext_.baseData.cOffset;
    tPipePtr_->Reset();
    MMImpl_.Init(aGM, bGM, cGM, MMcontext_.baseData.biasGM, nullptr, nullptr, MMcontext_.tilingDataPtr, tPipePtr_);
    MMImpl_.Process();
}

template <typename MMTilingType, typename MMType>
inline __aicore__ void MC2FpMatmul<MMTilingType, MMType>::End()
{
    if ASCEND_IS_AIV {
        return;
    }
    MMImpl_.End();
}

// 计算节点的上下文数据类型声明
#ifndef DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP
#define DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ContextType)                                               \
    using ContextType = MC2KernelTemplate::MC2MMContext<MC2KernelTemplate::FpMMAdditionalData, Mc2MatMulV3TilingData>
#endif

// 使用matmulv3算子作为计算节点的计算实现，是否转置的参数通过算子的模板参数获取
#ifndef DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP
#define DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(x1Transpose, x2Transpose, ComputationType)                         \
    using ComputationType =                                                                                            \
        MC2KernelTemplate::MC2FpMatmul<Mc2MatMulV3TilingData,                                                          \
                                       Mc2MatmulV3Advanced::Mc2MatmulAswKernel<                                        \
                                           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, x1Transpose>,  \
                                           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, x2Transpose>,  \
                                           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>,                \
                                           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DtypeBias>,              \
                                           Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD>>
#endif

#ifndef DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_A3_FP
#define DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_A3_FP(ContextType)                                            \
    using ContextType = MC2KernelTemplate::MC2MMContext<MC2KernelTemplate::FpMMAdditionalData, Mc2MatmulV3TilingData>
#endif

#ifndef DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_A3_FP
#define DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_A3_FP(x1Transpose, x2Transpose, ComputationType)                      \
    using ComputationType = MC2KernelTemplate::MC2FpMatmul<                                                            \
        Mc2MatmulV3TilingData,                                                                                         \
        Mc2MatmulBaseKernel<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, x1Transpose>,                 \
                            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, x2Transpose>,                 \
                            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>,                               \
                            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DtypeBias>, Mc2MatmulBaseBlock,         \
                            MM_CFG_NO_PRELOAD>>
#endif
}; // namespace MC2KernelTemplate
#endif
