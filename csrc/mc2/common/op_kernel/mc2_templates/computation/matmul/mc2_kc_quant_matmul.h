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
 * \file mc2_kc_quant_matmul.h
 * \brief
 */

#ifndef MC2_KC_QUANT_MATMUL_H
#define MC2_KC_QUANT_MATMUL_H
#include "matmul_base.h"

namespace MC2KernelTemplate {
struct KCQuantMMAdditionalData {
    GM_ADDR x1Scale;
    GM_ADDR x2Scale;
    GM_ADDR x2Offset;
    uint64_t x1ScaleOffset;
};

// 量化场景的相关逻辑实现
template <typename MMTilingType, typename MMType>
class MC2KCQuantMatmul {
public:
    __aicore__ inline MC2KCQuantMatmul(AscendC::TPipe *tPipe) : tPipePtr_(tPipe){};
    // 初始化方法
    __aicore__ inline void Init();
    // 获取数据上下文引用
    __aicore__ inline MC2MMContext<KCQuantMMAdditionalData, MMTilingType> *GetContextPtr();
    // 执行一次计算的方法
    __aicore__ inline void Process(uint32_t taskIndex);
    // 结束方法
    __aicore__ inline void End();

protected:
    MC2MMContext<KCQuantMMAdditionalData, MMTilingType> MMcontext_;
    MMType MMImpl_;
    AscendC::TPipe *tPipePtr_;
};

template <typename MMTilingType, typename MMType>
inline __aicore__ void MC2KCQuantMatmul<MMTilingType, MMType>::Init()
{
}

template <typename MMTilingType, typename MMType>
inline __aicore__ MC2MMContext<KCQuantMMAdditionalData, MMTilingType> *
MC2KCQuantMatmul<MMTilingType, MMType>::GetContextPtr()
{
    return &MMcontext_;
}

template <typename MMTilingType, typename MMType>
inline __aicore__ void MC2KCQuantMatmul<MMTilingType, MMType>::Process(uint32_t taskIndex)
{
    GM_ADDR aGM = MMcontext_.baseData.aGM + taskIndex * MMcontext_.baseData.aOffset;
    GM_ADDR bGM = MMcontext_.baseData.bGM + taskIndex * MMcontext_.baseData.bOffset;
    GM_ADDR cGM = MMcontext_.baseData.cGM + taskIndex * MMcontext_.baseData.cOffset;
    GM_ADDR x1Scale = MMcontext_.additionalData.x1Scale + taskIndex * MMcontext_.additionalData.x1ScaleOffset;
    tPipePtr_->Reset();
    MMImpl_.Init(aGM, bGM, MMcontext_.additionalData.x2Scale, MMcontext_.additionalData.x2Offset,
                 MMcontext_.baseData.biasGM, x1Scale, cGM, nullptr, MMcontext_.tilingDataPtr, tPipePtr_);
    MMImpl_.Process();
}

template <typename MMTilingType, typename MMType>
inline __aicore__ void MC2KCQuantMatmul<MMTilingType, MMType>::End()
{
}

// 计算节点的上下文数据类型声明
#ifndef DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_QUANT
#define DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_QUANT(ContextType)                                            \
    using ContextType = MC2KernelTemplate::MC2MMContext<MC2KernelTemplate::KCQuantMMAdditionalData,                    \
                                                        DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams>
#endif

// 使用QuantBatchMatmulV3算子作为计算节点的计算实现，是否转置的参数通过算子的模板参数获取
#ifndef DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT
#define DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(ComputationType, MMDtypeX1, MMDtypeX2)                          \
    using ComputationType = MC2KernelTemplate::MC2KCQuantMatmul<                                                       \
        DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams,                                                             \
        Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel<                                                       \
            MMDtypeX1, MMDtypeX2, float, float, float, DTYPE_Y, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, false, \
            X2TRANSPOSE, float, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock>>
#endif
}; // namespace MC2KernelTemplate
#endif
