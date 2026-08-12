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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_H

#include "weight_quant_batch_matmul_v2_asw_block.h"
#include "../weight_quant_batch_matmul_v2_common.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../tool.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                                    \
    template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,             \
              Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
#define LOCAL_TEMPLATE_FUNC_PARAMS                                                                                     \
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using matmul::MatmulImpl;
using matmul::MatmulType;
namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

constexpr MatmulConfig MM_CFG_VEC_ND2NZ = GetMDLConfig(false, false, 0, true);
constexpr MatmulConfig MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, false);
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG = GetMDLConfig(false, false, 0, false, false, false, true);
constexpr MatmulConfig MM_CFG_PRELOAD = GetMDLConfig(false, false, 1, false, false, false, false);

LOCAL_TEMPLATE_CLASS_PARAMS
class Mc2WeightQuantBatchMatmulV2ASWKernel {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2ASWKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void *tilingData, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeBasicOptiLoop();
    __aicore__ inline void SetMMParaAndCompute();

    uint32_t blockIdx_;
    const WeightQuantBatchMatmulV2ASWTilingData *tiling_;

    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    Mc2WeightQuantBmmAswBlock block_;
    GlobalTensor<uint64_t> scaleGlobal_;
    using aType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, xType, aTrans>;
    using bType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, wType, bTrans>;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, yType>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    matmul::MatmulImpl<aType, bType, cType, biasMatmulType, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG> mm_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void *tilingData, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    tiling_ = static_cast<const WeightQuantBatchMatmulV2ASWTilingData *>(tilingData);
    blockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&tiling_->matmulTiling, tPipe);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    block_.Init(tiling_, blockIdx_);

    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) { // pertensor
        block_.offset_.scaleScalar = *((__gm__ uint64_t *)antiquantScale);
    } else {
        scaleGlobal_.SetGlobalBuffer((__gm__ uint64_t *)antiquantScale);
    }

    // update global buffer
    aGlobal_.SetGlobalBuffer((__gm__ xType *)x);
    bGlobal_.SetGlobalBuffer((__gm__ wType *)weight);
    cGlobal_.SetGlobalBuffer((__gm__ yType *)y);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    ComputeBasicOptiLoop();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicOptiLoop()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);
        // 1. Set single core param
        block_.UpdateBlockParams(roundIndex);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }
        mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, tiling_->matmulTiling.Ka);
        // 2. compute offset
        block_.CalcGMOffset<aTrans, bTrans, CubeFormat::ND>();
        // 3. set offset and compute
        SetMMParaAndCompute();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMMParaAndCompute()
{
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) { // pertensor
        mm_.SetQuantScalar(block_.offset_.scaleScalar);
    } else {
        mm_.SetQuantVector(scaleGlobal_[block_.offset_.offsetScale]);
    }
    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], aTrans);
    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], bTrans);
    mm_.Iterate();
    mm_.GetTensorC(cGlobal_[block_.offset_.offsetC]);
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SLIDING_WINDOW_H
