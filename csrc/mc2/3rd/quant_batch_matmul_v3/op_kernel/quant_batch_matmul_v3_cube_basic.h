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
 * \file quant_batch_matmul_v3_cube_basic.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_CUBE_BASIC_H
#define MC2_QUANT_BATCH_MATMUL_V3_CUBE_BASIC_H

#include "quant_batch_matmul_v3_block.h"
#include "quant_batch_matmul_v3_update.h"

namespace AscendC {
template <TemplateBasicType>
class Mc2QuantBatchMatmulV3BaseKernel { // 纯cube kernel，无pertoken，输出int8/fp16/int32
public:
    __aicore__ inline Mc2QuantBatchMatmulV3BaseKernel()
    {
    }

    __aicore__ inline void InitInputs(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR y);

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR y, GM_ADDR workSpace,
                                const Mc2QuantBatchMatmulV3TilingData *__restrict tilingData, TPipe *tPipe);

    __aicore__ inline void Process();
    __aicore__ inline UPDATE_TYPE &GetUpdateObj()
    {
        return update_;
    }

protected:
    __aicore__ inline void MMCompute();
    __aicore__ inline void OneTileCompute(uint64_t mTileIndex, uint64_t nTileIndex);

    Mc2QuantBatchMatmulV3BaseBlock block_;
    UPDATE_TYPE update_; // 量化mm或mc2的更新计算大小和地址的接口
    QBmmBlockOffset offset_;
    using A_TYPE = matmul::MatmulType<TPosition::GM, DequantBmm::GetFormat(x1Format), x1Type, aTrans>;
    using B_TYPE = matmul::MatmulType<TPosition::GM, DequantBmm::GetFormat(x2Format), x2Type, bTrans>;
    using BIAS_TYPE = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using C_TYPE = matmul::MatmulType<TPosition::GM, CubeFormat::ND, yType>;
    matmul::MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG> mm_;
    GlobalTensor<x1Type> x1Global_;
    GlobalTensor<x2Type> x2Global_;
    GlobalTensor<uint64_t> scaleGlobal_; // aic内随路反量化的数据类型是uint64, scaleType可能是int64/uint64
    uint64_t scaleScalar_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<int32_t> biasGlobal_; // aic内计算的bias int32类型
    bool isPerTensor_;
    TPipe *pipe_;
};

template <TemplateBasicType>
__aicore__ inline void Mc2QuantBatchMatmulV3BaseKernel<TemplateBasicValue>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR y, GM_ADDR workSpace,
    const Mc2QuantBatchMatmulV3TilingData *__restrict tilingData, TPipe *tPipe)
{
    block_.Init(tilingData);
    update_.template Init<x1Format, x2Format, aTrans, bTrans>(&tilingData->matmulTiling, block_.params_);
    pipe_ = tPipe;
    isPerTensor_ = tilingData->params.isPerTensor;
    InitInputs(x1, x2, scale, bias, y);

    mm_.SetSubBlockIdx(0);
    mm_.Init(block_.matmulTilingData_, pipe_);
}

template <TemplateBasicType>
__aicore__ inline void Mc2QuantBatchMatmulV3BaseKernel<TemplateBasicValue>::InitInputs(GM_ADDR x1, GM_ADDR x2,
                                                                                       GM_ADDR scale, GM_ADDR bias,
                                                                                       GM_ADDR y)
{
    x1Global_.SetGlobalBuffer((__gm__ x1Type *)x1);
    x2Global_.SetGlobalBuffer((__gm__ x2Type *)x2);
    // 双页表
    if (block_.matmulTilingData_->M <= block_.matmulTilingData_->baseM) {
        x2Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    scaleGlobal_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint64_t *>(scale)); // int64/uint64 -> uint64, 硬件不关心符号位
    if (isPerTensor_) {
        scaleScalar_ = *((__gm__ uint64_t *)scale);
    }
    biasGlobal_.SetGlobalBuffer((__gm__ int32_t *)bias); // 不存在时也可以set，真正计算时需要判断
    yGlobal_.SetGlobalBuffer((__gm__ yType *)y);
}

template <TemplateBasicType>
__aicore__ inline void Mc2QuantBatchMatmulV3BaseKernel<TemplateBasicValue>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }

    // 首次计算，兼容无L2cache切分场景，减少scalar计算
    block_.InitFirstTileBlockIndex();
    OneTileCompute(0, 0);
    bool reverse = true;
    for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            if (mTileIndex > 0 || nTileIndex > 0) { // 跳过首块
                block_.UpdateBlockCnt(mTileIndex, nTileIndex);
                block_.InitBlockIndex();
                OneTileCompute(mTileIndex, nTileIndex);
            }
        }
    }
    mm_.End();
}

template <TemplateBasicType>
__aicore__ inline void Mc2QuantBatchMatmulV3BaseKernel<TemplateBasicValue>::OneTileCompute(uint64_t mTileIndex,
                                                                                           uint64_t nTileIndex)
{
    for (uint64_t j = 0; j < block_.realRound_; j++) {
        // 更新此次基本块的大小和输入输出地址
        update_.template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans>(block_.params_, offset_,
                                                                                              mTileIndex, nTileIndex);

        MMCompute();
        block_.UpdateBlockIndex();
    }
}

template <TemplateBasicType>
__aicore__ inline void Mc2QuantBatchMatmulV3BaseKernel<TemplateBasicValue>::MMCompute()
{
    mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, block_.matmulTilingData_->singleCoreK);
    mm_.SetTensorA(x1Global_[offset_.offsetA], aTrans);
    mm_.SetTensorB(x2Global_[offset_.offsetB], bTrans);
    if constexpr (!IsSameType<yType, int32_t>::value) { // 非int32输出的，需要随路反量化
        if (isPerTensor_) {
            mm_.SetQuantScalar(scaleScalar_);
        } else {
            mm_.SetQuantVector(scaleGlobal_[offset_.offsetScale]);
        }
    }
    if (block_.matmulTilingData_->isBias) {
        mm_.SetBias(biasGlobal_[offset_.offsetBias]);
    }
    mm_.Iterate();
    mm_.GetTensorC(yGlobal_[offset_.offsetC]);
}
} // namespace AscendC
#endif // QUANT_BATCH_MATMUL_V3_CUBE_BASIC_H
