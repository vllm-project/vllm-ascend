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
 * \file qbmm_mix_online_dynamic_al1_full_load.h
 * \brief
 */
#ifndef MC2_QBMM_MIX_ONLINE_DYNAMIC_AL1_FULL_LOAD_H
#define MC2_QBMM_MIX_ONLINE_DYNAMIC_AL1_FULL_LOAD_H

#include "qbmm_mix_online_dynamic.h"
#include "qbmm_api_utils.h"

namespace Mc2QuantBatchMatmulV3 {

using namespace AscendC;
using namespace matmul;

template <class aType, class bType, class scaleType, class biasType, class ptScaleType, class cType, CubeFormat aFormat,
          CubeFormat bFormat, CubeFormat cFormat, bool aTrans, bool bTrans, class l0cDtype,
          class blockType = Mc2QuantBmmAswBlock, const MatmulConfig &mmCfg = MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>
class Mc2QuantBmmPertokenAL1FullLoad
    : public Mc2QuantBmmPertokenRegbaseKernel<aType, bType, scaleType, biasType, ptScaleType, cType, aFormat, bFormat,
                                              cFormat, aTrans, bTrans, l0cDtype, blockType, mmCfg> {
public:
    __aicore__ inline Mc2QuantBmmPertokenAL1FullLoad()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias, GM_ADDR ptScale,
                                GM_ADDR cGM, GM_ADDR workSpace, const void *tilingData, TPipe *pipe);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithoutBatch();

protected:
    __aicore__ inline void MMCompute();
    using aT = AscendC::MatmulType<TPosition::TSCM, aFormat, aType, aTrans>;
    using bT = AscendC::MatmulType<TPosition::GM, bFormat, bType, bTrans>;
    using biasT = AscendC::MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    using cT = AscendC::MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, l0cDtype>;
    AscendC::MatmulImpl<aT, bT, cT, biasT, mmCfg, AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                        AscendC::QBmmCustomMatmulPolicy>
        mm;
    TQue<QuePosition::A1, 1> InQueueAL1_;
    LocalTensor<aType> al1Local_;
    bool isMMultiCore_;
};

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenAL1FullLoad<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias, GM_ADDR ptScale, GM_ADDR cGM,
    GM_ADDR workSpace, const void *tilingData, TPipe *pipe)
{
    this->blockIdx_ = GetBlockIdx();
    if ASCEND_IS_AIV {
        this->blockIdx_ = this->blockIdx_ / AscendC::GetTaskRation();
        this->subBlockIdx_ = AscendC::GetSubBlockIdx();
    }
    this->pipe_ = pipe;
    this->tilingData_ = static_cast<const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *>(tilingData);

    this->biasDtype_ = this->tilingData_->params.biasDtype;
    this->isBiasEpilogue_ =
        IsSameType<aType, int8_t>::value &&
        (this->biasDtype_ == DT_BF16 || this->biasDtype_ == DT_FLOAT16 || this->biasDtype_ == DT_FLOAT);
    this->UpdateGlobalAddr(aGM, bGM, scale, bias, ptScale, cGM, workSpace);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenAL1FullLoad<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::InitBuffer()
{
    if ASCEND_IS_AIC {
        uint64_t aL1Size = DequantBmm::CalcAL1Size<aType, aTrans>(this->block_.tilingData_->matmulTiling.singleCoreM,
                                                                  this->block_.tilingData_->matmulTiling.Ka);
        this->pipe_->InitBuffer(InQueueAL1_, 1, aL1Size); // m k的计算
        al1Local_ = InQueueAL1_.AllocTensor<aType>();
        CopyInA1<aType, aTrans>(this->block_, this->blockIdx_, isMMultiCore_, al1Local_, this->aGlobal_);
        mm.Init(&this->block_.tilingData_->matmulTiling, this->pipe_);
    }
    uint32_t mForSingleVec = DequantBmm::CeilDiv(this->tilingData_->matmulTiling.baseM, CV_RATIO);
    this->pipe_->InitBuffer(this->vecQueMMRes_, 1,
                            mForSingleVec * this->tilingData_->matmulTiling.baseN * sizeof(l0cDtype));
    this->l0cOutUb_ = this->vecQueMMRes_.template AllocTensor<l0cDtype>();
    if ASCEND_IS_AIV {
        if (!static_cast<bool>(this->tilingData_->params.isPerTensor)) {
            this->pipe_->InitBuffer(this->vecQueScale_, 1, this->tilingData_->matmulTiling.baseN * sizeof(scaleType));
        }
        if (static_cast<bool>(this->tilingData_->params.isPertoken)) {
            this->pipe_->InitBuffer(
                this->vecQuePertokenScale_, 1,
                DequantBmm::Align(mForSingleVec * sizeof(ptScaleType), static_cast<uint64_t>(DATA_BLOCK)));
        }
        if (this->isBiasEpilogue_) {
            if (this->biasDtype_ == DT_FLOAT) {
                this->pipe_->InitBuffer(this->vecQueBias_, 1, this->tilingData_->matmulTiling.baseN * sizeof(float));
            } else {
                this->pipe_->InitBuffer(this->vecQueBias_, 1,
                                        this->tilingData_->matmulTiling.baseN * sizeof(bfloat16_t));
            }
        }
        // fp16/bf16分两次输出，fp32分四次输出
        this->pipe_->InitBuffer(this->vecQueOut_, BUFFER_NUM,
                                DequantBmm::CeilDiv(mForSingleVec, FP32_OUTPUT_TIMES) *
                                    this->tilingData_->matmulTiling.baseN * sizeof(cType));
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenAL1FullLoad<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Process()
{
    isMMultiCore_ = this->block_.tilingData_->matmulTiling.singleCoreM < this->block_.tilingData_->matmulTiling.M;
    uint64_t innerAlignedBlock = ONE_BLK_SIZE / sizeof(aType);
    uint64_t mAligned = 0;
    uint64_t kAligned = 0;
    if constexpr (aTrans) {
        mAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreM, innerAlignedBlock);
        kAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, BMM_BLOCK_NUM);
    } else {
        mAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.singleCoreM, BMM_BLOCK_NUM);
        kAligned = DequantBmm::Align(this->block_.tilingData_->matmulTiling.Ka, innerAlignedBlock);
    }
    if (!isMMultiCore_) {
        this->block_.params_.mCnt = 1;
        this->block_.params_.mainRow = 0;
        this->block_.params_.totalCnt = this->block_.params_.nCnt;
    }
    InitBuffer();
    if (likely(this->tilingData_->params.batchC == 1)) {
        this->block_.offset_.batchCOffset = 0;
        this->block_.offset_.batchAOffset = 0;
        this->block_.offset_.batchBOffset = 0;
        ProcessWithoutBatch();
    } else {
        ProcessWithBatch<Mc2QuantBmmPertokenAL1FullLoad>(this->block_, *this);
    }
    if ASCEND_IS_AIC {
        InQueueAL1_.FreeTensor(al1Local_);
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenAL1FullLoad<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::ProcessWithoutBatch()
{
    bool isVecSetSyncCom = false;
    for (uint64_t j = 0; j < this->block_.params_.round; j++) {
        if (j == (this->block_.params_.round - 1) && this->block_.params_.totalTailTile > 1) {
            this->block_.UpdateBasicIndex4AL1FullLoad(j);
        } else {
            this->block_.UpdateBasicIndex(j);
        }
        if (this->block_.params_.index < this->block_.params_.totalCnt) {
            if (j == (this->block_.params_.round - 1) && this->block_.params_.totalTailTile > 1) {
                this->block_.template UpdateBlockParams4AL1FullLoad<bTrans, bFormat>(j);
            } else {
                this->block_.template UpdateBlockParams<bTrans, bFormat>(j);
            }
            if (this->block_.params_.singleCoreM > 0 && this->block_.params_.singleCoreN > 0) {
                this->block_.template CalcGMOffset<aTrans, bTrans, aType, scaleType, bFormat>();
                if ASCEND_IS_AIC {
                    mm.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                                      this->tilingData_->matmulTiling.singleCoreK);
                    if (this->block_.offset_.batchCOffset * this->block_.params_.round + j > 0) {
                        this->WaitForVector();
                    }
                    MMCompute();
                    this->NotifyVector();
                }
                isVecSetSyncCom = true;
                if ASCEND_IS_AIV {
                    this->WaitForCube();
                    this->DequantCompute();
                    this->NotifyCube();
                }
            }
        }
    }
    // 由于vec最后一次会通过NotifyCube多发一次硬同步，所以cube侧需要额外加一次硬同步
    if ASCEND_IS_AIC {
        if (this->block_.offset_.batchCOffset == this->tilingData_->params.batchC - 1 && isVecSetSyncCom) {
            this->WaitForVector();
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenAL1FullLoad<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::MMCompute()
{
    mm.SetTensorA(al1Local_, aTrans);
    mm.SetTensorB(this->bGlobal_[this->block_.offset_.offsetB], bTrans);
    if (static_cast<bool>(this->tilingData_->matmulTiling.isBias) && !this->isBiasEpilogue_) {
        mm.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
    }

    if (isMMultiCore_) {
        // MDL模板，L1输入场景默认al1M=M，分核全载需要通过设置SetOrgShape指定al1M=singleCoreM
        mm.SetOrgShape(this->block_.params_.singleCoreM, this->block_.tilingData_->matmulTiling.N,
                       this->block_.tilingData_->matmulTiling.Ka);
    }
    mm.Iterate();
    mm.GetTensorC(this->l0cOutUb_, 0, true);
}

} // namespace Mc2QuantBatchMatmulV3

#endif // QBMM_MIX_ONLINE_DYNAMIC_AL1_FULL_LOAD_H
