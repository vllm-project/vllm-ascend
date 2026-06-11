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
* \file attention_update_v2_regbase.h
* \brief
 */
#ifndef ATTENTION_UPDATE_V2_REGBASE_H_
#define ATTENTION_UPDATE_V2_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_utils.h"

namespace AttentionUpdateV2Opt {
using namespace AscendC;

static constexpr uint32_t BUFFER_NUM = 1;

template <typename lseType, typename outType, bool updateType>
class AttentionUpdateV2 {
public:
    __aicore__ inline AttentionUpdateV2(TPipe *pipe, const AttentionUpdateV2TilingData* __restrict tiling)
        : pPipe_(pipe), tilingData_(tiling) {};
    __aicore__ inline void Init(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax, GM_ADDR workSpace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeMaxVF(LocalTensor<lseType> &lseTensor, LocalTensor<lseType> &expTensor, uint32_t blockStride);
    __aicore__ inline void ComputeOutVF(LocalTensor<lseType> &sumTensor, LocalTensor<lseType> &goFp32Tensor, uint32_t curBlockNum);
    __aicore__ inline void ComputeMuls(LocalTensor<lseType> &expTensor, LocalTensor<lseType> &goFp32Tensor,  uint32_t curInnerBlockNum,
                                       uint32_t curBlockNum, uint32_t innerFormerLength, uint32_t innerLoopCount);

    __aicore__ inline void CopyLseToUb(uint32_t curBlockFactor);
    __aicore__ inline void CopyLseMaxToGm(uint32_t curBlockNum);
    __aicore__ inline void CopyGoToUb(uint32_t goOffset, uint32_t curBlockNum);
    __aicore__ inline void CopyOutToGm(uint32_t outOffset, uint32_t curBlockNum);

    TPipe *pPipe_ = nullptr;
    const AttentionUpdateV2TilingData* tilingData_;

    uint32_t blockIdx;
    uint32_t usedCoreNum;
    uint32_t formerBlockNum;
    uint32_t tailBlockNum;
    uint32_t blockNum;
    uint32_t formerLength; // 满载bsh长度
    uint32_t tailLength;   // bsh尾核长度
    uint32_t hDim;         // d
    uint32_t bshSize;
    uint32_t goSize;
    uint32_t sp;
    uint32_t curLseOffset;
    uint32_t outerLoopLimit;
    uint32_t innerLoopLimit;
    uint32_t tailInnerLoopLimit;
    bool isTailCore;

    uint32_t innerFormerBlockNum;   // bsh满载核o切分块数
    uint32_t innerTailBlockNum;     // bsh满载核o切分块尾块数
    uint32_t innerFormerLength;
    uint32_t innerTailLength;

    uint32_t tailInnerFormerBlockNum;   // bsh尾块o切分块数
    uint32_t tailInnerTailBlockNum;     // bsh尾块o切分尾块数
    uint32_t tailInnerFormerLength;
    uint32_t tailInnerTailLength;

    /* Tensor List */
    ListTensorDesc lseList;
    ListTensorDesc goList;

    /* global memory address */
    GlobalTensor<lseType> lseGm;
    GlobalTensor<lseType> outLseMaxGm;
    GlobalTensor<outType> goGm;
    GlobalTensor<outType> outGm;

    /* ascendc variable */
    TQue<QuePosition::VECIN, BUFFER_NUM> lseQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> goQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outLseMaxQueue;

    TBuf<> ubTmpBuf;
};

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::Init(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax, GM_ADDR workSpace) {
    usedCoreNum = tilingData_->usedCoreNum;
    blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= usedCoreNum) {
        return;
    }

    hDim = tilingData_->hDim;
    bshSize = tilingData_->bshSize;
    sp = tilingData_->sp;
    goSize = tilingData_->goSize;

    blockNum = tilingData_->formerBlockNum + tilingData_->tailBlockNum;
    formerBlockNum = tilingData_->formerBlockNum;
    tailBlockNum = tilingData_->tailBlockNum;
    formerLength = tilingData_->formerLength;
    tailLength = tilingData_->tailLength;

    innerFormerBlockNum = tilingData_->innerFormerBlockNum;
    innerTailBlockNum = tilingData_->innerTailBlockNum;
    innerFormerLength = tilingData_->innerFormerLength;
    innerTailLength = tilingData_->innerTailLength;
    tailInnerFormerBlockNum = tilingData_->tailInnerFormerBlockNum;
    tailInnerTailBlockNum = tilingData_->tailInnerTailBlockNum;
    tailInnerFormerLength = tilingData_->tailInnerFormerLength;
    tailInnerTailLength = tilingData_->tailInnerTailLength;

    innerLoopLimit = innerFormerBlockNum + innerTailBlockNum;
    tailInnerLoopLimit = tailInnerFormerBlockNum + tailInnerTailBlockNum;

    uint32_t highLoadCoreNum = blockNum % usedCoreNum;
    if (highLoadCoreNum == 0 || highLoadCoreNum > blockIdx) {
        outerLoopLimit = (usedCoreNum == 0) ? 0u : (blockNum + usedCoreNum - 1) / usedCoreNum;
        curLseOffset = blockIdx * outerLoopLimit * formerLength;
    } else {
        outerLoopLimit = (usedCoreNum == 0) ? 0u : (blockNum + usedCoreNum - 1) / usedCoreNum;
        outerLoopLimit = outerLoopLimit - 1;
        curLseOffset = (blockIdx * outerLoopLimit + highLoadCoreNum) * formerLength;
    }
    isTailCore = (blockIdx + 1 == usedCoreNum);

    lseList = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void*>(lse));
    goList = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void*>(go));

    outLseMaxGm.SetGlobalBuffer((__gm__ lseType*)outLseMax, bshSize);
    outGm.SetGlobalBuffer((__gm__ outType*)out, bshSize * hDim);

    uint32_t maxLength = AscendC::Std::max(formerLength, tailLength);
    maxLength = (maxLength + 7) / 8 * 8;
    uint32_t maxInnerLength = AscendC::Std::max(innerFormerLength, innerTailLength);
    if (maxInnerLength == 0) {
        maxInnerLength = AscendC::Std::max(tailInnerFormerLength, tailInnerTailLength);
    }
    uint32_t maxInnerLengthAlign = (maxInnerLength * sizeof(outType) + 31) / 32 * 32;
    pPipe_->InitBuffer(lseQueue, BUFFER_NUM, sp * maxLength * sizeof(lseType));
    pPipe_->InitBuffer(outLseMaxQueue, BUFFER_NUM, maxLength * sizeof(lseType));
    pPipe_->InitBuffer(outQueue, BUFFER_NUM, maxInnerLength * sizeof(outType));
    pPipe_->InitBuffer(goQueue, BUFFER_NUM, sp * maxInnerLengthAlign);

    uint32_t ubMaxLen = (4 * sp + AscendC::Std::max(2u, sp)) * maxLength * sizeof(lseType);
    uint32_t ubTmpBufLen = ubMaxLen - (sp + 1) * (maxInnerLength * sizeof(outType) + maxLength * sizeof(lseType));
    pPipe_->InitBuffer(ubTmpBuf, ubTmpBufLen);
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::Process() {
    if (blockIdx >= usedCoreNum) {
        return;
    }

    for (uint32_t outerLoopCount = 0; outerLoopCount < outerLoopLimit; outerLoopCount++) {
        uint32_t curBlockNum = formerLength;
        uint32_t curBlockNumAlign = formerLength;
        if (isTailCore && outerLoopCount == outerLoopLimit - 1) {
            curBlockNum = tailLength;
            curBlockNumAlign = (curBlockNum + 7) / 8 * 8;
        }
        CopyLseToUb(curBlockNum);

        LocalTensor<lseType> lseTensor = lseQueue.template DeQue<lseType>();
        LocalTensor<lseType> expTensor = ubTmpBuf.Get<lseType>(sp * curBlockNumAlign);
        ComputeMaxVF(lseTensor, expTensor, curBlockNumAlign);
        lseQueue.template FreeTensor<lseType>(lseTensor);
        if constexpr (updateType) {
            CopyLseMaxToGm(curBlockNum);
        }

        uint32_t curInnerLoopLimit = innerLoopLimit;
        uint32_t curInnerFormerLength = innerFormerLength;
        uint32_t curInnerTailLength = innerTailLength;
        uint32_t curIinnerFormerBlockNum = innerFormerBlockNum;
        if (isTailCore) {
            curInnerLoopLimit = tailInnerLoopLimit;
            curInnerFormerLength = tailInnerFormerLength;
            curInnerTailLength = tailInnerTailLength;
            curIinnerFormerBlockNum = tailInnerFormerBlockNum;
        }
        for (uint32_t innerLoopCount = 0; innerLoopCount < curInnerLoopLimit; innerLoopCount++) {
            uint32_t curInnerBlockNum = curInnerFormerLength;
            if (innerLoopCount == curIinnerFormerBlockNum) {
                curInnerBlockNum = curInnerTailLength;
            }
            uint32_t curInnerBlockNumAlign = (curInnerBlockNum * sizeof(outType) + 31) / 32 * 32 / sizeof(outType);
            uint32_t goOffset = curLseOffset * hDim + innerLoopCount * curInnerFormerLength;
            CopyGoToUb(goOffset, curInnerBlockNum);
            LocalTensor<outType> goTensor = goQueue.template DeQue<outType>();
            LocalTensor<lseType> goFp32Tensor = ubTmpBuf.GetWithOffset<lseType>(sp * curInnerBlockNumAlign, sp * curBlockNumAlign * sizeof(lseType));
            if constexpr (IsSameType<outType, float>::value) {
                Muls(goFp32Tensor, goTensor, 1.0f, sp * curInnerBlockNumAlign);
            } else {
                Cast(goFp32Tensor, goTensor, RoundMode::CAST_NONE, sp * curInnerBlockNumAlign);
            }
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            ComputeMuls(expTensor, goFp32Tensor, curInnerBlockNumAlign, curBlockNumAlign, curInnerFormerLength, innerLoopCount);
            LocalTensor<lseType> sumTensor = ubTmpBuf.GetWithOffset<lseType>(curInnerBlockNumAlign, sp * curBlockNumAlign * sizeof(lseType));
            ComputeOutVF(sumTensor, goFp32Tensor, curInnerBlockNumAlign);
            LocalTensor<outType> outTensor = outQueue.template AllocTensor<outType>();
            if constexpr (IsSameType<outType, float>::value) {
                Muls(outTensor, sumTensor, 1.0f, curInnerBlockNumAlign);
            } else {
                Cast(outTensor, sumTensor, RoundMode::CAST_RINT, curInnerBlockNumAlign);
            }
            outQueue.template EnQue<outType>(outTensor);
            goQueue.template FreeTensor<outType>(goTensor);
            CopyOutToGm(goOffset, curInnerBlockNum);
        }
        curLseOffset += curBlockNumAlign;
    }
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::ComputeMuls(LocalTensor<lseType> &expTensor, LocalTensor<lseType> &goFp32Tensor,
                                                                                  uint32_t innerBlockNum, uint32_t curBlockNum,
                                                                                  uint32_t innerFormerLength, uint32_t innerLoopCount) {
    for (uint32_t i = 0; i < sp; i++) {
        for (uint32_t j = 0; j < innerBlockNum / hDim; j ++) {
            uint32_t scalarOffset = innerLoopCount * innerFormerLength / hDim + j;
            lseType scalar = expTensor.GetValue(i * curBlockNum + scalarOffset);
            AscendC::Muls(goFp32Tensor[i * innerBlockNum + j * hDim], goFp32Tensor[i * innerBlockNum + j * hDim], scalar, hDim);
        }
    }
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::ComputeMaxVF(LocalTensor<lseType> &lseTensor, LocalTensor<lseType> &expTensor, uint32_t curBlockNum) {
    LocalTensor<lseType> maxTensor = outLseMaxQueue.template AllocTensor<lseType>();

    __local_mem__ lseType* lseAddr = (__local_mem__ lseType*)lseTensor.GetPhyAddr();
    __local_mem__ lseType* maxAddr = (__local_mem__ lseType*)maxTensor.GetPhyAddr();
    __local_mem__ lseType* expAddr = (__local_mem__ lseType*)expTensor.GetPhyAddr();

    uint16_t blockStride = static_cast<uint16_t>(curBlockNum);
    uint16_t spSize = static_cast<uint16_t>(sp);

    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (blockStride + VL - 1) / VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<lseType> vreg1; //搬入的lse
        AscendC::MicroAPI::RegTensor<lseType> vreg2; //reduce max
        AscendC::MicroAPI::RegTensor<lseType> vreg3;
        AscendC::MicroAPI::RegTensor<lseType> vreg4;
        AscendC::MicroAPI::RegTensor<lseType> vreg5; // 累加 reduce sum
        AscendC::MicroAPI::RegTensor<lseType> vreg6;
        AscendC::MicroAPI::RegTensor<lseType> vreg7; // 可选输出 lse_max
        AscendC::MicroAPI::RegTensor<lseType> vreg8;
        AscendC::MicroAPI::RegTensor<lseType> vreg9;

        AscendC::MicroAPI::MaskReg preg1;
        for (uint16_t i = 0; i < vfLoop; i ++) {
            preg1 = AscendC::MicroAPI::UpdateMask<lseType, MicroAPI::RegTraitNumOne>(curBlockNum);
            AscendC::MicroAPI::DataCopy<lseType>(vreg1, lseAddr + i * VL);
            for (uint16_t j = 0; j < spSize; j++) {
                AscendC::MicroAPI::DataCopy<lseType>(vreg1, lseAddr + i * VL + j * blockStride);
                AscendC::MicroAPI::Max<lseType>(vreg2, vreg2, vreg1, preg1);
            }

            AscendC::MicroAPI::Duplicate(vreg5, 0, preg1);
            for (uint16_t j = 0; j < spSize; j++) {
                AscendC::MicroAPI::DataCopy<lseType>(vreg1, lseAddr + i * VL + j * blockStride);
                AscendC::MicroAPI::Sub<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg3, vreg1, vreg2, preg1);
                AscendC::MicroAPI::Exp<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg3, preg1);
                AscendC::MicroAPI::Add<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg4, vreg5, preg1);
            }
            AscendC::MicroAPI::Log<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg6, vreg5, preg1);
            AscendC::MicroAPI::Add<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg7, vreg6, vreg2, preg1);
            AscendC::MicroAPI::DataCopy<lseType>(maxAddr + i * VL, vreg7, preg1);
            for (uint16_t j = 0; j < spSize; j ++) {
                AscendC::MicroAPI::DataCopy<lseType>(vreg1, lseAddr + i * VL + j * blockStride);
                AscendC::MicroAPI::Sub<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg8, vreg1, vreg7, preg1);
                AscendC::MicroAPI::Exp<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg9, vreg8, preg1);
                AscendC::MicroAPI::DataCopy<lseType>(expAddr + i * VL + j * blockStride, vreg9, preg1);
            }
        }
    }
    if constexpr (updateType) {
        outLseMaxQueue.template EnQue<lseType>(maxTensor);
    }
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::ComputeOutVF(LocalTensor<lseType> &sumTensor, LocalTensor<lseType> &goFp32Tensor, uint32_t curBlockNum) {
    __local_mem__ lseType* sumAddr = (__local_mem__ lseType*)sumTensor.GetPhyAddr();
    __local_mem__ lseType* goFp32Addr = (__local_mem__ lseType*)goFp32Tensor.GetPhyAddr();

    uint16_t blockStride = static_cast<uint16_t>(curBlockNum);
    uint16_t spSize = static_cast<uint16_t>(sp);

    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (blockStride + VL - 1) / VL;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<lseType> vreg1; //搬入的lse
        AscendC::MicroAPI::RegTensor<lseType> vreg2; //累加 reduce max

        AscendC::MicroAPI::MaskReg preg1;
        for (uint16_t i = 0; i < vfLoop; i ++) {
            preg1 = AscendC::MicroAPI::UpdateMask<lseType, MicroAPI::RegTraitNumOne>(curBlockNum);
            AscendC::MicroAPI::Duplicate(vreg2, 0, preg1);
            for (uint16_t j = 0; j < spSize; j++) {
                AscendC::MicroAPI::DataCopy<lseType>(vreg1, goFp32Addr + i * VL + j * blockStride);
                AscendC::MicroAPI::Add<lseType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg2, vreg2, vreg1, preg1);
            }
            AscendC::MicroAPI::DataCopy<lseType>(sumAddr + i * VL, vreg2, preg1);
        }
    }
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::CopyLseToUb(uint32_t curBlockNum)
{
    LocalTensor<lseType> lseTensor = lseQueue.template AllocTensor<lseType>();
    uint32_t blockLen = curBlockNum * sizeof(lseType);

    DataCopyExtParams params = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(blockLen),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPadExtParams<lseType> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<lseType>(0)};
    uint32_t alignBlockNum = (blockLen + 31) / 32 * 32 / sizeof(lseType);
    for (uint32_t i = 0; i < sp; i++) {
        lseGm.SetGlobalBuffer((__gm__ lseType*)lseList.GetDataPtr<lseType>(i));
        DataCopyPad(lseTensor[i * alignBlockNum], lseGm[curLseOffset], params, padParamIdx);
    }
    lseQueue.template EnQue<lseType>(lseTensor);
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::CopyGoToUb(uint32_t goOffset, uint32_t curBlockNum)
{
    LocalTensor<outType> goTensor = goQueue.template AllocTensor<outType>();
    uint32_t blockLen = curBlockNum * sizeof(outType);
    DataCopyExtParams params = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(blockLen),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPadExtParams<outType> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<outType>(0)};
    uint32_t alignBlockNum = (blockLen + 31) / 32 * 32 / sizeof(outType);
    for (uint32_t i = 0; i < sp; i++) {
        goGm.SetGlobalBuffer((__gm__ outType*)goList.GetDataPtr<outType>(i));
        DataCopyPad(goTensor[i * alignBlockNum], goGm[goOffset], params, padParamIdx);
    }
    goQueue.template EnQue<outType>(goTensor);
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::CopyLseMaxToGm(uint32_t curBlockNum)
{
    LocalTensor<lseType> outLseMaxTensor = outLseMaxQueue.template DeQue<lseType>();
    uint32_t blockLen = curBlockNum * sizeof(lseType);

    DataCopyExtParams params = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(blockLen),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPad(outLseMaxGm[curLseOffset], outLseMaxTensor, params);
    outLseMaxQueue.template FreeTensor<lseType>(outLseMaxTensor);
}

template <typename lseType, typename outType, bool updateType>
__aicore__ inline void AttentionUpdateV2<lseType, outType, updateType>::CopyOutToGm(uint32_t outOffset, uint32_t curBlockNum)
{
    LocalTensor<outType> outTensor = outQueue.template DeQue<outType>();
    uint32_t blockLen = curBlockNum * sizeof(outType);
    DataCopyExtParams params = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(blockLen),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPad(outGm[outOffset], outTensor, params);
    outQueue.template FreeTensor<outType>(outTensor);
}
}
#endif  // ATTENTION_UPDATE_V2_REGBASE_H_