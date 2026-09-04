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
 * \file infer_flash_attention_kvcache.h
 * \brief
 */
#ifndef INFER_FLASH_ATTENTION_KVCACHE_H
#define INFER_FLASH_ATTENTION_KVCACHE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "infer_flash_attention_comm.h"
#include "infer_flash_attention_sparse.h"
#define NUM_128 128
using namespace matmul;

TEMPLATE_INTF
__aicore__ inline void InitQueryLeftPaddingSize(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer,
    hasRope>& constInfo, int64_t& actualS1Size)
{
    if (!constInfo.isQHasLeftPadding) {
        runParam.queryLeftPaddingSize = 0;
    } else {
        int64_t qLeftPaddingSize = 0;
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
            qLeftPaddingSize = constInfo.s1Size - actualS1Size / constInfo.gSize - constInfo.queryRightPaddingSize;
        } else {
            qLeftPaddingSize = constInfo.s1Size - actualS1Size - constInfo.queryRightPaddingSize;
        }
        runParam.queryLeftPaddingSize = qLeftPaddingSize > 0 ? qLeftPaddingSize : 0;
        if (qLeftPaddingSize < 0) {
            actualS1Size = 0;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitKVLeftPaddingSize(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer,
    hasRope>& constInfo, int64_t& actualS2Size)
{
    if (!constInfo.isKVHasLeftPadding) {
        runParam.kvLeftPaddingSize = 0;
    } else {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualS2Size - constInfo.kvRightPaddingSize;
        runParam.kvLeftPaddingSize = kvLeftPaddingSize > 0 ? kvLeftPaddingSize : 0;
        if (kvLeftPaddingSize < 0) {
            actualS2Size = 0;
        }
    }
}

static constexpr uint32_t DIM_NUM2 = 2;
TEMPLATE_INTF
__aicore__ inline void GetKVSeqLengthForTensorList(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx, GlobalTensor<INPUT_T>& keyGm)
{
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void*)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, bIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(DIM_NUM2);
        } else {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
}

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS1Size(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx,
    __gm__ int64_t* actualSeqQlenAddr, int64_t actualSeqMin)
{
    if (constInfo.isActualLenDimsNull) {
        int64_t actualMSize = constInfo.s1Size;
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) { // IFA MLA
            actualMSize = constInfo.gS1;
            runParam.actualSeqLengthOfMlaPerBatch = constInfo.s1Size;
        }
        return actualMSize;
    }
    
    int64_t actualMSize = 0;
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            runParam.actualSeqLengthOfMlaPerBatch = 
                (constInfo.actualSeqLenSize == actualSeqMin) ?
                actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx];
            actualMSize = runParam.actualSeqLengthOfMlaPerBatch * constInfo.gSize;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            runParam.actualSeqLengthOfMlaPerBatch = (bIdx == 0) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
            actualMSize = runParam.actualSeqLengthOfMlaPerBatch * constInfo.gSize;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            runParam.actualSeqLengthOfMlaPerBatch = 
                (constInfo.actualSeqLenSize == actualSeqMin) ?
                actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx];
            if (runParam.actualSeqLengthOfMlaPerBatch == 0) {
                actualMSize = 0;
            } else {
                actualMSize = constInfo.gS1;
            }
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || 
                         layout == LayOutTypeEnum::LAYOUT_NTD) {
        actualMSize = (bIdx == 0) ? actualSeqQlenAddr[0] :
            actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
    } else {
        actualMSize = (constInfo.actualSeqLenSize == actualSeqMin) ? 
            actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx];
    }
    
    return actualMSize;
}

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS2Size(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx,
    __gm__ int64_t* actualSeqKvlenAddr, int64_t actualSeqKVMin)
{
    int64_t actualS2Size = 0;
    
    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = (constInfo.isKvContinuous == 1) ? constInfo.s2Size :
            runParam.s2InCurrentBatch;
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
            actualS2Size = (isPa && constInfo.actualSeqLenKVSize == actualSeqKVMin) ? 
                actualSeqKvlenAddr[0] : actualSeqKvlenAddr[bIdx];
            if ((bIdx > 0) && (!isPa)) {
                actualS2Size -= actualSeqKvlenAddr[bIdx - 1];
            }
        } else {
            actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ? 
                actualSeqKvlenAddr[0] : actualSeqKvlenAddr[bIdx];
        }
    }
    
    return actualS2Size;
}

TEMPLATE_INTF
__aicore__ inline void AdjustActualS1Size(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo)
{
    if (!constInfo.isGqa) {
        if constexpr (enableKVPrefix) {
            runParam.actualS1Size = (runParam.actualS1Size >
                runParam.actualS2Size + constInfo.actualKVPrefixSize + runParam.preTokensPerBatch) ?
                runParam.actualS2Size + constInfo.actualKVPrefixSize + runParam.preTokensPerBatch :
                runParam.actualS1Size;
        } else {
            if constexpr ((hasRope && (dTemplateType == DTemplateType::Aligned576)) &&
                layout != LayOutTypeEnum::LAYOUT_BNSD) {
                runParam.actualS1Size = (runParam.actualS1Size >
                    runParam.actualS2Size * constInfo.gSize + runParam.preTokensPerBatch) ?
                    runParam.actualS2Size * constInfo.gSize + runParam.preTokensPerBatch :
                    runParam.actualS1Size;
            } else {
                runParam.actualS1Size = (runParam.actualS1Size >
                    runParam.actualS2Size + runParam.preTokensPerBatch) ?
                    runParam.actualS2Size + runParam.preTokensPerBatch :
                    runParam.actualS1Size;
            }
        }
        // 计算S1的尾块大小，非对齐
        if (runParam.nextTokensPerBatch < 0) {
            runParam.actualS1Size = runParam.actualS1Size + runParam.nextTokensPerBatch;
        }
    }

    if (runParam.actualS1Size < 0) { // 修正preToken/nextToken导致全无效场景的qs值
        runParam.actualS1Size = 0;
    }
}

TEMPLATE_INTF
__aicore__ inline void CalculateQueryOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx,
    __gm__ int64_t* actualSeqQlenAddr)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        runParam.qBOffset = bIdx * constInfo.s1Size * constInfo.n2GD + 
                            runParam.queryLeftPaddingSize * constInfo.n2GD;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runParam.qBOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1] * constInfo.n2GD;
    } else {
        runParam.qBOffset = bIdx * constInfo.s1Size * constInfo.n2GD + 
                            runParam.queryLeftPaddingSize * constInfo.dSize;
    }
}

TEMPLATE_INTF
__aicore__ inline void CalculateRopeOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx,
    __gm__ int64_t* actualSeqQlenAddr)
{
    if constexpr (hasRope) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            runParam.qRopeBOffset = bIdx * constInfo.s1Size * constInfo.n2GDR +
                runParam.queryLeftPaddingSize * constInfo.n2GDR;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            runParam.qRopeBOffset = (bIdx == 0) ? 0 : 
                                   actualSeqQlenAddr[bIdx - 1] * constInfo.n2GDR;
        } else {
            runParam.qRopeBOffset = bIdx * constInfo.s1Size * constInfo.n2GDR +
                runParam.queryLeftPaddingSize * constInfo.dSizeRope;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr<isInfer>& runParam, 
    const ConstInfo<isInfer, hasRope> &constInfo, const AttenMaskInfo &attenMaskInfo, int32_t bIdx,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t * actualSeqKvlenAddr)
{
    constexpr int64_t actualSeqMin = 1;
    constexpr int64_t actualSeqKVMin = 1;
    
    // TensorList场景获取不同batch的KvSeq长度
    GetKVSeqLengthForTensorList<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, keyGm);

    // 计算实际序列长度
    int64_t actualS1Size = CalculateActualS1Size<TEMPLATE_INTF_ARGS>(
        runParam, constInfo, bIdx, actualSeqQlenAddr, actualSeqMin);
    int64_t actualS2Size = CalculateActualS2Size<TEMPLATE_INTF_ARGS>(
        runParam, constInfo, bIdx, actualSeqKvlenAddr, actualSeqKVMin);

    // 初始化padding大小
    InitQueryLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS1Size);
    InitKVLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS2Size);

    runParam.actualS1Size = actualS1Size;
    runParam.actualS2Size = actualS2Size;
    GetSparseParam<TEMPLATE_INTF_ARGS>(constInfo, attenMaskInfo, runParam);

    // 调整S1大小以适应per token、next token设置
    AdjustActualS1Size<TEMPLATE_INTF_ARGS>(runParam, constInfo);

    // 计算query偏移量
    CalculateQueryOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqQlenAddr);
    
    // 推理的TND场景的mask和pse都是padding过的
    runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
    // 推理的mask的sequence length可能大于qk的sequence length
    runParam.b1SSAttenMaskOffset = runParam.boIdx * (uint64_t)attenMaskInfo.attenMaskS1Size *
        (uint64_t)attenMaskInfo.attenMaskS2Size;
        
    // 计算RoPE偏移量
    CalculateRopeOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqQlenAddr);
}

TEMPLATE_INTF
__aicore__ inline void GetKeyCoreOffsetParam(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t keyInnerOffsetSize = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        if (constInfo.isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            keyInnerOffsetSize = bIdx * constInfo.n2S2D + runParam.kvLeftPaddingSize * constInfo.n2D;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;

    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            keyInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2D;
        } else {
            keyInnerOffsetSize = bIdx * constInfo.n2S2D;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;
    } else {
        uint64_t headStrideK = 0;
        if (constInfo.isKvContinuous == 1) {
            headStrideK = constInfo.s2D;
            keyInnerOffsetSize = bIdx * constInfo.n2Size * headStrideK +
                runParam.kvLeftPaddingSize * constInfo.dSize;
        } else {
            headStrideK = constInfo.dSize * runParam.s2InCurrentBatch;
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStrideK;
    }
}

TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBSH(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx)
{
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        valueInnerOffsetSize = bIdx * constInfo.n2S2Dv + runParam.kvLeftPaddingSize * constInfo.n2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
}

TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetTND(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx, __gm__ int64_t* actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    if constexpr (!isPa) {
        valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2Dv;
    } else {
        valueInnerOffsetSize = bIdx * constInfo.n2S2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
}

TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetNTD(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx, __gm__ int64_t* actualSeqKvlenAddr)
{
    uint64_t actualSeqKVLen = 0;
    uint64_t valueInnerOffsetSize = 0;
    if constexpr (isPa) {
        actualSeqKVLen = constInfo.s2Size;
        valueInnerOffsetSize = bIdx * constInfo.s2Dv;
    } else {
        actualSeqKVLen = (bIdx == 0) ? actualSeqKvlenAddr[0] : actualSeqKvlenAddr[bIdx] - actualSeqKvlenAddr[bIdx - 1];
        valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.dSizeV;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize +
        runParam.n2oIdx * constInfo.bSize * actualSeqKVLen * constInfo.dSizeV;
}

TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBNSD(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx)
{
    uint64_t headStrideV = 0;
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        headStrideV = constInfo.s2Dv;
        valueInnerOffsetSize = bIdx * constInfo.n2Size * headStrideV +
            runParam.kvLeftPaddingSize * constInfo.dSizeV;
    } else {
        headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
        valueInnerOffsetSize = 0;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
}

TEMPLATE_INTF
__aicore__ inline void SyncKeyCoreOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int32_t bIdx, __gm__ int64_t* actualSeqKvlenAddr)
{
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}

TEMPLATE_INTF
__aicore__ inline void CalcPrefixCoreOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo)
{
    if constexpr (enableKVPrefix && layout == LayOutTypeEnum::LAYOUT_BSH) {
        uint64_t prefixInnerOffsetSize = runParam.kvLeftPaddingSize * constInfo.n2Dv;
        runParam.prefixCoreOffset = prefixInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
    } else if constexpr (enableKVPrefix) {
        uint64_t headStrideV = constInfo.kvPrefixSize * constInfo.dSizeV;
        uint64_t prefixInnerOffsetSize = runParam.kvLeftPaddingSize * constInfo.dSizeV;
        runParam.prefixCoreOffset = prefixInnerOffsetSize + runParam.n2oIdx * headStrideV;
    }
}

TEMPLATE_INTF
__aicore__ inline void GetValueCoreOffsetParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        CalcValueCoreOffsetBSH<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx);
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        CalcValueCoreOffsetTND<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        CalcValueCoreOffsetNTD<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else {
        CalcValueCoreOffsetBNSD<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx);
    }
    SyncKeyCoreOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    CalcPrefixCoreOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo);
}

TEMPLATE_INTF
__aicore__ inline void GetKeyRopeCoreOffsetParam(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t kRopeInnerOffsetSize = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        kRopeInnerOffsetSize = bIdx * constInfo.n2S2DR + runParam.kvLeftPaddingSize * constInfo.n2DR;
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeRope;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            kRopeInnerOffsetSize = (bIdx == 0)? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2DR;
        } else {
            kRopeInnerOffsetSize = bIdx * constInfo.n2S2DR;
        }
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeRope;
    } else {
        uint64_t headStrideKV = 0;
        headStrideKV = constInfo.s2DR;
        kRopeInnerOffsetSize = bIdx * constInfo.n2Size * headStrideKV +
            runParam.kvLeftPaddingSize * constInfo.dSizeRope;
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.n2oIdx * headStrideKV;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    const AttenMaskInfo &attenMaskInfo,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, attenMaskInfo,
        runParam.boIdx, keyGm, actualSeqQlenAddr, actualSeqKvlenAddr);
    GetValueCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqKvlenAddr);
    if constexpr (hasRope) {
        GetKeyRopeCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqKvlenAddr);
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    bool lastBN, int64_t nextGs1Idx)
{
    constexpr int32_t s1BaseSize = static_cast<int32_t>(s1TemplateType);
    int32_t s1LoopTimes;
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
        s1LoopTimes = CeilDiv(runParam.actualS1Size, s1BaseSize);
    } else {
        if (constInfo.isGqa) {
            s1LoopTimes = CeilDiv(runParam.actualS1Size * constInfo.gSize, s1BaseSize);
        } else {
            s1LoopTimes = CeilDiv(runParam.actualS1Size, s1BaseSize);
        }
    }
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.s1LoopTimes = s1LoopTimes;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.s1LoopTimes = nextGs1Idx == 0 ? s1LoopTimes : nextGs1Idx;
    }
    if (constInfo.isGqa && constInfo.gSize > NUM_128 && layout == LayOutTypeEnum::LAYOUT_TND) {
        runParam.s1LoopTimes = s1LoopTimes;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * (uint32_t)s1TemplateType;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
    } else {
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
            runParam.s1RealSize = Min((uint32_t)s1TemplateType, runParam.actualS1Size - cubeSOuterOffset);
        } else {
            if (constInfo.isGqa) {
                runParam.s1RealSize = Min((uint32_t)s1TemplateType, runParam.actualS1Size * constInfo.gSize - cubeSOuterOffset);
            } else {
                runParam.s1RealSize = Min((uint32_t)s1TemplateType, runParam.actualS1Size - cubeSOuterOffset);
            }
        }
    }

    cubeSOuterOffset += (runParam.nextTokensPerBatch < 0) ? -runParam.nextTokensPerBatch : 0;
    if constexpr (useDn) { 
        runParam.s1RealSizeAlign32 = ((runParam.s1RealSize + 31) >> 5 << 5); // 5 and 31 for 32align factor
        runParam.halfS1RealSize = runParam.s1RealSize <= 16 ? runParam.s1RealSize : runParam.s1RealSizeAlign32 / 2;
    } else {
        runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
        if (constInfo.s1Size > 1 && constInfo.isGqa && (layout == LayOutTypeEnum::LAYOUT_BSH ||
            layout == LayOutTypeEnum::LAYOUT_TND)) {
            runParam.halfS1RealSize = (runParam.halfS1RealSize + constInfo.gSize - 1) / constInfo.gSize *
                constInfo.gSize;
        }
    }
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
        runParam.sOuterOffset = cubeSOuterOffset + runParam.firstHalfS1RealSize;
    } else {
        runParam.sOuterOffset = cubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void CalcSeqLenAndOffset(const ConstInfo<isInfer, hasRope>& constInfo,
    int32_t bIdx, __gm__ int64_t* actualSeqQlenAddr, int64_t& actualSeqLen, int64_t& seqOffset)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
        actualSeqLen = (bIdx == 0) ? actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
        seqOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1];
    } else {
        actualSeqLen = constInfo.s1Size;
        seqOffset = bIdx * constInfo.s1Size;
    }
}

TEMPLATE_INTF
__aicore__ inline void CalcAICTensorQAndRopeOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int64_t actualSeqLen)
{
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
        runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
            runParam.cubeSOuterOffset * constInfo.dSize;
        runParam.qRopeNBGOffset = runParam.qRopeBOffset + runParam.n2oIdx * constInfo.gDR * actualSeqLen +
            runParam.cubeSOuterOffset * constInfo.dSizeRope;
    } else {
        if (constInfo.isGqa && constInfo.s1Size > 1) {
            if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
                runParam.tensorQOffset = runParam.qBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                    runParam.n2oIdx * constInfo.gD;
            } else {
                runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
                    runParam.cubeSOuterOffset * constInfo.dSize;
            }
        } else if (constInfo.isGqa) {
            runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
                runParam.cubeSOuterOffset * constInfo.dSize;
            if constexpr (hasRope) {
                runParam.qRopeNBGOffset = runParam.qRopeBOffset + runParam.n2oIdx * constInfo.gDR * actualSeqLen +
                    runParam.cubeSOuterOffset * constInfo.dSizeRope;
            }
        } else {
            if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
                runParam.tensorQOffset = runParam.qBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                    runParam.n2oIdx * constInfo.gD + runParam.goIdx * constInfo.dSize;
                if constexpr (hasRope) {
                    runParam.qRopeNBGOffset = runParam.qRopeBOffset + runParam.cubeSOuterOffset * constInfo.n2GDR +
                        runParam.n2oIdx * constInfo.gDR + runParam.goIdx * constInfo.dSizeRope;
                }
            } else {
                runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gS1D +
                    runParam.goIdx * constInfo.s1D + runParam.cubeSOuterOffset * constInfo.dSize;
                if constexpr (hasRope) {
                    runParam.qRopeNBGOffset = runParam.qRopeBOffset + runParam.n2oIdx * constInfo.gS1DR +
                        runParam.goIdx * constInfo.s1DR + runParam.cubeSOuterOffset * constInfo.dSizeRope;
                }
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void CalcAIVMlaAttentionOutOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int64_t seqOffset, int64_t actualSeqLen)
{
    int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
    runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
        runParam.sOuterOffset * constInfo.dSizeV;
    runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
        runParam.cubeSOuterOffset * constInfo.dSize;
    if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_NBSD)) {
        attentionOutSeqOffset = seqOffset * constInfo.dSizeV;
        int64_t curGIdx = runParam.sOuterOffset / constInfo.s1Size;
        int64_t curS1Idx = runParam.sOuterOffset % constInfo.s1Size;
        runParam.attentionOutOffset = attentionOutSeqOffset +
            curGIdx * constInfo.t1Size * constInfo.dSizeV + curS1Idx * constInfo.dSizeV;
    } else if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_NBSD) ||
        constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_NBSD) ||
        constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::TND_NTD)) {
        attentionOutSeqOffset = seqOffset * constInfo.dSizeV;
        int64_t curGIdx = runParam.goIdx;
        int64_t curS1Idx = runParam.cubeSOuterOffset / (uint32_t)s1TemplateType;
        if (constInfo.gSize == 128) {
            curGIdx = (curS1Idx % 2 == 0) ? curGIdx : (uint32_t)s1TemplateType;
            curS1Idx /= 2;
        } else if (constInfo.gSize <= 32) {
            curS1Idx = runParam.cubeSOuterOffset / constInfo.gSize;
        }
        if (constInfo.subBlockIdx == 1) {
            int64_t firstCurGIdx = curGIdx;
            curGIdx = (firstCurGIdx + runParam.firstHalfS1RealSize) % constInfo.gSize;
            curS1Idx += (firstCurGIdx + runParam.firstHalfS1RealSize) / constInfo.gSize;
        }
        runParam.attentionOutOffset = attentionOutSeqOffset +
            curGIdx * constInfo.t1Size * constInfo.dSizeV + curS1Idx * constInfo.dSizeV;
    }
}

TEMPLATE_INTF
__aicore__ inline void CalcAIVNonMlaAttentionOutOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int64_t seqOffset, int64_t actualSeqLen)
{
    int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
    if (constInfo.isGqa && constInfo.s1Size > 1) {
        runParam.sOuterOffset += (runParam.nextTokensPerBatch < 0) ? runParam.nextTokensPerBatch : 0;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.queryLeftPaddingSize * constInfo.n2GDv +
                runParam.sOuterOffset / constInfo.gSize * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv;
        } else {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
                runParam.sOuterOffset * constInfo.dSizeV;
        }
    } else if (constInfo.isGqa) {
        runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
            runParam.sOuterOffset * constInfo.dSizeV;
    } else {
        if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND) ||
            constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::NTD_TND) ||
            layout == LayOutTypeEnum::LAYOUT_TND ||
            (constInfo.transposeLayout != static_cast<uint32_t>(TransposeLayoutEnum::BSND_BNSD) &&
            constInfo.transposeLayout != static_cast<uint32_t>(TransposeLayoutEnum::BSH_BNSD) &&
            layout == LayOutTypeEnum::LAYOUT_BSH)) {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.queryLeftPaddingSize * constInfo.n2GDv +
                runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                runParam.goIdx * constInfo.dSizeV;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            attentionOutSeqOffset = seqOffset * constInfo.dSizeV;
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.queryLeftPaddingSize * constInfo.dSizeV +
                runParam.n2oIdx * constInfo.t1Size * constInfo.gDv +
                runParam.goIdx * constInfo.t1Size * constInfo.dSizeV +
                runParam.sOuterOffset * constInfo.dSizeV;
        } else {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gS1Dv +
                runParam.goIdx * constInfo.s1Dv + (runParam.sOuterOffset + runParam.queryLeftPaddingSize) *
                constInfo.dSizeV;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void CalcAIVSoftmaxLseOffset(RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, int64_t seqOffset, int64_t actualSeqLen)
{
    int64_t softmaxLseSeqOffset = seqOffset * constInfo.n2G;
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            int64_t nsOffset = runParam.n2oIdx * constInfo.gS1 + runParam.sOuterOffset;
            int64_t sOffset = nsOffset / constInfo.gSize;
            runParam.softmaxLseOffset = softmaxLseSeqOffset + (nsOffset % constInfo.gSize) * constInfo.s1Size + sOffset;
        } else {
            runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gSize * actualSeqLen +
                runParam.sOuterOffset;
        }
    } else {
        if (constInfo.isGqa && constInfo.s1Size > 1) {
            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
                runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.queryLeftPaddingSize * constInfo.n2G +
                    runParam.sOuterOffset / constInfo.gSize * constInfo.n2G + runParam.n2oIdx * constInfo.gSize;
            } else {
                runParam.softmaxLseOffset = softmaxLseSeqOffset +
                    runParam.n2oIdx * constInfo.gSize * actualSeqLen + runParam.sOuterOffset / constInfo.gSize;
            }
        } else if (constInfo.isGqa) {
            runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gSize * actualSeqLen +
                runParam.sOuterOffset;
        } else {
            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
                runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.sOuterOffset * constInfo.n2G +
                    runParam.n2oIdx * constInfo.gSize + runParam.goIdx;
            } else {
                runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gS1 +
                    runParam.goIdx * constInfo.s1Size + runParam.sOuterOffset + runParam.queryLeftPaddingSize;
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    int32_t bIdx, __gm__ int64_t *actualSeqQlenAddr, PseInfo& pseInfo)
{
    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        CalPseShiftCoreOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, runParam.sOuterOffset, pseInfo);
    }

    int64_t actualSeqLen = 0;
    int64_t seqOffset = 0;
    CalcSeqLenAndOffset<TEMPLATE_INTF_ARGS>(constInfo, bIdx, actualSeqQlenAddr, actualSeqLen, seqOffset);

    if ASCEND_IS_AIC {
        CalcAICTensorQAndRopeOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualSeqLen);
    } else {
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
            CalcAIVMlaAttentionOutOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, seqOffset, actualSeqLen);
        } else {
            CalcAIVNonMlaAttentionOutOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, seqOffset, actualSeqLen);
        }
        CalcAIVSoftmaxLseOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, seqOffset, actualSeqLen);
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int64_t *actualSeqQlenAddr, PseInfo& pseInfo)
{
    // 后续的函数依赖 sOuterOffset
    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);

    // 使用转换后的左上角的pretoken nexttoken
    if (runParam.s1RealSize == 0 || (runParam.nextTokensPerBatch < 0 && runParam.sOuterOffset < ((runParam.nextTokensPerBatch * (-1)) /
        runParam.halfS1RealSize * runParam.halfS1RealSize))) {
        return true;
    }

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqQlenAddr, pseInfo);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    __gm__ int64_t *actualSeqQlenAddr)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (runParam.boIdx > 0 && actualSeqQlenAddr[runParam.boIdx] - actualSeqQlenAddr[runParam.boIdx - 1] == 0) {
            return true;
        }
    } else {
        if (!constInfo.isActualLenDimsNull) {
            if (constInfo.actualSeqLenSize == 1) {
                return actualSeqQlenAddr[0] == 0;
            } else {
                return actualSeqQlenAddr[runParam.boIdx] == 0;
            }
        }
    }
    return false;
}

TEMPLATE_INTF
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo)
{
    constexpr int32_t s2BaseSize = static_cast<int32_t>(s2TemplateType);
    if constexpr (isFd) {
        runParam.s2LoopEndIdx = (runParam.s2LineEndIdx + s2BaseSize - 1) / s2BaseSize;
        return false;
    }

    int64_t sInnerFirstToken = 0;
    if constexpr ((hasRope && (dTemplateType == DTemplateType::Aligned576)) && layout != LayOutTypeEnum::LAYOUT_BNSD) {
        sInnerFirstToken = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>((runParam.cubeSOuterOffset - runParam.preTokensPerBatch) / constInfo.gSize,
            0, runParam.actualS2Size);
        runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(CeilDiv(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
            runParam.s1RealSize, constInfo.gSize), 0, runParam.actualS2Size);
    } else {
        if (constInfo.isGqa) {
            sInnerFirstToken = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.s1oIdx - runParam.preTokensPerBatch,
                0, runParam.actualS2Size);
            runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.s1oIdx + runParam.nextTokensPerBatch +
                runParam.s1RealSize, 0, runParam.actualS2Size);
        } else {
            sInnerFirstToken = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch,
                0, runParam.actualS2Size);
            runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
                runParam.s1RealSize, 0, runParam.actualS2Size);
        }
    }
    runParam.s2LoopEndIdx = (runParam.s2LineEndIdx + s2BaseSize - 1) / s2BaseSize - sInnerFirstToken / s2BaseSize;
    if constexpr (enableKVPrefix) {
        sInnerFirstToken =
            ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch, 0,
                                                    runParam.actualS2Size + constInfo.actualKVPrefixSize);
        runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            runParam.cubeSOuterOffset + runParam.nextTokensPerBatch + runParam.s1RealSize, 0,
            runParam.actualS2Size + constInfo.actualKVPrefixSize);
        runParam.s2LoopEndIdx = (constInfo.actualKVPrefixSize + s2BaseSize - 1) / s2BaseSize +
                                (runParam.s2LineEndIdx - constInfo.actualKVPrefixSize + s2BaseSize - 1) / s2BaseSize -
                                sInnerFirstToken / s2BaseSize;
    }
    if (runParam.s2LoopEndIdx <= 0) {
        return true;
    }
    if constexpr (hasAtten) {
        runParam.s2LineStartIdx = sInnerFirstToken / s2BaseSize * s2BaseSize;
    } else {
        runParam.s2LineStartIdx = sInnerFirstToken;
    }
    return false;
}

TEMPLATE_INTF
__aicore__ inline void ComputeOffsetAIV(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer>& runInfo)
{
    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        if constexpr (enableKVPrefix) {
            if (sInnerLoopIdx < constInfo.prefixLoopCount) {
                runInfo.pseShiftOffset = ComputePseShiftOffset<TEMPLATE_INTF_ARGS>(
                    runParam, sInnerLoopIdx * static_cast<int32_t>(s2TemplateType));
            } else {
                runInfo.pseShiftOffset = ComputePseShiftOffset<TEMPLATE_INTF_ARGS>(
                    runParam, (sInnerLoopIdx - constInfo.prefixLoopCount) *
                    static_cast<int32_t>(s2TemplateType) + constInfo.actualKVPrefixSize);
            }
        } else {
            runInfo.pseShiftOffset = ComputePseShiftOffset<TEMPLATE_INTF_ARGS>(
                runParam, sInnerLoopIdx * static_cast<int32_t>(s2TemplateType));
        }
        if constexpr (isFd) {
            runInfo.pseShiftOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize;
        }
    }

    if (!constInfo.isGqa) {
        runInfo.vecCoreOffset = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    } else {
        runInfo.vecCoreOffset = 0;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeOffsetPrefixAIC(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer>& runInfo)
{
    const bool inPrefixLoop =
        (sInnerLoopIdx * static_cast<int32_t>(s2TemplateType) < constInfo.actualKVPrefixSize);
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        if (inPrefixLoop) {
            runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            runInfo.valueOffset = 0;
            runInfo.keyOffset = 0;
        } else {
            runInfo.prefixOffset = 0;
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            runInfo.keyOffset = runInfo.valueOffset;
        }
    } else {
        if (inPrefixLoop) {
            runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            runInfo.valueOffset = 0;
            runInfo.keyOffset = 0;
        } else {
            runInfo.prefixOffset = 0;
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            runInfo.keyOffset = runInfo.valueOffset;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeOffsetNonPrefixAIC(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer>& runInfo)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
        if constexpr (isFd) {
            runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.n2D;
        }
        if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2D;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        if constexpr (hasRope) {
            runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
        if constexpr (isFd) {
            runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSizeV;
        }
        if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        if constexpr (hasRope) {
            runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
        }
    } else {
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
        if constexpr (isFd) {
            runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSize;
        }
        if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseD;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        if constexpr (hasRope) {
            runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeOffset(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope> &constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer> &runInfo)
{
    if ASCEND_IS_AIV {
        ComputeOffsetAIV<TEMPLATE_INTF_ARGS>(runParam, constInfo, sInnerLoopIdx, runInfo);
    } else {
        if constexpr (enableKVPrefix) {
            ComputeOffsetPrefixAIC<TEMPLATE_INTF_ARGS>(runParam, constInfo, sInnerLoopIdx, runInfo);
        } else {
            ComputeOffsetNonPrefixAIC<TEMPLATE_INTF_ARGS>(runParam, constInfo, sInnerLoopIdx, runInfo);
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeOffsetForAntiquant(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope> &constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer> &runInfo)
{
    if constexpr (enableKVPrefix) {
        const bool inPrefixLoop = sInnerLoopIdx < constInfo.prefixLoopCount;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            if (inPrefixLoop) {
                runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
                runInfo.valueOffset = 0;
                runInfo.keyOffset = 0;
            } else {
                runInfo.prefixOffset = 0;
                runInfo.valueOffset = runParam.valueCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * constInfo.s2BaseN2Dv;
                runInfo.keyOffset = runInfo.valueOffset;
            }
        } else {
            if (inPrefixLoop) {
                runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
                runInfo.valueOffset = 0;
                runInfo.keyOffset = 0;
            } else {
                runInfo.prefixOffset = 0;
                runInfo.valueOffset = runParam.valueCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * constInfo.s2BaseDv;
                runInfo.keyOffset = runInfo.valueOffset;
            }
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.n2D;
            }
            runInfo.keyOffset = runInfo.valueOffset;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
            }
        } else {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSize;
            }
            runInfo.keyOffset = runInfo.valueOffset;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr<isInfer>& runParam, RunInfo<isInfer> &runInfo)
{
    runInfo.keyOffset = runParam.keyOffset;
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatch = runParam.preTokensPerBatch;
    runInfo.nextTokensPerBatch = runParam.nextTokensPerBatch;
    runInfo.b1SSAttenMaskOffset = runParam.b1SSAttenMaskOffset;
    runInfo.b1SSOffset = runParam.b1SSOffset;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
    runInfo.s2InCurrentBatch = runParam.s2InCurrentBatch;
    runInfo.queryLeftPaddingSize = runParam.queryLeftPaddingSize;
    runInfo.kvLeftPaddingSize = runParam.kvLeftPaddingSize;
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) { // IFA MLA
        runInfo.nextTokensOfMlaPerBatch = runParam.nextTokensOfMlaPerBatch;
        runInfo.preTokensOfMlaPerBatch = runParam.preTokensOfMlaPerBatch;
        runInfo.actualSeqLengthOfMlaPerBatch = runParam.actualSeqLengthOfMlaPerBatch;
    }
}

TEMPLATE_INTF
__aicore__ inline void IterateAllPreProcess(RunInfo<isInfer>& runInfo, const ConstInfo<isInfer, hasRope> &constInfo,
    GlobalTensor<INPUT_T>& keyValueGm, GlobalTensor<INPUT_T>& tempKeyValueGm)
{
    if constexpr (isInfer) {
        if (constInfo.isKvContinuous != 0) {
            return;
        }
        ListTensorDesc keyValueListTensorDesc((__gm__ void*)keyValueGm.GetPhyAddr());
        __gm__ uint8_t* tempKeyValueGmPtr =
            (__gm__ uint8_t*)keyValueListTensorDesc.GetDataPtr<__gm__ uint8_t>(runInfo.boIdx);
        tempKeyValueGm.SetGlobalBuffer((__gm__ INPUT_T*)tempKeyValueGmPtr);
    }
}

#endif  // INFER_FLASH_ATTENTION_KVCACHE_H