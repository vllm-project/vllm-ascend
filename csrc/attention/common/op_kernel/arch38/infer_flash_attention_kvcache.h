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

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "infer_flash_attention_comm.h"
#include "infer_flash_attention_sparse.h"

using namespace matmul;

TEMPLATE_INTF
__aicore__ inline void InitQueryLeftPaddingSize(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope>& constInfo, int64_t& actualS1Size)
{
    if (!constInfo.isQHasLeftPadding) {
        runParam.queryLeftPaddingSize = 0;
    } else {
        int64_t qLeftPaddingSize = constInfo.s1Size - actualS1Size - constInfo.queryRightPaddingSize;
        runParam.queryLeftPaddingSize = qLeftPaddingSize > 0 ? qLeftPaddingSize : 0;
        if (qLeftPaddingSize < 0) {
            actualS1Size = 0;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitKVLeftPaddingSize(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope>& constInfo, int64_t& actualS2Size)
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

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr<isInfer>& runParam, 
    const ConstInfo<isInfer, hasRope> &constInfo, const AttenMaskInfo &attenMaskInfo, int32_t sIdx,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t * actualSeqKvlenAddr)
{
    // TensorList场景获取不同batch的KvSeq长度
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void*)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, sIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }

    int64_t actualS1Size = 0;
    int64_t actualS2Size = 0;
    int64_t actualSeqMin = 1;
    int64_t actualSeqKVMin = 1;
    if (constInfo.isActualLenDimsNull) {
        actualS1Size = constInfo.s1Size;
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) { // IFA MLA
            actualS1Size = constInfo.gS1;
            runParam.actualSeqLengthOfMlaPerBatch = constInfo.s1Size;
        }
        if (constInfo.isGqa) {
            actualS1Size = constInfo.gS1;
        }
    } else {
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576) && layout == LayOutTypeEnum::LAYOUT_BSH) {
            runParam.actualSeqLengthOfMlaPerBatch = ((constInfo.actualSeqLenSize == actualSeqMin) ?
                actualSeqQlenAddr[0] : actualSeqQlenAddr[sIdx]);
            actualS1Size = runParam.actualSeqLengthOfMlaPerBatch * constInfo.gSize;
        } else if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576) && layout == LayOutTypeEnum::LAYOUT_TND) {
            runParam.actualSeqLengthOfMlaPerBatch = ((sIdx == 0) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1]);
            actualS1Size = runParam.actualSeqLengthOfMlaPerBatch * constInfo.gSize;
        } else if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576) && layout == LayOutTypeEnum::LAYOUT_BNSD) {
            actualS1Size = constInfo.gS1;
            runParam.actualSeqLengthOfMlaPerBatch = (constInfo.actualSeqLenSize == actualSeqMin) ?
                actualSeqQlenAddr[0] : actualSeqQlenAddr[sIdx];
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            actualS1Size = (sIdx == 0) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1];
            if (constInfo.isGqa) {
                actualS1Size *= constInfo.gSize;
            }
        } else {
            actualS1Size = (constInfo.actualSeqLenSize == actualSeqMin) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[sIdx];
        }
    }
    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = (constInfo.isKvContinuous == 1) ? constInfo.s2Size :
            runParam.s2InCurrentBatch;
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            actualS2Size = actualSeqKvlenAddr[sIdx];
            if ((sIdx > 0) && (!isPa)) {
                actualS2Size -= actualSeqKvlenAddr[sIdx - 1];
            }
        } else {
            actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ? 
                actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
        }
    }

    InitQueryLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS1Size);
    InitKVLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS2Size);

    runParam.actualS1Size = actualS1Size;
    runParam.actualS2Size = actualS2Size;
    GetSparseParam<TEMPLATE_INTF_ARGS>(constInfo, attenMaskInfo, runParam);

    runParam.actualS1Size = 
        (runParam.actualS1Size > runParam.actualS2Size + runParam.preTokensPerBatch) ?
        runParam.actualS2Size + runParam.preTokensPerBatch : runParam.actualS1Size;

    // 计算S1的尾块大小，非对齐
    runParam.actualS1Size = (runParam.nextTokensPerBatch >= 0) ? runParam.actualS1Size :
        (runParam.actualS1Size + runParam.nextTokensPerBatch);

    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        runParam.qBOffset = sIdx * constInfo.s1Size * constInfo.n2GD + runParam.queryLeftPaddingSize * constInfo.n2GD;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runParam.qBOffset = (sIdx == 0) ? 0 : actualSeqQlenAddr[sIdx - 1] * constInfo.n2GD;
    } else {
        runParam.qBOffset = sIdx * constInfo.s1Size * constInfo.n2GD + runParam.queryLeftPaddingSize * constInfo.dSize;
    }

    // 推理的TND场景的mask和pse都是padding过的
    runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
    // 推理的mask的sequence length可能大于qk的sequence length
    runParam.b1SSAttenMaskOffset = runParam.boIdx * (uint64_t)attenMaskInfo.attenMaskS1Size * (uint64_t)attenMaskInfo.attenMaskS2Size;
    if constexpr (hasRope) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            runParam.qRopeBOffset = sIdx * constInfo.s1Size * constInfo.n2GDR +
                runParam.queryLeftPaddingSize * constInfo.n2GDR;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            runParam.qRopeBOffset = (sIdx == 0) ? 0 : actualSeqQlenAddr[sIdx - 1] * constInfo.n2GDR;
        } else {
            runParam.qRopeBOffset = sIdx * constInfo.s1Size * constInfo.n2GDR +
                runParam.queryLeftPaddingSize * constInfo.dSizeRope;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void GetKeyCoreOffsetParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo, 
    int32_t sIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t keyInnerOffsetSize = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        if (constInfo.isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            keyInnerOffsetSize = sIdx * constInfo.n2S2D + runParam.kvLeftPaddingSize * constInfo.n2D;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;
    // 遗留问题：TND PA
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            keyInnerOffsetSize = (sIdx == 0) ? 0 : actualSeqKvlenAddr[sIdx - 1] * constInfo.n2D;
        } else {
            keyInnerOffsetSize = sIdx * constInfo.n2S2D;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;
    } else {
        uint64_t headStrideK = 0;
        if (constInfo.isKvContinuous == 1) {
            headStrideK = constInfo.s2D;
            keyInnerOffsetSize = sIdx * constInfo.n2Size * headStrideK +
                runParam.kvLeftPaddingSize * constInfo.dSize;
        } else {
            headStrideK = constInfo.dSize * runParam.s2InCurrentBatch;
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStrideK;
    }
}

TEMPLATE_INTF
__aicore__ inline void GetValueCoreOffsetParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo, 
    int32_t sIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        if (constInfo.isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            valueInnerOffsetSize = sIdx * constInfo.n2S2Dv + runParam.kvLeftPaddingSize * constInfo.n2Dv;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            valueInnerOffsetSize = 0;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            valueInnerOffsetSize = (sIdx == 0) ? 0 : actualSeqKvlenAddr[sIdx - 1] * constInfo.n2Dv;
        } else {
            valueInnerOffsetSize = sIdx * constInfo.n2S2Dv;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
    } else {
        uint64_t headStrideV = 0;
        if (constInfo.isKvContinuous == 1) {
            headStrideV = constInfo.s2Dv;
            valueInnerOffsetSize = sIdx * constInfo.n2Size * headStrideV +
                runParam.kvLeftPaddingSize * constInfo.dSizeV;
        } else {
            headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
            valueInnerOffsetSize = 0;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
    }
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}

TEMPLATE_INTF
__aicore__ inline void GetKeyRopeCoreOffsetParam(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo, 
    int32_t sIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t kRopeInnerOffsetSize = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        kRopeInnerOffsetSize = sIdx * constInfo.n2S2DR + runParam.kvLeftPaddingSize * constInfo.n2DR;
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeRope;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            kRopeInnerOffsetSize = (sIdx == 0)? 0 : actualSeqKvlenAddr[sIdx - 1] * constInfo.n2DR;
        } else {
            kRopeInnerOffsetSize = sIdx * constInfo.n2S2DR;
        }
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeRope;
    } else {
        uint64_t headStrideKV = 0;
        headStrideKV = constInfo.s2DR;
        kRopeInnerOffsetSize = sIdx * constInfo.n2Size * headStrideKV +
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
__aicore__ inline void ComputeS1LoopInfo(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo, bool lastBN, 
    int64_t nextGs1Idx)
{
    constexpr int32_t s1BaseSize = static_cast<int32_t>(s1TemplateType);
    int32_t s1LoopTimes = CeilDiv(runParam.actualS1Size, s1BaseSize);
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.s1LoopTimes = s1LoopTimes;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.s1LoopTimes = nextGs1Idx == 0 ? s1LoopTimes : nextGs1Idx;
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
        runParam.s1RealSize = Min((uint32_t)s1TemplateType, runParam.actualS1Size - cubeSOuterOffset);
    }

    cubeSOuterOffset += (runParam.nextTokensPerBatch < 0) ? -runParam.nextTokensPerBatch : 0;
    if constexpr (useDn) { 
        runParam.s1RealSizeAlign32 = ((runParam.s1RealSize + 31) >> 5 << 5); // 5 and 31 for 32align factor
        runParam.halfS1RealSize = runParam.s1RealSize <= 16 ? runParam.s1RealSize : runParam.s1RealSizeAlign32; // 16 for 32align factor
    } else {
        runParam.halfS1RealSize = runParam.s1RealSize;
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
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    int32_t sIdx, __gm__ int64_t *actualSeqQlenAddr, PseInfo& pseInfo)
{
    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        CalPseShiftCoreOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, sIdx, runParam.sOuterOffset, pseInfo);
    }

    int64_t actualSeqLen = 0;
    int64_t seqOffset = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        actualSeqLen = (sIdx == 0) ? actualSeqQlenAddr[0] : actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1];
        seqOffset = (sIdx == 0) ? 0 : actualSeqQlenAddr[sIdx - 1];
    } else {
        actualSeqLen = constInfo.s1Size;
        seqOffset = sIdx * constInfo.s1Size;
    }
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) { // IFA MLA
        runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
            runParam.cubeSOuterOffset * constInfo.dSize; // IFA MLA场景, BSH与BNSD一致
        runParam.qRopeNBGOffset = runParam.qRopeBOffset + runParam.n2oIdx * constInfo.gDR * actualSeqLen +
            runParam.cubeSOuterOffset * constInfo.dSizeRope;
    } else {
        if (constInfo.isGqa) {
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
    int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) { // IFA MLA
        runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
            runParam.sOuterOffset * constInfo.dSizeV;
    } else {
        if (constInfo.isGqa) {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
                runParam.sOuterOffset * constInfo.dSizeV;
        } else {
            if (constInfo.isBSNDOut == 1 || layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
                runParam.attentionOutOffset = attentionOutSeqOffset + runParam.queryLeftPaddingSize * constInfo.n2GDv +
                    runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                    runParam.goIdx * constInfo.dSizeV;
            } else {
                runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gS1Dv +
                    runParam.goIdx * constInfo.s1Dv + (runParam.sOuterOffset + runParam.queryLeftPaddingSize) *
                    constInfo.dSizeV;
            }
        }
    }

    int64_t softmaxLseSeqOffset = seqOffset * constInfo.n2G;
    if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) { // IFA MLA
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            int64_t nsOffset = runParam.n2oIdx * constInfo.gS1 + runParam.sOuterOffset;
            int64_t sOffset = nsOffset / constInfo.gSize;
            runParam.softmaxLseOffset = softmaxLseSeqOffset + (nsOffset % constInfo.gSize) * constInfo.s1Size + sOffset;
        } else {
            runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gSize * actualSeqLen +
                runParam.sOuterOffset;
        }
    } else {
        if (constInfo.isGqa) {
            runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gSize * actualSeqLen +
                runParam.sOuterOffset;
        } else {
            if (layout == LayOutTypeEnum::LAYOUT_TND) {
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
__aicore__ inline bool ComputeParamS1(RunParamStr<isInfer>& runParam, const ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int64_t *actualSeqQlenAddr, PseInfo& pseInfo)
{
    // 后续的函数依赖 sOuterOffset
    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);

    // 使用转换后的左上角的pretoken nexttoken
    if (runParam.nextTokensPerBatch < 0 && runParam.sOuterOffset < ((runParam.nextTokensPerBatch * (-1)) /
        runParam.halfS1RealSize * runParam.halfS1RealSize)) {
        return true;
    }

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqQlenAddr, pseInfo);
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

    int64_t sInnerFirstToken = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch,
        0, runParam.actualS2Size);
    runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
        runParam.s1RealSize, 0, runParam.actualS2Size);

    runParam.s2LoopEndIdx = (runParam.s2LineEndIdx + s2BaseSize - 1) / s2BaseSize - sInnerFirstToken /s2BaseSize;

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
__aicore__ inline void ComputeOffset(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope> &constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer> &runInfo)
{
    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        runInfo.pseShiftOffset = ComputePseShiftOffset<TEMPLATE_INTF_ARGS>(runParam, sInnerLoopIdx * static_cast<int32_t>(s2TemplateType));
        if constexpr (isFd) {
            runInfo.pseShiftOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize;
        }
    }

    if (!constInfo.isGqa) {
        runInfo.vecCoreOffset = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    } else {
        runInfo.vecCoreOffset = 0;
    }
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
__aicore__ inline void ComputeOffsetForAntiquant(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope> &constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer> &runInfo)
{
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