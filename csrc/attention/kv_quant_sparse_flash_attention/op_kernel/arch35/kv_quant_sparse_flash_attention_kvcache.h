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
 * \file kv_quant_sparse_flash_attention_kvcache.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_KVCACHE_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kv_quant_sparse_flash_attention_common_arch35.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
static constexpr uint32_t sparseModeThree = 3;
static constexpr uint32_t sparseModeZero = 0;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr& runParam, const ConstInfo &constInfo,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t * actualSeqKvlenAddr)
{
    int32_t qsfaActualS1Size = 0;
    int32_t qsfaActualS2Size = 0;
    int32_t actualSeqMin = 1;
    int32_t actualSeqKVMin = 1;
    int32_t sIdx = runParam.boIdx;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        // actual seq length first
        if (actualSeqQlenAddr != nullptr) {
            qsfaActualS1Size = (sIdx == 0) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1];
        } else {
            qsfaActualS1Size = constInfo.s1Size;
        }
    } else {
        qsfaActualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size :
            actualSeqQlenAddr[sIdx];
    }

    if (constInfo.isActualLenDimsKVNull) {
        qsfaActualS2Size = constInfo.s2Size;
    } else {
        if constexpr (isPa) {
            if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
                qsfaActualS2Size = actualSeqKvlenAddr[sIdx];
            } else {
                qsfaActualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        } else {
            if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
                qsfaActualS2Size = (sIdx == 0) ? actualSeqKvlenAddr[0] :
                    actualSeqKvlenAddr[sIdx] - actualSeqKvlenAddr[sIdx - 1];
            } else {
                qsfaActualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        }
    }

    runParam.actualS1Size = qsfaActualS1Size;
    runParam.actualS2Size = qsfaActualS2Size;
    runParam.preTokensPerBatch = runParam.actualS1Size;
    if (constInfo.sparseMode == sparseModeZero) {
        runParam.nextTokensPerBatch = MAX_PRE_NEXT_TOKENS;
    } else {
        runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& runParam,
    const ConstInfo &constInfo, __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualSeqQlenAddr, actualSeqKvlenAddr);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo,
    bool lastBN, int64_t nextGs1Idx, int64_t gS1StartIdx)
{
    runParam.gs1LoopStartIdx = gS1StartIdx;
    runParam.qSNumInOneBlock = 1; // qsfa 不切G轴, 计算每个基本块可以拷贝多少行s

    if (runParam.nextTokensPerBatch < 0) {
        uint64_t invalidTokenCount = static_cast<uint64_t>(-(runParam.nextTokensPerBatch + 1)) + 1ULL;
        int64_t gs1LoopStartIdx =
            invalidTokenCount / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock;
        if (gs1LoopStartIdx > gS1StartIdx) {
            runParam.gs1LoopStartIdx = gs1LoopStartIdx;
        }
    }

    int32_t qsfaGs1LoopEndIdx = runParam.actualS1Size; // qsfa 不切G轴, 每次拷贝一行的topk，只算一行的qs

    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.gs1LoopEndIdx = qsfaGs1LoopEndIdx;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.gs1LoopEndIdx = nextGs1Idx == 0 ? qsfaGs1LoopEndIdx : nextGs1Idx;
    }

    if (runParam.gs1LoopStartIdx > runParam.gs1LoopEndIdx) {
        runParam.gs1LoopStartIdx = runParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t qsfaCubeSOuterOffset = sOuterLoopIdx * runParam.qSNumInOneBlock;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
        runParam.mRealSize = 0;
    } else {
        runParam.s1RealSize = Min(runParam.qSNumInOneBlock, runParam.actualS1Size - qsfaCubeSOuterOffset);
        runParam.mRealSize = runParam.s1RealSize * constInfo.gSize;
        if constexpr (IS_SPLIT_G) {
            runParam.mRealSize = runParam.mRealSize >> 1;
        }
    }

    runParam.cubeMOuterOffset = qsfaCubeSOuterOffset * constInfo.gSize;
    runParam.halfMRealSize = (runParam.mRealSize + 1) >> 1;
    runParam.firstHalfMRealSize = runParam.halfMRealSize;
    if (constInfo.subBlockIdx == 0) {
        runParam.mOuterOffset = runParam.cubeMOuterOffset;
    } else {
        runParam.halfMRealSize = runParam.mRealSize - runParam.halfMRealSize;
        runParam.mOuterOffset = runParam.cubeMOuterOffset + runParam.firstHalfMRealSize;
    }
    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;

    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
        runParam.sOuterOffset = qsfaCubeSOuterOffset + runParam.halfMRealSize / constInfo.gSize;
    } else {
        runParam.sOuterOffset = qsfaCubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = qsfaCubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr& runParam, const ConstInfo &constInfo,
    int32_t sIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if ASCEND_IS_AIV {
        int64_t qsfaSeqOffset = 0;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            qsfaSeqOffset = sIdx == 0 ? 0 : cuSeqlensQAddr[sIdx - 1];
        } else {
            qsfaSeqOffset = sIdx * constInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = qsfaSeqOffset * constInfo.n2GDv;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND || LAYOUT_T == QSFA_LAYOUT::TND) {
            runParam.attentionOutOffset = attentionOutSeqOffset +
                runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                runParam.goIdx * constInfo.dSizeV;
        }
        if (constInfo.subBlockIdx == 1) {
            runParam.attentionOutOffset += runParam.firstHalfMRealSize * constInfo.dSizeV;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if (runParam.nextTokensPerBatch < 0) {
        uint64_t invalidTokenCount = static_cast<uint64_t>(-(runParam.nextTokensPerBatch + 1)) + 1ULL;
        if (runParam.s1oIdx <
            invalidTokenCount / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock) {
            return true;
        }
    }
    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);
    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo,
        runParam.boIdx, cuSeqlensQAddr);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr& runParam, __gm__ int32_t *cuSeqlensQAddr)
{
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (runParam.boIdx > 0 && ((runParam.boIdx == 0 && cuSeqlensQAddr[runParam.boIdx] == 0) || (cuSeqlensQAddr[runParam.boIdx] - cuSeqlensQAddr[runParam.boIdx - 1] == 0))) {
            return true;
        }
    }
    return false;
}

TEMPLATE_INTF
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t qsfaSInnerToken, int64_t minValue, int64_t maxValue)
{
    qsfaSInnerToken = qsfaSInnerToken > minValue ? qsfaSInnerToken : minValue;
    qsfaSInnerToken = qsfaSInnerToken < maxValue ? qsfaSInnerToken : maxValue;
    return qsfaSInnerToken;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo)
{
    if (runParam.actualS2Size == 0) {
        runParam.kvLoopEndIdx = 0;
        runParam.s2LoopEndIdx = 0;
        return true;
    }
    uint32_t qsfaS2BaseSize = constInfo.s2BaseSize;

    if (constInfo.sparseMode == sparseModeZero) {
        runParam.s2LineStartIdx = 0;
        runParam.s2LineEndIdx = Min(runParam.actualS2Size, constInfo.sparseBlockCount);
    } else if (constInfo.sparseMode == sparseModeThree) {
        runParam.s2LineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch,
            0, runParam.actualS2Size);
        runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
            runParam.s1RealSize, 0, runParam.actualS2Size);
        runParam.s2LineEndIdx = Min(runParam.s2LineEndIdx, constInfo.sparseBlockCount); // 当前LI输出的block size只可能是1
    }

    runParam.kvLoopEndIdx = (runParam.s2LineEndIdx + qsfaS2BaseSize - 1) / qsfaS2BaseSize;
    runParam.s2LoopEndIdx = runParam.kvLoopEndIdx;
    return false;
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr& runParam, RunInfo &runInfo)
{
    runInfo.boIdx = runParam.boIdx;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.preTokensPerBatch = runParam.preTokensPerBatch;
    runInfo.nextTokensPerBatch = runParam.nextTokensPerBatch;
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
    runInfo.qSNumInOneBlock = runParam.qSNumInOneBlock;
    runInfo.kvLoopEndIdx = runParam.kvLoopEndIdx;
}

#endif  // KV_QUANT_SPARSE_FLASH_ATTENTION_KVCACHE_H
