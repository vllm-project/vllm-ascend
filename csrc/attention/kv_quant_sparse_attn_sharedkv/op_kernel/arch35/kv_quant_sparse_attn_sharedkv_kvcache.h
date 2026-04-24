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
 * \file kv_quant_sparse_attn_sharedkv_kvcache.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_KVCACHE_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr& runParam, const ConstInfo &constInfo,
    __gm__ int32_t *cuSeqlensQAddr, __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t * actualSeqKvlenAddr)
{
    int32_t actualS1Size = 0;
    int32_t actualS2Size = 0;
    int32_t actualSeqMin = 1;
    int32_t actualSeqKVMin = 1;
    int32_t sIdx = runParam.boIdx;
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        // actual seq length first
        actualS1Size = (actualSeqQlenAddr == nullptr) ? (cuSeqlensQAddr[sIdx + 1] - cuSeqlensQAddr[sIdx]) :
            actualSeqQlenAddr[sIdx];
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size :
            actualSeqQlenAddr[sIdx];
    }

    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = constInfo.s2Size;
    } else {
        if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
            actualS2Size = actualSeqKvlenAddr[sIdx];
            if ((sIdx > 0) && (!isPa)) {
                actualS2Size -= actualSeqKvlenAddr[sIdx - 1];
            }
        } else {
            actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ? 
                actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
        }
    }

    runParam.actualS1Size = actualS1Size;
    runParam.actualS2Size = actualS2Size;
    runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;
    runParam.preTokensPerBatch = -(runParam.actualS2Size - runParam.actualS1Size - constInfo.oriWinLeft);
    runParam.preTokensPerBatch = Min(runParam.preTokensPerBatch, runParam.actualS1Size);
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& runParam, const ConstInfo &constInfo,
    __gm__ int32_t *cuSeqlensQAddr, __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, cuSeqlensQAddr, actualSeqQlenAddr, actualSeqKvlenAddr);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo, bool lastBN, 
    int64_t nextGs1Idx, int64_t gS1StartIdx)
{
    runParam.qSNumInOneBlock = constInfo.s1BaseSize / constInfo.gSize; // 不切G轴, 计算每个基本快可以拷贝多少行s
    runParam.gs1LoopStartIdx = gS1StartIdx;
    if (runParam.nextTokensPerBatch < 0) {
        int64_t gs1LoopStartIdx = runParam.nextTokensPerBatch * (-1) / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock;
        if (gs1LoopStartIdx > gS1StartIdx) {
            runParam.gs1LoopStartIdx = gs1LoopStartIdx;
        }
    }

    int32_t gs1LoopEndIdx = 0;
    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        gs1LoopEndIdx = runParam.actualS1Size; // 对于SCFA, 不切G轴, 每次拷贝一行的topk，只算一行的qs
    } else { // SWA/CFA
        // 不需要取topk, 每次计算gSize行, 循环qs次
        gs1LoopEndIdx = (runParam.actualS1Size + runParam.qSNumInOneBlock - 1) / runParam.qSNumInOneBlock;
    }
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.gs1LoopEndIdx = gs1LoopEndIdx;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.gs1LoopEndIdx = nextGs1Idx == 0 ? gs1LoopEndIdx : nextGs1Idx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * runParam.qSNumInOneBlock;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
        runParam.mRealSize = 0;
    } else {
        runParam.s1RealSize = Min(runParam.qSNumInOneBlock, runParam.actualS1Size - cubeSOuterOffset);
        runParam.mRealSize = runParam.s1RealSize * constInfo.gSize;
    }

    runParam.cubeMOuterOffset = cubeSOuterOffset * constInfo.gSize;
    runParam.halfMRealSize = (runParam.mRealSize + 1) >> 1;
    runParam.firstHalfMRealSize = runParam.halfMRealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfMRealSize = runParam.mRealSize - runParam.halfMRealSize;
        runParam.mOuterOffset = runParam.cubeMOuterOffset + runParam.firstHalfMRealSize;
    } else {
        runParam.mOuterOffset = runParam.cubeMOuterOffset;
    }

    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
        runParam.sOuterOffset = cubeSOuterOffset + runParam.halfMRealSize / constInfo.gSize;
    } else {
        runParam.sOuterOffset = cubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr& runParam, const ConstInfo &constInfo,
    int32_t sIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
            seqOffset = cuSeqlensQAddr[sIdx];
        } else {
            seqOffset = sIdx * constInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
        if constexpr (LAYOUT_T == SAS_LAYOUT::BSND || LAYOUT_T == SAS_LAYOUT::TND) {
            runParam.attentionOutOffset = attentionOutSeqOffset +
                runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                runParam.goIdx * constInfo.dSizeV;
        }
        if (constInfo.subBlockIdx == 1) {
            runParam.attentionOutOffset += runParam.halfMRealSize * constInfo.dSizeV;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if (runParam.nextTokensPerBatch < 0) {
        if (runParam.s1oIdx < (runParam.nextTokensPerBatch * (-1)) / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock) {
            return true;
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, cuSeqlensQAddr);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr& runParam, __gm__ int32_t *cuSeqlensQAddr) 
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (runParam.boIdx > 0 && cuSeqlensQAddr[runParam.boIdx + 1] - cuSeqlensQAddr[runParam.boIdx] == 0) {
            return true;
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
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo)
{
    if (runParam.actualS2Size == 0) {
        runParam.oriKvLoopEndIdx = 0;
        runParam.cmpKvLoopEndIdx = 0;
        runParam.s2LoopEndIdx = 0;
        return true;
    }
    uint32_t s2BaseSize = constInfo.s2BaseSize;

    runParam.s2LineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch,
        0, runParam.actualS2Size);
    runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
        runParam.s1RealSize, 0, runParam.actualS2Size);
    runParam.oriKvLoopEndIdx = (runParam.s2LineEndIdx - runParam.s2LineStartIdx + s2BaseSize - 1) / s2BaseSize;
    if constexpr (TEMPLATE_MODE == SASTemplateMode::SWA_TEMPLATE_MODE) {
        runParam.cmpKvLoopEndIdx = 0;
        runParam.s2CmpLineEndIdx = 0;
    } else if constexpr (TEMPLATE_MODE == SASTemplateMode::CFA_TEMPLATE_MODE) {
        runParam.s2CmpLineEndIdx = runParam.s2LineEndIdx / constInfo.cmpRatio;
        runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
    } else { // SCFA_TEMPLATE_MODE
        runParam.s2CmpLineEndIdx = Min(runParam.s2LineEndIdx / constInfo.cmpRatio, constInfo.sparseBlockCount); // 当前LI输出的block size只可能是1
        runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
    }
    runParam.s2LoopEndIdx = runParam.oriKvLoopEndIdx + runParam.cmpKvLoopEndIdx;
    return false;
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr& runParam, RunInfo &runInfo)
{
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatch = runParam.preTokensPerBatch;
    runInfo.nextTokensPerBatch = runParam.nextTokensPerBatch;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
    runInfo.qSNumInOneBlock = runParam.qSNumInOneBlock;
    runInfo.oriKvLoopEndIdx = runParam.oriKvLoopEndIdx;
    runInfo.cmpKvLoopEndIdx = runParam.cmpKvLoopEndIdx;
}

#endif  // KV_QUANT_SPARSE_ATTN_SHAREDKV_KVCACHE_H