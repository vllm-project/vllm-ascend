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
 * \file sparse_attn_sharedkv_kvcache.h
 * \brief
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_KVCACHE_ARCH35_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_KVCACHE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "fused_sparse_attention_overlap_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;

static constexpr uint32_t sparseModeZero = 0;
static constexpr uint32_t sparseModeThree = 3;

TEMPLATE_INTF
__aicore__ inline void CalculateQueryOffset(RunParamStr& runParam,
    const ConstInfo &constInfo, int32_t bIdx,
    __gm__ int32_t* actualSeqQlenAddr)
{
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
        runParam.qBOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1] * constInfo.gSize * 512;
        runParam.qRopeBOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1] * constInfo.gSize * 64;
    }
}

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr& runParam, const ConstInfo &constInfo,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t * actualSeqKvlenAddr)
{
    int32_t actualS1Size = 0;
    int32_t actualS2Size = 0;
    int32_t actualSeqMin = 1;
    int32_t actualSeqKVMin = 1;
    int32_t sIdx = runParam.boIdx;
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
        // actual seq length first
        if (actualSeqQlenAddr != nullptr) {
            actualS1Size = (sIdx == 0) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1];
        } else {
            actualS1Size = constInfo.s1Size;
        }
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size :
            actualSeqQlenAddr[sIdx];
    }

    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = constInfo.s2Size;
    } else {
        if constexpr (isPa) {
            if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
                actualS2Size = actualSeqKvlenAddr[sIdx];
            } else {
                actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        } else {
            if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
                actualS2Size = (sIdx == 0) ? actualSeqKvlenAddr[0] :
                    actualSeqKvlenAddr[sIdx] - actualSeqKvlenAddr[sIdx - 1];
            } else {
                actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        }
    }

    runParam.actualS1Size = actualS1Size;
    runParam.actualS2Size = actualS2Size;
    if (constInfo.sparseMode == sparseModeZero) {
        runParam.nextTokensPerBatch = MAX_PRE_NEXT_TOKENS;
    } else {
        runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;
    }
    runParam.preTokensPerBatch = runParam.actualS1Size;
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& runParam, const ConstInfo &constInfo,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualSeqQlenAddr, actualSeqKvlenAddr);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo, bool lastBN,
    int64_t nextGs1Idx, int64_t gS1StartIdx)
{
    runParam.qSNumInOneBlock = 1; // Do not split the G axis; calculate how many S rows each base block can copy
    runParam.gs1LoopStartIdx = gS1StartIdx;
    if (runParam.nextTokensPerBatch < 0) {
        int64_t gs1LoopStartIdx = runParam.nextTokensPerBatch * (-1) / runParam.qSNumInOneBlock
                                    * runParam.qSNumInOneBlock;
        if (gs1LoopStartIdx > gS1StartIdx) {
            runParam.gs1LoopStartIdx = gs1LoopStartIdx;
        }
    }

    int32_t gs1LoopEndIdx = runParam.actualS1Size; // Do not split G; copy one TopK row and compute one query row each time

    // For a non-final BN, assign souterBlockNum.
    if (!lastBN) {
        runParam.gs1LoopEndIdx = gs1LoopEndIdx;
    } else { // For the final BN, read the next element from the array
        runParam.gs1LoopEndIdx = nextGs1Idx == 0 ? gs1LoopEndIdx : nextGs1Idx;
    }

    if (runParam.gs1LoopStartIdx > runParam.gs1LoopEndIdx) {
        runParam.gs1LoopStartIdx = runParam.gs1LoopEndIdx;
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
        if constexpr (IS_SPLIT_G) {
            runParam.mRealSize = runParam.mRealSize >> 1;
        }
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
        if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
            seqOffset = sIdx == 0 ? 0 : cuSeqlensQAddr[sIdx - 1];
        } else {
            seqOffset = sIdx * constInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
        if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::BSND || LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
            runParam.attentionOutOffset = attentionOutSeqOffset +
                runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                runParam.goIdx * constInfo.dSizeV;
        }
        if (constInfo.subBlockIdx == 1) {
            runParam.attentionOutOffset += runParam.firstHalfMRealSize * constInfo.dSizeV;
        }
        if (constInfo.returnSoftmaxLse) {
            if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
                // [N2, T, G] (TND)
                runParam.softmaxLseOffset = runParam.n2oIdx * constInfo.s1Size * constInfo.gSize +
                    (seqOffset + runParam.sOuterOffset) * constInfo.gSize;
            } else {
                // [B, N2, S1, G] (BSND)
                runParam.softmaxLseOffset = sIdx * constInfo.n2Size * constInfo.s1Size * constInfo.gSize +
                    runParam.n2oIdx * constInfo.s1Size * constInfo.gSize +
                    runParam.sOuterOffset * constInfo.gSize;
            }
            uint32_t aicIdx = constInfo.aivIdx >> 1U;
            if (IS_SPLIT_G && aicIdx % 2U != 0) {
                runParam.softmaxLseOffset += 64; // Add an offset of 64 when G is split
            }
            if (constInfo.subBlockIdx == 1) {
                runParam.softmaxLseOffset += runParam.firstHalfMRealSize;
                }
            }
    } else {
        if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
            runParam.tensorQOffset = runParam.qBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                runParam.n2oIdx * constInfo.gD + runParam.goIdx * constInfo.dSize;
            runParam.tensorQRopeOffset = runParam.qRopeBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                runParam.n2oIdx * constInfo.gD + runParam.goIdx * constInfo.dSizeRope;
        } else {
            runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gS1D +
                runParam.goIdx * constInfo.s1D + runParam.cubeSOuterOffset * constInfo.dSize;
            runParam.tensorQRopeOffset = runParam.qRopeBOffset + runParam.n2oIdx * constInfo.gS1D +
                runParam.goIdx * constInfo.s1D + runParam.cubeSOuterOffset * constInfo.dSizeRope;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if (runParam.nextTokensPerBatch < 0) {
        if (runParam.s1oIdx < (runParam.nextTokensPerBatch * (-1)) \
            / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock) {
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
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::TND) {
        // For TND, return true when adjacent batches have equal actualSeqQlen values.
        if (runParam.boIdx > 0 && ((runParam.boIdx == 0 && cuSeqlensQAddr[runParam.boIdx] == 0) ||
            (cuSeqlensQAddr[runParam.boIdx] - cuSeqlensQAddr[runParam.boIdx - 1] == 0))) {
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
        runParam.kvLoopEndIdx = 0;
        runParam.s2LoopEndIdx = 0;
        return true;
    }
    uint32_t s2BaseSize = constInfo.s2BaseSize;

    if (constInfo.sparseMode == sparseModeZero) {
        runParam.s2LineStartIdx = 0;
        runParam.s2LineEndIdx = Min(runParam.actualS2Size, constInfo.sparseBlockCount);
    } else if (constInfo.sparseMode == sparseModeThree) {
        runParam.s2LineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            runParam.cubeSOuterOffset - runParam.preTokensPerBatch, 0, runParam.actualS2Size);
        runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
            runParam.s1RealSize, 0, runParam.actualS2Size);
        runParam.s2LineEndIdx = Min(runParam.s2LineEndIdx, constInfo.sparseBlockCount); // The current LI output block size can only be 1
    }

    runParam.kvLoopEndIdx = (runParam.s2LineEndIdx + s2BaseSize - 1) / s2BaseSize;
    runParam.s2LoopEndIdx = runParam.kvLoopEndIdx;
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
    runInfo.kvLoopEndIdx = runParam.kvLoopEndIdx;
}

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_KVCACHE_ARCH35_H
