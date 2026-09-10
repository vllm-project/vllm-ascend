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
 * \file sparse_flash_mla_kvcache.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_KVCACHE_H
#define SPARSE_FLASH_MLA_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "sparse_flash_mla_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace SMLAKernel;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr &runParam, const ConstInfo &constInfo,
                                          GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &cuSeqlensOriKvGm,
                                          GlobalTensor<int32_t> &cuSeqlensCmpKvGm,
                                          GlobalTensor<int32_t> &actualSeqQlenGm,
                                          GlobalTensor<int32_t> &actualSeqOriKvlenGm,
                                          GlobalTensor<int32_t> &actualSeqCmpKvlenGm,
                                          GlobalTensor<int32_t> &cmpResidualKvGm, bool hasActualSeqQlen,
                                          bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv)
{
    int32_t actualS1Size = 0;
    int32_t actualS2OriSize = 0;
    int32_t actualS2CmpSize = 0;
    int32_t bIdx = runParam.boIdx;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        actualS1Size = (!hasActualSeqQlen) ? (cuSeqlensQGm.GetValue(bIdx + 1) - cuSeqlensQGm.GetValue(bIdx)) :
                                             actualSeqQlenGm.GetValue(bIdx);
    } else {
        actualS1Size = (!hasActualSeqQlen) ? constInfo.s1Size : actualSeqQlenGm.GetValue(bIdx);
    }

    if (constInfo.isActualLenDimsOriKVNull) {
        actualS2OriSize = constInfo.s2Size;
    } else {
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
            if (hasActualSeqOriKvlen) {
                actualS2OriSize = actualSeqOriKvlenGm.GetValue(bIdx);
            } else {
                actualS2OriSize = cuSeqlensOriKvGm.GetValue(bIdx + 1) - cuSeqlensOriKvGm.GetValue(bIdx);
            }
        } else {
            actualS2OriSize = actualSeqOriKvlenGm.GetValue(bIdx);
        }
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            if (hasActualSeqCmpKvlen) {
                actualS2CmpSize = actualSeqCmpKvlenGm.GetValue(bIdx);
            } else if (hasCuSeqlensCmpKv) {
                actualS2CmpSize = cuSeqlensCmpKvGm.GetValue(bIdx + 1) - cuSeqlensCmpKvGm.GetValue(bIdx);
            } else {
                actualS2CmpSize = actualS2OriSize / constInfo.cmpRatio;
            }
        } else {
            if (!hasActualSeqCmpKvlen) {
                actualS2CmpSize = actualS2OriSize / constInfo.cmpRatio;
            } else {
                actualS2CmpSize = actualSeqCmpKvlenGm.GetValue(bIdx);
            }
        }
    }

    runParam.actualS1Size = actualS1Size;
    runParam.actualS2OriSize = actualS2OriSize;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        runParam.actualS2CmpSize = actualS2CmpSize;
        if (constInfo.cmpMaskMode == 0) {
            runParam.nextTokensPerBatchCmp = runParam.actualS2CmpSize * constInfo.cmpRatio;
        } else {
            runParam.cmpResidual = (cmpResidualKvGm.GetPhyAddr() != nullptr) ? cmpResidualKvGm.GetValue(bIdx) : 0;
            runParam.nextTokensPerBatchCmp =
                (int64_t)runParam.actualS2CmpSize * constInfo.cmpRatio + runParam.cmpResidual - runParam.actualS1Size;
        }
    }

    if (constInfo.oriMaskMode == 3) {
        runParam.nextTokensPerBatchOri = runParam.actualS2OriSize - runParam.actualS1Size;
        runParam.preTokensPerBatchOri = runParam.actualS1Size;
    } else if (constInfo.oriMaskMode == 4) {
        const int64_t casualOffset = runParam.actualS2OriSize - runParam.actualS1Size;

        runParam.preTokensPerBatchOri =
            (constInfo.oriWinLeft == -1) ? runParam.actualS1Size : constInfo.oriWinLeft - casualOffset;

        runParam.nextTokensPerBatchOri =
            (constInfo.oriWinRight == -1) ? runParam.actualS2OriSize : casualOffset + constInfo.oriWinRight;

        runParam.preTokensPerBatchOri = Min(runParam.preTokensPerBatchOri, static_cast<int64_t>(runParam.actualS1Size));
    } else if (constInfo.oriMaskMode == 0) {
        runParam.nextTokensPerBatchOri = runParam.actualS2OriSize;
        runParam.preTokensPerBatchOri = runParam.actualS1Size;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr &runParam, const ConstInfo &constInfo,
                                         GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &cuSeqlensOriKvGm,
                                         GlobalTensor<int32_t> &cuSeqlensCmpKvGm,
                                         GlobalTensor<int32_t> &actualSeqQlenGm,
                                         GlobalTensor<int32_t> &actualSeqOriKvlenGm,
                                         GlobalTensor<int32_t> &actualSeqCmpKvlenGm,
                                         GlobalTensor<int32_t> &cmpResidualKvGm, bool hasActualSeqQlen,
                                         bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, cuSeqlensQGm, cuSeqlensOriKvGm, cuSeqlensCmpKvGm,
                                           actualSeqQlenGm, actualSeqOriKvlenGm, actualSeqCmpKvlenGm, cmpResidualKvGm,
                                           hasActualSeqQlen, hasActualSeqOriKvlen, hasActualSeqCmpKvlen,
                                           hasCuSeqlensCmpKv);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr &runParam, const ConstInfo &constInfo, bool lastBN,
                                         int64_t nextGs1Idx, int64_t gS1StartIdx, int64_t s2EndIdx = 0)
{
    runParam.qSNumInOneBlock = 1;
    runParam.gs1LoopStartIdx = gS1StartIdx;
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (runParam.nextTokensPerBatchOri < 0) {
            int64_t gs1LoopStartIdx =
                runParam.nextTokensPerBatchOri * (-1) / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock;
            if (gs1LoopStartIdx > gS1StartIdx) {
                runParam.gs1LoopStartIdx = gs1LoopStartIdx;
            }
        }
    }

    int32_t gs1LoopEndIdx = 0;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        gs1LoopEndIdx = runParam.actualS1Size;
    } else { // SWA/HCA
        // 不需要取topk, 每次计算gSize行, 循环qs次
        gs1LoopEndIdx = (runParam.actualS1Size + runParam.qSNumInOneBlock - 1) / runParam.qSNumInOneBlock;
    }
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.gs1LoopEndIdx = gs1LoopEndIdx;
    } else { // 最后一个bn, 从数组下一个元素取值
        uint32_t actualNextGs1Idx = s2EndIdx == 0 ? nextGs1Idx : nextGs1Idx + 1;
        runParam.gs1LoopEndIdx = (nextGs1Idx == 0 && s2EndIdx == 0) ? gs1LoopEndIdx : actualNextGs1Idx;
    }

    if (runParam.gs1LoopStartIdx > runParam.gs1LoopEndIdx) {
        runParam.gs1LoopStartIdx = runParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr &runParam, const ConstInfo &constInfo, uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * runParam.qSNumInOneBlock;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
        runParam.mRealSize = 0;
    } else {
        runParam.s1RealSize = Min(runParam.qSNumInOneBlock, runParam.actualS1Size - cubeSOuterOffset);
        runParam.mRealSize = runParam.s1RealSize * constInfo.gSize;
        if constexpr (IS_SPLIT_G) {
            runParam.mRealSize = runParam.s1RealSize * runParam.gSplitSize;
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
        runParam.sOuterOffset = cubeSOuterOffset + runParam.firstHalfMRealSize / constInfo.gSize;
    } else {
        runParam.sOuterOffset = cubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr &runParam, const ConstInfo &constInfo, int32_t sIdx,
                                            GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            seqOffset = cuSeqlensQGm.GetValue(sIdx);
        } else {
            seqOffset = sIdx * constInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND || LAYOUT_T == SMLA_LAYOUT::TND) {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.sOuterOffset * constInfo.n2GDv +
                                          runParam.n2oIdx * constInfo.gDv + runParam.goIdx * constInfo.dSizeV;
        }
        if (constInfo.subBlockIdx == 1) {
            runParam.attentionOutOffset += runParam.firstHalfMRealSize * constInfo.dSizeV;
        }
        if (constInfo.returnSoftmaxLse) {
            if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
                // [N2, T, G] (TND)
                runParam.softmaxLseOffset = runParam.n2oIdx * constInfo.s1Size * constInfo.gSize +
                                            (seqOffset + runParam.sOuterOffset) * constInfo.gSize;
            } else {
                // [B, N2, S1, G] (BSND)
                runParam.softmaxLseOffset = sIdx * constInfo.n2Size * constInfo.s1Size * constInfo.gSize +
                                            runParam.n2oIdx * constInfo.s1Size * constInfo.gSize +
                                            runParam.sOuterOffset * constInfo.gSize;
            }
            if (IS_SPLIT_G) {
                runParam.softmaxLseOffset += runParam.goIdx;
            }
            if (constInfo.subBlockIdx == 1) {
                runParam.softmaxLseOffset += runParam.firstHalfMRealSize;
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr &runParam, const ConstInfo &constInfo, uint32_t sOuterLoopIdx,
                                      GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (runParam.nextTokensPerBatchOri < 0) {
            if (runParam.s1oIdx <
                (runParam.nextTokensPerBatchOri * (-1)) / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock) {
                return true;
            }
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, cuSeqlensQGm);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr &runParam, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (runParam.boIdx > 0 &&
            cuSeqlensQGm.GetValue(runParam.boIdx + 1) - cuSeqlensQGm.GetValue(runParam.boIdx) == 0) {
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
__aicore__ inline bool ComputeS2LoopInfo(int64_t bnIndex, int64_t gS1Index, GlobalTensor<int32_t> &cuSeqlensQGm,
                                         GlobalTensor<int32_t> &oriTopkLengthGm, GlobalTensor<int32_t> &cmpTopkLengthGm,
                                         RunParamStr &runParam, const ConstInfo &constInfo)
{
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (runParam.actualS2OriSize == 0) {
            runParam.oriKvLoopEndIdx = 0;
            runParam.cmpKvLoopEndIdx = 0;
            runParam.s2LoopEndIdx = 0;
            runParam.s2CmpLineStartIdx = 0;
            return true;
        }
    }
    uint32_t s2BaseSize = constInfo.s2BaseSize;

    // 计算topk length
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runParam.boIdx);
        runParam.oriSparseBlockCount =
            constInfo.hasOriTopkLength ?
                Min(oriTopkLengthGm.GetValue(actualSeqQPrefixSum + runParam.s1oIdx), constInfo.oriSparseBlockCount) :
                constInfo.oriSparseBlockCount;
        runParam.cmpSparseBlockCount =
            constInfo.hasCmpTopkLength ?
                Min(cmpTopkLengthGm.GetValue(actualSeqQPrefixSum + runParam.s1oIdx), constInfo.cmpSparseBlockCount) :
                constInfo.cmpSparseBlockCount;
    } else {
        uint64_t bsndTopkIdx = runParam.boIdx * constInfo.s1Size + runParam.s1oIdx;
        runParam.oriSparseBlockCount = constInfo.hasOriTopkLength ?
                                           Min(oriTopkLengthGm.GetValue(bsndTopkIdx), constInfo.oriSparseBlockCount) :
                                           constInfo.oriSparseBlockCount;
        runParam.cmpSparseBlockCount = constInfo.hasCmpTopkLength ?
                                           Min(cmpTopkLengthGm.GetValue(bsndTopkIdx), constInfo.cmpSparseBlockCount) :
                                           constInfo.cmpSparseBlockCount;
    }

    // orikv
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        runParam.s2OriLineStartIdx = 0;
        runParam.s2OriLineEndIdx = runParam.oriSparseBlockCount;
    } else {
        runParam.s2OriLineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            runParam.cubeSOuterOffset - runParam.preTokensPerBatchOri, 0, runParam.actualS2OriSize);
        runParam.s2OriLineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            runParam.cubeSOuterOffset + runParam.nextTokensPerBatchOri + runParam.s1RealSize, 0,
            runParam.actualS2OriSize);
    }
    runParam.oriKvLoopEndIdx = (runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx + s2BaseSize - 1) / s2BaseSize;

    // cmpkv
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SWA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        runParam.s2CmpLineStartIdx = 0;
        runParam.s2CmpLineEndIdx = 0;
        runParam.cmpKvLoopEndIdx = 0;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::HCA_TEMPLATE_MODE) {
        runParam.s2CmpLineStartIdx = 0;
        runParam.s2CmpLineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            (runParam.cubeSOuterOffset + runParam.s1RealSize + runParam.nextTokensPerBatchCmp) / constInfo.cmpRatio, 0,
            runParam.actualS2CmpSize);
        runParam.s2CmpLineEndIdx = Min(runParam.s2CmpLineEndIdx, runParam.actualS2CmpSize);
        runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        runParam.s2CmpLineStartIdx = 0;
        runParam.s2CmpLineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            (runParam.cubeSOuterOffset + runParam.s1RealSize + runParam.nextTokensPerBatchCmp) / constInfo.cmpRatio, 0,
            runParam.actualS2CmpSize);
        runParam.s2CmpLineEndIdx = Min(runParam.s2CmpLineEndIdx, runParam.cmpSparseBlockCount);
        runParam.s2CmpLineEndIdx = Min(runParam.s2CmpLineEndIdx, runParam.actualS2CmpSize);
        runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
    } else { // ORI_CMP_SPARSE_TEMPLATE_MODE
        runParam.s2CmpLineStartIdx = 0;
        runParam.s2CmpLineEndIdx = runParam.cmpSparseBlockCount;
        runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
    }

    runParam.s2LoopEndIdx = runParam.oriKvLoopEndIdx + runParam.cmpKvLoopEndIdx;
    return (runParam.s2LoopEndIdx == 0);
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr &runParam, RunInfo &runInfo, const ConstInfo &constInfo)
{
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatchOri = runParam.preTokensPerBatchOri;
    runInfo.nextTokensPerBatchOri = runParam.nextTokensPerBatchOri;
    runInfo.actualS1Size = runParam.actualS1Size;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        runInfo.actualS2CmpSize = runParam.actualS2CmpSize;
        runInfo.cmpResidual = runParam.cmpResidual;
    }
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
    runInfo.qSNumInOneBlock = runParam.qSNumInOneBlock;
    runInfo.oriKvLoopEndIdx = runParam.oriKvLoopEndIdx;
    runInfo.cmpKvLoopEndIdx = runParam.cmpKvLoopEndIdx;
    runInfo.isCmp = runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx;
    runInfo.oriSparseBlockCount = runParam.oriSparseBlockCount;
    runInfo.cmpSparseBlockCount = runParam.cmpSparseBlockCount;
}

#endif // SPARSE_FLASH_MLA_KVCACHE_H
