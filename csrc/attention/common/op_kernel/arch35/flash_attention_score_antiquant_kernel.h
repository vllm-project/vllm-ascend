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
 * \file flash_attention_score_antiquant_kernel.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_ANTIQUANT_KERNEL_H_
#define FLASH_ATTENTION_SCORE_ANTIQUANT_KERNEL_H_

const int CV_L1_EVENT[2] = {0, 1};
const int CV_MM1RES_EVENT[2] = {2, 3};
const int CV_MM2RES_EVENT[2] = {4, 5};

const int VC_L1_EVENT[2] = {6, 7};
const int VC_MM1RES_EVENT[2] = {8, 9};
const int VC_MM2RES_EVENT[2] = {10, 15};

#include "util_regbase.h"

#if __has_include("../../../common/op_kernel/matmul.h")
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/CopyInL1.h"
#else
#include "../matmul.h"
#include "../FixpipeOut.h"
#include "../CopyInL1.h"
#endif

#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_operator.h"
#include "flash_attention_score_antiquant_block_vec.h"
#include "flash_attention_score_antiquant_block_cube.h"
#include "infer_flash_attention_kvcache.h"
#include "infer_flash_attention_sparse.h"
#include "pse.h"
#include "attenmask.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using matmul::MatmulType;
using namespace optiling;

namespace BaseApi {
template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
class FlashAttentionScoreAntiquantKernel {
public:
    /*Template Variable*/
    ARGS_TRAITS_ANTIQUANT;
    using INPUT_T = KV_T;
    /*Compile Variable*/
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t dTemplateAlign64 = Align64FuncAntiquantup((uint16_t)dVTemplateType);
    static constexpr uint32_t kvAntiquantResSize = s2BaseSize * dTemplateAlign64 * sizeof(Q_T);
    static constexpr uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool useDn = false;
    __aicore__ inline FlashAttentionScoreAntiquantKernel() {};
    /*Public Function*/
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, 
        __gm__ uint8_t *pse, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
        __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *workspace, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe,
        __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
        __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
        __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
        __gm__ uint8_t* postQuantScale, __gm__ uint8_t* postQuantOffset);
    __aicore__ inline void InitActualKVPrefixLen(__gm__ uint8_t *actualSharedPrefixLen);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key,
        __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
        __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void Process();
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndx, int64_t gS1Index, RunParamStr<isInfer>& runParam);
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount,
        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline int64_t GetS2CurrentBatch();

    /*Public Variable*/
    ConstInfo<isInfer, hasRope> constInfo;
    PseInfo pseInfo;
    AttenMaskInfo attenMaskInfo;
    TPipe *pipe;
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1> mm2AL1Buffers;
    BuffersPolicyDB<BufferType::L1> kvAntiquantRes;
    TBuf<TPosition::VECIN> bmm1ResBuf[2];
    TBuf<TPosition::VECIN> bmm2ResBuf[2];
    __gm__ uint8_t *currentKey;
    /* Block */
    AntiquantCubeBlockType cubeBlock;
    AntiquantVecBlockType vecBlock;
    /*GM*/
    GlobalTensor<KV_T> keyGm;
    using prefixGmType = typename std::conditional<enableKVPrefix, GlobalTensor<KV_T>, int8_t>::type;
    prefixGmType keySharedPrefixGm;
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    int32_t aicIdx;
};

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, 
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe,
    __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
    __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
    __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
    __gm__ uint8_t* postQuantScale, __gm__ uint8_t* postQuantOffset)
{
    this->tilingData = tiling;
    this->pipe = tPipe;
    this->InitActualKVPrefixLen(actualSharedPrefixLen);
    this->ComputeConstexpr();
    this->InitInput(query, key, value, pse, attenMask, actualSeqLengths, actualSeqLengthsKv, 
        blockTable, queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
        softmaxLse, attentionOut, workspace, tiling, tPipe);
    this->InitBuffer();
    this->vecBlock.ClearOutput(this->constInfo);
    this->vecBlock.InitQuant(antiquantScale, antiquantOffset, keyAntiquantScale,
        keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, postQuantScale, postQuantOffset, this->constInfo);
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::InitActualKVPrefixLen(__gm__ uint8_t *actualSharedPrefixLen)
{
    if constexpr (isInfer) {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        if constexpr (enableKVPrefix) {
            constInfo.kvPrefixSize = inputParamsRegbase.prefixSeqInnerSize;
            if (!inputParamsRegbase.isActualSharedPrefixLenNull) {
                this->constInfo.actualKVPrefixSize = ((__gm__ int64_t *)actualSharedPrefixLen)[0];
            } else {
                this->constInfo.actualKVPrefixSize = inputParamsRegbase.prefixSeqInnerSize;
            }
            this->constInfo.prefixLoopCount = (this->constInfo.actualKVPrefixSize + s2BaseSize - 1) / s2BaseSize;
        }
    }
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    constInfo.n2Size = inputParamsRegbase.n2Size;
    constInfo.s1Size = inputParamsRegbase.s1Size;
    constInfo.s2Size = inputParamsRegbase.s2Size;
    constInfo.dSize = inputParamsRegbase.dSize;
    constInfo.dSizeV = inputParamsRegbase.dSizeV;
    if constexpr (hasRope) {
        constInfo.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = inputParamsRegbase.gSize;

    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    constInfo.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    constInfo.s1D = constInfo.s1Size * constInfo.dSize;
    constInfo.s2D = constInfo.s2Size * constInfo.dSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2D = constInfo.n2Size * constInfo.dSize;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.bN2D = inputParamsRegbase.bSize * constInfo.n2D;
    constInfo.gS1D = constInfo.gSize * constInfo.s1D;
    constInfo.n2S2D = constInfo.n2Size * constInfo.s2D;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.bN2GD = inputParamsRegbase.bSize * constInfo.n2GD;
    constInfo.n2GS1D = constInfo.n2Size * constInfo.gS1D;
    constInfo.s2BaseN2D = s2BaseSize * constInfo.n2D;
    constInfo.s1Dv = constInfo.s1D;
    constInfo.s2Dv = constInfo.s2D;
    constInfo.n2Dv = constInfo.n2D;
    constInfo.gDv = constInfo.gD;
    constInfo.gS1Dv = constInfo.gS1D;
    constInfo.n2S2Dv = constInfo.n2S2D;
    constInfo.n2GDv = constInfo.n2GD;
    constInfo.s2BaseN2Dv = constInfo.s2BaseN2D;
    constInfo.n2GS1Dv = constInfo.n2GS1D;
    constInfo.layoutType = inputParamsRegbase.layoutType;
    if ASCEND_IS_AIV {
        if constexpr (ANTIQUANT) {
            this->vecBlock.antiqSeqSize = inputParamsRegbase.antiquantParaSeqSize;
            this->vecBlock.antiquantPageAttentionFlag = inputParamsRegbase.antiquantPageAttentionFlag;
            this->vecBlock.antiquantPerTensorFlag = inputParamsRegbase.antiquantPerTensorFlag;
            this->vecBlock.antiquantPerHeadFlag = inputParamsRegbase.antiquantPerHeadFlag;
        }
        this->vecBlock.UbToL1Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    }
    if constexpr (hasRope) {
        constInfo.s1DR = constInfo.s1Size * constInfo.dSizeRope;
        constInfo.s2DR = constInfo.s2Size * constInfo.dSizeRope;
        constInfo.gDR = constInfo.gSize * constInfo.dSizeRope;
        constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
        constInfo.bN2DR = inputParamsRegbase.bSize * constInfo.n2DR;
        constInfo.gS1DR = constInfo.gSize * constInfo.s1DR;
        constInfo.n2S2DR = constInfo.n2Size * constInfo.s2DR;
        constInfo.n2GDR = constInfo.n2Size * constInfo.gDR;
        constInfo.bN2GDR = inputParamsRegbase.bSize * constInfo.n2GDR;
        constInfo.n2GS1DR = constInfo.n2Size * constInfo.gS1DR;
        constInfo.s2BaseN2DR = s2BaseSize * constInfo.n2DR;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
            constInfo.mm1RopeKa = constInfo.n2GDR;
            constInfo.mm1RopeKb = constInfo.n2DR;
        }

        constInfo.mm1Kb = constInfo.n2D;
        constInfo.mm1Ka = constInfo.n2GD;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            if (inputParamsRegbase.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
            }
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (inputParamsRegbase.isGqa && inputParamsRegbase.s1Size > 1) {
                    constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
                } else if (inputParamsRegbase.isGqa) {
                    constInfo.attentionOutStride = 0;
                }
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
            constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
            if constexpr (hasRope) {
                constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
                constInfo.mm1RopeKa = constInfo.n2GDR;
                constInfo.mm1RopeKb = constInfo.n2DR;
            }
            constInfo.mm1Ka = constInfo.n2GD;
            constInfo.mm1Kb = constInfo.n2D;
            constInfo.mm2Kb = constInfo.n2Dv;
            if constexpr (isInfer) {
                if (inputParamsRegbase.isGqa) {
                    constInfo.mm1Ka = constInfo.dSize;
                }
            }
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride =
                    (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
                if constexpr (isInfer) {
                    if (inputParamsRegbase.isGqa && inputParamsRegbase.s1Size > 1) {
                        constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
                    } else if (inputParamsRegbase.isGqa) {
                        constInfo.attentionOutStride = 0;
                    }
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            constInfo.s1BaseD = s1BaseSize * constInfo.dSize;
            constInfo.s2BaseD = s2BaseSize * constInfo.dSize;
            constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
            constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
            if constexpr (hasRope) {
                constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
                constInfo.s2BaseDR = s2BaseSize * constInfo.dSizeRope;
                constInfo.mm1RopeKa = constInfo.dSizeRope;
                constInfo.mm1RopeKb = constInfo.dSizeRope;
            }
            constInfo.mm1Ka = constInfo.dSize;
            constInfo.mm1Kb = constInfo.dSize;
            constInfo.mm2Kb = constInfo.dSizeV;
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride = 0;
            }
        }
    }
    if ASCEND_IS_AIC {
        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = inputParamsRegbase.preTokens;
            attenMaskInfo.nextTokens = inputParamsRegbase.nextTokens;
            attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
        }
    }
    if ASCEND_IS_AIV {
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            pseInfo.pseLayoutType = inputParamsRegbase.pseShapeType;
            pseInfo.pseType = inputParamsRegbase.pseType;
            pseInfo.pseBSize = inputParamsRegbase.pseBSize;
            pseInfo.pseS1Size = inputParamsRegbase.pseS1Size;
            pseInfo.pseS2Size = inputParamsRegbase.pseS2Size;
            pseInfo.pseEncodeType = (uint32_t)inputParamsRegbase.pseEncodeType;
            pseInfo.pseStride = (pseInfo.pseLayoutType == (uint32_t)PseLayoutTypeEnum::PSE_1S2)? 0 : s2BaseSize;
            pseInfo.qStartIdx = inputParamsRegbase.qStartIdx;
            pseInfo.kvStartIdx = inputParamsRegbase.kvStartIdx;
            if (inputParamsRegbase.pseShapeType == (uint32_t)PseLayoutTypeEnum::PSE_1S2) {
                constInfo.gS2 = constInfo.gSize * constInfo.s2Size;
            }
        }

        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = inputParamsRegbase.preTokens;
            attenMaskInfo.nextTokens = inputParamsRegbase.nextTokens;
            attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
            attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
            attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
        }
        constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
        constInfo.isSoftmaxLseEnable = inputParamsRegbase.isSoftMaxLseEnable;
    }
    if constexpr (isFd) {
        this->constInfo.splitKVNum = inputParamsRegbase.kvSplitPart;
        this->constInfo.sInnerLoopSize = CeilDiv(this->constInfo.s2Size, this->constInfo.splitKVNum);
        if (inputParamsRegbase.antiquantPerTensorFlag) {
            this->constInfo.sInnerLoopSize = AlignUp32(static_cast<uint64_t>(this->constInfo.sInnerLoopSize));
        }
        if constexpr (isPa) {
            this->constInfo.sInnerLoopSize = AlignUp32(static_cast<uint64_t>(this->constInfo.sInnerLoopSize));
        }
    }
    this->constInfo.isRowInvalid = inputParamsRegbase.isRowInvalid;
    this->constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
    this->constInfo.isGqa = inputParamsRegbase.isGqa;
    this->constInfo.isPfaGS1Merge = this->constInfo.isGqa && this->constInfo.s1Size > 1;
    this->constInfo.isKvContinuous = inputParamsRegbase.isKvContinuous;
    this->constInfo.actualSeqLenSize = inputParamsRegbase.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(inputParamsRegbase.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(inputParamsRegbase.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(inputParamsRegbase.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(inputParamsRegbase.isKVHasLeftPadding);
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        this->constInfo.blockSize = inputParamsRegbase.blockSize;
        this->constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
        this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }

    this->constInfo.transposeLayout = inputParamsRegbase.transposeLayout;
    if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND)) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    }
}


template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::InitInput(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
    }
    ListTensorDesc keyListTensorDescInit((__gm__ void *)key);
    currentKey = (__gm__ uint8_t *)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
        this->keyGm.SetGlobalBuffer((__gm__ KV_T *)currentKey);
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
    }
    if constexpr (enableKVPrefix) {
        this->keySharedPrefixGm.SetGlobalBuffer((__gm__ KV_T *)keySharedPrefix);
    }
    if (constInfo.isQHasLeftPadding) {
        constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
        if (constInfo.queryRightPaddingSize < 0) {
            constInfo.queryRightPaddingSize = 0;
        }
    }
    if (constInfo.isKVHasLeftPadding) {
        constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
        if (constInfo.kvRightPaddingSize < 0) {
            constInfo.kvRightPaddingSize = 0;
        }
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    } else {
        if constexpr (isInfer) {
            if (!constInfo.isActualLenDimsNull) {
                actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
            }
            if (!constInfo.isActualLenDimsKVNull) {
                actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
            }
        }
    }
    this->cubeBlock.InitCubeBlock(query, tiling, tPipe, &l1BufferManager);
    this->vecBlock.InitVecBlock(value, attentionOut, softmaxLse, pse, attenMask, blockTable,
         keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, workspace, tiling, tPipe, attenMaskInfo, pseInfo);
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::InitBuffer()
{
    /*Init Cube and Vec Common L1 Buffer*/
    l1BufferManager.Init(pipe, 524288);  // 524288 is 512 * 1024
    mm2AL1Buffers.Init(l1BufferManager, mm2LeftSize);
    kvAntiquantRes.Init(l1BufferManager, kvAntiquantResSize);
    this->cubeBlock.SendCrossCoreFlag();
    /*Init Cube and Vec Common UB Buffer*/
    this->pipe->InitBuffer(this->bmm1ResBuf[0], mm1ResultSize);
    this->pipe->InitBuffer(this->bmm1ResBuf[1], mm1ResultSize);
    this->pipe->InitBuffer(this->bmm2ResBuf[0], mm2ResultSize);
    this->pipe->InitBuffer(this->bmm2ResBuf[1], mm2ResultSize);
    this->vecBlock.SendCrossCoreFlag();
    /*Init Cube Local Buffer*/
    this->cubeBlock.InitLocalBuffer();
    /*Init Vec Local Buffer*/
    this->vecBlock.InitLocalBuffer(this->constInfo);
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::Process()
{
    if (this->tilingData->initOutputParams.needInit) {
        SyncAll<false>();
    }
    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    int32_t actualCoreNums = multiCoreParamsRegbase.coreNum;
    if constexpr (isFd) {
        actualCoreNums =  inputParamsRegbase.bSize * this->constInfo.n2Size
            * this->constInfo.splitKVNum;
    }
    if ((aicIdx) >= actualCoreNums) {
        if constexpr (isFd) {
            SyncAll();
        }
        return;
    }
    this->vecBlock.setConstAntiTaskParam(this->constInfo);
    // 确定核内切分起点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    if constexpr (!isFd) {
        bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
        gS1StartIdx = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
        if (likely((multiCoreParamsRegbase.coreNum - 1) > (aicIdx))) {
            bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
            if (nextGs1Idx != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = inputParamsRegbase.bSize * this->constInfo.n2Size *
                this->constInfo.headNumRatio;
        }
    } else {
        gS1StartIdx = 0;
        bnStartIdx = 0;
        bnEndIdx = 1;
        s2LoopLimit = 0;
    }
    int64_t taskId = 0;
    int64_t subTaskId = 0;
    bool isFirstAntiquantKey = true;
    bool isFirstAntiquantValue = true;
    bool isLastAntiquantKey = false;
    bool isLastAntiquantValue = false;
    bool isLastBmm1 = false;
    RunInfo<isInfer> runInfo[NUM_4];
    RunParamStr<isInfer> runParam;
    if constexpr (isFd) {
        runParam.boIdx = (aicIdx) / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        runParam.n2oIdx = ((aicIdx) / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        bnStartIdx = runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }
    int64_t multiCoreInnerIdx = 1;
    AscendC::ICachePreLoad(4);
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
            runParam.n2oIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, this->constInfo, lastBN, nextGs1Idx);
        if constexpr (isFd) {
            if (constInfo.sInnerLoopSize * (aicIdx % constInfo.splitKVNum) > runParam.actualS2Size) {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParam.actualS2Size -
                    this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum);
                runParam.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize :
                                        tailSInnerLoopSize;
            }
            if (constInfo.isGqa) {
                runParam.s1LoopTimes = CeilDiv(runParam.actualS1Size * constInfo.gSize, s1BaseSize);
            } else {
                runParam.s1LoopTimes = CeilDiv(runParam.actualS1Size, s1BaseSize);
            }
        }
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 2) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastTwoLoop = true;
            bool notLast = true;

            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastTwoLoop = false;
                        break;
                    case 1:
                        notLast = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastTwoLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, this->constInfo,
                    gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }

            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
                if (notLastTwoLoop) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];  // 3 is mod 4
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, runParam.s2LoopEndIdx - 1, multiCoreInnerIdx);
                    if ASCEND_IS_AIV {
                        GlobalTensor<KV_T> keyGmAnti;
                        if constexpr (enableKVPrefix) {
                            if ((runInfo1.s2LoopCount + runInfo1.s2StartIdx / s2BaseSize) < constInfo.prefixLoopCount) {
                                keyGmAnti = this->keySharedPrefixGm;
                            } else {
                                keyGmAnti = this->keyGm;
                                this->vecBlock.GetKvByTensorList(runInfo1, this->keyGm, keyGmAnti, this->constInfo);
                            }
                        } else {
                            keyGmAnti = this->keyGm;
                            this->vecBlock.GetKvByTensorList(runInfo1, this->keyGm, keyGmAnti, this->constInfo);
                        }
                        Buffer<BufferType::L1> outBufAntiKey = this->kvAntiquantRes.Get();
                        this->vecBlock.AntiquantKey(runInfo1, subTaskId, isFirstAntiquantKey,
                            runParam, outBufAntiKey, keyGmAnti, this->constInfo);
                        isFirstAntiquantKey = false;
                        subTaskId++;
                    }
                    if ASCEND_IS_AIC {
                        Buffer<BufferType::L1> inputBufmm1B = this->kvAntiquantRes.Get();
                        LocalTensor<T> outputTensorBmm1 = this->bmm1ResBuf[runInfo1.taskIdMod2].template Get<T>();
                        this->cubeBlock.IterateBmm1(runInfo1, runParam, subTaskId, inputBufmm1B, outputTensorBmm1, this->constInfo);
                        subTaskId++;
                    }
                }
                if (taskId >= 1 && notLast) {
                    RunInfo<isInfer> &runInfo2 = runInfo[(taskId - 1) & 3];  // 3 is mod 4
                    if ASCEND_IS_AIV {
                        LocalTensor<T> inputTensorVec1 = this->bmm1ResBuf[runInfo2.taskIdMod2].template Get<T>();
                        Buffer<BufferType::L1> outBufVec1 = this->mm2AL1Buffers.Get();
                        this->vecBlock.ProcessVec1(runInfo2, this->constInfo, inputTensorVec1, outBufVec1);
                        Buffer<BufferType::L1> outBufAntiValue = this->kvAntiquantRes.Get();
                        this->vecBlock.AntiquantValue(runInfo2, subTaskId, isFirstAntiquantValue, runParam,
                            outBufAntiValue, this->constInfo);
                        isFirstAntiquantValue = false;
                        subTaskId++;
                    }
                    if ASCEND_IS_AIC {
                        Buffer<BufferType::L1> inputBufmm2A = this->mm2AL1Buffers.Get();
                        Buffer<BufferType::L1> inputBufmm2B = this->kvAntiquantRes.Get();
                        LocalTensor<T> outputBufmm2 = this->bmm2ResBuf[runInfo2.taskIdMod2].template Get<T>();
                        this->cubeBlock.IterateBmm2(subTaskId, runInfo2, inputBufmm2A,
                            inputBufmm2B, outputBufmm2, this->constInfo);
                        subTaskId++;
                    }
                }
                if (taskId >= 2) {  // Later Than mm1 is 2 
                    if ASCEND_IS_AIV {
                        RunInfo<isInfer> &runInfo3 = runInfo[(taskId - 2) & 3]; // 3 is mod 4
                        LocalTensor<T> inputBufVec2 = this->bmm2ResBuf[runInfo3.taskIdMod2].template Get<T>();
                        this->vecBlock.ProcessVec2(runInfo3, inputBufVec2, this->constInfo);
                    }
                }
                taskId++;
            }
            multiCoreInnerIdx++;
        }
        gS1StartIdx = 0;
    }
    if constexpr (isFd) {
        int64_t s2InCurrentBatch = GetS2CurrentBatch();
        this->vecBlock.FlashDecode(this->constInfo, this->actualSeqQlenAddr, this->actualSeqKvlenAddr, s2InCurrentBatch);
    }
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::SetRunInfo(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;  // 3 is mod 3
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;  // 3 is mod num 
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, constInfo.s1Size, constInfo.s2Size);
    } else {
        runInfo.b1SSOffset = runInfo.boIdx * constInfo.s1S2;
        runInfo.b1SSOffsetAlign = runInfo.boIdx * constInfo.s1Size * Align(constInfo.s2Size);
    }

    if constexpr (isFd) {
        runInfo.flashDecodeS2Idx = this->aicIdx % constInfo.splitKVNum;
    }
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    if constexpr (isInfer) {
        runInfo.qRopeOffset = runParam.qRopeNBGOffset;
        InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, runInfo);
        ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, s2LoopCount + runInfo.s2StartIdx / s2BaseSize, runInfo);
        if ASCEND_IS_AIV{
            ComputeOffsetForAntiquant<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, s2LoopCount + runInfo.s2StartIdx / s2BaseSize, runInfo);
        }
    }
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if (runParam.boIdx == 0) {
            this->s1SizeAcc = 0;
            this->s2SizeAcc = 0;
        } else {
            this->s1SizeAcc = actualSeqQlenAddr[runParam.boIdx - 1];
            if constexpr (isPa) {
                this->s2SizeAcc = 0;
                for (uint32_t boIdx = 0; boIdx < runParam.boIdx; boIdx++) {
                    this->s2SizeAcc += actualSeqKvlenAddr[boIdx];
                }
            } else {
                this->s2SizeAcc = actualSeqKvlenAddr[runParam.boIdx - 1];
            }
        }
    }
    if (this->constInfo.isGqa) {
        runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParam.s1oIdx = gS1Index % constInfo.s1OuterSize;
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::GetSeqQlenKvlenByBoidx(
    int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvlen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    if constexpr (isPa) {
        actualSeqKvlen = actualSeqKvlenAddr[boIdx];
    } else {
        actualSeqKvlen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
    }
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if constexpr (enableKVPrefix) {
        if ((runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) < constInfo.prefixLoopCount) {
            if ((runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize) > constInfo.actualKVPrefixSize) {
                runInfo.s2RealSize = constInfo.actualKVPrefixSize - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
                runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
            }
        } else {
            if (runInfo.s2StartIdx + (runInfo.s2LoopCount - constInfo.prefixLoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx - constInfo.actualKVPrefixSize) {
                runInfo.s2RealSize = (runInfo.s2EndIdx - constInfo.actualKVPrefixSize) - (runInfo.s2LoopCount - constInfo.prefixLoopCount) * runInfo.s2RealSize - runInfo.s2StartIdx;
                runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
            }
        }
    } else {
        if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    }
}

template <typename AntiquantCubeBlockType, typename AntiquantVecBlockType>
__aicore__ inline int64_t FlashAttentionScoreAntiquantKernel<AntiquantCubeBlockType, AntiquantVecBlockType>::GetS2CurrentBatch()
{
    int64_t bIdx = constInfo.aivIdx / constInfo.n2Size;
    int64_t s2InCurrentBatch = constInfo.s2Size;
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, bIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    return s2InCurrentBatch;
}
}
#endif