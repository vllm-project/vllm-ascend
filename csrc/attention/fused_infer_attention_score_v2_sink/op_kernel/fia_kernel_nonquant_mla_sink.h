/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_kernel_nonquant_mla_sink.h
 * \brief
 */

#ifndef FIA_KERNEL_NONQUANT_MLA_SINK_H
#define FIA_KERNEL_NONQUANT_MLA_SINK_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#if __has_include("../common/op_kernel/fia_public_define.h")
#include "fia_public_define.h"
#include "vector_common.h"
#include "memory_copy.h"
#else
#include "fia_public_define.h"
#include "vector_common.h"
#include "memory_copy.h"
#endif
#include "kernel_common_sink.h"
#include "fia_block_vec_nonquant_mla_sink.h"
#include "fia_block_cube_nonquant_mla_sink.h"
#include "fia_block_vec_flashdecode_sink.h"
#include "fused_infer_attention_score_v2_sink_metadata.h"
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FIAT, typename CubeBlockType,
    typename VecBlockType, typename FdBlockType> class FiaKernelNonQuantMla {
  public:
    __aicore__ inline FiaKernelNonQuantMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query,
                                __gm__ uint8_t *key,
                                __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift,
                                __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *deqScale1,
                                __gm__ uint8_t *quantScale1,
                                __gm__ uint8_t *deqScale2,
                                __gm__ uint8_t *quantScale2,
                                __gm__ uint8_t *quantOffset2,
                                __gm__ uint8_t *antiquantScale,
                                __gm__ uint8_t *antiquantOffset,
                                __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryPaddingSize,
                                __gm__ uint8_t *kvPaddingSize,
                                __gm__ uint8_t *keyAntiquantScale,
                                __gm__ uint8_t *keyAntiquantOffset,
                                __gm__ uint8_t *valueAntiquantScale,
                                __gm__ uint8_t *valueAntiquantOffset,
                                __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *keyRopeAntiquantScale,
                                __gm__ uint8_t *metadata,
                                __gm__ uint8_t *learnableSink,
                                __gm__ uint8_t *keySink,
                                __gm__ uint8_t *keyRopeSink,
                                __gm__ uint8_t *valueSink,
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *softmaxMax,
                                __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *workspace,
                                const FusedInferAttentionScoreV2SinkTilingData *__restrict tiling,
                                __gm__ uint8_t *gmTiling,
                                TPipe *tPipe,
                                bool isPrefix = false);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    using ORIGIN_T = typename FIAT::orginalType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr bool FLASH_DECODE = FIAT::flashDecode;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t ANTIQUANT_MODE = FIAT::antiquantMode;
    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using ANTIQ_PARAMS_T = Q_T;
    using Q_ROPE_T = typename AscendC::Conditional<ANTIQUANT, Q_T, ORIGIN_T>::type;
    using K_ROPE_T = typename AscendC::Conditional<ANTIQUANT, KV_T, ORIGIN_T>::type;
    using UPDATE_T = typename AscendC::Conditional<QUANT || ANTIQUANT, half, T>::type;
    using TMP_T = typename AscendC::Conditional<ANTIQUANT, half, T>::type;
    using MM1_OUT_T = typename AscendC::Conditional<QUANT, int32_t, TMP_T>::type;
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, half, TMP_T>::type;

    CubeBlockType matmulService;
    VecBlockType vectorService;
    FdBlockType fdService;

    // =================================常量区=================================
    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t FIA_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V2_C2_FLAG = 2;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr int64_t fdPrefetchLen = 2;

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    // ==============================TilingData&TPipe==============================
    const FusedInferAttentionScoreV2SinkTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;
    GlobalTensor<uint32_t> metadataGm;
    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    // ================================Global Buffer区=================================
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    GlobalTensor<KV_T> vec1ResGm; // 存放A1, A2
    GlobalTensor<MM2_OUT_T> mm2ResGm; // 存放O

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    // ================================Task Info====================================
    ConstInfo constInfo{};
    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;
    uint64_t bn2IdxInCurCore = 0ULL;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;
    uint32_t curS2Start;
    uint32_t curS2End;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    uint32_t firstTokenWithSinkNumber = 0U;

    bool skipInitOutputFlag = false;
    bool fdFlag = false;
    // ================================Util functions==================================
    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a > b) ? (b) : (a);
    }

    template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b)
    {
        return (a > b) ? (a) : (b);
    }
    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitActualSeqLenQ(__gm__ uint8_t *actualSeqLengthsQ);
    __aicore__ inline void InitActualSeqLenKV(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline bool IsInitAttentionOutGm();
    __aicore__ inline void InitOutputSingleCore();
    // ================================Tool============================================
    __aicore__ inline uint32_t GetBIdx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetN2Idx(uint32_t bN2Idx);
    __aicore__ inline void
    GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preToken, int64_t &nextToken);
    // ================================Process functions================================
    __aicore__ inline void FlashAttention();
    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfo &info);
    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur);
    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur);
    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur);
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void FlashDecode();
    // ================================PIPE Control=====================================
    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur);

    __aicore__ inline void DealZeroActSeqLen(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur);
    __aicore__ inline bool IsSkip(uint32_t bN2Cur);
    __aicore__ inline void UpdateAxisInfo(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur);
    __aicore__ inline void CreateTask(uint64_t loop,
                                      uint32_t bN2Cur,
                                      uint32_t gS1Cur,
                                      uint32_t s2Cur,
                                      RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline bool ShouldExecuteTask(RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE]);
};

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::InitTilingData()
{
    usedCoreNum = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::USED_CORE_NUM_INDEX));
    constInfo.batchSize = tilingData->baseParams.bSize;
    constInfo.gSize = tilingData->baseParams.gSize;
    constInfo.kvHeadNum = tilingData->baseParams.n2Size;
    constInfo.qHeadNum = constInfo.gSize * constInfo.kvHeadNum;
    constInfo.kvSeqSize = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::S2_SIZE_INDEX));
    constInfo.qSeqSize = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::S1_SIZE_INDEX));
    constInfo.qTSize = tilingData->baseParams.qTSize;
    constInfo.kTSize = tilingData->baseParams.kTSize;
    constInfo.attenMaskFlag = (tilingData->maskParams.attenMaskFlag != 0) ? true : false;
    constInfo.attenMaskSize = tilingData->maskParams.attenMaskSize;
    constInfo.attenMaskStride = tilingData->maskParams.attenMaskStride;
    constInfo.sparseMode = tilingData->maskParams.sparseMode;
    constInfo.preToken = tilingData->maskParams.preToken;
    constInfo.nextToken = tilingData->maskParams.nextToken;
    constInfo.maxBlockNumPerBatch = tilingData->pageAttenParams.maxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->pageAttenParams.blockSize;
    constInfo.outputLayout = static_cast<FIA_LAYOUT>(tilingData->baseParams.outputLayout);
    constInfo.mBaseSize = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::M_BASE_SIZE_INDEX));
    constInfo.s2BaseSize = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::S_INNER_SIZE_INDEX));
    constInfo.batchContinuous = tilingData->baseParams.batchContinuous;
    constInfo.batchInvariant = (tilingData->baseParams.batchInvariant != 0);
    constInfo.softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;
    constInfo.softmaxMaxSumFlag = (tilingData->baseParams.softmaxMaxSumFlag != 0);

    constInfo.headDim = tilingData->baseParams.headDim;
    constInfo.headDimRope = tilingData->baseParams.headDimRope;
    constInfo.headDimAlign = Align(constInfo.headDim, (uint64_t)AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK);

    constInfo.mmResUbSize = tilingData->workspaceParams.mm1ResSize;
    constInfo.bmm2ResUbSize = tilingData->workspaceParams.mm2ResSize;
    constInfo.vec1ResUbSize = constInfo.mmResUbSize;
    constInfo.isRowInvalid = (tilingData->maskParams.isRowInvalid != 0);

    constInfo.needInit = tilingData->baseParams.needInit;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    constInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    constInfo.syncV2C2 = SYNC_V2_C2_FLAG;
    constInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
    constInfo.sinkNumber = tilingData->maskParams.sinkNumber;
    constInfo.keySinkNumber = tilingData->maskParams.keySinkNumber;
// 读取非连续 stride
#ifdef ASCENDC_CPU_DEBUG
    const uint64_t *keyStridePtr = tilingData->pageAttenParams.keyStride;
    const uint64_t *valueStridePtr = tilingData->pageAttenParams.valueStride;
    const uint64_t *keyRopeStridePtr = tilingData->pageAttenParams.keyRopeStride;
#else
    uint64_t keyStridePtr[ARRAY_SIZE(tilingData->pageAttenParams.keyStride)];
    uint64_t valueStridePtr[ARRAY_SIZE(tilingData->pageAttenParams.valueStride)];
    uint64_t keyRopeStridePtr[ARRAY_SIZE(tilingData->pageAttenParams.keyRopeStride)];
    copy_data_align64(
        (uint8_t *)keyStridePtr, (uint8_t *)(tilingData->pageAttenParams.keyStride), sizeof(keyStridePtr));
    copy_data_align64(
        (uint8_t *)valueStridePtr, (uint8_t *)(tilingData->pageAttenParams.valueStride), sizeof(valueStridePtr));
    copy_data_align64(
        (uint8_t *)keyRopeStridePtr, (uint8_t *)(tilingData->pageAttenParams.keyRopeStride), sizeof(keyRopeStridePtr));
#endif
    if (keyStridePtr[0] > 0) {
        for (uint32_t i = 0; i < 5; i++) {
            constInfo.keyStride[i] = keyStridePtr[i];
            constInfo.valueStride[i] = valueStridePtr[i];
            constInfo.keyRopeStride[i] = keyRopeStridePtr[i];
        }
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::InitActualSeqLenQ(
    __gm__ uint8_t *actualSeqLengthsQ)
{
    constInfo.actualLenQDims = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::ACUTAL_LEN_Q_DIM_INDEX));
    constInfo.accumQSeqFlag = tilingData->baseParams.accumQSeqFlag;
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    qActSeqLensParser.Init(actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::InitActualSeqLenKV(
    __gm__ uint8_t *actualSeqLengths)
{
    constInfo.actualLenDims = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::ACUTAL_LEN_KV_DIM_INDEX));
    ;
    constInfo.accumKVSeqFlag = tilingData->baseParams.accumKVSeqFlag;
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
    kvActSeqLensParser.Init(actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline bool FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::IsInitAttentionOutGm()
{
    // TND、NTD场景且无attentionMask,不需要初始化
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        if (tilingData->maskParams.attenMaskFlag == 0 && (constInfo.qTSize == qActSeqLensParser.GetTSize())) {
            return false;
        }
    } else {
        if (tilingData->baseParams.actualSeqS1Dims == 0 && tilingData->maskParams.attenMaskFlag == 0) {
            return false;
        }
    }

    return true;
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::InitOutputSingleCore()
{
    if (skipInitOutputFlag) {
        return;
    }
    if (usedCoreNum != 0) {
        int32_t aivCoreNum = usedCoreNum * constInfo.subBlockNum;
        uint32_t initOutputEventId = 0U;
        SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        uint64_t tSize = constInfo.batchSize * constInfo.qSeqSize;
        if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
            tSize = constInfo.qTSize;
        }
        // TND、NTD场景,S1和actualSeq相等,不需要初始化
        if (IsInitAttentionOutGm()) {
            uint64_t totalOutputSize = tSize * constInfo.qHeadNum * constInfo.headDim;
            uint64_t singleCoreSize = (totalOutputSize + aivCoreNum - 1) / aivCoreNum;
            uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
            uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if (tmpBlockIdx * singleCoreSize < totalOutputSize && singleInitOutputSize > 0) {
                matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }

        // lse/max/sum 三者 shape 相同，共用 size 计算
        uint64_t totalLseSize = tSize * constInfo.qHeadNum;
        uint64_t singleCoreLseSize = (totalLseSize + aivCoreNum - 1) / aivCoreNum;
        uint64_t tailLseSize = totalLseSize - tmpBlockIdx * singleCoreLseSize;
        uint64_t singleInitOutputLseSize = tailLseSize < singleCoreLseSize ? tailLseSize : singleCoreLseSize;

        // softmaxLseFlag 独立控制 lse init
        if (constInfo.softmaxLseFlag) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if (tmpBlockIdx * singleCoreLseSize < totalLseSize && singleInitOutputLseSize > 0) {
                matmul::InitOutput<float>(
                    softmaxLseGm[tmpBlockIdx * singleCoreLseSize], singleInitOutputLseSize, constInfo.FLOAT_INF);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }

        // softmaxMaxSumFlag 独立控制 max/sum init
        if (constInfo.softmaxMaxSumFlag) {
            // softmaxMax init: fill with -inf
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if (tmpBlockIdx * singleCoreLseSize < totalLseSize && singleInitOutputLseSize > 0) {
                matmul::InitOutput<float>(
                    softmaxMaxGm[tmpBlockIdx * singleCoreLseSize], singleInitOutputLseSize, -constInfo.FLOAT_INF);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            // softmaxSum init: fill with 0
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if (tmpBlockIdx * singleCoreLseSize < totalLseSize && singleInitOutputLseSize > 0) {
                matmul::InitOutput<float>(
                    softmaxSumGm[tmpBlockIdx * singleCoreLseSize], singleInitOutputLseSize, 0.0f);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        SyncAll(); // 硬同步要求所有核都进行同步，此处对实际使用的核进行数据同步
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::Init(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask,
    __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1,
    __gm__ uint8_t *quantScale1,
    __gm__ uint8_t *deqScale2,
    __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2,
    __gm__ uint8_t *antiquantScale,
    __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset,
    __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale,
    __gm__ uint8_t *metadata,
    __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *keySink,
    __gm__ uint8_t *keyRopeSink,
    __gm__ uint8_t *valueSink,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace,
    const FusedInferAttentionScoreV2SinkTilingData *__restrict tiling,
    __gm__ uint8_t *gmTiling,
    TPipe *tPipe,
    bool isPrefix)
{
    if ASCEND_IS_AIV {
        constInfo.subBlockNum = GetSubBlockNum(); // CV1:2场景,返回值为2，其他场景返回值为1
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / constInfo.subBlockNum;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;
    metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
    // 初始化计算参数
    InitActualSeqLenQ(actualSeqLengthsQ);
    InitActualSeqLenKV(actualSeqLengths);
    skipInitOutputFlag = !IsInitAttentionOutGm() &&
                         !tilingData->baseParams.softmaxLseFlag &&
                         !tilingData->baseParams.softmaxMaxSumFlag;
    if (aiCoreIdx >= tilingData->baseParams.usedCoreNum) {
        if ASCEND_IS_AIV {
            if (!skipInitOutputFlag) {
                // superkernel 场景，启动核数大于实际运行核数时，未启动的核仅需要保留 SyncAll
                SyncAll();
            }
        }
        return;
    }

    InitTilingData();
    InitCalcParamsEach();
    constInfo.ropeSplitMode = (queryRope != nullptr);
    fdFlag = (metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::NUM_OF_FD_INDEX)) > 0U);
    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    // init global buffer
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    if (constInfo.softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
    if (constInfo.softmaxMaxSumFlag) {
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    }

    if ASCEND_IS_AIV {
        InitOutputSingleCore();
    }

    // workspace 内存排布
    uint64_t offset = 0;

    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    vec2ResGm.SetGlobalBuffer(
        (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T);

    if constexpr (FLASH_DECODE) {
        if (fdFlag) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
            offset = offset + tilingData->workspaceParams.fdAccumOutSize * sizeof(float);
            lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
            lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) +
                                       tilingData->workspaceParams.fdLogSumExpSize / 2);
            offset = offset + tilingData->workspaceParams.fdLogSumExpSize * sizeof(float);
        }
    }

    if ASCEND_IS_AIV {
        if constexpr (FLASH_DECODE) {
            if (fdFlag) {
                fdService.InitParams(constInfo);
                fdService.InitGlobalTensor(
                    lseMaxFdGm, lseSumFdGm, accumOutGm, attentionOutGm, actualSeqLengthsGmQ, actualSeqLengthsGm);
                if (constInfo.softmaxLseFlag) {
                    fdService.InitSoftmaxLseGm(softmaxLseGm);
                }
            }
        }

        // 由于softmaxMax、softmaxSum输出的前提条件是batchInvariant=true，FD关闭，所以仅在ProcessVec1流程传入计算
        vectorService.InitParams(constInfo);
        vectorService.Init(query,
                           key,
                           value,
                           pseShift,
                           attenMask,
                           actualSeqLengthsQ,
                           actualSeqLengths,
                           deqScale1,
                           quantScale1,
                           deqScale2,
                           quantScale2,
                           quantOffset2,
                           antiquantScale,
                           antiquantOffset,
                           blockTable,
                           queryPaddingSize,
                           kvPaddingSize,
                           keyAntiquantScale,
                           keyAntiquantOffset,
                           valueAntiquantScale,
                           valueAntiquantOffset,
                           queryRope,
                           keyRope,
                           keyRopeAntiquantScale,
                           metadata,
                           attentionOut,
                           softmaxLse,
                           softmaxMax,
                           softmaxSum,
                           tilingData);
        if (constInfo.batchInvariant) {
            vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm);
        } else {
            vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, mm2ResInt32Gm);
        }
        vectorService.InitVec2GlobalTensor(vec2ResGm, mm2ResGm);
        vectorService.InitFlashDecodeGlobalTensor(accumOutGm, lseMaxFdGm, lseSumFdGm);
    }

    if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.InitMm1GlobalTensor(mm1ResGm);
        matmulService.InitMm2GlobalTensor(vec1ResGm, mm2ResGm);
        matmulService.Init(query,
                           key,
                           value,
                           pseShift,
                           attenMask,
                           actualSeqLengthsQ,
                           actualSeqLengths,
                           deqScale1,
                           quantScale1,
                           deqScale2,
                           quantScale2,
                           quantOffset2,
                           antiquantScale,
                           antiquantOffset,
                           blockTable,
                           queryPaddingSize,
                           kvPaddingSize,
                           keyAntiquantScale,
                           keyAntiquantOffset,
                           valueAntiquantScale,
                           valueAntiquantOffset,
                           queryRope,
                           keyRope,
                           keyRopeAntiquantScale,
                           keySink,
                           keyRopeSink,
                           valueSink,
                           attentionOut,
                           softmaxLse);
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::InitCalcParamsEach()
{
    // 任务起始位置
    if (aiCoreIdx == 0) {
        constInfo.bN2Start = 0;
        constInfo.gS1Start = 0;
        constInfo.s2Start = 0;
    } else {
        constInfo.bN2Start =
            metadataGm.GetValue(optiling::GetAICMetaAbsIndex(aiCoreIdx - 1, optiling::BN2_END_PTR_INDEX));
        constInfo.gS1Start =
            metadataGm.GetValue(optiling::GetAICMetaAbsIndex(aiCoreIdx - 1, optiling::GS1_END_PTR_INDEX));
        constInfo.s2Start = metadataGm.GetValue(optiling::GetAICMetaAbsIndex(aiCoreIdx - 1, optiling::S2_END_PTR));
    }
    // 任务结束位置
    constInfo.bN2End = metadataGm.GetValue(optiling::GetAICMetaAbsIndex(aiCoreIdx, optiling::BN2_END_PTR_INDEX));
    constInfo.gS1End = metadataGm.GetValue(optiling::GetAICMetaAbsIndex(aiCoreIdx, optiling::GS1_END_PTR_INDEX));
    constInfo.s2End = metadataGm.GetValue(optiling::GetAICMetaAbsIndex(aiCoreIdx, optiling::S2_END_PTR));

    // 首个S1G块、最后一个S1G块的S2是否被切分，在sparse模式下，需要和mask之后的起止点对比才能确定
    constInfo.headS2Split = false;
    constInfo.tailS2Split = false;

    // xxx
    constInfo.coreStartKVSplitPos =
        metadataGm.GetValue(optiling::GetAICMetaAbsIndex(aiCoreIdx, optiling::S2_SPLIT_START_IDX_OF_CORE_INDEX));
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::CalcParams(
    uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfo &info)
{
    info.loop = loop;
    info.isValid = true; // info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;
    info.bIdx = GetBIdx(bN2Cur);
    info.n2Idx = GetN2Idx(bN2Cur);
    info.gS1Idx = gS1Cur * constInfo.mBaseSize;
    info.s2Idx = s2Cur;

    info.actS1Size = actSeqLensQ;
    info.actS2Size = actSeqLensKv;

    info.actMBaseSize = constInfo.mBaseSize;
    // 计算实际基本块size
    uint64_t gS1Size = info.actS1Size * constInfo.gSize;
    if (((gS1Cur + 1) * constInfo.mBaseSize) > gS1Size) {
        info.actMBaseSize = gS1Size - gS1Cur * constInfo.mBaseSize;
    }

    info.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    if (((s2Cur + 1) * constInfo.s2BaseSize) > info.actS2Size) {
        info.actualSingleProcessSInnerSize = info.actS2Size - s2Cur * constInfo.s2BaseSize;
    }
    info.actualSingleProcessSInnerSizeAlign =
        Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK);

    // 命名修改为isUpdateKV
    if (constInfo.batchContinuous) {
        info.isChangeBatch = false;
    } else {
        if (loop == 0) { // 第一个有效任务才需要重置KV的tensor
            info.isChangeBatch = true;
        } else {
            info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
        }
    }
    uint32_t sinkNumber = (constInfo.sinkNumber > 0) ? constInfo.sinkNumber : constInfo.keySinkNumber;
    int64_t safePreToken = constInfo.preToken;
    int64_t safeNextToken = constInfo.nextToken;
    AiInfraInferenceCommonFaBaseVector::GetSafeActToken(
        info.actS1Size, info.actS2Size, safePreToken, safeNextToken, constInfo.sparseMode);

    if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND) {
        info.preTokensPerBatch = safePreToken;
        info.nextTokensPerBatch =
            static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) + safeNextToken;
        if (sinkNumber > 0) {
            info.nextTokensPerBatch = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(sinkNumber) -
                                      static_cast<int32_t>(info.actS1Size) + static_cast<int32_t>(safeNextToken);
        }
    } else if ((constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::DEFAULT_MASK) && constInfo.attenMaskFlag) {
        info.nextTokensPerBatch = safeNextToken;
        info.preTokensPerBatch =
            static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) + safePreToken;
    } else {
        info.nextTokensPerBatch = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size);
        info.preTokensPerBatch = 0;
        if (sinkNumber > 0) {
            if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::RIGHT_DOWN_CAUSAL &&
                static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(sinkNumber) <
                    static_cast<int32_t>(info.actS1Size)) {
                info.nextTokensPerBatch = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(sinkNumber) -
                                          static_cast<int32_t>(info.actS1Size);
            } else if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::LEFT_UP_CAUSAL) {
                info.nextTokensPerBatch = safeNextToken;
            }
        }
    }
    // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
    // 情况2: loop=0时, 如果(bN2Start, gS1Start, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
    // 情况3: loop=0时, 如果(bN2Start, gS1Start, s2Start)任务无效, 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
    info.isFirstSInnerLoop = ((loop == 0) || (s2Cur == curS2Start));
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    info.tndIsS2SplitCore = false;
    info.tndCoreStartKVSplitPos = 0;
    info.isLastS2Loop = (s2Cur + 1 == curS2End);
    info.curSInnerLoopTimes = curS2End - curS2Start;
    if (constInfo.bN2Start == constInfo.bN2End && constInfo.gS1Start == constInfo.gS1End) {
        // 所有任务属于同一个S1G
        info.tndIsS2SplitCore = true;
        info.tndCoreStartKVSplitPos = constInfo.coreStartKVSplitPos;
    } else {
        if (constInfo.headS2Split && (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1Start)) {
            // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
            info.tndIsS2SplitCore = true;
            info.tndCoreStartKVSplitPos = constInfo.coreStartKVSplitPos;
        } else if (constInfo.tailS2Split && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1End)) {
            // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
            info.tndIsS2SplitCore = true;
        }
    }

    if (sinkNumber > 0 && constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND) {
        if constexpr (FLASH_DECODE) {
            if (info.tndIsS2SplitCore) { // sinkNumber性能优化场景，sparseMode4中间块跳过不算，但需要将LSE拷贝出去
                uint32_t sinkBlockLastIndex =
                    (static_cast<uint32_t>(sinkNumber) + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize - 1;
                if (s2Cur == sinkBlockLastIndex && sinkBlockLastIndex + 1 < firstTokenWithSinkNumber &&
                    curS2End <= firstTokenWithSinkNumber) {
                    info.isLastS2Loop |= true;
                }
            }
        }
    }
    uint64_t sInnerOffsetDataSize = info.s2Idx * constInfo.s2BaseSize;

    info.s2BatchOffset = sInnerOffsetDataSize;
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::ComputeMm1(const RunInfo &info)
{
    matmulService.ComputeMm1(info);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::ComputeMm2(const RunInfo &info)
{
    matmulService.ComputeMm2(info);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::ComputeVec1(const RunInfo &info)
{
    vectorService.ProcessVec1L(info);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::ComputeVec2(const RunInfo &info)
{
    vectorService.ProcessVec2L(info);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::FlashDecode()
{
    const uint32_t usedVecNumOfFd =
        metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::USED_VEC_NUM_OF_FD_INDEX));
    if (tmpBlockIdx < usedVecNumOfFd) {
        FDparams fdParams;
        fdService.InitBuffers(pipe);
        AscendC::ICachePreLoad(fdPrefetchLen);
        const uint32_t gS1BaseSizeOfFd =
            metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::M_FD_BASE_SIZE_INDEX));
        uint32_t bN2IdxOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};
        uint32_t gS1IdxOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};
        uint32_t s2SplitNumOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};
        uint32_t gS1SplitNumOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};
        uint32_t gS1LastPartSizeOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};

        for (uint32_t i = 0; i < usedCoreNum; ++i) {
            bN2IdxOfFdHead_local[i] =
                metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::BN2_IDX_OF_FD_HEAD_INDEX));
            gS1IdxOfFdHead_local[i] =
                metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::GS1_IDX_OF_FD_HEAD_INDEX));
            s2SplitNumOfFdHead_local[i] =
                metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::S2_SPLIT_NUM_OF_FD_HEAD_INDEX));
            gS1SplitNumOfFdHead_local[i] =
                metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::GS1_SPLIT_NUM_OF_FD_HEAD_INDEX));
            gS1LastPartSizeOfFdHead_local[i] =
                metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::GS1_LAST_PART_SIZE_OF_FD_HEAD_INDEX));
        }

        uint32_t gS1IdxEndOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM * 2] = {0};
        uint32_t gS1IdxEndOfFdHeadSplit_local[optiling::FIA_MAX_AIC_CORE_NUM * 2] = {0};

        for (uint32_t i = 0; i < usedCoreNum * 2; ++i) {
            gS1IdxEndOfFdHead_local[i] =
                metadataGm.GetValue(optiling::GetAIVMetaAbsIndex(i, optiling::GS1_IDX_END_OF_FD_HEAD_INDEX));
            gS1IdxEndOfFdHeadSplit_local[i] =
                metadataGm.GetValue(optiling::GetAIVMetaAbsIndex(i, optiling::GS1_IDX_END_OF_FD_HEAD_SPLIT_INDEX));
        }

        fdParams = {bN2IdxOfFdHead_local,
                    gS1IdxOfFdHead_local,
                    s2SplitNumOfFdHead_local,
                    gS1SplitNumOfFdHead_local,
                    gS1LastPartSizeOfFdHead_local,
                    gS1IdxEndOfFdHead_local,
                    gS1IdxEndOfFdHeadSplit_local,
                    usedVecNumOfFd,
                    gS1BaseSizeOfFd};
        SyncAll();

        fdService.AllocEventID();
        fdService.InitDecodeParams();
        fdService.FlashDecode(fdParams);
        fdService.FreeEventID();
    } else {
        // superkernel 场景，启动核数大于实际运行核数时，未启动的核仅需要保留 SyncAll
        SyncAll();
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::Process()
{
    // usedCoreNum: 使用的总核数
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
            vectorService.InitBuffers(pipe);
            vectorService.AllocEventID();
        } else {
            matmulService.InitBuffers(pipe);
            matmulService.AllocEventID();
        }
        FlashAttention();

        if ASCEND_IS_AIV {
            vectorService.FreeEventID();
        } else {
            matmulService.FreeEventID();
        }
    }

    if constexpr (FLASH_DECODE) {
        if ASCEND_IS_AIV {
            if (fdFlag) {
                FlashDecode();
            }
        }
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline uint32_t
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::GetBIdx(uint32_t bN2Idx)
{
    return (bN2Idx / constInfo.kvHeadNum);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline uint32_t
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::GetN2Idx(uint32_t bN2Idx)
{
    return (bN2Idx % constInfo.kvHeadNum);
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline bool FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::ShouldDispatchTask(
    uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
{
    return ((bN2Cur != constInfo.bN2End) || (gS1Cur != constInfo.gS1End) || (s2Cur != constInfo.s2End));
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::DealZeroActSeqLen(
    uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
{
    // 对整个batch的结果置0
    if constexpr (POST_QUANT) { // out int8
    } else {
        uint32_t n2Idx = GetN2Idx(bN2Cur);
        uint32_t bIdx = GetBIdx(bN2Cur);
        if (constInfo.outputLayout == FIA_LAYOUT::BSND || constInfo.outputLayout == FIA_LAYOUT::BSH) {
            OffsetCalculator<GmFormat::BSNGD> offsetCalculator;
            offsetCalculator.Init(constInfo.batchSize,
                                  constInfo.kvHeadNum,
                                  constInfo.gSize,
                                  constInfo.qSeqSize,
                                  constInfo.headDim,
                                  actualSeqLengthsGmQ,
                                  constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::BSNGD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
            OffsetCalculator<GmFormat::BNGSD> offsetCalculator;
            offsetCalculator.Init(constInfo.batchSize,
                                  constInfo.kvHeadNum,
                                  constInfo.gSize,
                                  constInfo.qSeqSize,
                                  constInfo.headDim,
                                  actualSeqLengthsGmQ,
                                  constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::BNGSD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
            OffsetCalculator<GmFormat::NGBSD> offsetCalculator;
            offsetCalculator.Init(constInfo.batchSize,
                                  constInfo.kvHeadNum,
                                  constInfo.gSize,
                                  constInfo.qSeqSize,
                                  constInfo.headDim,
                                  actualSeqLengthsGmQ,
                                  constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::NGBSD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
            OffsetCalculator<GmFormat::TNGD> offsetCalculator;
            offsetCalculator.Init(
                constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims);
            DealActSeqLenIsZero<GmFormat::TNGD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
            OffsetCalculator<GmFormat::NGTD> offsetCalculator;
            offsetCalculator.Init(constInfo.kvHeadNum,
                                  constInfo.gSize,
                                  constInfo.headDim,
                                  actualSeqLengthsGmQ,
                                  constInfo.actualLenQDims,
                                  constInfo.qTSize);
            DealActSeqLenIsZero<GmFormat::NGTD, OUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        }
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::UpdateAxisInfo(
    uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
{
    uint64_t s2LoopTimes = (actSeqLensKv + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
    uint64_t gS1LoopTimes = (gS1Size + constInfo.mBaseSize - 1) / constInfo.mBaseSize;

    // 当前S2未处理完
    if (s2Cur + 1 < s2LoopTimes) {
        s2Cur++;
        return;
    }

    // 当前BN2未处理完
    s2Cur = 0;
    if (gS1Cur + 1 < gS1LoopTimes) {
        gS1Cur++;
        return;
    }

    // 当前BN2已处理完
    gS1Cur = 0;
    bN2Cur++;
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::CreateTask(
    uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % FIA_PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, bN2Cur, gS1Cur, s2Cur, extraInfo0);
    extraInfo0.isValid = true;
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline bool FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::ShouldExecuteTask(
    RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE])
{
    for (uint32_t i = 0; i < FIA_PRELOAD_TASK_CACHE_SIZE; i++) {
        if (extraInfo[i].isValid) {
            return true;
        }
    }
    return false;
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::ExecuteTask(
    uint64_t loop, RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % FIA_PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % FIA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0);
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec1(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            ComputeVec2(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline TASK_DEAL_MODE FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::GetTaskDealMode(
    uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
{
    bool isFirstTask = (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1Start) && (s2Cur == constInfo.s2Start);
    uint32_t bIdx = GetBIdx(bN2Cur);
    if (isFirstTask || prevBIdx != bIdx) {
        prevBIdx = bIdx;
        if (constInfo.actualLenDims == 0 && !constInfo.batchContinuous) {
            actSeqLensKv = constInfo.keySinkNumber
                         + FusedInferAttentionScoreV2SinkBaseKernel::SeqLenFromTensorList<LAYOUT_T>(keyPtr, bIdx);
        } else {
            actSeqLensKv = constInfo.keySinkNumber + kvActSeqLensParser.GetActualSeqLength(bIdx);
        }
        if (actSeqLensKv == constInfo.keySinkNumber) {
            actSeqLensKv = actSeqLensKv - constInfo.keySinkNumber;
        }
        actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
    }

    uint64_t s2LoopTimes = (actSeqLensKv + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
    uint64_t gS1LoopTimes = (gS1Size + constInfo.mBaseSize - 1) / constInfo.mBaseSize;

    if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
        if (gS1Cur == 0 && s2Cur == 0) {
            return TASK_DEAL_MODE::DEAL_ZERO;
        }
        return TASK_DEAL_MODE::SKIP;
    }

    if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
        if (constInfo.attenMaskFlag == 0U) {
            CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
        } else {
            CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
        }
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;
    }

    if (s2Cur < curS2Start || s2Cur >= curS2End) {
        return TASK_DEAL_MODE::SKIP;
    }
    uint32_t sinkNumber = (constInfo.sinkNumber > 0) ? constInfo.sinkNumber : constInfo.keySinkNumber;
    if (likely(sinkNumber > 0 && constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND)) {
        uint32_t sinkBlockLastIndex =
            (static_cast<uint32_t>(sinkNumber) + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize - 1;
        if (s2Cur > sinkBlockLastIndex && s2Cur < firstTokenWithSinkNumber) {
            return TASK_DEAL_MODE::SKIP;
        }
    }

    return TASK_DEAL_MODE::CREATE_TASK;
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::GetPreNextTokenLeftUp(
    int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp)
{
    preTokenLeftUp = constInfo.preToken;
    nextTokenLeftUp = constInfo.nextToken;
    AiInfraInferenceCommonFaBaseVector::GetSafeActToken(
        actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp, constInfo.sparseMode);

    if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND) {
        preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
    }

    if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::RIGHT_DOWN_CAUSAL) {
        nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
    } else if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND) {
        nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::CalcCurS2StartEndNoSparse(uint32_t bN2Cur,
                                                                                                uint32_t gS1Cur)
{
    curS2Start = 0U;
    curS2End = (static_cast<uint32_t>(actSeqLensKv) + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;

    if ((bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1Start)) {
        constInfo.headS2Split = constInfo.s2Start != 0U;
        curS2Start = constInfo.s2Start;
    }

    if ((bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1End)) {
        constInfo.tailS2Split = constInfo.s2End != 0U;
        curS2End = constInfo.s2End;
    }
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::CalcCurS2StartEndWithSparse(uint32_t bN2Cur,
                                                                                                  uint32_t gS1Cur)
{
    // 1. Calc preTokenLeftUp, nextTokenLeftUp
    int64_t preTokenLeftUp = 0;
    int64_t nextTokenLeftUp = 0;
    GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

    // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(constInfo.mBaseSize);
    int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(constInfo.mBaseSize),
                               static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.gSize)) -
                           1;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if constexpr (GetOutUbFormat<LAYOUT_T>() == UbFormat::S1G) {
        s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
        s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.gSize);
    } else {
        if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
            s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
        } else {
            // start and end locate in tow or more G, but working same as crossing one complete block
            s1FirstToken = 0;
            s1LastToken = static_cast<int64_t>(actSeqLensQ);
        }
    }

    // 3. trans index of token to index of block
    uint32_t s2StartWithSparse = 0U;
    uint32_t s2EndWithSparse = 0U;
    int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
    // no valid token
    if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        curS2Start = 0U;
        curS2End = 0U;
        return;
    }
    // get valid range
    s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
    s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

    s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / constInfo.s2BaseSize;
    s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / constInfo.s2BaseSize + 1U;

    // 增加sink_num场景sparseMode=4时，从零开始
    firstTokenWithSinkNumber = s2StartWithSparse;
    uint32_t sinkNumber = (constInfo.sinkNumber > 0) ? constInfo.sinkNumber : constInfo.keySinkNumber;
    if (sinkNumber > 0 && constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND) {
        s2StartWithSparse = 0;
    }

    // 4. Calc curS2Start, curS2End
    curS2Start = s2StartWithSparse;
    curS2End = s2EndWithSparse;

    if (bN2Cur == constInfo.bN2Start && gS1Cur == constInfo.gS1Start) { // first line
        constInfo.headS2Split = constInfo.s2Start > s2StartWithSparse ? true : false;
        curS2Start = Max(s2StartWithSparse, constInfo.s2Start);
        firstTokenWithSinkNumber = Max(firstTokenWithSinkNumber, constInfo.s2Start);
    }
    if (bN2Cur == constInfo.bN2End && gS1Cur == constInfo.gS1End) { // last line
        constInfo.tailS2Split = constInfo.s2End > 0U ? true : false;
        curS2End = constInfo.s2End > 0U ? Min(s2EndWithSparse, constInfo.s2End) : s2EndWithSparse;
    }
    return;
}

template <typename FIAT, typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FiaKernelNonQuantMla<FIAT, CubeBlockType, VecBlockType, FdBlockType>::FlashAttention()
{
    RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE];

    uint32_t bN2Cur = constInfo.bN2Start;
    uint32_t gS1Cur = constInfo.gS1Start;
    uint32_t s2Cur = constInfo.s2Start;
    prevBIdx = GetBIdx(bN2Cur);
    prevBN2Idx = bN2Cur;
    prevGS1Idx = gS1Cur;

    uint64_t createdTaskCount = 0;
    uint64_t executedTaskCount = 0;

    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;
    uint32_t WholeCount = 0;

    if (constInfo.batchInvariant) {
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV2C2);
        }
    }
    while (shouldDispatchTask || shouldExecuteTask) {
        WholeCount++;
        // 分发任务
        shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
            if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                // 创建任务
                CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, extraInfo);
                createdTaskCount++;
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
            } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                DealZeroActSeqLen(bN2Cur, gS1Cur, s2Cur);
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                continue;
            } else if (taskDealMode == TASK_DEAL_MODE::SKIP) {
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                continue;
            }
        }
        // 执行任务
        shouldExecuteTask = ShouldExecuteTask(extraInfo);
        if (shouldExecuteTask) {
            ExecuteTask(executedTaskCount, extraInfo);
            executedTaskCount++;
        }
    }

    if (constInfo.batchInvariant) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(constInfo.syncV2C2);
        }
    }
}
#endif // FIA_KERNEL_NONQUANT_MLA_SINK_H
