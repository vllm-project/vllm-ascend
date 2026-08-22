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
 * \file incre_flash_attention_antiquant_Bbn2s2_Us2.h
 * \brief
 */

#ifndef INCRE_FLASH_ATTENTION_ANTIQUANT_BBN2S2_US2_H
#define INCRE_FLASH_ATTENTION_ANTIQUANT_BBN2S2_US2_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "../incre_flash_attention_pub.h"

#include "../service/mm1_processor.h"
#include "../service/mm2_processor.h"
#include "../service/antiquant_processor.h"
#include "../service/vec1_processor.h"
#include "../service/vec2_processor.h"
#include "../service/antiquant_preprocessor_q.h"
#include "../incre_flash_attention_tiling_regbase.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::Nd2NzParams;

// BUFFER_NUM 是指K Buffer流水方案中每个模块的重复执行次数
#define BUFFER_NUM 2

#define KV_INPUT_QUEUE_SIZE  BUFFER_SIZE_BYTE_16K
struct ExtraInfo {
    uint64_t bIdx = 0;
    uint64_t n1Idx = 0;
    uint64_t n2Idx = 0;
    uint64_t s2Idx = 0;
    uint64_t s1Idx = 0; // PFA sOuterIdx or IFA gIdx
    uint32_t bnIdxInCurCore = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t antiqKeyParamOffset = 0;
    uint64_t antiqValueParamOffset = 0;
    uint32_t gDealSize = 0;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint64_t attenMaskOffset = 0;
    uint64_t attenMaskOffsetPre = 0;
    uint64_t attenMaskCoreOffset = 0;
    uint64_t pseShiftOffset;
    uint64_t pseShiftCoreOffset;
    bool isFirstSInnerLoop = false;
    uint8_t keyAntiquantResScmIdx = 0;
    uint8_t valueAntiquantResScmIdx = 0;
    uint64_t s2BatchOffset = 0;  // For Flash Decode
    uint32_t flashDecodeS2Idx;
    uint32_t sInnerLoopTimes;
    uint64_t kvPaddingBeginOffset;
    uint64_t qPaddingBeginOffset;
    uint32_t sInnerStartIndex = 0;
};

struct TaskContext {
    uint64_t bIdx = 0;
    uint64_t nIdx = 0; // PFA指n1, IFA指n2
    uint64_t s2Idx = 0;
    uint64_t s1Idx = 0; // PFA sOuterIdx or IFA gIdx
};

template <typename IFAT>
class IncreFlashAttentionAntiqSplitBbn2s2Us2
{
public:
    __aicore__ inline IncreFlashAttentionAntiqSplitBbn2s2Us2(
        const optiling::IncreFlashAttentionTilingDataRegbase& __restrict tilingData);
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
                                __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
                                __gm__ uint8_t* blockTable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                const optiling::IncreFlashAttentionTilingDataRegbase* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void InitQuant(__gm__ uint8_t* deqScale1, __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2,
                                     __gm__ uint8_t* quantScale2, __gm__ uint8_t* quantOffset2,
                                     __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
                                     __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
                                     __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
                                     __gm__ uint8_t* workspace);
    __aicore__ inline void Process();

    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool KV_CONTINUOUS = IFAT::kvContinuous;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr INPUTKVTYPE KVTYPE_T = IFAT::inputKVType;
    static constexpr IFAProfile PROFILE = IFAT::profile;
    static constexpr bool ATTEN_MASK = IFAT::attenMask;
    static constexpr bool PSE_SHIFT = IFAT::pseShift;
    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    // 目前只有FP4类型是pertoken伪量化，antiquant参数类型是fp16
    static constexpr bool KVFP4 = (IsSameType<KV_T, fp4x2_e1m2_t>::value || IsSameType<KV_T, fp4x2_e2m1_t>::value);

    // 目前只有FP4支持per group,所以使用数据类型判断
    static constexpr bool ANTIQUANT_PER_GROUP = (IsSameType<KV_T, fp4x2_e1m2_t>::value
        || IsSameType<KV_T, fp4x2_e2m1_t>::value);

    static constexpr bool PAGE_ATTENTION_ANTIQUANT = (IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_PAGE_ATTENTION
        || IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_HEAD_PAGE_ATTENTION
    );
    static constexpr bool KEY_ANTIQUANT_PER_TOKEN = (IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN
        || IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_PAGE_ATTENTION 
        || IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_HEAD_PAGE_ATTENTION
    );
    static constexpr bool VALUE_ANTIQUANT_PER_TOKEN = (IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN 
        || IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_PAGE_ATTENTION
        || IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_HEAD_PAGE_ATTENTION 
        || IFAT::antiquantMode == ANTIQUANTMODE::K_PER_CHANNEL_V_PER_TOKEN // split antiquant mode
    );
    using KEY_ANTIQ_PARAMS_T = typename std::conditional<KEY_ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    using VALUE_ANTIQ_PARAMS_T = typename std::conditional<VALUE_ANTIQUANT_PER_TOKEN, T, Q_T>::type;

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    // Due to UB capacity limit, can only apply 3 KV_INPUT_BUFFER when D is 512
    static constexpr uint8_t KV_INPUT_BUFFER_NUM = (PROFILE.D == 512) ? 3 : 4;
    static constexpr uint32_t QUERY_QUEUE_SIZE = (PROFILE.D == 64) ? BUFFER_SIZE_BYTE_2K : BUFFER_SIZE_BYTE_4K;
    static constexpr uint32_t ANTIQ_PARAM_QUEUE_SIZE = (PROFILE.D == 64) ? BUFFER_SIZE_BYTE_2K : BUFFER_SIZE_BYTE_1K;

    static constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
    static constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
    static constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
    static constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
    static constexpr uint32_t SPARSE_MODE_BAND = 4;
    static constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

    using MM_OUT_T = T;
    using MM_IN_T = Q_T;

    using mm1AType =
        MatmulType<TPosition::TSCM, CubeFormat::NZ, MM_IN_T, false, LayoutMode::NONE, false, TPosition::VECOUT>;
    using mm1BType =
        MatmulType<TPosition::TSCM, CubeFormat::NZ, MM_IN_T, true, LayoutMode::NONE, false, TPosition::VECOUT>;
    using mm1BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using mm1CType = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, MM_OUT_T>;

    static constexpr auto mm1Tiling =
        GetMatmulApiTiling<mm1AType, mm1BType, mm1CType, mm1BiasType>(GenConfMM1(PROFILE));

    Matmul<mm1AType, mm1BType, mm1CType, mm1BiasType, mm1Tiling> mm;
    using mm2AType =
        MatmulType<TPosition::TSCM, CubeFormat::NZ, MM_IN_T, false, LayoutMode::NONE, false, TPosition::VECOUT>;
    using mm2BType =
        MatmulType<TPosition::TSCM, CubeFormat::NZ, MM_IN_T, false, LayoutMode::NONE, false, TPosition::VECOUT>;
    using mm2BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using mm2CType = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, MM_OUT_T>;
    using PSE_SHIFT_T = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    static constexpr auto mm2Tiling =
        GetMatmulApiTiling<mm2AType, mm2BType, mm2CType, mm2BiasType>(GenConfMM2(PROFILE));

    Matmul<mm2AType, mm2BType, mm2CType, mm2BiasType, mm2Tiling> bmm2;
    using mm1Type = Matmul<mm1AType, mm1BType, mm1CType, mm1BiasType, mm1Tiling>;
    Mm1Processor<IFAT, mm1Type> mm1Processor;
    using mm2Type = Matmul<mm2AType, mm2BType, mm2CType, mm2BiasType, mm2Tiling>;
    Mm2Processor<IFAT, mm2Type> mm2Processor;

    AntiquantProcessor<IFAT, KEY_ANTIQUANT_PER_TOKEN> keyAntiquantProcessor;
    AntiquantProcessor<IFAT, VALUE_ANTIQUANT_PER_TOKEN> valueAntiquantProcessor;
    AntiQuantQPreprocessor<IFAT> antiQuantQPreprocessor;
    Vec1Processor<IFAT> vec1Processor;
    Vec2Processor<IFAT> vec2Processor;

protected:
    const optiling::IncreFlashAttentionTilingDataRegbase* __restrict tilingData;
    TPipe* pipe;

    __gm__ uint8_t* keyPtr;
    __gm__ uint8_t* valuePtr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;
    GlobalTensor<uint8_t> attenMaskGm;
    GlobalTensor<PSE_SHIFT_T> pseShiftGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<int32_t> blockTableGm;

    // antiquant
    GlobalTensor<KEY_ANTIQ_PARAMS_T> keyAntiquantOffsetGm;
    GlobalTensor<KEY_ANTIQ_PARAMS_T> keyAntiqScaleGm;
    GlobalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiquantOffsetGm;
    GlobalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiqScaleGm;

    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<uint64_t> actualSeqLengthsQGm;
    GlobalTensor<int64_t> queryPaddingSizeGm;
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // tilingData
    const uint32_t& batchSize;
    const uint32_t& kvHeadNum;
    const uint32_t& qHeadNum;
    const uint32_t& gSize;
    const uint32_t& headNumRatio;
    const uint32_t& headDim;
    const uint32_t& kvSeqSize;
    const uint32_t& qSeqSize;
    const uint32_t& actualLenDims;
    const uint32_t& actualLenQDims;

    uint32_t curNIdx = 0;
    uint32_t curBIdx = 0;
    uint32_t curS1Idx = 0;
    uint64_t curActualSeqLen;
    uint64_t curQueryActualSeqLen;
    int64_t remainQSeqLen;
    int64_t remainSeqLen;
    uint64_t kvPaddingBeginOffset = 0;
    uint64_t qPaddingBeginOffset = 0;
    bool isPFABSH = false; // PFA的Q需要区分BSH与BNSD
    uint32_t headNum;
    uint32_t sInnerStartIndex = 0;

    const uint32_t& batchContinuous;
    const uint32_t& singleProcessSInnerSize;

    uint32_t sInnerLoopTimes;
    uint32_t sOuterLoopEnd;
    uint32_t sOuterLoopIdx;
    uint32_t sInnerLoopSize;
    uint32_t actualSInnerLoopSize;
    uint32_t singleProcessSInnerSizeTail;

    const uint32_t& blockSplitBn2Range;
    const uint32_t& tailBlockSplitBn2Range;
    const uint32_t& usedCoreNum;
    const uint32_t& formerCoreNum;
    const uint32_t* coreStartIdx;
    const uint32_t* coreSposStartIdx;
    const uint32_t& splitKVNum;
    const uint32_t& antiquantPerTensorFlag;
    const uint32_t& antiquantPerHeadFlag;

    const uint32_t& attenMaskBatch;
    const uint32_t& attenMaskQSize;
    const uint32_t& attenMaskKVSize;
    const uint32_t& sparseMode;
    const int32_t& preToken;
    const int32_t& nextToken;
    const uint32_t& isRowInvalid;
    const uint32_t& selectWithByteMaskTmpMinSize;

    int64_t preTokenPerBatch = 0;
    int64_t nextTokenPerBatch = 0;
    int64_t nextTokenInvalidOffset = 0;

    const uint32_t& pseShiftB;
    const uint32_t& pseShiftS;
    const uint32_t& pseShiftS0;

    const uint32_t& qPaddingFlag;
    const uint32_t& kvPaddingFlag;
    const uint32_t& isPerChnU8Out;
    const uint32_t& isOutQuantTypeBf16;
    bool softmaxLseFlag = false;

    const uint32_t& antiqSeqSize;

    const uint32_t& paKvShapeType;  // pa场景下kv的shape, 0:(BBH) 1:(BNBD)
    const uint32_t& kvCacheBlockSize = 0;
    const uint32_t& maxBlockNumPerSeq = 0;

    uint32_t tmpBlockIdx;
    uint32_t subBlockIdx;      // aicore内 vector核id

    uint32_t beforeBlockSplitBnNums;
    uint32_t bnNums = 1;
    uint32_t bnIdxInCurCore = 0;
    uint32_t bnLoopTimes = 0;
    bool antiqOffsetExistFlag;
    ExtraInfo extraInfo[BUFFER_NUM]{};
    uint64_t tensorACoreOffset;
    uint64_t tensorBCoreOffset;
    uint64_t tensorAStride;
    uint64_t tensorBStride;
    uint64_t attenOutStride;

    uint64_t headDimAlign;
    uint64_t headDimAlignFp32;
    uint64_t headDimAlignBlock;
    // flash decode
    uint32_t actualCombineLoopSize;
    uint64_t combineLseOffset;
    uint64_t combineAccumOutOffset;
    // InitGm
    bool needInit = false;
    uint32_t singleCoreSize;
    uint32_t singleCoreLseSize;
    int64_t totalOutputSize;
    int64_t totalLseOutputSize;

    bool isBSNDOut = false;
    // UB
    TQue<QuePosition::VECIN, 1> queryInputQue;
    TQue<QuePosition::VECOUT, 1> queryOutputQue;
    TQue<QuePosition::VECIN, 1> kvInputQue;
    TQue<QuePosition::VECOUT, 1> kvOutputQue;
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue;
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;

    TQue<QuePosition::VECIN, 1> keyAntiqScaleInputQue;
    TQue<QuePosition::VECIN, 1> keyAntiqOffsetInputQue;
    TQue<QuePosition::VECIN, 1> valueAntiqScaleInputQue;
    TQue<QuePosition::VECIN, 1> valueAntiqOffsetInputQue;
    TBuf<> kvAntiqMxScaleRes;  // for w4

    TQue<QuePosition::VECIN, 1> maskInputQue;
    TQue<QuePosition::VECIN, 1> pseInputQue;

    TBuf<> entireMmResBuffer;
    LocalTensor<T> mmResultBufferArray[BUFFER_NUM];
    TQue<QuePosition::VECOUT, 1> softmaxResOutputQue;
    TQue<QuePosition::VECOUT, 1> updateResOutputQue;
    TBuf<> softmaxBuffBig;
    TBuf<> lseTmpBuff;

    TQue<QuePosition::VECOUT, 1> softmaxLseOutputQue;
    // L1 Buffer
    TSCM<TPosition::VECIN, 1> kvAntiquantResScm[BUFFER_NUM];
    TSCM<TPosition::VECIN, 1> queryScm[BUFFER_NUM];
    TSCM<TPosition::VECIN, 1> mm2AScm[BUFFER_NUM];
    TSCM<TPosition::VECIN, 1> reservedScm;

    // LocalTensor
    LocalTensor<KEY_ANTIQ_PARAMS_T> keyAntiqScale;
    LocalTensor<KEY_ANTIQ_PARAMS_T> keyAntiqOffset;
    LocalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiqScale;
    LocalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiqOffset;

    LocalTensor<T> softmaxMaxUb[BUFFER_NUM];
    LocalTensor<T> softmaxSumUb[BUFFER_NUM];
    LocalTensor<T> softmaxExpUb[BUFFER_NUM];
    LocalTensor<uint8_t> softmaxTmpUb;
    LocalTensor<MM_OUT_T> bmm2PreResUb;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);

    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitL1Buffers();
    __aicore__ inline void InitFDBuffers();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void SetLoopTimes();
    __aicore__ inline void UpdateSOuterLoopEnd(uint32_t bnIdx);
    __aicore__ inline void UpdateZeroActualSeqLen(uint32_t bIdx, uint64_t& paddingBeginOffset,
                                                  uint32_t& loopTimes, uint64_t& actualSeqLen);
    __aicore__ inline void UpdateZeroActualSeqLenPFA(uint32_t bIdx, uint64_t& paddingBeginOffset,
                                                     uint32_t& loopTimes, uint64_t& actualSeqLen,
                                                     uint64_t s1Idx, int64_t preTokenPerBatch,
                                                     int64_t nextTokenPerBatch, uint32_t& sInnerStartIndex);
    __aicore__ inline void DealActualSeqLenIsZero(uint32_t bnIdx);
    __aicore__ inline void InitAllZeroOutput(uint32_t bnIdx);
    __aicore__ inline void GetQueryActualSeqLen(uint32_t bIdx, uint64_t& qActualSeqLen);
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint64_t& actualSeqLen);
    __aicore__ inline bool ComputePaddingBeginOffset(uint64_t actualSeqLen, uint64_t& paddingBeginOffset,
                                                     bool paddingFlag, int64_t seqLen);
    __aicore__ inline void InitQueryLeftPaddingData();
    __aicore__ inline void InitLeftPaddingData();
    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void InitLseOutputSingleCore();

    __aicore__ inline void GetBNId(uint32_t bnIdx);
    __aicore__ inline void CalcParams(uint32_t loop, TaskContext& task);

    __aicore__ inline void CalcPreNextTokenPerBatch(int64_t& preTokenPerBatch,
        int64_t& nextTokenPerBatch, uint64_t curQueryActualSeqLen, uint64_t curActualSeqLen);
    __aicore__ inline void FixParamWithRowInvalid(int64_t& preTokenPerBatch, int64_t& nextTokenPerBatch,
        int64_t& nextTokenInvalidOffset, uint64_t& curQueryActualSeqLen, uint64_t curActualSeqLen);
    __aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue);
    __aicore__ inline uint64_t CalcAttenMaskCoreOffset(uint64_t bIdx, uint64_t s1Idx,
        uint64_t qPaddingBeginOffset, uint64_t kvPaddingBeginOffset);
    __aicore__ inline uint64_t CalcAttenMaskOffset(uint64_t attenMaskCoreOffset, uint64_t s2Idx, uint64_t s1Idx,
        int64_t nextTokenPerBatch, int64_t nextTokenInvalidOffset);
    __aicore__ inline uint64_t CalcAttenMaskOffsetPre(uint64_t attenMaskCoreOffset, uint64_t s2Idx, uint64_t s1Idx,
        int64_t preTokenPerBatch);

    __aicore__ inline void AntiquantKey(uint32_t loop);
    __aicore__ inline void AntiquantValue(uint32_t loop);
    __aicore__ inline void CopyQueryToL1(uint32_t loop);
    __aicore__ inline void WaitQK(uint32_t loop);
    __aicore__ inline void SendQK(uint32_t loop);
    __aicore__ inline void Vector1(uint32_t loop);
    __aicore__ inline void WaitSV(uint32_t loop);
    __aicore__ inline void SendSV(uint32_t loop);
    __aicore__ inline void Vector2(uint32_t loop);
    __aicore__ inline void Update(uint32_t loop);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T>& attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CopyLseIn(uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(uint64_t attenOutOffset, LocalTensor<T>& accumOutLocal, uint32_t startRow,
                                           uint32_t dealRowCount);
    __aicore__ inline void CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx, LocalTensor<T>& dst, LocalTensor<T>& lseLocal,
                                          uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CombineSplitKVRes(uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx);
    
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, 
                                             uint32_t splitSize, uint64_t lseOffset);
    __aicore__ inline void FlashDecodeCompute();

    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);
};

template <typename IFAT>
__aicore__ inline IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::IncreFlashAttentionAntiqSplitBbn2s2Us2(
    const optiling::IncreFlashAttentionTilingDataRegbase& __restrict tilingData)
    : singleProcessSInnerSize(tilingData.increFlashAttentionSingleCoreParams.singleProcessSInnerSize),
      usedCoreNum(tilingData.increFlashAttentionSingleCoreParams.usedCoreNum),
      formerCoreNum(tilingData.increFlashAttentionSingleCoreParams.formerCoreNum),
      blockSplitBn2Range(tilingData.increFlashAttentionSingleCoreParams.blockSplitBn2Range),
      tailBlockSplitBn2Range(tilingData.increFlashAttentionSingleCoreParams.tailSplitedBatchRange),
      splitKVNum(tilingData.splitKVParams.s2),
      batchSize(tilingData.baseParams.batchSize),
      kvHeadNum(tilingData.baseParams.kvHeadNum),
      qHeadNum(tilingData.baseParams.qHeadNum),
      headNumRatio(tilingData.baseParams.headNumRatio),
      gSize(tilingData.baseParams.nNumOfQInOneGroup),
      qSeqSize(tilingData.baseParams.qSeqSize),
      kvSeqSize(tilingData.baseParams.seqSize),
      actualLenQDims(tilingData.baseParams.actualLenQDims),
      actualLenDims(tilingData.baseParams.actualLenDims),
      headDim(tilingData.baseParams.headSize),
      batchContinuous(tilingData.baseParams.batchContinuousFlag),
      coreStartIdx(tilingData.increFlashAttentionCoreParams.coreSidxEndRegbase),
      coreSposStartIdx(tilingData.increFlashAttentionCoreParams.coreSposStartRegbase),
      // msdIterNum(tilingData.baseParams.msdIterNum),
      antiquantPerTensorFlag(tilingData.baseParams.antiquantPerTensorFlag),
      antiquantPerHeadFlag(tilingData.baseParams.antiquantPerHeadFlag),
      attenMaskBatch(tilingData.baseParams.attenMaskBatch),
      attenMaskQSize(tilingData.baseParams.attenMaskQSize),
      attenMaskKVSize(tilingData.baseParams.attenMaskSize),
      sparseMode(tilingData.baseParams.sparseMode),
      preToken(tilingData.baseParams.preToken),
      nextToken(tilingData.baseParams.nextToken),
      isRowInvalid(tilingData.baseParams.isRowInvalid),
      selectWithByteMaskTmpMinSize(tilingData.baseParams.selectWithByteMaskTmpMinSize),
      pseShiftB(tilingData.baseParams.pseShiftB),
      pseShiftS0(tilingData.baseParams.pseShiftS0),
      pseShiftS(tilingData.baseParams.pseShiftS),
      qPaddingFlag(tilingData.baseParams.qPaddingFlag),
      kvPaddingFlag(tilingData.baseParams.kvPaddingFlag),
      isPerChnU8Out(tilingData.outputParams.isPerChnOut),
      isOutQuantTypeBf16(tilingData.outputParams.isOutQuantTypeBf16),
      antiqSeqSize(tilingData.baseParams.antiqSeqSize),
      paKvShapeType(tilingData.baseParams.paKvShapeType),
      kvCacheBlockSize(tilingData.baseParams.blockSize),
      maxBlockNumPerSeq(tilingData.baseParams.maxBlockNumPerSeq),
      softmaxLseFlag(tilingData.baseParams.softmaxLseFlag),
      totalOutputSize(tilingData.outputParams.totalOutputSize),
      totalLseOutputSize(tilingData.outputParams.totalLseOutputSize),
      singleCoreSize(tilingData.outputParams.singleCoreSize),
      singleCoreLseSize(tilingData.outputParams.singleCoreLseSize),
      needInit(tilingData.outputParams.needInit),
      isBSNDOut(tilingData.outputParams.isBSNDOut)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
    __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* blockTable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    const optiling::IncreFlashAttentionTilingDataRegbase* __restrict tiling, TPipe* tPipe)
{
    tmpBlockIdx = GetBlockIdx();
    subBlockIdx = tmpBlockIdx % 2;

    // Only one vector core use one cube core when B*N number is less than half of core number
    if constexpr (!FLASH_DECODE) {
#ifndef __DAV_310R6__
        if (tmpBlockIdx & 0x1) {
            tmpBlockIdx = (tmpBlockIdx + GetBlockNum() * 2) / 2;
        } else {
            tmpBlockIdx = tmpBlockIdx / 2;
        }
#endif
    }

    pipe = tPipe;
    queryGm.SetGlobalBuffer((__gm__ Q_T*)query);

    keyPtr = key;
    valuePtr = value;

    if (batchContinuous) {
        InitKeyGm(0);
        InitValueGm(0);
    }

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T*)attentionOut);
    if constexpr (ATTEN_MASK) {
        attenMaskGm.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
    }
    if constexpr (PSE_SHIFT) {
        pseShiftGm.SetGlobalBuffer((__gm__ PSE_SHIFT_T*)pseShift);
    }
    if (softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ T*)softmaxLse);
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
    }

    tilingData = tiling;
    InitTilingData();
    InitBuffers();
    InitL1Buffers();

    uint64_t offset = 0;
    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ T*)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(T);
        softmaxMaxGm.SetGlobalBuffer((__gm__ T*)(workspace + offset));
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(T);
        softmaxSumGm.SetGlobalBuffer((__gm__ T*)(workspace + offset));
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(T);
    }
    if constexpr (!FLASH_DECODE) {
        if (actualLenQDims != 0) {
            actualSeqLengthsQGm.SetGlobalBuffer((__gm__ uint64_t*)actualSeqLengthsQ, actualLenQDims);
        }
        if (needInit) {
           InitOutputSingleCore();
           if (softmaxLseFlag) {
            InitLseOutputSingleCore();
           }
        }
        if (qPaddingFlag == 1) {
            queryPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t*) queryPaddingSize);
            InitQueryLeftPaddingData();
        }
    }
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t*)actualSeqLengths, actualLenDims);
    }
    
    if (kvPaddingFlag == 1) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t*) kvPaddingSize);
        InitLeftPaddingData();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitQuant(
    __gm__ uint8_t* deqScale1, __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2, __gm__ uint8_t* quantScale2,
    __gm__ uint8_t* quantOffset2, __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
    __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset, __gm__ uint8_t* valueAntiquantScale,
    __gm__ uint8_t* valueAntiquantOffset, __gm__ uint8_t* workspace)
{
    if constexpr (ANTIQUANT) {
        if (keyAntiquantScale == nullptr) {
            int64_t antiValueOffsetInitPos = kvHeadNum * headDim;
            if constexpr (KVFP4) {
                // perTokenGroup V 的偏移，除以2是因为数是fp8e9m0的，GM是fp16的
                antiValueOffsetInitPos = (uint64_t)batchSize * kvHeadNum * kvSeqSize * headDim / 32 / 2;
            }
            if constexpr (KEY_ANTIQUANT_PER_TOKEN) {
                antiValueOffsetInitPos = (uint64_t)batchSize * antiqSeqSize;
            }
            if (antiquantPerTensorFlag) {
                antiValueOffsetInitPos = 1;
            }
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)antiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer(((__gm__ VALUE_ANTIQ_PARAMS_T*)antiquantScale) + antiValueOffsetInitPos);
            antiqOffsetExistFlag = (antiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiquantOffsetGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)antiquantOffset);
                valueAntiquantOffsetGm.SetGlobalBuffer(((__gm__ VALUE_ANTIQ_PARAMS_T*)antiquantOffset) + antiValueOffsetInitPos);
            }
        } else {
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)keyAntiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer((__gm__ VALUE_ANTIQ_PARAMS_T*)valueAntiquantScale);
            antiqOffsetExistFlag = (keyAntiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiquantOffsetGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)keyAntiquantOffset);
                valueAntiquantOffsetGm.SetGlobalBuffer((__gm__ VALUE_ANTIQ_PARAMS_T*)valueAntiquantOffset);
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitTilingData()
{
    isPFABSH = LAYOUT_T == LAYOUT::BSH && qSeqSize > 1;
    headDimAlignFp32 = ALIGN((uint64_t)headDim, BYTE_BLOCK / sizeof(T));  // 32B对齐
    headDimAlignBlock = ALIGN((uint64_t)headDim, BYTE_BLOCK / sizeof(Q_T));
    tensorAStride = isPFABSH? qHeadNum * headDim : headDim; // IFAqs=1时Q不区分BSH与BNSD
    tensorBStride = (LAYOUT_T == LAYOUT::BNSD) ? headDim : kvHeadNum * headDim;
    attenOutStride = isBSNDOut? qHeadNum * headDim : tensorAStride;
    headNum = qSeqSize > 1 ? qHeadNum : kvHeadNum;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitQueryLeftPaddingData()
{
    int64_t paddingSize = queryPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }
    remainQSeqLen = static_cast<int64_t>(qSeqSize) - paddingSize;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitLeftPaddingData()
{
    int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }
    remainSeqLen = static_cast<int64_t>(kvSeqSize) - paddingSize;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitBuffers()
{
    pipe->InitBuffer(queryInputQue, BUFFER_NUM, QUERY_QUEUE_SIZE);
    pipe->InitBuffer(queryOutputQue, BUFFER_NUM, QUERY_QUEUE_SIZE);
    // kvInputQue有4块16k大小的buffer
    pipe->InitBuffer(kvInputQue, KV_INPUT_BUFFER_NUM, KV_INPUT_QUEUE_SIZE);
    pipe->InitBuffer(kvOutputQue, BUFFER_NUM, BUFFER_SIZE_BYTE_32K + BUFFER_SIZE_BYTE_1K);

    pipe->InitBuffer(keyAntiqScaleInputQue, 1, ANTIQ_PARAM_QUEUE_SIZE);
    pipe->InitBuffer(keyAntiqOffsetInputQue, 1, ANTIQ_PARAM_QUEUE_SIZE);
    pipe->InitBuffer(valueAntiqScaleInputQue, 1, ANTIQ_PARAM_QUEUE_SIZE);
    pipe->InitBuffer(valueAntiqOffsetInputQue, 1, ANTIQ_PARAM_QUEUE_SIZE);

    if constexpr (KVFP4) {
        pipe->InitBuffer(kvAntiqMxScaleRes, BUFFER_SIZE_BYTE_4K);
    }
    if constexpr (ATTEN_MASK) {
        if (qSeqSize > 1) {
            if (sparseMode == SPARSE_MODE_BAND) {
                pipe->InitBuffer(maskInputQue, 1, PROFILE.G * PROFILE.S * 2); // band模式需要2块mask ub
            } else {
                pipe->InitBuffer(maskInputQue, 1, PROFILE.G * PROFILE.S); // 基本块大小 * sizeof(bool/int8/uint8)
            }
        } else {
            pipe->InitBuffer(maskInputQue, 1, BUFFER_SIZE_BYTE_1K); // ATTEN_MASK与G无关，仅需申请 S2*sizeof(uint8_t)即可
        }
    }
    if constexpr (PSE_SHIFT) {
        pipe->InitBuffer(pseInputQue, 1, PROFILE.G * PROFILE.S * sizeof(Q_T));
    }
    // 尽量一次初始化一大块。
    uint32_t maxValue = MAX(PROFILE.S, PROFILE.D);  // bmm1, bmm2复用输出内存, 大小取D分档和S分档中较大的值
    pipe->InitBuffer(entireMmResBuffer, PROFILE.G * maxValue * sizeof(T) * BUFFER_NUM);
    mmResultBufferArray[0] = entireMmResBuffer.GetWithOffset<T>(PROFILE.G * maxValue, 0);
    mmResultBufferArray[1] = entireMmResBuffer.GetWithOffset<T>(PROFILE.G * maxValue, PROFILE.G * maxValue * sizeof(T));

    pipe->InitBuffer(softmaxResOutputQue, 1, (PROFILE.G + 1) * PROFILE.S * sizeof(Q_T));  // 加1是为了解bank冲突
    pipe->InitBuffer(updateResOutputQue, 1, PROFILE.G * PROFILE.D * sizeof(T));
    pipe->InitBuffer(softmaxLseOutputQue, 1, BUFFER_SIZE_BYTE_512B);

    pipe->InitBuffer(softmaxBuffBig, BUFFER_SIZE_BYTE_512B * 7);
    softmaxMaxUb[0] = softmaxBuffBig.GetWithOffset<T>(BUFFER_SIZE_BYTE_512B / sizeof(T), 0);
    softmaxMaxUb[1] = softmaxBuffBig.GetWithOffset<T>(BUFFER_SIZE_BYTE_512B / sizeof(T), BUFFER_SIZE_BYTE_512B);
    softmaxSumUb[0] = softmaxBuffBig.GetWithOffset<T>(BUFFER_SIZE_BYTE_512B / sizeof(T), BUFFER_SIZE_BYTE_512B * 2);
    softmaxSumUb[1] = softmaxBuffBig.GetWithOffset<T>(BUFFER_SIZE_BYTE_512B / sizeof(T), BUFFER_SIZE_BYTE_512B * 3);
    softmaxExpUb[0] = softmaxBuffBig.GetWithOffset<T>(BUFFER_SIZE_BYTE_512B / sizeof(T), BUFFER_SIZE_BYTE_512B * 4);
    softmaxExpUb[1] = softmaxBuffBig.GetWithOffset<T>(BUFFER_SIZE_BYTE_512B / sizeof(T), BUFFER_SIZE_BYTE_512B * 5);
    softmaxTmpUb = softmaxBuffBig.GetWithOffset<uint8_t>(BUFFER_SIZE_BYTE_512B, BUFFER_SIZE_BYTE_512B * 6);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitFDBuffers()
{
    pipe->Reset();
    pipe->InitBuffer(lseTmpBuff, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxMaxInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxSumInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(accumOutInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxLseOutputQue, 1, BUFFER_SIZE_BYTE_512B);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitL1Buffers()
{
    uint32_t kvAntiquantResBytes =
        ALIGN((uint32_t)singleProcessSInnerSize, 16U) * PROFILE.D * sizeof(Q_T);  // 正常64K
    // 当前伪量化模板，左矩阵类型Q_T为fp16或bf16，L1需要16元素对齐（注：D=64且使能pse，PROFILE.G为8）
    uint32_t queryBytes = FP16_ONE_BLOCK_SIZE * PROFILE.D * sizeof(Q_T);
     // 当前伪量化模板，左矩阵类型Q_T为fp16或bf16，L1需要16元素对齐，每行大小是PROFILE.S
    uint32_t mm2ABytes = FP16_ONE_BLOCK_SIZE * PROFILE.S * sizeof(Q_T);

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        pipe->InitBuffer(kvAntiquantResScm[i], 1, kvAntiquantResBytes);
    }
    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        pipe->InitBuffer(queryScm[i], 1, queryBytes);
    }
    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        pipe->InitBuffer(mm2AScm[i], 1, mm2ABytes);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitCalcParamsEach()
{
    bnNums = ((coreStartIdx[tmpBlockIdx + 1] - coreStartIdx[tmpBlockIdx]) > 0 || qSeqSize == 1) ?
       (coreStartIdx[tmpBlockIdx + 1] - coreStartIdx[tmpBlockIdx]) : 1; // PFA需要对Q的S方向分核，多个核会共b/n
    bnNums += (coreSposStartIdx[tmpBlockIdx + 1] != 0 && coreStartIdx[tmpBlockIdx + 1] != coreStartIdx[tmpBlockIdx]) ?
        1 : 0; // PFA分核时会跨b/n取s，这时需要加1
    beforeBlockSplitBnNums = coreStartIdx[tmpBlockIdx];
    curBIdx = beforeBlockSplitBnNums / headNum;
    curNIdx = beforeBlockSplitBnNums % headNum;
    curS1Idx = coreSposStartIdx[tmpBlockIdx];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::DealActualSeqLenIsZero(uint32_t bnIdx)
{
    InitAllZeroOutput(bnIdx);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitAllZeroOutput(uint32_t bnIdx)
{
    uint32_t dealRow = qSeqSize == 1 ? gSize : qSeqSize;
    uint32_t copySize = dealRow * headDim;
    if constexpr (POST_QUANT) {  // out int8
                                 //
    } else {
        matmul::InitOutput<OUT_T>(attentionOutGm[bnIdx * copySize], copySize, 0);
    }

    if (softmaxLseFlag && qSeqSize == 1) {
        LocalTensor<T> lseUb = softmaxLseOutputQue.template AllocTensor<T>();
        Duplicate(lseUb, FLOAT_MAX, gSize);
        softmaxLseOutputQue.template EnQue(lseUb);
        softmaxLseOutputQue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float) * gSize;
        intriParams1.blockCount = 1;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[bnIdx * gSize], lseUb, intriParams1);
        softmaxLseOutputQue.FreeTensor(lseUb);
    }
}

template<typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitOutputSingleCore()
{
    uint32_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
    InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
    SyncAll();
}

template<typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitLseOutputSingleCore()
{
    uint32_t tailSize = totalLseOutputSize - tmpBlockIdx * singleCoreLseSize;
    uint32_t singleInitLseOutputSize = tailSize < singleCoreLseSize ? tailSize : singleCoreLseSize;
    InitOutput<float>(softmaxLseGm[tmpBlockIdx * singleCoreLseSize], singleInitLseOutputSize, 3e+99);
    SyncAll();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::GetQueryActualSeqLen(uint32_t bIdx,
                                                                                     uint64_t& qActualSeqLen)
{
    if (actualLenQDims == 0) {
        qActualSeqLen = qSeqSize;
    } else if (actualLenQDims == 1) {
        qActualSeqLen = actualSeqLengthsQGm.GetValue(0);
    } else {
        qActualSeqLen = actualSeqLengthsQGm.GetValue(bIdx);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::GetActualSeqLen(uint32_t bIdx,
                                                                                     uint64_t& actualSeqLen)
{
    if (actualLenDims == 0) {
        actualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            actualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        actualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        actualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename IFAT>
__aicore__ inline bool IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::ComputePaddingBeginOffset(uint64_t actualSeqLen,
                                                                                                uint64_t& paddingBeginOffset,
                                                                                                bool paddingFlag,
                                                                                                int64_t seqLen)
{
    if (paddingFlag != 1) {
        return true;
    }

    int64_t startPosition = seqLen - static_cast<int64_t>(actualSeqLen);
    
    if (startPosition < 0) {
        return false;
    }

    paddingBeginOffset = static_cast<uint64_t>(startPosition);
    return true;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void*)keyPtr);
    __gm__ uint8_t* key_ = (__gm__ uint8_t*)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
    keyGm.SetGlobalBuffer((__gm__ KV_T*)key_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void*)valuePtr);
    __gm__ uint8_t* value_ = (__gm__ uint8_t*)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
    valueGm.SetGlobalBuffer((__gm__ KV_T*)value_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::SetLoopTimes()
{
    uint32_t gLoopCount = (gSize + PROFILE.G - 1) / PROFILE.G;
    curQueryActualSeqLen = gSize; // IFA的GQA场景的gSize等价于PFA的Qs处理
    sOuterLoopEnd = gLoopCount;
    if constexpr (FLASH_DECODE) {
        GetActualSeqLen(tmpBlockIdx / (kvHeadNum * splitKVNum), curActualSeqLen);
        sInnerLoopSize = (kvSeqSize + splitKVNum - 1) / splitKVNum;
        if constexpr (PAGE_ATTENTION_ANTIQUANT) {  // PA管理伪量化参数时，dstOffset必须32B对齐
            sInnerLoopSize = ALIGN((uint64_t)sInnerLoopSize, BYTE_BLOCK / sizeof(T));
        }
        if (sInnerLoopSize * (tmpBlockIdx % splitKVNum) > curActualSeqLen) {
            actualSInnerLoopSize = 0;
        } else {
            actualSInnerLoopSize = (curActualSeqLen - sInnerLoopSize * (tmpBlockIdx % splitKVNum)) > sInnerLoopSize ? 
                                    sInnerLoopSize : (curActualSeqLen - sInnerLoopSize * (tmpBlockIdx % splitKVNum));
        }
        sInnerLoopTimes = (actualSInnerLoopSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
        bnLoopTimes = 1;
        ComputePaddingBeginOffset(curActualSeqLen, kvPaddingBeginOffset, kvPaddingFlag, remainSeqLen);
    } else {
        for (uint32_t bnIdx = beforeBlockSplitBnNums; bnIdx < beforeBlockSplitBnNums + bnNums; bnIdx++) {
            GetActualSeqLen(bnIdx / headNum, curActualSeqLen);
            if (curActualSeqLen == 0 ||
                !ComputePaddingBeginOffset(curActualSeqLen, kvPaddingBeginOffset, kvPaddingFlag, remainSeqLen)) {
                DealActualSeqLenIsZero(bnIdx);
                continue;
            }
        }
        bnLoopTimes = bnNums;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::SeqLenFromTensorList(uint32_t bIdx)
{
    uint64_t dimInfo[4];  // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void*)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIdx);
    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        return keyTensorDesc.GetShape(2);  // BNSD, idx of s is 2
    } else {
        return keyTensorDesc.GetShape(1);  // BSH, idx of s is 1
    }
}

template<typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CalcPreNextTokenPerBatch(int64_t& preTokenPerBatch,
    int64_t& nextTokenPerBatch, uint64_t curQueryActualSeqLen, uint64_t curActualSeqLen)
{
    preTokenPerBatch = preToken;
    nextTokenPerBatch = nextToken;
    if (sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        preTokenPerBatch = SPARSE_MODE_INT_MAX;
        nextTokenPerBatch = static_cast<int64_t>(curActualSeqLen) - static_cast<int64_t>(curQueryActualSeqLen);
    }
    if (sparseMode == SPARSE_MODE_BAND) {
        preTokenPerBatch = preToken - static_cast<int64_t>(curActualSeqLen) + static_cast<int64_t>(curQueryActualSeqLen);
        nextTokenPerBatch = nextToken + static_cast<int64_t>(curActualSeqLen) - static_cast<int64_t>(curQueryActualSeqLen);
    }
}

template<typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::FixParamWithRowInvalid(int64_t& preTokenPerBatch,
    int64_t& nextTokenPerBatch, int64_t& nextTokenInvalidOffset, uint64_t& curQueryActualSeqLen, uint64_t curActualSeqLen)
{
    int64_t nextTokensError = (nextTokenPerBatch < 0) ? -nextTokenPerBatch : 0;
    nextTokensError = nextTokensError > curQueryActualSeqLen ? curQueryActualSeqLen : nextTokensError;
    int64_t preTokensError = (static_cast<int64_t>(curQueryActualSeqLen) > static_cast<int64_t>(curActualSeqLen) + preTokenPerBatch) ?
        (static_cast<int64_t>(curQueryActualSeqLen) - static_cast<int64_t>(curActualSeqLen) - preTokenPerBatch) : 0;
    preTokensError = preTokensError > curQueryActualSeqLen ? curQueryActualSeqLen : preTokensError;

    nextTokenPerBatch += nextTokensError;
    preTokenPerBatch -= nextTokensError;
    curQueryActualSeqLen -= nextTokensError;
    nextTokenInvalidOffset = nextTokensError;

    curQueryActualSeqLen -= preTokensError;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::UpdateSOuterLoopEnd(uint32_t bnIdx)
{
    if (qSeqSize > 1) { // qSeqSize = 1时为gSize,不需要更新
        GetQueryActualSeqLen(curBIdx, curQueryActualSeqLen);
        GetActualSeqLen(curBIdx, curActualSeqLen);
        if (!ComputePaddingBeginOffset(curQueryActualSeqLen, qPaddingBeginOffset, qPaddingFlag, remainQSeqLen)) {
            sOuterLoopEnd = 0; // q_s - actulen - queryPaddingSize < 0时不做处理，输出全0
        } else {
            CalcPreNextTokenPerBatch(preTokenPerBatch, nextTokenPerBatch, curQueryActualSeqLen, curActualSeqLen);
            FixParamWithRowInvalid(preTokenPerBatch, nextTokenPerBatch, nextTokenInvalidOffset, curQueryActualSeqLen,
                curActualSeqLen); // 修正preTokenPerBatch、nextTokenPerBatch、curQueryActualSeqLen
            if (coreStartIdx[tmpBlockIdx + 1] / qHeadNum != curBIdx || coreStartIdx[tmpBlockIdx + 1] % qHeadNum != curNIdx) {
                sOuterLoopEnd = (curQueryActualSeqLen + PROFILE.G -1) / PROFILE.G;
            } else {
                sOuterLoopEnd = coreSposStartIdx[tmpBlockIdx + 1];
            }
        }
        if (bnIdx > 0) {
            curS1Idx = 0;
        }
    }
}

template<typename IFAT>
__aicore__ inline int64_t IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::ClipSInnerToken(int64_t sInnerToken,
    int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::UpdateZeroActualSeqLenPFA(uint32_t bIdx, uint64_t& paddingBeginOffset,
                                                                                               uint32_t& loopTimes, uint64_t& actualSeqLen,
                                                                                               uint64_t s1Idx, int64_t preTokenPerBatch,
                                                                                               int64_t nextTokenPerBatch, uint32_t& sInnerStartIndex)
{
    GetActualSeqLen(bIdx, actualSeqLen);
    if (!ComputePaddingBeginOffset(actualSeqLen, paddingBeginOffset, kvPaddingFlag, remainSeqLen)) {
        actualSeqLen = 0;
    }
    int64_t sInnerFirstToken = ClipSInnerToken(static_cast<int64_t>(s1Idx * PROFILE.G) -
        preTokenPerBatch, 0, actualSeqLen);
    int64_t sInnerLastToken = ClipSInnerToken(static_cast<int64_t>(s1Idx * PROFILE.G) +
        nextTokenPerBatch + PROFILE.G, 0, actualSeqLen);

    sInnerStartIndex = (uint32_t)sInnerFirstToken / singleProcessSInnerSize;
    loopTimes = ((uint32_t)sInnerLastToken + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::UpdateZeroActualSeqLen(uint32_t bIdx, uint64_t& paddingBeginOffset,
                                                                                            uint32_t& loopTimes, uint64_t& actualSeqLen)
{
    GetActualSeqLen(bIdx, actualSeqLen);
    if (!ComputePaddingBeginOffset(actualSeqLen, paddingBeginOffset, kvPaddingFlag, remainSeqLen)) {
        actualSeqLen = 0;
    }
    loopTimes = (actualSeqLen + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
}

// 此函数是用来获取当前的s2、G、n2、B的index, 其中s1指q的s，s2和n2指kv的s和n
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::GetBNId(uint32_t bnIdx)
{
    if constexpr (FLASH_DECODE) {
        curBIdx = tmpBlockIdx / (kvHeadNum * splitKVNum);
        curNIdx = (tmpBlockIdx / splitKVNum) % kvHeadNum;
    } else {
        uint32_t bnCnt = beforeBlockSplitBnNums + bnIdx;
        curBIdx = bnCnt / headNum;
        curNIdx = bnCnt % headNum;
    }
}

template<typename IFAT> 
__aicore__ inline uint64_t IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CalcAttenMaskCoreOffset(uint64_t bIdx,
    uint64_t s1Idx, uint64_t qPaddingBeginOffset, uint64_t kvPaddingBeginOffset)
{
    if (qSeqSize > 1) {
        uint64_t attenMaskBatchOffset = 0;
        // 是否为多batch
        if (attenMaskBatch != 1) {
            attenMaskBatchOffset = bIdx * (uint64_t)attenMaskKVSize * (uint64_t)attenMaskQSize;
        }
        return (s1Idx * (uint64_t)PROFILE.G + qPaddingBeginOffset) *
            (uint64_t)attenMaskKVSize + kvPaddingBeginOffset + attenMaskBatchOffset;
    } else {
        return bIdx * (uint64_t)attenMaskKVSize + kvPaddingBeginOffset;
    }
}

template<typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CalcAttenMaskOffset(uint64_t attenMaskCoreOffset,
    uint64_t s2Idx, uint64_t s1Idx, int64_t nextTokenPerBatch, int64_t nextTokenInvalidOffset)
{
    if (qSeqSize > 1) {
        uint64_t attenMaskOffset = 0;
        // 2:leftUp mode of sparseMode, 3:rightdown mode of sparseMode, 4:band mode of sparseMode
        if (sparseMode == SPARSE_MODE_LEFT_UP || sparseMode == SPARSE_MODE_RIGHT_DOWN || sparseMode == SPARSE_MODE_BAND) {
            int64_t delta = 0;
            if (sparseMode == SPARSE_MODE_LEFT_UP) {
                delta = static_cast<int64_t>(s1Idx * PROFILE.G) -
                    static_cast<int64_t>(s2Idx * singleProcessSInnerSize) + nextToken;
            } else {
                delta = static_cast<int64_t>(s1Idx * PROFILE.G) -
                    static_cast<int64_t>(s2Idx * singleProcessSInnerSize) + nextTokenPerBatch;
            }

            if (delta < 0) {
                attenMaskOffset = (static_cast<int64_t>(PROFILE.G) + delta) > 0 ? (-delta) : PROFILE.G;
            } else {
                attenMaskOffset = ((static_cast<int64_t>(singleProcessSInnerSize) - delta) > 0 ?
                    delta : static_cast<int64_t>(singleProcessSInnerSize)) * attenMaskKVSize;
            }
        } else {
            attenMaskOffset = attenMaskCoreOffset + nextTokenInvalidOffset * attenMaskKVSize + s2Idx * singleProcessSInnerSize;
        }
        return attenMaskOffset;
    } else {
        return attenMaskCoreOffset + s2Idx * singleProcessSInnerSize;
    }
}

template<typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CalcAttenMaskOffsetPre(uint64_t attenMaskCoreOffset,
    uint64_t s2Idx, uint64_t s1Idx, int64_t preTokenPerBatch)
{
    if (sparseMode != SPARSE_MODE_BAND) {
        return 0;
    }
    uint64_t attenMaskOffsetPre = 0;  
    int64_t delta = static_cast<int64_t>(s1Idx * PROFILE.G) -
        static_cast<int64_t>(s2Idx * singleProcessSInnerSize) - preTokenPerBatch - 1;
    if (delta < 0) {
        attenMaskOffsetPre = (static_cast<int64_t>(PROFILE.G) + delta) > 0 ?
            (-delta) : PROFILE.G;
    } else {
        attenMaskOffsetPre = ((static_cast<int64_t>(singleProcessSInnerSize) - delta) > 0 ?
            delta : singleProcessSInnerSize) * attenMaskKVSize;
    }
    return attenMaskOffsetPre;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CalcParams(uint32_t loop, TaskContext& task)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];
    uint64_t qActualSeqLen = curQueryActualSeqLen;
    uint64_t kvActualSeqLen = curActualSeqLen;
    int64_t preTokenThisBatch = preTokenPerBatch;
    int64_t nextTokenThisBatch = nextTokenPerBatch;
    int64_t nextTokenInvalidOffsetThisBatch = nextTokenInvalidOffset;
    info.bIdx = task.bIdx;
    info.n1Idx = task.nIdx;
    info.n2Idx = task.nIdx;
    info.s1Idx = task.s1Idx;
    info.s2Idx = task.s2Idx;
    info.sInnerLoopTimes = sInnerLoopTimes;
    info.kvPaddingBeginOffset = kvPaddingBeginOffset;
    info.sInnerStartIndex = sInnerStartIndex;
    if constexpr (FLASH_DECODE) {
        info.flashDecodeS2Idx = tmpBlockIdx % splitKVNum;  // 用于偏移
    } else {
        if (qSeqSize > 1) {
            info.qPaddingBeginOffset = qPaddingBeginOffset;
            info.n2Idx = task.nIdx / headNumRatio;
            if (info.bIdx != curBIdx) { // 此loop相关参数是前面缓存的，因此与Batch相关数据可能发生改变需要重新获取
                GetQueryActualSeqLen(info.bIdx, qActualSeqLen);
                GetActualSeqLen(info.bIdx, kvActualSeqLen);
                ComputePaddingBeginOffset(qActualSeqLen, info.qPaddingBeginOffset, qPaddingFlag, remainQSeqLen);
                CalcPreNextTokenPerBatch(preTokenThisBatch, nextTokenThisBatch, qActualSeqLen, kvActualSeqLen);
                FixParamWithRowInvalid(preTokenThisBatch, nextTokenThisBatch, nextTokenInvalidOffsetThisBatch,
                    qActualSeqLen, kvActualSeqLen);
            }
            if (info.bIdx != curBIdx || info.n1Idx != curNIdx || info.s1Idx != sOuterLoopIdx) {
                UpdateZeroActualSeqLenPFA(info.bIdx, info.kvPaddingBeginOffset, info.sInnerLoopTimes, kvActualSeqLen,
                    info.s1Idx, preTokenThisBatch, nextTokenThisBatch, info.sInnerStartIndex);
            }
        } else if (info.bIdx != curBIdx) {
            UpdateZeroActualSeqLen(info.bIdx, info.kvPaddingBeginOffset, info.sInnerLoopTimes, kvActualSeqLen);
        }
    }

    info.isFirstSInnerLoop = (info.s2Idx == info.sInnerStartIndex);
    if (info.isFirstSInnerLoop) {
        bnIdxInCurCore++;
    }

    info.bnIdxInCurCore = bnIdxInCurCore - 1;

    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            tensorACoreOffset = info.bIdx * qHeadNum * headDim * qSeqSize +
                                (info.n1Idx * gSize * qSeqSize + info.qPaddingBeginOffset) * headDim +
                                nextTokenInvalidOffsetThisBatch * headDim;
            tensorBCoreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDim + info.n2Idx * kvSeqSize * headDim +
                                info.kvPaddingBeginOffset * headDim;
            if (!batchContinuous) {
                uint64_t seqSize = SeqLenFromTensorList(info.bIdx);
                tensorBCoreOffset = info.n2Idx * seqSize * headDim;
            }
            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize * headDim;
            }
        } else {
            tensorACoreOffset = (info.bIdx * qSeqSize + info.qPaddingBeginOffset) * qHeadNum * headDim
                                + info.n1Idx * gSize * headDim + nextTokenInvalidOffsetThisBatch * qHeadNum * headDim;
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim +
                                info.kvPaddingBeginOffset * kvHeadNum * headDim;
            if (!batchContinuous) {
                tensorBCoreOffset = info.n2Idx * headDim;
            }
            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize * kvHeadNum * headDim;
            }
        }
    }

    info.tensorAOffset = tensorACoreOffset + info.s1Idx * PROFILE.G * tensorAStride;
    info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * tensorBStride;
    if constexpr (FLASH_DECODE) {
        info.attenOutOffset = tensorACoreOffset;
    } else {
        if (isBSNDOut) {
            info.attenOutOffset = (info.bIdx * qSeqSize + info.qPaddingBeginOffset) * qHeadNum * headDim +
                                   info.n1Idx * gSize * headDim + nextTokenInvalidOffsetThisBatch * qHeadNum * headDim;
        } else {
            info.attenOutOffset = tensorACoreOffset;
        }
    }
    
    info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
    info.antiqKeyParamOffset = info.n2Idx * headDim;
    info.antiqValueParamOffset = info.n2Idx * headDim;
    if constexpr (ATTEN_MASK) {
        info.attenMaskCoreOffset = CalcAttenMaskCoreOffset(info.bIdx, info.s1Idx, info.qPaddingBeginOffset, info.kvPaddingBeginOffset);
    }
    info.s2BatchOffset = singleProcessSInnerSize * info.s2Idx + info.kvPaddingBeginOffset;
    if constexpr (FLASH_DECODE) {
        info.attenMaskCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize;
        info.s2BatchOffset +=
            info.flashDecodeS2Idx * sInnerLoopSize;  // FD场景 计算KV当前Sequence位置，需要加上FD切KV后的偏移量
    }
    if constexpr (ATTEN_MASK) {
        info.attenMaskOffset = CalcAttenMaskOffset(info.attenMaskCoreOffset, info.s2Idx, info.s1Idx, nextTokenThisBatch, nextTokenInvalidOffsetThisBatch);
        info.attenMaskOffsetPre = CalcAttenMaskOffsetPre(info.attenMaskCoreOffset, info.s2Idx, info.s1Idx, preTokenThisBatch);
    }
        if constexpr (PSE_SHIFT) {
        info.pseShiftCoreOffset =
            (pseShiftB == 1)
                ? (info.n1Idx * gSize * pseShiftS0 * pseShiftS + info.kvPaddingBeginOffset + info.qPaddingBeginOffset * pseShiftS)
                : (info.bIdx * qHeadNum * pseShiftS0 * pseShiftS + info.n1Idx * gSize * pseShiftS0 * pseShiftS +
                   info.kvPaddingBeginOffset + info.qPaddingBeginOffset * pseShiftS);
        if constexpr (FLASH_DECODE) {
            info.pseShiftCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize;
        }
        info.pseShiftOffset = info.pseShiftCoreOffset + info.s2Idx * singleProcessSInnerSize +
            nextTokenInvalidOffsetThisBatch * pseShiftS;
    }
    // s不对齐场景需按实际 tail 刷新 singleProcessSInnerSizeTail;
    uint32_t sInnerLoopTimesNoSparse = info.sInnerLoopTimes;
    if constexpr (ATTEN_MASK) {
        if (qSeqSize > 1) {
            if (softmaxLseFlag) {
                info.qPaddingBeginOffset += nextTokenInvalidOffsetThisBatch;
            }
            sInnerLoopTimesNoSparse = (kvActualSeqLen + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
        }
    }
    if (info.s2Idx == sInnerLoopTimesNoSparse - 1) {  // 尾块
        info.actualSingleProcessSInnerSize =
            FLASH_DECODE ? actualSInnerLoopSize - (sInnerLoopTimesNoSparse - 1) * PROFILE.S : kvActualSeqLen - (sInnerLoopTimesNoSparse - 1) * PROFILE.S;
    }
    uint32_t loopCount = (qActualSeqLen + PROFILE.G - 1) / PROFILE.G;
    if (info.s1Idx == (loopCount - 1)) {
        info.gDealSize = qActualSeqLen - (loopCount - 1) * PROFILE.G;
    } else {
        info.gDealSize = PROFILE.G;
    }
    uint32_t kvIdx = loop / BUFFER_NUM * BUFFER_NUM + loop;
    info.keyAntiquantResScmIdx = kvIdx % (BUFFER_NUM);
    info.valueAntiquantResScmIdx = (kvIdx + BUFFER_NUM) % (BUFFER_NUM);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::AntiquantKey(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];
    uint8_t keyAntiquantResScmIdx = info.keyAntiquantResScmIdx;
    uint32_t copySplitS = KV_INPUT_QUEUE_SIZE / sizeof(KV_T) / PROFILE.D;
    uint32_t curSequence = info.s2BatchOffset;

    if (!batchContinuous && (loop == 0 || info.n2Idx == 0)) {
        InitKeyGm(info.bIdx);
    }

    AntiquantTaskParam taskParam;

    taskParam.batchSize = batchSize;
    taskParam.seqSize = kvSeqSize;
    taskParam.kvHeadNum = kvHeadNum;
    taskParam.headDim = headDim;
    taskParam.headDimAlignBlock = headDimAlignBlock;
    taskParam.kvStep = tensorBStride;

    taskParam.kvGmOffset = info.tensorBOffset;
    taskParam.copySplitS = copySplitS;
    taskParam.copyTotalS = info.actualSingleProcessSInnerSize;

    taskParam.s2BatchOffset = curSequence;
    taskParam.isPertensor = antiquantPerTensorFlag;
    taskParam.isPerHead = antiquantPerHeadFlag;
    taskParam.kvCacheBlockSize = kvCacheBlockSize;
    taskParam.maxBlockNumPerSeq = maxBlockNumPerSeq;
    taskParam.paKvShapeType = paKvShapeType;
    taskParam.kvPaddingBeginOffset = info.kvPaddingBeginOffset;

    if constexpr (KVFP4) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = false;
    } else if constexpr (KEY_ANTIQUANT_PER_TOKEN) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = true;
    } else {
        if (antiquantPerTensorFlag) {
            taskParam.isLoadAntiqParam = (loop == 0);
            taskParam.isFreeAntiqParam = (loop == 0);
        } else {
            taskParam.isLoadAntiqParam = info.isFirstSInnerLoop;
            taskParam.isFreeAntiqParam = (info.s2Idx + 1 == info.sInnerLoopTimes);
        }
    }

    taskParam.isExistOffset = antiqOffsetExistFlag;
    taskParam.antiqParamOffset = info.antiqKeyParamOffset;

    taskParam.bIdx = info.bIdx;
    taskParam.n2Idx = info.n2Idx;
    taskParam.s2Idx = info.s2Idx;
    taskParam.singleSInnerSize = singleProcessSInnerSize;
    taskParam.flashDecodeS2Idx = info.flashDecodeS2Idx;
    taskParam.sInnerLoopSize = sInnerLoopSize;
    taskParam.antiqSeqSize = antiqSeqSize;
    keyAntiquantProcessor.Process(kvAntiquantResScm[keyAntiquantResScmIdx], keyGm, keyAntiqScaleGm,
                              keyAntiquantOffsetGm, blockTableGm, kvInputQue, kvOutputQue, keyAntiqScaleInputQue,
                              keyAntiqOffsetInputQue, kvAntiqMxScaleRes, taskParam);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::AntiquantValue(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];
    uint8_t valueAntiquantResScmIdx = info.valueAntiquantResScmIdx;
    uint32_t copySplitS = KV_INPUT_QUEUE_SIZE / sizeof(KV_T) / PROFILE.D;
    uint32_t curSequence = info.s2BatchOffset;

    // tensorlist场景下跨batch时需要更新value的gm地址
    if (!batchContinuous && (loop == 0 || info.n2Idx == 0)) {
        InitValueGm(info.bIdx);
    }

    AntiquantTaskParam taskParam;

    taskParam.batchSize = batchSize;
    taskParam.seqSize = kvSeqSize;
    taskParam.kvHeadNum = kvHeadNum;
    taskParam.headDim = headDim;
    taskParam.headDimAlignBlock = headDimAlignBlock;
    taskParam.kvStep = tensorBStride;

    taskParam.kvGmOffset = info.tensorBOffset;
    taskParam.copySplitS = copySplitS;
    taskParam.copyTotalS = info.actualSingleProcessSInnerSize;

    taskParam.s2BatchOffset = curSequence;
    taskParam.isPertensor = antiquantPerTensorFlag;
    taskParam.isPerHead = antiquantPerHeadFlag;
    taskParam.kvCacheBlockSize = kvCacheBlockSize;
    taskParam.maxBlockNumPerSeq = maxBlockNumPerSeq;
    taskParam.paKvShapeType = paKvShapeType;
    taskParam.kvPaddingBeginOffset = info.kvPaddingBeginOffset;


    if constexpr (VALUE_ANTIQUANT_PER_TOKEN || ANTIQUANT_PER_GROUP) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = true;
    } else {
        if (antiquantPerTensorFlag) {
            taskParam.isLoadAntiqParam = (loop == 0);
            taskParam.isFreeAntiqParam = (loop == 0);
        } else {
            taskParam.isLoadAntiqParam = info.isFirstSInnerLoop;
            taskParam.isFreeAntiqParam = (info.s2Idx + 1 == info.sInnerLoopTimes);
        }
    }

    taskParam.isExistOffset = antiqOffsetExistFlag;
    taskParam.antiqParamOffset = info.antiqValueParamOffset;

    taskParam.bIdx = info.bIdx;
    taskParam.n2Idx = info.n2Idx;
    taskParam.s2Idx = info.s2Idx;
    taskParam.singleSInnerSize = singleProcessSInnerSize;
    taskParam.flashDecodeS2Idx = info.flashDecodeS2Idx;
    taskParam.sInnerLoopSize = sInnerLoopSize;
    taskParam.antiqSeqSize = antiqSeqSize;
    valueAntiquantProcessor.Process(kvAntiquantResScm[valueAntiquantResScmIdx], valueGm,
                                valueAntiqScaleGm, valueAntiquantOffsetGm, blockTableGm, kvInputQue, kvOutputQue,
                                valueAntiqScaleInputQue, valueAntiqOffsetInputQue, kvAntiqMxScaleRes, taskParam);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CopyQueryToL1(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];

    uint32_t queryScmIdx = info.bnIdxInCurCore % BUFFER_NUM;
    AntiquantQPreprocessorTaskParam taskParam;
    taskParam.queryInputQueBufferSize = QUERY_QUEUE_SIZE;
    taskParam.dealRowCount = info.gDealSize;
    taskParam.columnCount = headDim;
    taskParam.tensorAOffset = info.tensorAOffset;
    taskParam.tensorAStride = tensorAStride;
    taskParam.qHeadNum = qHeadNum;
    taskParam.isPFABSH = isPFABSH;
    taskParam.isFirstSInnerLoop = info.isFirstSInnerLoop;
    antiQuantQPreprocessor.Process(queryScm[queryScmIdx], queryInputQue, queryOutputQue, queryGm, taskParam);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::SendQK(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];
    uint32_t queryScmIdx = info.bnIdxInCurCore % BUFFER_NUM;

    uint8_t keyAntiquantResScmIdx = info.keyAntiquantResScmIdx;
    Mm1TaskParam mm1Param;
    mm1Param.sigM = info.gDealSize;
    mm1Param.sigN = info.actualSingleProcessSInnerSize;
    mm1Param.sigK = headDim;
    mm1Processor.Send(mmResultBufferArray[loop % BUFFER_NUM], queryScm[queryScmIdx],
                      kvAntiquantResScm[keyAntiquantResScmIdx], mm1Param);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::WaitQK(uint32_t loop)
{
    mm1Processor.Wait();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::Vector1(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];
    uint32_t outTensorIdx = loop % BUFFER_NUM;
    // softmax计算时，非第一次计算时会使用上一轮的sum、max结果作为输入
    uint32_t inTensorIdx = info.isFirstSInnerLoop ? outTensorIdx : (loop - 1) % BUFFER_NUM;
    Vec1TaskParam taskParam;
    taskParam.dealRowCount = info.gDealSize;
    taskParam.columnCount = ALIGN(info.actualSingleProcessSInnerSize, 32U);
    taskParam.actualColumnCount = info.actualSingleProcessSInnerSize;
    taskParam.isFirstSInnerLoop = info.isFirstSInnerLoop;
    taskParam.isLastSInnerLoop = info.s2Idx == info.sInnerLoopTimes - 1 ? true : false;
    taskParam.bIdx = info.bIdx;
    taskParam.n2Idx = info.n1Idx; // 名称先延用
    taskParam.gIdx = info.s1Idx;
    taskParam.qHeadNum = qHeadNum;
    taskParam.splitKVNum = splitKVNum;
    taskParam.gSize = gSize;
    taskParam.pseShiftS = pseShiftS;
    taskParam.flashDecodeS2Idx = info.flashDecodeS2Idx;
    taskParam.attenMaskOffset = info.attenMaskOffset;
    taskParam.attenMaskOffsetPre = info.attenMaskOffsetPre;
    taskParam.pseShiftOffset = info.pseShiftOffset;
    taskParam.scaleValue = tilingData->baseParams.scaleValue;
    taskParam.softmaxLseFlag = softmaxLseFlag;
    taskParam.qPaddingBeginOffset = info.qPaddingBeginOffset;
    taskParam.qSeqSize = qSeqSize;
    taskParam.attenMaskKVSize = attenMaskKVSize;
    taskParam.sparseMode = sparseMode;

    vec1Processor.Process(softmaxResOutputQue, mm2AScm[loop % BUFFER_NUM], softmaxMaxUb[outTensorIdx],
                          softmaxSumUb[outTensorIdx], softmaxExpUb[outTensorIdx], mmResultBufferArray[loop % BUFFER_NUM],
                          softmaxMaxUb[inTensorIdx], softmaxSumUb[inTensorIdx], softmaxTmpUb, attenMaskGm,
                          maskInputQue, pseShiftGm, pseInputQue, softmaxLseOutputQue, softmaxLseGm, taskParam,
                          softmaxMaxGm, softmaxSumGm);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::SendSV(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];

    uint8_t valueAntiquantResScmIdx = info.valueAntiquantResScmIdx;
    Mm2TaskParam taskParam;
    taskParam.sigM = info.gDealSize;
    taskParam.sigN = headDim;
    taskParam.sigK = info.actualSingleProcessSInnerSize;
    mm2Processor.Send(mmResultBufferArray[loop % BUFFER_NUM], mm2AScm[loop % BUFFER_NUM],
                      kvAntiquantResScm[valueAntiquantResScmIdx], taskParam);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::WaitSV(uint32_t loop)
{
    mm2Processor.Wait();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::Bmm2DataCopyOut(
    uint64_t attenOutOffset, LocalTensor<OUT_T>& attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::Update(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % BUFFER_NUM];

    Vec2TaskParam taskParam;
    taskParam.isFirstSInnerLoop = info.isFirstSInnerLoop;
    taskParam.isLastSInnerLoop = info.s2Idx == info.sInnerLoopTimes - 1 ? true : false;

    taskParam.startRow = info.s1Idx * PROFILE.G;
    taskParam.dealRowCount = info.gDealSize;
    taskParam.columnCount = headDimAlign;
    taskParam.actualColumnCount = headDim;

    taskParam.attenOutOffset = info.attenOutOffset;
    taskParam.attenOutStride = attenOutStride;
    taskParam.splitKVNum = splitKVNum;
    taskParam.gSize = gSize;
    taskParam.attenOutHeadNum = qHeadNum;
    taskParam.isPFABSH = isPFABSH || isBSNDOut;
    taskParam.flashDecodeS2Idx = info.flashDecodeS2Idx;
    taskParam.isRowInvalid = isRowInvalid == 1 ? true : false;

    vec2Processor.Process(mmResultBufferArray[loop % BUFFER_NUM], updateResOutputQue, softmaxSumUb[loop % BUFFER_NUM],
                          softmaxExpUb[loop % BUFFER_NUM], softmaxMaxUb[loop % BUFFER_NUM], attentionOutGm, accumOutGm, taskParam);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::Vector2(uint32_t loop)
{
    Update(loop);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CopyLseIn(uint32_t bIdx, uint32_t n2Idx,
                                                                               uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.AllocTensor<T>();
    LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = splitKVNum;
    copyInParams.blockLen = dealRowCount * FP32_ONE_BLOCK_SIZE * sizeof(T);
    copyInParams.srcStride = (gSize - dealRowCount) * FP32_ONE_BLOCK_SIZE * sizeof(T);
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    combineLseOffset = ((uint64_t)bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize * FP32_ONE_BLOCK_SIZE +
                       startRow * FP32_ONE_BLOCK_SIZE;

    DataCopyPad(softmaxMaxLocal, softmaxMaxGm[combineLseOffset], copyInParams, copyInPadParams);
    DataCopyPad(softmaxSumLocal, softmaxSumGm[combineLseOffset], copyInParams, copyInPadParams);
    softmaxMaxInputQue.EnQue(softmaxMaxLocal);
    softmaxSumInputQue.EnQue(softmaxSumLocal);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CopyFinalResOut(uint64_t attenOutOffset,
                                                                                     LocalTensor<T>& accumOutLocal,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    if constexpr (!POST_QUANT) {
        LocalTensor<OUT_T> tmpBmm2ResCastTensor = updateResOutputQue.AllocTensor<OUT_T>();  // 复用内存
        uint32_t shapeArray[] = {(uint32_t)dealRowCount, (uint32_t)headDim};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * PROFILE.D);

        updateResOutputQue.EnQue(tmpBmm2ResCastTensor);
        updateResOutputQue.DeQue<OUT_T>();
        Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, PROFILE.D, headDim);
        updateResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx,
                                                                                    uint32_t splitKVIndex,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();  // 复用内存

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (PROFILE.D - headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (PROFILE.D - headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;

    combineAccumOutOffset =
        ((uint64_t)bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim +
        startRow * headDim;
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx,
                                                                                    LocalTensor<T>& dst,
                                                                                    LocalTensor<T>& lseLocal,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = PROFILE.D / FP32_ONE_BLOCK_SIZE;
    repeatParams.dstRepStride = PROFILE.D / FP32_ONE_BLOCK_SIZE;
    int32_t dtype_mask = 256 / sizeof(float);
    int32_t mul_loop = PROFILE.D / dtype_mask;
    int32_t mul_remain = PROFILE.D % dtype_mask;

    // 第一次，mul结果直接放到dst里
    CopyAccumOutIn(bIdx, n2Idx, 0, startRow, dealRowCount);
    LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();

    FaVectorApi::ReduceFinalRes_const_VF<T, PROFILE.D>(dst, lseLocal, accumOutLocal, dealRowCount, 0);

    accumOutInputQue.FreeTensor(accumOutLocal);

    for (uint32_t j = 1; j < actualCombineLoopSize; ++j) {
        CopyAccumOutIn(bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();

        FaVectorApi::ReduceFinalRes_const_VF<T, PROFILE.D>(dst, lseLocal, accumOutLocal, dealRowCount, j);

        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::ComputeScaleValue(LocalTensor<T> lseMaxUb,
                                                                                       LocalTensor<T> lseSumUb,
                                                                                       uint32_t splitSize,
                                                                                       uint64_t lseOffset)
{   
    LocalTensor<T> lseOutputUb;
    if (softmaxLseFlag) {
        lseOutputUb = softmaxLseOutputQue.template AllocTensor<T>();
    }
    LocalTensor<bfloat16_t> tmpSinkUb;
    bool learnableSinkFlag = false;
    FaVectorApi::ComputeScaleValue_VF(tmpSinkUb, lseMaxUb, lseSumUb, lseOutputUb, splitSize, actualCombineLoopSize, softmaxLseFlag, learnableSinkFlag);
    if (softmaxLseFlag) {
        softmaxLseOutputQue.template EnQue<T>(lseOutputUb);
        softmaxLseOutputQue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = splitSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[lseOffset], lseOutputUb, intriParams1);
        softmaxLseOutputQue.FreeTensor(lseOutputUb);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::CombineSplitKVRes(uint64_t attenOutOffset,
                                                                                       uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_32K / (BYTE_BLOCK * splitKVNum);  // 8k / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = BUFFER_SIZE_BYTE_32K / sizeof(float) / PROFILE.D;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (gSize > PROFILE.G) {
        gSplitSize = (gSplitSize > PROFILE.G) ? PROFILE.G : gSplitSize;
    } else {
        gSplitSize = (gSplitSize > gSize) ? gSize : gSplitSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>();  // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, FP32_ONE_BLOCK_SIZE};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    uint64_t lseOffset = 0;

    // 非尾块处理
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        uint32_t startRow = i * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, gSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * kvHeadNum + n2Idx) * gSize + i * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, gSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitSize);
        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, gSplitSize);
    }
    // 尾块处理
    if (tailSplitSize > 0) {
        uint32_t startRow = (loopCount - 1) * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, tailSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * kvHeadNum + n2Idx) * gSize + (loopCount - 1) * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, tailSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, tailSplitSize);
        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, tailSplitSize);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::FlashDecodeCompute()
{
    uint32_t bIdx = tmpBlockIdx / kvHeadNum;  // 局部变量
    uint32_t n2Idx = tmpBlockIdx % kvHeadNum;
    if (tmpBlockIdx >= batchSize * kvHeadNum) {
        return;
    }
    GetActualSeqLen(bIdx, curActualSeqLen);
    if (curActualSeqLen == 0 ||
        !ComputePaddingBeginOffset(curActualSeqLen, kvPaddingBeginOffset, kvPaddingFlag, remainSeqLen)) {
        DealActualSeqLenIsZero(bIdx * kvHeadNum + n2Idx);
        return;
    }
    uint64_t attenOutOffset = (uint64_t)bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
    actualCombineLoopSize = (curActualSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;
    CombineSplitKVRes(attenOutOffset, bIdx, n2Idx);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAntiqSplitBbn2s2Us2<IFAT>::Process()
{
    if (tmpBlockIdx >= usedCoreNum) {
        // skip cores
    } else {
        if constexpr (!FLASH_DECODE) {
            InitCalcParamsEach();
        }
        SetLoopTimes();
        TaskContext taskContext[BUFFER_NUM];
        uint32_t gLoop = 0;
        uint32_t taskIdx = 0;
        for (uint32_t bnIdx = 0; bnIdx < bnLoopTimes; bnIdx++) {
            GetBNId(bnIdx);
            if constexpr (!FLASH_DECODE) {
                UpdateSOuterLoopEnd(bnIdx);
                UpdateZeroActualSeqLen(curBIdx, kvPaddingBeginOffset, sInnerLoopTimes, curActualSeqLen);
            }
            for (sOuterLoopIdx = curS1Idx; sOuterLoopIdx < sOuterLoopEnd; sOuterLoopIdx++) {
                if constexpr (!FLASH_DECODE) {
                    if (qSeqSize > 1) {
                        UpdateZeroActualSeqLenPFA(curBIdx, kvPaddingBeginOffset, sInnerLoopTimes, curActualSeqLen,
                            sOuterLoopIdx, preTokenPerBatch, nextTokenPerBatch, sInnerStartIndex);
                    }
                }
                for (uint32_t sInnerLoopIdx = sInnerStartIndex; sInnerLoopIdx < sInnerLoopTimes; sInnerLoopIdx++) {
                    TaskContext& task = taskContext[taskIdx++];
                    task.bIdx = curBIdx;
                    task.nIdx = curNIdx;
                    task.s1Idx = sOuterLoopIdx;
                    task.s2Idx = sInnerLoopIdx;
                    bool isLast = (bnIdx == bnLoopTimes - 1) && (sOuterLoopIdx == sOuterLoopEnd - 1) && (sInnerLoopIdx == sInnerLoopTimes - 1);
                    if (taskIdx < BUFFER_NUM && !isLast) {
                        continue;
                    }
                    for (uint32_t i = 0; i < taskIdx; i++) {
                        uint32_t loop = gLoop + i;
                        if (loop >= BUFFER_NUM) {
                            WaitSV(loop - BUFFER_NUM);
                            Vector2(loop - BUFFER_NUM);
                        }
                        CalcParams(loop, taskContext[loop % BUFFER_NUM]);
                        AntiquantKey(loop);
                        CopyQueryToL1(loop);
                        SendQK(loop); 
                    }
                    for (uint32_t i = 0; i < taskIdx; i++) {
                        uint32_t loop = gLoop + i;
                        WaitQK(loop);
                        Vector1(loop);
                        AntiquantValue(loop);
                        SendSV(loop);
                    }
                    gLoop += taskIdx;
                    taskIdx = 0;
                }
            }
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            uint32_t loop = gLoop + i;
            if (loop >= BUFFER_NUM) {
                WaitSV(loop - BUFFER_NUM);
                Vector2(loop - BUFFER_NUM);
            }
        }
    }
    if constexpr (FLASH_DECODE) {
        SyncAll();
        InitFDBuffers();
        FlashDecodeCompute();
    }
}

#endif