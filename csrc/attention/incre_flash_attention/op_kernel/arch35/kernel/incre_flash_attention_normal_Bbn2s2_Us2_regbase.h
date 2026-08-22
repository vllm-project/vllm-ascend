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
 * \file incre_flash_attention_normal_Bbn2s2_Us2.h
 * \brief
 */

#ifndef INCRE_FLASH_ATTENTION_NORMAL_BBN2S2_US2_H
#define INCRE_FLASH_ATTENTION_NORMAL_BBN2S2_US2_H

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
#include "../matmul_modules/ifa_matmul_policy.h"

#include "../service/mm1_processor.h"
#include "../service/mm2_processor.h"
#include "../service/vec1_processor.h"
#include "../service/vec2_processor.h"
#include "../incre_flash_attention_tiling_regbase.h"

using namespace matmul;

#define PRE_LOAD_NUM 2
#define TSCM_DOUBLE_BUFFER 2
#define L1_SELF_CONTROL

struct ExtraInfoNormal {
    uint64_t bIdx = 0;
    uint64_t n2Idx = 0;
    uint64_t s2Idx = 0;
    uint32_t gIdx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t attenOutOffset = 0;
    uint32_t gDealSize = 0;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint64_t attenMaskOffset;
    uint64_t attenMaskCoreOffset;
    uint64_t pseShiftOffset;
    uint64_t pseShiftCoreOffset;
    bool isFirstSInnerLoop = false;
    uint32_t flashDecodeS2Idx;
    uint32_t sInnerLoopTimes;
};

#ifndef L1_SELF_CONTROL
static constexpr MatmulConfig GenConfMM1Normal(const IFAProfile& ifa)
{
    MatmulConfig conf = GenConfMM1(ifa);
    conf.enableSetTail = true;
    conf.enableSetOrgShape = true;
    return conf;
}
#endif

static constexpr MatmulConfig GenConfMM2Normal(const IFAProfile& ifa)
{
    MatmulConfig conf = GenConfMM2(ifa);
    conf.enableSetTail = true;
    conf.enableSetOrgShape = true;
    return conf;
}

template <typename IFAT>
class IncreFlashAttentionNormalSplitBbn2s2Us2
{
public:
    __aicore__ inline IncreFlashAttentionNormalSplitBbn2s2Us2(
        const optiling::IncreFlashAttentionTilingDataRegbase& __restrict tilingData);
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
                                __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
                                __gm__ uint8_t* blockTable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* attentionOut,
                                __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, const optiling::IncreFlashAttentionTilingDataRegbase* __restrict tiling,
                                TPipe* tPipe);
    __aicore__ inline void InitQuant(__gm__ uint8_t* deqScale1, __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2,
                                     __gm__ uint8_t* quantScale2, __gm__ uint8_t* quantOffset2,
                                     __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
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
    static constexpr bool PER_TOKEN = IFAT::perToken;
    static constexpr INPUTKVTYPE KVTYPE_T = IFAT::inputKVType;
    static constexpr IFAProfile PROFILE = IFAT::profile;
    static constexpr bool ATTEN_MASK = IFAT::attenMask;
    static constexpr bool PSE_SHIFT = IFAT::pseShift;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;

    using MM_OUT_T = T;
    using MM_IN_T = Q_T;

#ifdef L1_SELF_CONTROL
    using mm1AType = MatmulType<TPosition::TSCM, CubeFormat::NZ, MM_IN_T, false, LayoutMode::NONE, false, TPosition::GM>;
    using mm1BType = MatmulType<TPosition::TSCM, CubeFormat::NZ, MM_IN_T, true, LayoutMode::NONE, false, TPosition::GM>;
#else
    using mm1AType = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false>;
    using mm1BType = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, true>;
#endif
    using mm1BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using mm1CType = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, MM_OUT_T>;

#ifdef L1_SELF_CONTROL
    __aicore__ inline static constexpr MatmulApiStaticTiling MM1GetMatmulApiTiling() {
        MatmulConfig conf = GenConfMM1MDL(PROFILE);
        MatmulApiStaticTiling mm1TilingTmp = GetMatmulApiTiling<mm1AType, mm1BType, mm1CType, mm1BiasType>(conf);
        mm1TilingTmp.depthA1 = 1;
        mm1TilingTmp.depthB1 = 2;  // 自管理搬入L1的基本块个数为2
        mm1TilingTmp.stepM = 1;
        mm1TilingTmp.stepN = 2;  // N方向搬入基本快个数为2
        mm1TilingTmp.stepKa = 1;
        mm1TilingTmp.stepKb = 1;
        mm1TilingTmp.dbL0A = 2;  // 2表示L0A开启db
        mm1TilingTmp.dbL0B = 2;  // 2表示L0B开启db
        mm1TilingTmp.dbL0C = 1;
        return mm1TilingTmp;
    }
    static constexpr auto mm1Tiling = MM1GetMatmulApiTiling();
#else
    static constexpr auto mm1Tiling =
        GetMatmulApiTiling<mm1AType, mm1BType, mm1CType, mm1BiasType>(GenConfMM1Normal(PROFILE));
#endif

    Matmul<mm1AType, mm1BType, mm1CType, mm1BiasType, mm1Tiling> mm;
    using mm2AType =
        MatmulType<TPosition::TSCM, CubeFormat::NZ, MM_IN_T, false, LayoutMode::NONE, false, TPosition::VECOUT>;
    using mm2BType = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false>;
    using mm2BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using mm2CType = MatmulType<TPosition::VECCALC, CubeFormat::ND, MM_OUT_T>;

    using PSE_SHIFT_T = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    static constexpr auto mm2Tiling =
        GetMatmulApiTiling<mm2AType, mm2BType, mm2CType, mm2BiasType>(GenConfMM2Normal(PROFILE));

    Matmul<mm2AType, mm2BType, mm2CType, mm2BiasType, mm2Tiling> bmm2;
    using mm1Type = Matmul<mm1AType, mm1BType, mm1CType, mm1BiasType, mm1Tiling>;
    Mm1Processor<IFAT, mm1Type> mm1Processor;
    using mm2Type = Matmul<mm2AType, mm2BType, mm2CType, mm2BiasType, mm2Tiling>;
    Mm2Processor<IFAT, mm2Type> mm2Processor;

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

    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // tilingData
    const uint32_t& batchSize;
    const uint32_t& kvHeadNum;
    const uint32_t& qHeadNum;
    const uint32_t& gSize;
    const uint32_t& headDim;
    const uint32_t& kvSeqSize;
    const uint32_t& actualLenDims;

    uint32_t curS2Idx;
    uint32_t curN2Idx;
    uint32_t curBIdx;
    uint32_t curGIdx;
    uint64_t curActualSeqLen;
    int64_t remainSeqLen;
    uint64_t kvPaddingBeginOffset = 0;

    const uint32_t& batchContinuous;
    const uint32_t& singleProcessSInnerSize;

    uint32_t sInnerLoopTimes;
    uint32_t sInnerLoopSize;
    uint32_t actualSInnerLoopSize;
    uint32_t singleProcessSInnerSizeTail;

    const uint32_t& blockSplitBn2Range;
    const uint32_t& tailBlockSplitBn2Range;
    const uint32_t& usedCoreNum;
    const uint32_t& formerCoreNum;
    const uint32_t* coreStartIdx;
    const uint32_t& splitKVNum;

    const uint32_t& attentMaskSize;
    const uint32_t& selectWithByteMaskTmpMinSize;

    const uint32_t& pseShiftB;
    const uint32_t& pseShiftS;

    const uint32_t& kvPaddingFlag;
    const uint32_t& isPerChnU8Out;
    const uint32_t& isOutQuantTypeBf16;
    bool softmaxLseFlag = false;

    uint32_t tmpBlockIdx;
    uint32_t beforeBlockSplitBn2Nums;
    uint32_t bn2Nums;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bn2s2LoopTimes;
    uint64_t tensorACoreOffset;
    uint64_t tensorBCoreOffset;
    uint64_t tensorBStride;
    uint64_t headDimAlign;
    uint64_t headDimAlignFp32;
    ExtraInfoNormal extraInfo[PRE_LOAD_NUM]{};
    // flash decode
    uint32_t actualCombineLoopSize;
    uint64_t combineLseOffset;
    uint64_t combineAccumOutOffset;

    // UB
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue;
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECIN, 1> maskInputQue;
    TQue<QuePosition::VECIN, 1> pseInputQue;

    TBuf<> mmResBuffBig;
    LocalTensor<T> mmResBuffArray[PRE_LOAD_NUM];
    TQue<QuePosition::VECOUT, 1> softmaxResOutputQue;
    TQue<QuePosition::VECOUT, 1> updateResOutputQue;
    TBuf<> softmaxBuffBig;
    TBuf<> lseTmpBuff;

    TQue<QuePosition::VECOUT, 1> softmaxLseOutputQue;

    LocalTensor<T> softmaxMaxUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxSumUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxExpUb[PRE_LOAD_NUM];
    LocalTensor<uint8_t> softmaxTmpUb;
    LocalTensor<MM_OUT_T> bmm2PreResUb;

#ifdef L1_SELF_CONTROL
    TSCM<TPosition::GM, 1> mm1AScm[TSCM_DOUBLE_BUFFER];
    TSCM<TPosition::GM, 1> mm1BScm[TSCM_DOUBLE_BUFFER];
#endif
    TSCM<TPosition::VECIN, 1> mm2AScm[PRE_LOAD_NUM];

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);

    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitL1Buffers();
    __aicore__ inline void InitFDBuffers();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitCalcParams();
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void SetLoopTimes();
    __aicore__ inline void UpdateZeroActSeqLen();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bn2Idx);
    __aicore__ inline void InitAllZeroOutput(uint32_t bn2Idx);
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint64_t& actualSeqLen);
    __aicore__ inline bool ComputeKVPaddingBeginOffset();
    __aicore__ inline void InitLeftPaddingDate(); 

    __aicore__ inline void GetBN2S2GId(uint32_t loop, uint32_t gLoopCount);
    __aicore__ inline void CalcParams(uint32_t loop);

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
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, uint32_t splitSize, 
                                             uint64_t lseOffset);
    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);
#ifdef L1_SELF_CONTROL
    __aicore__ inline void CopyQueryToL1(uint32_t loop);
    __aicore__ inline void CopyKeyToL1(uint32_t loop);
#endif
};

template <typename IFAT>
__aicore__ inline IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::IncreFlashAttentionNormalSplitBbn2s2Us2(
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
      gSize(tilingData.baseParams.nNumOfQInOneGroup),
      kvSeqSize(tilingData.baseParams.seqSize),
      actualLenDims(tilingData.baseParams.actualLenDims),
      headDim(tilingData.baseParams.headSize),
      batchContinuous(tilingData.baseParams.batchContinuousFlag),
      coreStartIdx(tilingData.increFlashAttentionCoreParams.coreSidxEndRegbase),
      attentMaskSize(tilingData.baseParams.attenMaskSize),
      selectWithByteMaskTmpMinSize(tilingData.baseParams.selectWithByteMaskTmpMinSize),
      pseShiftB(tilingData.baseParams.pseShiftB),
      pseShiftS(tilingData.baseParams.pseShiftS),
      kvPaddingFlag(tilingData.baseParams.kvPaddingFlag),
      isPerChnU8Out(tilingData.outputParams.isPerChnOut),
      isOutQuantTypeBf16(tilingData.outputParams.isOutQuantTypeBf16),
      softmaxLseFlag(tilingData.baseParams.softmaxLseFlag)
{
#ifdef __DAV_C310_CUBE__
    // cube上预留出L1自管理空间
    TSCM<TPosition::VECIN, 1, 0x04> tmpQue;
#ifdef L1_SELF_CONTROL
    // L1共512KB空间 preload=2 C:V=1:2 分给单次mm的空间最大为512/4=128KB
    size_t size = ((BUFFER_SIZE_BYTE_16K + BUFFER_SIZE_BYTE_16K + BUFFER_SIZE_BYTE_64K) * PRE_LOAD_NUM);
#else
    size_t size = (BUFFER_SIZE_BYTE_16K * PRE_LOAD_NUM);
#endif
    GetTPipePtr()->InitBuffer(tmpQue, 1, size * 2);  // 2, vector0/1
#endif
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
    __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* blockTable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    const optiling::IncreFlashAttentionTilingDataRegbase* __restrict tiling, TPipe* tPipe)
{
    tmpBlockIdx = GetBlockIdx();

    // Only one vector core use one cube core when B*N number is less than half of core number
    if constexpr (!FLASH_DECODE) {
        if (tmpBlockIdx & 0x1) {
            tmpBlockIdx = (tmpBlockIdx + GetBlockNum() * 2) / 2;
        } else {
            tmpBlockIdx = tmpBlockIdx / 2;
        }
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

    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t*)actualSeqLengths, actualLenDims);
    }

    if (kvPaddingFlag == 1) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t*) kvPaddingSize);
        InitLeftPaddingDate();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitQuant(
    __gm__ uint8_t* deqScale1, __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2, __gm__ uint8_t* quantScale2,
    __gm__ uint8_t* quantOffset2, __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
    __gm__ uint8_t* workspace)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitTilingData()
{
    headDimAlign = ALIGN((uint64_t)headDim, BYTE_BLOCK);                  // 32个数字对齐
    headDimAlignFp32 = ALIGN((uint64_t)headDim, BYTE_BLOCK / sizeof(T));  // 32B对齐
    tensorBStride = (LAYOUT_T == LAYOUT::BNSD) ? headDim : kvHeadNum * headDim;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitLeftPaddingDate()
{
    int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }
    remainSeqLen = kvSeqSize - paddingSize;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitBuffers()
{
    if constexpr (FLASH_DECODE) {
        pipe->InitBuffer(accumOutInputQue, 1, BUFFER_SIZE_BYTE_32K);
    }
    pipe->InitBuffer(softmaxLseOutputQue, 1, BUFFER_SIZE_BYTE_512B);

    if constexpr (ATTEN_MASK) {
        pipe->InitBuffer(maskInputQue, 1, BUFFER_SIZE_BYTE_4K);
    }
    if constexpr (PSE_SHIFT) {
        pipe->InitBuffer(pseInputQue, 1, BUFFER_SIZE_BYTE_8K);
    }
    // 尽量一次初始化一大块
    uint32_t maxElemCnt =
        PROFILE.G * MAX(PROFILE.S, PROFILE.D);  // bmm1, bmm2复用输出内存, 大小取D分档和S分档中的较大的值
    pipe->InitBuffer(mmResBuffBig, maxElemCnt * sizeof(T) * PRE_LOAD_NUM);
    mmResBuffArray[0] = mmResBuffBig.GetWithOffset<T>(maxElemCnt, 0);
    mmResBuffArray[1] = mmResBuffBig.GetWithOffset<T>(maxElemCnt, maxElemCnt * sizeof(T));

    pipe->InitBuffer(softmaxResOutputQue, 1, (PROFILE.G + 1) * PROFILE.S * sizeof(Q_T));
    pipe->InitBuffer(updateResOutputQue, 1, PROFILE.G * PROFILE.D * sizeof(T));

    uint32_t softmaxElemCnt = BUFFER_SIZE_BYTE_512B / sizeof(T);
    pipe->InitBuffer(softmaxBuffBig, BUFFER_SIZE_BYTE_512B * 7);
    softmaxMaxUb[0] = softmaxBuffBig.GetWithOffset<T>(softmaxElemCnt, 0);
    softmaxMaxUb[1] = softmaxBuffBig.GetWithOffset<T>(softmaxElemCnt, BUFFER_SIZE_BYTE_512B);
    softmaxSumUb[0] = softmaxBuffBig.GetWithOffset<T>(softmaxElemCnt, BUFFER_SIZE_BYTE_512B * 2);
    softmaxSumUb[1] = softmaxBuffBig.GetWithOffset<T>(softmaxElemCnt, BUFFER_SIZE_BYTE_512B * 3);
    softmaxExpUb[0] = softmaxBuffBig.GetWithOffset<T>(softmaxElemCnt, BUFFER_SIZE_BYTE_512B * 4);
    softmaxExpUb[1] = softmaxBuffBig.GetWithOffset<T>(softmaxElemCnt, BUFFER_SIZE_BYTE_512B * 5);
    softmaxTmpUb = softmaxBuffBig.GetWithOffset<uint8_t>(BUFFER_SIZE_BYTE_512B, BUFFER_SIZE_BYTE_512B * 6);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitFDBuffers()
{
    pipe->Reset();
    pipe->InitBuffer(lseTmpBuff, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxMaxInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxSumInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(accumOutInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxLseOutputQue, 1, BUFFER_SIZE_BYTE_512B);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitL1Buffers()
{
#ifdef L1_SELF_CONTROL
    uint32_t queryBytes = PROFILE.M1 * headDimAlign * sizeof(Q_T);
    uint32_t keyBytes = PROFILE.S * headDimAlign * sizeof(KV_T);
    for (uint32_t i = 0; i < TSCM_DOUBLE_BUFFER; i++) {
        pipe->InitBuffer(mm1AScm[i], 1, queryBytes);
    }
    for (uint32_t i = 0; i < TSCM_DOUBLE_BUFFER; i++) {
        pipe->InitBuffer(mm1BScm[i], 1, keyBytes);
    }
#endif
    for (uint32_t i = 0; i < PRE_LOAD_NUM; i++) {
        pipe->InitBuffer(mm2AScm[i], 1, BUFFER_SIZE_BYTE_16K);
    }
}

#ifdef L1_SELF_CONTROL
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::CopyQueryToL1(uint32_t loop) {
    auto& info = extraInfo[loop % PRE_LOAD_NUM];
    uint32_t queryScmIdx = info.bn2IdxInCurCore % TSCM_DOUBLE_BUFFER;
    LocalTensor<Q_T> queryScmTensor = mm1AScm[queryScmIdx].AllocTensor<Q_T>();
    if (info.isFirstSInnerLoop) {
        Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.nValue = info.gDealSize;
        intriParams.dValue = headDim;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = headDim;
        intriParams.dstNzC0Stride = PROFILE.M1;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;
        DataCopy(queryScmTensor , queryGm[info.tensorAOffset], intriParams);
    }
    mm1AScm[queryScmIdx].template EnQue(queryScmTensor);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::CopyKeyToL1(uint32_t loop) {
    auto& info = extraInfo[loop % PRE_LOAD_NUM];
    uint32_t keyScmIdx = loop % TSCM_DOUBLE_BUFFER;
    LocalTensor<KV_T> keyScmTensor = mm1BScm[keyScmIdx].AllocTensor<KV_T>();
    uint64_t keyOffset = info.tensorBOffset;
    if (!batchContinuous && (loop == 0 || info.n2Idx == 0)) {
        InitKeyGm(info.bIdx);
    }
    Nd2NzParams intriParams;
    intriParams.ndNum = 1;
    intriParams.nValue = info.actualSingleProcessSInnerSize;
    intriParams.dValue = headDim;
    intriParams.srcNdMatrixStride = 0;
    intriParams.srcDValue = tensorBStride;
    intriParams.dstNzC0Stride = PROFILE.S;
    intriParams.dstNzNStride = 1;
    intriParams.dstNzMatrixStride = 0;
    DataCopy(keyScmTensor , keyGm[keyOffset], intriParams);
    mm1BScm[keyScmIdx].template EnQue(keyScmTensor);
}
#endif

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitCalcParams()
{
    bn2Nums = blockSplitBn2Range;
    beforeBlockSplitBn2Nums = tmpBlockIdx * blockSplitBn2Range;
    // tail core
    if (tmpBlockIdx >= formerCoreNum) {
        bn2Nums = tailBlockSplitBn2Range;
        beforeBlockSplitBn2Nums =
            formerCoreNum * blockSplitBn2Range + (tmpBlockIdx - formerCoreNum) * tailBlockSplitBn2Range;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitCalcParamsEach()
{
    bn2Nums = coreStartIdx[tmpBlockIdx + 1] - coreStartIdx[tmpBlockIdx];
    beforeBlockSplitBn2Nums = coreStartIdx[tmpBlockIdx];
    curBIdx = beforeBlockSplitBn2Nums / kvHeadNum;
    curN2Idx = beforeBlockSplitBn2Nums % kvHeadNum;
    curS2Idx = 0;
    curGIdx = 0;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::DealActSeqLenIsZero(uint32_t bn2Idx)
{
    InitAllZeroOutput(bn2Idx);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitAllZeroOutput(uint32_t bn2Idx)
{
    uint32_t copySize = gSize * headDim;
    if constexpr (POST_QUANT) {  // out int8
                                 //
    } else {
        matmul::InitOutput<OUT_T>(attentionOutGm[bn2Idx * copySize], copySize, 0);
    }
    
    if (softmaxLseFlag) {
        LocalTensor<T> lseUb = softmaxLseOutputQue.template AllocTensor<T>();
        Duplicate(lseUb, FLOAT_MAX, gSize);
        softmaxLseOutputQue.template EnQue(lseUb);
        softmaxLseOutputQue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float) * gSize;
        intriParams1.blockCount = 1;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[bn2Idx * gSize], lseUb, intriParams1);
        softmaxLseOutputQue.FreeTensor(lseUb);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::GetActualSeqLen(uint32_t bIdx,
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
__aicore__ inline bool IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::ComputeKVPaddingBeginOffset()
{
    if (kvPaddingFlag != 1) {
        return true;
    }

    int64_t startPosition = remainSeqLen - curActualSeqLen;
    
    if (startPosition < 0) {
        return false;
    }

    kvPaddingBeginOffset = static_cast<uint64_t>(startPosition);
    return true;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void*)keyPtr);
    __gm__ uint8_t* key_ = (__gm__ uint8_t*)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
    keyGm.SetGlobalBuffer((__gm__ KV_T*)key_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void*)valuePtr);
    __gm__ uint8_t* value_ = (__gm__ uint8_t*)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
    valueGm.SetGlobalBuffer((__gm__ KV_T*)value_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::SetLoopTimes()
{
    uint32_t gLoopCount = (gSize + PROFILE.G - 1) / PROFILE.G;
    if constexpr (FLASH_DECODE) {
        GetActualSeqLen(tmpBlockIdx / (kvHeadNum * splitKVNum), curActualSeqLen);
        sInnerLoopSize = (kvSeqSize + splitKVNum - 1) / splitKVNum;
        if (sInnerLoopSize * (tmpBlockIdx % splitKVNum) > curActualSeqLen) {
            actualSInnerLoopSize = 0;
        } else {
            actualSInnerLoopSize = (curActualSeqLen - sInnerLoopSize * (tmpBlockIdx % splitKVNum)) > sInnerLoopSize ? 
                                    sInnerLoopSize : (curActualSeqLen - sInnerLoopSize * (tmpBlockIdx % splitKVNum));
        }
        sInnerLoopTimes = (actualSInnerLoopSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
        bn2s2LoopTimes = sInnerLoopTimes * gLoopCount;
        ComputeKVPaddingBeginOffset();
    } else {
        for (uint32_t bn2Idx = beforeBlockSplitBn2Nums; bn2Idx < beforeBlockSplitBn2Nums + bn2Nums; bn2Idx++) {
            GetActualSeqLen(bn2Idx / kvHeadNum, curActualSeqLen);
            if (curActualSeqLen == 0 || !ComputeKVPaddingBeginOffset()) {
                DealActSeqLenIsZero(bn2Idx);
                continue;
            }
            bn2s2LoopTimes += (curActualSeqLen + singleProcessSInnerSize - 1) / singleProcessSInnerSize * gLoopCount;
        }
        GetActualSeqLen(beforeBlockSplitBn2Nums / kvHeadNum, curActualSeqLen);
        sInnerLoopTimes = (curActualSeqLen + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::SeqLenFromTensorList(uint32_t bIdx)
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

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::UpdateZeroActSeqLen()
{
    while (curActualSeqLen == 0 || !ComputeKVPaddingBeginOffset()) {
        curBIdx = curBIdx + 1;
        GetActualSeqLen(curBIdx, curActualSeqLen);
    }
    sInnerLoopTimes = (curActualSeqLen + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
    curN2Idx = 0;
    curGIdx = 0;
    curS2Idx = 0;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::GetBN2S2GId(uint32_t loop, uint32_t gLoopCount)
{
    if (loop == 0) {
        if (curActualSeqLen == 0 || !ComputeKVPaddingBeginOffset()) {
            UpdateZeroActSeqLen();
            return;
        }
    } else {
        if (curS2Idx == sInnerLoopTimes - 1) {
            curS2Idx = 0;
            if (curGIdx == gLoopCount - 1) {
                curGIdx = 0;
                if (curN2Idx == kvHeadNum - 1) {
                    curN2Idx = 0;
                    curBIdx = curBIdx + 1;
                    // update curActualSeqLen
                    GetActualSeqLen(curBIdx, curActualSeqLen);
                    UpdateZeroActSeqLen();
                } else {
                    curN2Idx = curN2Idx + 1;
                }
            } else {
                curGIdx = curGIdx + 1;
            }
        } else {
            curS2Idx = curS2Idx + 1;
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::CalcParams(uint32_t loop)
{
    uint32_t gLoopCount = (gSize + PROFILE.G - 1) / PROFILE.G;
    auto& info = extraInfo[loop % PRE_LOAD_NUM];
    if constexpr (FLASH_DECODE) {
        info.bIdx = tmpBlockIdx / (kvHeadNum * splitKVNum);
        info.n2Idx = (tmpBlockIdx / splitKVNum) % kvHeadNum;
        info.s2Idx = loop % sInnerLoopTimes;  // 用于循环结束
        info.gIdx = (loop / sInnerLoopTimes) % gLoopCount;
        info.flashDecodeS2Idx = tmpBlockIdx % splitKVNum;  // 用于偏移
        info.sInnerLoopTimes = sInnerLoopTimes;
    } else {
        GetBN2S2GId(loop, gLoopCount);
        info.bIdx = curBIdx;
        info.n2Idx = curN2Idx;
        info.gIdx = curGIdx;
        info.s2Idx = curS2Idx;
        info.sInnerLoopTimes = sInnerLoopTimes;
    }

    info.isFirstSInnerLoop = (info.s2Idx == 0);
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }

    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;

    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            tensorACoreOffset = info.bIdx * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            tensorBCoreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDim + info.n2Idx * kvSeqSize * headDim +
                                kvPaddingBeginOffset * headDim;
            if (!batchContinuous) {
                uint64_t seqSize = SeqLenFromTensorList(info.bIdx);
                tensorBCoreOffset = info.n2Idx * seqSize * headDim;
            }
            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize * headDim;
            }
        } else {
            tensorACoreOffset = info.bIdx * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim +
                                kvPaddingBeginOffset * kvHeadNum * headDim;
            if (!batchContinuous) {
                tensorBCoreOffset = info.n2Idx * headDim;
            }
            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize * kvHeadNum * headDim;
            }
        }
    }

    info.tensorAOffset = tensorACoreOffset + info.gIdx * PROFILE.G * headDim;
    info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * tensorBStride;
    info.attenOutOffset = tensorACoreOffset;
    info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
    info.attenMaskCoreOffset = info.bIdx * attentMaskSize + kvPaddingBeginOffset;
    if constexpr (FLASH_DECODE) {
        info.attenMaskCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize;
    }
    info.attenMaskOffset = info.attenMaskCoreOffset + info.s2Idx * singleProcessSInnerSize;

    // s不对齐场景需按实际 tail 刷新 singleProcessSInnerSizeTail;
    if (info.s2Idx == sInnerLoopTimes - 1) {  // 尾块
        info.actualSingleProcessSInnerSize =
            FLASH_DECODE ? actualSInnerLoopSize - (sInnerLoopTimes - 1) * PROFILE.S : curActualSeqLen - (sInnerLoopTimes - 1) * PROFILE.S;
    }
    if (info.gIdx == (gLoopCount - 1)) {
        info.gDealSize = gSize - (gLoopCount - 1) * PROFILE.G;
    } else {
        info.gDealSize = PROFILE.G;
    }
    if constexpr (PSE_SHIFT) {
        info.pseShiftCoreOffset =
            (pseShiftB == 1)
                ? (info.n2Idx * gSize * pseShiftS + kvPaddingBeginOffset)
                : (info.bIdx * qHeadNum * pseShiftS + info.n2Idx * gSize * pseShiftS + kvPaddingBeginOffset);
        if constexpr (FLASH_DECODE) {
            info.pseShiftCoreOffset += info.flashDecodeS2Idx * sInnerLoopSize;
        }
        info.pseShiftOffset = info.pseShiftCoreOffset + info.s2Idx * singleProcessSInnerSize;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::SendQK(uint32_t loop)
{
    auto& info = extraInfo[loop % PRE_LOAD_NUM];
#ifdef L1_SELF_CONTROL
    CopyKeyToL1(loop);
    CopyQueryToL1(loop);
    uint32_t queryScmIdx = info.bn2IdxInCurCore % TSCM_DOUBLE_BUFFER;
    uint32_t keyScmIdx = loop % TSCM_DOUBLE_BUFFER;
#else
    if (!batchContinuous && (loop == 0 || info.n2Idx == 0)) {
        InitKeyGm(info.bIdx);
    }
#endif

    Mm1TaskParam taskParam;
    taskParam.sigM = info.gDealSize;
    taskParam.sigN = info.actualSingleProcessSInnerSize;
    taskParam.sigK = headDim;
    taskParam.orgM = gSize;
    taskParam.orgN = tilingData->baseParams.seqSize;
    taskParam.orgKa = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH) {
        taskParam.orgKb = kvHeadNum * headDim;
    } else {
        taskParam.orgKb = headDim;
    }

    taskParam.orgKc = PROFILE.S;
    taskParam.tscmIdx = GetSubBlockIdx() * 2 + info.bn2IdxInCurCore % PRE_LOAD_NUM;
    taskParam.tensorAOffset = info.tensorAOffset;
    taskParam.tensorBOffset = info.tensorBOffset;

#ifdef L1_SELF_CONTROL
    mm1Processor.Send(mmResBuffArray[loop % PRE_LOAD_NUM], mm1AScm[queryScmIdx], mm1BScm[keyScmIdx], taskParam);
#else
    mm1Processor.Send(mmResBuffArray[loop % PRE_LOAD_NUM], queryGm, keyGm, taskParam);
#endif
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::WaitQK(uint32_t loop)
{
    mm1Processor.Wait();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::Vector1(uint32_t loop)
{
    auto& info = extraInfo[loop % PRE_LOAD_NUM];
    uint32_t outTensorIdx = loop % PRE_LOAD_NUM;
    uint32_t inTensorIdx = info.isFirstSInnerLoop ? outTensorIdx : (loop - 1) % PRE_LOAD_NUM;
    Vec1TaskParam taskParam;
    taskParam.dealRowCount = info.gDealSize;
    taskParam.columnCount = ALIGN(info.actualSingleProcessSInnerSize, 32U);
    taskParam.actualColumnCount = info.actualSingleProcessSInnerSize;
    taskParam.isFirstSInnerLoop = info.isFirstSInnerLoop;
    taskParam.isLastSInnerLoop = info.s2Idx == info.sInnerLoopTimes - 1 ? true : false;
    taskParam.bIdx = info.bIdx;
    taskParam.n2Idx = info.n2Idx;
    taskParam.gIdx = info.gIdx;
    taskParam.qHeadNum = qHeadNum;
    taskParam.splitKVNum = splitKVNum;
    taskParam.gSize = gSize;
    taskParam.pseShiftS = pseShiftS;
    taskParam.flashDecodeS2Idx = info.flashDecodeS2Idx;
    taskParam.attenMaskOffset = info.attenMaskOffset;
    taskParam.pseShiftOffset = info.pseShiftOffset;
    taskParam.scaleValue = tilingData->baseParams.scaleValue;
    taskParam.softmaxLseFlag = softmaxLseFlag;
    taskParam.qSeqSize = 1;
    taskParam.qPaddingBeginOffset = 0;
    taskParam.sparseMode = 0;
    taskParam.attenMaskKVSize = 0;
    taskParam.attenMaskOffsetPre = 0;

    vec1Processor.Process(softmaxResOutputQue, mm2AScm[loop % PRE_LOAD_NUM], softmaxMaxUb[outTensorIdx],
                          softmaxSumUb[outTensorIdx], softmaxExpUb[outTensorIdx], mmResBuffArray[loop % PRE_LOAD_NUM],
                          softmaxMaxUb[inTensorIdx], softmaxSumUb[inTensorIdx], softmaxTmpUb, attenMaskGm,
                          maskInputQue, pseShiftGm, pseInputQue, softmaxLseOutputQue, softmaxLseGm, taskParam,
                          softmaxMaxGm, softmaxSumGm);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::SendSV(uint32_t loop)
{
    auto& info = extraInfo[loop % PRE_LOAD_NUM];
    if (!batchContinuous && (loop == 0 || info.n2Idx == 0)) {
        InitValueGm(info.bIdx);
    }
    Mm2TaskParam taskParam;
    taskParam.sigM = info.gDealSize;
    taskParam.sigN = headDim;
    taskParam.sigK = info.actualSingleProcessSInnerSize;

    taskParam.orgM = gSize;
    if constexpr (LAYOUT_T == LAYOUT::BSH) {
        taskParam.orgN = kvHeadNum * headDim;
    } else {
        taskParam.orgN = headDim;
    }
    taskParam.orgKa = PROFILE.S;
    taskParam.orgKb = PROFILE.S;
    taskParam.orgKc = PROFILE.D;
    taskParam.tensorBOffset = info.tensorBOffset;

    mm2Processor.Send(mmResBuffArray[loop % PRE_LOAD_NUM], mm2AScm[loop % PRE_LOAD_NUM], valueGm, taskParam);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::WaitSV(uint32_t loop)
{
    mm2Processor.Wait();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::Bmm2DataCopyOut(
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
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::Update(uint32_t loop)
{
    auto& info = extraInfo[loop % PRE_LOAD_NUM];

    Vec2TaskParam taskParam;
    taskParam.isFirstSInnerLoop = info.isFirstSInnerLoop;
    taskParam.isLastSInnerLoop = info.s2Idx == info.sInnerLoopTimes - 1 ? true : false;

    taskParam.startRow = info.gIdx * PROFILE.G;
    taskParam.dealRowCount = info.gDealSize;
    taskParam.columnCount = headDimAlign;
    taskParam.actualColumnCount = headDim;

    taskParam.attenOutOffset = info.attenOutOffset;
    taskParam.splitKVNum = splitKVNum;
    taskParam.gSize = gSize;
    taskParam.flashDecodeS2Idx = info.flashDecodeS2Idx;
    taskParam.attenOutStride = headDim;
    taskParam.isPFABSH = false;
    taskParam.isRowInvalid = false;

    vec2Processor.Process(mmResBuffArray[loop % PRE_LOAD_NUM], updateResOutputQue, softmaxSumUb[loop % PRE_LOAD_NUM],
                          softmaxExpUb[loop % PRE_LOAD_NUM], softmaxMaxUb[loop % PRE_LOAD_NUM], attentionOutGm, accumOutGm, taskParam);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::Vector2(uint32_t loop)
{
    Update(loop);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::CopyLseIn(uint32_t bIdx, uint32_t n2Idx,
                                                                                uint32_t startRow,
                                                                                uint32_t dealRowCount)
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
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::CopyFinalResOut(uint64_t attenOutOffset,
                                                                                      LocalTensor<T>& accumOutLocal,
                                                                                      uint32_t startRow,
                                                                                      uint32_t dealRowCount)
{
    if constexpr (!POST_QUANT) {
        LocalTensor<OUT_T> tmpBmm2ResCastTensor = updateResOutputQue.AllocTensor<OUT_T>();  // 复用内存
        uint32_t shapeArray[] = {(uint32_t)dealRowCount, (uint32_t)headDim};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * headDimAlignFp32);

        updateResOutputQue.EnQue(tmpBmm2ResCastTensor);
        updateResOutputQue.DeQue<OUT_T>();

        Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, headDimAlignFp32, headDim);
        updateResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
    } else {
#ifdef IFA_TO_RECOVER
        LocalTensor<OUT_T> bmm2ResUbInt8 = outputQue1.AllocTensor<OUT_T>();
        uint32_t shapeArray[] = {(uint32_t)dealRowCount, (uint32_t)headDim};
        bmm2ResUbInt8.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        PostQuant(accumOutLocal, bmm2ResUbInt8, 0, dealRowCount, headDimAlignFp32, headDim);
        outputQue1.EnQue(bmm2ResUbInt8);
        outputQue1.DeQue<OUT_T>();
        Bmm2DataCopyOut(bmm2ResUbInt8, startRow, dealRowCount, headDimAlignFp32, headDim);
        outputQue1.FreeTensor(bmm2ResUbInt8);
#endif
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx,
                                                                                     uint32_t splitKVIndex,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlignFp32 - headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlignFp32 - headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;

    combineAccumOutOffset =
        ((uint64_t)bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim +
        startRow * headDim;
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx,
                                                                                     LocalTensor<T>& dst,
                                                                                     LocalTensor<T>& lseLocal,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = headDimAlignFp32 / FP32_ONE_BLOCK_SIZE;
    repeatParams.dstRepStride = headDimAlignFp32 / FP32_ONE_BLOCK_SIZE;
    int32_t dtype_mask = 256 / sizeof(float);
    int32_t mul_loop = headDimAlignFp32 / dtype_mask;
    int32_t mul_remain = headDimAlignFp32 % dtype_mask;

    for (uint32_t j = 0; j < actualCombineLoopSize; ++j) {
        // 第一次，mul结果直接放到dst里
        CopyAccumOutIn(bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();
        FaVectorApi::ReduceFinalRes_const_VF<T, PROFILE.D>(dst, lseLocal, accumOutLocal, dealRowCount, j);
        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::ComputeScaleValue(LocalTensor<T> lseMaxUb,
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
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::CombineSplitKVRes(uint64_t attenOutOffset,
                                                                                        uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_32K / (BYTE_BLOCK * splitKVNum);  // 16k / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = BUFFER_SIZE_BYTE_32K / sizeof(float) / headDimAlign;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (gSize > PROFILE.G) {
        gSplitSize = (gSplitSize > PROFILE.G) ? PROFILE.G : gSplitSize;
    } else {
        gSplitSize = (gSplitSize > gSize) ? gSize : gSplitSize;
    }
    uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;
    uint64_t lseOffset = 0;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>();  // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, FP32_ONE_BLOCK_SIZE};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

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
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::FlashDecodeCompute()
{
    uint32_t bIdx = tmpBlockIdx / kvHeadNum;  // 局部变量
    uint32_t n2Idx = tmpBlockIdx % kvHeadNum;
    if (tmpBlockIdx >= batchSize * kvHeadNum) {
        return;
    }
    GetActualSeqLen(bIdx, curActualSeqLen);
    if (curActualSeqLen == 0 || !ComputeKVPaddingBeginOffset()) {
        DealActSeqLenIsZero(bIdx * kvHeadNum + n2Idx);
        return;
    }
    uint64_t attenOutOffset = (uint64_t)bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
    actualCombineLoopSize = (curActualSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;

    CombineSplitKVRes(attenOutOffset, bIdx, n2Idx);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionNormalSplitBbn2s2Us2<IFAT>::Process()
{
    if (tmpBlockIdx >= usedCoreNum) {
        // skip cores, syncAll() should be called for each core in flashdeding
    } else {
        if constexpr (FLASH_DECODE) {
            InitCalcParams();
        } else {
            InitCalcParamsEach();
        }
        SetLoopTimes();

        for (uint32_t i = 0; i < bn2s2LoopTimes; i = i + PRE_LOAD_NUM) {
            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;

                if (loop >= PRE_LOAD_NUM) {
                    WaitSV(loop - PRE_LOAD_NUM);
                    Vector2(loop - PRE_LOAD_NUM);
                }

                if (loop >= bn2s2LoopTimes) {
                    break;
                }
                CalcParams(loop);
                SendQK(loop);
            }

            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;
                if (loop >= bn2s2LoopTimes) {
                    break;
                }
                WaitQK(loop);
                Vector1(loop);
                SendSV(loop);
            }

            if (i + PRE_LOAD_NUM >= bn2s2LoopTimes) {
                for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                    uint32_t loop = i + j;
                    if (loop >= bn2s2LoopTimes) {
                        break;
                    }
                    WaitSV(loop);
                    Vector2(loop);
                }
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