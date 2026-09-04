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
 * \file incre_flash_attention_preload_dd.h
 * \brief
 */
#ifndef INCRE_FLASH_ATTENTION_PRELOAD_DD
#define INCRE_FLASH_ATTENTION_PRELOAD_DD

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
#include "../ifa_public_define.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

#define PRE_LOAD_NUM_DD 4
#define ATTENMASK_STRIDE 2048U
#define EXTRA_NUM 6
#define TASK_NUM 2
#define IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
#define Q_AMAX_BUF_BYTES (BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B)
#define P_AMAX_BUF_BYTES (BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B)
#define Q_ROWSUM_BUF_BYTES (BUFFER_SIZE_BYTE_512B + BUFFER_SIZE_BYTE_256B)
#define P_ROWSUM_BUF_BYTES (BUFFER_SIZE_BYTE_512B + BUFFER_SIZE_BYTE_256B)
#define Q_AMAX_BUF_SIZE (Q_AMAX_BUF_BYTES / sizeof(T))
#define P_AMAX_BUF_SIZE (P_AMAX_BUF_BYTES / sizeof(T))
#define Q_ROWSUM_BUF_SIZE (Q_ROWSUM_BUF_BYTES / sizeof(T))
#define P_ROWSUM_BUF_SIZE (P_ROWSUM_BUF_BYTES / sizeof(T))

#define SOFTMAX_MAX_BUF_BYTES BUFFER_SIZE_BYTE_512B
#define SOFTMAX_EXP_BUF_BYTES BUFFER_SIZE_BYTE_512B
#define SOFTMAX_SUM_BUF_BYTES BUFFER_SIZE_BYTE_512B
#define SOFTMAX_MAX_BUF_SIZE (SOFTMAX_MAX_BUF_BYTES / sizeof(T))
#define SOFTMAX_EXP_BUF_SIZE (SOFTMAX_EXP_BUF_BYTES / sizeof(T))
#define SOFTMAX_SUM_BUF_SIZE (SOFTMAX_SUM_BUF_BYTES / sizeof(T))

template <typename IFAT> class IncreFlashAttentionAttenPreloadDD {
struct ExtraInfo {
    uint32_t loop = 0;
    uint32_t bIdx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t n2Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t curSInnerLoopTimes = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t attenMaskOffset = 0;
    uint64_t antiqKeyParamOffset = 0;
    uint64_t antiqValueParamOffset = 0;
    uint64_t antiqKeyParamOffsetPerToken = 0;
    uint64_t antiqValueParamOffsetPerToken = 0;
    uint64_t perChannelQuantOffset = 0ULL;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    uint32_t s2BatchOffset = 0;
    uint32_t s1Size = 0;
    uint32_t actS1Size = 0;
    uint32_t s2Size = 0;
    uint32_t gSize = 0;
    uint32_t mSize = 0;
    uint32_t mSizeV = 0;
    uint32_t mSizeVStart = 0;
    uint64_t actSeqLenQ = 0;
    bool isFirstSInnerLoop = false;
    bool isChangeBatch = false;
};

struct TaskContext {
    uint32_t bidx;
    uint32_t gidx;
    uint32_t s1idx;
    uint32_t s2idx;
    uint32_t s2loops;
    uint32_t s2SizeTail;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t isFirstLoop;
    uint32_t nidx;
    uint64_t actS1Size = 1;
    uint64_t actSeqLenQ = 0;
};

public:
    __aicore__ inline IncreFlashAttentionAttenPreloadDD(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *workspace, const IncreFlashAttentionTilingData *__restrict tiling,
                                __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix = false);
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                     __gm__ uint8_t *workspace);
    __aicore__ inline void InitAntiquant(__gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                        __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffse);
    __aicore__ inline void InitPostQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                        __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2);
    __aicore__ inline void Process();

    // 中间计算数据类型为float，高精度模式
    using T = float;

    __aicore__ inline void Bmm2ResCopyOut(const ExtraInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool KV_CONTINUOUS = IFAT::kvContinuous;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr LAYOUT KV_LAYOUT_T = IFAT::kvLayout;
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t PER_TOKEN_MODE = 1; // 伪量化: K V per-token
    static constexpr uint8_t PER_CHANNEL_TOKEN_MODE = 2; // 伪量化: K per-channel and V per-token
    static constexpr uint8_t ANTIQUANT_MODE = IFAT::antiquantMode;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr bool ANTIQUANT_PER_CHANNEL_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using ANTIQ_PARAMS_T = typename AscendC::Conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    using MM_OUT_T = typename AscendC::Conditional<ANTIQUANT, half, T>::type;
    using L0C_T = typename AscendC::Conditional<ANTIQUANT, int32_t, T>::type;

protected:
    const IncreFlashAttentionTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int32_t> blockTableGm;

    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;

    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqScaleGm;
    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqScaleGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    // out quant
    GlobalTensor<float> quantScale2Gm;
    GlobalTensor<float> quantOffset2Gm;
    GlobalTensor<bfloat16_t> quantScale2Bf16Gm;
    GlobalTensor<bfloat16_t> quantOffset2Bf16Gm;
    // workspace
    GlobalTensor<KV_T> queryPreProcessResGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<MM_OUT_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // kv_left_padding
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // queue
    TBuf<> inputBuf1; // 32K, inque
    TBuf<> inputBuf2; // 16K, inque
    TBuf<> outputBuf1; // 32K, outque
    TBuf<> outputBuf2; // 8K, outque
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 3;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;

    TBuf<TPosition::A1> tmpBufL1Q;
    LocalTensor<KV_T> qL1Buffers;
    TBuf<TPosition::A1> tmpBufL1KP;
    LocalTensor<KV_T> kpL1Buffers;
    TBuf<TPosition::A1> tmpBufL1V;
    LocalTensor<KV_T> vL1Buffers;

    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong;
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong;
    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<L0C_T> cL0TensorPingPong;

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K
    TBuf<> tmpBuff3; // 2K

    // 常驻tbuf
    TBuf<> vec2ResBuff; // 16k 伪量化场景
    TBuf<> antiqScaleBuff; // 4K
    TBuf<> antiqOffsetBuff; // 4K
    TBuf<> qAmaxBuff; // PRE_LOAD_NUM_DD * (2K + 256B)
    TBuf<> softmaxResAmaxBuff; // 2K + 256B
    TBuf<> qRowSumBuff; // 2K + 256B
    TBuf<> softmaxResRowSumBuff; // 2K + 256B
    TBuf<> softmaxMaxBuff; // PRE_LOAD_NUM_DD * 2K
    TBuf<> softmaxExpBuff; // PRE_LOAD_NUM_DD * 2K
    TBuf<> softmaxSumBuff; // PRE_LOAD_NUM_DD * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    // antiquant msd
    LocalTensor<T> aMaxBmm1Ub;
    LocalTensor<T> aMaxBmm2Ub;
    LocalTensor<T> softmaxScaleResRowSumUb;
    LocalTensor<half> vec2ResUb;
    LocalTensor<T> antiqScaleUb;
    LocalTensor<T> antiqOffsetUb;
    LocalTensor<T> qRowSumUb;

    static constexpr float quantScaleC1S1 = 1.0 / (1024);
    static constexpr float quantScaleC1S2 = 1.0 / (1024 * 254);
    static constexpr float quantScaleC2O1 = 1.0 / (1024 * 127);
    static constexpr float quantScaleC2O2 = 1.0 / (1024 * 254 * 127);

    // L1
    static constexpr uint32_t L1Q_BLOCK_SIZE = 64 * 1024; // 64k
    static constexpr uint32_t L1KP_BLOCK_SIZE = 64 * 1024; // 64k
    static constexpr uint32_t L1V_BLOCK_SIZE = 64 * 1024; // 64k

    // L0
    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1
    static constexpr uint32_t L1Q_EVENT0 = EVENT_ID0;
    static constexpr uint32_t L1KP_EVENT0 = EVENT_ID1;
    static constexpr uint32_t L1KP_EVENT1 = EVENT_ID2;
    static constexpr uint32_t L1KP_EVENT2 = EVENT_ID3;
    static constexpr uint32_t L1V_EVENT0 = EVENT_ID4;
    static constexpr uint32_t L1V_EVENT1 = EVENT_ID5;
    static constexpr uint32_t L1V_EVENT2 = EVENT_ID6;
    static constexpr uint32_t L1V_EVENT3 = EVENT_ID7;

    // m <> mte1
    static constexpr uint32_t L0A_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0A_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0B_EVENT0 = EVENT_ID5;
    static constexpr uint32_t L0B_EVENT1 = EVENT_ID6;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    uint32_t qL1BufIter = 0;
    uint32_t kpL1BufIter = -1;
    uint32_t vL1BufIter = -1;
    uint32_t aL0BufIter = 0;
    uint32_t bL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint64_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t SYNC_C2_V2_FLAG = 9;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM = 512 / sizeof(KV_T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM_THRESHLOD = 128 / sizeof(KV_T);
    MM_OUT_T antiquantExpandCoeff = KVINT4 ? 14.98 : 254;
    static constexpr T antiqCoeff1 = KVINT4 ? 7.49 : 127;
    static constexpr T antiqCoeff2 = 1 / antiqCoeff1;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    static constexpr T FP16_ATTEN_MASK_SCALAR_VALUE = -10000;           // 用于mask为fp16类型
    static constexpr uint32_t P_LOAD_TO_L1_ROW_NUM = 128 / sizeof(KV_T);
    static constexpr uint32_t KV_LOAD_TO_L1_ROW_NUM = 512 / sizeof(KV_T);
    bool antiqOffsetExistFlag = false;
    uint32_t msdIterNum = 0U;
    bool antiquantPerHeadFlag = false;
    bool antiquantParamsInPagedAttentionFlag = false;
    uint32_t antiquantPerTensorFlag = 0U;

    // kv_left_padding
    uint32_t kvPaddingFlag = 0;
    uint64_t kvPaddingBeginOffset = 0;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRE_LOAD_NUM_DD;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    __gm__ uint8_t *blocktablePtr = nullptr;

    // tilingdata
    uint64_t singleProcessSInnerSize = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;

    uint32_t vec1ResUbSize = 0U;
    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;
    uint32_t batchContinuous = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t gSize = 0ULL;

    uint64_t actS1Size = 1ULL;
    uint64_t gSizeSub = 0ULL;
    uint64_t gSizeTail = 0ULL;
    uint64_t s1SizeSub = 0ULL;
    uint64_t s1SizeTail = 0ULL;
    uint64_t gOuter = 1ULL;
    uint64_t s1Outer = 1ULL;
    uint64_t gIdx = 0ULL;
    uint32_t s1Idx = 0U;
    uint64_t mSizeVector = 0ULL;
    uint64_t mSizeVStart = 0ULL;
    uint64_t qSeqSize = 0ULL;

    uint64_t kvSeqSize = 0ULL;
    uint64_t headDim = 0ULL;
    uint64_t headDimAlign = 0ULL;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t s2BatchBaseOffset = 0;

    // 是否返回lse
    bool softmaxLseFlag = false;

    // attention mask
    bool attenMaskFlag = false;
    uint32_t selectWithByteMaskTmpMinSize = 0U;
    uint32_t attenMaskSizeAlign = 0U;
    // pse mask
    uint32_t pseMaskSizeAlign = 0U;
    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t valueOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t antiqParamOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t antiqKeyParamCoreOffsetPerToken = 0ULL;
    uint64_t antiqParamOffsetPerToken = 0ULL;
    uint64_t attenMaskSize = 0ULL;
    uint64_t antiqSeqSize = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint32_t s2IdxFD = 0U;
    uint64_t sInnerLoopSize = 0ULL;
    uint32_t actualCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenQ = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenDims = 0U;
    uint32_t actualLenQDims = 0U;
    // out quant
    bool isPerChnU8Out = false;
    bool isOutQuantTypeBf16 = false;
    float quantScale2Value = 0;
    float quantOffset2Value = 0;
    bool isQuantOffset2Exist = false;
    uint64_t perChannelQuantOffset = 0ULL;
    static constexpr float scaleC1 = 1024.0 / 127;
    static constexpr float scaleC2 = 1024.0;

    // 记录当前轮的bIdx nIdx s2Idx actualLen
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bn2s2LoopTimes = 0;
    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void CalcParams(uint32_t loop, ExtraInfo& info, TaskContext &task);
    __aicore__ inline void ComputeMm1(ExtraInfo& info);
    __aicore__ inline void ProcessVec1L(ExtraInfo& info);
    __aicore__ inline void ComputeMm2(ExtraInfo& info);
    __aicore__ inline void ProcessVec2L(ExtraInfo& info);
    __aicore__ inline void SetLoopTimes();
    __aicore__ inline void QueryPreProcessL(ExtraInfo& info);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t nCopyIdx, uint32_t nCopyRowCount, uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void LoadDataMm1A(ExtraInfo& info, LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor);

    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t mCopyIdx, uint32_t mCopyRowCount, uint32_t mActCopyRowCount, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount);
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount);
    __aicore__ inline void LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize);
    __aicore__ inline void LoadDataMm2B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor);

    bool curActSeqLenIsZero = false;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
    }
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParams();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline uint64_t GetQueryOffSetBySIdx(uint32_t bIdx, uint32_t sIdx, uint32_t n2Idx, uint32_t gIdx);
    __aicore__ inline void GetTndActualSeqLenQ(uint32_t bIdx, uint32_t s1Idx);
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline bool ComputeKVPaddingBeginOffset();

    __aicore__ inline void GetBN2Gid(const uint32_t bn2Idx);

    __aicore__ inline void CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount, uint32_t actualColumnCount);
    __aicore__ inline void AttenMaskCopyIn(const ExtraInfo& info);

    __aicore__ inline void CopyAntiquantScale(LocalTensor<T> &castUb, GlobalTensor<Q_T> srcGm, uint64_t offset);

    __aicore__ inline void CopyAntiquantParamsPerToken(GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset,
                                                       uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiquantParamsPerTokenHead(GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset,
                                                           uint32_t columnCount);
    __aicore__ inline void CopyAntiquantParamsPerTokenParamsPagedAttention(GlobalTensor<ANTIQ_PARAMS_T> srcGm,
                                                                           uint64_t offset, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiqQuery(LocalTensor<T> &queryCastUb, uint64_t qOffset, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb, LocalTensor<T> tmpAUb,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<half> &tmpA1, LocalTensor<half> &tmpA2,
                                                uint32_t calcSize, bool isFirst, uint64_t outOffset);
    __aicore__ inline void AntiquantMatmulPreProcess(ExtraInfo& info, GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb,
                                                     LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void AntiquantSoftmaxResPreProcess(ExtraInfo& info, GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb,
                                                         LocalTensor<T> tmpAFloorUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount);
    __aicore__ inline void DealQueryPreProcessBaseBlock(ExtraInfo& info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount);
    __aicore__ inline void DealQueryPreProcessBaseBlockPerToken(ExtraInfo& info, uint32_t startRow, uint32_t dealRowCount,
                                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void QueryPreProcessInner(ExtraInfo& info);

    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void Bmm1ComputeCommon(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void Bmm2ComputeCommon(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void SysPrefixBmm1Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void SysPrefixBmm2Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);

    __aicore__ inline void DealBmm1ResBaseBlock(ExtraInfo& info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlock(ExtraInfo& info, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlockPerToken(ExtraInfo& info, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMatmulResCombine(ExtraInfo& info, LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMatmulResCombineDD(ExtraInfo& info, LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount, float scaleC);
    __aicore__ inline void AntiquantMM2ResCombine(const ExtraInfo& info, LocalTensor<MM_OUT_T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec1Inner(ExtraInfo& info);
    __aicore__ inline void PreProcessVec1(uint32_t sInnerLoopIdx);

    __aicore__ inline void DealAntiqBmm2ResBaseBlock(ExtraInfo& info, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm2ResBaseBlockPerToken(ExtraInfo& info,  uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(ExtraInfo& info);

    __aicore__ inline void SoftmaxFlashV2Compute(ExtraInfo& info, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline bool IsSkipAttenMask(const ExtraInfo &info, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ElewiseCompute(ExtraInfo& info, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutNcon(const ExtraInfo& info, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm2CastAndCopyOut(const ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                              uint32_t columnCount, uint32_t actualColumnCount);

    template <typename RT>
    __aicore__ inline void DealInvalidRows(const ExtraInfo &info, LocalTensor<RT> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CombineSplitKVRes();
    __aicore__ inline void CopyAccumOutIn(uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void Bmm2FDDataCopyOut(const ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax, uint32_t startRow,
                                             uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow,
                                          uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void PostQuant(uint64_t perChannelQuantOffset, LocalTensor<T> &bmm2ResUb, LocalTensor<int8_t> &bmm2ResUbInt8, uint32_t startRow,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void SysPrefixInitAllZeroOutput();
    __aicore__ inline void InitAllZeroInt8Output(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);

    __aicore__ inline void CopyFixedUbToGm(const GlobalTensor<T> &dst, const LocalTensor<T> &src, size_t size);
    __aicore__ inline void SoftmaxLseOutput(LocalTensor<T> &lse);

    __aicore__ inline void DealKvInt4ColumnOdd(LocalTensor<T> mmResUb, uint32_t dealRowCount,
        uint32_t columnCount, uint32_t actualColumnCount);
};

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSize;
    sInnerLoopTimes = tilingData->increFlashAttentionSingleCoreParams.sInnerLoopTimes;
    singleProcessSInnerSizeTail = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSizeTail;
    usedCoreNum = tilingData->increFlashAttentionSingleCoreParams.usedCoreNum;
    formerCoreNum = tilingData->increFlashAttentionSingleCoreParams.formerCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    vec1ResUbSize = mmResUbSize * msdIterNum;
    mmResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.mmResUbSize;
    bmm2ResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    gSize = tilingData->baseParams.nNumOfQInOneGroup;
    qSeqSize = tilingData->baseParams.qSeqSize;
    kvSeqSize = tilingData->baseParams.seqSize;
    headDim = tilingData->baseParams.headSize;
    batchContinuous = tilingData->baseParams.batchContinuousFlag;
    msdIterNum = tilingData->baseParams.msdIterNum;
    antiquantPerTensorFlag = tilingData->baseParams.antiquantPerTensorFlag;
    antiquantPerHeadFlag = (tilingData->baseParams.antiquantPerHeadFlag == 1);
    antiquantParamsInPagedAttentionFlag = (tilingData->baseParams.antiquantParamsInPagedAttentionFlag == 1);

    headDimAlign = Align(headDim, BYTE_BLOCK);

    gSizeSub = tilingData->increFlashAttentionSingleCoreParams.groupSplitSize; //切块大小Gi
    gOuter = (gSize + gSizeSub - 1) / gSizeSub;
    gSizeTail = gSize - (gOuter - 1) * gSizeSub;
    s1SizeSub = tilingData->increFlashAttentionSingleCoreParams.s1SplitSize; //切块大小Si
    s1Outer = (qSeqSize + s1SizeSub - 1) / s1SizeSub;
    s1SizeTail = qSeqSize - (s1Outer - 1) * s1SizeSub;

    attenMaskFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attenMaskSize = tilingData->baseParams.attenMaskSize;
    selectWithByteMaskTmpMinSize = tilingData->baseParams.selectWithByteMaskTmpMinSize;

    antiqSeqSize = tilingData->baseParams.antiqSeqSize;

    kvPaddingFlag = tilingData->baseParams.kvPaddingFlag;

    // out quant
    isPerChnU8Out = tilingData->outputParams.isPerChnOut == 0 ? false : true;
    isOutQuantTypeBf16 = tilingData->outputParams.isOutQuantTypeBf16 == 0 ? false : true;

    // 是否输出lse
    softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        pipe->InitBuffer(inputBuf1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(inputBuf2, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(outputBuf1, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(vec2ResBuff, BUFFER_SIZE_BYTE_16K);
        
        pipe->InitBuffer(outputBuf2, BUFFER_SIZE_BYTE_8K);

        // tmpBuff
        pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff3, BUFFER_SIZE_BYTE_2K);

        // 常驻buffer
        pipe->InitBuffer(antiqScaleBuff, BUFFER_SIZE_BYTE_4K);
        pipe->InitBuffer(antiqOffsetBuff, BUFFER_SIZE_BYTE_4K);
        // 预留空间2K = 64 * 32，支持 gSize = 64
        // brcb 操作每次操作8*32字节输出，startRow接近64时，
        // 输出最多可能超出2k空间7*32字节， 这里预留256B防止越界
        pipe->InitBuffer(qAmaxBuff, Q_AMAX_BUF_BYTES * PRE_LOAD_NUM_DD);
        pipe->InitBuffer(softmaxResAmaxBuff, P_AMAX_BUF_BYTES * PRE_LOAD_NUM_DD); // perToken使用,perChannel不需要
        pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_MAX_BUF_BYTES * PRE_LOAD_NUM_DD);
        pipe->InitBuffer(softmaxExpBuff, SOFTMAX_EXP_BUF_BYTES * PRE_LOAD_NUM_DD);
        pipe->InitBuffer(softmaxSumBuff, SOFTMAX_SUM_BUF_BYTES * PRE_LOAD_NUM_DD);
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
        pipe->InitBuffer(softmaxMaxDefaultBuff, SOFTMAX_MAX_BUF_BYTES);
        pipe->InitBuffer(softmaxSumDefaultBuff, SOFTMAX_SUM_BUF_BYTES);
#else
        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_512B);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_512B);
#endif
    } else {

        // L1
        pipe->InitBuffer(tmpBufL1Q, L1Q_BLOCK_SIZE);
        qL1Buffers = tmpBufL1Q.Get<KV_T>();
        pipe->InitBuffer(tmpBufL1KP, 3 * L1KP_BLOCK_SIZE);
        kpL1Buffers = tmpBufL1KP.Get<KV_T>();
        pipe->InitBuffer(tmpBufL1V, 4 * L1V_BLOCK_SIZE);
        vL1Buffers = tmpBufL1V.Get<KV_T>();

        pipe->InitBuffer(tmpBufL0A, 2 * L0A_PP_SIZE);
        aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
        // L0B
        pipe->InitBuffer(tmpBufL0B, 2 * L0B_PP_SIZE);
        bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
        // L0C
        pipe->InitBuffer(tmpBufL0C, 2 * L0C_PP_SIZE);
        cL0TensorPingPong = tmpBufL0C.Get<L0C_T>();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths)
{
    actualLenQDims = tilingData->baseParams.actualLenQDims;
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, actualLenQDims);
    }
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitAllZeroInt8Output(uint32_t bIdx, uint32_t n2Idx)
{
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenPreloadDD<IFAT>::GetQueryOffSetBySIdx(uint32_t bIdx, uint32_t sIdx,
    uint32_t n2Idx, uint32_t gIdx)
{
    uint64_t sBase = bIdx * qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        sBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
    }
    return (sBase + sIdx) * qHeadNum * headDim + n2Idx * gSize * headDim + gIdx * headDim;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t copySize = gSize * headDim;
    if constexpr (POST_QUANT) { // out int8
        InitAllZeroInt8Output(bIdx, n2Idx);
    } else {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            uint64_t attenOutOffset = bIdx * kvHeadNum * gSize * qSeqSize * headDim +           //B轴偏移
                                      n2Idx * gSize * qSeqSize * headDim;     //N2轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], gSize * qSeqSize * headDim, 0);
        } else if constexpr (LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::TND) {
            for (int s1Idx = 0; s1Idx < curActualSeqLenQ; s1Idx++) {
                uint64_t attenOutOffset = GetQueryOffSetBySIdx(bIdx, s1Idx, n2Idx, 0);
                matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], gSize * headDim, 0);
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::GetTndActualSeqLenQ(uint32_t bIdx, uint32_t s1Idx)
{
    if (bIdx > 0) {
        curActualSeqLenQ = actualSeqLengthsGmQ.GetValue(bIdx) - actualSeqLengthsGmQ.GetValue(bIdx - 1);
    } else if (bIdx == 0) {
        curActualSeqLenQ = actualSeqLengthsGmQ.GetValue(0);
    }

    if (s1Idx * s1SizeSub >= curActualSeqLenQ) {
        actS1Size = 0;
    } else if ((s1Idx + 1) * s1SizeSub > curActualSeqLenQ) {
        actS1Size = curActualSeqLenQ - s1Idx * s1SizeSub;
    } else {
        actS1Size = s1SizeSub;
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }

    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        actS1Size = (s1Idx == s1Outer - 1) ? s1SizeTail : s1SizeSub;
        curActualSeqLenQ = qSeqSize;
    } else if constexpr (LAYOUT_T == LAYOUT::TND) {
        GetTndActualSeqLenQ(bIdx, s1Idx);
    } else {
        actS1Size = qSeqSize;
        curActualSeqLenQ = qSeqSize;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::GetBN2Gid(const uint32_t bn2gIdx)
{
    if constexpr (FLASH_DECODE) {
        bIdx = aiCoreIdx / (kvHeadNum * splitKVNum);
        n2Idx = (aiCoreIdx / splitKVNum) % kvHeadNum;
        s2IdxFD = aiCoreIdx % splitKVNum;
    } else if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) {
        uint32_t bs1n2 = beforeBlockSplitBn2Nums + bn2gIdx;
        uint32_t s1n2 = bs1n2 % (kvHeadNum * s1Outer);
        bIdx = bs1n2 / (kvHeadNum * s1Outer);
        s1Idx = s1n2 / kvHeadNum;
        n2Idx = s1n2 % kvHeadNum;
    } else {
        uint32_t bn2g = beforeBlockSplitBn2Nums + bn2gIdx;
        uint32_t n2g = bn2g % (kvHeadNum * gOuter);
        bIdx = bn2g / (kvHeadNum * gOuter);
        gIdx = n2g % gOuter;
        n2Idx = n2g / gOuter;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, n2Idx);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::UpdateInnerLoopCond()
{
    if (curActualSeqLen == 0) {
        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    int32_t remainSinnerSize = (int32_t)curActualSeqLen;
    int32_t computeSinnerSize = (int32_t)curActualSeqLen;
    if constexpr (FLASH_DECODE) {
        remainSinnerSize = (int32_t)curActualSeqLen - sInnerLoopSize * s2IdxFD;
        computeSinnerSize = remainSinnerSize >= sInnerLoopSize ? sInnerLoopSize : remainSinnerSize;
        if (aiCoreIdx >= batchSize * kvHeadNum * splitKVNum) {
            remainSinnerSize = 0;
        }
    }
    if (remainSinnerSize > 0) {
        if (computeSinnerSize <= singleProcessSInnerSize) {
            singleProcessSInnerSizeTail = computeSinnerSize;
            sInnerLoopTimes = 1;
        } else {
            sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
        }
    } else {
        sInnerLoopTimes = 0;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenPreloadDD<IFAT>::SeqLenFromTensorList(uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <typename IFAT>
__aicore__ inline bool IncreFlashAttentionAttenPreloadDD<IFAT>::ComputeKVPaddingBeginOffset()
{
    if (kvPaddingFlag != 1) {
        return true;
    }
    int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }

    int64_t startPosition = kvSeqSize - paddingSize - curActualSeqLen;

    if (startPosition < 0) {
        return false;
    }

    kvPaddingBeginOffset = static_cast<uint64_t>(startPosition);
    return true;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const IncreFlashAttentionTilingData *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }
    // init tiling data
    tilingData = tiling;

    InitTilingData();
    // 初始化计算参数
    if constexpr (FLASH_DECODE) {
        InitCalcParams();
    } else {
        InitCalcParamsEach();
    }

    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;
    blocktablePtr = blockTable;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    if (batchContinuous) {
        InitKeyGm(0);
        InitValueGm(0);
    }

    if (pipe != nullptr) {
        InitBuffers();
    }

    if (attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }

    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    if (kvPaddingFlag == 1) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t *)kvPaddingSize);
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    if ASCEND_IS_AIV {
        softmaxMaxUb = softmaxMaxBuff.Get<T>();
        softmaxSumUb = softmaxSumBuff.Get<T>();
        softmaxExpUb = softmaxExpBuff.Get<T>();
        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
    }

    uint64_t offset = 0;
    if constexpr (ANTIQUANT) {
        if constexpr (KVINT4) {
            queryPreProcessResGm.SetGlobalBuffer(
                (__gm__ KV_T *)(workspace + offset + (aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(KV_T) >> 1)));
            offset += (GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(KV_T) >> 1);
        } else {
            queryPreProcessResGm.SetGlobalBuffer(
                (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(KV_T)));
            offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(KV_T);
        }
    }

    mm1ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T);

    if constexpr (KVINT4) {
        vec1ResGm.SetGlobalBuffer(
            (__gm__ KV_T *)(workspace + offset + (aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T) >> 1)));
        offset += (GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T) >> 1);
    } else {
        vec1ResGm.SetGlobalBuffer(
            (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T);
    }


    mm2ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T);

    if constexpr (ANTIQUANT) {
        vec2ResGm.SetGlobalBuffer(
                (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio* bmm2ResUbSize * sizeof(MM_OUT_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T);
    } else {
        vec2ResGm.SetGlobalBuffer(
                (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio* bmm2ResUbSize * sizeof(T)));
        offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T);
    }
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(T);

    // GM for pse


    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if (softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);

    if (tilingData->baseParams.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);

    if (tilingData->baseParams.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitQuant(
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *workspace)
{
    InitAntiquant(antiquantScale, antiquantOffset, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset);
    InitPostQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitAntiquant(
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset)
{
    if ASCEND_IS_AIC {
        return;
    }
    if constexpr (ANTIQUANT) {
        if (keyAntiquantScale == nullptr) {
            int64_t antiValueOffsetInitPos = kvHeadNum * headDim;
            if (antiquantPerTensorFlag == 1) {
                antiValueOffsetInitPos = 1;
            }
            if constexpr (ANTIQUANT_PER_TOKEN) {
                antiValueOffsetInitPos = batchSize * antiqSeqSize;
            }
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)antiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer(((__gm__ ANTIQ_PARAMS_T *)antiquantScale) + antiValueOffsetInitPos);
            antiqOffsetExistFlag = (antiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)antiquantOffset);
                valueAntiqOffsetGm.SetGlobalBuffer(((__gm__ ANTIQ_PARAMS_T *)antiquantOffset) + antiValueOffsetInitPos);
            }
        } else {
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)keyAntiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)valueAntiquantScale);
            antiqOffsetExistFlag = (keyAntiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)keyAntiquantOffset);
                valueAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)valueAntiquantOffset);
            }
        }
        aMaxBmm1Ub = qAmaxBuff.Get<T>();
        aMaxBmm2Ub = softmaxResAmaxBuff.Get<T>();

        if constexpr (ANTIQUANT_PER_TOKEN) {
            vec2ResUb = vec2ResBuff.Get<half>();
        } else if constexpr (ANTIQUANT_PER_CHANNEL) {
            vec2ResUb = vec2ResBuff.Get<half>();
            antiqScaleUb = antiqScaleBuff.Get<T>();
            antiqOffsetUb = antiqOffsetBuff.Get<T>();
        }
    }
}
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitPostQuant(__gm__ uint8_t *deqScale1,
    __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
{
    if constexpr (POST_QUANT) {
        if (!isPerChnU8Out && !isOutQuantTypeBf16) {
            if (quantScale2 != nullptr) {
                quantScale2Gm.SetGlobalBuffer((__gm__ float *)quantScale2);
                quantScale2Value = quantScale2Gm.GetValue(0);
            }
            if (quantOffset2 != nullptr) {
                quantOffset2Gm.SetGlobalBuffer((__gm__ float *)quantOffset2);
                quantOffset2Value = quantOffset2Gm.GetValue(0);
            } else {
                quantOffset2Value = 0;
            }
        }
        if (quantScale2 != nullptr && !isPerChnU8Out && isOutQuantTypeBf16) {
            quantScale2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantScale2);
            quantScale2Value = ToFloat(quantScale2Bf16Gm.GetValue(0));
        }
        if (!isPerChnU8Out && isOutQuantTypeBf16) {
            if (quantOffset2 != nullptr) {
                quantOffset2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantOffset2);
                quantOffset2Value = ToFloat(quantOffset2Bf16Gm.GetValue(0));
            } else {
                quantOffset2Value = 0;
            }
        }

        if (isPerChnU8Out && !isOutQuantTypeBf16) {
            if (quantScale2 != nullptr) {
                quantScale2Gm.SetGlobalBuffer((__gm__ float *)quantScale2);
            }
            if (quantOffset2 != nullptr) {
                isQuantOffset2Exist = true;
                quantOffset2Gm.SetGlobalBuffer((__gm__ float *)quantOffset2);
            }
        }

        if (isPerChnU8Out && isOutQuantTypeBf16) {
            if (quantScale2 != nullptr) {
                quantScale2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantScale2);
            }
            if (quantOffset2 != nullptr) {
                isQuantOffset2Exist = true;
                quantOffset2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantOffset2);
            }
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitCalcParams()
{
    bn2LoopTimes = 1U;
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个increFlashAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间。
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *coreSidxEnd = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
#else
    uint32_t coreSidxEnd[50];
    copy_data_align64((uint8_t *)coreSidxEnd, (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreSidxEnd),
                      sizeof(coreSidxEnd));
#endif
    bn2LoopTimes = coreSidxEnd[aiCoreIdx + 1] - coreSidxEnd[aiCoreIdx];
    beforeBlockSplitBn2Nums = coreSidxEnd[aiCoreIdx];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::AttenMaskCopyIn(uint64_t offset,
                                                                                     uint32_t dealRowCount,
                                                                                     uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputBuf1.Get<bool>();
    attenMaskSizeAlign = Align(actualColumnCount, 32U);
    maskUb.SetSize(dealRowCount * attenMaskSizeAlign);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 32 == 0) {
        DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(bool);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualColumnCount * sizeof(bool);
        intriParams.blockCount = 1;
        intriParams.dstStride = (attenMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<bool> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (attenMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        DataCopyPad(maskUb, attenMaskBoolGm[offset], intriParams, padParams);
    }
#else
    DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
#endif
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::AttenMaskCopyIn(const ExtraInfo& info)
{
    uint32_t offset;
    {
        int32_t delta =
            info.s1Idx * s1SizeSub - info.s2Idx * singleProcessSInnerSize + info.s2Size - info.actSeqLenQ; // s1idx = 0
        if (delta < 0) {
            offset = (-delta) < (int32_t)info.s1Size ? (-delta) : info.s1Size; //min (-delta, s1Size)
        } else {
            offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                    ATTENMASK_STRIDE; // min(delta, s2inner)
        }
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = info.s1Size;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) / 32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    LocalTensor<bool> maskUb = inputBuf1.Get<bool>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(maskUb, attenMaskBoolGm[offset], dataCopyParams);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyAntiquantScale(LocalTensor<T> &castUb,
                                                                                        GlobalTensor<Q_T> srcGm,
                                                                                        uint64_t offset)
{
    if (antiquantPerHeadFlag || antiquantPerTensorFlag == 1) {
        if constexpr (AscendC::IsSameType<Q_T, half>::value) {
            Duplicate(castUb, static_cast<T>(srcGm.GetValue(offset)), headDimAlign);
        } else if constexpr (AscendC::IsSameType<Q_T, bfloat16_t>::value) {
            Duplicate(castUb, ToFloat(srcGm.GetValue(offset)), headDimAlign);
        }
    } else {
        uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<Q_T> copyInPadParams;
        // antiq scale copy in
        copyInParams.blockCount = 1;
        copyInParams.blockLen = headDim * sizeof(Q_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (headDimAlign - headDim) / qTypeElementSize;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (headDimAlign - headDim) % qTypeElementSize;
        copyInPadParams.paddingValue = 0;

        LocalTensor<Q_T> inputUb = inputBuf2.Get<Q_T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        DataCopyPad(inputUb, srcGm[offset], copyInParams, copyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        Cast(castUb, inputUb, RoundMode::CAST_NONE, headDim);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyAntiquantParamsPerTokenHead(
    GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset, uint32_t columnCount) {
    LocalTensor<ANTIQ_PARAMS_T> dstUb = inputBuf1.Get<ANTIQ_PARAMS_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(dstUb, srcGm[offset], columnCount);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
}


template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyAntiquantParamsPerTokenParamsPagedAttention(
    GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset, uint32_t actualColumnCount)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyAntiquantParamsPerToken(
    GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset, uint32_t columnCount, uint32_t actualColumnCount)
{
    if (antiquantPerHeadFlag) {
        CopyAntiquantParamsPerTokenHead(srcGm, offset, columnCount);
        return;
    }
    if (antiquantParamsInPagedAttentionFlag) {
        CopyAntiquantParamsPerTokenParamsPagedAttention(srcGm, offset, actualColumnCount);
        return;
    }
    uint32_t paramsTypeElementSize = BYTE_BLOCK / sizeof(ANTIQ_PARAMS_T);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = 1;
    copyInParams.blockLen = actualColumnCount * sizeof(ANTIQ_PARAMS_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % paramsTypeElementSize;
    copyInPadParams.paddingValue = 0;
    LocalTensor<ANTIQ_PARAMS_T> paramsUb = inputBuf2.Get<ANTIQ_PARAMS_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    DataCopyPad(paramsUb, srcGm[offset], copyInParams, copyInPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::CopyAntiqQuery(LocalTensor<T> &queryCastUb, uint64_t qOffset,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = actualColumnCount * sizeof(Q_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (columnCount - actualColumnCount) / qTypeElementSize;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % qTypeElementSize;
    copyInPadParams.paddingValue = 0;

    LocalTensor<Q_T> inputUb = inputBuf1.Get<Q_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopyPad(inputUb, queryGm[qOffset], copyInParams, copyInPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    Cast(queryCastUb, inputUb, RoundMode::CAST_NONE, dealRowCount * columnCount);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb,
                                                        LocalTensor<T> tmpAUb, uint32_t dealRowCount,
                                                        uint32_t columnCount, uint32_t actualColumnCount)
{
    Abs(tmpAUb, srcUb, dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpRowMaxUb = tmpBuff3.Get<T>();
    RowMaxForLongColumnCount(tmpRowMaxUb, tmpAUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    Brcb(tmpAMaxRes, tmpRowMaxUb, (dealRowCount + 7) / 8, {1, 8});
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<half> &tmpA1,
                                                                   LocalTensor<half> &tmpA2, uint32_t calcSize,
                                                                   bool isFirst, uint64_t outOffset)
{
    if (!isFirst) {
        Sub(tmpA2, tmpA1, tmpA2, calcSize);
        PipeBarrier<PIPE_V>();
        Muls(tmpA2, tmpA2, antiquantExpandCoeff, calcSize);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<KV_T> aResOutUbI8 = outputBuf1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    Cast(aResOutUbI8, tmpA2, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (KVINT4) {
        GlobalTensor<int8_t> dstGmI8;
        dstGmI8.SetGlobalBuffer(((__gm__ int8_t *)(dstGm[outOffset].GetPhyAddr())));
        DataCopy<int8_t>(dstGmI8, aResOutUbI8, calcSize >> 1);
    } else {
        DataCopy(dstGm[outOffset], aResOutUbI8, calcSize);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::AntiquantMatmulPreProcess(ExtraInfo& info,
    GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = info.mSize * columnCount;
    uint64_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    LocalTensor<T> tmpAMaxRes = aMaxResUb[startRow * BLOCK_ELEMENT_NUM];
    AbsRowMax(tmpAMaxRes, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // 128/(1.001*Amax)*A
    Duplicate(tmpAFloorUb, antiqCoeff1, dealRowCount * BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    Div(tmpAFloorUb, tmpAFloorUb, tmpAMaxRes, dealRowCount * BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    RowMuls(srcUb, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // step4: 取出Qtmp整数部分
    Cast(tmpAFloorUb, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> tmpAFloorUbFp16 = tmpAFloorUb.template ReinterpretCast<half>();
    tmpAFloorUbFp16.SetSize(tmpAFloorUb.GetSize());
    Cast(tmpAFloorUbFp16, tmpAFloorUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    // step5: 将Qtmp转成fp16
    LocalTensor<half> srcUbFp16 = srcUb.template ReinterpretCast<half>();
    srcUbFp16.SetSize(srcUb.GetSize());
    Cast(srcUbFp16, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUbFp16, tmpAFloorUbFp16, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::AntiquantSoftmaxResPreProcess(ExtraInfo &info,
    GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = info.mSize * columnCount;
    uint64_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    Muls(srcUb, srcUb, antiqCoeff1, calcSize);
    PipeBarrier<PIPE_V>();

    Cast(tmpAFloorUb, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> tmpAFloorUbFp16 = tmpAFloorUb.template ReinterpretCast<half>();
    tmpAFloorUbFp16.SetSize(tmpAFloorUb.GetSize());
    Cast(tmpAFloorUbFp16, tmpAFloorUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    // step5: 将Qtmp转成fp16
    LocalTensor<half> srcUbFp16 = srcUb.template ReinterpretCast<half>();
    srcUbFp16.SetSize(srcUb.GetSize());
    Cast(srcUbFp16, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUbFp16, tmpAFloorUbFp16, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::DealQueryPreProcessBaseBlock(ExtraInfo& info,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t bufferId = info.bn2IdxInCurCore % PRE_LOAD_NUM_DD;
    LocalTensor<T> queryUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();
    uint64_t qOffset = info.tensorAOffset + (mSizeVStart + startRow) * actualColumnCount;

    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) {
        if (kvHeadNum != 1 && gOuter == 1) {
            uint32_t startS1Idx = (mSizeVStart + startRow) / info.gSize;
            uint32_t startGOfffset = (mSizeVStart + startRow) % info.gSize;
            uint32_t endS1Idx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
            uint32_t curStartRow = (mSizeVStart + startRow);
            uint32_t curDealRowCount = 0;
            uint32_t ubOffset = 0;
            for (uint32_t curS1idx = startS1Idx; curS1idx <= endS1Idx; curS1idx++) {
                qOffset = GetQueryOffSetBySIdx(info.bIdx, curS1idx + info.s1Idx * s1SizeSub, info.n2Idx, startGOfffset);
                if (curS1idx != endS1Idx) {
                    curDealRowCount = (curS1idx + 1) * gSize - curStartRow;
                }
                else {
                    curDealRowCount = mSizeVStart + startRow + dealRowCount - curStartRow;
                }
                ubOffset = (curStartRow - mSizeVStart) * columnCount;
                LocalTensor<T> curQueryUb = queryUb[ubOffset];
                CopyAntiqQuery(curQueryUb, qOffset, curDealRowCount, columnCount, actualColumnCount);
                PipeBarrier<PIPE_V>();
                curStartRow += curDealRowCount;
                startGOfffset = 0;
            }
        }
        else {
            CopyAntiqQuery(queryUb, qOffset, dealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
        }
    }
    else {
        CopyAntiqQuery(queryUb, qOffset, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
    }
    // mul scale
    VecMulMat(queryUb, antiqScaleUb, queryUb, dealRowCount, columnCount, actualColumnCount);

    PipeBarrier<PIPE_V>();

    // A pre process
    size_t dstOffset = bufferId * bmm2ResUbSize + mSizeVStart * columnCount;

    AntiquantMatmulPreProcess(info, queryPreProcessResGm[dstOffset], aMaxBmm1Ub[bufferId * Q_AMAX_BUF_SIZE],
        queryUb, aFloorUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::DealQueryPreProcessBaseBlockPerToken(ExtraInfo& info,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t bufferId = info.bn2IdxInCurCore % PRE_LOAD_NUM_DD;
    LocalTensor<T> queryUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();
    uint64_t queryOffset = info.tensorAOffset + (mSizeVStart + startRow) * actualColumnCount;

    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::TND) {
        if (kvHeadNum != 1 && gOuter == 1) {
            uint32_t curDealRowCount = 0;
            uint32_t ubOffset = 0;
            uint32_t startS1Idx = (mSizeVStart + startRow) / info.gSize;
            uint32_t startGOfffset = (mSizeVStart + startRow) % info.gSize;
            uint32_t endS1Idx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
            uint32_t curStartRow = (mSizeVStart + startRow);
            for (uint32_t curS1idx = startS1Idx; curS1idx <= endS1Idx; curS1idx++) {
                queryOffset = GetQueryOffSetBySIdx(info.bIdx, curS1idx + info.s1Idx * s1SizeSub, info.n2Idx, startGOfffset);
                if (curS1idx != endS1Idx) {
                    curDealRowCount = (curS1idx + 1) * gSize - curStartRow;
                }
                else {
                    curDealRowCount = mSizeVStart + startRow + dealRowCount - curStartRow;
                }
                ubOffset = (curStartRow - mSizeVStart) * columnCount;
                LocalTensor<T> curQueryUb = queryUb[ubOffset];
                CopyAntiqQuery(curQueryUb, queryOffset, curDealRowCount, columnCount, actualColumnCount);
                PipeBarrier<PIPE_V>();
                curStartRow += curDealRowCount;
                startGOfffset = 0;
            }
        }
        else {
            CopyAntiqQuery(queryUb, queryOffset, dealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
        }
    }
    else {
        CopyAntiqQuery(queryUb, queryOffset, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
    }

    // A pre process
    size_t dstOffset = bufferId * bmm2ResUbSize + mSizeVStart * columnCount;

    AntiquantMatmulPreProcess(info, queryPreProcessResGm[dstOffset], aMaxBmm1Ub[bufferId * Q_AMAX_BUF_SIZE],
        queryUb, aFloorUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::QueryPreProcessInner(ExtraInfo& info)
{
    if (!info.isFirstSInnerLoop) {
        return;
    }

    if constexpr (ANTIQUANT_PER_CHANNEL) {
        CopyAntiquantScale(antiqScaleUb, keyAntiqScaleGm, info.antiqKeyParamOffset);
        if (softmaxLseFlag && antiqOffsetExistFlag) {
            CopyAntiquantScale(antiqOffsetUb, keyAntiqOffsetGm, info.antiqKeyParamOffset);
        }
    }

    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }
    uint32_t loopCount = (mSizeVector + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSizeVector - (loopCount - 1) * mSplitSize;

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT_PER_CHANNEL) {
            DealQueryPreProcessBaseBlock(info, i * mSplitSize, dealSize, headDimAlign, headDim);
        } else if constexpr (ANTIQUANT_PER_TOKEN) {
            DealQueryPreProcessBaseBlockPerToken(info, i * mSplitSize, dealSize, headDimAlign, headDim);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseSum = inputBuf2.Get<T>();
    LocalTensor<T> lseMax = inputBuf1.Get<T>();
    uint64_t combineLoopOffset;

    combineLseOffset = ((bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize + startRow) * FP32_ONE_BLOCK_SIZE;
    combineLoopOffset = qSeqSize * gSize * FP32_ONE_BLOCK_SIZE;
    uint64_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    for (uint32_t i = 0; i < actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset], dealRowCountAlign);  // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset], dealRowCountAlign);
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyAccumOutIn(uint32_t splitKVIndex,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = inputBuf1.Get<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlign - headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlign - headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    combineAccumOutOffset =
        (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim + startRow * headDim;
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax,
                                                                uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMaxUb = softmaxMaxUb[0];
    LocalTensor<T> lseSumUb = softmaxSumUb[0];
    LocalTensor<T> lseExpUb = tmpBuff1.Get<T>();

    // lseLocal的shape为[actualCombineLoopSize, dealRowCount * FP32_ONE_BLOCK_SIZE]
    Duplicate(lseMaxUb, -FLOAT_MAX, dealRowCount * FP32_ONE_BLOCK_SIZE);
    Duplicate(lseSumUb, FLOAT_ZERO, dealRowCount * FP32_ONE_BLOCK_SIZE);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Max(lseMaxUb, lseMaxUb, lseMax[i * dealRowCount * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Sub(lseExpUb[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseMax[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseMaxUb,
            dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    PipeBarrier<PIPE_V>();
    Exp(lseExpUb, lseExpUb, actualCombineLoopSize * dealRowCount * FP32_ONE_BLOCK_SIZE);
    PipeBarrier<PIPE_V>();

    Mul(lseSum, lseSum, lseExpUb, actualCombineLoopSize * dealRowCount * FP32_ONE_BLOCK_SIZE);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Add(lseSumUb, lseSumUb, lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Div(lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseSumUb,
            dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    PipeBarrier<PIPE_V>();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                                             uint32_t startRow, uint32_t dealRowCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    repeatParams.dstRepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    int32_t dtypeMask = 256 / sizeof(float);
    int32_t mulLoop = headDimAlign / dtypeMask;
    int32_t mulRemain = headDimAlign % dtypeMask;

    // 第一次，mul结果直接放到dst里
    CopyAccumOutIn(0, startRow, dealRowCount);
    LocalTensor<T> accumOutLocal = inputBuf1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    for (int i = 0; i < mulLoop; i++) {
        Mul(dst[i * dtypeMask], lseLocal, accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
    }
    if (mulRemain > 0) {
        Mul(dst[mulLoop * dtypeMask], lseLocal, accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount,
            repeatParams);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);

    for (uint32_t j = 1; j < actualCombineLoopSize; ++j) {
        CopyAccumOutIn(j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = inputBuf1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
        for (int i = 0; i < mulLoop; i++) {
            Mul(accumOutLocal[i * dtypeMask], lseLocal[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(accumOutLocal[mulLoop * dtypeMask], lseLocal[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount, repeatParams);
        }
        PipeBarrier<PIPE_V>();
        Add(dst, dst, accumOutLocal, dealRowCount * headDimAlign);
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    if constexpr (!POST_QUANT) {
        LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuf1.Get<OUT_T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * headDimAlign);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * headDimAlign);
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, headDimAlign, headDim);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    } else {
        LocalTensor<OUT_T> bmm2ResUbInt8 = outputBuf1.Get<OUT_T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
        bmm2ResUbInt8.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        PostQuant(perChannelQuantOffset, accumOutLocal, bmm2ResUbInt8, startRow, dealRowCount, headDimAlign, headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        Bmm2DataCopyOut(attenOutOffset, bmm2ResUbInt8, startRow, dealRowCount, headDimAlign, headDim);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CombineSplitKVRes()
{
    if (curActualSeqLen != 0) {
        uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_16K / (BYTE_BLOCK * splitKVNum);
        uint32_t gSplitSizeAccumOut = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
        // 取两者较小的，用来切g，保证ub够用
        uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
        gSplitSize = (gSplitSize > gSize) ? gSize : gSplitSize; // 最大为gSize
        uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

        // 尾块与非尾块都使用这些ub，减少处理次数
        for (uint32_t i = 0, actualGSplitSize = gSplitSize; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if ((i + 1) == loopCount) {
                actualGSplitSize = tailSplitSize;
            }
            CopyLseIn(startRow, actualGSplitSize);
            LocalTensor<T> lseSum = inputBuf2.Get<T>();
            LocalTensor<T> lseMax = inputBuf1.Get<T>();
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
            ComputeScaleValue(lseSum, lseMax, startRow, actualGSplitSize);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);

            uint32_t gSplitBmm2UbSize = headDimAlign * actualGSplitSize;
            LocalTensor<T> tmp1 = tmpBuff1.Get<T>(gSplitBmm2UbSize);
            ReduceFinalRes(tmp1, lseSum, startRow, actualGSplitSize);

            CopyFinalResOut(tmp1, startRow, actualGSplitSize);
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::FlashDecodeCompute()
{
    bIdx = tmpBlockIdx / kvHeadNum;
    n2Idx = tmpBlockIdx % kvHeadNum;
    attenOutOffset = bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
    perChannelQuantOffset = n2Idx * headDim * gSize;
    mSizeVStart = 0;
    if (tmpBlockIdx >= batchSize * kvHeadNum) {
        return;
    }

    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }

    actualCombineLoopSize = (curActualSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;

    CombineSplitKVRes();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxSumUb,
                                                                          LocalTensor<T> &softmaxMaxUb)
{
    size_t size = mSizeVector * FP32_ONE_BLOCK_SIZE;
    size_t offset = bIdx * kvHeadNum * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE +
                    n2Idx * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE + 
                    s2IdxFD * gSize * FP32_ONE_BLOCK_SIZE + mSizeVStart * FP32_ONE_BLOCK_SIZE;
    CopyFixedUbToGm(lseSumFdGm[offset], softmaxSumUb, size);
    CopyFixedUbToGm(lseMaxFdGm[offset], softmaxMaxUb, size);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::SoftmaxLseCopyOut(LocalTensor<T> &softmaxSumUb,
                                                                                       LocalTensor<T> &softmaxMaxUb)
{
}

template <typename IFAT>
__aicore__ inline bool
IncreFlashAttentionAttenPreloadDD<IFAT>::IsSkipAttenMask(const ExtraInfo &info, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t actualSeqQ = info.actSeqLenQ;

    // s2<s1时，必然走mask
    if (info.s2Size < actualSeqQ) {
        return false;
    }

    // 当前的s2位置不超过需要打标记的位置时，不需要mask
    uint32_t s2EndPos = info.s2Idx * singleProcessSInnerSize + info.actualSingleProcessSInnerSize;
    if (s2EndPos <= (info.s2Size - actualSeqQ + 1)) {
        return true;
    }

    // BSH/BSND/TND格式切G，最后一个s1不需要mask
    if constexpr (LAYOUT_T != LAYOUT::BNSD) {
        uint32_t baseS1StartIdx = info.s1Idx * s1SizeSub;
        uint32_t s1StartIdx = (mSizeVStart + startRow) / info.gSize + baseS1StartIdx;
        uint32_t s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize + baseS1StartIdx;
        if (s1StartIdx == s1EndIdx && s1StartIdx == actualSeqQ - 1) {
            return true;
        }
    }
    return false;
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::ElewiseCompute(ExtraInfo& info, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    if constexpr (!ANTIQUANT) {
        Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
    }

    // attenMask
    if (attenMaskFlag == 1) {
        LocalTensor<bool> attenMaskUb;
        if (qSeqSize == 1 && LAYOUT_T != LAYOUT::TND) {
            AttenMaskCopyIn(info.attenMaskOffset, dealRowCount, actualColumnCount);
            attenMaskUb = inputBuf1.Get<bool>();
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
            for (int i = 1; i < dealRowCount; i++) {
                DataCopy(attenMaskUb[i * attenMaskSizeAlign], attenMaskUb, attenMaskSizeAlign);
            }
        } else {
            if (IsSkipAttenMask(info, startRow, dealRowCount)) {
                return;
            }
            if constexpr (LAYOUT_T == LAYOUT::BNSD) {
                AttenMaskCopyIn(info);
                attenMaskUb = inputBuf1.Get<bool>();
                LocalTensor<bool> attenMaskUbDst = attenMaskUb[BUFFER_SIZE_BYTE_32K / 2];

                uint32_t headS1Count= 0;
                uint32_t headS1Start = (mSizeVStart + startRow) % info.s1Size;
                if (headS1Start + dealRowCount > info.s1Size) {
                    headS1Count = info.s1Size - headS1Start;
                } else {
                    headS1Count = dealRowCount;
                }
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
                // head
                DataCopy(attenMaskUbDst, attenMaskUb[headS1Start * attenMaskSizeAlign], headS1Count * attenMaskSizeAlign);
                // mid
                uint32_t reminRowCount = dealRowCount - headS1Count;
                uint32_t midGCount = reminRowCount / info.s1Size;
                uint32_t tailS1Size = reminRowCount % info.s1Size;
                for (uint32_t i = 0; i < midGCount; i++) {
                    DataCopy(attenMaskUbDst[(headS1Count + i * info.s1Size) * attenMaskSizeAlign],
                        attenMaskUb, info.s1Size * attenMaskSizeAlign);
                }
                // tail
                if (tailS1Size > 0) {
                    DataCopy(attenMaskUbDst[(headS1Count + midGCount * info.s1Size) * attenMaskSizeAlign],
                        attenMaskUb, tailS1Size * attenMaskSizeAlign);
                }
                attenMaskUb = attenMaskUbDst;
            } else { // BSH/BSND/TND
                uint32_t s1StartIdx = (mSizeVStart + startRow) / info.gSize;
                uint32_t s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
                uint32_t s1Count = s1EndIdx - s1StartIdx + 1;

                uint32_t offset;
                uint32_t actualSeqQ = info.actSeqLenQ;

                int32_t delta = (info.s1Idx * s1SizeSub + s1StartIdx) - info.s2Idx * singleProcessSInnerSize +
                                (info.s2Size - actualSeqQ);
                if (delta < 0) {
                    offset = (-delta) < (int32_t)s1Count ? (-delta) : s1Count;
                } else {
                    offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                            ATTENMASK_STRIDE;
                }

                attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

                DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = s1Count;
                dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) / 32;
                dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
                dataCopyParams.dstStride = 0;

                attenMaskUb = inputBuf1.Get<bool>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
                DataCopy(attenMaskUb, attenMaskBoolGm[offset], dataCopyParams);

                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
                LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
                LocalTensor<int16_t> attenMaskUbDst = mask16[BUFFER_SIZE_BYTE_16K / 4];

                uint32_t headGCount = 0;
                uint32_t firstGIdx = (mSizeVStart + startRow) % info.gSize;
                if (s1Count > 1) {
                    headGCount = info.gSize - firstGIdx;
                } else {
                    headGCount = dealRowCount;
                }
                uint32_t dstMaskOffset = 0;
                uint32_t srcMaskBaseOffset = 0;
                // head
                SetMaskCount();
                SetVectorMask<int16_t, MaskMode::COUNTER>(attenMaskSizeAlign / 2);
                Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset],
                                     AscendC::MASK_PLACEHOLDER, headGCount,
                                     {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
                dstMaskOffset += headGCount * attenMaskSizeAlign / 2;
                srcMaskBaseOffset += attenMaskSizeAlign / 2;
                // mid
                uint32_t reminRowCount = dealRowCount - headGCount;
                uint32_t midS1Count = reminRowCount / info.gSize;
                uint32_t tailGSize =  reminRowCount % info.gSize;
                for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
                    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset],
                                         AscendC::MASK_PLACEHOLDER, info.gSize,
                                         {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
                    dstMaskOffset += info.gSize * attenMaskSizeAlign / 2;
                    srcMaskBaseOffset += attenMaskSizeAlign / 2;
                }
                // tail
                if (tailGSize > 0) {
                    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset],
                                         AscendC::MASK_PLACEHOLDER, tailGSize,
                                         {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
                    }
                    SetMaskNorm();
                    ResetMask();
                    attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
                }
            }

        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
        attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign); // Select接口要求mask size与参数匹配
        mmResUb.SetSize(dealRowCount * columnCount);            // Select接口要求src size与参数匹配
        SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
        mmResUb.SetSize(BUFFER_SIZE_BYTE_32K / sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);

        PipeBarrier<PIPE_V>();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::SoftmaxFlashV2Compute(ExtraInfo& info,
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
    uint32_t baseOffset = startRow;
#else
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
#endif
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t outIdx = info.loop % (PRE_LOAD_NUM_DD);
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (PRE_LOAD_NUM_DD);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_MAX_BUF_SIZE + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_SUM_BUF_SIZE + baseOffset];
    }

#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[outIdx * SOFTMAX_SUM_BUF_SIZE + baseOffset],
        softmaxMaxUb[outIdx * SOFTMAX_MAX_BUF_SIZE + baseOffset], mmResUb, softmaxExpUb[outIdx * SOFTMAX_EXP_BUF_SIZE + baseOffset],
        inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
#else
    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb, softmaxSumUb[outIdx * SOFTMAX_SUM_BUF_SIZE + baseOffset],
        softmaxMaxUb[outIdx * SOFTMAX_MAX_BUF_SIZE + baseOffset], mmResUb, softmaxExpUb[outIdx * SOFTMAX_EXP_BUF_SIZE + baseOffset],
        inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
#endif
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::Bmm2FDDataCopyOut(const ExtraInfo& info, LocalTensor<T> &attenOutUb, uint32_t startRow,
                                                                uint32_t dealRowCount, uint32_t columnCount,
                                                                uint32_t actualColumnCount)
{
    DealInvalidRows(info, attenOutUb, startRow, dealRowCount, columnCount, actualColumnCount);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    LocalTensor<T> tmp = outputBuf1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, attenOutUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    size_t base = (bIdx * qHeadNum + n2Idx * gSize) * splitKVNum * headDim;
    DataCopyPad(accumOutGm[base + s2IdxFD * gSize * actualColumnCount + 
                           startRow * actualColumnCount + mSizeVStart * actualColumnCount], tmp, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb, dataCopyParams);
}


template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::Bmm2DataCopyOutNcon(const ExtraInfo& info, LocalTensor<OUT_T> &attenOutUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (LAYOUT_T ==LAYOUT::BSH || LAYOUT_T ==LAYOUT::BSND || LAYOUT_T ==LAYOUT::TND) {
        uint64_t outOffset = 0;
        if (kvHeadNum != 1 && gOuter == 1) {
            uint32_t startS1Idx = (mSizeVStart + startRow) / info.gSize;
            uint32_t startGOfffset = (mSizeVStart + startRow) % info.gSize;
            uint32_t endS1Idx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
            uint32_t curStartRow = (mSizeVStart + startRow);
            uint32_t curDealRowCount = 0;
            uint32_t ubOffset = 0;
            for (uint32_t curS1idx = startS1Idx; curS1idx <= endS1Idx; curS1idx++) {
                outOffset = GetQueryOffSetBySIdx(info.bIdx, curS1idx + info.s1Idx * s1SizeSub, info.n2Idx, startGOfffset);
                if (curS1idx != endS1Idx) {
                    curDealRowCount = (curS1idx + 1) * gSize - curStartRow;
                }
                else {
                    curDealRowCount = mSizeVStart + startRow + dealRowCount - curStartRow;
                }
                ubOffset = (curStartRow - mSizeVStart) * columnCount;
                LocalTensor<OUT_T> curAttenOutUb = attenOutUb[ubOffset];
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = curDealRowCount;
                dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
                dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
                dataCopyParams.dstStride = 0;
                DataCopyPad(attentionOutGm[outOffset], curAttenOutUb, dataCopyParams);
                PipeBarrier<PIPE_V>();
                curStartRow += curDealRowCount;
                startGOfffset = 0;
            }
            return;
        }
        Bmm2DataCopyOut(info.attenOutOffset, attenOutUb, startRow, dealRowCount,columnCount, actualColumnCount);
    }
    else {
        Bmm2DataCopyOut(info.attenOutOffset, attenOutUb, startRow, dealRowCount,columnCount, actualColumnCount);
    }
}

template <typename IFAT>
template <typename RT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::DealInvalidRows(const ExtraInfo &info, LocalTensor<RT> &attenOutUb,
                                                          uint32_t startRow, uint32_t dealRowCount,
                                                          uint32_t columnCount, uint32_t actualColumnCount)
{
    if (!attenMaskFlag || (info.actSeqLenQ <= info.s2Size)) {
        return;
    }
    PipeBarrier<PIPE_V>();
    uint32_t s1Tok = info.actSeqLenQ - info.s2Size;

    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        uint32_t s1 = (mSizeVStart + startRow) % info.s1Size;
        for (uint32_t i = 0; i < dealRowCount;) {
            if (s1 < s1Tok) {
                uint32_t s1Num = s1Tok - s1;
                if (i + s1Num > dealRowCount) {
                    s1Num = dealRowCount - i;
                }
                Duplicate(attenOutUb[i * columnCount], static_cast<RT>(FLOAT_ZERO), columnCount * s1Num);
            }
            i += info.s1Size - s1;
            s1 = 0;
        }
        return;
    }
    // BSH and TND
    uint32_t s1 = info.s1Idx * s1SizeSub + (mSizeVStart + startRow) / info.gSize;
    uint32_t gIdx = (mSizeVStart + startRow) % info.gSize;
    for (uint32_t i = 0; i < dealRowCount;) {
        if (s1 < s1Tok) {
            uint32_t gNum = info.gSize - gIdx;
            if (i + gNum > dealRowCount) {
                gNum = dealRowCount - i;
            }
            Duplicate(attenOutUb[i * columnCount], static_cast<RT>(FLOAT_ZERO), columnCount * gNum);
            i += gNum;
            s1++;
            gIdx = 0;
            continue;
        }
        break;
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::Bmm2ResCopyOut(const ExtraInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                                 uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        Bmm2FDDataCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::Bmm2CastAndCopyOut(const ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                                 uint32_t actualColumnCount)
{
    if constexpr (!POST_QUANT) {
        LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuf1.Get<OUT_T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
        } else {
            Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
        }
        DealInvalidRows(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        Bmm2DataCopyOutNcon(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    } else {
        LocalTensor<OUT_T> bmm2ResUbInt8 = outputBuf1.Get<OUT_T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        PostQuant(info.perChannelQuantOffset, bmm2ResUb, bmm2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
        DealInvalidRows(info, bmm2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        Bmm2DataCopyOut(info.attenOutOffset, bmm2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::DealBmm1ResBaseBlock(ExtraInfo& info, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<MM_OUT_T> mmResUb = tmpBuff1.Get<MM_OUT_T>();

    size_t batchBase = 0;

    uint64_t inOutGmOffset = (info.loop % PRE_LOAD_NUM_DD) * mmResUbSize + (mSizeVStart + startRow) * columnCount;
    LocalTensor<MM_OUT_T> tmpMmResUb = inputBuf1.Get<MM_OUT_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(tmpMmResUb, mm1ResGm[inOutGmOffset + batchBase], computeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(mmResUb, tmpMmResUb, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    PipeBarrier<PIPE_V>();

    ElewiseCompute(info, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<KV_T> tmpMMResCastTensor = outputBuf1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset + batchBase], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}


template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::AntiquantMatmulResCombine(ExtraInfo &info,
    LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
}

// 由于S在GM上已完成atomatic，此处只需要把数据拷贝出来有fp16转成fp32
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::AntiquantMatmulResCombineDD(ExtraInfo &info,
    LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount, float scaleC)
{
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;

    LocalTensor<MM_OUT_T> tmpMMRes = inputBuf1.Get<MM_OUT_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(tmpMMRes, srcGm[baseOffset], copySize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    Cast(bmmResUb, tmpMMRes, AscendC::RoundMode::CAST_NONE, copySize);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    Muls(bmmResUb, bmmResUb, scaleC, copySize);
    PipeBarrier<PIPE_V>();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::DealAntiqBmm1ResBaseBlock(ExtraInfo& info, uint32_t startRow,
                                                                        uint32_t dealRowCount, uint32_t columnCount,
                                                                        uint32_t actualColumnCount)
{
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_DD) * mmResUbSize + mSizeVStart * columnCount;
    AntiquantMatmulResCombineDD(info, mmResUb, mm1ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount, scaleC1 * static_cast<T>(tilingData->baseParams.scaleValue));
    PipeBarrier<PIPE_V>();
    LocalTensor<T> aMax = aMaxBmm1Ub[(info.bn2IdxInCurCore % (PRE_LOAD_NUM_DD)) * Q_AMAX_BUF_SIZE + startRow * BLOCK_ELEMENT_NUM];
    RowMuls(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // mul scalar and mask
    ElewiseCompute(info, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);
    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    DealKvInt4ColumnOdd(mmResUb, dealRowCount, columnCount, actualColumnCount);

    uint64_t dstOffset = (info.loop % PRE_LOAD_NUM_DD) * mmResUbSize + mSizeVStart * columnCount;
    AntiquantSoftmaxResPreProcess(info, vec1ResGm[dstOffset], mmResUb, tmpAFloorUb, startRow, dealRowCount, columnCount,
                                  actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::DealAntiqBmm1ResBaseBlockPerToken(ExtraInfo& info,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_DD) * mmResUbSize + mSizeVStart * columnCount;

    AntiquantMatmulResCombineDD(info, mmResUb, mm1ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount, scaleC1 * static_cast<T>(tilingData->baseParams.scaleValue));
    PipeBarrier<PIPE_V>();
    LocalTensor<T> aMax = aMaxBmm1Ub[(info.bn2IdxInCurCore % (PRE_LOAD_NUM_DD)) * Q_AMAX_BUF_SIZE + startRow * BLOCK_ELEMENT_NUM];
    RowMuls(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    CopyAntiquantParamsPerToken(keyAntiqScaleGm, info.antiqKeyParamOffsetPerToken, columnCount, actualColumnCount);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    LocalTensor<T> antiqScalePerTokenUb = inputBuf2.Get<T>();
    // (Amax * C + rowsum(A) * offset) * scale
    VecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    PipeBarrier<PIPE_V>();
    // mul scalar and mask
    ElewiseCompute(info, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);
    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    DealKvInt4ColumnOdd(mmResUb, dealRowCount, columnCount, actualColumnCount);
    CopyAntiquantParamsPerToken(valueAntiqScaleGm, info.antiqValueParamOffsetPerToken, columnCount, actualColumnCount);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    antiqScalePerTokenUb = inputBuf2.Get<T>();
    VecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    PipeBarrier<PIPE_V>();
    Adds(tmpAFloorUb, mmResUb, (T)0, dealRowCount * columnCount); // mmResUb need to be stored
    PipeBarrier<PIPE_V>();
    uint64_t dstOffset = (info.loop % PRE_LOAD_NUM_DD) * mmResUbSize + mSizeVStart * columnCount;
    AntiquantMatmulPreProcess(info, vec1ResGm[dstOffset], aMaxBmm2Ub[(info.loop % PRE_LOAD_NUM_DD)*P_AMAX_BUF_SIZE], mmResUb, tmpAFloorUb,
        startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::ProcessVec1Inner(ExtraInfo& info)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512，mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;
#endif
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }
    uint32_t loopCount = (mSizeVector + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSizeVector - (loopCount - 1) * mSplitSize;

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT) {
            if constexpr (ANTIQUANT_PER_CHANNEL) {
                DealAntiqBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                          info.actualSingleProcessSInnerSize);
            } else if constexpr (ANTIQUANT_PER_TOKEN) {
                DealAntiqBmm1ResBaseBlockPerToken(info, i * mSplitSize, dealSize,
                                                  info.actualSingleProcessSInnerSizeAlign, info.actualSingleProcessSInnerSize);
            }
        } else {
            DealBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                 info.actualSingleProcessSInnerSize);
        }
    }

    if (info.s2Idx == sInnerLoopTimes - 1) {
        if constexpr (FLASH_DECODE) {
            uint32_t outIdx = info.loop % (PRE_LOAD_NUM_DD);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_SUM_BUF_SIZE];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_MAX_BUF_SIZE];
            ComputeLogSumExpAndCopyToGm(sumTensor, maxTensor);
            return;
        }
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::DealAntiqBmm2ResBaseBlock(ExtraInfo& info, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
    uint32_t baseOffset = startRow;
#else
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
#endif
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_DD) * bmm2ResUbSize + mSizeVStart * columnCount;
    LocalTensor<half> bmm2ResUb = tmpBuff1.Get<half>();
    AntiquantMM2ResCombine(info, bmm2ResUb, mm2ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount);
    uint32_t vec2ComputeSize = dealRowCount * columnCount;

    size_t batchBase = 0;

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        LocalTensor<half> bmm2ResPreUb = inputBuf2.Get<half>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        PipeBarrier<PIPE_V>();
        DataCopy(bmm2ResPreUb, vec2ResUb, dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % PRE_LOAD_NUM_DD;
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
        //step2: 将softmaxExpUb转换为FP16
        LocalTensor<half> tmpSoftmaxFp16 = tmpBuff3.Get<half>();
        Cast(tmpSoftmaxFp16, softmaxExpUb[idx * SOFTMAX_EXP_BUF_SIZE + baseOffset], AscendC::RoundMode::CAST_ROUND, dealRowCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> tmpExpBrcbResUb = tmpBuff2.Get<half>();
        Brcb(tmpExpBrcbResUb, tmpSoftmaxFp16, (dealRowCount + 7) / 8, {1, 8});
        // step3: Oi = (Oi - 1)*tmpSoftmaxFp16 + Otmp
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        //step2: 将softmaxExpUb转换为FP16
        LocalTensor<T> tmpSoftmaxFp32 = tmpBuff3.Get<T>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = 1;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 1;
        DataCopy(tmpSoftmaxFp32, softmaxExpUb[idx * SOFTMAX_EXP_BUF_SIZE + baseOffset], dataCopyParams);
        DataCopy(tmpSoftmaxFp32[BLOCK_ELEMENT_NUM], softmaxExpUb[idx * SOFTMAX_EXP_BUF_SIZE + baseOffset], dataCopyParams);
        LocalTensor<half> tmpSoftmaxFp16 = tmpBuff3.Get<half>();
        PipeBarrier<PIPE_V>();
        Cast(tmpSoftmaxFp16, tmpSoftmaxFp32, AscendC::RoundMode::CAST_ROUND, dealRowCount * BLOCK_ELEMENT_NUM * 2);
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, tmpSoftmaxFp16,
            dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> bmm2ResUbFp32 = tmpBuff2.Get<T>();
        Cast(bmm2ResUbFp32, bmm2ResUb, RoundMode::CAST_NONE, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % PRE_LOAD_NUM_DD;
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
        LocalTensor<T> tmpSumBrcbResUb = tmpBuff1.Get<T>();
        Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * SOFTMAX_SUM_BUF_SIZE + baseOffset], (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUbFp32, bmm2ResUbFp32, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowDivs(bmm2ResUbFp32, bmm2ResUbFp32, softmaxSumUb[idx * SOFTMAX_SUM_BUF_SIZE + baseOffset],
            dealRowCount, columnCount, actualColumnCount);
#endif
        
        PipeBarrier<PIPE_V>();

        if (antiqOffsetExistFlag) {
            // bmm2res+offsetV
            CopyAntiquantScale(antiqOffsetUb, valueAntiqOffsetGm, info.antiqValueParamOffset);
            PipeBarrier<PIPE_V>();
            VecAddMat(bmm2ResUbFp32, antiqOffsetUb, bmm2ResUbFp32, dealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
        }

        Muls(bmm2ResUbFp32, bmm2ResUbFp32, scaleC2, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        CopyAntiquantScale(antiqScaleUb, valueAntiqScaleGm, info.antiqValueParamOffset);
        PipeBarrier<PIPE_V>();
        // ScaleV * bmm2res
        VecMulMat(bmm2ResUbFp32, antiqScaleUb, bmm2ResUbFp32, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUbFp32, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        DataCopy(vec2ResUb, bmm2ResUb, dealRowCount * columnCount);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadDD<IFAT>::PostQuant(uint64_t perChannelQuantOffset, LocalTensor<T> &bmm2ResUb, LocalTensor<int8_t> &bmm2ResUbInt8,
                                                        uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount)
{
    // post quant function
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::AntiquantMM2ResCombine(const ExtraInfo& info,
    LocalTensor<MM_OUT_T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;
    LocalTensor<MM_OUT_T> tmpCInt = inputBuf1.Get<MM_OUT_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(tmpCInt, srcGm[baseOffset], copySize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(bmmResUb, tmpCInt, copySize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::DealAntiqBmm2ResBaseBlockPerToken(ExtraInfo& info,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
    uint32_t baseOffset = startRow;
#else
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
#endif
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_DD) * bmm2ResUbSize + mSizeVStart * columnCount;

    LocalTensor<half> bmm2ResUb = tmpBuff1.Get<half>();
    AntiquantMM2ResCombine(info, bmm2ResUb, mm2ResGm[srcGmOffset], startRow, dealRowCount, columnCount,
                           actualColumnCount);
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    size_t batchBase = 0;
    PipeBarrier<PIPE_V>();
    LocalTensor<T> aRowMax = aMaxBmm2Ub[(info.loop % PRE_LOAD_NUM_DD) * P_AMAX_BUF_SIZE][baseOffset]; //shape [dealRowCount,32B/sizeof(float)]
    LocalTensor<T> tmpaRowMaxFp32 = tmpBuff3.Get<T>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 1;
    DataCopy(tmpaRowMaxFp32, aRowMax, dataCopyParams);
    DataCopy(tmpaRowMaxFp32[BLOCK_ELEMENT_NUM], aRowMax, dataCopyParams);
    LocalTensor<half> tmpaRowMaxFp16 = tmpBuff3.Get<half>();
    PipeBarrier<PIPE_V>();
    Cast(tmpaRowMaxFp16, tmpaRowMaxFp32, AscendC::RoundMode::CAST_ROUND, dealRowCount * BLOCK_ELEMENT_NUM * 2);
    PipeBarrier<PIPE_V>();
    RowMuls(bmm2ResUb, bmm2ResUb, tmpaRowMaxFp16, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        LocalTensor<half> bmm2ResPreUBPerToken = inputBuf2.Get<half>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        DataCopy(bmm2ResPreUBPerToken, vec2ResUb, dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
        uint32_t loopIdx = info.loop % PRE_LOAD_NUM_DD;
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
        //step2: 将softmaxExpUb转换成FP16
        LocalTensor<half> tmpSoftmaxFp16 = tmpBuff3.Get<half>();
        Cast(tmpSoftmaxFp16, softmaxExpUb[loopIdx * SOFTMAX_EXP_BUF_SIZE + baseOffset], AscendC::RoundMode::CAST_ROUND, dealRowCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> tmpExpBrcbResUb = tmpBuff2.Get<half>();
        Brcb(tmpExpBrcbResUb, tmpSoftmaxFp16, (dealRowCount + 7) / 8, {1, 8});
        //step3: Oi = (Oi - 1)*tmpSoftmaxFp16 + Otmp
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUBPerToken, bmm2ResPreUBPerToken, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        //step2: 将softmaxExpUb转换成FP16
        LocalTensor<T> tmpSoftmaxFp32 = tmpBuff3.Get<T>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = 1;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 1;
        DataCopy(tmpSoftmaxFp32, softmaxExpUb[loopIdx * SOFTMAX_EXP_BUF_SIZE + baseOffset], dataCopyParams);
        DataCopy(tmpSoftmaxFp32[BLOCK_ELEMENT_NUM], softmaxExpUb[loopIdx * SOFTMAX_EXP_BUF_SIZE + baseOffset], dataCopyParams);
        LocalTensor<half> tmpSoftmaxFp16 = tmpBuff3.Get<half>();
        PipeBarrier<PIPE_V>();
        Cast(tmpSoftmaxFp16, tmpSoftmaxFp32, AscendC::RoundMode::CAST_ROUND, dealRowCount * BLOCK_ELEMENT_NUM * 2);
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUBPerToken, bmm2ResPreUBPerToken, tmpSoftmaxFp16, dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUBPerToken, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> bmm2ResUbFp32PerToken = tmpBuff2.Get<T>();
        Cast(bmm2ResUbFp32PerToken, bmm2ResUb, RoundMode::CAST_NONE, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        uint32_t loopIdx = info.loop % PRE_LOAD_NUM_DD;
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
        LocalTensor<T> tmpSumBrcbResUb = tmpBuff1.Get<T>();
        Brcb(tmpSumBrcbResUb, softmaxSumUb[loopIdx * SOFTMAX_SUM_BUF_SIZE + baseOffset], (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUbFp32PerToken, bmm2ResUbFp32PerToken, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowDivs(bmm2ResUbFp32PerToken, bmm2ResUbFp32PerToken, softmaxSumUb[loopIdx * SOFTMAX_SUM_BUF_SIZE + baseOffset],
            dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        Muls(bmm2ResUbFp32PerToken, bmm2ResUbFp32PerToken, scaleC2, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUbFp32PerToken, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        DataCopy(vec2ResUb, bmm2ResUb, dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::ProcessVec2Inner(ExtraInfo& info)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }
    uint32_t loopCount = (mSizeVector + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSizeVector - (loopCount - 1) * mSplitSize;

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT) {
            if constexpr (ANTIQUANT_PER_CHANNEL) {
                DealAntiqBmm2ResBaseBlock(info, i * mSplitSize, dealSize, headDimAlign, headDim);
            } else if constexpr (ANTIQUANT_PER_TOKEN) {
                DealAntiqBmm2ResBaseBlockPerToken(info, i * mSplitSize, dealSize, headDimAlign, headDim);
            }
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::QueryPreProcessL(ExtraInfo& info)
{
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    QueryPreProcessInner(info);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CalcParams(uint32_t loop, ExtraInfo& info, TaskContext &task) {
    info.loop = loop;
    info.bIdx = task.bidx;
    info.s1Idx = task.s1idx;
    info.s2Idx = task.s2idx;
    info.n2Idx = task.nidx;
    info.curSInnerLoopTimes = task.s2loops;
    info.s1Size = task.s1Size;
    info.s2Size = task.s2Size;
    info.gIdx = task.gidx;
    info.gSize = (task.gidx == gOuter - 1) ? gSizeTail : gSizeSub;
    info.actS1Size = task.actS1Size;
    info.mSize = info.gSize * info.s1Size;
    info.actSeqLenQ = task.actSeqLenQ;

    if ASCEND_IS_AIV {
        info.mSizeV = (info.mSize + 1) / 2;
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    info.isFirstSInnerLoop = (info.s2Idx == 0);
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            // B,N2,G,1,D
            tensorACoreOffset = info.bIdx * qHeadNum * qSeqSize * headDim + info.n2Idx * gSize * qSeqSize * headDim +
                                info.gIdx * gSizeSub * qSeqSize * headDim;
            // B,N2,S2,D
            tensorBCoreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDim + info.n2Idx * kvSeqSize * headDim;
            if (!batchContinuous) {
                uint64_t seqSize = SeqLenFromTensorList(bIdx);
                tensorBCoreOffset = info.n2Idx * seqSize * headDim;
            }
            tensorBCoreOffset += kvPaddingBeginOffset * headDim;
            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += s2IdxFD * sInnerLoopSize * headDim;
            }
        } else {
            // B,1,N2,G,D
            tensorACoreOffset = GetQueryOffSetBySIdx(info.bIdx, info.s1Idx * s1SizeSub, info.n2Idx, 0);
            // B,S2,N2,D
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            if (!batchContinuous) {
                tensorBCoreOffset = info.n2Idx * headDim;
            }
            tensorBCoreOffset += kvPaddingBeginOffset * kvHeadNum * headDim;
            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDim;
            }
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * headDim;
    } else {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * kvHeadNum * headDim;
    }
    info.attenOutOffset = tensorACoreOffset;
    info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        info.actualSingleProcessSInnerSize = task.s2SizeTail;
    }
    if constexpr (KVINT4) {
        info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)64L);
    } else {
        info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)BYTE_BLOCK);
    }

    if (antiquantPerTensorFlag == 1) {
        info.antiqKeyParamOffset = 0;
    } else if (antiquantPerHeadFlag) {
        info.antiqKeyParamOffset = info.n2Idx;
    } else {
        info.antiqKeyParamOffset = info.n2Idx * headDim;
    }

    if (antiquantPerTensorFlag == 1) {
        info.antiqValueParamOffset = 0;
    } else if (antiquantPerHeadFlag) {
        info.antiqValueParamOffset = info.n2Idx;
    } else {
        info.antiqValueParamOffset = info.n2Idx * headDim;
    }
    uint64_t sInnerOffsetDataSize = info.s2Idx * singleProcessSInnerSize;
    if constexpr (FLASH_DECODE) {
        sInnerOffsetDataSize += s2IdxFD * sInnerLoopSize;
    }

    info.antiqKeyParamOffsetPerToken = info.bIdx * antiqSeqSize + sInnerOffsetDataSize + kvPaddingBeginOffset;// info.n2Idx * headDim;
    info.antiqValueParamOffsetPerToken = info.bIdx * antiqSeqSize + sInnerOffsetDataSize + kvPaddingBeginOffset;
    if (antiquantPerHeadFlag) {
        info.antiqKeyParamOffsetPerToken = info.bIdx * antiqSeqSize * kvHeadNum + info.n2Idx * antiqSeqSize + sInnerOffsetDataSize + kvPaddingBeginOffset;// info.n2Idx * headDim;
        info.antiqValueParamOffsetPerToken = info.bIdx * antiqSeqSize * kvHeadNum + info.n2Idx * antiqSeqSize + sInnerOffsetDataSize + kvPaddingBeginOffset;
    }
    // out quant
    info.perChannelQuantOffset = info.n2Idx * headDim * gSize;
    attenMaskCoreOffset = info.bIdx * attenMaskSize + kvPaddingBeginOffset;
    info.attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;
    info.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::ProcessVec2L(ExtraInfo& info) {
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;
    if (mSizeVector == 0) {
        return;
    }
    ProcessVec2Inner(info);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::ProcessVec1L(ExtraInfo& info)
{
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    ProcessVec1Inner(info);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info) {
    uint32_t mmRowCount = msdIterNum * info.mSize;
    uint32_t copyStride = 16 * headDimAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++) {
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdIterNum * info.mSize - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数  16
        }
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForA.dValue = headDimAlign / 2;
            mm1Nd2NzParamsForA.srcDValue = headDimAlign / 2; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
        } else {
            mm1Nd2NzParamsForA.dValue = headDimAlign; // 单个ND矩阵的列数, 单位为元素个数   128
            mm1Nd2NzParamsForA.srcDValue = headDimAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  128
        }
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数  16
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移   
        if constexpr (ANTIQUANT) {
            DataCopy(aL1Tensor[i * copyStride], queryPreProcessResGm[(info.bn2IdxInCurCore % (PRE_LOAD_NUM_DD)) * bmm2ResUbSize + i * copyStride], mm1Nd2NzParamsForA);
        } else {
            DataCopy(aL1Tensor, queryGm[info.tensorAOffset], mm1Nd2NzParamsForA);
        }
    }
    auto b = queryPreProcessResGm[(info.bn2IdxInCurCore % (PRE_LOAD_NUM_DD)) * bmm2ResUbSize];
}

// nCopyRowCount需要32元素对齐
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyInMm1BToL1(
    LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t nCopyIdx, uint32_t nCopyRowCount,
    uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign)
{
    uint64_t step = headDim;
    if constexpr (KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) {
        step = headDim * kvHeadNum;
    }
    uint64_t keyGmBaseOffset = info.tensorBOffset + nCopyIdx * nCopyRowCount * step;

#ifdef L1_LAYOUT_zN
    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    if constexpr (KVINT4) {
        mm1Nd2NzParamsForB.dValue = headDim / 2;
        mm1Nd2NzParamsForB.srcDValue = step / 2;
    } else {
        mm1Nd2NzParamsForB.dValue = headDim;
        mm1Nd2NzParamsForB.srcDValue = step;
    }
    mm1Nd2NzParamsForB.dstNzC0Stride = nActCopyRowCountAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor, keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
#else
    uint32_t baseN = 256 / sizeof(KV_T);
    uint32_t ndNum_tmp = nActCopyRowCount / baseN;
    uint32_t nTail = nActCopyRowCount % baseN;
    uint32_t nTailAlign = nActCopyRowCountAlign - ndNum_tmp * baseN;

    Nd2NzParams mm1Nd2NzParamsForB;
    if (ndNum_tmp != 0){
        mm1Nd2NzParamsForB.nValue = baseN; // 单个ND矩阵的行数, 单位为元素个数   256
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = headDim / 2; // 单个ND矩阵的列数, 单位为元素个数   128
            mm1Nd2NzParamsForB.srcDValue = step / 2; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        } else {
            mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
            mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = baseN;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数   256
        mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数

        if ((KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) && (baseN * step > 65535)) {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
            for (uint32_t i = 0; i < ndNum_tmp; i++) {
                DataCopy(bL1Tensor[i * baseN * headDimAlign], keyGm[keyGmBaseOffset + i * baseN * step], mm1Nd2NzParamsForB);
            }
        } else {
            mm1Nd2NzParamsForB.ndNum = ndNum_tmp;
            if constexpr (KVINT4) {
                mm1Nd2NzParamsForB.srcNdMatrixStride = baseN * step / 2; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
                mm1Nd2NzParamsForB.dstNzMatrixStride = baseN * headDimAlign / 2; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
            } else {
                mm1Nd2NzParamsForB.srcNdMatrixStride = baseN * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
                mm1Nd2NzParamsForB.dstNzMatrixStride = baseN * headDimAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
            }
            DataCopy(bL1Tensor, keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
        }
    }

    if (nTail != 0){
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数   actualSingleProcessSInnerSizeAlign为32对齐，nTail已经是32对齐
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = headDim / 2; // 单个ND矩阵的列数, 单位为元素个数   128
            mm1Nd2NzParamsForB.srcDValue = step / 2; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        } else {
            mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
            mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        }
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 0
        mm1Nd2NzParamsForB.dstNzC0Stride = nTailAlign;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
        mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数  0

        DataCopy(bL1Tensor[ndNum_tmp * baseN * headDimAlign], keyGm[keyGmBaseOffset + ndNum_tmp * baseN * step],
                 mm1Nd2NzParamsForB);  //需要调整偏移地址，bL1Tensor的偏移， keyGm的偏移
    }
#endif
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyInMm1BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    uint32_t baseN = 256 / sizeof(KV_T);
    uint32_t blockElementCnt = 32 / sizeof(KV_T);
    if constexpr (KVINT4) {
        blockElementCnt = 64;
    }

    uint64_t step = headDim;
    if constexpr (KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) {
        step = headDim * kvHeadNum;
    } else if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        step = blockElementCnt;
    }

    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInBaseN = copyStartRowCnt % baseN;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInBaseN + nActCopyRowCount <= baseN) {
        nHead = 0;
        midNdNum = 0;
        nTail = nActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInBaseN;
    } else {
        if (copyStartRowCntInBaseN == 0) {
            nHead = 0;
        } else {
            nHead = baseN - copyStartRowCntInBaseN;
        }
        midNdNum = (nActCopyRowCount - nHead) / baseN;
        nTail = (nActCopyRowCount - nHead) % baseN;
        copyStartRowCntInBaseNTail = 0;
    }

    uint32_t dstNzC0StrideTail = baseN;
    if (copyTotalRowCnt % baseN != 0) {
        if ((copyStartRowCnt + nActCopyRowCount) > (copyTotalRowCnt / baseN * baseN)) {
            dstNzC0StrideTail = copyTotalRowCnt % baseN;
        }
    }

    Nd2NzParams mm1Nd2NzParamsForB;
    if (nHead != 0) {
        if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
            DataCopyParams intriParams;
            intriParams.blockLen = nHead;
            intriParams.blockCount = headDim / blockElementCnt;
            intriParams.dstStride = baseN - nHead;
            intriParams.srcStride = kvCacheBlockSize - nHead;

            uint32_t ndNumFinish = copyStartRowCnt / baseN;
            DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))], keyGm[keyGmBaseOffset], intriParams);
            copyTotalRowCnt -= nHead;
        } else {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = nHead; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzC0Stride = baseN;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
            mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

            uint32_t ndNumFinish = copyStartRowCnt / baseN;
            DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))], keyGm[keyGmBaseOffset],
                    mm1Nd2NzParamsForB);
        }
    }

    if (midNdNum != 0) {
        if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
            DataCopyParams intriParams;
            intriParams.blockLen = baseN;
            intriParams.blockCount = headDim / blockElementCnt;
            intriParams.dstStride = 0;
            intriParams.srcStride = kvCacheBlockSize - baseN;

            int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN;
            for (uint32_t i = 0; i < midNdNum; i++) {
                DataCopy(bL1Tensor[(ndNumFinish + i) * baseN * headDimAlign], keyGm[keyGmBaseOffset + nHead * step + i * baseN * step], intriParams);
            }
            copyTotalRowCnt -= midNdNum * baseN;
        } else {
            mm1Nd2NzParamsForB.nValue = baseN; // 单个ND矩阵的行数, 单位为元素个数   256
            mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
            mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
            mm1Nd2NzParamsForB.dstNzC0Stride = baseN;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数   256
            mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数

            int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN;
            if ((KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) && (baseN * step > 65535)) {
                mm1Nd2NzParamsForB.ndNum = 1;
                mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
                mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128
                for (uint32_t i = 0; i < midNdNum; i++) {
                    DataCopy(bL1Tensor[(ndNumFinish + i) * baseN * headDimAlign], keyGm[keyGmBaseOffset + nHead * step + i * baseN * step], mm1Nd2NzParamsForB);
                }
            } else {
                mm1Nd2NzParamsForB.ndNum = midNdNum;
                mm1Nd2NzParamsForB.srcNdMatrixStride = baseN * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 256*128
                mm1Nd2NzParamsForB.dstNzMatrixStride = baseN * headDimAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数 256*128

                DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign], keyGm[keyGmBaseOffset + nHead * step], mm1Nd2NzParamsForB);
            }
        }
    }

    if (nTail != 0) {
        if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
            DataCopyParams intriParams;
            intriParams.blockLen = nTail;
            intriParams.blockCount = headDim / blockElementCnt;
            intriParams.dstStride = dstNzC0StrideTail - nTail;
            intriParams.srcStride = kvCacheBlockSize - nTail;

            uint32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN + midNdNum;
            DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))],
                keyGm[keyGmBaseOffset + (nHead + midNdNum * baseN) * step], intriParams);
        } else {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数   actualSingleProcessSInnerSizeAlign为32对齐，nTail已经是32对齐
            mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数 0
            mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
            mm1Nd2NzParamsForB.dstNzC0Stride = dstNzC0StrideTail;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
            mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数  0

            int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN + midNdNum;
            DataCopy(bL1Tensor[ndNumFinish * baseN * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], keyGm[keyGmBaseOffset + (nHead + midNdNum * baseN) * step],
                    mm1Nd2NzParamsForB);  //需要调整偏移地址，bL1Tensor的偏移， keyGm的偏移
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::LoadDataMm1A(ExtraInfo& info, LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor) {
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    if constexpr (KVINT4) {
        loadData2DParams.repeatTimes = (msdIterNum * info.mSize + 15) / 16 * headDimAlign / 64;
    } else {
        loadData2DParams.repeatTimes = (msdIterNum * info.mSize + 15) / 16 * headDimAlign / (32 / sizeof(KV_T));
    }
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::LoadDataMm1B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor) {
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::ComputeMm1(ExtraInfo& info) {
    LocalTensor<KV_T> aL1Tensor = qL1Buffers[qL1BufIter * L1Q_BLOCK_SIZE / sizeof(KV_T)];
    if (info.s2Idx == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0 + qL1BufIter);
        CopyInMm1AToL1(aL1Tensor, info);
        SetFlag<HardEvent::MTE2_MTE1>(L1Q_EVENT0 + qL1BufIter);
        WaitFlag<HardEvent::MTE2_MTE1>(L1Q_EVENT0 + qL1BufIter);
    }
    constexpr uint32_t nCopyRowCount = KV_LOAD_TO_L1_ROW_NUM;
    uint32_t nCopyTimes = (info.actualSingleProcessSInnerSize + nCopyRowCount - 1) / nCopyRowCount;
    uint32_t nTailCopyRowCount = info.actualSingleProcessSInnerSize - (nCopyTimes - 1) * nCopyRowCount;
    uint32_t nTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (nCopyTimes - 1) * nCopyRowCount;
    for (uint32_t nCopyIdx = 0, nActCopyRowCount = nCopyRowCount, nActCopyRowCountAlign = nCopyRowCount; nCopyIdx < nCopyTimes; nCopyIdx++) {
        if (nCopyIdx + 1 == nCopyTimes) {
            nActCopyRowCount = nTailCopyRowCount;
            nActCopyRowCountAlign = nTailCopyRowCountAlign;
        }
        // A:256*128 B:1024*128 A*B^T INT8  M=256 N=1024 K=128
        // A全载到L1，B的N方向按照512切分，计算2次
        kpL1BufIter++;
        LocalTensor<KV_T> bL1Tensor = kpL1Buffers[(kpL1BufIter % 3) * L1KP_BLOCK_SIZE / sizeof(KV_T)];
        WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));

        if constexpr (PAGE_ATTENTION) {
            uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
            uint32_t curSeqIdx = info.s2BatchOffset + nCopyIdx * nCopyRowCount;
            uint32_t copyFinishRowCnt = 0;
            while (copyFinishRowCnt < nActCopyRowCount) {
                uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                uint64_t idInBlockTable =
                    blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
                // 计算可以拷贝行数
                uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                if (copyFinishRowCnt + copyRowCnt > nActCopyRowCount) {
                    copyRowCnt = nActCopyRowCount - copyFinishRowCnt;
                }
                uint64_t keyOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
                    uint32_t blockElementCnt = 32 / sizeof(KV_T);
                    if constexpr (KVINT4) {
                        blockElementCnt = 64;
                    }
                    keyOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                } else {
                    if constexpr (KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) {
                        keyOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                    } else {
                        keyOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                    }
                }
                CopyInMm1BToL1ForPA(bL1Tensor, keyOffset, nActCopyRowCountAlign, copyFinishRowCnt, copyRowCnt);

                // 更新循环变量
                copyFinishRowCnt += copyRowCnt;
                curSeqIdx += copyRowCnt;
            }
        } else {
            CopyInMm1BToL1(bL1Tensor, info, nCopyIdx, nCopyRowCount, nActCopyRowCount, nActCopyRowCountAlign);
        }

        SetFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));
        WaitFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));

        // A:256*128，M方向按照64循环
        constexpr uint32_t baseM = 64 / sizeof(KV_T);
        uint32_t mActCopyRowCount = msdIterNum * info.mSize;
        uint32_t mLoopTimes = (mActCopyRowCount + baseM - 1) / baseM;
        uint32_t mTail = mActCopyRowCount - (mLoopTimes - 1) * baseM;
        bool isHead = false;
        bool isMid = false;
        bool isTail = false;
        bool isOdd = (mLoopTimes % 2) != 0 ? true : false;
        bool hasTail = (mTail == baseM) ? false : true;
        for (uint32_t i = 0, actualBaseM = baseM; i < mLoopTimes; i++) {
            if (i + 1 == mLoopTimes) {
                actualBaseM = mTail;
            }
            isHead = (!isOdd && ((!hasTail && (i < (mLoopTimes / 2))) || (i < (mLoopTimes / 2 - 1)))) || (isOdd && (i < (mLoopTimes / 2)));
            isMid = (!isOdd && hasTail && (i == (mLoopTimes / 2 - 1))) || (isOdd && (i == (mLoopTimes / 2)));
            isTail = (!isOdd && (!hasTail || (i > (mLoopTimes / 2 - 1)))) || (isOdd && (i > (mLoopTimes / 2)));
            LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * L0A_PP_SIZE / sizeof(KV_T)];
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
            LoadData2DParams loadData2DParamsForA;
            loadData2DParamsForA.startIndex = 0;
            if constexpr (KVINT4) {
                loadData2DParamsForA.repeatTimes = (actualBaseM + 15) / 16 * headDimAlign / 64;
            } else {
                loadData2DParamsForA.repeatTimes = (actualBaseM + 15) / 16 * headDimAlign / (32 / sizeof(KV_T));
            }
            loadData2DParamsForA.srcStride = 1;
            loadData2DParamsForA.dstGap = 0;
            loadData2DParamsForA.ifTranspose = false;
            LoadData(aL0Tensor, aL1Tensor[i * baseM * headDimAlign], loadData2DParamsForA);
            SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));
            WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));

            constexpr uint32_t baseN = 256 / sizeof(KV_T);
            uint32_t nLoopTimes = (nActCopyRowCountAlign + baseN - 1) / baseN;
            uint32_t nTail = nActCopyRowCountAlign - (nLoopTimes - 1) * baseN;
            for (uint32_t j = 0, actualBaseN = baseN; j < nLoopTimes; j++) {
                if (j + 1 == nLoopTimes) {
                    actualBaseN = nTail;
                }
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * L0B_PP_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));
                uint32_t blockElementCnt = 32 / sizeof(KV_T);
                if constexpr (KVINT4) {
                    blockElementCnt = 64;
                }
                LoadData2DParams loadData2DParamsForB;
                loadData2DParamsForB.startIndex = 0;
                loadData2DParamsForB.srcStride = 1;
                loadData2DParamsForB.dstGap = 0;
                loadData2DParamsForB.ifTranspose = false;
#ifdef L1_LAYOUT_zN
                loadData2DParamsForB.repeatTimes = actualBaseN / 16;
                uint32_t loadLoopTimes = headDimAlign / blockElementCnt;
                for(uint32_t k = 0; k < loadLoopTimes; k++){
                    LoadData(bL0Tensor[k * actualBaseN * blockElementCnt],
                        bL1Tensor[(j * baseN + k * nActCopyRowCountAlign) * blockElementCnt], loadData2DParamsForB);
                }
#else
                loadData2DParamsForB.repeatTimes = (actualBaseN / 16) * (headDimAlign / blockElementCnt);
                LoadData(bL0Tensor, bL1Tensor[baseN * headDimAlign * j], loadData2DParamsForB);
#endif
                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));

                MmadParams mmadParams;
                mmadParams.m = actualBaseM;
                if (mmadParams.m == 1) {  // m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = actualBaseN; // 无效数据不参与计算
                mmadParams.k = 128;
                mmadParams.cmatrixInitVal = true;
                mmadParams.cmatrixSource = false;

                LocalTensor<L0C_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * L0C_PP_SIZE / sizeof(L0C_T)];
                WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));

                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
                WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));

                if (mLoopTimes == 1) {
                    for (uint32_t mIter = 0; mIter < msdIterNum; mIter++) {
                        float tmp = quantScaleC1S1;
                        if (mIter == 1) {
                            tmp = quantScaleC1S2;
                        }
                        FixpipeParamsV220 fixParams;
                        fixParams.nSize = actualBaseN;
                        fixParams.mSize = actualBaseM / msdIterNum; // 有效数据不足16行，只需要输出部分行即可
                        fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                        fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                        fixParams.ndNum = 1;
                        fixParams.quantPre = QuantMode_t::DEQF16;
                        fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                        if (mIter == 1) {
                            SetAtomicAdd<half>();
                        }
                        Fixpipe(mm1ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * mmResUbSize + i * baseM * info.actualSingleProcessSInnerSizeAlign + nCopyIdx * nCopyRowCount + j * baseN],
                                cL0Tensor[info.mSize * mIter * 16], fixParams);
                        if (mIter == 1) {
                            SetAtomicNone();
                        }
                        PipeBarrier<PIPE_FIX>();
                    }
                } else if (isHead) {
                    float tmp = quantScaleC1S1;
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = actualBaseN;
                    fixParams.mSize = actualBaseM; // 有效数据不足16行，只需要输出部分行即可
                    fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;
                    fixParams.quantPre = QuantMode_t::DEQF16;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    Fixpipe(mm1ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * mmResUbSize + i * baseM * info.actualSingleProcessSInnerSizeAlign + nCopyIdx * nCopyRowCount + j * baseN],
                            cL0Tensor, fixParams);
                    PipeBarrier<PIPE_FIX>();
                } else if (isTail) {
                    float tmp = quantScaleC1S2;
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = actualBaseN;
                    fixParams.mSize = actualBaseM; // 有效数据不足16行，只需要输出部分行即可
                    fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;
                    fixParams.quantPre = QuantMode_t::DEQF16;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    SetAtomicAdd<half>();
                    Fixpipe(mm1ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * mmResUbSize + (i * baseM - info.mSize) * info.actualSingleProcessSInnerSizeAlign + nCopyIdx * nCopyRowCount + j * baseN],
                            cL0Tensor, fixParams);
                    SetAtomicNone();
                    PipeBarrier<PIPE_FIX>();
                } else {
                    float tmp = quantScaleC1S1;
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = actualBaseN;
                    fixParams.mSize = info.mSize - i * baseM; // 有效数据不足16行，只需要输出部分行即可
                    fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;
                    fixParams.quantPre = QuantMode_t::DEQF16;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    Fixpipe(mm1ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * mmResUbSize + i * baseM * info.actualSingleProcessSInnerSizeAlign + nCopyIdx * nCopyRowCount + j * baseN],
                                cL0Tensor, fixParams);
                    PipeBarrier<PIPE_FIX>();
                    tmp = quantScaleC1S2;
                    fixParams.mSize = (i + 1) * baseM - info.mSize;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    SetAtomicAdd<half>();
                    Fixpipe(mm1ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * mmResUbSize + nCopyIdx * nCopyRowCount + j * baseN], cL0Tensor[(info.mSize - i * baseM) * 16], fixParams);
                    SetAtomicNone();
                    PipeBarrier<PIPE_FIX>();
                }
                SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
                cL0BufIter++;
                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));
                bL0BufIter++;
            }
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
            aL0BufIter++;
        }
        SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));
    }
    // will change batch
    if ((info.s2Idx + 1) == info.curSInnerLoopTimes) {
        SetFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0 + qL1BufIter);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t mCopyIdx, uint32_t mCopyRowCount, uint32_t mActCopyRowCount, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount) {
    uint32_t mmRowCount = mActCopyRowCount;
    uint32_t copyStrideL1 = 16 * kActCopyRowCount;
    uint32_t copyStrideGm = 16 * info.actualSingleProcessSInnerSizeAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++){
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = mmRowCount - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数  16
        }
        mm1Nd2NzParamsForA.dValue = kActCopyRowCount; // 单个ND矩阵的列数, 单位为元素个数
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForA.srcDValue = info.actualSingleProcessSInnerSizeAlign / 2;
        } else {
            mm1Nd2NzParamsForA.srcDValue = info.actualSingleProcessSInnerSizeAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
        }
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移
        DataCopy(aL1Tensor[i * copyStrideL1], vec1ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * mmResUbSize + (mCopyIdx * mCopyRowCount) * info.actualSingleProcessSInnerSizeAlign + kCopyIdx * kCopyRowCount + i * copyStrideGm], mm1Nd2NzParamsForA);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount) {
    uint64_t step = headDim;
    if constexpr (KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) {
        step = headDim * kvHeadNum;
    }
    uint64_t valueGmBaseOffset = info.tensorBOffset + kCopyIdx * kCopyRowCount * step;

#ifdef L1_LAYOUT_zN
    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = kActCopyRowCount;
    if constexpr (KVINT4) {
        mm1Nd2NzParamsForB.dValue = headDim / 2;
        mm1Nd2NzParamsForB.srcDValue = step / 2;
    } else {
        mm1Nd2NzParamsForB.dValue = headDim;
        mm1Nd2NzParamsForB.srcDValue = step;
    }
    mm1Nd2NzParamsForB.dstNzC0Stride = Align(kActCopyRowCount, 32U);
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor, valueGm[valueGmBaseOffset], mm1Nd2NzParamsForB);
#else
    uint32_t blockK = 32 / sizeof(KV_T); // 单个ND矩阵的行数
    if constexpr (KVINT4) {
        blockK = 64;
    }
    uint32_t kTail = kActCopyRowCount % blockK;
    uint32_t ndNum_tmp = kActCopyRowCount / blockK;

    Nd2NzParams mm1Nd2NzParamsForB;
    if constexpr (KVINT4) {
        mm1Nd2NzParamsForB.dValue = headDim / 2;
        mm1Nd2NzParamsForB.srcDValue = step / 2;
    } else {
        mm1Nd2NzParamsForB.dValue = headDim;
        mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
    }
    mm1Nd2NzParamsForB.dstNzC0Stride = blockK;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    if (ndNum_tmp != 0){
        if ((KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) && (blockK * step > 65535)) {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = blockK;
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum_tmp; i++) {
                DataCopy(bL1Tensor[i * blockK * headDimAlign], valueGm[valueGmBaseOffset + i * blockK * step], mm1Nd2NzParamsForB);
            }
        } else {
            mm1Nd2NzParamsForB.ndNum = ndNum_tmp;
            mm1Nd2NzParamsForB.nValue = blockK;
            if constexpr (KVINT4) {
                mm1Nd2NzParamsForB.srcNdMatrixStride = blockK * step / 2;
                mm1Nd2NzParamsForB.dstNzMatrixStride = blockK * headDimAlign / 2;
            } else {
                mm1Nd2NzParamsForB.srcNdMatrixStride = blockK * step;
                mm1Nd2NzParamsForB.dstNzMatrixStride = blockK * headDimAlign;
            }
            DataCopy(bL1Tensor, valueGm[valueGmBaseOffset], mm1Nd2NzParamsForB);
        }
    }
    if (kTail != 0){
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = kTail;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[ndNum_tmp * blockK * headDimAlign], valueGm[valueGmBaseOffset + ndNum_tmp * blockK * step], mm1Nd2NzParamsForB);
    }
#endif
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyInMm2BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount)
{
    uint32_t blockK = 32 / sizeof(KV_T); // 单个ND矩阵的行数
    uint32_t blockElementCnt = 32 / sizeof(KV_T);
    if constexpr (KVINT4) {
        blockElementCnt = 64;
    }

    uint64_t step = headDim;
    if constexpr (KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) {
        step = headDim * kvHeadNum;
    } else if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        step = blockElementCnt;
    }

    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInABlockK = copyStartRowCnt % blockK;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInABlockK + kActCopyRowCount <= blockK) {
        nHead = 0;
        midNdNum = 0;
        nTail = kActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInABlockK;
    } else {
        if (copyStartRowCntInABlockK == 0) {
            nHead = 0;
        } else {
            nHead = blockK - copyStartRowCntInABlockK;
        }
        midNdNum = (kActCopyRowCount - nHead) / blockK;
        nTail = (kActCopyRowCount - nHead) % blockK;
        copyStartRowCntInBaseNTail = 0;
    }

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.dValue = headDim;
    mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
    mm1Nd2NzParamsForB.dstNzC0Stride = blockK;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    if (nHead != 0) {
        if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
            DataCopyParams intriParams;
            intriParams.blockLen = nHead;
            intriParams.blockCount = headDim / blockElementCnt;
            intriParams.dstStride = blockK - nHead;
            intriParams.srcStride = kvCacheBlockSize - nHead;

            uint32_t ndNumFinish = copyStartRowCnt / blockK;
            DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInABlockK * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset], intriParams);
        } else {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = nHead; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

            uint32_t ndNumFinish = copyStartRowCnt / blockK;
            DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInABlockK * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset],
                    mm1Nd2NzParamsForB);
        }
    }

    if (midNdNum != 0) {
        if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
            DataCopyParams intriParams;
            intriParams.blockLen = blockK;
            intriParams.blockCount = headDim / blockElementCnt;
            intriParams.dstStride = 0;
            intriParams.srcStride = kvCacheBlockSize - blockK;

            int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK;
            for (uint32_t i = 0; i < midNdNum; i++) {
                DataCopy(bL1Tensor[(ndNumFinish + i) * blockK * headDimAlign], valueGm[valueGmBaseOffset + nHead * step + i * blockK * step], intriParams);
            }
        } else {
            int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK;
            if ((KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) && (blockK * step > 65535)) {
                mm1Nd2NzParamsForB.ndNum = 1;
                mm1Nd2NzParamsForB.nValue = blockK; // 单个ND矩阵的行数, 单位为元素个数
                mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
                mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

                for (uint32_t i = 0; i < midNdNum; i++) {
                    DataCopy(bL1Tensor[(ndNumFinish + i) * blockK * headDimAlign], valueGm[valueGmBaseOffset + nHead * step + i * blockK * step], mm1Nd2NzParamsForB);
                }
            } else {
                mm1Nd2NzParamsForB.ndNum = midNdNum;
                mm1Nd2NzParamsForB.nValue = blockK; // 单个ND矩阵的行数, 单位为元素个数
                mm1Nd2NzParamsForB.srcNdMatrixStride = blockK * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
                mm1Nd2NzParamsForB.dstNzMatrixStride = blockK * headDimAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

                DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign], valueGm[valueGmBaseOffset + nHead * step], mm1Nd2NzParamsForB);
            }
        } 
    }

    if (nTail != 0) {
        if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
            DataCopyParams intriParams;
            intriParams.blockLen = nTail;
            intriParams.blockCount = headDim / blockElementCnt;
            intriParams.dstStride = blockK - nTail;
            intriParams.srcStride = kvCacheBlockSize - nTail;

            int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK + midNdNum;
            DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset + (nHead + midNdNum * blockK) * step], intriParams);
        } else {
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

            int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK + midNdNum;
            DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset + (nHead + midNdNum * blockK) * step],
                    mm1Nd2NzParamsForB);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize) {
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    if constexpr (KVINT4) {
        loadData2DParams.repeatTimes = kSize / 64;
    } else {
        loadData2DParams.repeatTimes = kSize / (32 / sizeof(KV_T));
    }
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::LoadDataMm2B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor) {
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::ComputeMm2(ExtraInfo& info) {
    constexpr uint32_t mCopyRowCount = P_LOAD_TO_L1_ROW_NUM;
    uint32_t mActRowCount = msdIterNum * info.mSize;
    uint32_t mCopyTimes = (mActRowCount + mCopyRowCount - 1) / mCopyRowCount;
    uint32_t mTailCopyRowCount = mActRowCount - (mCopyTimes - 1) * mCopyRowCount;

    constexpr uint32_t kCopyRowCount = KV_LOAD_TO_L1_ROW_NUM;
    uint32_t kCopyTimes = (info.actualSingleProcessSInnerSize + kCopyRowCount - 1) / kCopyRowCount;
    uint32_t kTailCopyRowCount = info.actualSingleProcessSInnerSize - (kCopyTimes - 1) * kCopyRowCount;
    uint32_t kTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (kCopyTimes - 1) * kCopyRowCount;

    for (uint32_t mCopyIdx = 0, mActCopyRowCount = mCopyRowCount; mCopyIdx < mCopyTimes; mCopyIdx++) {
        if (mCopyIdx + 1 == mCopyTimes) {
            mActCopyRowCount = mTailCopyRowCount;
        }
        LocalTensor<L0C_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * L0C_PP_SIZE / sizeof(L0C_T)];
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
        for (uint32_t kCopyIdx = 0, kActCopyRowCount = kCopyRowCount, kActCopyRowCountAlign = kCopyRowCount; kCopyIdx < kCopyTimes; kCopyIdx++) {
            if (kCopyIdx + 1 == kCopyTimes) {
                kActCopyRowCount = kTailCopyRowCount;
                kActCopyRowCountAlign = kTailCopyRowCountAlign;
            }
            kpL1BufIter++;
            LocalTensor<KV_T> aL1Tensor = kpL1Buffers[(kpL1BufIter % 3) * L1KP_BLOCK_SIZE / sizeof(KV_T)];
            WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));
            CopyInMm2AToL1(aL1Tensor, info, mCopyIdx, mCopyRowCount, mActCopyRowCount, kCopyIdx, kCopyRowCount, kActCopyRowCountAlign);
            SetFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));
            WaitFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));
            LocalTensor<KV_T> bL1Tensor = vL1Buffers[(vL1BufIter % 4) * L1V_BLOCK_SIZE / sizeof(KV_T)];
            uint32_t kb = 0;
            if (mCopyIdx == 0) {
                vL1BufIter++;
                bL1Tensor = vL1Buffers[(vL1BufIter % 4) * L1V_BLOCK_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0 + (vL1BufIter % 4));
                if constexpr (PAGE_ATTENTION) {
                    uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
                    uint32_t curSeqIdx = info.s2BatchOffset + kCopyIdx * kCopyRowCount;
                    uint32_t copyFinishRowCnt = 0;
                    while (copyFinishRowCnt < kActCopyRowCount) {
                        uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                        uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                        uint64_t idInBlockTable =
                            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
                        // 计算可以拷贝行数
                        uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                        if (copyFinishRowCnt + copyRowCnt > kActCopyRowCount) {
                            copyRowCnt = kActCopyRowCount - copyFinishRowCnt;
                        }
                        uint64_t valueOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                        if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
                            uint32_t blockElementCnt = 32 / sizeof(KV_T);
                            if constexpr (KVINT4) {
                                blockElementCnt = 64;
                            }
                            valueOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                        } else {
                            if constexpr (KV_LAYOUT_T == LAYOUT::BSH || KV_LAYOUT_T == LAYOUT::BSND) {
                                valueOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                            } else {
                                valueOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                            }
                        }
                        CopyInMm2BToL1ForPA(bL1Tensor, valueOffset, kActCopyRowCount, copyFinishRowCnt, copyRowCnt);

                        // 更新循环变量
                        copyFinishRowCnt += copyRowCnt;
                        curSeqIdx += copyRowCnt;
                    }
                } else {
                    CopyInMm2BToL1(bL1Tensor, info, kCopyIdx, kCopyRowCount, kActCopyRowCount);
                }
                SetFlag<HardEvent::MTE2_MTE1>(L1V_EVENT0 + (vL1BufIter % 4));
                WaitFlag<HardEvent::MTE2_MTE1>(L1V_EVENT0 + (vL1BufIter % 4));
                kb = vL1BufIter;
            } else {
                kb = vL1BufIter - (kCopyTimes - kCopyIdx - 1);
                bL1Tensor = vL1Buffers[(kb % 4) * L1V_BLOCK_SIZE / sizeof(KV_T)];
            }
            constexpr uint32_t baseK = 128 / sizeof(KV_T);
            uint32_t kLoopTimes = (kActCopyRowCountAlign + baseK - 1) / baseK;
            uint32_t kTailAlign = kActCopyRowCountAlign - (kLoopTimes - 1) * baseK;
            uint32_t kTail = kActCopyRowCount - (kLoopTimes - 1) * baseK;
            for (uint32_t i = 0, actualBaseKAlign = baseK, actualBaseK = baseK; i < kLoopTimes; i++) {
                if (i + 1 == kLoopTimes) {
                    actualBaseKAlign = kTailAlign;
                    actualBaseK = kTail;
                }

                LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * L0A_PP_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * L0B_PP_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));

                LocalTensor<KV_T> curAL1Tensor = aL1Tensor[16 * baseK * i];

                uint32_t mmRowCount = mActCopyRowCount;
                uint32_t copyStrideL0 = 16 * actualBaseKAlign;
                uint32_t copyStrideL1 = 16 * kActCopyRowCountAlign;
                uint32_t copyIterNum = (mmRowCount + 15) / 16;
                for(int i = 0; i < copyIterNum; i++){
                    LoadDataMm2A(aL0Tensor[i * copyStrideL0], curAL1Tensor[i * copyStrideL1], actualBaseKAlign);
                }
                
                SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));
                WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));

                uint32_t blockElementCnt = 32 / sizeof(KV_T);
                if constexpr (KVINT4) {
                    blockElementCnt = 64;
                }
                LoadData2dTransposeParams loadData2DTransposeParamsForB;
                loadData2DTransposeParamsForB.startIndex = 0;
                loadData2DTransposeParamsForB.srcStride = 1;
                loadData2DTransposeParamsForB.dstFracGap = 0;
#ifdef L1_LAYOUT_zN
                loadData2DTransposeParamsForB.repeatTimes = actualBaseKAlign / blockElementCnt;
                loadData2DTransposeParamsForB.dstGap = headDimAlign / 16 - 1;
                uint32_t loadLoopTimes = headDimAlign / blockElementCnt;
                for(uint32_t j = 0; j < loadLoopTimes; j++){
                    LoadDataWithTranspose(bL0Tensor[j * blockElementCnt * blockElementCnt],
                        bL1Tensor[(i * baseK + j * kActCopyRowCountAlign) * blockElementCnt], loadData2DTransposeParamsForB);
                }
#else
                loadData2DTransposeParamsForB.repeatTimes = (actualBaseKAlign / blockElementCnt) * (headDimAlign / blockElementCnt);
                loadData2DTransposeParamsForB.dstGap = blockElementCnt / 16 - 1;
                uint32_t l1BaseOffset = baseK * headDimAlign * i;
                LoadDataWithTranspose(bL0Tensor, bL1Tensor[l1BaseOffset], loadData2DTransposeParamsForB);
#endif
                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));

                MmadParams mmadParams;
                mmadParams.m = mActCopyRowCount;
                if (mmadParams.m == 1) {  // m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = 128;
                mmadParams.k = actualBaseK; // 无效数据不参与计算
                mmadParams.cmatrixInitVal = (kCopyIdx == 0) && (i == 0);
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                PipeBarrier<PIPE_M>();

                SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
                aL0BufIter++;
                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));
                bL0BufIter++;
            }

            SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));
            if ((mCopyIdx + 1) == mCopyTimes) {
                SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0 + (kb % 4));
            }
        }         
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
        if (mCopyTimes == 1) {
            for (uint32_t mIter = 0; mIter < msdIterNum; mIter++) {
                float tmp = quantScaleC2O1;
                if (mIter == 1) {
                    tmp = quantScaleC2O2;
                }
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = mActCopyRowCount / msdIterNum; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((msdIterNum * fixParams.mSize + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                if (mIter == 1) {
                    SetAtomicAdd<half>();
                }
                Fixpipe(mm2ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * bmm2ResUbSize], cL0Tensor[mIter * fixParams.mSize * 16], fixParams);
                if (mIter == 1) {
                    SetAtomicNone();
                }
                PipeBarrier<PIPE_FIX>();
            }
        } else {
            if (mTailCopyRowCount != mCopyRowCount && mCopyIdx == 0) {
                float tmp = quantScaleC2O1;
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = info.mSize; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((mActCopyRowCount + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                Fixpipe(mm2ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * bmm2ResUbSize], cL0Tensor, fixParams);
                PipeBarrier<PIPE_FIX>();
                tmp = quantScaleC2O2;
                fixParams.mSize = mActCopyRowCount - info.mSize;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                SetAtomicAdd<half>();
                Fixpipe(mm2ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * bmm2ResUbSize], cL0Tensor[info.mSize * 16], fixParams);
                SetAtomicNone();
                PipeBarrier<PIPE_FIX>();
            } else if (mCopyIdx == 0) {
                float tmp = quantScaleC2O1;
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = mActCopyRowCount; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((mActCopyRowCount + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                Fixpipe(mm2ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * bmm2ResUbSize], cL0Tensor, fixParams);
                PipeBarrier<PIPE_FIX>();
            } else {
                float tmp = quantScaleC2O2;
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = mActCopyRowCount; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((mActCopyRowCount + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                SetAtomicAdd<half>();
                Fixpipe(mm2ResGm[(info.loop % (PRE_LOAD_NUM_DD)) * bmm2ResUbSize + (mCopyIdx * mCopyRowCount - info.mSize) * headDimAlign], cL0Tensor, fixParams);
                SetAtomicNone();
                PipeBarrier<PIPE_FIX>();
            }
        }
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
        cL0BufIter++;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::Process()
{

    if ASCEND_IS_AIV {
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    }
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_MAX_BUF_SIZE);
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, SOFTMAX_SUM_BUF_SIZE);
#else
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_512B / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_512B / sizeof(T));
#endif   
        } else {
            SetFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT1);
            SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT2);
            SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT1);
            SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT2);
            SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT3);
           
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

            SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
            SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }

        ExtraInfo extraInfo[EXTRA_NUM];
        TaskContext taskContext[TASK_NUM];
        uint32_t gloop = 0;
        uint32_t gtasks = 0; //gtasks记录全局已处理了多少个Task
        uint32_t gtasksC1V1 = 0;
        uint32_t gtasksC2V2 = 0;
        uint32_t tasks = 0; //记录每组Task数量（预加载task的数量）

        if constexpr (!FLASH_DECODE) {
            for (uint32_t bn2gIdx = bn2LoopTimes; bn2gIdx > 0; bn2gIdx--) {
                GetBN2Gid(bn2gIdx - 1);
                GetActualSeqLen(bIdx, s1Idx);
                if (curActualSeqLen == 0) {
                    DealActSeqLenIsZero(bIdx, n2Idx);
                }
                if (curActualSeqLen != 0 && actS1Size != 0) {
                    break;
                }
                bn2LoopTimes--;
            }
        } 

        for (uint32_t bn2gIdx = 0; bn2gIdx < bn2LoopTimes; bn2gIdx++) {
            GetBN2Gid(bn2gIdx);
            GetActualSeqLen(bIdx, s1Idx);
            UpdateInnerLoopCond();
            if (curActSeqLenIsZero) {
                DealActSeqLenIsZero(bIdx, n2Idx);
                continue;
            }
            if (actS1Size == 0) {
                continue;
            }
            uint32_t curS2LoopTimes = (bn2gIdx == bn2LoopTimes - 1) ? sInnerLoopTimes + 2 : sInnerLoopTimes;
            for (uint32_t sInnerLoopIdx = 0; sInnerLoopIdx < curS2LoopTimes; sInnerLoopIdx++) {
                if (sInnerLoopIdx < sInnerLoopTimes) {
                    TaskContext &ctx = taskContext[tasks++];
                    ctx.bidx = bIdx;
                    ctx.nidx = n2Idx;
                    ctx.s2idx = sInnerLoopIdx;
                    ctx.s2loops = sInnerLoopTimes;
                    ctx.gidx = gIdx;
                    ctx.s2SizeTail = singleProcessSInnerSizeTail;
                    ctx.s1Size = actS1Size;
                    ctx.s2Size = curActualSeqLen;
                    ctx.s1idx = s1Idx;
                    ctx.actSeqLenQ = curActualSeqLenQ;
                    bool isLast = (bn2gIdx == bn2LoopTimes - 1) && (sInnerLoopIdx >= sInnerLoopTimes - 1);
                    if (tasks < TASK_NUM && !isLast) {
                        continue;
                    }
                    //遍历Task任务
                    for (uint32_t i = 0; i < tasks; i++) {
                        uint32_t loop = gtasks;
                        CalcParams(loop, extraInfo[loop % EXTRA_NUM], taskContext[i]);

                        if ASCEND_IS_AIV {
                            if constexpr (ANTIQUANT) {
                                QueryPreProcessL(extraInfo[loop % EXTRA_NUM]);
                                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V0_C1_FLAG);
                            }
                        }
                        gtasks++;
                    }
                }              

                for (uint32_t i = 0; i < TASK_NUM; i++) {
                    uint32_t loop = gtasksC1V1;
                    if (gloop > 0 && gtasksC1V1 < gtasks) {
                        if ASCEND_IS_AIC {
                            if constexpr (ANTIQUANT) {
                                CrossCoreWaitFlag(SYNC_V0_C1_FLAG);
                            }
                            ComputeMm1(extraInfo[loop % EXTRA_NUM]);
                            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
                        }
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
                            ProcessVec1L(extraInfo[loop % EXTRA_NUM]);
                            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);
                        }
                        gtasksC1V1++;
                    }
                }

                for (uint32_t i = 0; i < TASK_NUM; i++) {
                    uint32_t loop = gtasksC2V2;
                    if (gloop > 1 && gtasksC2V2 < gtasks) {
                        if ASCEND_IS_AIC {
                            CrossCoreWaitFlag(SYNC_V1_C2_FLAG);
                            ComputeMm2(extraInfo[loop % EXTRA_NUM]);
                            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG);
                        }
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                            ProcessVec2L(extraInfo[loop % EXTRA_NUM]);
                        }
                        gtasksC2V2++;
                    }
                }
                tasks = 0;
                gloop++;
            }
        }

        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT1);
            WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT2);
            WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT1);
            WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT2);
            WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT3);

            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }
    }

    if constexpr (FLASH_DECODE) {
        // 多核同步
        SyncAll();
        if ASCEND_IS_AIV {
            FlashDecodeCompute();
        }
    }
    if ASCEND_IS_AIV {
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::CopyFixedUbToGm(const GlobalTensor<T> &dst,
                                                                                     const LocalTensor<T> &src,
                                                                                     size_t size)
{
    LocalTensor<T> tmp = outputBuf2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
#ifdef IFA_SOFTMAX_WITHOUT_BRC_PRELOAD
    Brcb(tmp, src, (mSizeVector + 7) / 8, {1, 8}); //将m*1数据扩展为m*8匹配后续使用
#else
    DataCopy(tmp, src, size);
#endif
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopy(dst, tmp, size);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadDD<IFAT>::DealKvInt4ColumnOdd(LocalTensor<T> mmResUb,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (KVINT4) {
        if (actualColumnCount % 2 == 1) {
            uint32_t blockIdx = actualColumnCount / FP32_ONE_BLOCK_SIZE;
            uint32_t offset = blockIdx * FP32_ONE_BLOCK_SIZE;
            uint32_t maskIdx = actualColumnCount % FP32_ONE_BLOCK_SIZE;
            uint64_t mask[2] = {1U << maskIdx, 0};
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Duplicate(mmResUb[i * columnCount + offset], FLOAT_ZERO, mask, 1, 1, 8);
            }
            PipeBarrier<PIPE_V>();
        }
    }
}
#endif // INCRE_FLASH_ATTENTION_PRELOAD
