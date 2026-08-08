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
 * \file incre_flash_attention_preload_mla.h
 * \brief
 */

#ifndef INCRE_FLASH_ATTENTION_PRELOAD_MLA
#define INCRE_FLASH_ATTENTION_PRELOAD_MLA

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#include "adv_api/activation/softmax.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../ifa_public_define.h"
#include "ifa_service_matmul_full_quant.h"
#include "service_vector_flashdecode_mla.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using namespace AttentionCommonFlashDecode;

#define USE_SERVICE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define QMAX_BUF_SIZE  (BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B)

namespace {
    struct TaskContext {
        uint32_t bidx;
        uint32_t gidx;
        uint32_t s1idx;
        uint32_t s2idx;
        uint32_t s2loops;
        uint32_t s2SizeTail;
        uint32_t s1Size;
        uint32_t s2Size;
        uint32_t tndIsS2SplitCore;  // TND格式，[B, N2, S1]三根轴确定后，S2是否被切到了多核上
        uint32_t tndCoreStartKVSplitPos;
        uint32_t isFirstLoop;
        static constexpr uint32_t nidx = 0;
        uint64_t actS1Size = 1;
        bool isValid = false;
        bool isLastTask = false;
    };

    struct TransposeInfo {
        // 以下是FlashDecode分支区分的信息
        uint32_t n2Idx = 0;
        uint32_t bIdx = 0;
        uint32_t gSize = 0;
        uint32_t s1Size = 0;
        // 以下是需要用公式计算的信息
        uint32_t s1StartIdx = 0;
        uint32_t s1EndIdx = 0;
        uint32_t s1Count = 0;
        uint32_t gStartIdx = 0;
        uint32_t gEndIdx = 0;
        uint32_t gCount = 0;
    };
}

template <typename IFAT> class IncreFlashAttentionAttenPreloadMla {
public:
    __aicore__ inline IncreFlashAttentionAttenPreloadMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const IncreFlashAttentionTilingDataMla *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe, bool isPrefix = false);
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                     __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *dequantScaleQuery,
                                     __gm__ uint8_t *workspace);

    __aicore__ inline void InitAntiquant(__gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                         __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                         __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                         __gm__ uint8_t *keyRopeAntiquantScale);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void ProcessBXXD();

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr LAYOUT KV_LAYOUT_T = IFAT::kvLayout;
    static constexpr AMLAMODE AMLA = IFAT::isAMla;
    static constexpr bool BALANCE = IFAT::isBalance;
    static constexpr bool ENABLE_TREE = IFAT::enableTree;
    
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t ANTIQUANT_MODE = IFAT::antiquantMode;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;

    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));

    using ANTIQ_PARAMS_T = Q_T;

    using Q_ROPE_T = typename AscendC::Conditional<ANTIQUANT, Q_T, ORIGIN_T>::type;
    using K_ROPE_T = typename AscendC::Conditional<ANTIQUANT, KV_T, ORIGIN_T>::type;

#ifdef QUANT_MM2_FP16
    using UPDATE_T = typename AscendC::Conditional<QUANT || ANTIQUANT, half, T>::type;
#else
    using UPDATE_T = typename AscendC::Conditional<ANTIQUANT, half, T>::type;
#endif

    using TMP_T = typename AscendC::Conditional<ANTIQUANT, half, T>::type;
    using MM1_OUT_T = typename AscendC::Conditional<QUANT, int32_t, TMP_T>::type;
#ifdef QUANT_MM2_FP16
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, half, TMP_T>::type;
#else
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, int32_t, TMP_T>::type;
#endif

    static constexpr float deqScaleV = 256.0;
    static constexpr float quantScaleC = 1.0 / 256;
    static constexpr float scaleP1 = 127.0;
    // static constexpr float scaleP2 = 1 / 127.0;
    static constexpr float scaleP2 = 0.0078740157f;
    static constexpr uint32_t PRELOAD_NUM = 2;
    ServiceFlashDecode<IFAT> fdService;
    static constexpr uint32_t mFdBaseSizeMla = 8;
    static constexpr int64_t fdPrefetchLen = 2;

protected:
    const IncreFlashAttentionTilingDataMla *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_ROPE_T> qRopeGm;
    GlobalTensor<K_ROPE_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int32_t> blockTableGm;

    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;

    // antiquant
    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqScaleGm;
    GlobalTensor<ANTIQ_PARAMS_T> keyRopeAntiquantScaleGm;

    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqScaleGm;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

    // quant
    GlobalTensor<T> deqScale1Gm;
    T deqScaleKey;
    T deqScale2Val;

    // workspace
    GlobalTensor<KV_T> queryPreProcessResGm; // 存放Q1, Q2
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    GlobalTensor<KV_T> vec1ResGm; // 存放A1, A2
    GlobalTensor<MM2_OUT_T> mm2ResGm; // 存放O

    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<int32_t> nUpdateGm;
    GlobalTensor<T> softmaxSumGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1; // 32K, inque
    TQue<QuePosition::VECIN, 1> inputQue2; // 16K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 32K, outque
    TQue<QuePosition::VECOUT, 1> outputQue2; // 8K, outque

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K
    TBuf<> tmpBuff3; // 2K

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff; // PRE_LOAD_NUM_MLA * 2K
    TBuf<> softmaxExpBuff; // PRE_LOAD_NUM_MLA * 2K
    TBuf<> softmaxSumBuff; // PRE_LOAD_NUM_MLA * 2K

    TBuf<> dequantScale1Buff;
    LocalTensor<T> dequantScale1Ub;
    TBuf<> pMaxBuff;
    LocalTensor<T> pMaxUb;

    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    TBuf<> antiqScaleBuff; // 4K
    TBuf<> antiqOffsetBuff; // 4K
    TBuf<> qAmaxBuff; // PRE_LOAD_NUM_MLA * (2K + 256B)

    // antiquant msd
    LocalTensor<T> qAmaxUb;
    LocalTensor<T> antiqScaleUb;
    LocalTensor<T> antiqOffsetUb;
    AttentionCommonFlashDecode::ConstInfoFD constInfo{};

    uint64_t msdRowMaxSize = 0;
    uint64_t msdRowSumSize = 0;
    uint64_t softmaxMaxSumExpSize = 0;

    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint64_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t SYNC_C2_V2_FLAG = 9;

    static constexpr int32_t FP32_MAX_MASK_ELEMENT_NUM = 64;
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T); // 256/4=64
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t ADDRESS_ALIGN_NUM = 512 / sizeof(KV_T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM_THRESHLOD = 128 / sizeof(KV_T);

    half antiquantExpandCoeff = 254;

    static constexpr T antiqCoeff1 = 127;
    static constexpr T antiqCoeff2 = 1 / antiqCoeff1;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T FLOAT_E_SCALAR = 8388608; // pow(2, 23)
    static constexpr T FLOAT_INF = std::numeric_limits<T>::infinity();
    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint64_t headDimAll  = 576ULL;
    static constexpr bool batchContinuous = true;
    static constexpr uint32_t n2Idx = 0U;
    // static constexpr float scaleC1  = 1024.0 / 127;
    static constexpr float scaleC1  = 8.06299213f;
    static constexpr float scaleC2  = 1024.0;
    static constexpr uint32_t msdIterNum = 2U;

    bool antiqOffsetExistFlag = false;
    uint32_t antiquantPerTensorFlag = 0U;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRE_LOAD_NUM_MLA;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t subBlockNum = 2U; // AICORE 上AIC与AIV的数量默认是1:2

    // tilingdata
    uint64_t singleProcessSInnerSize = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t s1Idx = 0U;

    uint32_t mmResUbSize = 0U;

    uint32_t vec1ResUbSize = 0U;

    uint32_t bmm2ResUbSize = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t gSize = 0ULL;

    uint64_t actS1Size = 1ULL;
    uint64_t gSizeSub = 0ULL;
    uint64_t gSizeTail = 0ULL;
    uint64_t s1SizeSub = 0ULL;
    uint64_t s1SizeTail = 0ULL;
    uint64_t gOuter = 1ULL;
    uint64_t s1Outer = 1ULL;
    uint64_t gIdx = 0ULL;

    uint64_t mSizeVector = 0ULL;
    uint64_t mSizeVStart = 0ULL;
    uint64_t kvSeqSize = 0ULL;
    uint64_t qSeqSize = 1ULL;
    uint64_t tSeqSize = 1ULL;   // TND T轴总长度（含padding）

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t s2BatchBaseOffset = 0;

    bool softmaxLseFlag = false;

    // attention mask
    bool attenMaskFlag = false;
    uint32_t attenMaskSizeAlign = 0U;
    uint32_t sparseMode = 3U;

    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t attenMaskSize = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint32_t s2IdxFD = 0U;
    uint64_t sInnerLoopSize = 0ULL;
    uint32_t actualCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenQDims = 0U;
    uint32_t actualLenDims = 0U;
    uint32_t gMax = 128U;

    uint32_t tndSgBasicSize; // TND格式，cube M轴的tiling大小
    // TND分核信息
    uint32_t bEnd = 0U;
    static constexpr uint32_t n2End = 0U;
    uint32_t s1OuterIdxEnd = 0U;
    uint32_t s2End = 0U;
    uint32_t bStart = 0U;
    uint32_t n2Start = 0U;
    uint32_t s1OuterIdxStart = 0U;
    uint32_t s2Start = 0U;
    uint32_t bEndPrev = 0U;
    static constexpr uint32_t n2EndPrev = 0U;
    uint32_t s1OuterIdxEndPrev = 0U;
    uint32_t s2EndPrev = 0U;
    uint32_t tndFDCoreArrLen = 0U;
    uint32_t coreStartKVSplitPos = 0U;
    TNDFDSplitInfo tndFDCoreInfo;

    // 输出transpose信息
    LAYOUT outputLayout;
    // 记录当前轮的bIdx nIdx s2Idx actualLen
    uint32_t bn2IdxInCurCore = 0;
    using MatmulServiceFullQuant = IfaMatmulFullQuant<IFAT>;
    using MatmulServiceQuant = MatmulServiceFullQuant;

    using MatmulServiceType = MatmulServiceQuant;
    MatmulServiceType matmulService;

    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void CalcParams(uint32_t loop, ExtraInfoMla &info, TaskContext &task);

    __aicore__ inline void NBufferPipeline(uint32_t sInnerLoopIdx, uint32_t sInnerLoopTimes, uint32_t tasks,
        uint32_t gloop, ExtraInfoMla extraInfo[PRE_LOAD_NUM_MLA], TaskContext taskContext[PRE_LOAD_NUM_MLA]);
    __aicore__ inline void PreloadPipeline(uint32_t loop, ExtraInfoMla extraInfo[IFA_PRELOAD_TASK_CACHE_SIZE], TaskContext &ctx);

    __aicore__ inline void ComputeMm1(const ExtraInfoMla &info);
    __aicore__ inline void ProcessVec1L(const ExtraInfoMla &info);
    __aicore__ inline void ComputeMm2(const ExtraInfoMla &info);
    __aicore__ inline void ProcessVec2L(const ExtraInfoMla &info);
    __aicore__ inline void QueryPreProcessL(const ExtraInfoMla &info);

    bool curActSeqLenIsZero = false;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
    }

    template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T, int4b_t>::value) {
            return (s + 63) / 64 * 64;
        }
        size_t n = (32 / sizeof(KV_T));
        return (s + n - 1) / n * n;
    }

    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void GetBN2Gid(const uint32_t bn2gIdx);

    __aicore__ inline void GetTNDAxisStartId(uint32_t &bStart, uint32_t &n2Start, uint32_t &s1OuterStart,
                                            uint32_t &s2Start, int bEnd, int n2End, int s1OuterIdxEnd, int s2End);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<uint64_t> &actualSeqLengthsTND, int bIdx);

    __aicore__ inline uint64_t GetTNDBatchOffset(int bIdx);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount, uint32_t actualColumnCount);
    __aicore__ inline void AttenMaskCopyIn(const ExtraInfoMla& info);

    __aicore__ inline void CopyAntiquantScale(LocalTensor<T> &castUb, GlobalTensor<Q_T> srcGm, uint64_t offset);

    __aicore__ inline void CopyAntiqQuery(LocalTensor<T> &dst, GlobalTensor<T> &srcGm, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);

    template <typename RT>
    __aicore__ inline void RowsCopyGmToUb(const LocalTensor<RT> &dst, const GlobalTensor<RT> &src, uint32_t rows,
                                          uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb, LocalTensor<T> tmpAUb,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<half> &tmpA1, LocalTensor<half> &tmpA2,
                                                uint32_t calcSize, bool isFirst, uint64_t outOffset);

    __aicore__ inline void AntiquantMatmulPreProcess(const ExtraInfoMla &info, GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb,
                                                     LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void AntiquantSoftmaxResPreProcess(const ExtraInfoMla &info, GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb,
                                                         LocalTensor<T> tmpAFloorUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount);
    __aicore__ inline void DealQueryPreProcessBaseBlock(const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount);

    __aicore__ inline void QueryPreProcessInner(const ExtraInfoMla &info);

    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void FlashDecodeComputeND();
    __aicore__ inline void FlashDecode();

    __aicore__ inline void DealBmm1ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMatmulResCombine(const ExtraInfoMla &info, LocalTensor<T> bmmResUb, GlobalTensor<MM1_OUT_T> srcGm,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount, float scaleC);
    __aicore__ inline void AntiquantMM2ResCombine(const ExtraInfoMla &info, LocalTensor<MM2_OUT_T> bmmResUb, GlobalTensor<MM2_OUT_T> srcGm,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec1Inner(const ExtraInfoMla &info);

    __aicore__ inline void ComputeSoftmaxLse(LocalTensor<T> softmaxlseUb, LocalTensor<T> &lseSumUb,
                                             LocalTensor<T> &lseMaxUb, uint32_t dealRowCountAlign);
    __aicore__ inline void DealSoftmaxLseInvalidRows(LocalTensor<T> &softmaxlseUb, LocalTensor<T> &lseMaxUb,
                                                     uint32_t dealRowCount, uint64_t s1Size, uint32_t curS1Idx);
    __aicore__ inline void SoftmaxLseCopyOut(const ExtraInfoMla &info, LocalTensor<T> &lseSumUb,
                                             LocalTensor<T> &lseMaxUb, uint32_t dealRowCount);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    __aicore__ inline void DealBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void DealQuantBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);

    __aicore__ inline void DealBmm2ResDualBaseBlock(uint32_t innerCount, uint32_t mStartRow, const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void ProcessVec2DualInner(uint32_t mIdx, const ExtraInfoMla &info, uint32_t mStartRow, uint32_t mdealSize);

    __aicore__ inline void ProcessVec2Inner(const ExtraInfoMla &info);

    __aicore__ inline void SoftmaxFlashV2Compute(const ExtraInfoMla &info, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline bool IsSkipAttenMask(const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount);
    // 针对BSND/BSH/TND等切G的格式拷贝AttentionMask
    __aicore__ inline void AttenMaskCopyForSplitG(const ExtraInfoMla &info, LocalTensor<bool> &attenMaskUb, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void AttenMaskCopyForTree(const ExtraInfoMla &info, LocalTensor<bool> &attenMaskUb, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void AttenMaskCopyNoFull(LocalTensor<bool> &attenMaskUb, const ExtraInfoMla &info, uint32_t s1StartIdx, uint32_t s1EndIdx);

    __aicore__ inline void ElewiseCompute(const ExtraInfoMla &info, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutNBSDGTiling(LocalTensor<OUT_T> &attenOutUb, const TransposeInfo& transInfo);
    __aicore__ inline void Bmm2DataCopyOutNBSDMTiling(LocalTensor<OUT_T> &attenOutUb, const TransposeInfo& transInfo);
    __aicore__ inline void Bmm2DataCopyOutTrans(const ExtraInfoMla& info, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                              uint32_t columnCount, uint32_t actualColumnCount);
    template <typename RT>
    __aicore__ inline void DealInvalidRows(const ExtraInfoMla &info, LocalTensor<RT> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void CombineSplitKVRes(uint64_t baseOffset = 0);
    __aicore__ inline void CopyAccumOutIn(uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset = 0);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const ExtraInfoMla &info, LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void Bmm2FDDataCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax, uint32_t startRow,
                                             uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow,
                                          uint32_t dealRowCount, uint64_t baseOffset = 0);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void InitSoftmaxLseAllInfOutput(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);

    __aicore__ inline void CopyFixedUbToGm(const GlobalTensor<T> &dst, const LocalTensor<T> &src, size_t size);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bIdx, uint32_t s1Idx);
};

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSize;
    usedCoreNum = tilingData->increFlashAttentionSingleCoreParams.usedCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    mmResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.mmResUbSize;

    vec1ResUbSize = mmResUbSize * msdIterNum;

    bmm2ResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    qHeadNum = gSize = tilingData->baseParams.nNumOfQInOneGroup;

    gSizeSub = tilingData->increFlashAttentionSingleCoreParams.groupSplitSize; // 切块大小Gi
    gOuter = (gSize + gSizeSub - 1) / gSizeSub;
    gSizeTail = gSize - (gOuter - 1) * gSizeSub;

    kvSeqSize = tilingData->baseParams.seqSize;
    qSeqSize = tilingData->baseParams.qSeqSize;
    tSeqSize = tilingData->baseParams.tSeqSize;

    s1SizeSub = tilingData->increFlashAttentionSingleCoreParams.s1SplitSize; // 切块大小Si
    s1Outer = (qSeqSize + s1SizeSub - 1) / s1SizeSub;
    s1SizeTail = qSeqSize - (s1Outer - 1) * s1SizeSub;

    attenMaskFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attenMaskSize = tilingData->baseParams.attenMaskSize;
    sparseMode = tilingData->baseParams.sparseMode;

    softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;
    outputLayout = static_cast<LAYOUT> (tilingData->baseParams.outputLayout);
    tndSgBasicSize = s1SizeSub * gSize;
    // flashdecode新增结构体
    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.batchSize = tilingData->baseParams.batchSize;
    constInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.qHeadNum = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.headDimAlign = headDimAlign;
    constInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    constInfo.outputLayout = static_cast<LAYOUT> (tilingData->baseParams.outputLayout);
    constInfo.mBaseSize = tilingData->baseParams.nNumOfQInOneGroup * tilingData->increFlashAttentionSingleCoreParams.s1SplitSize;
}

#define QMAX_UB_SIZE (BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_256B)

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        // queue
        pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_4K);

        // tmpBuff
        pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff3, BUFFER_SIZE_BYTE_2K);

        if constexpr (QUANT) {
            pipe->InitBuffer(dequantScale1Buff, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_256B); // G_MAX / 2 * sizeof(T)
            pipe->InitBuffer(pMaxBuff, QMAX_UB_SIZE * PRE_LOAD_NUM_MLA);
        }

        if constexpr (ANTIQUANT) {
            pipe->InitBuffer(qAmaxBuff, QMAX_BUF_SIZE * PRE_LOAD_NUM_MLA);
            qAmaxUb = qAmaxBuff.Get<T>();

            pipe->InitBuffer(antiqScaleBuff, BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_1K); // 576 * sizeof(FLOAT)
            antiqScaleUb = antiqScaleBuff.Get<T>();
        }

        // M_MAX=512, 512 * sizeof(T) * N_Buffer
        if constexpr (AMLA != AMLAMODE::NORMAL) {
            pipe->InitBuffer(nValueBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
            pipe->InitBuffer(cofValueBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
            pipe->InitBuffer(aMlaSumBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
        }

        pipe->InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);

        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_2K);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths)
{
    actualLenQDims = tilingData->baseParams.actualLenQDims;
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, actualLenDims);
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    if (outputLayout == LAYOUT::TND) {
        uint32_t tSize = actualSeqLengthsGmQ.GetValue(batchSize - 1);
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint32_t s1Count = actS1Size;

        for (int s1Idx = 0; s1Idx < s1Count; s1Idx++) {
            uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * gSize * headDim +       //T轴、s1轴偏移
                                      n2Idx * gSize * headDim;                     //N2轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], gSize * headDim, 0);
        }
    } else if (outputLayout == LAYOUT::NTD) {
        uint32_t tSize = actualSeqLengthsGmQ.GetValue(batchSize - 1);
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint32_t s1Count = actS1Size;

        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = n2Idx * gSize * tSize * headDim + gIdx * tSize * headDim +    //N2轴偏移，G轴偏移
                                      tBase * headDim;
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * headDim, 0);
        }
    } else if (outputLayout == LAYOUT::BNSD) {
        uint64_t attenOutOffset = bIdx *kvHeadNum * gSize * qSeqSize * headDim +           //B轴偏移
                                  n2Idx * gSize * qSeqSize * headDim;     //N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], gSize * qSeqSize * headDim, 0);
    } else if (outputLayout == LAYOUT::BSND || outputLayout == LAYOUT::BSH) {
        for (int s1Idx = 0; s1Idx < qSeqSize; s1Idx++) {
            uint64_t attenOutOffset = bIdx * qSeqSize * kvHeadNum * gSize * headDim + s1Idx * kvHeadNum * gSize * headDim +     //B轴、S1轴偏移
                                      n2Idx * gSize * headDim;    //N2轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], gSize * headDim, 0);
        }
    } else if (outputLayout == LAYOUT::NBSD) {
        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = n2Idx * gSize * batchSize * qSeqSize * headDim +        //N2轴偏移
                                      gIdx * batchSize * qSeqSize * headDim + bIdx * qSeqSize * headDim; //G轴、B轴偏移
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], qSeqSize * headDim, 0);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitSoftmaxLseAllInfOutput(uint32_t bIdx,
                                                                                            uint32_t n2Idx)
{
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        uint32_t tokenPrefixSum = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
        uint64_t softmaxLseOffset = tokenPrefixSum * kvHeadNum * gSize + n2Idx * gSize;
        matmul::InitOutput<float>(softmaxLseGm[softmaxLseOffset], gSize * actS1Size, FLOAT_INF);
    } else { // BSH BSND
        uint64_t softmaxLseOffset = bIdx * kvHeadNum * gSize * qSeqSize + n2Idx * gSize * qSeqSize;
        matmul::InitOutput<T>(softmaxLseGm[softmaxLseOffset], gSize * qSeqSize, FLOAT_INF);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitOutputSingleCore()
{
    if (usedCoreNum != 0) {
        uint32_t initOutputEventId = 0U;
        SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        // TND 场景：sparse9 有 s2=0 的 batch，其 Q token 不会被计算路径写入，必须清零
        if constexpr (LAYOUT_T == LAYOUT::TND && ENABLE_TREE) {
            if (actualLenDims > 1U) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                // 按 batch 轮询分配，跳过 s2≠0 的 batch
                for (uint64_t bIdx = (uint64_t)tmpBlockIdx; bIdx < batchSize;
                    bIdx += (uint64_t)usedCoreNum) {
                    uint64_t s2 = actualSeqLengthsGm.GetValue(bIdx);
                    if (s2 != 0) {
                        continue;
                    }
                    uint64_t tBase = (bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
                    uint64_t s1 = actualSeqLengthsGmQ.GetValue(bIdx) - tBase;
                    if (s1 == 0) {
                        continue;
                    }
                    // TND: [T, N, D]，batch 对应的 T 区间连续，一次清零
                    uint64_t outputOffset = tBase * qHeadNum * headDim;
                    uint64_t outputSize = s1 * qHeadNum * headDim;
                    if constexpr (IsSameType<OUT_T, int8_t>::value) {
                        GlobalTensor<half> attentionOutTmpGm;
                        attentionOutTmpGm.SetGlobalBuffer(
                            reinterpret_cast<__gm__ half *>(attentionOutGm.GetPhyAddr(0)));
                        matmul::InitOutput<half>(
                            attentionOutTmpGm[outputOffset / 2], outputSize / 2, 0);
                    } else {
                        matmul::InitOutput<OUT_T>(
                            attentionOutGm[outputOffset], outputSize, 0);
                    }
                }
                SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            }
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        SyncAll();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
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

    if constexpr (BALANCE) {
        actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bIdx);
    } else {
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            actS1Size = (s1Idx == s1Outer - 1) ? s1SizeTail : s1SizeSub;
        } else {
            actS1Size = qSeqSize;
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::GetBN2Gid(const uint32_t bn2gIdx)
{
    if constexpr (FLASH_DECODE) {
        bIdx = aiCoreIdx / (kvHeadNum * splitKVNum);
        // n2Idx = (aiCoreIdx / splitKVNum) % kvHeadNum;
        s2IdxFD = aiCoreIdx % splitKVNum;
    } else if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        uint32_t bs1n2 = beforeBlockSplitBn2Nums + bn2gIdx;
        uint32_t s1n2 = bs1n2 % (kvHeadNum * s1Outer);
        bIdx = bs1n2 / (kvHeadNum * s1Outer);
        s1Idx = s1n2 / kvHeadNum;
    } else {
        uint32_t bn2g = beforeBlockSplitBn2Nums + bn2gIdx;
        uint32_t n2g = bn2g % (kvHeadNum * gOuter);
        bIdx = bn2g / (kvHeadNum * gOuter);
        gIdx = n2g % gOuter;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, n2Idx);
        if (softmaxLseFlag) {
            InitSoftmaxLseAllInfOutput(bIdx, n2Idx);
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::UpdateInnerLoopCond()
{
    if ((curActualSeqLen == 0) || (actS1Size == 0)) {
        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    if constexpr (BALANCE) {
        singleProcessSInnerSizeTail = curActualSeqLen % singleProcessSInnerSize;
        singleProcessSInnerSizeTail = (singleProcessSInnerSizeTail == 0) ? singleProcessSInnerSize : singleProcessSInnerSizeTail;
        sInnerLoopTimes = 0;
    } else {
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
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenPreloadMla<IFAT>::SeqLenFromTensorList(uint32_t bIndex)
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
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const IncreFlashAttentionTilingDataMla *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix)
{
    if ASCEND_IS_AIV {
        subBlockNum = GetSubBlockNum(); // CV1:2场景,返回值为2，其他场景返回值为1
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / subBlockNum;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    // 初始化计算参数
    InitCalcParamsEach();

    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    actualSingleProcessSInnerSize = 0ULL;
    actualSingleProcessSInnerSizeAlign = 0ULL;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    qRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)queryRope);
    kRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)keyRope);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    if constexpr (batchContinuous) {
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

    // sparse9 TND padding 场景初始化 output 为 0
    if constexpr (ENABLE_TREE) {
        if ASCEND_IS_AIV {
            InitOutputSingleCore();
        }
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    if constexpr (AMLA == AMLAMODE::AMLA_3BUF) {
        gMax = 512U;
    }

    if ASCEND_IS_AIV {
        if constexpr (AMLA != AMLAMODE::NORMAL) {
            nValueUb = nValueBuff.Get<T>();
            cofValueUb = cofValueBuff.Get<T>();
            aMlaSumUb = aMlaSumBuff.Get<T>();
        }
        softmaxMaxUb = softmaxMaxBuff.Get<T>();
        softmaxSumUb = softmaxSumBuff.Get<T>();
        softmaxExpUb = softmaxExpBuff.Get<T>();

        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
        if constexpr (QUANT) {
            pMaxUb = pMaxBuff.Get<T>();
            dequantScale1Ub = dequantScale1Buff.Get<T>();
        }
    }

     // workspace 内存排布
     // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm|nUpdateGm
     // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;
    if constexpr (ANTIQUANT) {
        size_t qPreSizeMla = msdIterNum * gSize * (headDim + headDimRope) * qSeqSize;

        // 每个核处理gSize * (headDim + headDimRope) * qSeqSize个Q
        queryPreProcessResGm.SetGlobalBuffer(
            (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * qPreSizeMla * sizeof(KV_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * qPreSizeMla * sizeof(KV_T);
    }

    mm1ResGm.SetGlobalBuffer(
            (__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(MM1_OUT_T);
    if constexpr (ANTIQUANT) {
        vec1ResGm.SetGlobalBuffer(
            (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * vec1ResUbSize * sizeof(KV_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * vec1ResUbSize * sizeof(KV_T);
    } else {
        vec1ResGm.SetGlobalBuffer(
            (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T);
    }

    mm2ResGm.SetGlobalBuffer(
            (__gm__ MM2_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM2_OUT_T);

    if constexpr (ANTIQUANT) {
        vec2ResGm.SetGlobalBuffer(
                (__gm__ UPDATE_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(UPDATE_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(UPDATE_T);
    } else if constexpr (QUANT) {
        vec2ResGm.SetGlobalBuffer(
                (__gm__ UPDATE_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(UPDATE_T)));
        offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(UPDATE_T);
    } else {
        vec2ResGm.SetGlobalBuffer(
                (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(T)));
        offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(T);
    }

    if constexpr (AMLA != AMLAMODE::NORMAL) {
        nUpdateGm.SetGlobalBuffer(
                (__gm__ int32_t *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * gMax * sizeof(int32_t)));
        offset += GetBlockNum() * dbWorkspaceRatio * gMax * sizeof(int32_t);
        softmaxSumGm.SetGlobalBuffer(
                (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * gMax * sizeof(T)));
        offset += GetBlockNum() * dbWorkspaceRatio * gMax * sizeof(T);
    }

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

    if ASCEND_IS_AIV {
        if constexpr (FLASH_DECODE) {
            fdService.InitParams(constInfo);
            fdService.InitGlobalTensor(lseMaxFdGm, lseSumFdGm, accumOutGm, attentionOutGm, actualSeqLengthsGmQ);
        }
    }

    if ASCEND_IS_AIC {
        matmulService.InitParams(qHeadNum, kvHeadNum, headDim, headDimRope, qSeqSize, mmResUbSize, bmm2ResUbSize);
        if constexpr (ANTIQUANT) {
            matmulService.InitMm1GlobalTensor(queryPreProcessResGm, keyGm, kRopeGm, mm1ResGm);
            matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        } else {
            matmulService.InitMm1GlobalTensor(queryGm, qRopeGm, keyGm, kRopeGm, mm1ResGm);
            matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        }

        matmulService.InitPageAttentionInfo(blockTableGm, kvCacheBlockSize, maxBlockNumPerBatch);
        if constexpr (AMLA != AMLAMODE::NORMAL) {
            matmulService.InitNUpdateGlobalTensor(nUpdateGm);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitQuant(
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *workspace)
{
    if constexpr (ANTIQUANT) {
        InitAntiquant(antiquantScale, antiquantOffset, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale,
            valueAntiquantOffset, keyRopeAntiquantScale);
    }
    if constexpr (QUANT) {
        deqScale1Gm.SetGlobalBuffer((__gm__ T *)dequantScaleQuery);
        deqScaleKey = *(__gm__ T *)keyAntiquantScale;
        deqScale2Val = *(__gm__ T *)valueAntiquantScale;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitAntiquant(
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t * keyRopeAntiquantScale)
{
    if ASCEND_IS_AIC {
        return;
    }
    keyRopeAntiquantScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)keyRopeAntiquantScale);
    keyAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)keyAntiquantScale);
    valueAntiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)valueAntiquantScale);
    antiqOffsetExistFlag = (keyAntiquantOffset != nullptr);
    if (antiqOffsetExistFlag) {
        keyAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)keyAntiquantOffset);
        valueAntiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)valueAntiquantOffset);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个increFlashAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间。

    if constexpr (BALANCE) {
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t *coreSidxEnd = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
        const uint32_t *coreBEnd = tilingData->increFlashAttentionCoreParams.coreBEnd;
        const uint32_t *coreS1OuterEnd = tilingData->increFlashAttentionCoreParams.coreS1OuterEnd;
        const uint32_t *coreS2End = tilingData->increFlashAttentionCoreParams.coreS2End;
        const uint32_t *balanceFDCoreBArr = tilingData->tndSplitCoreParams.balanceFDCoreBArr;
        const uint32_t *balanceFDCoreS1Arr = tilingData->tndSplitCoreParams.balanceFDCoreS1Arr;
        const uint32_t *balanceFDCoreKVSplitArr = tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr;
        const uint32_t *balanceFDCoreStartKVSplitNum = tilingData->tndSplitCoreParams.balanceFDCoreStartKVSplitNum;
#else
        uint32_t coreBEnd[ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreBEnd)];
        uint32_t coreS1OuterEnd[ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreS1OuterEnd)];
        uint32_t coreS2End[ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreS2End)];
        uint32_t balanceFDCoreBArr[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreBArr)];
        uint32_t balanceFDCoreS1Arr[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr)];
        uint32_t balanceFDCoreKVSplitArr[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr)];
        uint32_t balanceFDCoreStartKVSplitNum[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreStartKVSplitNum)];
        copy_data_align64((uint8_t *)coreBEnd, (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreBEnd),
                        sizeof(coreBEnd));
        copy_data_align64((uint8_t *)coreS1OuterEnd, (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreS1OuterEnd),
                        sizeof(coreS1OuterEnd));
        copy_data_align64((uint8_t *)coreS2End, (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreS2End),
                        sizeof(coreS2End));
        copy_data_align64((uint8_t *)balanceFDCoreBArr, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreBArr),
                        sizeof(balanceFDCoreBArr));
        copy_data_align64((uint8_t *)balanceFDCoreS1Arr, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr),
                        sizeof(balanceFDCoreS1Arr));
        copy_data_align64((uint8_t *)balanceFDCoreKVSplitArr, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr),
                        sizeof(balanceFDCoreKVSplitArr));
        copy_data_align64((uint8_t *)balanceFDCoreStartKVSplitNum, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreStartKVSplitNum),
                        sizeof(balanceFDCoreStartKVSplitNum));
#endif
        // TND分核信息
        bEnd = coreBEnd[aiCoreIdx];
        s1OuterIdxEnd = coreS1OuterEnd[aiCoreIdx];
        s2End = coreS2End[aiCoreIdx];
        if (aiCoreIdx != 0) {
            bEndPrev = coreBEnd[aiCoreIdx - 1];
            s1OuterIdxEndPrev = coreS1OuterEnd[aiCoreIdx - 1];
            s2EndPrev = coreS2End[aiCoreIdx - 1];
        }
        GetTNDAxisStartId(bStart, n2Start, s1OuterIdxStart, s2Start, bEnd, n2End, s1OuterIdxEnd, s2End);
        tndFDCoreArrLen = tilingData->tndSplitCoreParams.tndFDCoreArrLen;
        if (tmpBlockIdx < tndFDCoreArrLen) {
            // 当前核要规约的[B,N,S1]
            tndFDCoreInfo.bIdx = balanceFDCoreBArr[tmpBlockIdx];
            tndFDCoreInfo.n2Idx = 0;
            tndFDCoreInfo.s1Idx = balanceFDCoreS1Arr[tmpBlockIdx];
            tndFDCoreInfo.kvSplitNum = balanceFDCoreKVSplitArr[tmpBlockIdx];
            for (int i = 0; i < tmpBlockIdx; i++) {
                tndFDCoreInfo.combineTaskPrefixSum += balanceFDCoreKVSplitArr[i];
            }
        }
        coreStartKVSplitPos = balanceFDCoreStartKVSplitNum[aiCoreIdx];
    } else {
        if constexpr (FLASH_DECODE) {
            bn2LoopTimes = 1U;
        } else {
#ifdef ASCENDC_CPU_DEBUG
            const uint32_t *coreSidxEnd = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
#else
            uint32_t coreSidxEnd[ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreSidxEnd)];
            copy_data_align64((uint8_t *)coreSidxEnd,
                              (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreSidxEnd), sizeof(coreSidxEnd));
#endif
            bn2LoopTimes = coreSidxEnd[aiCoreIdx + 1] - coreSidxEnd[aiCoreIdx];
            beforeBlockSplitBn2Nums = coreSidxEnd[aiCoreIdx];
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AttenMaskCopyIn(uint64_t offset,
                                                                                     uint32_t dealRowCount,
                                                                                     uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    attenMaskSizeAlign = Align(actualColumnCount, 32U);
    maskUb.SetSize(dealRowCount * attenMaskSizeAlign);
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
    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AttenMaskCopyIn(const ExtraInfoMla& info)
{
    #define ATTENMASK_STRIDE  2048U
    uint32_t offset;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        int32_t delta = info.s1Idx * s1SizeSub - info.s2Idx * singleProcessSInnerSize + info.s2Size - info.actS1Size; // s1idx = 0
        if (delta < 0) {
            offset = (-delta) < (int32_t)s1SizeSub ? (-delta) : s1SizeSub; // min (-delta, s1Size)
        } else  {
            offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                    ATTENMASK_STRIDE; // min(delta, s2inner)
        }
    } else {
        int32_t delta =
            info.s1Idx * s1SizeSub - info.s2Idx * singleProcessSInnerSize + info.s2Size - qSeqSize; // s1idx = 0
        if (delta < 0) {
            offset = (-delta) < (int32_t)info.s1Size ? (-delta) : info.s1Size; // min (-delta, s1Size)
        } else  {
            offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                    ATTENMASK_STRIDE; // min(delta, s2inner)
        }
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = info.s1Size;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) /32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    DataCopy(maskUb, attenMaskBoolGm[offset], dataCopyParams);

    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyAntiquantScale(LocalTensor<T> &castUb,
    GlobalTensor<Q_T> srcGm, uint64_t offset)
{
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

    LocalTensor<Q_T> inputUb = inputQue2.AllocTensor<Q_T>();
    DataCopyPad(inputUb, srcGm[offset], copyInParams, copyInPadParams);
    inputQue2.template EnQue(inputUb);

    inputUb = inputQue2.DeQue<Q_T>();
    Cast(castUb, inputUb, RoundMode::CAST_NONE, headDim);
    inputQue2.FreeTensor(inputUb);
}

template <typename IFAT>
template <typename RT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::RowsCopyGmToUb(const LocalTensor<RT> &dst, const GlobalTensor<RT> &src,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = (actualColumnCount * sizeof(RT)) / BYTE_BLOCK;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (columnCount - actualColumnCount) * sizeof(RT) / BYTE_BLOCK;
    DataCopy(dst, src, dataCopyParams);
}


template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb,
    LocalTensor<T> tmpAUb, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
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
IncreFlashAttentionAttenPreloadMla<IFAT>::AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<half> &tmpA1,
    LocalTensor<half> &tmpA2, uint32_t calcSize, bool isFirst, uint64_t outOffset)
{
    // Q 被分成2阶，2阶值Q2 = (127 * Qnorm - Q1) *254, 第一次计算将Q1放在tmpA2中，第二次tmpA1-Q1
    if (!isFirst) {
        Sub(tmpA2, tmpA1, tmpA2, calcSize);
        PipeBarrier<PIPE_V>();
        Muls(tmpA2, tmpA2, antiquantExpandCoeff, calcSize);
        PipeBarrier<PIPE_V>();
    }

    // step3: 申请与块int8空间，并将fp16空间 round到int8
    LocalTensor<KV_T> aResOutUbI8 = outputQue1.template AllocTensor<KV_T>();
    Cast(aResOutUbI8, tmpA2, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    // copyOut Ak
    outputQue1.template EnQue(aResOutUbI8);
    outputQue1.template DeQue<KV_T>();
    // 将Q1或Q2拷贝到GM， Q1在前，Q2在后
    DataCopy(dstGm[outOffset], aResOutUbI8, calcSize);
    outputQue1.FreeTensor(aResOutUbI8);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AntiquantMatmulPreProcess(const ExtraInfoMla &info,
    GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = info.gSize * info.s1Size * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    // step1:计算每行的max(abs), tmpAMaxRes=[]
    LocalTensor<T> tmpAMaxRes = aMaxResUb[startRow * BLOCK_ELEMENT_NUM];
    // tmpAMaxRes=RowMax(|srcUb|), tmpAFloorUb 是临时变量，此时还没有值 tmpAFloorUb=[dealRowCount, columnCount]
    AbsRowMax(tmpAMaxRes, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // step2: 127/Qmax
    Duplicate(tmpAFloorUb, antiqCoeff1, dealRowCount * BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    Div(tmpAFloorUb, tmpAFloorUb, tmpAMaxRes, dealRowCount * BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();

	// step3: 计算Qtmp=Q*(127/Qmax)
    RowMuls(srcUb, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // step4：取出Qtmp整数部分
    Cast(tmpAFloorUb, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> tmpAFloorUbFp16 = tmpAFloorUb.template ReinterpretCast<half>();
    tmpAFloorUbFp16.SetSize(tmpAFloorUb.GetSize());
    Cast(tmpAFloorUbFp16, tmpAFloorUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    // step5：将Qtmp转成fp16
    LocalTensor<half> srcUbFp16 = srcUb.template ReinterpretCast<half>();
    srcUbFp16.SetSize(srcUb.GetSize());
    Cast(srcUbFp16, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    // msdIterNum 为Q分解的阶数，当前为2
    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUbFp16, tmpAFloorUbFp16, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}


template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AntiquantSoftmaxResPreProcess(const ExtraInfoMla &info,
    GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = info.gSize * info.s1Size * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    Muls(srcUb, srcUb, antiqCoeff1, calcSize);
    PipeBarrier<PIPE_V>();

    Cast(tmpAFloorUb, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> tmpAFloorUbFp16 = tmpAFloorUb.template ReinterpretCast<half>();
    tmpAFloorUbFp16.SetSize(tmpAFloorUb.GetSize());
    Cast(tmpAFloorUbFp16, tmpAFloorUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    // step5：将Qtmp转成fp16
    LocalTensor<half> srcUbFp16 = srcUb.template ReinterpretCast<half>();
    srcUbFp16.SetSize(srcUb.GetSize());
    Cast(srcUbFp16, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUbFp16, tmpAFloorUbFp16, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealQueryPreProcessBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint64_t qOffset = info.tensorAOffset + (mSizeVStart + startRow) * headDim;
    uint64_t qRopeOffset = info.tensorARopeOffset + (mSizeVStart + startRow) * headDimRope;

    uint32_t bufferId = info.bn2IdxInCurCore % PRE_LOAD_NUM_MLA;  // 0 or 1

    LocalTensor<T> queryUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();

    {
        // 拷贝 Q + QRope 并拼接
        LocalTensor<Q_T> inputUb = inputQue1.AllocTensor<Q_T>();
        RowsCopyGmToUb(inputUb, queryGm[qOffset], dealRowCount, headDimAll, headDim);
        RowsCopyGmToUb(inputUb[headDim], qRopeGm[qRopeOffset], dealRowCount, headDimAll, headDimRope);
        inputQue1.template EnQue(inputUb);
        inputUb = inputQue1.DeQue<Q_T>();
        Cast(queryUb, inputUb, RoundMode::CAST_NONE, dealRowCount * columnCount);
        inputQue1.FreeTensor(inputUb);
    }

    PipeBarrier<PIPE_V>();

    // mul scale
    VecMulMat(queryUb, antiqScaleUb, queryUb, dealRowCount, columnCount, actualColumnCount);

    PipeBarrier<PIPE_V>();

    // A pre process
    size_t qPreSize = qSeqSize * gSize * (headDim + headDimRope) * msdIterNum;
    size_t dstBase = bufferId * qPreSize + mSizeVStart * columnCount;

    LocalTensor<T> aMaxBmm1Ub = qAmaxUb[bufferId * QMAX_BUF_SIZE / sizeof(T)];
    AntiquantMatmulPreProcess(info, queryPreProcessResGm[dstBase], aMaxBmm1Ub, queryUb, aFloorUb, startRow, dealRowCount,
                              columnCount, actualColumnCount);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::QueryPreProcessInner(const ExtraInfoMla &info)
{
    if (!info.isFirstSInnerLoop) {
        return;
    }

    if constexpr (ANTIQUANT_PER_CHANNEL) {
        // 拷贝
        LocalTensor<Q_T> inputUb = inputQue1.AllocTensor<Q_T>();
        RowsCopyGmToUb(inputUb, keyAntiqScaleGm[info.antiqKeyParamOffset], 1, headDimAll, headDim);
        RowsCopyGmToUb(inputUb[headDim], keyRopeAntiquantScaleGm[info.antiqKeyRopeParamOffset], 1, headDimAll, headDimRope);
        inputQue1.template EnQue(inputUb);
        inputUb = inputQue1.DeQue<Q_T>();
        Cast(antiqScaleUb, inputUb, RoundMode::CAST_NONE, headDimAll);
        inputQue1.FreeTensor(inputUb);
    }

    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAll;
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
            DealQueryPreProcessBaseBlock(info, i * mSplitSize, dealSize, headDimAll, headDimAll);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount,
                                                                           uint64_t baseOffset)
{
    LocalTensor<T> lseMax = inputQue1.AllocTensor<T>();
    LocalTensor<T> lseSum = inputQue2.AllocTensor<T>();
    uint64_t combineLoopOffset;

    if constexpr (BALANCE) {
        //  { B,   N2, splitKVNum, S1, G，D }
        combineLseOffset = (baseOffset + startRow) * FP32_ONE_BLOCK_SIZE;
        combineLoopOffset = tndSgBasicSize * FP32_ONE_BLOCK_SIZE;
    } else {
        combineLseOffset = ((bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize + startRow) * FP32_ONE_BLOCK_SIZE;
        combineLoopOffset = qSeqSize * gSize * FP32_ONE_BLOCK_SIZE;
    }
    uint64_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    for (uint32_t i = 0; i < actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset], dealRowCountAlign);  // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset], dealRowCountAlign);
    }
    inputQue2.EnQue(lseSum);
    inputQue1.EnQue(lseMax);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyAccumOutIn(uint32_t splitKVIndex,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = inputQue1.AllocTensor<T>();

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

    if constexpr (BALANCE) {
        combineAccumOutOffset = startRow * headDim +                        // taskoffset + g轴offset
                                splitKVIndex * tndSgBasicSize * headDim;    // 份数offset
    } else {
        combineAccumOutOffset =
            (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim + startRow * headDim;
    }

    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    inputQue1.EnQue(accumOutLocal);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax,
                                                                uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMaxUb = softmaxMaxUb[0];
    LocalTensor<T> lseSumUb = softmaxSumUb[0];
    LocalTensor<T> lseExpUb = tmpBuff1.Get<T>();

    // lseLocal的shape为[actualCombineLoopSize, dealRowCount * FP32_ONE_BLOCK_SIZE]
    uint32_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    Duplicate(lseMaxUb, -FLOAT_MAX, dealRowCountAlign);
    Duplicate(lseSumUb, FLOAT_ZERO, dealRowCountAlign);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Max(lseMaxUb, lseMaxUb, lseMax[i * dealRowCountAlign], dealRowCountAlign);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Sub(lseExpUb[i * dealRowCountAlign], lseMax[i * dealRowCountAlign], lseMaxUb, dealRowCountAlign);
    }
    PipeBarrier<PIPE_V>();
    Exp(lseExpUb, lseExpUb, actualCombineLoopSize * dealRowCountAlign);
    PipeBarrier<PIPE_V>();

    Mul(lseSum, lseSum, lseExpUb, actualCombineLoopSize * dealRowCountAlign);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Add(lseSumUb, lseSumUb, lseSum[i * dealRowCountAlign], dealRowCountAlign);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Div(lseSum[i * dealRowCountAlign], lseSum[i * dealRowCountAlign], lseSumUb, dealRowCountAlign);
    }
    PipeBarrier<PIPE_V>();

    if (softmaxLseFlag) {
        LocalTensor<T> softmaxlseUb = outputQue2.template AllocTensor<T>();
        ComputeSoftmaxLse(softmaxlseUb, lseSumUb, lseMaxUb, dealRowCountAlign);

        outputQue2.EnQue(softmaxlseUb);
        outputQue2.DeQue<T>();

        uint64_t softmaxLseOutOffset = attenOutOffset / headDim + startRow;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = sizeof(T);
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[softmaxLseOutOffset], softmaxlseUb, dataCopyParams);        
        outputQue2.FreeTensor(softmaxlseUb);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                                             uint32_t startRow, uint32_t dealRowCount,
                                                             uint64_t baseOffset)
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
    CopyAccumOutIn(0, baseOffset + startRow, dealRowCount);
    LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
    for (int i = 0; i < mulLoop; i++) {
        Mul(dst[i * dtypeMask], lseLocal, accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
    }
    if (mulRemain > 0) {
        Mul(dst[mulLoop * dtypeMask], lseLocal, accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount,
            repeatParams);
    }
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(accumOutLocal);

    uint32_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    for (uint32_t j = 1; j < actualCombineLoopSize; ++j) {
        CopyAccumOutIn(j, baseOffset + startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
        for (int i = 0; i < mulLoop; i++) {
            Mul(accumOutLocal[i * dtypeMask], lseLocal[j * dealRowCountAlign],
                accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(accumOutLocal[mulLoop * dtypeMask], lseLocal[j * dealRowCountAlign],
                accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount, repeatParams);
        }
        PipeBarrier<PIPE_V>();
        Add(dst, dst, accumOutLocal, dealRowCount * headDimAlign);
        PipeBarrier<PIPE_V>();
        inputQue1.FreeTensor(accumOutLocal);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * headDimAlign);
    } else {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * headDimAlign);
    }

    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    if (outputLayout == LAYOUT::NBSD || outputLayout == LAYOUT::NTD) {
        TransposeInfo transInfo;
        transInfo.n2Idx = n2Idx;
        transInfo.bIdx = bIdx;
        transInfo.gSize = gSize;
        transInfo.s1Size = qSeqSize;
        if constexpr (BALANCE) {
            transInfo.s1StartIdx = s1Idx;
            transInfo.s1Count = 1; // 当前TND一次只处理一个S1
            transInfo.gStartIdx = startRow;
            transInfo.gCount = dealRowCount;
            Bmm2DataCopyOutNBSDMTiling(tmpBmm2ResCastTensor, transInfo);
        } else { // 当前只支持非TND格式的S1=1
            transInfo.gStartIdx = startRow;
            transInfo.gCount = dealRowCount;
            transInfo.s1StartIdx = 0;
            transInfo.s1EndIdx = 0;
            Bmm2DataCopyOutNBSDGTiling(tmpBmm2ResCastTensor, transInfo);
        }
    } else {
        Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, headDimAlign, headDim);
    }
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CombineSplitKVRes(
    uint64_t baseOffset)
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
            CopyLseIn(startRow, actualGSplitSize, baseOffset);
            LocalTensor<T> lseSum = inputQue2.DeQue<T>();
            LocalTensor<T> lseMax = inputQue1.DeQue<T>();
            ComputeScaleValue(lseSum, lseMax, startRow, actualGSplitSize);
            inputQue1.FreeTensor(lseMax);

            uint32_t gSplitBmm2UbSize = headDimAlign * actualGSplitSize;
            LocalTensor<T> tmp1 = tmpBuff1.Get<T>(gSplitBmm2UbSize);
            ReduceFinalRes(tmp1, lseSum, startRow, actualGSplitSize, baseOffset);
            inputQue2.FreeTensor(lseSum);

            CopyFinalResOut(tmp1, startRow, actualGSplitSize);
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::FlashDecodeComputeND()
{
    mSizeVStart = 0;
    // 根据核id从数组把bns1Idx和份数拿出来
    bIdx = tndFDCoreInfo.bIdx;
    s1Idx = tndFDCoreInfo.s1Idx;
    actualCombineLoopSize = tndFDCoreInfo.kvSplitNum;

    uint32_t actualSeqQ = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bIdx);
    curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    int sOuterLoopTailIdx = ((actualSeqQ + s1SizeSub - 1) / s1SizeSub) - 1;
    int s1RowSize = s1SizeSub;
    if (s1Idx == sOuterLoopTailIdx) {   // 尾块场景
        s1RowSize = (actualSeqQ % s1SizeSub == 0) ? s1SizeSub : (actualSeqQ % s1SizeSub);
    }

    s1Idx = s1Idx * s1SizeSub;  // 这里s1Idx 指行号
    uint64_t s1IdxStart = s1Idx;
    uint32_t s1IdxEnd = s1Idx + s1RowSize;
    uint64_t taskOffset = tndFDCoreInfo.combineTaskPrefixSum * kvHeadNum * tndSgBasicSize;

    // 每次循环完成一次GD块规约
    for (; s1Idx < s1IdxEnd; s1Idx++) {
        uint64_t baseOffset = 0;
        if constexpr (BALANCE) {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == LAYOUT::TND) {
                actualSeqQPrefixSum = (bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
            } else { // BSND
                actualSeqQPrefixSum = (bIdx <= 0) ? 0 : bIdx * qSeqSize;
            }
            uint64_t batchOutOffset = actualSeqQPrefixSum * qHeadNum * headDim; // 累积计算当前Batch对应的 offset
            baseOffset = taskOffset + (s1Idx - s1IdxStart) * kvHeadNum * gSize;
            attenOutOffset = batchOutOffset + s1Idx * (kvHeadNum * gSize * headDim) + n2Idx * (gSize * headDim);
        } else {
            // { B, S1, N2, G，D} BS1N2分核 GD 规约
            attenOutOffset = bIdx * qSeqSize * kvHeadNum * gSize * headDim + s1Idx * (kvHeadNum * gSize * headDim) +
                             n2Idx * (gSize * headDim);
        }
        CombineSplitKVRes(baseOffset);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::FlashDecodeCompute()
{
    if constexpr (BALANCE) {
        if (tmpBlockIdx >= tndFDCoreArrLen) {
            return;
        }
        FlashDecodeComputeND();
    } else {
        bIdx = tmpBlockIdx / kvHeadNum;
        attenOutOffset = bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
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
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeLogSumExpAndCopyToGm(const ExtraInfoMla &info, LocalTensor<T> &softmaxSumUb,
                                                                          LocalTensor<T> &softmaxMaxUb)
{
    if constexpr (BALANCE) {
        // workspace同步修改
        //  src-Shape  { gsizeV, S1, FP32_ONE_BLOCK_SIZE }
        //  dst-Shape  { B  N2, splitKV s1, G, FP32_ONE_BLOCK_SIZE}
        // 这里的offset计算，后续FD切G改切M时，同步改掉
        s2IdxFD = info.tndCoreStartKVSplitPos;
        size_t size = mSizeVector * FP32_ONE_BLOCK_SIZE;
        uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.s1Idx);
        uint64_t offset = (accumTmpOutNum * kvHeadNum * tndSgBasicSize +     // taskoffset
                          s2IdxFD * kvHeadNum * tndSgBasicSize +            // 份数offset
                          mSizeVStart) * FP32_ONE_BLOCK_SIZE;                                      // m轴offset

        LocalTensor<T> tmp = outputQue2.AllocTensor<T>();

#ifdef IFA_SOFTMAX_WITHOUT_BRC
        Brcb(tmp, softmaxSumUb, (mSizeVector + 7) / 8, {1, 8});
#else
        DataCopy(tmp, softmaxSumUb, size);
#endif
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseSumFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);

        tmp = outputQue2.AllocTensor<T>();
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        Brcb(tmp, softmaxMaxUb, (mSizeVector + 7) / 8, {1, 8});
#else
        DataCopy(tmp, softmaxMaxUb, size);
#endif
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseMaxFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
    } else {
        size_t size = mSizeVector * FP32_ONE_BLOCK_SIZE;
        size_t offset = (info.bIdx * kvHeadNum * splitKVNum * gSize + info.n2Idx * splitKVNum * gSize +
                        s2IdxFD * gSize + mSizeVStart) * FP32_ONE_BLOCK_SIZE;
        // lseSumFdGm: batchQ * kvHeadNum * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE
        CopyFixedUbToGm(lseSumFdGm[offset], softmaxSumUb, size);

        CopyFixedUbToGm(lseMaxFdGm[offset], softmaxMaxUb, size);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::AttenMaskCopyForSplitG(const ExtraInfoMla &info, LocalTensor<bool> &attenMaskUb, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t s1StartIdx = (mSizeVStart + startRow) / info.gSize;
    uint32_t s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;

    #define ATTENMASK_STRIDE  2048U

    uint32_t offset;
    uint32_t actualSeqQ = qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        actualSeqQ = info.actS1Size;
    }
    int32_t delta = (info.s1Idx * s1SizeSub + s1StartIdx) - info.s2Idx * singleProcessSInnerSize +
                    (info.s2Size - actualSeqQ);
    if (delta < 0) {
        offset = (-delta) < (int32_t)s1Count ? (-delta) : s1Count;
    } else  {
        offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                ATTENMASK_STRIDE;
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Count;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) /32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    attenMaskUb = inputQue2.AllocTensor<bool>();
    DataCopy(attenMaskUb, attenMaskBoolGm[offset], dataCopyParams);

    inputQue2.template EnQue(attenMaskUb);

    attenMaskUb = inputQue2.DeQue<bool>();
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
    uint32_t tailGSize = reminRowCount % info.gSize;
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

// 新增sparse9的处理
template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::AttenMaskCopyNoFull(LocalTensor<bool> &attenMaskUb, const ExtraInfoMla &info, uint32_t s1StartIdx, uint32_t s1EndIdx)
{
    uint32_t actualSeqQ = qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        actualSeqQ = info.actS1Size;
    }

    uint32_t treeMaskStart = info.s2Size - actualSeqQ;
    uint32_t curS2EndPos = info.s2Idx * singleProcessSInnerSize + info.actualSingleProcessSInnerSize;
    uint32_t curS2StartPos = info.s2Idx * singleProcessSInnerSize;

    uint64_t attenMaskBatchStride = actualSeqQ * actualSeqQ;
    uint64_t attenMaskStride = actualSeqQ;

    uint64_t bOffset;
    uint64_t s1Offset;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        // TND格式下mask是一维张量∑s1²，batch偏移是前面所有batch的s1²累积和
        uint64_t batchOffset = 0;
        for (uint32_t i = 0; i < info.bIdx; i++) {
            uint64_t s1i = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, i);
            batchOffset += s1i * s1i;
        }
        bOffset = batchOffset;
        // TND格式下s1StartIdx是当前S1块内的局部偏移，完整局部S1索引需加上s1Idx*s1SizeSub
        s1Offset = (static_cast<uint64_t>(info.s1Idx) * s1SizeSub + s1StartIdx) * attenMaskStride;
    } else {
        bOffset = static_cast<uint64_t>(info.bIdx) * static_cast<uint64_t>(attenMaskBatchStride);
        s1Offset = ((info.s1Idx * s1SizeSub + s1StartIdx) % actualSeqQ) * attenMaskStride;
    }
    uint64_t s2Offset = curS2StartPos > treeMaskStart ? curS2StartPos - treeMaskStart : 0;

    uint64_t maskOffset = bOffset + s1Offset + s2Offset;

    if (curS2StartPos < treeMaskStart) { //mask占不满UB空间
        // 不对齐场景下的搬运
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
        dataCopyParams.blockLen = curS2EndPos - treeMaskStart;
        dataCopyParams.srcStride = attenMaskStride - (curS2EndPos - treeMaskStart);
        dataCopyParams.dstStride = (treeMaskStart - curS2StartPos) / 32;

        DataCopyPadExtParams<bool> padParams;
        padParams.isPad = true;
        padParams.leftPadding = static_cast<uint8_t>(treeMaskStart % 32);
        padParams.rightPadding = static_cast<uint8_t>(Align(static_cast<uint32_t>(curS2EndPos - treeMaskStart + treeMaskStart % 32), 32U) - \
                                (curS2EndPos - treeMaskStart + treeMaskStart % 32));
        padParams.paddingValue = 0;
        DataCopyPad(attenMaskUb[(treeMaskStart - curS2StartPos) / 32 * 32], attenMaskBoolGm[maskOffset], dataCopyParams, padParams);
    } else { // mask 占满UB空间
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
        dataCopyParams.blockLen = info.actualSingleProcessSInnerSize;
        dataCopyParams.srcStride = attenMaskStride - info.actualSingleProcessSInnerSize;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<bool> padParams{true, 0, static_cast<uint8_t>(info.actualSingleProcessSInnerSizeAlign - info.actualSingleProcessSInnerSize), 0};
        DataCopyPad(attenMaskUb, attenMaskBoolGm[maskOffset], dataCopyParams, padParams);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AttenMaskCopyForTree(const ExtraInfoMla &info,
                                        LocalTensor<bool> &attenMaskUb, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t s1StartIdx = (mSizeVStart + startRow) / info.gSize;
    uint32_t s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;

    uint32_t actualSeqQ = qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        actualSeqQ = info.actS1Size;
    }


    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);
    // 第一步，将mask的 ub空间赋值为0
    // 申请的16k空间，前8k给从GM拷贝到Ub时使用，后8k用来在拓展G轴时使用
    attenMaskUb = inputQue2.AllocTensor<bool>();
    LocalTensor<int16_t> mask16_0 = attenMaskUb.template ReinterpretCast<int16_t>();
    AscendC::Duplicate(mask16_0, static_cast<int16_t>(0), s1Count * attenMaskSizeAlign / sizeof(int16_t));
    attenMaskUb = mask16_0.template ReinterpretCast<bool>();
    // V→MTE2 跨流水线同步：确保 V pipeline 的 Duplicate 完成后，MTE2 的 DataCopyPad 才能写入
    event_t eventIdVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVMte2);

    // 第二步，计算偏移 从Gm上把mask拷过来
    AttenMaskCopyNoFull(attenMaskUb, info, s1StartIdx, s1EndIdx + 1);

    // 第三步，添加拷贝同步信号
    inputQue2.template EnQue(attenMaskUb);
    attenMaskUb = inputQue2.DeQue<bool>();

    LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> attenMaskUbDst = mask16[BUFFER_SIZE_BYTE_16K / 4];

    uint32_t headGCount = 0;
    uint32_t firstGIdx = (mSizeVStart + startRow) % info.gSize;
    if (s1Count > 1) {
        headGCount = info.gSize - firstGIdx;
    } else {
        headGCount = dealRowCount;
    }
    uint64_t dstMaskOffset = 0;
    uint64_t srcMaskBaseOffset = 0;
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
    uint32_t tailGSize = reminRowCount % info.gSize;
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

template <typename IFAT>
__aicore__ inline bool
IncreFlashAttentionAttenPreloadMla<IFAT>::IsSkipAttenMask(const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t actualSeqQ = qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        actualSeqQ = info.actS1Size;
    }

    uint32_t s2EndPos = info.s2Idx * singleProcessSInnerSize + info.actualSingleProcessSInnerSize;
    // 新增sparse9的处理，sparse9和sparse3类似，但是sparse9是用户自己传入的mask矩阵，算子无法感知具体数值，所以涉及到基本块都要参与计算
    if constexpr (ENABLE_TREE) {
        if (s2EndPos > (info.s2Size - actualSeqQ)) {
            return false;
        } else {
            return true;
        }
    }

    // s2<s1时，必然走mask
    if (info.s2Size < actualSeqQ) {
        return false;
    }

    // 当前的s2位置不超过需要打标记的位置时，不需要mask
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
IncreFlashAttentionAttenPreloadMla<IFAT>::ElewiseCompute(const ExtraInfoMla &info, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    if constexpr (!ANTIQUANT && !QUANT) {
        // 由于伪量化方案中scaleValue已经在MM1的fixpipe中乘过了，故此处不再需要; 全量化也在V1开始计算过了。
        Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
    }
    // attenMask
    if (attenMaskFlag == 1) {
        LocalTensor<bool> attenMaskUb;
        if ((qSeqSize == 1) && (LAYOUT_T != LAYOUT::TND)) {
            AttenMaskCopyIn(info.attenMaskOffset, dealRowCount, actualColumnCount);
            attenMaskUb = inputQue2.DeQue<bool>();
            for (int i = 1; i < dealRowCount; i++) {
                DataCopy(attenMaskUb[i * attenMaskSizeAlign], attenMaskUb, attenMaskSizeAlign);
            }
        } else {
            if (IsSkipAttenMask(info, startRow, dealRowCount)) {
                return;
            }
            if constexpr (LAYOUT_T == LAYOUT::BNSD) {
                AttenMaskCopyIn(info);
                attenMaskUb = inputQue2.DeQue<bool>();
                LocalTensor<bool> attenMaskUbDst = attenMaskUb[BUFFER_SIZE_BYTE_16K / 2];

                uint32_t headS1Count = 0;
                uint32_t headS1Start = (mSizeVStart + startRow) % info.s1Size;
                if (headS1Start + dealRowCount > info.s1Size) {
                    headS1Count = info.s1Size - headS1Start;
                } else {
                    headS1Count = dealRowCount;
                }

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
                if constexpr (ENABLE_TREE) {
                    AttenMaskCopyForTree(info, attenMaskUb, startRow, dealRowCount);
                } else {
                    AttenMaskCopyForSplitG(info, attenMaskUb, startRow, dealRowCount);
                }
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
        inputQue2.FreeTensor(attenMaskUb);

        PipeBarrier<PIPE_V>();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::SoftmaxFlashV2Compute(const ExtraInfoMla &info,
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
#ifdef IFA_SOFTMAX_WITHOUT_BRC
    uint32_t baseOffset = startRow;
#else
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
#endif
    uint32_t outIdx = info.loop % (PRE_LOAD_NUM_MLA);
    uint32_t softmaxOutOffset = outIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (PRE_LOAD_NUM_MLA);
        inMaxTensor = softmaxMaxUb[inIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset];
    }

#ifdef SOFTMAX_NEW
    LocalTensor<T> tmpRowMaxUb = tmpBuff3.Get<T>();
    #ifdef IFA_SOFTMAX_WITHOUT_BRC
        SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, tmpRowMaxUb, softmaxSumUb[softmaxOutOffset],
            softmaxMaxUb[softmaxOutOffset], mmResUb, softmaxExpUb[softmaxOutOffset],
            inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    #else
        SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
            mmResUb, tmpRowMaxUb, softmaxSumUb[softmaxOutOffset],
            softmaxMaxUb[softmaxOutOffset], mmResUb, softmaxExpUb[softmaxOutOffset],
            inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    #endif
 
    if constexpr (QUANT) {
        PipeBarrier<PIPE_V>();
    #ifdef IFA_SOFTMAX_WITHOUT_BRC
        uint32_t calQuantCount = dealRowCount;
    #else
        uint32_t calQuantCount = dealRowCount * BLOCK_ELEMENT_NUM;
    #endif
        Sub(tmpRowMaxUb, tmpRowMaxUb, softmaxMaxUb[softmaxOutOffset], calQuantCount);
        PipeBarrier<PIPE_V>();
        Exp(tmpRowMaxUb, tmpRowMaxUb, calQuantCount);
        PipeBarrier<PIPE_V>();
        Adds(tmpRowMaxUb, tmpRowMaxUb, FLOAT_EPS, calQuantCount);
        PipeBarrier<PIPE_V>();
    }
#else
    #ifdef IFA_SOFTMAX_WITHOUT_BRC
        SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset],
            softmaxMaxUb[softmaxOutOffset], mmResUb, softmaxExpUb[softmaxOutOffset],
            inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    #else
        SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
            mmResUb, softmaxSumUb[softmaxOutOffset],
            softmaxMaxUb[softmaxOutOffset], mmResUb, softmaxExpUb[softmaxOutOffset],
            inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    #endif
#endif

    if constexpr (AMLA != AMLAMODE::NORMAL) {
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        uint32_t calCount = dealRowCount;
#else
        uint32_t calCount = dealRowCount * BLOCK_ELEMENT_NUM;
#endif
        // compute n(i)
        LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
        LocalTensor<T> nUpdateTmp = nTmp[BUFFER_SIZE_BYTE_2K / sizeof(T)];
        PipeBarrier<PIPE_V>();
        Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);
        PipeBarrier<PIPE_V>();
        Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
        PipeBarrier<PIPE_V>();

        uint32_t prOutIdx = (info.loop - 1) % (PRE_LOAD_NUM_MLA);
        uint32_t PreSoftmaxOutOffset = prOutIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset;

        // n(i) - n(i-1)
        if (info.isFirstSInnerLoop) {
            Duplicate(nUpdateTmp, FLOAT_ZERO, calCount); // n1=n0
        } else {
            Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
        }
        PipeBarrier<PIPE_V>();
        // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
        Adds(nValueUb[softmaxOutOffset], nTmp, FLOAT_ZERO, calCount);
        PipeBarrier<PIPE_V>();

        // update softmax res
        LocalTensor<T> nUpdateTmp2 = nTmp[2 * BUFFER_SIZE_BYTE_2K / sizeof(T)];
        LocalTensor<KV_T> nTmp_KvT = nTmp[3 * BUFFER_SIZE_BYTE_2K / sizeof(T)].template ReinterpretCast<KV_T>();
        LocalTensor<T> tmpCofUb = nTmp[4 * BUFFER_SIZE_BYTE_2K / sizeof(T)];
        LocalTensor<T> epsUb = nTmp[5 * BUFFER_SIZE_BYTE_2K / sizeof(T)];
        Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
        PipeBarrier<PIPE_V>();
        Add(nTmp, nUpdateTmp2, nTmp, calCount);
        PipeBarrier<PIPE_V>();
        Muls(nTmp, nTmp, LN2, calCount);
        PipeBarrier<PIPE_V>();
        Exp(nTmp, nTmp, calCount);
        PipeBarrier<PIPE_V>();
        Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount); // fp32->fp16/bf16
        PipeBarrier<PIPE_V>();
        Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
        PipeBarrier<PIPE_V>();
        if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
            Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
        }
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        LocalTensor<T> nTmp3 = nTmp[6 * BUFFER_SIZE_BYTE_2K / sizeof(T)];
        Brcb(nTmp3, nUpdateTmp2, (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(mmResUb, mmResUb, nTmp3, dealRowCount, columnCount, actualColumnCount);
#else
        RowMuls(mmResUb, mmResUb, nUpdateTmp2, dealRowCount, columnCount, actualColumnCount);
#endif

        Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
        if (info.isFirstSInnerLoop) {
            Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount); // cof_0=1
            PipeBarrier<PIPE_V>();
            Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount); // 1 / cof(i)
        } else {
            PipeBarrier<PIPE_V>();
            Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
        }
        PipeBarrier<PIPE_V>();

        Adds(cofValueUb[softmaxOutOffset], tmpCofUb, FLOAT_ZERO, calCount); // store cof(i)
        Adds(epsUb, epsUb, (T)(-1.0), calCount); // cof(i - 1) / cof(i) - 1
        PipeBarrier<PIPE_V>();
        Muls(epsUb, epsUb, (T)1.5, calCount); // (cof(i - 1) - cof(i)) / cof(i) * 1.5

        Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
        PipeBarrier<PIPE_V>();
        Adds(epsUb, epsUb, (T)(0.000001), calCount);
        PipeBarrier<PIPE_V>();
        Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
        PipeBarrier<PIPE_V>();
        Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
        PipeBarrier<PIPE_V>();

        // nUpdate int32 out
        LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();

#ifndef IFA_SOFTMAX_WITHOUT_BRC
        constexpr int32_t srcRepStride = FP32_MAX_MASK_ELEMENT_NUM / BLOCK_ELEMENT_NUM; // =8
        BlockReduceMax(nUpdateTmp, nUpdateTmp, (dealRowCount + 7) / 8, FP32_MAX_MASK_ELEMENT_NUM, 1, 1, srcRepStride);
        PipeBarrier<PIPE_V>();
#endif
        Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
        PipeBarrier<PIPE_V>();
        outputQue2.EnQue(nInt32Out);
        outputQue2.DeQue<int32_t>();
        if ((dealRowCount % 8) == 0) { // 32B对齐
            DataCopy(nUpdateGm[outIdx * gMax + mSizeVStart + startRow], nInt32Out, dealRowCount);
        } else {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = dealRowCount * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(nUpdateGm[outIdx * gMax + mSizeVStart + startRow], nInt32Out, dataCopyParams);
        }
        outputQue2.FreeTensor(nInt32Out);
    } else {
        if (info.isBmm2Output) {
            PipeBarrier<PIPE_V>();
#ifdef IFA_SOFTMAX_WITHOUT_BRC
            LocalTensor<T> tmpSoftmaxSumUb = softmaxTmpUb.template ReinterpretCast<T>();
            Brcb(tmpSoftmaxSumUb, softmaxSumUb[softmaxOutOffset], (dealRowCount + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            RowDivs(mmResUb, mmResUb, tmpSoftmaxSumUb, dealRowCount,
                    columnCount, actualColumnCount);
#else
            RowDivs(mmResUb, mmResUb, softmaxSumUb[softmaxOutOffset], dealRowCount,
                    columnCount, actualColumnCount);
#endif
        }
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2FDDataCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                                uint32_t dealRowCount, uint32_t columnCount,
                                                                uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    if constexpr (BALANCE) {
        LocalTensor<T> tmp = outputQue1.AllocTensor<T>();
        DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
        outputQue1.EnQue(tmp);
        outputQue1.DeQue<T>();
        s2IdxFD = info.tndCoreStartKVSplitPos;

        uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.s1Idx);
        uint64_t offset = accumTmpOutNum * kvHeadNum * tndSgBasicSize * headDim +               // taskoffset
                          info.tndCoreStartKVSplitPos * kvHeadNum * tndSgBasicSize * headDim +  // 份数offset
                          (mSizeVStart + startRow) * actualColumnCount;                         // m轴offset
        GlobalTensor<T> dst = accumOutGm[offset];
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
        outputQue1.FreeTensor(tmp);
    } else {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        LocalTensor<T> tmp = outputQue1.AllocTensor<T>();
        DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
        outputQue1.EnQue(tmp);
        outputQue1.DeQue<T>();

        size_t base = (info.bIdx * qHeadNum + info.n2Idx * gSize) * splitKVNum * headDim;
        // accumOutGm: batchQ * kvHeadNum * gSize * kvSplitPart_ * headDimAlign_
        DataCopyPad(accumOutGm[base + s2IdxFD * gSize * actualColumnCount +
                               startRow * actualColumnCount + mSizeVStart * actualColumnCount], tmp, dataCopyParams);
        outputQue1.FreeTensor(tmp);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
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
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2DataCopyOutNBSDGTiling(LocalTensor<OUT_T> &attenOutUb,
    const TransposeInfo& transInfo)
{
    bool hasHeadBlock = transInfo.s1StartIdx != 0;
    bool hasTailBlock = (transInfo.s1EndIdx + 1) != transInfo.s1Size;
    uint32_t attenOutUbOffset = 0;
    if (hasHeadBlock) { // 头块单独一条DataCopy指令
        DataCopyParams dataCopyParamsHead;
        dataCopyParamsHead.blockCount = 1;
        dataCopyParamsHead.blockLen = (transInfo.s1Size - transInfo.s1StartIdx) * headDim * sizeof(OUT_T) / 32U;
        dataCopyParamsHead.srcStride = 0;
        dataCopyParamsHead.dstStride = 0; // blockCount = 1 无所谓跳写
        uint64_t  attenOutOffset = transInfo.n2Idx * gSize * batchSize * qSeqSize * headDim + // N2轴的偏移
                                   transInfo.gStartIdx * batchSize * qSeqSize * headDim +     // G轴的偏移
                                   transInfo.bIdx * qSeqSize * headDim +                      // B轴的偏移
                                   transInfo.s1StartIdx * headDim;                            // S1轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParamsHead);
        attenOutUbOffset += (transInfo.s1Size - transInfo.s1StartIdx) * headDim;
    }
    // 中间块DataCopy指令
    uint64_t  attenOutOffset = transInfo.n2Idx * gSize * batchSize * qSeqSize * headDim +                                     // N2轴的偏移
                               (transInfo.gStartIdx + static_cast<uint32_t>(hasHeadBlock)) * batchSize * qSeqSize * headDim + // G轴的偏移
                               transInfo.bIdx * qSeqSize * headDim;                                                           // B轴的偏移
    bool dstStrideFlag = ((batchSize * qSeqSize - transInfo.s1Size) * headDim * sizeof(OUT_T) / 32U) > UINT16_MAX ? 1 : 0;
    if (dstStrideFlag) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = transInfo.s1Size * headDim * sizeof(OUT_T); // 一个S1*D的大小
        dataCopyParams.srcStride = 0;                                                                         // 连读
        dataCopyParams.dstStride = (batchSize * qSeqSize - transInfo.s1Size) * headDim * sizeof(OUT_T); // 跳写
        DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (transInfo.s1Size * headDim);
    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = transInfo.gCount - static_cast<uint32_t>(hasHeadBlock) - static_cast<uint32_t>(hasTailBlock); // 处理多少个G
        dataCopyParams.blockLen = transInfo.s1Size * headDim * sizeof(OUT_T) / 32U; // 一个S1*D的大小
        dataCopyParams.srcStride = 0;                                                                         // 连读
        dataCopyParams.dstStride = (batchSize * qSeqSize - transInfo.s1Size) * headDim * sizeof(OUT_T) / 32U; // 跳写
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        attenOutUbOffset += dataCopyParams.blockCount * (transInfo.s1Size * headDim);
    }
    if (hasTailBlock) { // 尾块单独一条DataCopy指令
        DataCopyParams dataCopyParamsTail;
        dataCopyParamsTail.blockCount = 1;
        dataCopyParamsTail.blockLen = (transInfo.s1EndIdx + 1) * headDim * sizeof(OUT_T) / 32U;
        dataCopyParamsTail.srcStride = 0;
        dataCopyParamsTail.dstStride = 0; // blockCount = 1 无所谓跳写
        uint64_t  attenOutOffset = transInfo.n2Idx * gSize * batchSize * qSeqSize * headDim +                      // N2轴的偏移
                                   (transInfo.gStartIdx + transInfo.gCount - 1) * batchSize * qSeqSize * headDim + // G轴的偏移
                                   transInfo.bIdx * qSeqSize * headDim;                                            // B轴的偏移
        DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParamsTail);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2DataCopyOutNBSDMTiling(LocalTensor<OUT_T> &attenOutUb,
    const TransposeInfo& transInfo)
{
    uint32_t tSize = batchSize * qSeqSize;
    uint32_t tBase = transInfo.bIdx * qSeqSize;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        tSize = actualSeqLengthsGmQ.GetValue(batchSize - 1);
        tBase = transInfo.bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(transInfo.bIdx - 1);
    }

    uint32_t s1Idx = transInfo.s1StartIdx;
    uint32_t attenOutUbOffset = 0;
    for (int i = 0; i < transInfo.s1Count; i++) {
        uint32_t gIdx = 0;                       // 中间块
        uint32_t gCountOneS1 = transInfo.gSize;
        if (i == 0) {                            // 首块
            gIdx = transInfo.gStartIdx;
            gCountOneS1 = (transInfo.gSize - transInfo.gStartIdx) < transInfo.gCount ?
                (transInfo.gSize - transInfo.gStartIdx) : transInfo.gCount; // min(info.gSize - gStartIdx, gCount);
        } else if (i == transInfo.s1Count - 1) { // 尾块
            gIdx = 0;
            gCountOneS1 = transInfo.gEndIdx + 1;
        }
        uint64_t  attenOutOffset = transInfo.n2Idx * gSize * tSize * headDim +  // N2轴的偏移
                                   gIdx * tSize * headDim +                     // G轴的偏移
                                   tBase * headDim +                            // B轴的偏移
                                   s1Idx * headDim;                             // S1轴的偏移
        bool dstStrideFlag = ((batchSize * qSeqSize - transInfo.s1Size) * headDim * sizeof(OUT_T) / 32U) > UINT16_MAX ? 1 : 0;
        if (dstStrideFlag) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = headDim * sizeof(OUT_T); // 一个D的大小
            dataCopyParams.srcStride = 0;                                           // 连读
            dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T); // 跳写
            DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = gCountOneS1;
            dataCopyParams.blockLen = headDim * sizeof(OUT_T) / 32U; // 一个D的大小
            dataCopyParams.srcStride = 0;                                           // 连读
            dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T) / 32U; // 跳写
            DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
        }
        s1Idx++;
        attenOutUbOffset += gCountOneS1 * headDim;
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2DataCopyOutTrans(const ExtraInfoMla &info, LocalTensor<OUT_T> &attenOutUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    if (outputLayout == LAYOUT::NBSD || outputLayout == LAYOUT::NTD) {
        TransposeInfo transInfo;
        transInfo.n2Idx = info.n2Idx;
        transInfo.bIdx = info.bIdx;
        transInfo.gSize = info.gSize;
        transInfo.s1Size = info.s1Size;
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            transInfo.gStartIdx = info.gIdx * gSizeSub + (mSizeVStart + startRow) / info.s1Size;
            uint32_t gEndIdx = info.gIdx * gSizeSub + (mSizeVStart + startRow + dealRowCount - 1) / info.s1Size;
            transInfo.gCount = gEndIdx - transInfo.gStartIdx + 1;
            transInfo.s1StartIdx = (mSizeVStart + startRow) % info.s1Size;
            transInfo.s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) % info.s1Size;
            Bmm2DataCopyOutNBSDGTiling(attenOutUb, transInfo);
        } else {
            transInfo.s1StartIdx = info.s1Idx * s1SizeSub + (mSizeVStart + startRow) / info.gSize;
            uint32_t s1EndIdx = info.s1Idx * s1SizeSub + (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
            transInfo.s1Count = s1EndIdx - transInfo.s1StartIdx + 1;
            transInfo.gStartIdx = (mSizeVStart + startRow) % info.gSize;
            transInfo.gEndIdx = (mSizeVStart + startRow + dealRowCount - 1) % info.gSize;
            transInfo.gCount = dealRowCount;
            Bmm2DataCopyOutNBSDMTiling(attenOutUb, transInfo);
        }
        return;
    }
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename IFAT>
template <typename RT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealInvalidRows(const ExtraInfoMla &info, LocalTensor<RT> &attenOutUb,
                                                          uint32_t startRow, uint32_t dealRowCount,
                                                          uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (LAYOUT_T != LAYOUT::TND) {
        if (!attenMaskFlag || (qSeqSize <= info.s2Size)) {
            return;
        }
    } else {
        if (!attenMaskFlag || (info.actS1Size <= info.s2Size)) {
            return;
        }
    }
    PipeBarrier<PIPE_V>();
    uint32_t s1Tok = qSeqSize - info.s2Size;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        s1Tok = info.actS1Size - info.s2Size;
    }

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
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2ResCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    if constexpr (BALANCE) {
        if constexpr (FLASH_DECODE) {
            if (info.tndIsS2SplitCore) {
                Bmm2FDDataCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
            } else {
                Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
            }
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        if constexpr (FLASH_DECODE) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        }
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2CastAndCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    DealInvalidRows(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealBmm1ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();

    size_t batchBase = 0;


    uint64_t inOutGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * mmResUbSize + (mSizeVStart + startRow) * columnCount;

    if constexpr (QUANT) {
#ifdef MM1_RES_ADD
    #ifdef MM1_DATO
        LocalTensor<MM1_OUT_T> mm1ResInt32 = inputQue1.AllocTensor<MM1_OUT_T>();
        DataCopy(mm1ResInt32, mm1ResGm[inOutGmOffset + batchBase], computeSize);
        inputQue1.EnQue(mm1ResInt32);
        inputQue1.DeQue<MM1_OUT_T>();
        LocalTensor<T> mm1ResFp32 = mm1ResInt32.template ReinterpretCast<T>();
        RowMuls(mmResUb, mm1ResFp32, dequantScale1Ub[startRow * BLOCK_ELEMENT_NUM], dealRowCount, columnCount, actualColumnCount);
        inputQue1.FreeTensor(mm1ResInt32);
    #else
        LocalTensor<MM1_OUT_T> mm1NopeRes = inputQue1.AllocTensor<MM1_OUT_T>();
        DataCopy(mm1NopeRes, mm1ResGm[inOutGmOffset + batchBase], computeSize);
        inputQue1.EnQue(mm1NopeRes);
        inputQue1.DeQue<MM1_OUT_T>();
        Cast(mmResUb, mm1NopeRes, AscendC::RoundMode::CAST_NONE, computeSize);
        inputQue1.FreeTensor(mm1NopeRes);

        uint64_t nopeRopeStride = info.mSize * columnCount;
        LocalTensor<MM1_OUT_T> mm1RopeRes = inputQue1.AllocTensor<MM1_OUT_T>();
        DataCopy(mm1RopeRes, mm1ResGm[inOutGmOffset + nopeRopeStride + batchBase], computeSize);
        inputQue1.EnQue(mm1RopeRes);
        inputQue1.DeQue<MM1_OUT_T>();
        PipeBarrier<PIPE_V>();
        LocalTensor<T> mm1RopeResFp32 = mm1RopeRes.template ReinterpretCast<T>();
        Add(mmResUb, mmResUb, mm1RopeResFp32, computeSize);
        inputQue1.FreeTensor(mm1RopeRes);

        PipeBarrier<PIPE_V>();
        RowMuls(mmResUb, mmResUb, dequantScale1Ub[startRow * BLOCK_ELEMENT_NUM], dealRowCount, columnCount, actualColumnCount);
    #endif
#else
        LocalTensor<MM1_OUT_T> mm1NopeRes = inputQue1.AllocTensor<MM1_OUT_T>();
        DataCopy(mm1NopeRes, mm1ResGm[inOutGmOffset + batchBase], computeSize);
        inputQue1.EnQue(mm1NopeRes);
        inputQue1.DeQue<MM1_OUT_T>();
        Cast(mmResUb, mm1NopeRes, AscendC::RoundMode::CAST_NONE, computeSize);
        inputQue1.FreeTensor(mm1NopeRes);

        uint64_t mm1NopeRopeStride = info.mSize * columnCount;
        LocalTensor<MM1_OUT_T> mm1RopeRes = inputQue1.AllocTensor<MM1_OUT_T>();
        DataCopy(mm1RopeRes, mm1ResGm[inOutGmOffset + mm1NopeRopeStride + batchBase], computeSize);
        inputQue1.EnQue(mm1RopeRes);

        PipeBarrier<PIPE_V>();
        RowMuls(mmResUb, mmResUb, dequantScale1Ub[startRow * BLOCK_ELEMENT_NUM], dealRowCount, columnCount, actualColumnCount);

        inputQue1.DeQue<MM1_OUT_T>();
        PipeBarrier<PIPE_V>();
        LocalTensor<T> mm1RopeResFp32 = mm1RopeRes.template ReinterpretCast<T>();
        Add(mmResUb, mmResUb, mm1RopeResFp32, computeSize);
        inputQue1.FreeTensor(mm1RopeRes);
#endif
    } else {
        LocalTensor<MM1_OUT_T> tmpMmResUb = inputQue1.AllocTensor<MM1_OUT_T>();
        DataCopy(tmpMmResUb, mm1ResGm[inOutGmOffset + batchBase], computeSize);
        inputQue1.EnQue(tmpMmResUb);
        inputQue1.DeQue<MM1_OUT_T>();
        DataCopy(mmResUb, tmpMmResUb, computeSize);
        inputQue1.FreeTensor(tmpMmResUb);
    }

    PipeBarrier<PIPE_V>();
    ElewiseCompute(info, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<KV_T> tmpMMResCastTensor = outputQue1.AllocTensor<KV_T>();

    if constexpr (QUANT) {
        uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
        uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
        LocalTensor<T> maxP = pMaxUb[idx * QMAX_UB_SIZE / sizeof(T) + baseOffset];

        LocalTensor<T> tmpRowMaxUb = tmpBuff3.Get<T>();
        LocalTensor<T> tmpSrcUb = tmpBuff2.Get<T>();

#ifndef SOFTMAX_NEW
        // Row Max 计算过程会修改src 内容，先复制一份
        DataCopy(tmpSrcUb, mmResUb, computeSize);
        PipeBarrier<PIPE_V>();
        RowMax(tmpRowMaxUb, tmpSrcUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
#endif
        // 防止tmpRowMaxUb 值为 0
        Brcb(maxP, tmpRowMaxUb, (dealRowCount + 7) / 8, {1, 8});

        LocalTensor<T> quantScaleP = tmpSrcUb;
        Duplicate(quantScaleP, scaleP1, dealRowCount * BLOCK_ELEMENT_NUM);
        PipeBarrier<PIPE_V>();
        Div(quantScaleP, quantScaleP, maxP, dealRowCount * BLOCK_ELEMENT_NUM);
        PipeBarrier<PIPE_V>();
        RowMuls(mmResUb, mmResUb, quantScaleP, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();

        // 先cast 到 half
        LocalTensor<half> tmpf16Ub = mmResUb.template ReinterpretCast<half>();
        Cast(tmpf16Ub, mmResUb, AscendC::RoundMode::CAST_RINT, computeSize);
        PipeBarrier<PIPE_V>();
        Cast(tmpMMResCastTensor, tmpf16Ub, AscendC::RoundMode::CAST_RINT, computeSize);
    } else {
        Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    }

    outputQue1.EnQue(tmpMMResCastTensor);
    outputQue1.DeQue<KV_T>();

    DataCopy(vec1ResGm[inOutGmOffset + batchBase], tmpMMResCastTensor, computeSize);
    outputQue1.FreeTensor(tmpMMResCastTensor);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AntiquantMM2ResCombine(const ExtraInfoMla &info,
    LocalTensor<MM2_OUT_T> bmmResUb, GlobalTensor<MM2_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;
    LocalTensor<MM2_OUT_T> tmpCInt = inputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(tmpCInt, srcGm[baseOffset], copySize);
    inputQue1.template EnQue(tmpCInt);
    tmpCInt = inputQue1.DeQue<MM2_OUT_T>();
    DataCopy(bmmResUb, tmpCInt, copySize);
    inputQue1.FreeTensor(tmpCInt);
}

// 由于S在GM上已经完成atomic，此处只需要把数据拷贝出来有fp16转成fp32
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AntiquantMatmulResCombine(const ExtraInfoMla &info,
    LocalTensor<T> bmmResUb, GlobalTensor<MM1_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount, float scaleC)
{
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;
    LocalTensor<MM1_OUT_T> tmpMMRes = inputQue1.AllocTensor<MM1_OUT_T>();
    DataCopy(tmpMMRes, srcGm[baseOffset], copySize);
    inputQue1.template EnQue(tmpMMRes);
    tmpMMRes = inputQue1.DeQue<MM1_OUT_T>();
    Cast(bmmResUb, tmpMMRes, AscendC::RoundMode::CAST_NONE, copySize);
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(tmpMMRes);
    Muls(bmmResUb, bmmResUb, scaleC, copySize);
    PipeBarrier<PIPE_V>();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealAntiqBmm1ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * mmResUbSize + mSizeVStart * columnCount;
    AntiquantMatmulResCombine(info, mmResUb, mm1ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount, scaleC1 * static_cast<T>(tilingData->baseParams.scaleValue));
    PipeBarrier<PIPE_V>();

    LocalTensor<T> aMaxBmm1Ub = qAmaxUb[(info.bn2IdxInCurCore % (PRE_LOAD_NUM_MLA)) * QMAX_BUF_SIZE / sizeof(T)];
    LocalTensor<T> aMax = aMaxBmm1Ub[startRow * BLOCK_ELEMENT_NUM];
    RowMuls(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    // mul scalar and mask
    ElewiseCompute(info, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);
    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    size_t dstOffset = (info.loop % PRE_LOAD_NUM_MLA) * vec1ResUbSize + mSizeVStart * columnCount;
    AntiquantSoftmaxResPreProcess(info, vec1ResGm[dstOffset], mmResUb, tmpAFloorUb, startRow, dealRowCount, columnCount,
                                  actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1Inner(const ExtraInfoMla &info)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
#ifdef IFA_SOFTMAX_WITHOUT_BRC
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
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
            }
        } else {
            DealBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                 info.actualSingleProcessSInnerSize);
        }
    }

    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        uint32_t outIdx = info.loop % (PRE_LOAD_NUM_MLA);
        uint32_t softmaxOutOffset = outIdx * BUFFER_SIZE_BYTE_2K / sizeof(T);
        LocalTensor<T> sumTensor = softmaxSumUb[softmaxOutOffset];
        LocalTensor<T> maxTensor = softmaxMaxUb[softmaxOutOffset];

        if constexpr (BALANCE) {
            if (!info.tndIsS2SplitCore) {
                if (softmaxLseFlag) {
#ifdef IFA_SOFTMAX_WITHOUT_BRC
                    LocalTensor<T> lseSumUb = tmpBuff1.Get<T>(BUFFER_SIZE_BYTE_2K);
                    LocalTensor<T> lseMaxUb = tmpBuff1.GetWithOffset<T>(BUFFER_SIZE_BYTE_2K, BUFFER_SIZE_BYTE_2K);
                    Brcb(lseSumUb, sumTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                         {1, BLOCK_ELEMENT_NUM});
                    PipeBarrier<PIPE_V>();
                    Brcb(lseMaxUb, maxTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                         {1, BLOCK_ELEMENT_NUM});
                    PipeBarrier<PIPE_V>();
#endif
                    SoftmaxLseCopyOut(info, lseSumUb, lseMaxUb, mSizeVector);
                }
                return;
            }
        }

        if constexpr (FLASH_DECODE) {
            ComputeLogSumExpAndCopyToGm(info, sumTensor, maxTensor);
            return;
        }

        if (softmaxLseFlag) {
#ifdef IFA_SOFTMAX_WITHOUT_BRC
            LocalTensor<T> lseSumUb = tmpBuff1.Get<T>(BUFFER_SIZE_BYTE_2K);
            LocalTensor<T> lseMaxUb = tmpBuff1.GetWithOffset<T>(BUFFER_SIZE_BYTE_2K, BUFFER_SIZE_BYTE_2K);
            Brcb(lseSumUb, sumTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                 {1, BLOCK_ELEMENT_NUM});
            PipeBarrier<PIPE_V>();
            Brcb(lseMaxUb, maxTensor, (mSizeVector + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
                 {1, BLOCK_ELEMENT_NUM});
            PipeBarrier<PIPE_V>();
#endif
            SoftmaxLseCopyOut(info, lseSumUb, lseMaxUb, mSizeVector);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::SoftmaxLseCopyOut(const ExtraInfoMla &info, LocalTensor<T> &lseSumUb,
                                                                                   LocalTensor<T> &lseMaxUb, uint32_t dealRowCount)
{
    uint64_t dealRowCountAlign = dealRowCount * FP32_ONE_BLOCK_SIZE;
    LocalTensor<T> softmaxlseUb = outputQue2.template AllocTensor<T>();
    ComputeSoftmaxLse(softmaxlseUb, lseSumUb, lseMaxUb, dealRowCountAlign);
    uint64_t curS1Size = LAYOUT_T == LAYOUT::TND ? info.actS1Size : qSeqSize;
    DealSoftmaxLseInvalidRows(softmaxlseUb, lseMaxUb, dealRowCount, curS1Size, info.s1Idx * s1SizeSub);

    outputQue2.EnQue(softmaxlseUb);
    outputQue2.DeQue<T>();

    uint32_t startS1Idx = info.s1Idx * s1SizeSub + mSizeVStart / info.gSize;
    uint32_t startGIdx = mSizeVStart % info.gSize;
    uint32_t endS1Idx = info.s1Idx * s1SizeSub + (mSizeVStart + dealRowCount - 1) / info.gSize;
    uint32_t endGIdx = (mSizeVStart + dealRowCount - 1) % info.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    if constexpr (LAYOUT_T == LAYOUT::TND) {
        uint64_t tokenPrefixSum = (info.bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
        uint64_t bN2Offset = tokenPrefixSum * qHeadNum + info.n2Idx * info.gSize;
        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            outOffset = bN2Offset + s1Idx * kvHeadNum * info.gSize + startGIdx;
            if (s1Idx != endS1Idx) {
                curDealRowCount = info.gSize - startGIdx;
            } else {
                curDealRowCount = endGIdx + 1 - startGIdx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxLseGm[outOffset], softmaxlseUb[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * FP32_ONE_BLOCK_SIZE;
        }
    } else { // BSH BSND
        uint64_t bN2Offset = info.bIdx * qHeadNum * qSeqSize + info.n2Idx * info.gSize * qSeqSize;
        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            outOffset = bN2Offset + startGIdx * qSeqSize + s1Idx;
            if (s1Idx != endS1Idx) {
                curDealRowCount = info.gSize - startGIdx;
            } else {
                curDealRowCount = endGIdx + 1 - startGIdx;
            }
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (qSeqSize - 1) * sizeof(T);
            DataCopyPad(softmaxLseGm[outOffset], softmaxlseUb[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * FP32_ONE_BLOCK_SIZE;
        }
    }
    outputQue2.FreeTensor(softmaxlseUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeSoftmaxLse(LocalTensor<T> softmaxlseUb, LocalTensor<T> &lseSumUb,
                                                                                   LocalTensor<T> &lseMaxUb, uint32_t dealRowCountAlign)
{
    Log(softmaxlseUb, lseSumUb, dealRowCountAlign);
    PipeBarrier<PIPE_V>();
    Add(softmaxlseUb, softmaxlseUb, lseMaxUb, dealRowCountAlign);
    PipeBarrier<PIPE_V>();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::DealSoftmaxLseInvalidRows(LocalTensor<T> &softmaxlseUb,
                                                                                           LocalTensor<T> &lseMaxUb,
                                                                                           uint32_t dealRowCount,
                                                                                           uint64_t s1Size,
                                                                                           uint32_t curS1Idx)
{
    if (!attenMaskFlag) {
        return;
    }

    if (s1Size < curActualSeqLen) {
        return;
    }

    uint64_t seqLenGap = s1Size - curActualSeqLen;
    uint64_t startS1Idx = curS1Idx + mSizeVStart / gSize;
    if (startS1Idx < seqLenGap) {
        SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(dealRowCount), static_cast<uint32_t>(BLOCK_ELEMENT_NUM),
                                          static_cast<uint32_t>(dealRowCount), static_cast<uint32_t>(BLOCK_ELEMENT_NUM)};
        AdjustSoftMaxRes<T, T>(softmaxlseUb, lseMaxUb, negativeIntScalar, FLOAT_INF, softmaxShapeInfo);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealAntiqBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + mSizeVStart * columnCount;
    LocalTensor<half> bmm2ResUb = tmpBuff1.Get<half>();
    AntiquantMM2ResCombine(info, bmm2ResUb, mm2ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount);
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    size_t batchBase = 0;

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        uint64_t vec2ResGmOffset = ((info.loop - 1) % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + (mSizeVStart + startRow) * columnCount;
        LocalTensor<half> bmm2ResPreUb = inputQue2.AllocTensor<half>();
        // step1： 将上一次计算的Oi-1结果从GM拷贝到bmm2ResPreUb
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + batchBase], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);
        inputQue2.DeQue<T>();
        PipeBarrier<PIPE_V>();
        uint32_t loopIdx = info.loop % PRE_LOAD_NUM_MLA;
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        // step2： 将softmaxExpUb转换成FP16
        LocalTensor<half> tmpSoftmaxFp16 = tmpBuff3.Get<half>();
        Cast(tmpSoftmaxFp16, softmaxExpUb[loopIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], AscendC::RoundMode::CAST_ROUND, dealRowCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> tmpExpBrcbResUb = tmpBuff2.Get<half>();
        Brcb(tmpExpBrcbResUb, tmpSoftmaxFp16, (dealRowCount + 7) / 8, {1, 8});
        // step3: Oi = (Oi-1)*tmpSoftmaxFp16 + Otmp
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[loopIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset],
            dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> bmm2ResUbFp32 = tmpBuff2.Get<T>();
        Cast(bmm2ResUbFp32, bmm2ResUb, RoundMode::CAST_NONE, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % PRE_LOAD_NUM_MLA;
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        LocalTensor<T> tmpSumBrcbResUb = tmpBuff1.Get<T>();
        Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUbFp32, bmm2ResUbFp32, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowDivs(bmm2ResUbFp32, bmm2ResUbFp32, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset],
            dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        if (antiqOffsetExistFlag) {
            // bmm2Res + offsetV
            CopyAntiquantScale(antiqOffsetUb, valueAntiqOffsetGm, info.antiqValueParamOffset);
            PipeBarrier<PIPE_V>();
            VecAddMat(bmm2ResUbFp32, antiqOffsetUb, bmm2ResUbFp32, dealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
        }

        Muls(bmm2ResUbFp32, bmm2ResUbFp32, scaleC2, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        CopyAntiquantScale(antiqScaleUb, valueAntiqScaleGm, info.antiqValueParamOffset);
        PipeBarrier<PIPE_V>();
        // ScaleV * bmm2Res
        VecMulMat(bmm2ResUbFp32, antiqScaleUb, bmm2ResUbFp32, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUbFp32, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        // 将从GM workspace中拷贝出来的MM2数据拷贝到临时UB，再拷贝出去
        PipeBarrier<PIPE_V>();
        LocalTensor<MM2_OUT_T> tmpBmm2Res = outputQue1.AllocTensor<MM2_OUT_T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<MM2_OUT_T>();
        uint64_t vec2ResGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + (mSizeVStart + startRow) * columnCount;
        DataCopy(vec2ResGm[vec2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealQuantBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow, 
                                                                    uint32_t dealRowCount, uint32_t columnCount,
                                                                    uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
    LocalTensor<MM2_OUT_T> bmm2ResUb = tmpBuff1.Get<MM2_OUT_T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;

    uint64_t inOutBaseOffset = (mSizeVStart + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    inputQue1.EnQue(tmpBmm2ResUb);
    inputQue1.DeQue<MM2_OUT_T>();

    LocalTensor<T> maxP = pMaxUb[idx * QMAX_UB_SIZE / sizeof(T) + baseOffset];
    // pMaxUb(maxPU32) [dealRowCount, 32B / sizeof(float)] fp32 -> maxPDoubleU32 [dealRowCount, 32B / sizeof(half)] fp32
    LocalTensor<uint32_t> maxPU32 = maxP.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> maxPDoubleU32 = tmpBuff3.Get<uint32_t>();
    uint32_t repeatNums = (dealRowCount * BLOCK_ELEMENT_NUM + REPEAT_ELEMENT_NUM - 1) / REPEAT_ELEMENT_NUM;
    ShiftLeft(maxPDoubleU32, maxPU32, static_cast<uint32_t>(0), static_cast<uint64_t>(FP32_MAX_MASK_ELEMENT_NUM), repeatNums, {2, 1, 16, 8});
    ShiftLeft(maxPDoubleU32[BLOCK_ELEMENT_NUM], maxPU32, static_cast<uint32_t>(0), static_cast<uint64_t>(FP32_MAX_MASK_ELEMENT_NUM), repeatNums, {2, 1, 16, 8});
    PipeBarrier<PIPE_V>();
    LocalTensor<T> maxPFp32 = tmpBuff3.Get<T>();
    LocalTensor<MM2_OUT_T> maxPFp16 = tmpBuff3.Get<MM2_OUT_T>();
    // same buffer fp32 to fp16
    Cast(maxPFp16, maxPFp32, RoundMode::CAST_RINT, dealRowCount * BLOCK_ELEMENT_NUM * static_cast<uint32_t>(sizeof(T) / sizeof(MM2_OUT_T))); // fp32 to fp16
    PipeBarrier<PIPE_V>();
    RowMuls(bmm2ResUb, tmpBmm2ResUb, maxPFp16, dealRowCount, columnCount, actualColumnCount); // max(p)
    inputQue1.FreeTensor(tmpBmm2ResUb);
    PipeBarrier<PIPE_V>();

    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        if (info.isLastTask) {
            event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        }
        LocalTensor<MM2_OUT_T> bmm2ResPreUb = inputQue2.AllocTensor<MM2_OUT_T>();
        uint64_t vec2ResGmOffset = ((info.loop - 1) % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + batchBase], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);
        inputQue2.DeQue<MM2_OUT_T>();

#ifdef IFA_SOFTMAX_WITHOUT_BRC
        LocalTensor<MM2_OUT_T> tmpTensor = tmpBuff3.Get<MM2_OUT_T>();
        Cast(tmpTensor, softmaxExpUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], RoundMode::CAST_RINT, dealRowCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<MM2_OUT_T> tmpExpBrcbResUb = tmpBuff2.Get<MM2_OUT_T>();
        Brcb(tmpExpBrcbResUb, tmpTensor, (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset],
            dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> bmm2ResUb1 = tmpBuff2.Get<T>();
        PipeBarrier<PIPE_V>();
        Cast(bmm2ResUb1, bmm2ResUb, RoundMode::CAST_NONE, vec2ComputeSize);
#ifdef QUANT_MM2_FP16
        PipeBarrier<PIPE_V>();
        Muls(bmm2ResUb1, bmm2ResUb1, (T)(1024.0), dealRowCount * columnCount);
#endif

#ifdef IFA_SOFTMAX_WITHOUT_BRC
        LocalTensor<T> tmpSumBrcbResUb = tmpBuff1.Get<T>();
        Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb1, bmm2ResUb1, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowDivs(bmm2ResUb1, bmm2ResUb1, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        Muls(bmm2ResUb1, bmm2ResUb1, deqScale2Val * scaleP2, dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb1, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<MM2_OUT_T> tmpBmm2Res = outputQue1.AllocTensor<MM2_OUT_T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<MM2_OUT_T>();
        uint64_t vec2ResGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    if constexpr (AMLA != AMLAMODE::NORMAL) {
        uint32_t mSizeAct = info.gSize * info.s1Size;
        uint32_t mSize = Align(mSizeAct, 16U);
        uint32_t vec2ComputeSize = dealRowCount * columnCount;
        LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
        bmm2ResUb.SetSize(vec2ComputeSize);

        LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();

        // GNZ
        constexpr uint64_t dGroupSize = 128;
        uint64_t inOutBaseOffset = (mSizeVStart + startRow) * dGroupSize;
        uint64_t srcGmOffset = (info.bn2IdxInCurCore % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = headDim / dGroupSize; // 4
        dataCopyParams.blockLen = dGroupSize * 16 * sizeof(T) / 32; // 256
        dataCopyParams.srcStride = dGroupSize * (mSize - 16) * sizeof(T) / 32; // 128 * (mSize / 16 - 1) * 2
        dataCopyParams.dstStride = 0;
        DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], dataCopyParams);

        inputQue1.EnQue(tmpBmm2ResUb);
        inputQue1.DeQue<MM2_OUT_T>();
        ConfusionTransposeTiling transposeTiling;
        int64_t numR = 16; // 转置后的行, dealRowCount不一定16对齐
        int64_t numC = columnCount; // 转置后的列, 长和宽的元素数需要是16的倍数
        GetConfusionTransposeTiling(numR, numC, numR * numC, sizeof(T), transposeTiling);
        ConfusionTranspose(bmm2ResUb, tmpBmm2ResUb, TransposeType::TRANSPOSE_ND2ND_ONLY, transposeTiling);
        inputQue1.FreeTensor(tmpBmm2ResUb);

        PipeBarrier<PIPE_V>();
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        uint32_t baseOffset = startRow;
        uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
        LocalTensor<T> tmpSumUb = tmpBuff2.Get<T>();
        Brcb(tmpSumUb, aMlaSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
#else
        uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
        uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
        RowDivs(bmm2ResUb, bmm2ResUb, aMlaSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], dealRowCount, columnCount, actualColumnCount);
#endif

        // 将绝对值大于1e10的数置为0
        PipeBarrier<PIPE_V>();
        LocalTensor<T> absBmm2ResUb = tmpBuff2.Get<T>();
        Abs(absBmm2ResUb, bmm2ResUb, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
        CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        Select(bmm2ResUb, cmpMaskUb, bmm2ResUb, FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);

        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        uint32_t vec2ComputeSize = dealRowCount * columnCount;
        uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
        LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
        bmm2ResUb.SetSize(vec2ComputeSize);

        size_t batchBase = 0;

        uint64_t inOutBaseOffset = (mSizeVStart + startRow) * columnCount;
        uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();
        DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
        inputQue1.EnQue(tmpBmm2ResUb);
        inputQue1.DeQue<MM2_OUT_T>();
        
        DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
        inputQue1.FreeTensor(tmpBmm2ResUb);

        // 除第一个循环外，均需要更新中间计算结果
        if (!info.isFirstSInnerLoop) {
            event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
            uint64_t vec2ResGmOffset = ((info.loop - 1) % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
            DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + batchBase], vec2ComputeSize);
            inputQue2.EnQue(bmm2ResPreUb);

            inputQue2.DeQue<T>();
            PipeBarrier<PIPE_V>();
            uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
#ifdef IFA_SOFTMAX_WITHOUT_BRC
            LocalTensor<T> tmpExpBrcbResUb = tmpBuff2.Get<T>();
            Brcb(tmpExpBrcbResUb, softmaxExpUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], (dealRowCount + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            RowMuls(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
            RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], dealRowCount, columnCount, actualColumnCount);
#endif
            PipeBarrier<PIPE_V>();
            Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
            inputQue2.FreeTensor(bmm2ResPreUb);
        }

        // 最后一次输出计算结果，否则将中间结果暂存至workspace
        if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
            PipeBarrier<PIPE_V>();
            uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
#ifdef IFA_SOFTMAX_WITHOUT_BRC
            LocalTensor<T> tmpSumBrcbResUb = tmpBuff2.Get<T>();
            Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], (dealRowCount + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            RowDivs(bmm2ResUb, bmm2ResUb, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
            RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], dealRowCount, columnCount, actualColumnCount);
#endif
            PipeBarrier<PIPE_V>();
            Bmm2ResCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
        } else {
            PipeBarrier<PIPE_V>();
            LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
            DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
            outputQue1.EnQue(tmpBmm2Res);
            outputQue1.DeQue<T>();
            uint64_t vec2ResGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
            DataCopy(vec2ResGm[vec2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);

            outputQue1.FreeTensor(tmpBmm2Res);
        }
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealBmm2ResDualBaseBlock(uint32_t mIdx, uint32_t mStartRow, const ExtraInfoMla &info, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    uint32_t mSizeAct = info.gSize * info.s1Size;
    uint32_t mSize = Align(mSizeAct, 16U);
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();
    // GNZ
    // ————————————————————————————————————————————————
    // |1 z z z z |5 z z z z |9  z z z z |13 z z z z |
    // |2 z z z z |6 z z z z |10 z z z z |14 z z z z |
    // |3 z z z z |7 z z z z |11 z z z z |15 z z z z |
    // |4 z z z z |8 z z z z |12 z z z z |16 z z z z |
    // ————————————————————————————————————————————————
    constexpr uint64_t dGroupSize = 128;
    constexpr uint64_t mGroupSize = 128;
    uint64_t inOutBaseOffset = (startRow - mStartRow) * dGroupSize;
    // mStartRow算m轴偏移 按128一个大偏来算 (mSizeVStart + mStartRow) / 128 * d * 128,
    // 中偏算一行数据 (mSizeVStart + mStartRow) % 128 * 128, startRaw是在每个大块之间的偏移 要减去mStartRow再*128
    uint64_t srcGmOffset = (info.bn2IdxInCurCore % PRE_LOAD_NUM_MLA) * bmm2ResUbSize +
        (mSizeVStart + mStartRow) / mGroupSize * columnCount * mGroupSize +
        ((mSizeVStart + mStartRow) % mGroupSize) * mGroupSize + inOutBaseOffset;
    // msize 有尾块 按尾块长度算，没有按128算
    uint32_t mTail = mSize % mGroupSize;
    uint32_t gnzStride = mGroupSize;
    if (mTail != 0 && (mSize - (mSizeVStart + mStartRow) == mTail)) {
        gnzStride = mTail;
    }
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = headDim / dGroupSize; // 4
    dataCopyParams.blockLen = dGroupSize * 16 * sizeof(T) / 32; // 256
    dataCopyParams.srcStride = dGroupSize * (gnzStride - 16) * sizeof(T) / 32; // 128 * (mSize / 16 - 1) * 2
    dataCopyParams.dstStride = 0;
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], dataCopyParams);

    inputQue1.EnQue(tmpBmm2ResUb);
    inputQue1.DeQue<MM2_OUT_T>();
    ConfusionTransposeTiling transposeTilingDual;
    int64_t numR = 16; // 转置后的行, dealRowCount不一定16对齐
    int64_t numC = columnCount; // 转置后的列, 长和宽的元素数需要是16的倍数
    GetConfusionTransposeTiling(numR, numC, numR * numC, sizeof(T), transposeTilingDual);
    ConfusionTranspose(bmm2ResUb, tmpBmm2ResUb, TransposeType::TRANSPOSE_ND2ND_ONLY, transposeTilingDual);
    inputQue1.FreeTensor(tmpBmm2ResUb);
    PipeBarrier<PIPE_V>();
    uint32_t baseOffset = startRow;
    uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
    LocalTensor<T> tmpSumUb = tmpBuff2.Get<T>();
    Brcb(tmpSumUb, aMlaSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    RowDivs(bmm2ResUb, bmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
    // 将绝对值大于1e10的数置为0
    PipeBarrier<PIPE_V>();
    LocalTensor<T> absBmm2ResDualUb = tmpBuff2.Get<T>();
    Abs(absBmm2ResDualUb, bmm2ResUb, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResDualUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResDualUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    Select(bmm2ResUb, cmpMaskUb, bmm2ResUb, FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    Bmm2ResCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec2Inner(const ExtraInfoMla &info)
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
            }
        } else if constexpr (QUANT) {
            DealQuantBmm2ResBaseBlock(info, i * mSplitSize, dealSize, headDimAlign, headDim);
        } else {
            DealBmm2ResBaseBlock(info, i * mSplitSize, dealSize, headDimAlign, headDim);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec2DualInner(uint32_t mIdx, const ExtraInfoMla &info, uint32_t mStartRow, uint32_t mdealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }

    uint32_t loopCount = (mdealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mdealSize  - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResDualBaseBlock(mIdx, mStartRow, info, i * mSplitSize + mStartRow, dealSize, headDimAlign, headDim);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::QueryPreProcessL(const ExtraInfoMla &info)
{
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    QueryPreProcessInner(info);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CalcParams(uint32_t loop, ExtraInfoMla &info, TaskContext &task) {
    info.loop = loop;
    info.bIdx = task.bidx;
    info.s1Idx = task.s1idx;
    info.s2Idx = task.s2idx;
    info.curSInnerLoopTimes = task.s2loops;
    info.s1Size = task.s1Size;
    info.s2Size = task.s2Size;
    info.gIdx = task.gidx;
    info.gSize = (task.gidx == gOuter - 1) ? gSizeTail : gSizeSub;
    info.tndIsS2SplitCore = task.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = task.tndCoreStartKVSplitPos;
    info.isBmm2Output = false;
    info.actS1Size = task.actS1Size;
    info.isValid = task.isValid;
    info.isLastTask = task.isLastTask;

    if constexpr (ANTIQUANT) {
        info.isBmm2Output = false;
    }

    if ASCEND_IS_AIV {
        info.mSize = info.gSize * info.s1Size;
        if constexpr (AMLA != AMLAMODE::NORMAL) {
            info.mSizeV = (info.mSize <= 16) ? info.mSize : (((info.mSize + 15) / 16 + 1) / 2 * 16);
        } else {
            info.mSizeV = (info.mSize + 1) / 2;
        }
        info.mSizeVStart = 0;
        if (subBlockNum == 1) { // CV 1:1
            info.mSizeV = info.mSize;
        } else if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    if constexpr (batchContinuous) {
       info.isChangeBatch = false;
    } else {
       if (loop == 0) {
           info.isChangeBatch = true;
       } else {
           info.isChangeBatch = (task.nidx == 0 && task.s2idx == 0);
       }
    }

    if constexpr (BALANCE) {
        info.isFirstSInnerLoop = task.isFirstLoop;
    } else {
        info.isFirstSInnerLoop = (info.s2Idx == 0);
    }
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            // B,N2,G,1,D
            tensorACoreOffset = info.bIdx * qHeadNum * qSeqSize * headDim + info.n2Idx * gSize * qSeqSize * headDim +
                                info.gIdx * gSizeSub * qSeqSize * headDim;
            tensorARopeCoreOffset = info.bIdx * qHeadNum * qSeqSize * headDimRope +
                                    info.n2Idx * gSize * qSeqSize * headDimRope + info.gIdx * gSizeSub * qSeqSize * headDimRope;
            // B,N2,S2,D
            tensorBCoreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDim + info.n2Idx * kvSeqSize * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDimRope + info.n2Idx * kvSeqSize * headDimRope;
            if (!batchContinuous) {
                uint64_t seqSize = SeqLenFromTensorList(info.bIdx);
                tensorBCoreOffset = info.n2Idx * seqSize * headDim;
                tensorBRopeCoreOffset = info.n2Idx * seqSize * headDimRope;
            }

            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += s2IdxFD * sInnerLoopSize * headDim;
                tensorBRopeCoreOffset += s2IdxFD * sInnerLoopSize * headDimRope;
            }
        } else if constexpr (BALANCE) {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == LAYOUT::TND) {
                actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
            } else {
                actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * qSeqSize;
            }
            info.tndBIdxOffset = actualSeqQPrefixSum * qHeadNum * headDim;
            uint64_t tndBIdxRopeOffset = actualSeqQPrefixSum * qHeadNum * headDimRope;
            tensorACoreOffset = info.tndBIdxOffset + (info.s1Idx * s1SizeSub * qHeadNum * headDim);
            tensorARopeCoreOffset = tndBIdxRopeOffset + (info.s1Idx * s1SizeSub * qHeadNum * headDimRope);
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDimRope + info.n2Idx * headDimRope;
        } else {
            // B,S,N2,G,D
            tensorACoreOffset = info.bIdx * qSeqSize * qHeadNum * headDim +
                                info.s1Idx * s1SizeSub * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            tensorARopeCoreOffset = info.bIdx * qSeqSize * qHeadNum * headDimRope +
                                    info.s1Idx * s1SizeSub * qHeadNum * headDimRope + info.n2Idx * gSize * headDimRope;
            // B,S2,N2,D
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDimRope + info.n2Idx * headDimRope;

            if (!batchContinuous) {
                tensorBCoreOffset = info.n2Idx * headDim;
                tensorBRopeCoreOffset = info.n2Idx * headDimRope;
            }

            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDim;
                tensorBRopeCoreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDimRope;
            }
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    info.tensorARopeOffset = tensorARopeCoreOffset;
    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * headDim;
        info.tensorBRopeOffset = tensorBRopeCoreOffset + info.s2Idx * singleProcessSInnerSize * headDimRope;
    } else {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * kvHeadNum * headDim;
        info.tensorBRopeOffset = tensorBRopeCoreOffset + info.s2Idx * singleProcessSInnerSize * kvHeadNum * headDimRope;
    }
    info.attenOutOffset = tensorACoreOffset;

    if constexpr (BALANCE) {
        info.actualSingleProcessSInnerSize = task.s2SizeTail;
    } else {
        info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
        if (info.s2Idx == info.curSInnerLoopTimes - 1) {
            info.actualSingleProcessSInnerSize = task.s2SizeTail;
        }
    }

    info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)BYTE_BLOCK);

    info.antiqKeyParamOffset = info.n2Idx * headDim;
    info.antiqValueParamOffset = info.n2Idx * headDim;
    info.antiqKeyRopeParamOffset = info.n2Idx * headDimRope;

    uint64_t sInnerOffsetDataSize = info.s2Idx * singleProcessSInnerSize;
    if constexpr (FLASH_DECODE) {
        sInnerOffsetDataSize += s2IdxFD * sInnerLoopSize;
    }

    attenMaskCoreOffset = info.bIdx * attenMaskSize;
    info.attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;

    info.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec2L(const ExtraInfoMla &info) {
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    if constexpr (AMLA != AMLAMODE::NORMAL) {
        if (info.s2Idx + 1 != info.curSInnerLoopTimes) {
            return;
        }
    } else {
        if (info.isBmm2Output) {
            return;
        }
    }
    if constexpr (AMLA == AMLAMODE::AMLA_3BUF) {
        uint32_t mGroupSize = 128;
        uint32_t mHead = mSizeVector;
        if ((mSizeVector + mSizeVStart) < mGroupSize) {
            mHead = mSizeVector;
        } else {
            mHead = mGroupSize - (mSizeVStart % mGroupSize);
        }
        uint32_t mloops = (mSizeVector - mHead + mGroupSize - 1) / mGroupSize + 1;
        uint32_t remainMSize = mHead; // 表示还剩多少m需要计算 如：mSizeVector=192，remain=64 表示剩余m长度为64
        uint32_t mStartRaw = 0; // 表示m轴开始计算的位置
        for (uint32_t i = 0; i < mloops; i++) {
            if (i != 0) {
                remainMSize = ((mSizeVector - mHead) % mGroupSize == 0) ? mGroupSize : (mSizeVector - mHead) % mGroupSize;
                mStartRaw = mHead + (i - 1) * mGroupSize;
            }
            ProcessVec2DualInner(i, info, mStartRaw, remainMSize);
        }
    } else {
        ProcessVec2Inner(info);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1L(const ExtraInfoMla &info)
{
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    if constexpr (QUANT) {
        if (info.isFirstSInnerLoop) {
            // preToken_Head, Query->deqScale1Gm的3种场景下(BSND->BSN;BNSD->BNS;TND->TN), query的偏移除以D即为deqScale1Gm的偏移
            uint64_t quantScale1Offset = info.tensorAOffset / headDim;
            LocalTensor<T> tmpDequantScale1 = inputQue2.AllocTensor<T>();
            if ((mSizeVector % 8) == 0) { // 32B对齐
                DataCopy(tmpDequantScale1, deqScale1Gm[quantScale1Offset + mSizeVStart], mSizeVector);
            } else {
                DataCopyExtParams intriParams;
                intriParams.blockLen = mSizeVector * sizeof(T);
                intriParams.blockCount = 1;
                intriParams.dstStride = 0;
                intriParams.srcStride = 0;
                DataCopyPadExtParams<T> padParams;
                padParams.isPad = true;
                padParams.leftPadding = 0;
                padParams.rightPadding = 8 - (mSizeVector % 8);
                padParams.paddingValue = 0;
                DataCopyPad(tmpDequantScale1, deqScale1Gm[quantScale1Offset + mSizeVStart], intriParams, padParams);
            }
            inputQue2.EnQue(tmpDequantScale1);
            inputQue2.DeQue<T>();
            Muls(tmpDequantScale1, tmpDequantScale1, deqScaleKey, mSizeVector);
            PipeBarrier<PIPE_V>();
            Muls(tmpDequantScale1, tmpDequantScale1, static_cast<T>(tilingData->baseParams.scaleValue), mSizeVector); // 乘以1/sqrt(d)
            PipeBarrier<PIPE_V>();
            Brcb(dequantScale1Ub, tmpDequantScale1, (mSizeVector + 7) / 8, {1, 8});
            inputQue2.FreeTensor(tmpDequantScale1);
            PipeBarrier<PIPE_V>();
        }
    }
    ProcessVec1Inner(info);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeMm1(const ExtraInfoMla &info)
{
    if (info.isChangeBatch) {
        InitKeyGm(info.bIdx);
        matmulService.UpdateKey(keyGm);
    }
    matmulService.ComputeMm1(info);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeMm2(const ExtraInfoMla &info)
{
    if (info.isChangeBatch) {
        InitValueGm(info.bIdx);
        matmulService.UpdateValue(valueGm);
    }
    matmulService.ComputeMm2(info);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::FlashDecode()
{
    if (tmpBlockIdx < tilingData->tndSplitCoreParams.usedVecNumOfFd) {
        FDparams fdParams;
        fdService.InitBuffers(pipe);
        AscendC::ICachePreLoad(fdPrefetchLen);
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t *bN2IdxOfFdHead = tilingData->tndSplitCoreParams.balanceFDCoreBArr;
        const uint32_t *gS1IdxOfFdHead = tilingData->tndSplitCoreParams.balanceFDCoreS1Arr;
        const uint32_t *s2SplitNumOfFdHead = tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr;
        const uint32_t *gS1IdxEndOfFdHead = tilingData->tndSplitCoreParams.gS1IdxEndOfFdHead;
        const uint32_t *gS1IdxEndOfFdHeadSplit = tilingData->tndSplitCoreParams.gS1IdxEndOfFdHeadSplit;
        const uint32_t *gS1SplitNumOfFdHead = tilingData->tndSplitCoreParams.gS1SplitNumOfFdHeadMla;
        const uint32_t *gS1LastPartSizeOfFdHead = tilingData->tndSplitCoreParams.gS1LastPartSizeOfFdHead;
#else
        uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreBArr)];
        uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr)];
        uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr)];
        uint32_t gS1IdxEndOfFdHead[ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHead)];
        uint32_t gS1IdxEndOfFdHeadSplit[ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHeadSplit)];
        uint32_t gS1SplitNumOfFdHead[ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1SplitNumOfFdHeadMla)];
        uint32_t gS1LastPartSizeOfFdHead[ARRAY_SIZE(tilingData->tndSplitCoreParams.gS1LastPartSizeOfFdHead)];
        copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreBArr),
                    sizeof(bN2IdxOfFdHead));
        copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr),
                    sizeof(gS1IdxOfFdHead));
        copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr),
                    sizeof(s2SplitNumOfFdHead));
        copy_data_align64((uint8_t *)gS1IdxEndOfFdHead, (uint8_t *)(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHead),
                    sizeof(gS1IdxEndOfFdHead));
        copy_data_align64((uint8_t *)gS1IdxEndOfFdHeadSplit,
                    (uint8_t *)(tilingData->tndSplitCoreParams.gS1IdxEndOfFdHeadSplit),
                    sizeof(gS1IdxEndOfFdHeadSplit));
        copy_data_align64((uint8_t *)gS1SplitNumOfFdHead, (uint8_t *)(tilingData->tndSplitCoreParams.gS1SplitNumOfFdHeadMla),
                    sizeof(gS1SplitNumOfFdHead));
        copy_data_align64((uint8_t *)gS1LastPartSizeOfFdHead,
                    (uint8_t *)(tilingData->tndSplitCoreParams.gS1LastPartSizeOfFdHead),
                    sizeof(gS1LastPartSizeOfFdHead));
#endif
        fdParams = {bN2IdxOfFdHead, gS1IdxOfFdHead, s2SplitNumOfFdHead, gS1SplitNumOfFdHead, gS1LastPartSizeOfFdHead,
                gS1IdxEndOfFdHead, gS1IdxEndOfFdHeadSplit, tilingData->tndSplitCoreParams.usedVecNumOfFd,
                mFdBaseSizeMla};
 
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

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::Process()
{
    //usedCoreNum: 使用的总核数
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
#ifdef IFA_SOFTMAX_WITHOUT_BRC
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_2K / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_2K / sizeof(T));
#else
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_2K / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_2K / sizeof(T));
#endif
        } else {
            matmulService.AllocEventID();
        }

        if constexpr (BALANCE) {
            ProcessBalance();
        } else {
            ProcessBXXD();
        }

        if ASCEND_IS_AIC {
            matmulService.FreeEventID();
        }
    }

    if constexpr (FLASH_DECODE) {
        // 多核同步
        SyncAll();
        if ASCEND_IS_AIV {
            if (tilingData->tndSplitCoreParams.FdBalanceFlag == 1) {
                FlashDecode();
            } else {
                FlashDecodeCompute();
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessBalance()
{
    ExtraInfoMla extraInfo[IFA_PRELOAD_TASK_CACHE_SIZE];
    TaskContext taskContext[PRE_LOAD_NUM_MLA];

    uint32_t gloop = 0;
    uint32_t tasks = 0;

    int tmpN2LoopEnd, tmpS1IdxLoopEnd, tmpS2LoopEnd;

    bool globalLoopStart = true;

    for (int bIdx = bStart; bIdx <= bEnd; bIdx++) {
        GetActualSeqLen(bIdx);
        UpdateInnerLoopCond();
        if (bIdx == bEnd) {
            tmpN2LoopEnd = n2End;
        } else {
            tmpN2LoopEnd = kvHeadNum - 1;
        }

        for (int n2Idx = n2Start; n2Idx <= tmpN2LoopEnd; n2Idx++) {
            if (curActSeqLenIsZero) {
                DealActSeqLenIsZero(bIdx, n2Idx);
                continue;
            }
            if ((bIdx == bEnd) && (n2Idx == n2End)) {
                tmpS1IdxLoopEnd = s1OuterIdxEnd;
            } else {
                tmpS1IdxLoopEnd = (actS1Size + s1SizeSub - 1) / s1SizeSub;      // 切分个数
                tmpS1IdxLoopEnd = tmpS1IdxLoopEnd - 1;                          // 循环上界是<=，这里-1
            }
            for (int sOuterLoopIdx = s1OuterIdxStart; sOuterLoopIdx <= tmpS1IdxLoopEnd; sOuterLoopIdx++) {
                int s2SplitNum = (curActualSeqLen + singleProcessSInnerSize - 1) / singleProcessSInnerSize;      // S2切分份数
                bool isEnd = (bIdx == bEnd) && (n2Idx == n2End) && (sOuterLoopIdx == s1OuterIdxEnd);
                if (isEnd) {
                    tmpS2LoopEnd = s2End;
                } else {
                    tmpS2LoopEnd = s2SplitNum - 1;                                       // 循环上界是<=，这里-1
                }
                // 当前s2是否被切，决定了输出是否要写到attenOut上
                bool tndIsS2SplitCore = ((s2Start == 0) && (tmpS2LoopEnd == s2SplitNum - 1)) ? false : true;

                uint32_t extraLoop = 0;
                if (isEnd) {
                    extraLoop = 2;
                }

                for (int s2Idx = s2Start; s2Idx <= (tmpS2LoopEnd + extraLoop); s2Idx++) {
                    TaskContext &ctx = taskContext[tasks++];
                    ctx.bidx = bIdx;
                    ctx.s1idx = sOuterLoopIdx;
                    ctx.s2idx = s2Idx;
                    ctx.s2loops = tmpS2LoopEnd + 1;
                    ctx.gidx = gIdx;
                    ctx.s2SizeTail = singleProcessSInnerSize;       // 当前任务要处理的S2长度，复用s2SizeTail
                    if (s2Idx == s2SplitNum - 1) {
                        ctx.s2SizeTail = singleProcessSInnerSizeTail;
                    }
                    ctx.tndIsS2SplitCore = tndIsS2SplitCore;
                    ctx.tndCoreStartKVSplitPos = globalLoopStart ? coreStartKVSplitPos : 0;
                    ctx.isFirstLoop = s2Idx == s2Start;
                    int sOuterLoopTailIdx = ((actS1Size + s1SizeSub - 1) / s1SizeSub) - 1;      // 尾块idx=切分份数 - 1
                    if (sOuterLoopIdx == sOuterLoopTailIdx) {           // 尾块场景
                        ctx.s1Size = (actS1Size % s1SizeSub == 0) ? s1SizeSub : (actS1Size % s1SizeSub);
                    } else {
                        ctx.s1Size = s1SizeSub;
                    }
                    ctx.s2Size = curActualSeqLen;
                    ctx.actS1Size = actS1Size;
                    ctx.isValid = (s2Idx <= tmpS2LoopEnd);
                    bool isLast = isEnd && (s2Idx == s2End);
                    ctx.isLastTask = isLast;

                    // PreloadPipeline loop初始值要求为PRE_LOAD_NUM_MLA
                    PreloadPipeline(gloop + PRE_LOAD_NUM_MLA, extraInfo, ctx);

                    gloop += tasks;
                    tasks = 0;
                }
                globalLoopStart = false;
                s2Start = 0;
            }
            s1OuterIdxStart = 0;
        }
        n2Start = 0;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessBXXD()
{
    // 0,1 拥有不同的ExtraInfoMla 和TaskContext
    ExtraInfoMla extraInfo[IFA_PRELOAD_TASK_CACHE_SIZE];
    TaskContext taskContext[PRE_LOAD_NUM_MLA];
    uint32_t gloop = 0;  // gloop 记录全局已处理了多少个Task
    uint32_t tasks = 0;  // 记录每组Task数量（预加载task的数量）

    if constexpr (!FLASH_DECODE) {
        for (uint32_t bn2gIdx = bn2LoopTimes; bn2gIdx > 0; bn2gIdx--) {
            GetBN2Gid(bn2gIdx - 1);
            GetActualSeqLen(bIdx, s1Idx);
            if (curActualSeqLen == 0) {
                DealActSeqLenIsZero(bIdx, n2Idx);
            }
            if (curActualSeqLen != 0) {
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

        uint32_t extraLoop = 0;
        bool isOuterLoopLast = (bn2gIdx == bn2LoopTimes - 1);
        if (isOuterLoopLast) {
            extraLoop = PRE_LOAD_NUM_MLA;
        }

        // 遍历当前S2，当前有Task预加载能力，每PRE_LOAD_NUM_MLA（2）个sInner组成一组Task并发
        for (uint32_t sInnerLoopIdx = 0; sInnerLoopIdx < (sInnerLoopTimes + extraLoop); sInnerLoopIdx++) {
            TaskContext &ctx = taskContext[tasks++];
            ctx.bidx = bIdx;
            ctx.s2idx = sInnerLoopIdx;
            ctx.s2loops = sInnerLoopTimes;
            ctx.gidx = gIdx;
            ctx.s2SizeTail = singleProcessSInnerSizeTail;
            ctx.s1Size = actS1Size;
            ctx.s2Size = curActualSeqLen;
            ctx.s1idx = s1Idx;
            ctx.isValid = (sInnerLoopIdx < sInnerLoopTimes);

            bool isLast = isOuterLoopLast && (sInnerLoopIdx == sInnerLoopTimes - 1);
            ctx.isLastTask = isLast;

            if constexpr (ANTIQUANT || QUANT) {
                // PreloadPipeline loop初始值要求为PRE_LOAD_NUM_MLA
                PreloadPipeline(gloop + PRE_LOAD_NUM_MLA, extraInfo, ctx);
            } else {
                // NBufferPipeline每PRE_LOAD_NUM_MLA轮发一次任务
                if (tasks < PRE_LOAD_NUM_MLA && !isLast) {
                    continue;
                }
                NBufferPipeline(sInnerLoopIdx, sInnerLoopTimes, tasks, gloop, extraInfo, taskContext);
            }

            gloop += tasks;
            tasks = 0;
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::NBufferPipeline(uint32_t sInnerLoopIdx, uint32_t sInnerLoopTimes,
    uint32_t tasks, uint32_t gloop, ExtraInfoMla extraInfo[PRE_LOAD_NUM_MLA], TaskContext taskContext[PRE_LOAD_NUM_MLA])
{
    if (sInnerLoopIdx >= sInnerLoopTimes) {
        for (uint32_t j = 0; j < PRE_LOAD_NUM_MLA; j++) {
            uint32_t loop = gloop + j;
            if (loop >= PRE_LOAD_NUM_MLA) {
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                    ProcessVec2L(extraInfo[(loop - PRE_LOAD_NUM_MLA) % PRE_LOAD_NUM_MLA]);
                }
            }
        }
        return;
    }

    // 遍历Task任务
    for (uint32_t i = 0; i < tasks; i++) {
        uint32_t loop = gloop + i;
        if (loop >= PRE_LOAD_NUM_MLA) {
            if ASCEND_IS_AIV {
                // 当处理到每组最后一个Task时，提前让Vec2处于等待状态，
                CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                // 处理MM2的输出数据A1，A2复原成A，并与SoftMax输出值的操作，并输出结果
                ProcessVec2L(extraInfo[(loop - PRE_LOAD_NUM_MLA) % PRE_LOAD_NUM_MLA]);
            }
        }

        CalcParams(loop, extraInfo[loop % PRE_LOAD_NUM_MLA], taskContext[loop % PRE_LOAD_NUM_MLA]);

        if ASCEND_IS_AIV {
            if constexpr (ANTIQUANT) {
                QueryPreProcessL(extraInfo[loop % PRE_LOAD_NUM_MLA]);
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V0_C1_FLAG);
            }
        }
        if ASCEND_IS_AIC {
            if constexpr (ANTIQUANT) {
                CrossCoreWaitFlag(SYNC_V0_C1_FLAG);
            }

            ComputeMm1(extraInfo[loop % PRE_LOAD_NUM_MLA]);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
        }
    }

    for (uint32_t i = 0; i < tasks; i++) {
        uint32_t loop = gloop + i;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
            ProcessVec1L(extraInfo[loop % PRE_LOAD_NUM_MLA]);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V1_C2_FLAG);
            ComputeMm2(extraInfo[loop % PRE_LOAD_NUM_MLA]);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::PreloadPipeline(uint32_t loop,
    ExtraInfoMla extraInfo[IFA_PRELOAD_TASK_CACHE_SIZE], TaskContext &ctx)
{
    ExtraInfoMla &extraInfo2 = extraInfo[loop % IFA_PRELOAD_TASK_CACHE_SIZE];           // 本轮任务
    ExtraInfoMla &extraInfo1 = extraInfo[(loop - 1) % IFA_PRELOAD_TASK_CACHE_SIZE];     // 上一轮任务
    ExtraInfoMla &extraInfo0 = extraInfo[(loop - 2) % IFA_PRELOAD_TASK_CACHE_SIZE];     // 上两轮任务

    CalcParams(loop, extraInfo2, ctx);

    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            if constexpr (ANTIQUANT) {
                QueryPreProcessL(extraInfo2);
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V0_C1_FLAG);
            }
        }

        if ASCEND_IS_AIC {
            if constexpr (ANTIQUANT) {
                CrossCoreWaitFlag(SYNC_V0_C1_FLAG);
            }
            ComputeMm1(extraInfo2);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
            ProcessVec1L(extraInfo1);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V1_C2_FLAG);
            ComputeMm2(extraInfo1);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG);
        }
    }
    if (extraInfo0.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
            ProcessVec2L(extraInfo0);
        }
        extraInfo0.isValid = false;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenPreloadMla<IFAT>::GetBalanceActualSeqLengths(
    GlobalTensor<uint64_t> &actualSeqLengths, int bIdx)
{
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        return qSeqSize;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::GetTNDAxisStartId(
    uint32_t &bStart, uint32_t &n2Start, uint32_t &s1OuterStart, uint32_t & s2Start,
    int bEnd, int n2End, int s1OuterIdxEnd, int s2End)
{
    if constexpr (BALANCE) {
        if (aiCoreIdx == 0) {
            bStart = 0;
            n2Start = 0;
            s1OuterStart = 0;
            s2Start = 0;
        } else {
            uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsGmQ, bEndPrev);
            uint32_t actualSeqKVPrev = actualSeqLengthsGm.GetValue(bEndPrev);
            uint32_t s1PrevSplitNum = (actualSeqQPrev + s1SizeSub - 1) / s1SizeSub;
            uint32_t s2PrevSplitNum = (actualSeqKVPrev + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            bStart = bEndPrev;
            n2Start = n2EndPrev;
            s1OuterStart = s1OuterIdxEndPrev;
            if (s2EndPrev >= s2PrevSplitNum - 1) {     // 上个核把S2处理完了
                s2Start = 0;
                if (s1OuterIdxEndPrev >= s1PrevSplitNum - 1) {       // 上个核把S1处理完了
                    s1OuterStart = 0;
                    if constexpr (n2EndPrev >= kvHeadNum - 1) {      // 上个核把N2处理完了
                        n2Start = 0;
                        bStart++;
                    } else {
                        n2Start++;
                    }
                    if (bEndPrev >= batchSize) {
                    }
                } else {
                    s1OuterStart++;
                }
            } else {                                // 上个核没有把S2处理完，只需要更新s2Start，其他与上个核保持一致
                s2Start = s2EndPrev + 1;
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyFixedUbToGm(const GlobalTensor<T> &dst,
                                                                                     const LocalTensor<T> &src,
                                                                                     size_t size)
{
    LocalTensor<T> tmp = outputQue2.template AllocTensor<T>();
#ifdef IFA_SOFTMAX_WITHOUT_BRC
    Brcb(tmp, src, (mSizeVector + 7) / 8, {1, 8});
#else
    DataCopy(tmp, src, size);
#endif
    outputQue2.EnQue(tmp);
    outputQue2.DeQue();
    DataCopy(dst, tmp, size);
    outputQue2.FreeTensor(tmp);
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenPreloadMla<IFAT>::CalcAccumOffset(uint32_t bIdx, uint32_t s1Idx)
{
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *balanceFDCoreBArr = tilingData->tndSplitCoreParams.balanceFDCoreBArr;
    const uint32_t *balanceFDCoreS1Arr = tilingData->tndSplitCoreParams.balanceFDCoreS1Arr;
    const uint32_t *balanceFDCoreKVSplitArr = tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr;
#else
    uint32_t balanceFDCoreBArr[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreBArr)];
    uint32_t balanceFDCoreS1Arr[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr)];
    uint32_t balanceFDCoreKVSplitArr[ARRAY_SIZE(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr)];
    copy_data_align64((uint8_t *)balanceFDCoreBArr, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreBArr),
                    sizeof(balanceFDCoreBArr));
    copy_data_align64((uint8_t *)balanceFDCoreS1Arr, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreS1Arr),
                    sizeof(balanceFDCoreS1Arr));
    copy_data_align64((uint8_t *)balanceFDCoreKVSplitArr, (uint8_t *)(tilingData->tndSplitCoreParams.balanceFDCoreKVSplitArr),
                    sizeof(balanceFDCoreKVSplitArr));
#endif
    uint64_t accumTmpOutNum = 0;
    int taskId = 0;
    while (balanceFDCoreBArr[taskId] != bIdx || balanceFDCoreS1Arr[taskId] != s1Idx) {
        accumTmpOutNum += balanceFDCoreKVSplitArr[taskId]; // 计算前面的workspace数
        taskId++;
    }
    return accumTmpOutNum;
}

#endif // INCRE_FLASH_ATTENTION_PRELOAD_MLA
