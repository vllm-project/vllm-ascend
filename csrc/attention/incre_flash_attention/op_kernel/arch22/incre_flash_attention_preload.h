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
 * \file incre_flash_attention_preload.h
 * \brief
 */
#ifndef INCRE_FLASH_ATTENTION_PRELOAD
#define INCRE_FLASH_ATTENTION_PRELOAD

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "../ifa_public_define.h"
#include "lib/matrix/matmul/tiling.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

#define PRE_LOAD_NUM 4
struct ExtraInfo {
    uint32_t loop = 0;
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t curSInnerLoopTimes = 0;
    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t pseShiftCoreOffset = 0;
    uint64_t attenMaskOffset = 0;
    uint64_t antiqKeyParamOffset = 0;
    uint64_t antiqValueParamOffset = 0;
    uint64_t antiqKeyParamOffsetPerToken = 0;
    uint64_t antiqValueParamOffsetPerToken = 0;
    uint64_t perChannelQuantOffset = 0ULL;
    uint32_t actualSinglePreProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;
    bool isFirstSInnerLoop = false;
    bool needSetOrgShape = false;
    bool isChangeBatch = false;
    uint32_t s2BatchOffset = 0;
    ExtraInfo() {
      loop = 0;
      bIdx = 0;
      n2Idx = 0;
      s2Idx = 0;
      bn2IdxInCurCore = 0;
      curSInnerLoopTimes = 0;
      tensorAOffset = 0;
      tensorBOffset = 0;
      attenOutOffset = 0;
      pseShiftCoreOffset = 0;
      attenMaskOffset = 0;
      antiqKeyParamOffset = 0;
      antiqValueParamOffset = 0;
      antiqKeyParamOffsetPerToken = 0;
      antiqValueParamOffsetPerToken = 0;
      perChannelQuantOffset = 0;
      actualSinglePreProcessSInnerSize = 0;
      actualSingleProcessSInnerSizeAlign = 0;
      isFirstSInnerLoop = false;
      needSetOrgShape = false;
      isChangeBatch = false;
      s2BatchOffset = 0;
    }
};

template <typename IFAT> class IncreFlashAttentionAttenPreload {
public:
    __aicore__ inline IncreFlashAttentionAttenPreload(){};
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                     __gm__ uint8_t *workspace);
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const IncreFlashAttentionTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe, bool isPrefix = false);
    __aicore__ inline void Process();
    __aicore__ inline void InitAntiquant(__gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffse);
    __aicore__ inline void InitPostQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2);
    __aicore__ inline bool IsFinish(uint32_t loop);

    __aicore__ inline void ProcessSysPrefixCombine();
    __aicore__ inline void InitPrefix(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                      __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask,
                                      __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                      __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
                                      __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                      const IncreFlashAttentionTilingDataPrefix *__restrict tiling,
                                      __gm__ uint8_t *gmTiling, TPipe *tPipe);

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool KV_CONTINUOUS = IFAT::kvContinuous;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr uint8_t PER_CHANNEL_MODE_PRE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t PER_TOKEN_MODE_PRE = 1; // 伪量化: K V per-token
    static constexpr uint8_t PER_CHANNEL_TOKEN_MODE_PRE = 2; // 伪量化: K per-channel and V per-token
    static constexpr uint8_t ANTIQUANT_MODE_PRE = IFAT::antiquantMode;
    static constexpr bool SHARED_PREFIX_PRE = IFAT::sharedPrefix;

    static constexpr bool ANTIQUANT_PRE = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr bool ANTIQUANT_PER_CHANNEL_TOKEN = (ANTIQUANT_PRE && (ANTIQUANT_MODE_PRE == PER_CHANNEL_TOKEN_MODE_PRE));
    static constexpr bool ANTIQUANT_PER_TOKEN = (ANTIQUANT_PRE && (ANTIQUANT_MODE_PRE == PER_TOKEN_MODE_PRE));
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT_PRE && (ANTIQUANT_MODE_PRE == PER_CHANNEL_MODE_PRE));
    using ANTIQ_PARAMS_T = typename AscendC::Conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    // define pse datetype
    using pseShiftType = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT_PRE || QUANT), int32_t, T>::type;

#ifdef USE_MM_API
    using singleRowAType = MatmulType<TPosition::GM, CubeFormat::VECTOR, KV_T, false>;
    using multiRowAType = MatmulType<TPosition::GM, CubeFormat::ND, KV_T, false>;
    // using AType = typename AscendC::Conditional<ANTIQUANT_PRE, multiRowAType, singleRowAType>::type;
    using AType = multiRowAType;

    template <typename PRE_T> static __aicore__ inline constexpr int32_t GetC0SizeBySrcType()
    {
        if (sizeof(PRE_T) == sizeof(float)) {
            return 8;
        } else if (sizeof(PRE_T) == sizeof(int8_t)) {
            return 32;
        }
        return 16;
    }

    // 参考mamtul_impl.h中实现
    template <typename PRE_T>
    static __aicore__ void
    CopyND2NZ(const LocalTensor<PRE_T> &dst, const GlobalTensor<PRE_T> &src, const int row, const int col,
              const int height, const int width, const int gCol, const int ndNum = 1, const int srcNdMatrixStride = 0,
              const int dstNzMatrixStride = 0, const int dstNzC0Stride = 0)
    {
        int64_t srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);

        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;

        nd2nzParams.dstNzC0Stride = dstNzC0Stride;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;

        DataCopy(dst, src[srcOffset], nd2nzParams);
    }

    // bmm1 回调，row方向对应k、d；col方向对应n、s2
    static __aicore__ void bmm1CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col,
                                      int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
    }

    // bmm2 回调，row方向对应k、s2；col方向对应n、d
    static __aicore__ void bmm2CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col,
                                      int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
    }

    // define matmul1
    typedef MatmulType<TPosition::GM, CubeFormat::ND, KV_T, true> b1Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, float> bias1Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, MM_OUT_T> c1Type;
    using mm1Type = typename AscendC::Conditional<PAGE_ATTENTION,
                                                  MatmulImpl<AType, b1Type, c1Type, bias1Type, CFG_MDL_EXCEED_INIT_CALLBACK,
                                                         matmul::MatmulCallBackFunc<nullptr, nullptr, bmm1CopyB1>>,
                                                  MatmulImpl<AType, b1Type, c1Type, bias1Type, CFG_MDL_EXCEED_INIT>>::type;
    mm1Type mm;

    // define matmul2
    typedef MatmulType<TPosition::GM, CubeFormat::VECTOR, KV_T, false> a2Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, KV_T, false> b2Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, float> bias2Type;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, MM_OUT_T> c2Type;
    using mm2Type = typename AscendC::Conditional<PAGE_ATTENTION,
                                                  MatmulImpl<AType, b2Type, c2Type, bias2Type, CFG_MDL_EXCEED_INIT_CALLBACK,
                                                         matmul::MatmulCallBackFunc<nullptr, nullptr, bmm2CopyB1>>,
                                                MatmulImpl<AType, b2Type, c2Type, bias2Type, CFG_NORM_EXCEED_INIT>>::type;
    mm2Type bmm2;

#endif

protected:
    const IncreFlashAttentionTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    // PSE
    GlobalTensor<pseShiftType> pseShiftGm;

    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;

    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // antiquant
    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> keyAntiqScaleGm;
    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> valueAntiqScaleGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // out quant
    GlobalTensor<float> quantScale2Gm;
    GlobalTensor<float> quantOffset2Gm;
    GlobalTensor<bfloat16_t> quantScale2Bf16Gm;
    GlobalTensor<bfloat16_t> quantOffset2Bf16Gm;
    // workspace
    GlobalTensor<Q_T> prefixQueryPreProcessResGm;
    GlobalTensor<KV_T> queryPreProcessResGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<uint32_t> bmm1CallBackDataGm;
    GlobalTensor<uint32_t> bmm2CallBackDataGm;

    // kv_left_padding
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1;   // 32K, inque
    TQue<QuePosition::VECIN, 1> inputQue2;   // 16K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 32K, outque
    TQue<QuePosition::VECOUT, 1> outputQue2; // 8K, outque
#ifndef USE_MM_API
    TQue<QuePosition::A1, 1> inputQueL1A;
    TQue<QuePosition::B1, 1> inputQueL1B;
    // L0_A
    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong[2];
    BufferManager<TPosition::A2, 2> l0aBufferManager;   // 2:l0a buf num
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong[2];
    BufferManager<TPosition::B2, 2> l0bBufferManager;   // 2:l0b buf num
    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<MM_OUT_T> cL0TensorPingPong[2];
    BufferManager<TPosition::CO1, 2> l0cBufferManager;  // 2:l0c buf num
#endif

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K
    TBuf<> tmpBuff3; // 2K

    // 常驻tbuf
    TBuf<> antiqScaleBuff;            // 4K
    TBuf<> antiqOffsetBuff;           // 4K
    TBuf<> qAmaxBuff[PRE_LOAD_NUM];   // PRE_LOAD_NUM * (2K + 256B)
    TBuf<> softmaxResAmaxBuff[PRE_LOAD_NUM];        // 2K + 256B
    TBuf<> qRowSumBuff[PRE_LOAD_NUM];               // 2K + 256B
    TBuf<> softmaxResRowSumBuff[PRE_LOAD_NUM];      // 2K + 256B
    TBuf<> softmaxMaxBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff[PRE_LOAD_NUM]; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff;     // 2K
    TBuf<> softmaxSumDefaultBuff;     // 2K
    TBuf<> bmm1PageAttentionDataBuff; // 64B
    TBuf<> bmm2PageAttentionDataBuff; // 64B

    LocalTensor<T> softmaxMaxUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxSumUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxExpUb[PRE_LOAD_NUM];
    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<uint32_t> bmm1PageAttentionDataUb;
    LocalTensor<uint32_t> bmm2PageAttentionDataUb;

    // antiquant msd
    LocalTensor<T> aMaxBmm1Ub[PRE_LOAD_NUM];
    LocalTensor<T> aMaxBmm2Ub[PRE_LOAD_NUM];
    LocalTensor<T> softmaxScaleResRowSumUb[PRE_LOAD_NUM];
    LocalTensor<T> antiqScaleUb;
    LocalTensor<T> antiqOffsetUb;
    LocalTensor<T> qRowSumUb[PRE_LOAD_NUM];

    // sys prefix tmpBuffer
    GlobalTensor<T> softmaxRowMaxGm;
    GlobalTensor<T> softmaxRowSumGm;
    GlobalTensor<T> softmaxRowExpGm;
    GlobalTensor<T> lseSumGm;
    GlobalTensor<T> lseMaxGm;
    GlobalTensor<T> msdRowMax1Gm;
    GlobalTensor<T> msdRowMax2Gm;
    GlobalTensor<T> msdRowSum1Gm;
    GlobalTensor<T> msdRowSum2Gm;
    GlobalTensor<T> sysPrefixAttenOutGm;
    GlobalTensor<T> usrPromptAttenOutGm;

    uint64_t msdRowPreMaxSize = 0;
    uint64_t msdRowPreSumSize = 0;
    uint64_t softmaxMaxSumPreExpSize = 0;

    uint64_t sysPrefixLen = 0;
    uint32_t formerCoreNumSp = 0;
    uint32_t blockSplitBn2RangeSp = 0;
    uint32_t tailBlockSplitBn2RangeSp = 0;
    uint32_t usedPreCoreNumSp = 0;
    bool calcSysPrefixFlag = false;
    uint32_t batchSizeQ = 0;

    static constexpr uint64_t PRE_SYNC_MODE2 = 2;
    static constexpr uint64_t PRE_SYNC_V0_C1_FLAG = 6;
    static constexpr uint64_t PRE_SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t PRE_SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t PRE_SYNC_C2_V2_FLAG = 9;

    static constexpr uint32_t PRE_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t PRE_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM = 512 / sizeof(KV_T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM_THRESHLOD = 128 / sizeof(KV_T);
    static constexpr T antiquantPreExpandCoeff = KVINT4 ? 14.98 : 254;
    static constexpr T antiqCoeff1 = KVINT4 ? 7.49 : 127;
    static constexpr T antiqCoeff2 = 1 / antiqCoeff1;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    static constexpr T FP16_ATTEN_MASK_SCALAR_VALUE = -10000;           // 用于mask为fp16类型
    static constexpr uint32_t KV_PRE_LOAD_L1_ROW_NUM = 512 / sizeof(KV_T);
    bool antiqOffsetExistFlag = false;
    uint32_t msdPreIterNum = 0U;
    bool antiquantPerHeadFlag = false;
    bool antiquantParamsInPagedAttentionFlag = false;
    uint32_t antiquantPerTensorFlag = 0U;
    uint64_t sPreUnitSize = 0;

    // kv_left_padding
    uint32_t kvPaddingPreFlag = 0;
    uint64_t kvPaddingBeginOffset = 0;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRE_LOAD_NUM;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;

    uint32_t tmpPreBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    __gm__ uint8_t *blocktablePrePtr = nullptr;
    __gm__ uint32_t *bmm1CallBackDataPtr = nullptr;
    __gm__ uint32_t *bmm2CallBackDataPtr = nullptr;

    // tilingdata
    uint64_t singlePreProcessSInnerSize = 0U;
    uint32_t sInnerPreLoopTimes = 0U;
    uint64_t singlePreProcessSInnerSizeTail = 0U;
    uint32_t formerPreCoreNum = 0U;
    uint32_t usedPreCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;

    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;
    uint32_t batchPreContinuous = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t gSizeVector = 0ULL;
    uint64_t gSizeCube = 0ULL;
    uint64_t gSizeStart = 0ULL;
    uint64_t kvSeqSize = 0ULL;
    uint64_t headDim = 0ULL;
    uint64_t headDimAlign = 0ULL;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t s2BatchBaseOffset = 0;

    // 是否返回lse
    bool softmaxLseFlag;

    // attention mask
    bool attenMaskPreFlag = false;
    uint32_t selectWithByteMaskTmpMinSize = 0U;
    uint32_t attenMaskPreSizeAlign = 0U;
    // pse mask
    bool pseShiftPreFlag = false;
    uint32_t pseShiftB = 0U;
    uint32_t pseShiftS = 0U;
    uint64_t pseShiftOffset = 0U;
    uint64_t pseShiftCoreOffset = 0ULL;
    uint32_t pseMaskPreSizeAlign = 0U;
    // offset
    uint64_t tensorACorePreOffset = 0ULL;
    uint64_t tensorBCorePreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t valueOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t antiqPreParamOffset = 0ULL;
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
    uint32_t actualPreCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;
    bool flashDecodeFlag = false;

    uint64_t curActualPreSeqLen = 0ULL;
    uint64_t curSinglePreProcessSInnerSizeAlign = 0ULL;
    uint64_t actualSinglePreProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockPreSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenDims = 0U;
    // out quant
    bool isPerChnU8Out = false;
    bool isPreOutQuantTypeBf16 = false;
    float quantScale2Value = 0;
    float quantOffset2Value = 0;
    bool isQuantOffset2Exist = false;
    uint64_t perChannelQuantOffset = 0ULL;

    // 记录当前轮的bIdx nIdx s2Idx actualLen
    uint32_t curBIdx = 0;
    uint32_t curN2Idx = 0;
    uint32_t curS2Idx = 0;
    uint32_t curSInnerLoopTimes = 0;

    uint32_t bn2IdxInCurCore = 0;
    uint32_t bn2s2LoopTimes = 0;
    ExtraInfo extraInfo[PRE_LOAD_NUM]{};
    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void CalcParams(uint32_t loop);
    __aicore__ inline void ComputeMm1(uint32_t loop);
    __aicore__ inline void ProcessVec1L(uint32_t loop);
    __aicore__ inline void ComputeMm2(uint32_t loop);
    __aicore__ inline void ProcessVec2L(uint32_t loop);
      __aicore__ inline void SetLoopTimes();
    __aicore__ inline void QueryPreProcessL(uint32_t loop);

    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t nCopyIdx, uint32_t nCopyRowCount, uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor);

    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount);
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount);
    __aicore__ inline void LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize);
    __aicore__ inline void LoadDataMm2B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor);

    bool curActSeqLenIsZero = false;
    // PA
    const uint32_t mmPACallBackDataSize = 64U;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
    }
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void CalculateSUnitSize();
    __aicore__ inline bool ComputeKVPaddingBeginOffset();
    __aicore__ inline void GetActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void InitCalcParams();
    __aicore__ inline void InitCalcParamsEach();

    __aicore__ inline void GetBN2id(const uint32_t bn2Idx);
    __aicore__ inline void CalcBN2Offset();
    __aicore__ inline void CalcBN2Params();

    __aicore__ inline void CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx);
    __aicore__ inline void UpdateOffsetsVec(uint32_t sInnerLoopIdx);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount, uint32_t actualColumnCount);

    __aicore__ inline void CopyAntiquantScale(LocalTensor<T> &castUb, GlobalTensor<Q_T> srcGm, uint64_t offset);

    __aicore__ inline void CopyAntiquantParamsPerToken(GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset,
                                                       uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiquantParamsPerTokenHead(GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset,
                                                           uint32_t columnCount);
    __aicore__ inline void CopyAntiquantParamsPerTokenParamsPagedAttention(GlobalTensor<ANTIQ_PARAMS_T> srcGm,
                                                                           uint64_t offset, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiqQuery(LocalTensor<T> &queryCastUb, uint64_t qOffset, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMatmulPreProcess(GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb,
                                                     LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void AntiquantSoftmaxResPreProcess(GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb,
                                                         LocalTensor<T> tmpAFloorUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount);
    __aicore__ inline void AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb, LocalTensor<T> tmpAUb,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<T> &tmpA1, LocalTensor<T> &tmpA2,
                                                uint32_t calcSize, bool isFirst, uint64_t outOffset);
    __aicore__ inline void DealQueryPreProcessBaseBlock(uint32_t loop, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount);
    __aicore__ inline void DealQueryPreProcessBaseBlockPerToken(uint32_t loop, uint32_t startRow, uint32_t dealRowCount,
                                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void QueryPreProcessInner(uint32_t loop);

    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void SetMMOrgShape();
    __aicore__ inline void SetMM1OrgShape(ExtraInfo& info);
    __aicore__ inline void SetMM2OrgShape(ExtraInfo& info);
    __aicore__ inline void SysPrefixSetMMOrgShape();
    __aicore__ inline void Bmm1Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void Bmm2Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void Bmm1ComputeCommon(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void Bmm2ComputeCommon(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void SysPrefixBmm1Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);
    __aicore__ inline void SysPrefixBmm2Compute(const uint32_t bn2Idx, const uint32_t sInnerLoopIdx);

    __aicore__ inline void DealBmm1ResBaseBlock(const uint32_t loop, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlock(uint32_t loop, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm1ResBaseBlockPerToken(uint32_t loop,  uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMatmulResCombine(LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm,
                                                     uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec1Inner(uint32_t loop);
    __aicore__ inline void PostProcessVec1();

    __aicore__ inline void DealBmm2ResBaseBlock(const uint32_t loop, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm2ResBaseBlock(uint32_t loop, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount);
    __aicore__ inline void DealAntiqBmm2ResBaseBlockPerToken(uint32_t loop,  uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(uint32_t loop);

    __aicore__ inline void SoftmaxFlashV2Compute(uint32_t loop, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void PseShiftCopyIn(uint32_t loop, uint32_t startRow, uint32_t rowCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(uint32_t loop, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                              uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void CombineSplitKVRes();
    __aicore__ inline void CopyAccumOutIn(uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax, uint32_t startRow,
                                             uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow,
                                          uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void Bmm2FDDataCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void PostQuant(uint64_t perChannelQuantOffset, LocalTensor<T> &bmm2ResUb, LocalTensor<int8_t> &bmmPre2ResUbInt8, uint32_t startRow,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void SysPrefixInitAllZeroOutput();
    __aicore__ inline void InitAllZeroInt8Output(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);

    __aicore__ inline void SysPrefixAttenResCombine();
    __aicore__ inline void SysPrefixSaveLseFd(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax, uint32_t start,
                                              uint32_t count);
    __aicore__ inline void SysPrefixSaveLseFA();
    __aicore__ inline void SysPrefixSaveAttenRes(uint32_t bIndex, uint32_t n2Index, LocalTensor<T> &bmm2ResUb,
                                                 uint32_t startRow, uint32_t rows, bool isPrefix);
    __aicore__ inline void SysPrefixLseToScales(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax);
    __aicore__ inline void SysPrefixAttenReduce(LocalTensor<T> &dst, GlobalTensor<T> &atten1, GlobalTensor<T> &atten2,
                                                LocalTensor<T> scales, uint32_t startRow, uint32_t rows);
    __aicore__ inline void SysPrefixAttenOutput(GlobalTensor<OUT_T> &dst, LocalTensor<T> &attenOut, uint32_t startRow,
                                                uint32_t rows);
    __aicore__ inline void SysPrefixSaveLse(uint32_t bIndex, uint32_t n2Index, LocalTensor<T> &softmaxSumUb,
                                            LocalTensor<T> &softmaxMaxUb, uint32_t start, uint32_t count,
                                            bool isPrefix);

    __aicore__ inline void SysPrefixSaveZeroLse(uint32_t bIndex, uint32_t n2Index, bool isPrefix);
    __aicore__ inline void SysPrefixSaveZeroAttenRes(uint32_t bIndex, uint32_t n2Index, bool isPrefix);

    __aicore__ inline void SysPrefixSaveMsdSum1(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdSum1(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveMsdSum2(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdSum2(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveMsdMax1(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdMax1(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveMsdMax2(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadMsdMax2(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveSoftmaxMax(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadSoftmaxMax(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveSoftmaxSum(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadSoftmaxSum(uint32_t bIndex);

    __aicore__ inline void SysPrefixSaveSoftmaxExp(uint32_t bIndex);
    __aicore__ inline void SysPrefixLoadSoftmaxExp(uint32_t bIndex);

    __aicore__ inline void CopyDataInByQueue1(LocalTensor<T> &dst, const GlobalTensor<T> &src, size_t size);
    __aicore__ inline void CopyDataInByQueue2(LocalTensor<T> &dst, const GlobalTensor<T> &src, size_t size);

    __aicore__ inline void CopyGmToFixedUb(LocalTensor<T> &dst, const GlobalTensor<T> &src, size_t size);
    __aicore__ inline void CopyFixedUbToGm(const GlobalTensor<T> &dst, const LocalTensor<T> &src, size_t size);
    __aicore__ inline void SoftmaxLseOutput(LocalTensor<T> &lse);

    __aicore__ inline void DealKvInt4ColumnOdd(LocalTensor<T> mmResUb, uint32_t dealRowCount,
        uint32_t columnCount, uint32_t actualColumnCount);
};

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitTilingData()
{
    singlePreProcessSInnerSize = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSize;
    sInnerPreLoopTimes = tilingData->increFlashAttentionSingleCoreParams.sInnerLoopTimes;
    singlePreProcessSInnerSizeTail = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSizeTail;
    usedPreCoreNum = tilingData->increFlashAttentionSingleCoreParams.usedCoreNum;
    formerPreCoreNum = tilingData->increFlashAttentionSingleCoreParams.formerCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;
    flashDecodeFlag = splitKVNum > 0;

    mmResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.mmResUbSize;
    bmm2ResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    gSize = tilingData->baseParams.nNumOfQInOneGroup;
    gSizeCube = gSize;
    gSizeVector = (gSize + 1) / 2; // vector0 : half gsize, align up
    gSizeStart = 0;
    if (tmpPreBlockIdx % 2 == 1) { // vector1
      gSizeStart = gSizeVector;
      gSizeVector = gSize - gSizeVector;
    }
    kvSeqSize = tilingData->baseParams.seqSize;
    headDim = tilingData->baseParams.headSize;
    batchPreContinuous = tilingData->baseParams.batchContinuousFlag;
    msdPreIterNum = tilingData->baseParams.msdIterNum;
    antiquantPerTensorFlag = tilingData->baseParams.antiquantPerTensorFlag;
    antiquantPerHeadFlag = (tilingData->baseParams.antiquantPerHeadFlag == 1);
    antiquantParamsInPagedAttentionFlag = (tilingData->baseParams.antiquantParamsInPagedAttentionFlag == 1);

    headDimAlign = Align(headDim, BYTE_BLOCK);

    attenMaskPreFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attenMaskSize = tilingData->baseParams.attenMaskSize;
    selectWithByteMaskTmpMinSize = tilingData->baseParams.selectWithByteMaskTmpMinSize;

    antiqSeqSize = tilingData->baseParams.antiqSeqSize;

    pseShiftPreFlag = (tilingData->baseParams.pseShiftFlag == 1) ? true : false;
    if (pseShiftPreFlag) {
        pseShiftB = tilingData->baseParams.pseShiftB;
        pseShiftS = tilingData->baseParams.pseShiftS;
    }

    kvPaddingPreFlag = tilingData->baseParams.kvPaddingFlag;

    // out quant
    isPerChnU8Out = tilingData->outputParams.isPerChnOut == 0 ? false : true;
    isPreOutQuantTypeBf16 = tilingData->outputParams.isOutQuantTypeBf16 == 0 ? false : true;

    // 是否输出lse
    softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        // queue
        pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_8K);

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
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            pipe->InitBuffer(qAmaxBuff[i], BUFFER_SIZE_BYTE_256B + BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(softmaxResAmaxBuff[i], BUFFER_SIZE_BYTE_256B + BUFFER_SIZE_BYTE_256B); // perToken使用,perChannel不需要
            pipe->InitBuffer(qRowSumBuff[i], BUFFER_SIZE_BYTE_256B + BUFFER_SIZE_BYTE_256B); // perToken,systenfix,lse需要使用,perChannel不需要
            pipe->InitBuffer(softmaxResRowSumBuff[i], BUFFER_SIZE_BYTE_256B + BUFFER_SIZE_BYTE_256B); // perToken,systenfix需要使用,perChannel不需要
        }
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            pipe->InitBuffer(softmaxMaxBuff[i], BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(softmaxExpBuff[i], BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(softmaxSumBuff[i], BUFFER_SIZE_BYTE_256B);
        }
        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_256B);
    } else {
#ifndef USE_MM_API
        // L1
        pipe->InitBuffer(inputQueL1A, 2, 32 * KV_PRE_LOAD_L1_ROW_NUM * sizeof(KV_T)); // mm1:msdPreIterNum * gSize * headDimAlign, mm2:msdPreIterNum * gSize * singlePreProcessSInnerSize
        pipe->InitBuffer(inputQueL1B, 2, KV_PRE_LOAD_L1_ROW_NUM * headDimAlign * sizeof(KV_T));
        // L0A
        pipe->InitBuffer(tmpBufL0A, 2 * 32 * 256); // mm1:16*128, mm2:16*256
        aL0TensorPingPong[0] = tmpBufL0A.Get<KV_T>();
        aL0TensorPingPong[1] = aL0TensorPingPong[0][32 * 256 / sizeof(KV_T)];
        // L0B
        pipe->InitBuffer(tmpBufL0B, 2 * 256 * 128);
        bL0TensorPingPong[0] = tmpBufL0B.Get<KV_T>();
        bL0TensorPingPong[1] = bL0TensorPingPong[0][256 / sizeof(KV_T) * 128];
        // L0C
        pipe->InitBuffer(tmpBufL0C, 2 * 32 * KV_PRE_LOAD_L1_ROW_NUM * sizeof(MM_OUT_T)); // mm1:16*1024*4, mm2:16*128*4
        cL0TensorPingPong[0] = tmpBufL0C.Get<MM_OUT_T>();
        cL0TensorPingPong[1] = cL0TensorPingPong[0][32 * KV_PRE_LOAD_L1_ROW_NUM];
#endif
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitAllZeroInt8Output(uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gPreSplitSize > gSize) {
        gPreSplitSize = gSize;
    }
    uint32_t loopCount = (gSize + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = gSize - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        uint32_t startRow = gPreSplitSize * i;
        uint32_t dealRowCount = dealSize;
        uint32_t columnCount = headDimAlign;
        uint32_t actualColumnCount = headDim;
        LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>(); // bmm2 result is zero
        Duplicate(bmm2ResUb, static_cast<float>(0), dealRowCount * columnCount);
        LocalTensor<OUT_T> bmmPre2ResUbInt8 = outputQue1.AllocTensor<OUT_T>();
        uint64_t perChannelQuantOffset = n2Idx * headDim * gSize;
        PostQuant(perChannelQuantOffset, bmm2ResUb, bmmPre2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
        outputQue1.EnQue(bmmPre2ResUbInt8);
        outputQue1.DeQue<OUT_T>();
        uint64_t attenOutOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        Bmm2DataCopyOut(attenOutOffset, bmmPre2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
        outputQue1.FreeTensor(bmmPre2ResUbInt8);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t copySize = gSize * headDim;
    if constexpr (POST_QUANT) { // out int8
        InitAllZeroInt8Output(bIdx, n2Idx);
    } else {
        matmul::InitOutput<OUT_T>(attentionOutGm[(bIdx * kvHeadNum + n2Idx) * copySize], copySize, 0);
    }

    if (softmaxLseFlag) {
        LocalTensor<T> softmaxlsePreOut = outputQue1.template AllocTensor<T>();
        float minf = -3.40E+38;
        Duplicate(softmaxlsePreOut, minf, gSize);
        outputQue1.EnQue(softmaxlsePreOut);
        outputQue1.DeQue();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float) * gSize;
        intriParams1.blockCount = 1;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[(bIdx * kvHeadNum + n2Idx) * gSize], softmaxlsePreOut, intriParams1);
        outputQue1.FreeTensor(softmaxlsePreOut);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::GetActualSeqLen()
{
    if (actualLenDims == 0) {
        curActualPreSeqLen = kvSeqSize;
        if (!batchPreContinuous) {
            curActualPreSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualPreSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualPreSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::GetBN2id(const uint32_t bn2Idx)
{
    if (flashDecodeFlag) {
        bIdx = aiCoreIdx / (kvHeadNum * splitKVNum);
        n2Idx = (aiCoreIdx / splitKVNum) % kvHeadNum;
        s2IdxFD = aiCoreIdx % splitKVNum;
    } else {
        bIdx = (beforeBlockPreSplitBn2Nums + bn2Idx) / kvHeadNum;
        n2Idx = (beforeBlockPreSplitBn2Nums + bn2Idx) % kvHeadNum;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        if constexpr (SHARED_PREFIX_PRE) {
            SysPrefixInitAllZeroOutput();
        } else {
            InitAllZeroOutput(bIdx, n2Idx);
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::UpdateInnerLoopCond()
{
    if (curActualPreSeqLen == 0) {
        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    int32_t remainSPreinnerSize = (int32_t)curActualPreSeqLen;
    int32_t computeSinnerSize = (int32_t)curActualPreSeqLen;
    if (flashDecodeFlag) {
        remainSPreinnerSize = (int32_t)curActualPreSeqLen - sInnerLoopSize * s2IdxFD;
        computeSinnerSize = remainSPreinnerSize >= sInnerLoopSize ? sInnerLoopSize : remainSPreinnerSize;
        if (aiCoreIdx >= batchSize * kvHeadNum * splitKVNum) {
            remainSPreinnerSize = 0;
        }
    }
    if (remainSPreinnerSize > 0) {
        if (computeSinnerSize <= singlePreProcessSInnerSize) {
            singlePreProcessSInnerSizeTail = computeSinnerSize;
            sInnerPreLoopTimes = 1;
        } else {
            sInnerPreLoopTimes = (computeSinnerSize + singlePreProcessSInnerSize - 1) / singlePreProcessSInnerSize;
            singlePreProcessSInnerSizeTail = computeSinnerSize - (sInnerPreLoopTimes - 1) * singlePreProcessSInnerSize;
        }
    } else {
        sInnerPreLoopTimes = 0;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenPreload<IFAT>::SeqLenFromTensorList(uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyPreTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyPreTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyPreTensorDesc, bIndex);
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        return keyPreTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyPreTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CalculateSUnitSize()
{
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        sPreUnitSize = kvHeadNum * headDim;
    } else {
        sPreUnitSize = headDim;
    }
    return;
}

template <typename IFAT>
__aicore__ inline bool IncreFlashAttentionAttenPreload<IFAT>::ComputeKVPaddingBeginOffset()
{
    if (kvPaddingPreFlag != 1) {
        return true;
    }
    int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }

    int64_t startPrePosition = kvSeqSize - paddingSize - curActualPreSeqLen;

    if (startPrePosition < 0) {
        return false;
    }

    kvPaddingBeginOffset = static_cast<uint64_t>(startPrePosition);
    return true;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitPrefix(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const IncreFlashAttentionTilingDataPrefix *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    sysPrefixLen = tiling->prefixLen;
    formerCoreNumSp = tiling->formerCoreNum;
    blockSplitBn2RangeSp = tiling->blockSplitBn2Range;
    tailBlockSplitBn2RangeSp = tiling->tailSplitedBatchRange;
    usedPreCoreNumSp = tiling->usedCoreNum;
    batchSizeQ = tiling->batchSizeQ;

    sysPrefixAttenOutGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->prefixAttenOutOffset));
    usrPromptAttenOutGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->userPromptAttenOutOffset));
    lseSumGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->tmpLseOffset));
    lseMaxGm.SetGlobalBuffer((__gm__ T *)(workspace + tiling->tmpLseOffset +
                                          2 * batchSizeQ * tiling->base.baseParams.qHeadNum * BYTE_BLOCK));

    Init(query, key, value, pseShift, attenMask, actualSeqLengths, blockTable, kvPaddingSize, attentionOut, softmaxLse,
         workspace, &tiling->base, gmTiling, tPipe, true);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const IncreFlashAttentionTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix)
{
    if ASCEND_IS_AIV {
        tmpPreBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpPreBlockIdx / 2;
    } else {
        tmpPreBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpPreBlockIdx;
    }
    // init tiling data
    tilingData = tiling;
#ifdef USE_MM_API
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tiling->bmm1TilingData, tPipe);
        bmm2.SetSubBlockIdx(0);
        bmm2.Init(&tiling->bmm2TilingData, tPipe);
    }
#endif
    InitTilingData();
    // 初始化计算参数
    if (flashDecodeFlag) {
        InitCalcParams();
    } else {
        InitCalcParamsEach();
    }

    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;
    blocktablePrePtr = blockTable;
    calcSysPrefixFlag = isPrefix;
    curSinglePreProcessSInnerSizeAlign = 0ULL; // prefix场景计算user prompt前必须重新初始化
    actualSinglePreProcessSInnerSize = 0ULL;
    actualSingleProcessSInnerSizeAlign = 0ULL;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    if (batchPreContinuous) {
        InitKeyGm(0);
        InitValueGm(0);
    }

    if (pipe != nullptr) {
        InitBuffers();
    }

    if (attenMaskPreFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }

    InitActualSeqLen(actualSeqLengths);

    if (kvPaddingPreFlag == 1) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t *)kvPaddingSize);
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    if ASCEND_IS_AIV {
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            softmaxMaxUb[i] = softmaxMaxBuff[i].Get<T>();
            softmaxSumUb[i] = softmaxSumBuff[i].Get<T>();
            softmaxExpUb[i] = softmaxExpBuff[i].Get<T>();
        }
        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
    }

    uint64_t offset = 0;
    if constexpr (ANTIQUANT_PRE) {
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

    vec2ResGm.SetGlobalBuffer(
            (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio* bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(T);

    // GM for pse
    if (pseShiftPreFlag) {
        pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType *)pseShift);
    }

    if (flashDecodeFlag) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if (softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }

    if constexpr (PAGE_ATTENTION) {
        // dcci cacheline 64B 对齐
        bmm1CallBackDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(workspace + offset + tmpPreBlockIdx * mmPACallBackDataSize));
        bmm1CallBackDataPtr = (__gm__ uint32_t *)(workspace + offset + tmpPreBlockIdx * mmPACallBackDataSize);
        offset = offset + GetBlockNum() * GetTaskRation() * mmPACallBackDataSize;

        bmm2CallBackDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(workspace + offset + tmpPreBlockIdx * mmPACallBackDataSize));
        bmm2CallBackDataPtr = (__gm__ uint32_t *)(workspace + offset + tmpPreBlockIdx * mmPACallBackDataSize);
        offset = offset + GetBlockNum() * GetTaskRation() * mmPACallBackDataSize;
    }

    if constexpr (SHARED_PREFIX_PRE) {
        if (isPrefix) {
            if constexpr (ANTIQUANT_PRE) {
                size_t blockSize = gSize * BYTE_BLOCK * batchSizeQ;
                msdRowMax1Gm.SetGlobalBuffer((__gm__ T *)(workspace + offset + tmpPreBlockIdx * blockSize * 4));
                msdRowMax2Gm = msdRowMax1Gm[blockSize / sizeof(T)];
                msdRowSum1Gm = msdRowMax1Gm[2 * blockSize / sizeof(T)];
                msdRowSum2Gm = msdRowMax1Gm[3 * blockSize / sizeof(T)];
                offset = offset + GetBlockNum() * GetTaskRation() * blockSize * 4;
                msdRowPreMaxSize = gSize * BYTE_BLOCK / sizeof(T);
                msdRowPreSumSize = msdRowPreMaxSize;
            }

            size_t blockSize = gSize * BYTE_BLOCK * batchSizeQ;
            softmaxRowMaxGm.SetGlobalBuffer((__gm__ T *)(workspace + offset + tmpPreBlockIdx * blockSize * 3));
            softmaxRowSumGm = softmaxRowMaxGm[blockSize / sizeof(T)];
            softmaxRowExpGm = softmaxRowMaxGm[2 * blockSize / sizeof(T)];
            offset = offset + GetBlockNum() * GetTaskRation() * blockSize * 3;
            softmaxMaxSumPreExpSize = gSize * BYTE_BLOCK / sizeof(T);

            if constexpr (!ANTIQUANT_PRE) {
                size_t blockSize = batchSizeQ * gSize * headDimAlign * sizeof(Q_T);
                prefixQueryPreProcessResGm.SetGlobalBuffer(
                    (__gm__ Q_T *)(workspace + offset + tmpPreBlockIdx * blockSize));
                offset = offset + GetBlockNum() * GetTaskRation() * blockSize;
            }
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitKeyGm(uint32_t bIdx)
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
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitValueGm(uint32_t bIdx)
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
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitQuant(
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *workspace)
{
    InitAntiquant(antiquantScale, antiquantOffset, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset);
    InitPostQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitAntiquant(
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset)
{
    if ASCEND_IS_AIC {
        return;
    }
    if constexpr (ANTIQUANT_PRE) {
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
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            aMaxBmm1Ub[i] = qAmaxBuff[i].Get<T>();
            aMaxBmm2Ub[i] = softmaxResAmaxBuff[i].Get<T>();
        }
        if constexpr (ANTIQUANT_PER_TOKEN) {
            for (int i = 0; i < PRE_LOAD_NUM; i++) {
                qRowSumUb[i] = qRowSumBuff[i].Get<T>();
                softmaxScaleResRowSumUb[i] = softmaxResRowSumBuff[i].Get<T>();
            }
        } else if constexpr (ANTIQUANT_PER_CHANNEL) {
            antiqScaleUb = antiqScaleBuff.Get<T>();
            antiqOffsetUb = antiqOffsetBuff.Get<T>();
        }
    }
}
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitPostQuant(__gm__ uint8_t *deqScale1,
    __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
{
    if constexpr (POST_QUANT) {
        if (!isPerChnU8Out && !isPreOutQuantTypeBf16) {
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
        if (quantScale2 != nullptr && !isPerChnU8Out && isPreOutQuantTypeBf16) {
            quantScale2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantScale2);
            quantScale2Value = ToFloat(quantScale2Bf16Gm.GetValue(0));
        }
        if (!isPerChnU8Out && isPreOutQuantTypeBf16) {
            if (quantOffset2 != nullptr) {
                quantOffset2Bf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)quantOffset2);
                quantOffset2Value = ToFloat(quantOffset2Bf16Gm.GetValue(0));
            } else {
                quantOffset2Value = 0;
            }
        }

        if (isPerChnU8Out && !isPreOutQuantTypeBf16) {
            if (quantScale2 != nullptr) {
                quantScale2Gm.SetGlobalBuffer((__gm__ float *)quantScale2);
            }
            if (quantOffset2 != nullptr) {
                isQuantOffset2Exist = true;
                quantOffset2Gm.SetGlobalBuffer((__gm__ float *)quantOffset2);
            }
        }

        if (isPerChnU8Out && isPreOutQuantTypeBf16) {
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

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitCalcParams()
{
    bn2LoopTimes = tilingData->increFlashAttentionSingleCoreParams.blockSplitBn2Range;
    beforeBlockPreSplitBn2Nums = aiCoreIdx * tilingData->increFlashAttentionSingleCoreParams.blockSplitBn2Range;
    // tail core
    if (aiCoreIdx >= formerPreCoreNum) {
        bn2LoopTimes = tilingData->increFlashAttentionSingleCoreParams.tailSplitedBatchRange;
        beforeBlockPreSplitBn2Nums =
            formerPreCoreNum * tilingData->increFlashAttentionSingleCoreParams.blockSplitBn2Range +
            (aiCoreIdx - formerPreCoreNum) * tilingData->increFlashAttentionSingleCoreParams.tailSplitedBatchRange;
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个increFlashAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间。
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *corePreSidxEnd = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
#else
    uint32_t corePreSidxEnd[50];
    copy_data_align64((uint8_t *)corePreSidxEnd, (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreSidxEnd),
                      sizeof(corePreSidxEnd));
#endif
    bn2LoopTimes = corePreSidxEnd[aiCoreIdx + 1] - corePreSidxEnd[aiCoreIdx];
    beforeBlockPreSplitBn2Nums = corePreSidxEnd[aiCoreIdx];
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CalcBN2Offset()
{
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        // B,1,N2,G,D
        tensorACorePreOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        // B,S2,N2,D
        tensorBCorePreOffset =
            bIdx * kvSeqSize * kvHeadNum * headDim + n2Idx * headDim + kvPaddingBeginOffset * kvHeadNum * headDim;

        if (!batchPreContinuous) {
            tensorBCorePreOffset = n2Idx * headDim;
        }

        if (flashDecodeFlag) {
            tensorBCorePreOffset += s2Idx * sInnerLoopSize * kvHeadNum * headDim;
        }
    } else {
        tensorACorePreOffset = bIdx * qHeadNum * headDim + n2Idx * gSize * headDim;
        // B,N2,S2,D
        tensorBCorePreOffset =
            bIdx * kvHeadNum * kvSeqSize * headDim + n2Idx * kvSeqSize * headDim + kvPaddingBeginOffset * headDim;

        if (!batchPreContinuous) {
            uint64_t seqSize = SeqLenFromTensorList(bIdx);
            tensorBCorePreOffset = n2Idx * seqSize * headDim;
        }
        if (flashDecodeFlag) {
            tensorBCorePreOffset += s2Idx * sInnerLoopSize * headDim;
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CalcBN2Params()
{
    attenMaskCoreOffset = bIdx * attenMaskSize + kvPaddingBeginOffset;
    if (flashDecodeFlag) {
        attenMaskCoreOffset += s2Idx * sInnerLoopSize;
    }
    // antiquant的offset和scale参数数据排列是先key后value
    if (antiquantPerTensorFlag == 1) {
        antiqPreParamOffset = 0;
    } else {
        antiqPreParamOffset = n2Idx * headDim;
    }
    antiqKeyParamCoreOffsetPerToken = bIdx * antiqSeqSize + kvPaddingBeginOffset;
    if (antiquantPerHeadFlag) {
        antiqKeyParamCoreOffsetPerToken = bIdx * antiqSeqSize * kvHeadNum + kvPaddingBeginOffset +
                                          n2Idx * antiqSeqSize;
    }
    if (flashDecodeFlag) {
        antiqKeyParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
    }
    if (antiquantPerHeadFlag) {
        antiqPreParamOffset = n2Idx;
    }
    // out quant
    perChannelQuantOffset = n2Idx * headDim * gSize;
    if (!batchPreContinuous) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
        ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
        __gm__ uint8_t *key = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        __gm__ uint8_t *value = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
        if (tilingData->baseParams.l2CacheOffFlag) {
            // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
            keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
    }
    // 更新actualSingleProcessSInnerSize，防止尾块值，影响第二次loop
    actualSinglePreProcessSInnerSize = singlePreProcessSInnerSize;
    if constexpr (KVINT4) {
        actualSingleProcessSInnerSizeAlign = Align(singlePreProcessSInnerSize, 64UL);
    } else {
        actualSingleProcessSInnerSizeAlign = Align(singlePreProcessSInnerSize, BYTE_BLOCK);
    }

    if (pseShiftPreFlag) {
        pseShiftCoreOffset = (pseShiftB == 1) ? (n2Idx * gSize * pseShiftS) : (bIdx * qHeadNum * pseShiftS + n2Idx * gSize * pseShiftS);
        if (flashDecodeFlag) {
            pseShiftCoreOffset += s2Idx * sInnerLoopSize;
        }
        pseShiftCoreOffset += kvPaddingBeginOffset; // kv_padding_size
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::UpdateOffsetsVec(uint32_t sInnerLoopIdx)
{
    // antiquant的offset和scale参数数据排列是先key后value
    if (antiquantPerTensorFlag == 1) {
        antiqPreParamOffset = 0;
    } else {
        antiqPreParamOffset = n2Idx * headDim;
    }
    antiqKeyParamCoreOffsetPerToken = bIdx * antiqSeqSize + kvPaddingBeginOffset;
    if (antiquantPerHeadFlag) {
        antiqPreParamOffset = n2Idx;
        antiqKeyParamCoreOffsetPerToken = bIdx * kvHeadNum * antiqSeqSize + n2Idx * antiqSeqSize +
                                          kvPaddingBeginOffset;
    }
    if (flashDecodeFlag) {
        antiqKeyParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
    }
    // out quant
    perChannelQuantOffset = n2Idx * headDim * gSize;

    if (pseShiftPreFlag) {
        if (pseShiftB == 1) {
            pseShiftCoreOffset = n2Idx * gSize * pseShiftS;
        } else {
            pseShiftCoreOffset = bIdx * qHeadNum * pseShiftS + n2Idx * gSize * pseShiftS;
        }
        if (flashDecodeFlag) {
            pseShiftCoreOffset += s2Idx * sInnerLoopSize;
        }
    }

    uint64_t sInnerOffsetDataSize = sInnerLoopIdx * singlePreProcessSInnerSize;

    attenMaskCoreOffset = bIdx * attenMaskSize; // 前缀不用考虑左kvpadding
    if (flashDecodeFlag) {
        attenMaskCoreOffset += s2Idx * sInnerLoopSize;
    }
    attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;
    antiqParamOffsetPerToken = antiqKeyParamCoreOffsetPerToken + sInnerOffsetDataSize;

    // pse offset
    if (pseShiftPreFlag) {
        pseShiftOffset = pseShiftCoreOffset + sInnerOffsetDataSize;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::AttenMaskCopyIn(uint64_t offset,
                                                                                     uint32_t dealRowCount,
                                                                                     uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    attenMaskPreSizeAlign = Align(actualColumnCount, 32U);
    maskUb.SetSize(dealRowCount * attenMaskPreSizeAlign);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 32 == 0) {
        DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskPreSizeAlign);
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(bool);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualColumnCount * sizeof(bool);
        intriParams.blockCount = 1;
        intriParams.dstStride = (attenMaskPreSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<bool> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (attenMaskPreSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        DataCopyPad(maskUb, attenMaskBoolGm[offset], intriParams, padParams);
    }
#else
    DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskPreSizeAlign);
#endif
    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyAntiquantScale(LocalTensor<T> &castUb,
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
        uint32_t qPreTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<Q_T> copyInPadParams;
        // antiq scale copy in
        copyInParams.blockCount = 1;
        copyInParams.blockLen = headDim * sizeof(Q_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (headDimAlign - headDim) / qPreTypeElementSize;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (headDimAlign - headDim) % qPreTypeElementSize;
        copyInPadParams.paddingValue = 0;

        LocalTensor<Q_T> inputPreUb = inputQue2.AllocTensor<Q_T>();
        DataCopyPad(inputPreUb, srcGm[offset], copyInParams, copyInPadParams);
        inputQue2.template EnQue(inputPreUb);

        inputPreUb = inputQue2.DeQue<Q_T>();
        Cast(castUb, inputPreUb, RoundMode::CAST_NONE, headDim);
        inputQue2.FreeTensor(inputPreUb);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyAntiquantParamsPerTokenHead(
    GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset, uint32_t columnCount)
{
    LocalTensor<ANTIQ_PARAMS_T> dstUb = inputQue1.AllocTensor<ANTIQ_PARAMS_T>();
    DataCopy(dstUb, srcGm[offset], columnCount);
    inputQue1.template EnQue(dstUb);
}


template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyAntiquantParamsPerTokenParamsPagedAttention(
    GlobalTensor<ANTIQ_PARAMS_T> srcGm, uint64_t offset, uint32_t actualColumnCount)
{
    uint64_t kvCacheBlockSize = tilingData->baseParams.blockSize;
    uint32_t maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    uint32_t paramsTypeElementSize = BYTE_BLOCK / sizeof(ANTIQ_PARAMS_T);
    __gm__ int32_t *blockTableGmAddr = reinterpret_cast<__gm__ int32_t *>(blocktablePrePtr);

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.paddingValue = 0;

    uint64_t allBlockBaseIndex = bIdx * maxBlockNumPerBatch;
    uint64_t dstOffset = 0;
    uint32_t copyFinishElmeCnt = 0;
    uint64_t curSeqIdx = offset - bIdx * kvSeqSize;
    LocalTensor<ANTIQ_PARAMS_T> paramsUb = inputQue1.AllocTensor<ANTIQ_PARAMS_T>();

    while (copyFinishElmeCnt < actualColumnCount) {
        uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize;
        uint64_t reaminPreElemCnt = curSeqIdx % kvCacheBlockSize;
        uint32_t blockId = *(blockTableGmAddr + allBlockBaseIndex + blockIdOffset);
        uint32_t copyElemCnt = kvCacheBlockSize - reaminPreElemCnt;
        if (copyFinishElmeCnt + copyElemCnt > actualColumnCount) {
            copyElemCnt = actualColumnCount - copyFinishElmeCnt;
        }
        uint32_t copyElemCntAilgin = Align(copyElemCnt, paramsTypeElementSize);

        copyInPadParams.rightPadding = copyElemCntAilgin - copyElemCntAilgin;
        copyInParams.blockLen = copyElemCnt * sizeof(ANTIQ_PARAMS_T);
        uint64_t srcOffst = blockId * kvCacheBlockSize + reaminPreElemCnt;
        // 此处需注意paramsUb偏移量的合法性（UB地址需要32Byte对齐）
        // 在 (blockSize % sInnerSize) * sizeof(ANTIQ_PARAMS_T)不是32对齐时，可能出现非法偏移
        // 当前blockSize为16对齐，sInnerSize为1024对齐，ANTIQ_PARAMS_T为FP32，不会出现非法偏移
        // 后续上述三个元素限制有改动时，需要小心此处
        DataCopyPad(paramsUb[dstOffset], srcGm[srcOffst], copyInParams, copyInPadParams);

        dstOffset += copyElemCnt;
        copyFinishElmeCnt += copyElemCnt;
        curSeqIdx += copyElemCnt;
    }
    inputQue1.template EnQue(paramsUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyAntiquantParamsPerToken(
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

    LocalTensor<ANTIQ_PARAMS_T> paramsUb = inputQue1.AllocTensor<ANTIQ_PARAMS_T>();
    DataCopyPad(paramsUb, srcGm[offset], copyInParams, copyInPadParams);
    inputQue1.template EnQue(paramsUb);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::CopyAntiqQuery(LocalTensor<T> &queryCastUb, uint64_t qOffset,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    uint32_t qPreTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = actualColumnCount * sizeof(Q_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (columnCount - actualColumnCount) / qPreTypeElementSize;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % qPreTypeElementSize;
    copyInPadParams.paddingValue = 0;

    LocalTensor<Q_T> inputPreUb = inputQue1.AllocTensor<Q_T>();
    DataCopyPad(inputPreUb, queryGm[qOffset], copyInParams, copyInPadParams);
    inputQue1.template EnQue(inputPreUb);

    inputPreUb = inputQue1.DeQue<Q_T>();
    Cast(queryCastUb, inputPreUb, RoundMode::CAST_NONE, dealRowCount * columnCount);
    inputQue1.FreeTensor(inputPreUb);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb,
                                                        LocalTensor<T> tmpAUb, uint32_t dealRowCount,
                                                        uint32_t columnCount, uint32_t actualColumnCount)
{
    Abs(tmpAUb, srcUb, dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpPreRowMaxUb = tmpBuff3.Get<T>();
    RowMaxForLongColumnCount(tmpPreRowMaxUb, tmpAUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    Brcb(tmpAMaxRes, tmpPreRowMaxUb, (dealRowCount + 7) / 8, {1, 8});
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::AntiquantAIterExpand(GlobalTensor<KV_T> dstGm, LocalTensor<T> &tmpA1,
                                                                   LocalTensor<T> &tmpA2, uint32_t calcSize,
                                                                   bool isFirst, uint64_t outOffset)
{
    if (!isFirst) {
        Sub(tmpA1, tmpA1, tmpA2, calcSize);
        PipeBarrier<PIPE_V>();
        Muls(tmpA1, tmpA1, antiquantPreExpandCoeff, calcSize);
        PipeBarrier<PIPE_V>();
    }

    Cast(tmpA2, tmpA1, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    // cast-fp16
    LocalTensor<half> aPreResOutUb = outputQue1.template AllocTensor<half>();
    Cast(aPreResOutUb, tmpA2, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    // cast-int8
    LocalTensor<KV_T> aResOutUbI8 = aPreResOutUb.template ReinterpretCast<KV_T>();
    aResOutUbI8.SetSize(aPreResOutUb.GetSize());
    Cast(aResOutUbI8, aPreResOutUb, RoundMode::CAST_ROUND, calcSize);

    // copyOut Ak
    outputQue1.template EnQue(aResOutUbI8);
    outputQue1.template DeQue<KV_T>();
    if constexpr (KVINT4) {
        GlobalTensor<int8_t> dstGmI8;
        dstGmI8.SetGlobalBuffer(((__gm__ int8_t *)(dstGm[outOffset].GetPhyAddr())));
        DataCopy<int8_t>(dstGmI8, aResOutUbI8.template ReinterpretCast<int8_t>(), calcSize >> 1);
    } else {
        DataCopy(dstGm[outOffset], aResOutUbI8, calcSize);
    }
    outputQue1.FreeTensor(aResOutUbI8);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::AntiquantMatmulPreProcess(
    GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = gSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    LocalTensor<T> tmpAMaxRes = aMaxResUb[startRow * PRE_BLOCK_ELEMENT_NUM];
    AbsRowMax(tmpAMaxRes, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // 128/(1.001*Amax)*A
    Duplicate(tmpAFloorUb, antiqCoeff1, dealRowCount * PRE_BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    Div(tmpAFloorUb, tmpAFloorUb, tmpAMaxRes, dealRowCount * PRE_BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    RowMuls(srcUb, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < msdPreIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUb, tmpAFloorUb, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::AntiquantSoftmaxResPreProcess(
    GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t stepPre = gSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    Muls(srcUb, srcUb, antiqCoeff1, calcSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < msdPreIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUb, tmpAFloorUb, calcSize, (i == 0 ? true : false), stepPre * i + baseOffset);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::DealQueryPreProcessBaseBlock(
    uint32_t loop, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint64_t qOffset = info.bIdx * qHeadNum * headDim + info.n2Idx * gSize * headDim +
        (gSizeStart + startRow) * actualColumnCount;

    uint32_t bufferPreId = info.bn2IdxInCurCore % PRE_LOAD_NUM;
    LocalTensor<T> queryUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();
    CopyAntiqQuery(queryUb, qOffset, dealRowCount, columnCount, actualColumnCount);

    PipeBarrier<PIPE_V>();
    // mul scale
    VecMulMat(queryUb, antiqScaleUb, queryUb, dealRowCount, columnCount, actualColumnCount);

    PipeBarrier<PIPE_V>();

    // A pre process
    size_t dstOffset = bufferPreId * bmm2ResUbSize + gSizeStart * columnCount;
    AntiquantMatmulPreProcess(queryPreProcessResGm[dstOffset], aMaxBmm1Ub[bufferPreId],
        queryUb, aFloorUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::DealQueryPreProcessBaseBlockPerToken(uint32_t loop,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];

    uint32_t baseOffset = startRow * PRE_BLOCK_ELEMENT_NUM;
    uint64_t qOffset = info.bIdx * qHeadNum * headDim + info.n2Idx * gSize * headDim +
        (gSizeStart+ startRow) * actualColumnCount;

    uint32_t bufferPreId = info.bn2IdxInCurCore % PRE_LOAD_NUM;
    LocalTensor<T> queryProcessUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();
    CopyAntiqQuery(queryProcessUb, qOffset, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    if (antiqOffsetExistFlag) {
        LocalTensor<T> tmpRowSumUb = tmpBuff3.Get<T>();
        Adds(aFloorUb, queryProcessUb, (T)0, dealRowCount * columnCount); // queryUb数据需要保留
        PipeBarrier<PIPE_V>();
        RowSum(tmpRowSumUb, aFloorUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Brcb(qRowSumUb[bufferPreId][baseOffset], tmpRowSumUb, (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
    }

    // A pre process
    size_t dstOffset = bufferPreId * bmm2ResUbSize + gSizeStart * columnCount;
    AntiquantMatmulPreProcess(queryPreProcessResGm[dstOffset], aMaxBmm1Ub[bufferPreId],
        queryProcessUb, aFloorUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::QueryPreProcessInner(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];

    if (!info.isFirstSInnerLoop) {
        return;
    }

    if constexpr (ANTIQUANT_PER_CHANNEL) {
        CopyAntiquantScale(antiqScaleUb, keyAntiqScaleGm, info.antiqKeyParamOffset);
        if (softmaxLseFlag && antiqOffsetExistFlag) {
            CopyAntiquantScale(antiqOffsetUb, keyAntiqOffsetGm, info.antiqKeyParamOffset);
        }
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gPreSplitSize > gSizeVector) {
        gPreSplitSize = gSizeVector;
    }
    uint32_t loopCount = (gSizeVector + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = gSizeVector - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT_PER_CHANNEL) {
            DealQueryPreProcessBaseBlock(loop, i * gPreSplitSize, dealSize, headDimAlign, headDim);
        } else if constexpr (ANTIQUANT_PER_TOKEN) {
            DealQueryPreProcessBaseBlockPerToken(loop, i * gPreSplitSize, dealSize, headDimAlign, headDim);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMax = inputQue1.AllocTensor<T>();
    LocalTensor<T> lseSum = inputQue2.AllocTensor<T>();

    combineLseOffset = (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize * FP32_ONE_BLOCK_SIZE +
                       startRow * FP32_ONE_BLOCK_SIZE;
    for (uint32_t k = 0; k < actualPreCombineLoopSize; k++) {
        DataCopy(lseSum[k * dealRowCount * FP32_ONE_BLOCK_SIZE],
                 lseSumFdGm[combineLseOffset + k * gSize * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        DataCopy(lseMax[k * dealRowCount * FP32_ONE_BLOCK_SIZE],
                 lseMaxFdGm[combineLseOffset + k * gSize * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    inputQue2.EnQue(lseSum);
    inputQue1.EnQue(lseMax);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyAccumOutIn(uint32_t splitKVIndex,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = inputQue1.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlign - headDim) / PRE_BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlign - headDim) % PRE_BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;

    combineAccumOutOffset =
        (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim + startRow * headDim;
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    inputQue1.EnQue(accumOutLocal);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax,
                                                                uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMaxUb = softmaxMaxUb[0];
    LocalTensor<T> lseSumUb = softmaxSumUb[0];
    LocalTensor<T> lseExpUb = tmpBuff1.Get<T>();

    // lseLocal的shape为[actualPreCombineLoopSize, dealRowCount * FP32_ONE_BLOCK_SIZE]
    Duplicate(lseMaxUb, -FLOAT_MAX, dealRowCount * FP32_ONE_BLOCK_SIZE);
    Duplicate(lseSumUb, FLOAT_ZERO, dealRowCount * FP32_ONE_BLOCK_SIZE);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < actualPreCombineLoopSize; ++i) {
        Max(lseMaxUb, lseMaxUb, lseMax[i * dealRowCount * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < actualPreCombineLoopSize; ++i) {
        Sub(lseExpUb[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseMax[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseMaxUb,
            dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    PipeBarrier<PIPE_V>();
    Exp(lseExpUb, lseExpUb, actualPreCombineLoopSize * dealRowCount * FP32_ONE_BLOCK_SIZE);
    PipeBarrier<PIPE_V>();

    Mul(lseSum, lseSum, lseExpUb, actualPreCombineLoopSize * dealRowCount * FP32_ONE_BLOCK_SIZE);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < actualPreCombineLoopSize; ++i) {
        Add(lseSumUb, lseSumUb, lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], dealRowCount * FP32_ONE_BLOCK_SIZE);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t i = 0; i < actualPreCombineLoopSize; ++i) {
        Div(lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseSum[i * dealRowCount * FP32_ONE_BLOCK_SIZE], lseSumUb,
            dealRowCount * FP32_ONE_BLOCK_SIZE);
    }
    PipeBarrier<PIPE_V>();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                                             uint32_t startRow, uint32_t dealRowCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    repeatParams.dstRepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    int32_t dtypePreMask = 256 / sizeof(float);
    int32_t mulPreLoop = headDimAlign / dtypePreMask;
    int32_t mulRemain = headDimAlign % dtypePreMask;

    // 第一次，mul结果直接放到dst里
    CopyAccumOutIn(0, startRow, dealRowCount);
    LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
    for (int i = 0; i < mulPreLoop; i++) {
        Mul(dst[i * dtypePreMask], lseLocal, accumOutLocal[i * dtypePreMask], dtypePreMask, dealRowCount, repeatParams);
    }
    if (mulRemain > 0) {
        Mul(dst[mulPreLoop * dtypePreMask], lseLocal, accumOutLocal[mulPreLoop * dtypePreMask], mulRemain, dealRowCount,
            repeatParams);
    }
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(accumOutLocal);

    for (uint32_t j = 1; j < actualPreCombineLoopSize; ++j) {
        CopyAccumOutIn(j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
        for (int i = 0; i < mulPreLoop; i++) {
            Mul(accumOutLocal[i * dtypePreMask], lseLocal[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[i * dtypePreMask], dtypePreMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(accumOutLocal[mulPreLoop * dtypePreMask], lseLocal[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[mulPreLoop * dtypePreMask], mulRemain, dealRowCount, repeatParams);
        }
        PipeBarrier<PIPE_V>();
        Add(dst, dst, accumOutLocal, dealRowCount * headDimAlign);
        PipeBarrier<PIPE_V>();
        inputQue1.FreeTensor(accumOutLocal);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    if constexpr (!POST_QUANT) {
        LocalTensor<OUT_T> tmpPreBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
        tmpPreBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpPreBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * headDimAlign);
        } else {
            Cast(tmpPreBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * headDimAlign);
        }

        outputQue1.EnQue(tmpPreBmm2ResCastTensor);
        outputQue1.DeQue<OUT_T>();
        Bmm2DataCopyOut(attenOutOffset, tmpPreBmm2ResCastTensor, startRow, dealRowCount, headDimAlign, headDim);
        outputQue1.FreeTensor(tmpPreBmm2ResCastTensor);
    } else {
        LocalTensor<OUT_T> bmmPre2ResUbInt8 = outputQue1.AllocTensor<OUT_T>();
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
        bmmPre2ResUbInt8.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        PostQuant(perChannelQuantOffset, accumOutLocal, bmmPre2ResUbInt8, startRow, dealRowCount, headDimAlign, headDim);
        outputQue1.EnQue(bmmPre2ResUbInt8);
        outputQue1.DeQue<OUT_T>();
        Bmm2DataCopyOut(attenOutOffset, bmmPre2ResUbInt8, startRow, dealRowCount, headDimAlign, headDim);
        outputQue1.FreeTensor(bmmPre2ResUbInt8);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CombineSplitKVRes()
{
    if (curActualPreSeqLen != 0) {
        uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_16K / (BYTE_BLOCK * splitKVNum);
        uint32_t gSplitSizeAccumOut = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
        // 取两者较小的，用来切g，保证ub够用
        uint32_t gPreSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
        gPreSplitSize = (gPreSplitSize > gSize) ? gSize : gPreSplitSize; // 最大为gSize
        uint32_t loopCount = (gSize + gPreSplitSize - 1) / gPreSplitSize;
        uint32_t tailSplitSize = gSize - (loopCount - 1) * gPreSplitSize;

        // 尾块与非尾块都使用这些ub，减少处理次数
        for (uint32_t i = 0, actualGSplitSize = gPreSplitSize; i < loopCount; i++) {
            uint32_t startRow = i * gPreSplitSize;
            if ((i + 1) == loopCount) {
                actualGSplitSize = tailSplitSize;
            }
            CopyLseIn(startRow, actualGSplitSize);
            LocalTensor<T> lseSum = inputQue2.DeQue<T>();
            LocalTensor<T> lseMax = inputQue1.DeQue<T>();
            ComputeScaleValue(lseSum, lseMax, startRow, actualGSplitSize);
            inputQue1.FreeTensor(lseMax);

            uint32_t gSplitBmm2UbSize = headDimAlign * actualGSplitSize;
            LocalTensor<T> tmp1 = tmpBuff1.Get<T>(gSplitBmm2UbSize);
            ReduceFinalRes(tmp1, lseSum, startRow, actualGSplitSize);
            inputQue2.FreeTensor(lseSum);

            if constexpr (SHARED_PREFIX_PRE) {
                SysPrefixSaveAttenRes(bIdx, n2Idx, tmp1, startRow, actualGSplitSize, calcSysPrefixFlag);
            } else {
                CopyFinalResOut(tmp1, startRow, actualGSplitSize);
            }
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::FlashDecodeCompute()
{
    bIdx = tmpPreBlockIdx / kvHeadNum;
    n2Idx = tmpPreBlockIdx % kvHeadNum;
    attenOutOffset = bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
    perChannelQuantOffset = n2Idx * headDim * gSize;
    gSizeStart = 0;
    if (tmpPreBlockIdx >= batchSize * kvHeadNum) {
        return;
    }

    if (actualLenDims == 0) {
        curActualPreSeqLen = kvSeqSize;
        if (!batchPreContinuous) {
            curActualPreSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualPreSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualPreSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }

    actualPreCombineLoopSize = (curActualPreSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;

    if (SHARED_PREFIX_PRE) {
        if (calcSysPrefixFlag) {
            for (bIdx = 0; bIdx < batchSizeQ; bIdx++) {
                CombineSplitKVRes();
            }
            return;
        }
    }

    CombineSplitKVRes();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::ComputeLogSumExpAndCopyToGm(LocalTensor<T> &softmaxSumUb,
                                                                          LocalTensor<T> &softmaxMaxUb)
{
    size_t size = gSizeVector * FP32_ONE_BLOCK_SIZE;
    size_t offset = bIdx * kvHeadNum * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE +
                    n2Idx * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE +
                    s2IdxFD * gSize * FP32_ONE_BLOCK_SIZE + gSizeStart * FP32_ONE_BLOCK_SIZE;
    // lseSumFdGm: batchQ * kvHeadNum * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE
    CopyFixedUbToGm(lseSumFdGm[offset], softmaxSumUb, size);
    CopyFixedUbToGm(lseMaxFdGm[offset], softmaxMaxUb, size);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SoftmaxLseCopyOut(LocalTensor<T> &softmaxSumUb,
                                                                                       LocalTensor<T> &softmaxMaxUb)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::Bmm1Compute(const uint32_t bn2Idx,
                                                                                 const uint32_t sInnerLoopIdx)
{
    if (SHARED_PREFIX_PRE) {
        if (calcSysPrefixFlag) {
            SysPrefixBmm1Compute(bn2Idx, sInnerLoopIdx);
            return;
        }
    }
    Bmm1ComputeCommon(bn2Idx, sInnerLoopIdx);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::Bmm2Compute(const uint32_t bn2Idx,
                                                                                 const uint32_t sInnerLoopIdx)
{
    if (SHARED_PREFIX_PRE) {
        if (calcSysPrefixFlag) {
            SysPrefixBmm2Compute(bn2Idx, sInnerLoopIdx);
            return;
        }
    }
    Bmm2ComputeCommon(bn2Idx, sInnerLoopIdx);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::ElewiseCompute(uint32_t loop, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();

    // pse shift mask
    if (pseShiftPreFlag) {
        PseShiftCopyIn(loop, startRow, dealRowCount, actualColumnCount);
        LocalTensor<pseShiftType> pseShiftUb = inputQue1.DeQue<pseShiftType>();
        LocalTensor<float> pseShiftUbFloat = tmpBuf.Get<float>();
        for (uint32_t i = 0; i < dealRowCount; ++i) {
            Cast(pseShiftUbFloat[i * columnCount], pseShiftUb[i * pseMaskPreSizeAlign], AscendC::RoundMode::CAST_NONE,
                 pseMaskPreSizeAlign);
        }

        inputQue1.FreeTensor(pseShiftUb);
        PipeBarrier<PIPE_V>();
        Add(mmResUb, mmResUb, pseShiftUbFloat, dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
        pseShiftOffset = info.pseShiftCoreOffset + dealRowCount * columnCount;
    }

    // attenMask
    if (attenMaskPreFlag == 1) {
        AttenMaskCopyIn(info.attenMaskOffset, dealRowCount, actualColumnCount);
        LocalTensor<bool> attenMaskPreUb = inputQue2.DeQue<bool>();
        for (int i = 1; i < dealRowCount; i++) {
            DataCopy(attenMaskPreUb[i * attenMaskPreSizeAlign], attenMaskPreUb, attenMaskPreSizeAlign);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>(selectWithByteMaskTmpMinSize);
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskPreSizeAlign;
        attenMaskPreUb.SetSize(dealRowCount * attenMaskPreSizeAlign); // Select接口要求mask size与参数匹配
        mmResUb.SetSize(dealRowCount * columnCount);            // Select接口要求src size与参数匹配
        SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskPreUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
        mmResUb.SetSize(BUFFER_SIZE_BYTE_32K / sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
        inputQue2.FreeTensor(attenMaskPreUb);

        PipeBarrier<PIPE_V>();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SoftmaxFlashV2Compute(uint32_t loop,
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * PRE_BLOCK_ELEMENT_NUM;
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t outIdx = loop % (PRE_LOAD_NUM);
    if (extraInfo[loop % PRE_LOAD_NUM].isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (loop - 1) % (PRE_LOAD_NUM);
        inMaxTensor = softmaxMaxUb[inIdx][baseOffset];
        inSumTensor = softmaxSumUb[inIdx][baseOffset];
    }


    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb, softmaxSumUb[outIdx][baseOffset], softmaxMaxUb[outIdx][baseOffset], mmResUb, softmaxExpUb[outIdx][baseOffset],
        inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::Bmm2FDDataCopyOut(LocalTensor<T> &attenOutUb, uint32_t startRow,
                                                                uint32_t dealPreRowCount, uint32_t columnCount,
                                                                uint32_t actualPreColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealPreRowCount;
    dataCopyParams.blockLen = actualPreColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualPreColumnCount) / (BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    LocalTensor<T> tmpTensor = outputQue1.AllocTensor<T>();
    DataCopy(tmpTensor, attenOutUb, columnCount * dealPreRowCount);
    outputQue1.EnQue(tmpTensor);
    outputQue1.DeQue<T>();

    size_t base = (bIdx * qHeadNum + n2Idx * gSize) * splitKVNum * headDim;
    // accumOutGm: batchQ * kvHeadNum * gSize * kvSplitPart_ * headDimAlign_
    DataCopyPad(accumOutGm[base + s2IdxFD * gSize * actualPreColumnCount +
                           startRow * actualPreColumnCount + gSizeStart * actualPreColumnCount], tmpTensor, dataCopyParams);
    outputQue1.FreeTensor(tmpTensor);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (gSizeStart + startRow) * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::Bmm2CastAndCopyOut(ExtraInfo& info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                                 uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (flashDecodeFlag) {
            Bmm2FDDataCopyOut(bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
            return;
        }
    }

    if constexpr (SHARED_PREFIX_PRE) {
        SysPrefixSaveAttenRes(bIdx, n2Idx, bmm2ResUb, startRow, dealRowCount, calcSysPrefixFlag);
    } else {
        if constexpr (!POST_QUANT) {
            LocalTensor<OUT_T> tmpPreBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
            if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
                Cast(tmpPreBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
            } else {
                Cast(tmpPreBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
            }
            outputQue1.EnQue(tmpPreBmm2ResCastTensor);
            outputQue1.DeQue<OUT_T>();
            Bmm2DataCopyOut(info.attenOutOffset, tmpPreBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
            outputQue1.FreeTensor(tmpPreBmm2ResCastTensor);
        } else {
            LocalTensor<OUT_T> bmmPre2ResUbInt8 = outputQue1.AllocTensor<OUT_T>();
            PostQuant(info.perChannelQuantOffset, bmm2ResUb, bmmPre2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
            outputQue1.EnQue(bmmPre2ResUbInt8);
            outputQue1.DeQue<OUT_T>();
            Bmm2DataCopyOut(info.attenOutOffset, bmmPre2ResUbInt8, startRow, dealRowCount, columnCount, actualColumnCount);
            outputQue1.FreeTensor(bmmPre2ResUbInt8);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::PseShiftCopyIn(uint32_t loop,  uint32_t startRow,
                                                                                    uint32_t rowCount,
                                                                                    uint32_t actualPreColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    pseShiftOffset = info.pseShiftCoreOffset + gSizeStart * pseShiftS;
    LocalTensor<pseShiftType> pseShiftUb = inputQue1.AllocTensor<pseShiftType>();
    pseMaskPreSizeAlign = Align(actualPreColumnCount, 16U); // 16: align to 32bytes
    uint32_t computeSize = rowCount * pseMaskPreSizeAlign;
    pseShiftUb.SetSize(computeSize);
#if (__CCE_AICORE__ > 200)
    if (actualPreColumnCount % 16 == 0) {
        for (uint32_t i = 0; i < rowCount; ++i) {
            DataCopy(pseShiftUb[i * pseMaskPreSizeAlign],
                     pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS], pseMaskPreSizeAlign);
        }
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(pseShiftType);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualPreColumnCount * sizeof(pseShiftType);
        intriParams.blockCount = 1;
        intriParams.dstStride = (pseMaskPreSizeAlign - actualPreColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<pseShiftType> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (pseMaskPreSizeAlign - actualPreColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        for (uint32_t i = 0; i < rowCount; ++i) {
            DataCopyPad(pseShiftUb[i * pseMaskPreSizeAlign],
                        pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS], intriParams, padParams);
        }
    }
#else
    for (uint32_t i = 0; i < rowCount; ++i) {
        DataCopy(pseShiftUb[i * pseMaskPreSizeAlign], pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS],
                 pseMaskPreSizeAlign);
    }
#endif
    inputQue1.EnQue(pseShiftUb);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::DealBmm1ResBaseBlock(const uint32_t loop, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();

    size_t batchBase = 0;

    uint64_t inOutGmOffset = (loop % PRE_LOAD_NUM) * mmResUbSize + (gSizeStart + startRow) * columnCount;
    LocalTensor<MM_OUT_T> tmpMmResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpMmResUb, mm1ResGm[inOutGmOffset + batchBase], computeSize);
    inputQue1.EnQue(tmpMmResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(mmResUb, tmpMmResUb, computeSize);
    inputQue1.FreeTensor(tmpMmResUb);
    PipeBarrier<PIPE_V>();

    ElewiseCompute(loop, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(loop, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<KV_T> tmpMMResCastTensor = outputQue1.AllocTensor<KV_T>();
    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);

    outputQue1.EnQue(tmpMMResCastTensor);
    outputQue1.DeQue<KV_T>();
    DataCopy(vec1ResGm[inOutGmOffset + batchBase], tmpMMResCastTensor, computeSize);
    outputQue1.FreeTensor(tmpMMResCastTensor);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::AntiquantMatmulResCombine(
    LocalTensor<T> bmmResUb, GlobalTensor<MM_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = gSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;

    if constexpr (SHARED_PREFIX_PRE) {
        if (calcSysPrefixFlag) {
            baseOffset += bIdx * gSize * msdPreIterNum * columnCount;
        }
    }

    T scale = 1;
    uint32_t offset = baseOffset;
    for (uint32_t i = 0; i < msdPreIterNum; i++) {
        LocalTensor<MM_OUT_T> tmpCInt = inputQue1.AllocTensor<MM_OUT_T>();
        DataCopy(tmpCInt, srcGm[offset], copySize); // offset = i * step + baseOffset
        inputQue1.template EnQue(tmpCInt);

        tmpCInt = inputQue1.DeQue<MM_OUT_T>();
        if (i == 0) {
            Cast(bmmResUb, tmpCInt, AscendC::RoundMode::CAST_NONE, copySize);
        } else {
            LocalTensor<T> tmpCFp;
            tmpCFp = tmpCInt.template ReinterpretCast<T>();
            tmpCFp.SetSize(tmpCInt.GetSize());
            Cast(tmpCFp, tmpCInt, AscendC::RoundMode::CAST_NONE, copySize);
            PipeBarrier<PIPE_V>();
            Muls(tmpCFp, tmpCFp, scale, copySize);
            PipeBarrier<PIPE_V>();
            Add(bmmResUb, bmmResUb, tmpCFp, copySize);
        }
        inputQue1.FreeTensor(tmpCInt);

        offset += step;
        scale = scale / antiquantPreExpandCoeff;
    }
    PipeBarrier<PIPE_V>();

    // muls 1/antiqCoeff1
    Muls(bmmResUb, bmmResUb, antiqCoeff2, copySize);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::DealAntiqBmm1ResBaseBlock(uint32_t loop, uint32_t startRow,
                                                                        uint32_t dealRowCount, uint32_t columnCount,
                                                                        uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];

    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    uint64_t srcGmOffset = (loop % PRE_LOAD_NUM) * mmResUbSize + gSizeStart * columnCount;
    AntiquantMatmulResCombine(mmResUb, mm1ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<T> aMax = aMaxBmm1Ub[info.bn2IdxInCurCore % (PRE_LOAD_NUM)][startRow * PRE_BLOCK_ELEMENT_NUM];
    RowMuls(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // mul scalar and mask
    ElewiseCompute(loop, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);
    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(loop, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    DealKvInt4ColumnOdd(mmResUb, dealRowCount, columnCount, actualColumnCount);

    size_t dstOffset = (loop % PRE_LOAD_NUM) * mmResUbSize + gSizeStart * columnCount;
    AntiquantSoftmaxResPreProcess(vec1ResGm[dstOffset], mmResUb, tmpAFloorUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::DealAntiqBmm1ResBaseBlockPerToken(uint32_t loop,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];

    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    uint64_t srcGmOffset = (loop % PRE_LOAD_NUM) * mmResUbSize + gSizeStart * columnCount;
    AntiquantMatmulResCombine(mmResUb, mm1ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount);

    uint32_t baseOffset = startRow * PRE_BLOCK_ELEMENT_NUM;
    LocalTensor<T> aMax = aMaxBmm1Ub[info.bn2IdxInCurCore % (PRE_LOAD_NUM)][baseOffset];
    LocalTensor<T> aRowSum = qRowSumUb[info.bn2IdxInCurCore % (PRE_LOAD_NUM)][baseOffset];
    uint32_t dtypeMask = PRE_REPEAT_ELEMENT_NUM;
    uint32_t mulPreLoop = actualColumnCount / dtypeMask;
    uint32_t mulRemain = actualColumnCount % dtypeMask;
    BinaryRepeatParams repeatParams;

    if (columnCount < REPEATE_STRIDE_UP_BOUND * PRE_BLOCK_ELEMENT_NUM) {
        if (antiqOffsetExistFlag) {
            CopyAntiquantParamsPerToken(keyAntiqOffsetGm, info.antiqKeyParamOffsetPerToken, columnCount, actualColumnCount);
            LocalTensor<T> antiqOffsetPerTokenUb = inputQue1.DeQue<T>();
            LocalTensor<T> tmpOffset = tmpBuff2.Get<T>();

            // rowsum(A) * offset
            repeatParams.src0RepStride = 1;
            repeatParams.src0BlkStride = 0;
            repeatParams.src1RepStride = 0;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = columnCount / PRE_BLOCK_ELEMENT_NUM;
            repeatParams.dstBlkStride = 1;
            for (uint32_t i = 0; i < mulPreLoop; i++) {
                Mul(tmpOffset[i * dtypeMask], aRowSum, antiqOffsetPerTokenUb[i * dtypeMask], dtypeMask, dealRowCount,
                    repeatParams);
            }
            if (mulRemain > 0) {
                Mul(tmpOffset[mulPreLoop * dtypeMask], aRowSum, antiqOffsetPerTokenUb[mulPreLoop * dtypeMask], mulRemain,
                    dealRowCount, repeatParams);
            }
            inputQue1.FreeTensor(antiqOffsetPerTokenUb);

            // Amax * C + rowsum(A) * offset
            repeatParams.src0RepStride = 1;
            repeatParams.src0BlkStride = 0;
            repeatParams.src1RepStride = columnCount / PRE_BLOCK_ELEMENT_NUM;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = columnCount / PRE_BLOCK_ELEMENT_NUM;
            repeatParams.dstBlkStride = 1;
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < mulPreLoop; j++) {
                FusedMulAdd(mmResUb[j * dtypeMask], aMax, tmpOffset[j * dtypeMask], dtypeMask, dealRowCount,
                            repeatParams);
            }
            if (mulRemain > 0) {
                FusedMulAdd(mmResUb[mulPreLoop * dtypeMask], aMax, tmpOffset[mulPreLoop * dtypeMask], mulRemain,
                            dealRowCount, repeatParams);
            }
        } else {
            // Amax * C
            repeatParams.src0RepStride = 1;
            repeatParams.src0BlkStride = 0;
            repeatParams.src1RepStride = columnCount / PRE_BLOCK_ELEMENT_NUM;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = columnCount / PRE_BLOCK_ELEMENT_NUM;
            repeatParams.dstBlkStride = 1;
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < mulPreLoop; i++) {
                Mul(mmResUb[i * dtypeMask], aMax, mmResUb[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
            }
            if (mulRemain > 0) {
                Mul(mmResUb[mulPreLoop * dtypeMask], aMax, mmResUb[mulPreLoop * dtypeMask], mulRemain, dealRowCount,
                    repeatParams);
            }
        }
    } else {
        repeatParams.src0RepStride = 8;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // Amax * C
            Mul(mmResUb[i * columnCount], mmResUb[i * columnCount], aMax[i * PRE_BLOCK_ELEMENT_NUM],
                dtypeMask, mulPreLoop, repeatParams);
            if (mulRemain > 0) {
                // Amax * C, deal column tail
                Mul(mmResUb[i * columnCount + mulPreLoop * dtypeMask], mmResUb[i * columnCount + mulPreLoop * dtypeMask],
                    aMax[i * PRE_BLOCK_ELEMENT_NUM], mulRemain, 1, repeatParams);
            }
        }

        if (antiqOffsetExistFlag) {
            CopyAntiquantParamsPerToken(keyAntiqOffsetGm, info.antiqKeyParamOffsetPerToken, columnCount, actualColumnCount);
            LocalTensor<T> antiqOffsetPerTokenUb = inputQue1.DeQue<T>();
            LocalTensor<T> tmpOffset = tmpBuff2.Get<T>();
            for (uint32_t i = 0; i < dealRowCount; i++) {
                // rowsum(A) * offset
                Mul(tmpOffset[i * columnCount], antiqOffsetPerTokenUb, aRowSum[i * PRE_BLOCK_ELEMENT_NUM],
                    dtypeMask, mulPreLoop, repeatParams);
                if (mulRemain > 0) {
                    // rowsum(A) * offset, deal column tail
                    Mul(tmpOffset[i * columnCount + mulPreLoop * dtypeMask], antiqOffsetPerTokenUb[mulPreLoop * dtypeMask],
                        aRowSum[i * PRE_BLOCK_ELEMENT_NUM], mulRemain, 1, repeatParams);
                }
            }
            inputQue1.FreeTensor(antiqOffsetPerTokenUb);

            PipeBarrier<PIPE_V>();
            // Amax * C + rowsum(A) * offset
            Add(mmResUb, mmResUb, tmpOffset, dealRowCount * columnCount);
        }
    }
    PipeBarrier<PIPE_V>();

    CopyAntiquantParamsPerToken(keyAntiqScaleGm, info.antiqKeyParamOffsetPerToken, columnCount, actualColumnCount);
    LocalTensor<T> antiqScalePerTokenUb = inputQue1.DeQue<T>();
    // (Amax * C + rowsum(A) * offset) * scale
    VecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(antiqScalePerTokenUb);

    // mul scalar and mask
    ElewiseCompute(loop, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(loop, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    DealKvInt4ColumnOdd(mmResUb, dealRowCount, columnCount, actualColumnCount);

    // mmResUb mul scale
    CopyAntiquantParamsPerToken(valueAntiqScaleGm, info.antiqValueParamOffsetPerToken, columnCount, actualColumnCount);
    antiqScalePerTokenUb = inputQue1.DeQue<T>();
    VecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(antiqScalePerTokenUb);

    Adds(tmpAFloorUb, mmResUb, (T)0, dealRowCount * columnCount); // mmResUb need to be stored
    PipeBarrier<PIPE_V>();
    if (antiqOffsetExistFlag) {
        LocalTensor<T> tmpAMax = tmpBuff3.Get<T>();

        // (mmResUb * scale) · offset = rowsum(mmResUb * scale * offset)
        CopyAntiquantParamsPerToken(valueAntiqOffsetGm, info.antiqValueParamOffsetPerToken, columnCount, actualColumnCount);
        antiqScalePerTokenUb = inputQue1.DeQue<T>();
        VecMulMat(tmpAFloorUb, antiqScalePerTokenUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
        inputQue1.FreeTensor(antiqScalePerTokenUb);
        PipeBarrier<PIPE_V>();
        RowSumForLongColumnCount(tmpAMax, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Brcb(softmaxScaleResRowSumUb[loop % (PRE_LOAD_NUM)][baseOffset], tmpAMax, (dealRowCount + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        Adds(tmpAFloorUb, mmResUb, (T)0, dealRowCount * columnCount); // mmResUb need to be stored
        PipeBarrier<PIPE_V>();
    }

    size_t dstOffset = (loop % PRE_LOAD_NUM) * mmResUbSize + gSizeStart * columnCount;
    AntiquantMatmulPreProcess(vec1ResGm[dstOffset], aMaxBmm2Ub[loop % PRE_LOAD_NUM], mmResUb, tmpAFloorUb,
        startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::PostProcessVec1()
{
    if constexpr (ANTIQUANT_PRE && ANTIQUANT_PER_TOKEN) {
        SysPrefixSaveMsdMax2(bIdx);
        if (antiqOffsetExistFlag) {
            SysPrefixSaveMsdSum2(bIdx);
        }
    }
    SysPrefixSaveSoftmaxMax(bIdx);
    SysPrefixSaveSoftmaxSum(bIdx);
    SysPrefixSaveSoftmaxExp(bIdx);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ProcessVec1Inner(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    if (gPreSplitSize > gSizeVector) {
        gPreSplitSize = gSizeVector;
    }
    uint32_t loopCount = (gSizeVector + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = gSizeVector - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT_PRE) {
            if constexpr (ANTIQUANT_PER_CHANNEL) {
                DealAntiqBmm1ResBaseBlock(loop, i * gPreSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                          info.actualSinglePreProcessSInnerSize);
            } else if constexpr (ANTIQUANT_PER_TOKEN) {
                DealAntiqBmm1ResBaseBlockPerToken(loop, i * gPreSplitSize, dealSize,
                                                  info.actualSingleProcessSInnerSizeAlign, info.actualSinglePreProcessSInnerSize);
            }
        } else {
            DealBmm1ResBaseBlock(loop, i * gPreSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                 info.actualSinglePreProcessSInnerSize);
        }
    }

    if (info.s2Idx == sInnerPreLoopTimes - 1) {
        if (flashDecodeFlag) {
            uint32_t outIdx = loop % (PRE_LOAD_NUM);
            ComputeLogSumExpAndCopyToGm(softmaxSumUb[outIdx], softmaxMaxUb[outIdx]);
            return;
        }
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::DealBmm2ResBaseBlock(const uint32_t loop, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * PRE_BLOCK_ELEMENT_NUM;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;
    if constexpr (SHARED_PREFIX_PRE) {
        if (calcSysPrefixFlag) {
            batchBase = bIdx * gSize * columnCount;
        }
    }

    uint64_t inOutBaseOffset = (gSizeStart + startRow) * columnCount;
    uint64_t srcGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<MM_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    inputQue1.EnQue(tmpBmm2ResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    inputQue1.FreeTensor(tmpBmm2ResUb);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
        uint64_t vecPre2ResGmOffset = ((loop - 1) % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vecPre2ResGmOffset + batchBase], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);

        inputQue2.DeQue<T>();
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();

        uint64_t vecPre2ResGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vecPre2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::PostQuant(uint64_t perChannelQuantOffset, LocalTensor<T> &bmm2ResUb, LocalTensor<int8_t> &bmmPre2ResUbInt8,
                                                        uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount)
{
    uint32_t copySize = dealRowCount * columnCount;
    if (!isPerChnU8Out) {
        LocalTensor<uint8_t> sharedTempBuffer = tmpBuff2.Get<uint8_t>(); // AscendQuant接口要求sharedTempBuffer不能太小
        AscendQuant(bmmPre2ResUbInt8, bmm2ResUb, sharedTempBuffer, quantScale2Value, quantOffset2Value, copySize);
    } else {
        if (!isPreOutQuantTypeBf16) { // fp32
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<float> copyInPadParams;
            copyInParams.blockCount = dealRowCount;
            copyInParams.blockLen = actualColumnCount * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = (columnCount - actualColumnCount) / PRE_BLOCK_ELEMENT_NUM;

            copyInPadParams.isPad = true;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = (columnCount - actualColumnCount) % PRE_BLOCK_ELEMENT_NUM;
            copyInPadParams.paddingValue = 0;
            {
                LocalTensor<float> quantPreScale2Ub = inputQue1.AllocTensor<float>();
                DataCopyPad(quantPreScale2Ub, quantScale2Gm[perChannelQuantOffset + startRow * actualColumnCount + gSizeStart * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue1.EnQue(quantPreScale2Ub);
                inputQue1.DeQue<float>();

                Mul(bmm2ResUb, quantPreScale2Ub, bmm2ResUb, copySize);
                inputQue1.FreeTensor(quantPreScale2Ub);
                PipeBarrier<PIPE_V>();
            }
            if (isQuantOffset2Exist) {
                LocalTensor<float> quantPreOffset2Ub = inputQue1.AllocTensor<float>();
                DataCopyPad(quantPreOffset2Ub, quantOffset2Gm[perChannelQuantOffset + startRow * actualColumnCount + gSizeStart * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue1.EnQue(quantPreOffset2Ub);
                inputQue1.DeQue<float>();

                Add(bmm2ResUb, quantPreOffset2Ub, bmm2ResUb, copySize);
                inputQue1.FreeTensor(quantPreOffset2Ub);
                PipeBarrier<PIPE_V>();
            }
        } else {
            uint32_t typeElementSize = BYTE_BLOCK / sizeof(bfloat16_t);
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<bfloat16_t> copyInPadParams;
            copyInParams.blockCount = dealRowCount;
            copyInParams.blockLen = actualColumnCount * sizeof(bfloat16_t);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = (columnCount - actualColumnCount) / typeElementSize;

            copyInPadParams.isPad = true;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = (columnCount - actualColumnCount) % typeElementSize;
            copyInPadParams.paddingValue = 0;
            LocalTensor<float> tempCastUb = tmpBuff2.Get<float>(copySize);
            {
                LocalTensor<bfloat16_t> quantPreScale2Ub = inputQue1.AllocTensor<bfloat16_t>();
                DataCopyPad(quantPreScale2Ub, quantScale2Bf16Gm[perChannelQuantOffset + startRow * actualColumnCount + gSizeStart * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue1.EnQue(quantPreScale2Ub);
                inputQue1.DeQue<bfloat16_t>();

                Cast(tempCastUb, quantPreScale2Ub, RoundMode::CAST_NONE, copySize);
                inputQue1.FreeTensor(quantPreScale2Ub);
                PipeBarrier<PIPE_V>();
            }

            Mul(bmm2ResUb, tempCastUb, bmm2ResUb, copySize);
            PipeBarrier<PIPE_V>();
            if (isQuantOffset2Exist) {
                LocalTensor<bfloat16_t> quantPreOffset2Ub = inputQue2.AllocTensor<bfloat16_t>();
                DataCopyPad(quantPreOffset2Ub, quantOffset2Bf16Gm[perChannelQuantOffset + startRow * actualColumnCount + gSizeStart * actualColumnCount],
                            copyInParams, copyInPadParams);
                inputQue2.EnQue(quantPreOffset2Ub);
                inputQue2.DeQue<bfloat16_t>();

                Cast(tempCastUb, quantPreOffset2Ub, RoundMode::CAST_NONE, copySize);
                inputQue2.FreeTensor(quantPreOffset2Ub);
                PipeBarrier<PIPE_V>();

                Add(bmm2ResUb, tempCastUb, bmm2ResUb, copySize);
                PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<half> quantResultHalf = tmpBuff1.Get<half>(copySize);
        Cast(quantResultHalf, bmm2ResUb, RoundMode::CAST_ROUND, copySize);
        PipeBarrier<PIPE_V>();

        Cast(bmmPre2ResUbInt8, quantResultHalf, RoundMode::CAST_ROUND, copySize);
        PipeBarrier<PIPE_V>();
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::DealAntiqBmm2ResBaseBlock(uint32_t loop, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];

    uint32_t baseOffset = startRow * PRE_BLOCK_ELEMENT_NUM;
    uint64_t srcGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + gSizeStart * columnCount;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    AntiquantMatmulResCombine(bmm2ResUb, mm2ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount);

    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    size_t batchBase = 0;
    if constexpr (SHARED_PREFIX_PRE) {
        if (calcSysPrefixFlag) {
            batchBase = bIdx * gSize * columnCount;
        }
    }
    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        uint64_t vec2ResGmOffset = ((loop - 1) % PRE_LOAD_NUM) * bmm2ResUbSize + (gSizeStart + startRow) * columnCount;
        LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + batchBase], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);

        inputQue2.DeQue<T>();
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[loop % PRE_LOAD_NUM][baseOffset],
            dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[loop % (PRE_LOAD_NUM)][baseOffset],
            dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();

        if (antiqOffsetExistFlag) {
            // bmm2Res + offsetV
            CopyAntiquantScale(antiqOffsetUb, valueAntiqOffsetGm, info.antiqValueParamOffset);
            PipeBarrier<PIPE_V>();
            VecAddMat(bmm2ResUb, antiqOffsetUb, bmm2ResUb, dealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
        }

        CopyAntiquantScale(antiqScaleUb, valueAntiqScaleGm, info.antiqValueParamOffset);
        PipeBarrier<PIPE_V>();
        // ScaleV * bmm2Res
        VecMulMat(bmm2ResUb, antiqScaleUb, bmm2ResUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();

        Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();
        uint64_t vec2ResGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + (gSizeStart + startRow) * columnCount;
        DataCopy(vec2ResGm[vec2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::DealAntiqBmm2ResBaseBlockPerToken(uint32_t loop,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];

    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    uint64_t srcGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + gSizeStart * columnCount;
    AntiquantMatmulResCombine(bmm2ResUb, mm2ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount);

    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * PRE_BLOCK_ELEMENT_NUM;
    LocalTensor<T> aRowMax = aMaxBmm2Ub[loop % (PRE_LOAD_NUM)][baseOffset];
    uint32_t dtypeMask = PRE_REPEAT_ELEMENT_NUM;
    int32_t mulPreLoop = actualColumnCount / dtypeMask;
    int32_t mulRemain = actualColumnCount % dtypeMask;
    BinaryRepeatParams repeatParams;
    if (antiqOffsetExistFlag) {
        LocalTensor<T> aRowSum = softmaxScaleResRowSumUb[loop % (PRE_LOAD_NUM)][baseOffset];

        PipeBarrier<PIPE_V>();
        repeatParams.src0RepStride = 1;
        repeatParams.src0BlkStride = 0;
        repeatParams.src1RepStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = columnCount / PRE_BLOCK_ELEMENT_NUM;
        repeatParams.dstBlkStride = 1;
        for (int j = 0; j < mulPreLoop; j++) {
            FusedMulAdd(bmm2ResUb[j * dtypeMask], aRowMax, aRowSum, dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            FusedMulAdd(bmm2ResUb[mulPreLoop * dtypeMask], aRowMax, aRowSum, mulRemain, dealRowCount, repeatParams);
        }
    } else {
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResUb, bmm2ResUb, aRowMax, dealRowCount, columnCount, actualColumnCount);
    }

    size_t batchBase = 0;
    if constexpr (SHARED_PREFIX_PRE) {
        if (calcSysPrefixFlag) {
            batchBase = bIdx * gSize * columnCount;
        }
    }

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
        uint64_t vec2ResGmOffset = ((loop - 1) % PRE_LOAD_NUM) * bmm2ResUbSize + (gSizeStart + startRow) * columnCount;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + batchBase], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);

        inputQue2.DeQue<T>();
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpPreBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpPreBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpPreBmm2Res);
        outputQue1.DeQue<T>();
        uint64_t vec2ResGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + (gSizeStart + startRow) * columnCount;
        DataCopy(vec2ResGm[vec2ResGmOffset + batchBase], tmpPreBmm2Res, vec2ComputeSize);
        outputQue1.FreeTensor(tmpPreBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ProcessVec2Inner(uint32_t loop)
{
    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (gPreSplitSize > gSizeVector) {
        gPreSplitSize = gSizeVector;
    }
    uint32_t loopCount = (gSizeVector + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = gSizeVector - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if constexpr (ANTIQUANT_PRE) {
            if constexpr (ANTIQUANT_PER_CHANNEL) {
                DealAntiqBmm2ResBaseBlock(loop, i * gPreSplitSize, dealSize, headDimAlign, headDim);
            } else if constexpr (ANTIQUANT_PER_TOKEN) {
                DealAntiqBmm2ResBaseBlockPerToken(loop, i * gPreSplitSize, dealSize, headDimAlign, headDim);
            }
        } else {
            DealBmm2ResBaseBlock(loop, i * gPreSplitSize, dealSize, headDimAlign, headDim);
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SetMMOrgShape()
{
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::QueryPreProcessL(uint32_t loop)
{
    if (gSizeVector == 0) {
        return;
    }
    QueryPreProcessInner(loop);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CalcParams(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    info.loop = loop;
    if (loop != 0) {
        info.isChangeBatch = false;
        if (flashDecodeFlag) {
            curS2Idx++;
        } else {
            if (curS2Idx + 1 >= curSInnerLoopTimes) {
                if ((!batchPreContinuous) && (curN2Idx + 1 >= kvHeadNum)) {
                    info.isChangeBatch = true;
                }
                curS2Idx = 0;
                curBIdx = curBIdx + (curN2Idx + 1) / kvHeadNum;
                curN2Idx = (curN2Idx + 1) % kvHeadNum;

                curActSeqLenIsZero = true;
                while (curActSeqLenIsZero) {
                    bIdx = curBIdx;
                    n2Idx = curN2Idx;
                    GetActualSeqLen();
                    UpdateInnerLoopCond();
                    if (curActSeqLenIsZero) {
                        curBIdx++;
                        curN2Idx = 0;
                        continue;
                    }
                }
                curS2Idx = s2Idx;
                curSInnerLoopTimes = sInnerPreLoopTimes;
                s2BatchBaseOffset = kvPaddingBeginOffset;
            } else {
                curS2Idx++;
            }
        }
    } else {
        info.isChangeBatch = (!batchPreContinuous);
        if (flashDecodeFlag) {
        } else {
            curActSeqLenIsZero = true;
            while (curActSeqLenIsZero) {
                bIdx = curBIdx;
                n2Idx = curN2Idx;
                GetActualSeqLen();
                // ComputeKVPaddingBeginOffset return false means this loop skip calculation
                // 根据当前块实际长度, 重配flashattention循环条件
                UpdateInnerLoopCond();
                if (curActSeqLenIsZero) {
                    curBIdx++;
                    curN2Idx = 0;
                    continue;
                }
            }
            curS2Idx = s2Idx;
            curSInnerLoopTimes = sInnerPreLoopTimes;
            s2BatchBaseOffset = kvPaddingBeginOffset;
        }
    }
    info.bIdx = curBIdx;
    info.n2Idx = curN2Idx;
    info.s2Idx = curS2Idx;
    info.curSInnerLoopTimes = curSInnerLoopTimes;

    info.isFirstSInnerLoop = (info.s2Idx == 0);
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            tensorACorePreOffset = info.bIdx * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            // B,N2,S2,D
            tensorBCorePreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDim + info.n2Idx * kvSeqSize * headDim;
            if (!batchPreContinuous) {
                uint64_t seqSize = SeqLenFromTensorList(bIdx);
                tensorBCorePreOffset = info.n2Idx * seqSize * headDim;
            }
            tensorBCorePreOffset += kvPaddingBeginOffset * headDim;
            if (flashDecodeFlag) {
                tensorBCorePreOffset += s2IdxFD * sInnerLoopSize * headDim;
            }
        } else {
            // B,1,N2,G,D
            tensorACorePreOffset = info.bIdx * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            // B,S2,N2,D
            tensorBCorePreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            if (!batchPreContinuous) {
                tensorBCorePreOffset = info.n2Idx * headDim;
            }
            tensorBCorePreOffset += kvPaddingBeginOffset * kvHeadNum * headDim;
            if (flashDecodeFlag) {
                tensorBCorePreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDim;
            }
        }
    }
    info.tensorAOffset = tensorACorePreOffset;
    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        info.tensorBOffset = tensorBCorePreOffset + info.s2Idx * singlePreProcessSInnerSize * headDim;
    } else {
        info.tensorBOffset = tensorBCorePreOffset + info.s2Idx * singlePreProcessSInnerSize * kvHeadNum * headDim;
    }
    info.attenOutOffset = tensorACorePreOffset;
    info.actualSinglePreProcessSInnerSize = singlePreProcessSInnerSize;
    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        info.actualSinglePreProcessSInnerSize = singlePreProcessSInnerSizeTail;
    }
    if constexpr (KVINT4) {
        info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSinglePreProcessSInnerSize, (uint32_t)64L);
    } else {
        info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSinglePreProcessSInnerSize, (uint32_t)BYTE_BLOCK);
    }
    info.needSetOrgShape = (curSinglePreProcessSInnerSizeAlign != info.actualSingleProcessSInnerSizeAlign);
    if (info.needSetOrgShape) {
        curSinglePreProcessSInnerSizeAlign = info.actualSingleProcessSInnerSizeAlign;
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
    uint64_t sInnerOffsetDataSize = info.s2Idx * singlePreProcessSInnerSize;
    if (flashDecodeFlag) {
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
    if (pseShiftPreFlag) {
        info.pseShiftCoreOffset = info.n2Idx * gSize * pseShiftS + sInnerOffsetDataSize + kvPaddingBeginOffset;
        if (pseShiftB != 1) {
            info.pseShiftCoreOffset += info.bIdx * qHeadNum * pseShiftS;
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ProcessVec2L(uint32_t loop) {
    if (gSizeVector == 0) {
        return;
    }
    ProcessVec2Inner(loop);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ProcessVec1L(uint32_t loop)
{
    if (gSizeVector == 0) {
        return;
    }
    ProcessVec1Inner(loop);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SetLoopTimes() {
    // 获取总的循环次数，此处假设所有head的S=Smax
    bn2s2LoopTimes = 0;
    bool hasInit = false;
    for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
        GetBN2id(bn2Idx);
        GetActualSeqLen();
        // ComputeKVPaddingBeginOffset return false means this loop skip calculation
        if (!ComputeKVPaddingBeginOffset()) {
            DealActSeqLenIsZero(bIdx, n2Idx);
            continue;
        }
        // 根据当前块实际长度, 重配flashattention循环条件
        UpdateInnerLoopCond();
        if (curActSeqLenIsZero) {
            DealActSeqLenIsZero(bIdx, n2Idx);
            continue;
        }
        if (!hasInit) {
            curBIdx = bIdx;
            curN2Idx = n2Idx;
            curS2Idx = s2Idx;
            curSInnerLoopTimes = sInnerPreLoopTimes;
            hasInit = true;
        }
        bn2s2LoopTimes += sInnerPreLoopTimes;
    }
}

#ifndef USE_MM_API
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info) {
    // ND[msdPreIterNum * gSize, headDimAlign]->NZ[headDimAlign / 32, 16, 32]
    uint32_t mmRowCount = msdPreIterNum * gSize;
    uint32_t copyStride = 16 * headDimAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++){
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdPreIterNum * gSize - i * 16;
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
        if constexpr (ANTIQUANT_PRE) {
            DataCopy(aL1Tensor[i * copyStride], queryPreProcessResGm[(info.bn2IdxInCurCore % (PRE_LOAD_NUM)) * bmm2ResUbSize + i * copyStride], mm1Nd2NzParamsForA);
        } else {
            DataCopy(aL1Tensor, queryGm[info.tensorAOffset], mm1Nd2NzParamsForA);
        }
    }
}

// nCopyRowCount需要32元素对齐
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyInMm1BToL1(
    LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t nCopyIdx, uint32_t nCopyRowCount,
    uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign)
{
    uint64_t step = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
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

        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) && (baseN * step > 65535)) {
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
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyInMm1BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    uint32_t baseN = 256 / sizeof(KV_T);
    uint64_t step = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        step = headDim * kvHeadNum;
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

    if (midNdNum != 0) {
        mm1Nd2NzParamsForB.nValue = baseN; // 单个ND矩阵的行数, 单位为元素个数   256
        mm1Nd2NzParamsForB.dValue = headDim; // 单个ND矩阵的列数, 单位为元素个数   128
        mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数  // 128
        mm1Nd2NzParamsForB.dstNzC0Stride = baseN;// 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数   256
        mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / baseN;
        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) && (baseN * step > 65535)) {
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

    if (nTail != 0) {
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

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor) {
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    if constexpr (KVINT4) {
        loadData2DParams.repeatTimes = (msdPreIterNum * gSize + 15) / 16 * headDimAlign / 64;
    } else {
        loadData2DParams.repeatTimes = (msdPreIterNum * gSize + 15) / 16 * headDimAlign / (32 / sizeof(KV_T));
    }
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ComputeMm1(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    gSize = gSizeCube;
    if (info.isChangeBatch) {
        InitKeyGm(info.bIdx);
    }

    LocalTensor<KV_T> aL1Tensor = inputQueL1A.AllocTensor<KV_T>();
    CopyInMm1AToL1(aL1Tensor, info);
    inputQueL1A.EnQue(aL1Tensor);
    inputQueL1A.DeQue<KV_T>();

    constexpr uint32_t nCopyRowCount = KV_PRE_LOAD_L1_ROW_NUM;
    uint32_t nCopyTimes = (info.actualSinglePreProcessSInnerSize + nCopyRowCount - 1) / nCopyRowCount;
    uint32_t nTailCopyRowCount = info.actualSinglePreProcessSInnerSize - (nCopyTimes - 1) * nCopyRowCount;
    uint32_t nTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (nCopyTimes - 1) * nCopyRowCount;
    for (uint32_t nCopyIdx = 0, nActCopyRowCount = nCopyRowCount, nActCopyRowCountAlign = nCopyRowCount; nCopyIdx < nCopyTimes; nCopyIdx++) {
        if (nCopyIdx + 1 == nCopyTimes) {
            nActCopyRowCount = nTailCopyRowCount;
            nActCopyRowCountAlign = nTailCopyRowCountAlign;
        }
        // A:16*128 B:1024*128 A*B^T INT8  M=16 N=1024 K128 baseM=16 baseN=256 baseK=128
        // A、B全载到L1，B的N方向按照256切分，计算4次
        LocalTensor<KV_T> bL1Tensor = inputQueL1B.AllocTensor<KV_T>();
        if constexpr (PAGE_ATTENTION) {
            uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
            uint32_t curSeqIdx = info.s2BatchOffset + nCopyIdx * nCopyRowCount;
            uint32_t copyFinishRowCnt = 0;
            while (copyFinishRowCnt < nActCopyRowCount) {
                uint64_t blockPreIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                uint64_t idInBlockTable =
                    blockTableGm.GetValue(blockTableBaseOffset + blockPreIdOffset); // 从block table上的获取编号
                // 计算可以拷贝行数
                uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                if (copyFinishRowCnt + copyRowCnt > nActCopyRowCount) {
                    copyRowCnt = nActCopyRowCount - copyFinishRowCnt;
                }
                uint64_t keyOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
                    keyOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                } else {
                    keyOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                }
                CopyInMm1BToL1ForPA(bL1Tensor, keyOffset, nActCopyRowCountAlign, copyFinishRowCnt, copyRowCnt);

                // 更新循环变量
                copyFinishRowCnt += copyRowCnt;
                curSeqIdx += copyRowCnt;
            }
        } else {
            CopyInMm1BToL1(bL1Tensor, info, nCopyIdx, nCopyRowCount, nActCopyRowCount, nActCopyRowCountAlign);
        }
        inputQueL1B.EnQue(bL1Tensor);
        inputQueL1B.DeQue<KV_T>();

        LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[l0cBufferManager.GetBuffer()];
        {
            LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[l0aBufferManager.GetBuffer()];
            LoadDataMm1A(aL0Tensor, aL1Tensor);
            l0aBufferManager.EnQue();
            l0aBufferManager.DeQue();

            constexpr uint32_t baseN = 256 / sizeof(KV_T);
            uint32_t nLoopTimes = (nActCopyRowCountAlign + baseN - 1) / baseN;
            uint32_t nTail = nActCopyRowCountAlign - (nLoopTimes - 1) * baseN;
            for (uint32_t i = 0, actualBaseN = baseN; i < nLoopTimes; i++) {
                if (i + 1 == nLoopTimes) {
                    actualBaseN = nTail;
                }
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[l0bBufferManager.GetBuffer()];
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
                for(uint32_t j = 0; j < loadLoopTimes; j++) {
                    LoadData(bL0Tensor[j * actualBaseN * blockElementCnt],
                        bL1Tensor[(i * baseN + j * nActCopyRowCountAlign) * blockElementCnt], loadData2DParamsForB);
                }
#else
                loadData2DParamsForB.repeatTimes = (actualBaseN / 16) * (headDimAlign / blockElementCnt);
                LoadData(bL0Tensor, bL1Tensor[baseN * headDimAlign * i], loadData2DParamsForB);
#endif
                l0bBufferManager.EnQue();
                l0bBufferManager.DeQue();
                MmadParams mmadParams;
                mmadParams.m = msdPreIterNum * gSize;
                if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = actualBaseN; // 无效数据不参与计算
                mmadParams.k = 128;
                mmadParams.cmatrixInitVal = true;
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor[((mmadParams.m + 15) / 16) * 16 * baseN * i], aL0Tensor, bL0Tensor, mmadParams);
                l0bBufferManager.ReleaseBuffer();
            }

            l0aBufferManager.ReleaseBuffer();
        }
        inputQueL1B.FreeTensor(bL1Tensor);
        l0cBufferManager.EnQue();
        l0cBufferManager.DeQue();
        FixpipeParamsV220 fixParams;
        fixParams.nSize = nActCopyRowCountAlign;
        fixParams.mSize = msdPreIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
        fixParams.srcStride = ((fixParams.mSize + 15) / 16) * 16;
        fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
        fixParams.ndNum = 1;
        Fixpipe(mm1ResGm[(loop % (PRE_LOAD_NUM)) * mmResUbSize + nCopyIdx * nCopyRowCount], cL0Tensor, fixParams);
        l0cBufferManager.ReleaseBuffer();
    }
    inputQueL1A.FreeTensor(aL1Tensor);
}
#else
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ComputeMm1(uint32_t loop) {
    gSize = gSizeCube;
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    if (info.isChangeBatch) {
        InitKeyGm(info.bIdx);
    }
    if (info.needSetOrgShape) {
        uint32_t orgKa;
        if constexpr (ANTIQUANT_PRE) {
            orgKa = headDimAlign;
        } else {
            orgKa = headDim;
        }
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || PAGE_ATTENTION) {
            mm.SetOrgShape(msdPreIterNum * gSize, tilingData->baseParams.seqSize,
                orgKa, kvHeadNum * headDim, info.actualSingleProcessSInnerSizeAlign);
        } else {
            mm.SetOrgShape(msdPreIterNum * gSize, tilingData->baseParams.seqSize,
                orgKa, headDim, info.actualSingleProcessSInnerSizeAlign);
        }
    }
    if constexpr (ANTIQUANT_PRE) {
        mm.SetTensorA(queryPreProcessResGm[(info.bn2IdxInCurCore % (PRE_LOAD_NUM)) * bmm2ResUbSize]);
    } else {
        mm.SetTensorA(queryGm[info.tensorAOffset]);
    }
    mm.SetTensorB(keyGm[info.tensorBOffset], true);

    mm.SetTail(msdPreIterNum * gSize, info.actualSinglePreProcessSInnerSize, headDim);
    mm.template IterateAll<false>(mm1ResGm[(loop % (PRE_LOAD_NUM)) * mmResUbSize], false, false, true);
    mm.End();
}
#endif

#ifndef USE_MM_API
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, ExtraInfo& info, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount) {
    // ND:[msdPreIterNum * gSize, info.actualSingleProcessSInnerSizeAlign]->NZ:[info.actualSingleProcessSInnerSizeAlign / 32, 16, 32]
    uint32_t mmRowCount = msdPreIterNum * gSize;
    uint32_t copyStrideL1 = 16 * kActCopyRowCount;
    uint32_t copyStrideGm = 16 * info.actualSingleProcessSInnerSizeAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++) {
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdPreIterNum * gSize - i * 16;
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
        DataCopy(aL1Tensor[i * copyStrideL1], vec1ResGm[(info.loop % (PRE_LOAD_NUM)) * mmResUbSize + kCopyIdx * kCopyRowCount + i * copyStrideGm], mm1Nd2NzParamsForA);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor, ExtraInfo& info, uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount) {
    uint64_t step = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
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
        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) && (blockK * step > 65535)) {
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
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyInMm2BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset, uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount)
{
    uint32_t blockK = 32 / sizeof(KV_T); // 单个ND矩阵的行数
    uint64_t step = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        step = headDim * kvHeadNum;
    }

    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInBaseN = copyStartRowCnt % blockK;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInBaseN + kActCopyRowCount <= blockK) {
        nHead = 0;
        midNdNum = 0;
        nTail = kActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInBaseN;
    } else {
        if (copyStartRowCntInBaseN == 0) {
            nHead = 0;
        } else {
            nHead = blockK - copyStartRowCntInBaseN;
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
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nHead; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        uint32_t ndNumFinish = copyStartRowCnt / blockK;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset],
                mm1Nd2NzParamsForB);
    }

    if (midNdNum != 0) {
        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK;
        if ((LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) && (blockK * step > 65535)) {
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

    if (nTail != 0) {
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK + midNdNum;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))], valueGm[valueGmBaseOffset + (nHead + midNdNum * blockK) * step],
                 mm1Nd2NzParamsForB);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize) {
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
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::LoadDataMm2B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor) {
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ComputeMm2(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    gSize = gSizeCube;
    if (info.isChangeBatch) {
        InitValueGm(info.bIdx);
    }

    constexpr uint32_t kCopyRowCount = KV_PRE_LOAD_L1_ROW_NUM;
    uint32_t kCopyTimes = (info.actualSinglePreProcessSInnerSize + kCopyRowCount - 1) / kCopyRowCount;
    uint32_t kTailCopyRowCount = info.actualSinglePreProcessSInnerSize - (kCopyTimes - 1) * kCopyRowCount;
    uint32_t kTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (kCopyTimes - 1) * kCopyRowCount;

    LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[l0cBufferManager.GetBuffer()];
    for (uint32_t kCopyIdx = 0, kActCopyRowCount = kCopyRowCount, kActCopyRowCountAlign = kCopyRowCount; kCopyIdx < kCopyTimes; kCopyIdx++) {
        if (kCopyIdx + 1 == kCopyTimes) {
            kActCopyRowCount = kTailCopyRowCount;
            kActCopyRowCountAlign = kTailCopyRowCountAlign;
        }
        // P[16, 1024] FP16/BF16; V[1024, 128] INT8
        LocalTensor<KV_T> aL1Tensor = inputQueL1A.AllocTensor<KV_T>();
        CopyInMm2AToL1(aL1Tensor, info, kCopyIdx, kCopyRowCount, kActCopyRowCountAlign);
        inputQueL1A.EnQue(aL1Tensor);
        LocalTensor<KV_T> bL1Tensor = inputQueL1B.AllocTensor<KV_T>();
        if constexpr (PAGE_ATTENTION) {
            uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
            uint32_t curSeqIdx = info.s2BatchOffset + kCopyIdx * kCopyRowCount;
            uint32_t copyFinishRowCnt = 0;
            while (copyFinishRowCnt < kActCopyRowCount) {
                uint64_t blockIdPreOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                uint64_t idInBlockTable =
                    blockTableGm.GetValue(blockTableBaseOffset + blockIdPreOffset); // 从block table上的获取编号
                // 计算可以拷贝行数
                uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                if (copyFinishRowCnt + copyRowCnt > kActCopyRowCount) {
                    copyRowCnt = kActCopyRowCount - copyFinishRowCnt;
                }
                uint64_t valueOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
                    valueOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                } else {
                    valueOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                }
                CopyInMm2BToL1ForPA(bL1Tensor, valueOffset, kActCopyRowCount, copyFinishRowCnt, copyRowCnt);

                // 更新循环变量
                copyFinishRowCnt += copyRowCnt;
                curSeqIdx += copyRowCnt;
            }
        } else {
            CopyInMm2BToL1(bL1Tensor, info, kCopyIdx, kCopyRowCount, kActCopyRowCount);
        }
        inputQueL1B.EnQue(bL1Tensor);

        {
            LocalTensor<KV_T> aL1Tensor = inputQueL1A.DeQue<KV_T>();
            LocalTensor<KV_T> bL1Tensor = inputQueL1B.DeQue<KV_T>();
            constexpr uint32_t baseK = 256 / sizeof(KV_T);
            uint32_t kLoopTimes = (kActCopyRowCountAlign + baseK - 1) / baseK;
            uint32_t kTailAlign = kActCopyRowCountAlign - (kLoopTimes - 1) * baseK;
            uint32_t kTail = kActCopyRowCount - (kLoopTimes - 1) * baseK;
            for (uint32_t i = 0, actualBaseKAlign = baseK, actualBaseK = baseK; i < kLoopTimes; i++) {
                if (i + 1 == kLoopTimes) {
                    actualBaseKAlign = kTailAlign;
                    actualBaseK = kTail;
                }
                LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[l0aBufferManager.GetBuffer()];
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[l0bBufferManager.GetBuffer()];
                LocalTensor<KV_T> curAL1Tensor = aL1Tensor[16 * baseK * i];

                uint32_t mmRowCount = msdPreIterNum * gSize;
                uint32_t copyStrideL0 = 16 * actualBaseKAlign;
                uint32_t copyStrideL1 = 16 * kActCopyRowCountAlign;
                uint32_t copyIterNum = (mmRowCount + 15) / 16;
                for(int i = 0; i < copyIterNum; i++){
                    LoadDataMm2A(aL0Tensor[i * copyStrideL0], curAL1Tensor[i * copyStrideL1], actualBaseKAlign);
                }

                l0aBufferManager.EnQue();

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
                l0bBufferManager.EnQue();

                l0aBufferManager.DeQue();
                l0bBufferManager.DeQue();
                MmadParams mmadParams;
                mmadParams.m = msdPreIterNum * gSize;
                if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = 128;
                mmadParams.k = actualBaseK; // 无效数据不参与计算
                mmadParams.cmatrixInitVal = (kCopyIdx == 0) && (i == 0);
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                PipeBarrier<PIPE_M>();
                l0aBufferManager.ReleaseBuffer();
                l0bBufferManager.ReleaseBuffer();
            }

            inputQueL1A.FreeTensor(aL1Tensor);
            inputQueL1B.FreeTensor(bL1Tensor);
        }
    }

    l0cBufferManager.EnQue();
    l0cBufferManager.DeQue();
    FixpipeParamsV220 fixParams;
    fixParams.nSize = 128;
    fixParams.mSize = msdPreIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
    fixParams.srcStride = ((fixParams.mSize + 15) / 16) * 16;
    fixParams.dstStride = 128;
    fixParams.ndNum = 1;
    Fixpipe(mm2ResGm[(loop % (PRE_LOAD_NUM)) * bmm2ResUbSize], cL0Tensor, fixParams);
    l0cBufferManager.ReleaseBuffer();
}
#else
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ComputeMm2(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    gSize = gSizeCube;
    if (info.isChangeBatch) {
        InitValueGm(info.bIdx);
    }

    if (info.needSetOrgShape) {
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND || PAGE_ATTENTION) {
            bmm2.SetOrgShape(msdPreIterNum * gSize, kvHeadNum * headDim,
                info.actualSingleProcessSInnerSizeAlign, tilingData->baseParams.seqSize, headDimAlign);
        } else {
            bmm2.SetOrgShape(msdPreIterNum * gSize, headDim,
                info.actualSingleProcessSInnerSizeAlign, tilingData->baseParams.seqSize, headDimAlign);
        }
    }
    bmm2.SetTensorA(vec1ResGm[(loop % (PRE_LOAD_NUM)) * mmResUbSize]);
    bmm2.SetTensorB(valueGm[info.tensorBOffset]);
    if constexpr (KVINT4) {
        if (info.actualSinglePreProcessSInnerSize % 2 == 1) {
            bmm2.SetTail(msdPreIterNum * gSize, headDim, info.actualSinglePreProcessSInnerSize + 1);
        } else {
            bmm2.SetTail(msdPreIterNum * gSize, headDim, info.actualSinglePreProcessSInnerSize);
        }
    } else {
        bmm2.SetTail(msdPreIterNum * gSize, headDim, info.actualSinglePreProcessSInnerSize);
    }
    bmm2.template IterateAll<false>(mm2ResGm[(loop % (PRE_LOAD_NUM)) * bmm2ResUbSize], false, false, true);
    bmm2.End();
}
#endif

template <typename IFAT>
__aicore__ inline bool IncreFlashAttentionAttenPreload<IFAT>::IsFinish(uint32_t loop)
{
    return (loop >= bn2s2LoopTimes);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::Process()
{
    if (aiCoreIdx < usedPreCoreNum) {
        SetLoopTimes();
        if ASCEND_IS_AIV {
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, gSize * 32 / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, gSize * 32 / sizeof(T));
        } else {
#ifndef USE_MM_API
            l0aBufferManager.Init();
            l0bBufferManager.Init();
            l0cBufferManager.Init();
#endif
        }
        for (uint32_t i = 0; i < bn2s2LoopTimes; i = i + PRE_LOAD_NUM) {
            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;
                if (i != 0) {
                    if (!IsFinish(loop - PRE_LOAD_NUM)) {
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(PRE_SYNC_C2_V2_FLAG);
                            ProcessVec2L(loop - PRE_LOAD_NUM);
                            DO_DUMP_DATA(vec2ResGm[((loop - PRE_LOAD_NUM) % PRE_LOAD_NUM) * bmm2ResUbSize + gSizeStart * headDimAlign], 5, gSizeVector * headDimAlign);
                        }
                    }
                }
                if (!IsFinish(loop)) {
                    CalcParams(loop);
                    if ASCEND_IS_AIV {
                        if constexpr (ANTIQUANT_PRE) {
                            QueryPreProcessL(loop);
                            CrossCoreSetFlag<PRE_SYNC_MODE2, PIPE_MTE3>(PRE_SYNC_V0_C1_FLAG);
                            DO_DUMP_DATA(queryPreProcessResGm[(extraInfo[loop % PRE_LOAD_NUM].bn2IdxInCurCore % (PRE_LOAD_NUM)) * bmm2ResUbSize + gSizeStart * headDimAlign], 1, gSizeVector * headDimAlign);
                            DO_DUMP_DATA(queryPreProcessResGm[(extraInfo[loop % PRE_LOAD_NUM].bn2IdxInCurCore % (PRE_LOAD_NUM)) * bmm2ResUbSize + (gSize + gSizeStart) * headDimAlign], 11, gSizeVector * headDimAlign);
                        }
                    }
                    if ASCEND_IS_AIC {
                        if constexpr (ANTIQUANT_PRE) {
                            CrossCoreWaitFlag(PRE_SYNC_V0_C1_FLAG);
                        }
                        ComputeMm1(loop);
                        DO_DUMP_DATA(mm1ResGm[(loop % PRE_LOAD_NUM) * mmResUbSize], 2, msdPreIterNum * gSize * extraInfo[loop % PRE_LOAD_NUM].actualSingleProcessSInnerSizeAlign);
                        CrossCoreSetFlag<PRE_SYNC_MODE2, PIPE_FIX>(PRE_SYNC_C1_V1_FLAG);
                    }
                }
            }

            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;
                if (!IsFinish(loop)) {
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag(PRE_SYNC_C1_V1_FLAG);
                        ProcessVec1L(loop);
                        CrossCoreSetFlag<PRE_SYNC_MODE2, PIPE_MTE3>(PRE_SYNC_V1_C2_FLAG);
                        DO_DUMP_DATA(vec1ResGm[(loop % PRE_LOAD_NUM) * mmResUbSize + gSizeStart * extraInfo[loop % PRE_LOAD_NUM].actualSingleProcessSInnerSizeAlign], 3, gSizeVector * extraInfo[loop % PRE_LOAD_NUM].actualSingleProcessSInnerSizeAlign);
                        DO_DUMP_DATA(vec1ResGm[(loop % PRE_LOAD_NUM) * mmResUbSize + (gSize + gSizeStart) * extraInfo[loop % PRE_LOAD_NUM].actualSingleProcessSInnerSizeAlign], 31, gSizeVector * extraInfo[loop % PRE_LOAD_NUM].actualSingleProcessSInnerSizeAlign);
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag(PRE_SYNC_V1_C2_FLAG);
                        ComputeMm2(loop);
                        DO_DUMP_DATA(mm2ResGm[(loop % PRE_LOAD_NUM) * bmm2ResUbSize], 4, msdPreIterNum * gSize * headDimAlign);
                        CrossCoreSetFlag<PRE_SYNC_MODE2, PIPE_FIX>(PRE_SYNC_C2_V2_FLAG);
                    }
                }
            }

            if (i + PRE_LOAD_NUM >= bn2s2LoopTimes) {
                for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                    uint32_t loop = i + j;
                    if (!IsFinish(loop)) {
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(PRE_SYNC_C2_V2_FLAG);
                            ProcessVec2L(loop);
                            DO_DUMP_DATA(vec2ResGm[(loop % PRE_LOAD_NUM) * bmm2ResUbSize + gSizeStart * headDimAlign], 5, gSizeVector * headDimAlign);
                        }
                    }
                }
            }
        }
#ifndef USE_MM_API
        if ASCEND_IS_AIC {
            l0aBufferManager.Destory();
            l0bBufferManager.Destory();
            l0cBufferManager.Destory();
        }
#endif
    }
    if (flashDecodeFlag) {
        if constexpr (FLASH_DECODE) {
            // 多核同步
            SyncAll();

            if ASCEND_IS_AIV {
                FlashDecodeCompute();
            }
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::ProcessSysPrefixCombine()
{
    // 多核同步
    SyncAll();

    if (tmpPreBlockIdx >= usedPreCoreNumSp) {
        return;
    }

    bn2LoopTimes = blockSplitBn2RangeSp;
    beforeBlockPreSplitBn2Nums = tmpPreBlockIdx * blockSplitBn2RangeSp;
    // tail cores
    if (tmpPreBlockIdx >= formerCoreNumSp) {
        bn2LoopTimes = tailBlockSplitBn2RangeSp;
        beforeBlockPreSplitBn2Nums =
            formerCoreNumSp * blockSplitBn2RangeSp + (tmpPreBlockIdx - formerCoreNumSp) * tailBlockSplitBn2RangeSp;
    }

    for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
        bIdx = (beforeBlockPreSplitBn2Nums + bn2Idx) / kvHeadNum;
        n2Idx = (beforeBlockPreSplitBn2Nums + bn2Idx) % kvHeadNum;
        SysPrefixAttenResCombine();
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyDataInByQueue1(LocalTensor<T> &dst,
                                                                                        const GlobalTensor<T> &src,
                                                                                        size_t size)
{
    dst = inputQue1.AllocTensor<T>();
    DataCopy(dst, src, size);
    inputQue1.EnQue(dst);
    inputQue1.DeQue<T>();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyDataInByQueue2(LocalTensor<T> &dst,
                                                                                        const GlobalTensor<T> &src,
                                                                                        size_t size)
{
    dst = inputQue2.AllocTensor<T>();
    DataCopy(dst, src, size);
    inputQue2.EnQue(dst);
    inputQue2.DeQue<T>();
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixAttenResCombine()
{
    size_t lseSize = 2 * gSize * FP32_ONE_BLOCK_SIZE;
    size_t preBn2 = bIdx * kvHeadNum + n2Idx;

    // InQueue2 is used by bf16 PostQuant, antiqScale ub is not used now, reuse it
    LocalTensor<T> lseSum = antiqScaleBuff.Get<T>(BUFFER_SIZE_BYTE_4K);
    CopyGmToFixedUb(lseSum, lseSumGm[preBn2 * lseSize], lseSize);
    LocalTensor<T> lseMax;
    CopyDataInByQueue1(lseMax, lseMaxGm[preBn2 * lseSize], lseSize);

    SysPrefixLseToScales(lseSum, lseMax);
    inputQue1.FreeTensor(lseMax);

    uint64_t attenOffset = preBn2 * gSize * headDimAlign;
    GlobalTensor<T> atten1 = sysPrefixAttenOutGm[attenOffset];
    GlobalTensor<T> atten2 = usrPromptAttenOutGm[attenOffset];
    LocalTensor<T> attenRes = tmpBuff1.Get<T>(BUFFER_SIZE_BYTE_32K);
    GlobalTensor<OUT_T> attenOutGm = attentionOutGm[preBn2 * gSize * headDim];

    uint32_t gPreSplitSize = BUFFER_SIZE_BYTE_32K / (headDimAlign * sizeof(T));
    uint32_t loops = (gSize + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t gTailSize = gSize - (loops - 1) * gPreSplitSize;

    for (uint32_t i = 0; i < loops; i++) {
        uint32_t rows = (i == loops - 1) ? gTailSize : gPreSplitSize;
        auto lseScale = lseSum[i * gPreSplitSize * FP32_ONE_BLOCK_SIZE];
        SysPrefixAttenReduce(attenRes, atten1, atten2, lseScale, i * gPreSplitSize, rows);
        SysPrefixAttenOutput(attenOutGm, attenRes, i * gPreSplitSize, rows);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::SysPrefixAttenReduce(LocalTensor<T> &dst, GlobalTensor<T> &atten1Gm,
                                                                   GlobalTensor<T> &atten2Gm, LocalTensor<T> scales,
                                                                   uint32_t startRow, uint32_t rows)
{
    uint64_t attenPreOffset = startRow * headDimAlign;
    size_t attenPreSize = rows * headDimAlign;
    LocalTensor<T> atten1;
    CopyDataInByQueue1(atten1, atten1Gm[attenPreOffset], attenPreSize);

    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = (headDimAlign * sizeof(T)) / BYTE_BLOCK;
    repeatParams.dstRepStride = (headDimAlign * sizeof(T)) / BYTE_BLOCK;
    uint64_t tempMask = 256 / sizeof(T);
    uint32_t loops = (headDimAlign + tempMask - 1) / tempMask;
    uint32_t tail = headDimAlign - (loops - 1) * tempMask;

    // 第一次，mul结果直接放到dst里
    for (uint32_t i = 0; i < loops; i++) {
        Mul(dst[i * tempMask], scales, atten1[i * tempMask], (i != loops - 1) ? tempMask : tail, rows, repeatParams);
    }
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(atten1);

    LocalTensor<T> atten2;
    CopyDataInByQueue1(atten2, atten2Gm[attenPreOffset], attenPreSize);
    LocalTensor<T> scales2 = scales[gSize * FP32_ONE_BLOCK_SIZE];
    for (uint32_t i = 0; i < loops; i++) {
        Mul(atten2[i * tempMask], scales2, atten2[i * tempMask], (i != loops - 1) ? tempMask : tail, rows, repeatParams);
    }

    PipeBarrier<PIPE_V>();
    Add(dst, dst, atten2, rows * headDimAlign);
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(atten2);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLseToScales(LocalTensor<T> &lseSum,
                                                                                          LocalTensor<T> &lseMax)
{
    size_t lsePreBlockSize = gSize * FP32_ONE_BLOCK_SIZE;
    LocalTensor<T> tmpMax = tmpBuff1.Get<T>(2 * lsePreBlockSize + 2 * lsePreBlockSize);
    LocalTensor<T> tmpSum = tmpMax[lsePreBlockSize];
    LocalTensor<T> tmpExp = tmpSum[lsePreBlockSize];

    LocalTensor<T> lseMax1 = lseMax[0];
    LocalTensor<T> lseMax2 = lseMax[lsePreBlockSize];

    Max(tmpMax, lseMax1, lseMax2, lsePreBlockSize);
    PipeBarrier<PIPE_V>();

    Sub(tmpExp, lseMax1, tmpMax, lsePreBlockSize);
    Sub(tmpExp[lsePreBlockSize], lseMax2, tmpMax, lsePreBlockSize);
    PipeBarrier<PIPE_V>();

    Exp(tmpExp, tmpExp, 2 * lsePreBlockSize);
    PipeBarrier<PIPE_V>();

    Mul(lseSum, lseSum, tmpExp, 2 * lsePreBlockSize);
    PipeBarrier<PIPE_V>();

    Add(tmpSum, lseSum[0], lseSum[lsePreBlockSize], lsePreBlockSize);
    PipeBarrier<PIPE_V>();

    Div(lseSum[0], lseSum[0], tmpSum, lsePreBlockSize);
    Div(lseSum[lsePreBlockSize], lseSum[lsePreBlockSize], tmpSum, lsePreBlockSize);
    PipeBarrier<PIPE_V>();

    if (softmaxLseFlag) {
        Log(tmpSum, tmpSum, lsePreBlockSize);
        PipeBarrier<PIPE_V>();
        Add(tmpSum, tmpSum, tmpMax, lsePreBlockSize);
        PipeBarrier<PIPE_V>();

        SoftmaxLseOutput(tmpSum);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::SysPrefixAttenOutput(GlobalTensor<OUT_T> &dst, LocalTensor<T> &attenRes,
                                                                   uint32_t startRow, uint32_t rows)
{
    LocalTensor<OUT_T> attenPreOut = outputQue1.AllocTensor<OUT_T>();
    if constexpr (!POST_QUANT) {
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(attenPreOut, attenRes, AscendC::RoundMode::CAST_RINT, rows * headDimAlign);
        } else {
            Cast(attenPreOut, attenRes, AscendC::RoundMode::CAST_ROUND, rows * headDimAlign);
        }
    } else {
        perChannelQuantOffset = n2Idx * headDim * gSize;
    }

    outputQue1.EnQue(attenPreOut);
    outputQue1.DeQue<OUT_T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = rows;
    dataCopyParams.blockLen = headDim * sizeof(OUT_T);
    dataCopyParams.srcStride = ((headDimAlign - headDim) * sizeof(OUT_T)) / BYTE_BLOCK;
    dataCopyParams.dstStride = 0;
    DataCopyPad(dst[startRow * headDim], attenPreOut, dataCopyParams);
    outputQue1.FreeTensor(attenPreOut);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveLse(uint32_t bIndex, uint32_t n2Index,
                                                                                      LocalTensor<T> &softmaxSumUb,
                                                                                      LocalTensor<T> &softmaxMaxUb,
                                                                                      uint32_t start, uint32_t count,
                                                                                      bool isPrefix)
{
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveLseFA()
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveLseFd(LocalTensor<T> &lseSum,
                                                                                        LocalTensor<T> &lseMax,
                                                                                        uint32_t start, uint32_t count)
{
    SysPrefixSaveLse(bIdx, n2Idx, lseSum, lseMax, start, count, calcSysPrefixFlag);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveZeroLse(uint32_t bIndex, uint32_t n2Index, bool isPrefix)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveZeroAttenRes(uint32_t bIndex,
                                                                                               uint32_t n2Index,
                                                                                               bool isPrefix)
{
    uint64_t attenOffset = (bIndex * kvHeadNum + n2Index) * gSize * headDimAlign;
    GlobalTensor<T> dst = isPrefix ? sysPrefixAttenOutGm[attenOffset] : usrPromptAttenOutGm[attenOffset];
    matmul::InitOutput<T>(dst, gSize * headDimAlign, 0);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixInitAllZeroOutput()
{
    if (calcSysPrefixFlag) {
        for (uint32_t k = 0; k < batchSizeQ; k++) {
            SysPrefixSaveZeroAttenRes(k, n2Idx, true);
            SysPrefixSaveZeroLse(k, n2Idx, true);
        }
    } else {
        SysPrefixSaveZeroAttenRes(bIdx, n2Idx, false);
        SysPrefixSaveZeroLse(bIdx, n2Idx, false);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveAttenRes(
    uint32_t bIndex, uint32_t n2Index, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t rows, bool isPrefix)
{
    LocalTensor<T> outputUb = outputQue1.template AllocTensor<T>();
    DataCopy(outputUb, bmm2ResUb, rows * headDimAlign);

    uint64_t attenOffset = (bIndex * kvHeadNum + n2Index) * gSize * headDimAlign + startRow * headDimAlign;
    GlobalTensor<T> dst = isPrefix ? sysPrefixAttenOutGm[attenOffset] : usrPromptAttenOutGm[attenOffset];

    outputQue1.EnQue(outputUb);
    outputQue1.DeQue();
    DataCopy(dst, outputUb, rows * headDimAlign);
    outputQue1.FreeTensor(outputUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SoftmaxLseOutput(LocalTensor<T> &lse)
{
    LocalTensor<T> softmaxlsePreOut = outputQue2.template AllocTensor<T>();
    DataCopy(softmaxlsePreOut, lse, gSize * FP32_ONE_BLOCK_SIZE);
    outputQue2.EnQue(softmaxlsePreOut);
    outputQue2.DeQue<T>();

    DataCopyExtParams param;
    param.blockLen = sizeof(T);
    param.blockCount = gSize;
    param.srcStride = 0;
    param.dstStride = 0;
    DataCopyPad(softmaxLseGm[(bIdx * kvHeadNum + n2Idx) * gSize], softmaxlsePreOut, param);
    outputQue2.FreeTensor(softmaxlsePreOut);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyFixedUbToGm(const GlobalTensor<T> &dst,
                                                                                     const LocalTensor<T> &src,
                                                                                     size_t size)
{
    LocalTensor<T> tmpTensor = outputQue2.template AllocTensor<T>();
    DataCopy(tmpTensor, src, size);

    outputQue2.EnQue(tmpTensor);
    outputQue2.DeQue();
    DataCopy(dst, tmpTensor, size);
    outputQue2.FreeTensor(tmpTensor);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::CopyGmToFixedUb(LocalTensor<T> &dst,
                                                                                     const GlobalTensor<T> &src,
                                                                                     size_t size)
{
    LocalTensor<T> tmpTensor = inputQue2.AllocTensor<T>();
    DataCopy(tmpTensor, src, size);
    inputQue2.EnQue(tmpTensor);
    inputQue2.DeQue<T>();
    DataCopy(dst, tmpTensor, size);
    inputQue2.FreeTensor(tmpTensor);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveMsdMax1(uint32_t bIndex)
{
    auto dst = msdRowMax1Gm[bIndex * msdRowPreMaxSize];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLoadMsdMax1(uint32_t bIndex)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveMsdMax2(uint32_t bIndex)
{
    auto dst = msdRowMax2Gm[bIndex * msdRowPreMaxSize];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLoadMsdMax2(uint32_t bIndex)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveMsdSum1(uint32_t bIndex)
{
    auto dst = msdRowSum1Gm[bIndex * msdRowPreSumSize];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLoadMsdSum1(uint32_t bIndex)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveMsdSum2(uint32_t bIndex)
{
    auto dst = msdRowSum2Gm[bIndex * msdRowPreSumSize];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLoadMsdSum2(uint32_t bIndex)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveSoftmaxMax(uint32_t bIndex)
{
    auto dst = softmaxRowMaxGm[bIndex * softmaxMaxSumPreExpSize];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLoadSoftmaxMax(uint32_t bIndex)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveSoftmaxSum(uint32_t bIndex)
{
    auto dst = softmaxRowSumGm[bIndex * softmaxMaxSumPreExpSize];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLoadSoftmaxSum(uint32_t bIndex)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixSaveSoftmaxExp(uint32_t bIndex)
{
    auto dst = softmaxRowExpGm[bIndex * softmaxMaxSumPreExpSize];
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::SysPrefixLoadSoftmaxExp(uint32_t bIndex)
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreload<IFAT>::DealKvInt4ColumnOdd(LocalTensor<T> mmResUb,
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