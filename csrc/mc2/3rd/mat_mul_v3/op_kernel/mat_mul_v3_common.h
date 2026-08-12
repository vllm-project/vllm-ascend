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
 * \file mat_mul_v3_common.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_COMMON_H__
#define __OP_KERNEL_MATMUL_V3_COMMON_H__
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "utils/std/algorithm.h"
#include "lib/matmul_intf.h"
#include "mat_mul_v3_tiling_data.h"


using namespace AscendC;
using namespace matmul;
#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

const uint64_t NUM_TWO = 2;
const uint64_t NUM_FOUR = 4;
const uint64_t NUM_256 = 256;
const uint64_t DATA_SIZE_FP16 = 2;
const uint64_t DATA_SIZE_FP32 = 4;
const uint64_t BANK_SIZE = 256;
const uint64_t ALIGN_BYTE = 256;
const uint64_t ALIGN_128_BYTE = 128;
const uint64_t MAX_BLOCK_NUM = 100;
const uint64_t DEFAULT_BLOCK_LEN = 8;
const uint64_t BLOCK_SIZE = 16;
constexpr uint64_t NUM_AIV_TO_AIC_RATIO = 2;

constexpr uint64_t ALIGNED_H = 16;

const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;

const uint32_t CONTROL_DB = 1;
const uint32_t ALL_L2_CACHE_ENABLE = 1;
const uint32_t C_L2_DISABLE = 16;

const uint64_t AIV_SYNC_AIC_FLAG = 4;
const uint64_t AIC_SYNC_AIV_FLAG = 6;

constexpr uint64_t BLOCK_BYTE_SIZE = 32;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
const uint8_t AIC_SYNC_AIV_MODE_4 = 4;
const uint64_t VEC0_FLAG_ID_OFFSET = 0;
const uint64_t VEC1_FLAG_ID_OFFSET = 16;
#endif
// common MDL config
constexpr MatmulConfig MM_CFG_MDL = GetMDLConfig();
// set isVecND2Nz
constexpr MatmulConfig MM_CFG_VEC_ND2NZ = GetMDLConfig(false, false, 0, true);
// set enUnitFlag
constexpr MatmulConfig MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, true);
// set doMTE2Preload
constexpr MatmulConfig MM_CFG_PRELOAD_MK = GetMDLConfig(false, false, 2);
constexpr MatmulConfig MM_CFG_PRELOAD_NK = GetMDLConfig(false, false, 1);
// set enMultiBatch Info
constexpr MatmulBatchParams batchBiasParams{false, BatchMode::BATCH_LESS_THAN_L1, true, BatchOutMode::DYNAMIC};
constexpr MatmulConfig MM_CFG_MULTI_BATCH_OUT_BATCH_BIAS = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(batchBiasParams);
constexpr MatmulBatchParams singBiasParams{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::DYNAMIC};
constexpr MatmulConfig MM_CFG_MULTI_BATCH_OUT_SING_BIAS = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(singBiasParams);
// set ORDER_M
constexpr MatmulConfig MM_CFG_ORDER_M =
    GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, true, IterateOrder::ORDER_M);
// set enMultiBatch & ORDER_M
constexpr MatmulConfig MM_CFG_MULTI_BATCH_OUT =
    GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, true, IterateOrder::ORDER_M,
                    ScheduleType::INNER_PRODUCT, true, false, BatchOutMode::MULTI_BATCH);

enum class ND2NZ_SELECT : int32_t {
    ONLY_A = 1,
    ONLY_B = 2,
    BOTH_AB = 3
};

enum class FIXPIPE_OPT_SELECT : int32_t {
    BASE = 0,
    BASE_ENABLE_ALIGNOUT = 1,
    VEC_NZ2ND_UNALIGNOUT = 2
};

#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

template <HardEvent event>
__aicore__ inline void TPipeSetWaitFlag()
{
    auto eventID = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

template <class T>
__aicore__ inline void GetSizeC0(uint64_t &c0Size)
{
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }
}

/**
 * if b is 0, return a
 */
__aicore__ inline uint64_t MMV3DivCeil(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * if b is 0, return 0
 */
__aicore__ inline uint64_t MMV3CeilAlign(uint64_t a, uint64_t b)
{
    return MMV3DivCeil(a, b) * b;
}

/**
 * if b is 0, return a
 */
__aicore__ inline uint64_t MMV3DivFloor(uint64_t a, uint64_t b)
{
    return b == 0 ? a : a / b;
}

/**
 * if b is 0, return 0
 */
__aicore__ inline uint64_t MMV3FloorAlign(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : a / b * b;
}

__aicore__ inline uint64_t GetCurrentBlockIdx()
{
    if ASCEND_IS_AIV {
        return GetBlockIdx() / GetTaskRation();
    }
    return GetBlockIdx();
}

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
__aicore__ inline uint64_t MMLcm(uint64_t m, uint64_t n)
{
    if (m == 0 || n == 0) {
        return 0; // 处理输入为0的情况
    }
    uint64_t total = m * n;
    uint64_t tmp = 0;
    while (n != 0) {
        tmp = m % n;
        m = n;
        n = tmp;
    }
    return total / m;
}

__aicore__ inline void WaitFlagDevLocal(int64_t flagID)
{
    CrossCoreWaitFlag(flagID);
}

// supportMmadS8S4平台无L2cacheUseInfo，用宏隔离
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
template <class A_T, class B_T, class C_T, class BiasT>
__aicore__ inline void SetL2CacheEnable(const Mc2L2cacheUseInfo &l2EnableInfo, GlobalTensor<A_T> &aGlobal,
                                        GlobalTensor<B_T> &bGlobal, GlobalTensor<C_T> &cGlobal,
                                        GlobalTensor<BiasT> &biasGlobal)
{
    if ((l2EnableInfo.l2CacheFlag & ALL_L2_CACHE_ENABLE) == 0) {
        if (l2EnableInfo.l2CacheFlag & C_L2_DISABLE) {
            cGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
    }
}
#endif

template <class T>
__aicore__ inline void CopyGmToUbufAlign(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint16_t nBurst,
                                         uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                                         uint32_t srcGap, uint32_t dstGap)
{
    DataCopyExtParams dataCopyExtParams{nBurst, lenBurst, srcGap, dstGap, 0};
    DataCopyPadExtParams<T> dataCopyPadExtParams{false, leftPaddingNum, rightPaddingNum, static_cast<T>(0)};
    DataCopyPad(dst, src, dataCopyExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyUbufToGmAlign(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint16_t nBurst,
                                         uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap)
{
    DataCopyExtParams dataCopyExtParams{nBurst, lenBurst, srcGap, dstGap, 0};
    DataCopyPad(dst, src, dataCopyExtParams);
}

template <typename T>
__aicore__ inline void CopyCast(const LocalTensor<float> &ubSrc, const LocalTensor<T> &ubDst,
                                const GlobalTensor<float> &src, const GlobalTensor<T> &dst, uint64_t offset,
                                uint16_t nBurst, uint16_t lenBurst, uint32_t gap, uint8_t pingpongEventId)
{
    CopyGmToUbufAlign<float>(ubSrc, src[offset], nBurst, lenBurst * sizeof(float), 0, 0, gap * sizeof(float), 0);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingpongEventId));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingpongEventId));
    Cast(ubDst, ubSrc, RoundMode::CAST_RINT, nBurst * lenBurst);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingpongEventId));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingpongEventId));
    CopyUbufToGmAlign<T>(dst[offset], ubDst, nBurst, lenBurst * sizeof(T), 0, gap * sizeof(T));
}

#define DBCAST
#ifdef DBCAST
// v220
template <typename T>
__aicore__ inline void Cast32to16V220(__gm__ T *dst, __gm__ float *src, uint64_t size, uint32_t nCoreUse, uint32_t n,
                                      TBuf<TPosition::VECCALC> &tmpBuf)
{
    uint16_t dataSize = static_cast<uint16_t>(TOTAL_UB_SIZE / NUM_TWO / sizeof(float));
    uint16_t dataSize1 = static_cast<uint16_t>(TOTAL_UB_SIZE / NUM_TWO / sizeof(T));
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<T> ubDstPong = ubDstPing[dataSize1];
    LocalTensor<float> ubSrcPong = ubSrcPing[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc = ubSrcPing;

    GlobalTensor<float> gmSrc;
    GlobalTensor<T> gmDst;
    gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), size);
    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst), size);

    uint8_t pingpongEventId = 0;

    if (nCoreUse >= dataSize) { // 操作的最小一行数据量超ub时，对于每一行数据循环做cast处理
        uint64_t mRepeat = size / nCoreUse;
        uint16_t nBurst = static_cast<uint16_t>(nCoreUse / dataSize);
        uint16_t tail = static_cast<uint16_t>(nCoreUse % dataSize);

        for (uint64_t i = 0; i < mRepeat; ++i) {
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
            for (uint16_t j = 0; j < nBurst; ++j) {
                if ((j & CONTROL_DB) == 0) {
                    pingpongEventId = 0;
                    ubDst = ubDstPing;
                    ubSrc = ubSrcPing;
                } else {
                    pingpongEventId = 1;
                    ubDst = ubDstPong;
                    ubSrc = ubSrcPong;
                }
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
                CopyCast<T>(ubSrc, ubDst, gmSrc, gmDst, i * n + j * dataSize, 1, dataSize, 0, pingpongEventId);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
            }
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
            if (tail > 0) {
                CopyCast<T>(ubSrc, ubDst, gmSrc, gmDst, i * n + nBurst * dataSize, 1, tail, 0, pingpongEventId);
            }
        }
        return;
    }

    uint16_t nBurst = static_cast<uint16_t>(dataSize / nCoreUse);
    uint64_t repeat = size / (nBurst * nCoreUse);
    uint16_t tail = static_cast<uint16_t>(size % (nBurst * nCoreUse));

    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    for (uint64_t i = 0; i < repeat; ++i) {
        if ((i & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            ubDst = ubDstPing;
            ubSrc = ubSrcPing;
        } else {
            pingpongEventId = 1;
            ubDst = ubDstPong;
            ubSrc = ubSrcPong;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
        CopyCast<T>(ubSrc, ubDst, gmSrc, gmDst, i * nBurst * n, nBurst, static_cast<uint16_t>(nCoreUse), n - nCoreUse,
                    pingpongEventId);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    if ((repeat & CONTROL_DB) == 0) {
        pingpongEventId = 0;
        ubDst = ubDstPing;
        ubSrc = ubSrcPing;
    } else {
        pingpongEventId = 1;
        ubDst = ubDstPong;
        ubSrc = ubSrcPong;
    }
    if (tail > 0) {
        uint16_t tailNBurst = static_cast<uint16_t>(tail / nCoreUse);
        CopyCast<T>(ubSrc, ubDst, gmSrc, gmDst, repeat * nBurst * n, tailNBurst, static_cast<uint16_t>(nCoreUse),
                    n - nCoreUse, pingpongEventId);
    }
    return;
}

template <typename T>
__aicore__ inline void UnAlignedCast32to16V220(__gm__ T *dst, __gm__ float *src, uint32_t offset, uint32_t size,
                                               TBuf<TPosition::VECCALC> &tmpBuf)
{
    uint32_t dataSize = TOTAL_UB_SIZE / NUM_TWO / sizeof(float);
    uint32_t dataSize1 = TOTAL_UB_SIZE / NUM_TWO / sizeof(T);
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<T> ubDstPong = ubDstPing[dataSize1];
    LocalTensor<float> ubSrcPong = ubSrcPing[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc = ubSrcPing;

    GlobalTensor<float> gmSrc;
    GlobalTensor<T> gmDst;
    gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), size);
    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst), size);

    uint32_t repeat = size / dataSize;
    uint32_t tail = size % dataSize;

    uint8_t pingpongEventId = 0;

    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));

    for (uint32_t i = 0; i < repeat; ++i) {
        if ((i & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            ubDst = ubDstPing;
            ubSrc = ubSrcPing;
        } else {
            pingpongEventId = 1;
            ubDst = ubDstPong;
            ubSrc = ubSrcPong;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
        CopyCast<T>(ubSrc, ubDst, gmSrc, gmDst, i * dataSize, 1, dataSize, 0, pingpongEventId);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    if ((repeat & CONTROL_DB) == 0) {
        pingpongEventId = 0;
        ubDst = ubDstPing;
        ubSrc = ubSrcPing;
    } else {
        pingpongEventId = 1;
        ubDst = ubDstPong;
        ubSrc = ubSrcPong;
    }
    if (tail > 0) {
        CopyCast<T>(ubSrc, ubDst, gmSrc, gmDst, repeat * dataSize, 1, tail, 0, pingpongEventId);
    }
    return;
}

#endif


template <typename T1, typename T2>
__aicore__ inline void CopyRemovePad(const GlobalTensor<T2> &outputGlobal, const GlobalTensor<T1> &inputGlobal,
                                     const LocalTensor<T1> &srcUb, const LocalTensor<T2> &castDstUb, uint32_t nBurst,
                                     uint32_t inputWidth, uint32_t outputWidth)
{
    CopyGmToUbufAlign<T1>(srcUb, inputGlobal, static_cast<uint16_t>(nBurst), inputWidth * sizeof(T1), 0, 0, 0, 0);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    Cast(castDstUb, srcUb, RoundMode::CAST_RINT, nBurst * inputWidth);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    uint32_t srcGap = (inputWidth - outputWidth) * sizeof(T2) / 32; // 32 is blocksize
    CopyUbufToGmAlign<T2>(outputGlobal, castDstUb, static_cast<uint16_t>(nBurst), outputWidth * sizeof(T2), srcGap, 0);
}

template <typename T1, typename T2>
__aicore__ inline void RemovePaddingImpl(GlobalTensor<T2> outputGlobal, GlobalTensor<T1> inputGlobal, uint32_t height,
                                         uint32_t width, uint32_t outputWidth, TBuf<TPosition::VECCALC> &tmpBuf)
{
    LocalTensor<T1> srcUb = tmpBuf.Get<T1>();
    LocalTensor<T2> castDstUb = srcUb.template ReinterpretCast<T2>();

    uint32_t nBurst = TOTAL_UB_SIZE / (width * sizeof(T1));
    if (nBurst == 0) {
        uint32_t maxWidthLen = TOTAL_UB_SIZE / sizeof(T1);
        uint32_t castTimes = width / maxWidthLen;
        uint32_t tailWidth = width - castTimes * maxWidthLen;
        uint32_t tailOutWidth = outputWidth - castTimes * maxWidthLen;
        for (uint32_t mIndex = 0; mIndex < height; ++mIndex) {
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            for (uint32_t index = 0; index < castTimes; ++index) {
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                UnAlignedCast32to16V220(
                    (__gm__ T2 *)(outputGlobal[index * maxWidthLen + mIndex * outputWidth].GetPhyAddr()),
                    (__gm__ float *)(inputGlobal[index * maxWidthLen + mIndex * width].GetPhyAddr()), 0, maxWidthLen,
                    tmpBuf);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            }
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            if (tailWidth != 0) {
                CopyRemovePad(outputGlobal[castTimes * maxWidthLen + mIndex * outputWidth],
                              inputGlobal[castTimes * maxWidthLen + mIndex * width], srcUb, castDstUb, 1, tailWidth,
                              tailOutWidth);
            }
        }
        return;
    }
    uint32_t nBurstTimes = height / nBurst;
    uint32_t nBurstTail = height - nBurstTimes * nBurst;
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    for (uint32_t i = 0; i < nBurstTimes; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        CopyRemovePad(outputGlobal[i * nBurst * outputWidth], inputGlobal[i * nBurst * width], srcUb, castDstUb, nBurst,
                      width, outputWidth);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    if (nBurstTail > 0) {
        CopyRemovePad(outputGlobal[nBurstTimes * nBurst * outputWidth], inputGlobal[nBurstTimes * nBurst * width],
                      srcUb, castDstUb, nBurstTail, width, outputWidth);
    }
}

#endif
#endif // __OP_KERNEL_MATMUL_V3_H__
