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
 * \file mat_mul_deterministic_splitk_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_DETERMINISTIC_SPLITK_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_DETERMINISTIC_SPLITK_KERNEL_H__

#include "mat_mul_v3_common.h"

const uint64_t ND2NZ_AIV_SYNC_AIC_FLAG = 8;
const uint64_t MAX_NUM = 12 * 1024; // ub最大容纳的元素个数

__aicore__ inline uint64_t AlignTo256B(uint64_t base)
{
    uint64_t alignedSize = base * DATA_SIZE_FP32;
    // 如果alignedSize小于256字节，则直接调整为256字节
    if (alignedSize < ALIGN_BYTE) {
        return ALIGN_BYTE / DATA_SIZE_FP32;
    }
    // 否则，计算向上取整到256字节的倍数
    return MMV3DivCeil(alignedSize, ALIGN_BYTE) * ALIGN_BYTE / DATA_SIZE_FP32;
}

template <class A_TYPE, class B_TYPE>
__aicore__ inline void SetOffset(uint64_t &offsetA, uint64_t &offsetB, uint64_t mOffset, uint64_t nOffset,
                                 uint64_t kOffset, uint64_t c0Size, uint64_t alignedOriM, uint64_t alignedOriN,
                                 uint64_t alignedKaSize, uint64_t alignedKbSize, const TCubeTiling &tiling)
{
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        offsetA = A_TYPE::isTrans ? (kOffset * tiling.M + mOffset) : (kOffset + mOffset * tiling.Ka);
    } else {
        offsetA =
            A_TYPE::isTrans ? (kOffset * c0Size + mOffset * alignedKaSize) : (kOffset * alignedOriM + mOffset * c0Size);
    }
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        offsetB = B_TYPE::isTrans ? (nOffset * tiling.Kb + kOffset) : (nOffset + kOffset * tiling.N);
    } else {
        offsetB =
            B_TYPE::isTrans ? (nOffset * c0Size + kOffset * alignedOriN) : (nOffset * alignedKbSize + kOffset * c0Size);
    }
}

__aicore__ inline uint64_t ComputOffsetL2cache(uint64_t gmSrcOffset, uint64_t singleCoreN, uint64_t n, uint64_t nIndex,
                                               uint64_t singleCoreM, uint64_t mIndex)
{
    if (singleCoreN == 0) {
        return 0;
    }
    // nIndex 是B在N方向执行singleCoreN的次数
    uint64_t mIdx = (gmSrcOffset / singleCoreN + mIndex * singleCoreM);
    uint64_t nIdx = (gmSrcOffset % singleCoreN + nIndex * singleCoreN);
    uint64_t dstOffset = mIdx * n + nIdx;
    return dstOffset;
}

template <class T>
__aicore__ inline void
SplitKVectorProcess(LocalTensor<float> ubSrc1, LocalTensor<float> ubSrc2, GlobalTensor<float> gmSrc,
                    LocalTensor<T> ubDst, GlobalTensor<T> gmDst, uint64_t vIndex, uint64_t index, uint64_t currentLoop,
                    uint64_t dataSizeToMove, uint64_t dataSize, uint64_t coreSize, uint64_t singleSize,
                    uint64_t singleCoreNum, uint64_t singleCoreN, uint64_t n, uint64_t cnt, bool orderFlag)
{
    uint64_t dstOffset = 0;
    uint64_t burst = 1;
    uint64_t burstLen = dataSizeToMove;
    uint64_t srcGap = 0;
    uint64_t dstGap = 0;
    uint64_t nCoreTail = n - (cnt - 1) * singleCoreN;
    uint64_t gmOffset = currentLoop * dataSize + vIndex * coreSize;
    uint64_t tmpOffset = 0;

    if (orderFlag && index == cnt - 1) {
        uint64_t outNCoreTail = AlignTo256B(nCoreTail);
        burst = (dataSizeToMove / outNCoreTail);
        burstLen = nCoreTail;
        srcGap = (outNCoreTail - nCoreTail);
        dstGap = 0;
    }

    CopyGmToUbufAlign<float>(ubSrc1, gmSrc[gmOffset], burst, burstLen * sizeof(float), 0, 0, srcGap * sizeof(float),
                             dstGap * sizeof(float) / 32);

    for (uint64_t j = 1; j < singleCoreNum; ++j) {
        tmpOffset += (singleSize << 1);
        CopyGmToUbufAlign<float>(ubSrc2, gmSrc[gmOffset + tmpOffset], burst, burstLen * sizeof(float), 0, 0,
                                 srcGap * sizeof(float), dstGap * sizeof(float) / 32);
        // MTE2 to V, enable pingpong
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        Add(ubSrc1, ubSrc1, ubSrc2, burst * burstLen);
        TPipeSetWaitFlag<HardEvent::V_MTE2>();
    }
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(ubDst, ubSrc1, RoundMode::CAST_RINT, burst * burstLen);
    }
    TPipeSetWaitFlag<HardEvent::V_MTE3>();

    if (orderFlag) {
        dstOffset = ComputOffsetL2cache(gmOffset, singleCoreN, n, index, 0, 0);
        burst = (dataSizeToMove / singleCoreN);
        burstLen = singleCoreN;
        dstGap = (n - singleCoreN);

        // 尾列对齐256B
        if (index == cnt - 1) {
            uint64_t outNCoreTail = AlignTo256B(nCoreTail);
            burst = (dataSizeToMove / outNCoreTail);
            burstLen = nCoreTail;
            srcGap = 0;
            dstGap = n - nCoreTail;
            uint64_t mIdx = gmOffset / outNCoreTail;
            uint64_t nIdx = gmOffset % outNCoreTail + index * singleCoreN;
            dstOffset = mIdx * n + nIdx;
        }

        if constexpr (sizeof(T) == sizeof(half)) {
            CopyUbufToGmAlign<T>(gmDst[dstOffset], ubDst, burst, burstLen * sizeof(T), srcGap * sizeof(T) / 32,
                                 dstGap * sizeof(T)); // 32 is blocksize

        } else if constexpr (sizeof(T) == sizeof(float)) {
            CopyUbufToGmAlign<T>(gmDst[dstOffset], ubSrc1, burst, burstLen * sizeof(T), srcGap * sizeof(T) / 32,
                                 dstGap * sizeof(T)); // 32 is blocksize
        }
    } else {
        if constexpr (sizeof(T) == sizeof(half)) {
            CopyUbufToGmAlign<T>(gmDst[gmOffset + index * singleSize], ubDst, 1, dataSizeToMove * sizeof(T), 0, 0);
        } else if constexpr (sizeof(T) == sizeof(float)) {
            CopyUbufToGmAlign<T>(gmDst[gmOffset + index * singleSize], ubSrc1, 1, dataSizeToMove * sizeof(T), 0, 0);
        }
    }
}

template <class T>
__aicore__ inline void SplitKVectorNZProcess(GlobalTensor<float> gmSrc, GlobalTensor<T> gmDst, uint64_t copyElemNum,
                                             uint64_t currSplitN, uint64_t singleCoreNum, uint64_t singleSize,
                                             uint64_t oriN, LocalTensor<float> ubSrc1, LocalTensor<float> ubSrc2,
                                             LocalTensor<T> ubDst)
{
    uint64_t vIndex = GetBlockIdx();
    auto copySize = copyElemNum;
    uint64_t repeatNums = MMV3DivCeil(copySize, MAX_NUM);
    if (copySize > MAX_NUM) {
        copySize = MAX_NUM;
    }
    uint64_t offsetcopysize = 0;
    for (uint64_t repeat = 0; repeat < repeatNums; repeat++) {
        uint64_t tmpOffset = 0;
        if (repeat == repeatNums - 1) {
            copySize = copyElemNum - MAX_NUM * (repeatNums - 1);
        }
        DataCopy(ubSrc1, gmSrc[offsetcopysize], copySize);
        for (uint64_t j = 1; j < singleCoreNum; ++j) {
            tmpOffset += (singleSize << 1);
            DataCopy(ubSrc2, gmSrc[tmpOffset + offsetcopysize], copySize);
            TPipeSetWaitFlag<HardEvent::MTE2_V>();
            Add(ubSrc1, ubSrc1, ubSrc2, copySize);
            TPipeSetWaitFlag<HardEvent::V_MTE2>();
        }
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == sizeof(half)) {
            Cast(ubDst, ubSrc1, RoundMode::CAST_RINT, copySize);
        }
        TPipeSetWaitFlag<HardEvent::V_MTE3>();
        if constexpr (sizeof(T) == sizeof(half)) {
            // copy out
            CopyUbufToGmAlign<T>(gmDst[repeat * MAX_NUM / 16 * oriN], ubDst, copySize / 16, // burst
                                 currSplitN * sizeof(T),                                    // burstLen   16
                                 0,                                                         // srcGap, block
                                 (oriN - currSplitN) * sizeof(T)                            // dstGap, element, bytes
            );
        } else if constexpr (sizeof(T) == sizeof(float)) {
            uint32_t srcGap = 0;
            if (currSplitN <= 8) {
                srcGap = 1;
            }
            // copy out
            CopyUbufToGmAlign<T>(gmDst[repeat * MAX_NUM / 16 * oriN], ubSrc1, copySize / 16, // burst
                                 currSplitN * sizeof(T),                                     // burstLen
                                 srcGap,                                                     // srcGap, block
                                 (oriN - currSplitN) * sizeof(T)                             // dstGap, element, bytes
            );
        }
        offsetcopysize += copySize;
        TPipeSetWaitFlag<HardEvent::MTE3_MTE2>();
    }
}


template <class C_TYPE>
__aicore__ inline void ReduceKInUb(GM_ADDR cGM, GM_ADDR mmGM, uint64_t coreSize, uint64_t singleSize,
                                   uint64_t totalSize, uint64_t outSize, uint64_t cnt, uint64_t singleCoreN, uint64_t n,
                                   TBuf<TPosition::VECCALC> &tmpBuf, bool orderFlag, const TCubeTiling &tiling)
{
    using T = typename C_TYPE::T;

    uint64_t nCoreTail = n - (cnt - 1) * singleCoreN;
    uint64_t dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float);
    uint64_t dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing1 = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<float> ubSrcPing2 = ubSrcPing1[dataSize];

    LocalTensor<T> ubDstPong = ubDstPing[dataSize1 * NUM_TWO];
    LocalTensor<float> ubSrcPong1 = ubSrcPing1[dataSize * NUM_TWO];
    LocalTensor<float> ubSrcPong2 = ubSrcPong1[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc1 = ubSrcPing1;
    LocalTensor<float> ubSrc2 = ubSrcPing2;

    GlobalTensor<T> gmDst;
    GlobalTensor<float> gmSrcPing;
    GlobalTensor<float> gmSrcPong;
    GlobalTensor<float> gmSrc = gmSrcPing;

    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cGM), outSize);
    gmSrcPing.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mmGM), totalSize);
    gmSrcPong = gmSrcPing[singleSize];

    uint64_t vIndex = GetBlockIdx();
    uint64_t singleCoreNum = totalSize / singleSize;
    uint8_t pingpongEventId = 0;
    uint8_t pingpongEventIdWS = 0;
    for (uint64_t index = 0; index < cnt; ++index) {
        uint64_t coreOutSize = tiling.M * tiling.singleCoreN;

        if (orderFlag && index == cnt - 1) {
            // 重新计算尾列参数
            uint64_t nCoreTail = AlignTo256B(tiling.N - (cnt - 1) * tiling.singleCoreN);
            coreOutSize = tiling.M * nCoreTail;
            coreSize =
                MMV3DivCeil(tiling.M, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) * nCoreTail;
            dataSize = dataSize / nCoreTail * nCoreTail;
            dataSize1 = dataSize1 / nCoreTail * nCoreTail;
        }
        if ((index & CONTROL_DB) == 0) {
            pingpongEventIdWS = 0;
            gmSrc = gmSrcPing;
        } else {
            pingpongEventIdWS = 1;
            gmSrc = gmSrcPong;
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        WaitEvent<PIPE_S>(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        WaitEvent(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#endif
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        bool preProcess =
            orderFlag ? (vIndex * coreSize >= coreOutSize) : (vIndex * coreSize + index * singleSize >= outSize);
        if (preProcess) {
            if (index < cnt - 1) {
                NotifyEvent<PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
                PipeBarrier<PIPE_ALL>();
            }
            continue;
        }

        uint64_t actualSize = 0;
        if (orderFlag) {
            actualSize = min(coreSize, coreOutSize - vIndex * coreSize);
        } else {
            actualSize = min(coreSize, outSize - (vIndex * coreSize + index * singleSize));
        }

        uint64_t repeat = actualSize / dataSize;
        uint64_t tail = actualSize % dataSize;

        // initialize flag, in order to match the relationship in the vector core loop
        auto eventMTE3toMTE2Zero = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        auto eventMTE3toMTE2One = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);

        for (uint64_t i = 0; i < repeat; ++i) {
            if ((i & CONTROL_DB) == 0) {
                pingpongEventId = eventMTE3toMTE2Zero;
                ubDst = ubDstPing;
                ubSrc1 = ubSrcPing1;
                ubSrc2 = ubSrcPing2;
            } else {
                pingpongEventId = eventMTE3toMTE2One;
                ubDst = ubDstPong;
                ubSrc1 = ubSrcPong1;
                ubSrc2 = ubSrcPong2;
            }
            WaitFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
            SplitKVectorProcess<T>(ubSrc1, ubSrc2, gmSrc, ubDst, gmDst, vIndex, index, i, dataSize, dataSize, coreSize,
                                   singleSize, singleCoreNum, singleCoreN, n, cnt, orderFlag);
            SetFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
        if ((repeat & CONTROL_DB) == 0) {
            ubDst = ubDstPing;
            ubSrc1 = ubSrcPing1;
            ubSrc2 = ubSrcPing2;
        } else {
            ubDst = ubDstPong;
            ubSrc1 = ubSrcPong1;
            ubSrc2 = ubSrcPong2;
        }
        if (tail > 0) {
            SplitKVectorProcess<T>(ubSrc1, ubSrc2, gmSrc, ubDst, gmDst, vIndex, index, repeat, tail, dataSize, coreSize,
                                   singleSize, singleCoreNum, singleCoreN, n, cnt, orderFlag);
        }
        if (index < cnt - 1) {
            NotifyEvent<PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
            PipeBarrier<PIPE_ALL>();
        }
    }
    return;
}

template <class C_TYPE>
__aicore__ inline void ReduceKNzInUb(GM_ADDR cGM, GM_ADDR mmGM, uint64_t coreSize, uint64_t singleSize,
                                     uint64_t totalSize, uint64_t outSize, uint64_t cnt, uint64_t singleCoreN,
                                     uint64_t n, TBuf<TPosition::VECCALC> &tmpBuf, bool orderFlag,
                                     const TCubeTiling &tiling, uint64_t mCnt, uint64_t nCnt, uint64_t originM)
{
    using T = typename C_TYPE::T;
    uint64_t totalVecCoreNum = tiling.usedCoreNum * NUM_AIV_TO_AIC_RATIO; // 40


    uint64_t mCoreTail = tiling.M - (mCnt - 1) * tiling.singleCoreM;
    uint64_t nCoreTail = n - (cnt - 1) * singleCoreN;
    uint64_t dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float);
    uint64_t dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing1 = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<float> ubSrcPing2 = ubSrcPing1[dataSize];

    LocalTensor<T> ubDstPong = ubDstPing[dataSize1 * NUM_TWO];
    LocalTensor<float> ubSrcPong1 = ubSrcPing1[dataSize * NUM_TWO];
    LocalTensor<float> ubSrcPong2 = ubSrcPong1[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc1 = ubSrcPing1;
    LocalTensor<float> ubSrc2 = ubSrcPing2;

    GlobalTensor<T> gmDst;
    GlobalTensor<float> gmSrcPing;
    GlobalTensor<float> gmSrcPong;
    GlobalTensor<float> gmSrc = gmSrcPing;

    uint64_t mCoreUse = tiling.singleCoreM; // 384
    uint64_t nCoreUse = tiling.singleCoreN;
    uint64_t oriM = tiling.M;
    uint64_t oriN = tiling.N;

    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cGM), outSize);
    gmSrcPing.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mmGM), totalSize);
    gmSrcPong = gmSrcPing[singleSize];

    uint64_t vIndex = GetBlockIdx();
    uint64_t singleCoreNum = totalSize / singleSize;
    uint8_t pingpongEventId = 0;
    uint8_t pingpongEventIdWS = 0;
    for (uint64_t index = 0; index < cnt; ++index) {
        if (orderFlag && index == (nCnt - 1)) {
            nCoreUse = nCoreTail;
            nCoreTail = AlignTo256B(nCoreTail);
        } else if (!orderFlag && index == (cnt - 1)) {
            mCoreUse = mCoreTail;
        }

        if ((index & CONTROL_DB) == 0) {
            pingpongEventIdWS = 0;
            gmSrc = gmSrcPing;
        } else {
            pingpongEventIdWS = 1;
            gmSrc = gmSrcPong;
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        WaitEvent<PIPE_S>(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        WaitEvent(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#endif
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        uint64_t actualM = mCoreUse; // 63
        uint64_t actualN = nCoreUse; // 64

        uint64_t alignedM = (actualM + 15) / 16 * 16; // 112
        uint64_t alignedN = (actualN + 15) / 16 * 16; // 160
        uint64_t rowBlockNum = alignedN / 16;         // nz base block, 16 x 16, 4  7
        uint64_t colBlockNum = alignedN / 16;         // nz base block, 16 x 16, 4  10
        uint64_t currSplitN = 16;
        uint64_t currOutCOffset = index * tiling.N * tiling.singleCoreM; // 0
        // it seems that singleCoreM is less than 384
        uint64_t pIndex = 0;
        auto eventMTE3toMTE2Zero = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        auto eventMTE3toMTE2One = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);

        for (uint64_t processIdx = vIndex; processIdx < colBlockNum; processIdx += totalVecCoreNum) {
            if (pIndex == 0) {
                pingpongEventId = eventMTE3toMTE2Zero;
                ubDst = ubDstPing;
                ubSrc1 = ubSrcPing1;
                ubSrc2 = ubSrcPing2;
            } else {
                pingpongEventId = eventMTE3toMTE2One;
                ubDst = ubDstPong;
                ubSrc1 = ubSrcPong1;
                ubSrc2 = ubSrcPong2;
            }
            pIndex = 1 - pIndex;
            // deal with alignedM x 16 nz matrix, double buffer inside the function
            if (processIdx == colBlockNum - 1)
                currSplitN = actualN - processIdx * 16;
            uint64_t nOffset = processIdx * 16;
            uint64_t currSrcOffset = processIdx * 16 * originM; // 1 * 16 * 2331
            uint64_t copySize = actualM * 16;
            WaitFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
            SplitKVectorNZProcess(gmSrc[currSrcOffset], gmDst[currOutCOffset + nOffset], copySize, currSplitN,
                                  singleCoreNum, singleSize, oriN, ubSrc1, ubSrc2, ubDst);
            SetFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);

        if (index < cnt - 1) {
            SyncAll();
            NotifyEvent<PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
            PipeBarrier<PIPE_ALL>();
        }
    }
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatMulMultiCoreSplitKDivide(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR mmOffsetGM,
                                                      uint64_t singleSize, bool isHf32, TPipe *que,
                                                      const TCubeTiling &tiling, bool isBias)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;

    SetAtomicNone();

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobalPing;
    GlobalTensor<C_T> cGlobalPong;
    GlobalTensor<C_T> cGlobal = cGlobalPing;
    GlobalTensor<BIAS_T> biasGlobal;

    uint64_t c0Size = 8;
    GetSizeC0<A_T>(c0Size);
    uint64_t alignedOriM = MMV3CeilAlign(tiling.M, ALIGNED_H);
    uint64_t alignedOriN = MMV3CeilAlign(tiling.N, c0Size);
    uint64_t alignedKaSize = MMV3CeilAlign(tiling.Ka, c0Size);
    uint64_t alignedKbSize = MMV3CeilAlign(tiling.Kb, ALIGNED_H);

    // A B矩阵都是对齐矩阵
    if constexpr (A_TYPE::isTrans) {
        alignedOriM = MMV3CeilAlign(tiling.M, c0Size);
        alignedKaSize = MMV3CeilAlign(tiling.Ka, ALIGNED_H);
    }
    if constexpr (B_TYPE::isTrans) {
        alignedOriN = MMV3CeilAlign(tiling.N, ALIGNED_H);
        alignedKbSize = MMV3CeilAlign(tiling.Kb, c0Size);
    }

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling.Kb * tiling.N);
    cGlobalPing.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(mmOffsetGM), singleSize * NUM_TWO);
    cGlobalPong = cGlobalPing[singleSize];
    if (isBias) {
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(biasGM), tiling.N);
    }
#ifdef __CCE_KT_TEST__
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_PRELOAD_MK> mmmk_33;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_PRELOAD_MK> mmmk;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_PRELOAD_NK> mmnk;
#else
    constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_MDL;
    constexpr static MatmulShapeParams shapeParams = {0, 0, 0, 128, 128, 256 / sizeof(A_T)};
    constexpr static MatmulQuantParams quantParams = {false, false};
    constexpr static MatmulBatchParams batchParams = {false, BatchMode::NONE};
    constexpr static MatmulFuncParams funcParamsMK{
        false, false, false, false, 2, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true}; // 2 preload左矩阵
    constexpr static MatmulFuncParams funcParamsNK{
        false, false, false, false, 1, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true}; // 1 preload右矩阵
    constexpr MatmulConfig mmStaticConfigMK =
        GetMMConfig<configMode>(shapeParams, quantParams, batchParams, funcParamsMK);
    constexpr MatmulConfig mmStaticConfigNK =
        GetMMConfig<configMode>(shapeParams, quantParams, batchParams, funcParamsNK);
    constexpr static MatmulApiStaticTiling staticTilingMK =
        GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmStaticConfigMK);
    constexpr static MatmulApiStaticTiling staticTilingNK =
        GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmStaticConfigNK);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, staticTilingMK, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
               AscendC::Impl::Detail::NBuffer33MatmulPolicy>
        mmmk_33;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, staticTilingMK> mmmk;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, staticTilingNK> mmnk;
#endif
    bool orderFlag = !tiling.iterateOrder;
    bool is33MK = (tiling.M <= NUM_256);
    if (orderFlag) {
        mmnk.SetSubBlockIdx(0);
        PRELOAD(4); // preload commands
        mmnk.Init(&tiling, que);
        if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
            mmnk.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, tiling.singleCoreN);
        } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
            mmnk.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, tiling.singleCoreN);
        } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
            mmnk.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, tiling.singleCoreN);
        } else {
            mmnk.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.singleCoreN);
        }
        isHf32 ? mmnk.SetHF32(true, 1) : mmnk.SetHF32(false, 0);
    } else {
        if (is33MK) {
            mmmk_33.SetSubBlockIdx(0);
            PRELOAD(4); // preload commands
            mmmk_33.Init(&tiling, que);
            if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
                mmmk_33.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, tiling.N);
            } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
                mmmk_33.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, tiling.N);
            } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
                mmmk_33.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, tiling.N);
            } else {
                mmmk_33.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.N);
            }
            isHf32 ? mmmk_33.SetHF32(true, 1) : mmmk_33.SetHF32(false, 0);
        } else {
            mmmk.SetSubBlockIdx(0);
            PRELOAD(4); // preload commands
            mmmk.Init(&tiling, que);
            if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
                mmmk.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, tiling.N);
            } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
                mmmk.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, tiling.N);
            } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
                mmmk.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, tiling.N);
            } else {
                mmmk.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.N);
            }
            isHf32 ? mmmk.SetHF32(true, 1) : mmmk.SetHF32(false, 0);
        }
    }

    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t nCnt = MMV3DivCeil(tiling.N, tiling.singleCoreN);
    uint64_t mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
    uint64_t kCnt = orderFlag ? MMV3DivCeil(tiling.Kb, tiling.singleCoreK) : MMV3DivCeil(tiling.Ka, tiling.singleCoreK);
    uint64_t nCoreTail = tiling.N - (nCnt - 1) * tiling.singleCoreN;
    uint64_t mCoreTail = tiling.M - (mCnt - 1) * tiling.singleCoreM;
    uint64_t kCoreTail =
        orderFlag ? tiling.Kb - (kCnt - 1) * tiling.singleCoreK : tiling.Ka - (kCnt - 1) * tiling.singleCoreK;
    uint64_t preCoreNum = kCnt % tiling.usedCoreNum;

    if (preCoreNum == 0) {
        preCoreNum = tiling.usedCoreNum;
    }
    uint64_t round = MMV3DivCeil(kCnt, tiling.usedCoreNum);
    uint64_t index = GetCurrentBlockIdx() * round;
    uint64_t realRound = round;
    if (GetCurrentBlockIdx() >= preCoreNum) {
        index = GetCurrentBlockIdx() * (round - 1) + preCoreNum;
        realRound = round - 1;
    }
    uint64_t mCoreUse = tiling.singleCoreM;
    uint64_t nCoreUse = tiling.singleCoreN;
    uint64_t kCoreUse = tiling.singleCoreK;
    uint64_t mTileOffset = 0;
    uint64_t nTileOffset = 0;
    uint64_t mOffset = 0;
    uint64_t nOffset = 0;
    uint64_t kOffset = 0;
    uint8_t pingpongEventId = 0;

    uint64_t OutCnt = orderFlag ? nCnt : mCnt;
    for (uint64_t outIndex = 0; outIndex < OutCnt; ++outIndex) {
        if ((outIndex & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            cGlobal = cGlobalPing;
        } else {
            pingpongEventId = 1;
            cGlobal = cGlobalPong;
        }
        if (outIndex > 1) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            WaitEvent<PIPE_S>(AIV_SYNC_AIC_FLAG + pingpongEventId);
            WaitEvent<PIPE_S>(AIV_SYNC_AIC_FLAG + pingpongEventId + FLAG_ID_MAX);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            WaitEvent(AIV_SYNC_AIC_FLAG + pingpongEventId);
#endif
            SyncAll();
            PipeBarrier<PIPE_ALL>();
        }
        if (orderFlag) {
            nOffset = outIndex * tiling.singleCoreN;
        } else {
            mOffset = outIndex * tiling.singleCoreM;
        }
        if (orderFlag && outIndex == (nCnt - 1)) {
            nCoreUse = nCoreTail;
            nCoreTail = AlignTo256B(nCoreTail);
            if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
                mmnk.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, nCoreTail);
            } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
                mmnk.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, nCoreTail);
            } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
                mmnk.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, nCoreTail);
            } else {
                mmnk.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, nCoreTail);
            }
        } else if (!orderFlag && outIndex == (mCnt - 1)) {
            mCoreUse = mCoreTail;
        }
        for (uint64_t kIndex = index; kIndex < (index + realRound); ++kIndex) {
            kOffset = kIndex * tiling.singleCoreK;
            kCoreUse = tiling.singleCoreK;
            if (kIndex == (kCnt - 1)) {
                kCoreUse = kCoreTail;
            }
            SetOffset<A_TYPE, B_TYPE>(offsetA, offsetB, mOffset, nOffset, kOffset, c0Size, alignedOriM, alignedOriN,
                                      alignedKaSize, alignedKbSize, tiling);
            if (orderFlag) {
                mmnk.SetSingleShape(mCoreUse, nCoreUse, kCoreUse);
                mmnk.SetTensorA(aGlobal[offsetA], A_TYPE::isTrans);
                mmnk.SetTensorB(bGlobal[offsetB], B_TYPE::isTrans);
                isBias &&kIndex == 0 ?
                    mmnk.SetBias(biasGlobal[outIndex * tiling.singleCoreN]) :
                    mmnk.ClearBias(); // set bias at the first k loop and clear bias tag in the following loop
                mmnk.IterateAll(cGlobal[offsetC], kIndex != index);
            } else {
                if (is33MK) {
                    mmmk_33.SetSingleShape(mCoreUse, nCoreUse, kCoreUse);
                    mmmk_33.SetTensorA(aGlobal[offsetA], A_TYPE::isTrans);
                    mmmk_33.SetTensorB(bGlobal[offsetB], B_TYPE::isTrans);
                    isBias &&kIndex == 0 ?
                        mmmk_33.SetBias(biasGlobal[0]) :
                        mmmk_33.ClearBias(); // set bias at the first k loop and clear bias tag in the following loop
                    mmmk_33.IterateAll(cGlobal[offsetC], kIndex != index);
                } else {
                    mmmk.SetSingleShape(mCoreUse, nCoreUse, kCoreUse);
                    mmmk.SetTensorA(aGlobal[offsetA], A_TYPE::isTrans);
                    mmmk.SetTensorB(bGlobal[offsetB], B_TYPE::isTrans);
                    isBias &&kIndex == 0 ?
                        mmmk.SetBias(biasGlobal[0]) :
                        mmmk.ClearBias(); // set bias at the first k loop and clear bias tag in the following loop
                    mmmk.IterateAll(cGlobal[offsetC], kIndex != index);
                }
            }
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId + FLAG_ID_MAX);
        NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId);
#endif
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    if (orderFlag) {
        mmnk.SetHF32(false, 0);
    } else {
        is33MK ? mmmk_33.SetHF32(false, 0) : mmmk.SetHF32(false, 0);
    }
}

__aicore__ inline void AddTail(LocalTensor<float> &ubSrc1, LocalTensor<float> &ubSrc2, uint64_t burst,
                               uint64_t burstLen, uint64_t realLen)
{
    for (uint64_t i = 0; i < burst; ++i) {
        Add(ubSrc1[i * burstLen], ubSrc1[i * burstLen], ubSrc2[i * burstLen], realLen);
    }
}

template <class T>
__aicore__ inline void
SplitKVectorProcessL2cache(LocalTensor<float> ubSrc1, LocalTensor<float> ubSrc2, GlobalTensor<float> gmSrc,
                           LocalTensor<T> ubDst, GlobalTensor<T> gmDst, uint64_t vIndex, uint64_t index,
                           uint64_t currentLoop, uint64_t dataSizeToMove, uint64_t dataSize, uint64_t coreSize,
                           uint64_t singleSize, uint64_t singleCoreNum, uint64_t singleCoreN, uint64_t singleCoreM,
                           uint64_t n, uint64_t mCnt, bool orderNMFlag, uint64_t nIndex, uint64_t nCnt)
{
    uint64_t dstOffset = 0;
    uint64_t burst = 1;
    uint64_t burstLen = dataSizeToMove;
    uint64_t srcGap = 0;
    uint64_t dstGap = 0;
    uint64_t nCoreTail = n - (nCnt - 1) * singleCoreN;
    uint64_t gmOffset = currentLoop * dataSize + vIndex * coreSize;
    uint64_t tmpOffset = 0;

    CopyGmToUbufAlign<float>(ubSrc1, gmSrc[gmOffset], burst, burstLen * sizeof(float), 0, 0, srcGap * sizeof(float),
                             dstGap * sizeof(float) / 32);

    for (uint64_t j = 1; j < singleCoreNum; ++j) {
        tmpOffset += (singleSize << 1);
        CopyGmToUbufAlign<float>(ubSrc2, gmSrc[gmOffset + tmpOffset], burst, burstLen * sizeof(float), 0, 0,
                                 srcGap * sizeof(float), dstGap * sizeof(float) / 32);
        // MTE2 to V, enable pingpong
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        if (nIndex == nCnt - 1 && AlignTo256B(nCoreTail) != nCoreTail) {
            AddTail(ubSrc1, ubSrc2, (dataSizeToMove / AlignTo256B(nCoreTail)), AlignTo256B(nCoreTail), nCoreTail);
        } else {
            Add(ubSrc1, ubSrc1, ubSrc2, burst * burstLen);
        }
        TPipeSetWaitFlag<HardEvent::V_MTE2>();
    }
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(ubDst, ubSrc1, RoundMode::CAST_RINT, burst * burstLen);
    }
    TPipeSetWaitFlag<HardEvent::V_MTE3>();


    dstOffset = ComputOffsetL2cache(gmOffset, singleCoreN, n, nIndex, singleCoreM, index);
    burst = (dataSizeToMove / singleCoreN);
    burstLen = singleCoreN;
    dstGap = (n - singleCoreN);

    // 尾列正常处理
    if (nIndex == nCnt - 1) {
        burst = (dataSizeToMove / AlignTo256B(nCoreTail));
        burstLen = nCoreTail;
        srcGap = AlignTo256B(nCoreTail) - nCoreTail;
        dstGap = n - nCoreTail;
        uint64_t mIdx = (gmOffset / AlignTo256B(nCoreTail) + index * singleCoreM);
        uint64_t nIdx = (gmOffset % AlignTo256B(nCoreTail) + nIndex * singleCoreN);
        dstOffset = mIdx * n + nIdx;
    }

    if constexpr (sizeof(T) == sizeof(half)) {
        CopyUbufToGmAlign<T>(gmDst[dstOffset], ubDst, burst, burstLen * sizeof(T), srcGap * sizeof(T) / 32,
                             dstGap * sizeof(T)); // 32 is blocksize

    } else if constexpr (sizeof(T) == sizeof(float)) {
        CopyUbufToGmAlign<T>(gmDst[dstOffset], ubSrc1, burst, burstLen * sizeof(T), srcGap * sizeof(T) / 32,
                             dstGap * sizeof(T)); // 32 is blocksize
    }
}

template <class C_TYPE>
__aicore__ inline void ReduceKInUbNzL2cache(GM_ADDR cGM, GM_ADDR mmGM, uint64_t coreSize, uint64_t singleSize,
                                            uint64_t totalSize, uint64_t outSize, uint64_t mCnt, uint64_t nCnt,
                                            uint64_t singleCoreN, uint64_t n, TBuf<TPosition::VECCALC> &tmpBuf,
                                            bool orderNMFlag, const TCubeTiling &tiling, uint64_t originM)
{
    using T = typename C_TYPE::T;

    uint64_t nCoreTail = n - (nCnt - 1) * singleCoreN;
    uint64_t mCoreTail = tiling.M - (mCnt - 1) * tiling.singleCoreM;

    uint64_t dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float);
    uint64_t dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);

    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing1 = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<float> ubSrcPing2 = ubSrcPing1[dataSize];

    LocalTensor<T> ubDstPong = ubDstPing[dataSize1 * NUM_TWO];
    LocalTensor<float> ubSrcPong1 = ubSrcPing1[dataSize * NUM_TWO];
    LocalTensor<float> ubSrcPong2 = ubSrcPong1[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc1 = ubSrcPing1;
    LocalTensor<float> ubSrc2 = ubSrcPing2;

    GlobalTensor<T> gmDst;
    GlobalTensor<float> gmSrcPing;
    GlobalTensor<float> gmSrcPong;
    GlobalTensor<float> gmSrc = gmSrcPing;

    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cGM), outSize);
    gmSrcPing.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mmGM), totalSize);
    gmSrcPong = gmSrcPing[singleSize];

    uint64_t vIndex = GetBlockIdx();
    uint64_t singleCoreNum = totalSize / singleSize;
    uint8_t pingpongEventId = 0;
    uint8_t pingpongEventIdWS = 1;

    uint64_t oriM = tiling.M;
    uint64_t oriN = tiling.N;
    uint64_t totalVecCoreNum = tiling.usedCoreNum * NUM_AIV_TO_AIC_RATIO; // 40

    uint64_t mCoreUse = tiling.singleCoreM;
    uint64_t nCoreUse = tiling.singleCoreN;

    uint64_t outCnt = orderNMFlag ? nCnt : mCnt;
    uint64_t inCnt = orderNMFlag ? mCnt : nCnt;

    uint64_t outCntSize = orderNMFlag ? tiling.singleCoreN : tiling.singleCoreM;
    uint64_t inCntSize = orderNMFlag ? tiling.singleCoreM : tiling.singleCoreN;

    uint64_t coreOutSize = tiling.singleCoreM * tiling.singleCoreN;
    for (uint64_t outIndex = 0; outIndex < outCnt; ++outIndex) {
        // 被尾列改过的参数都要恢复
        coreSize = MMV3DivCeil(tiling.singleCoreM, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) *
                   tiling.singleCoreN;
        dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float); // 恢复原始大小
        dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);
        dataSize = dataSize / singleCoreN * singleCoreN;
        dataSize1 = dataSize1 / singleCoreN * singleCoreN;
        coreOutSize = tiling.singleCoreM * tiling.singleCoreN;

        for (uint64_t inIndex = 0; inIndex < inCnt; ++inIndex) {
            uint64_t mIndex = orderNMFlag ? inIndex : outIndex;
            uint64_t nIndex = orderNMFlag ? outIndex : inIndex;

            if (mIndex == mCnt - 1 || nIndex == nCnt - 1) {
                // 重新计算尾列参数
                mCoreUse = (mIndex == mCnt - 1) ? mCoreTail : tiling.singleCoreM;
                nCoreUse = (nIndex == nCnt - 1) ? AlignTo256B(nCoreTail) : tiling.singleCoreN;
                coreOutSize = mCoreUse * nCoreUse;
                coreSize =
                    MMV3DivCeil(mCoreUse, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) * nCoreUse;
                dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float); // 恢复原始大小
                dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);
                dataSize = dataSize / nCoreUse * nCoreUse;
                dataSize1 = dataSize1 / nCoreUse * nCoreUse;
            }

            pingpongEventIdWS = (pingpongEventIdWS + 1) & 1;
            gmSrc = pingpongEventIdWS ? gmSrcPong : gmSrcPing;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            WaitEvent<PIPE_S>(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            CrossCoreWaitFlag(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#endif
            SyncAll();
            PipeBarrier<PIPE_ALL>();

            uint64_t actualM = mCoreUse; // 63
            uint64_t actualN = nCoreUse; // 64
            if (orderNMFlag && outIndex == outCnt - 1) {
                actualN = nCoreTail;
            }

            uint64_t alignedM = (actualM + 15) / 16 * 16; // 64
            uint64_t alignedN = (actualN + 15) / 16 * 16; // 64
            uint64_t rowBlockNum = alignedM / 16;         // nz base block, 16 x 16, 4
            uint64_t colBlockNum = alignedN / 16;         // nz base block, 16 x 16, 4
            uint64_t currSplitN = 16;
            uint64_t currOutCOffset = 0;
            if (orderNMFlag) {
                currOutCOffset = inIndex * tiling.N * tiling.singleCoreM + outIndex * tiling.singleCoreN;
            } else {
                currOutCOffset = outIndex * tiling.N * tiling.singleCoreM + inIndex * tiling.singleCoreN;
            }
            uint64_t pIndex = 0;
            auto eventMTE3toMTE2Zero = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
            auto eventMTE3toMTE2One = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);


            for (uint64_t processIdx = vIndex; processIdx < colBlockNum; processIdx += totalVecCoreNum) {
                if (pIndex == 0) {
                    pingpongEventId = eventMTE3toMTE2Zero;
                    ubDst = ubDstPing;
                    ubSrc1 = ubSrcPing1;
                    ubSrc2 = ubSrcPing2;
                } else {
                    pingpongEventId = eventMTE3toMTE2One;
                    ubDst = ubDstPong;
                    ubSrc1 = ubSrcPong1;
                    ubSrc2 = ubSrcPong2;
                }
                pIndex = 1 - pIndex;
                // deal with alignedM x 16 nz matrix, double buffer inside the function
                if (processIdx == colBlockNum - 1)
                    currSplitN = actualN - processIdx * 16;
                uint64_t nOffset = processIdx * 16;
                uint64_t currSrcOffset = processIdx * 16 * originM;
                uint64_t copySize = actualM * 16;

                WaitFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
                SplitKVectorNZProcess(gmSrc[currSrcOffset], gmDst[currOutCOffset + nOffset], copySize, currSplitN,
                                      singleCoreNum, singleSize, oriN, ubSrc1, ubSrc2, ubDst);
                SetFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
            if (mIndex < mCnt - 1 || nIndex < nCnt - 1) {
                SyncAll();
                CrossCoreSetFlag<0x2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    return;
}

template <class C_TYPE>
__aicore__ inline void ReduceKInUbL2cache(GM_ADDR cGM, GM_ADDR mmGM, uint64_t coreSize, uint64_t singleSize,
                                          uint64_t totalSize, uint64_t outSize, uint64_t mCnt, uint64_t nCnt,
                                          uint64_t singleCoreN, uint64_t n, TBuf<TPosition::VECCALC> &tmpBuf,
                                          bool orderNMFlag, const TCubeTiling &tiling)
{
    using T = typename C_TYPE::T;

    uint64_t nCoreTail = n - (nCnt - 1) * singleCoreN;
    uint64_t mCoreTail = tiling.M - (mCnt - 1) * tiling.singleCoreM;

    uint64_t dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float);
    uint64_t dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);

    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing1 = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<float> ubSrcPing2 = ubSrcPing1[dataSize];

    LocalTensor<T> ubDstPong = ubDstPing[dataSize1 * NUM_TWO];
    LocalTensor<float> ubSrcPong1 = ubSrcPing1[dataSize * NUM_TWO];
    LocalTensor<float> ubSrcPong2 = ubSrcPong1[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc1 = ubSrcPing1;
    LocalTensor<float> ubSrc2 = ubSrcPing2;

    GlobalTensor<T> gmDst;
    GlobalTensor<float> gmSrcPing;
    GlobalTensor<float> gmSrcPong;
    GlobalTensor<float> gmSrc = gmSrcPing;

    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cGM), outSize);
    gmSrcPing.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mmGM), totalSize);
    gmSrcPong = gmSrcPing[singleSize];

    uint64_t vIndex = GetBlockIdx();
    uint64_t singleCoreNum = totalSize / singleSize;
    uint8_t pingpongEventId = 0;
    uint8_t pingpongEventIdWS = 1;

    uint64_t outCnt = orderNMFlag ? nCnt : mCnt;
    uint64_t inCnt = orderNMFlag ? mCnt : nCnt;
    uint64_t coreOutSize = tiling.singleCoreM * tiling.singleCoreN;
    for (uint64_t outIndex = 0; outIndex < outCnt; ++outIndex) {
        // 被尾列改过的参数都要恢复
        coreSize = MMV3DivCeil(tiling.singleCoreM, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) *
                   tiling.singleCoreN;
        dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float); // 恢复原始大小
        dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);
        dataSize = dataSize / singleCoreN * singleCoreN;
        dataSize1 = dataSize1 / singleCoreN * singleCoreN;
        coreOutSize = tiling.singleCoreM * tiling.singleCoreN;

        for (uint64_t inIndex = 0; inIndex < inCnt; ++inIndex) {
            uint64_t mIndex = orderNMFlag ? inIndex : outIndex;
            uint64_t nIndex = orderNMFlag ? outIndex : inIndex;

            if (mIndex == mCnt - 1 || nIndex == nCnt - 1) {
                // 重新计算尾列参数
                uint64_t mCoreUse = (mIndex == mCnt - 1) ? mCoreTail : tiling.singleCoreM;
                uint64_t nCoreUse = (nIndex == nCnt - 1) ? AlignTo256B(nCoreTail) : tiling.singleCoreN;
                coreOutSize = mCoreUse * nCoreUse;
                coreSize =
                    MMV3DivCeil(mCoreUse, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) * nCoreUse;
                dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float); // 恢复原始大小
                dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);
                dataSize = dataSize / nCoreUse * nCoreUse;
                dataSize1 = dataSize1 / nCoreUse * nCoreUse;
            }

            pingpongEventIdWS = (pingpongEventIdWS + 1) & 1;
            gmSrc = pingpongEventIdWS ? gmSrcPong : gmSrcPing;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            WaitEvent<PIPE_S>(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            CrossCoreWaitFlag(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#endif
            SyncAll();
            PipeBarrier<PIPE_ALL>();

            bool preProcess = vIndex * coreSize >= coreOutSize;
            if (preProcess) {
                if (mIndex < mCnt - 1 || nIndex < nCnt - 1) { // 之后还有singleSize的处理过程，就需要发送同步，让AIC进行
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
                    PipeBarrier<PIPE_ALL>();
                }
                continue;
            }

            uint64_t actualSize = min(coreSize, coreOutSize - vIndex * coreSize);
            uint64_t repeat = actualSize / dataSize;
            uint64_t tail = actualSize % dataSize;

            // initialize flag, in order to match the relationship in the vector core loop
            auto eventMTE3toMTE2Zero = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
            auto eventMTE3toMTE2One = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);

            for (uint64_t i = 0; i < repeat; ++i) {
                if ((i & CONTROL_DB) == 0) {
                    pingpongEventId = eventMTE3toMTE2Zero;
                    ubDst = ubDstPing;
                    ubSrc1 = ubSrcPing1;
                    ubSrc2 = ubSrcPing2;
                } else {
                    pingpongEventId = eventMTE3toMTE2One;
                    ubDst = ubDstPong;
                    ubSrc1 = ubSrcPong1;
                    ubSrc2 = ubSrcPong2;
                }

                WaitFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
                SplitKVectorProcessL2cache<T>(ubSrc1, ubSrc2, gmSrc, ubDst, gmDst, vIndex, mIndex, i, dataSize,
                                              dataSize, coreSize, singleSize, singleCoreNum, singleCoreN,
                                              tiling.singleCoreM, n, mCnt, orderNMFlag, nIndex, nCnt);
                SetFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
            if ((repeat & CONTROL_DB) == 0) {
                ubDst = ubDstPing;
                ubSrc1 = ubSrcPing1;
                ubSrc2 = ubSrcPing2;
            } else {
                ubDst = ubDstPong;
                ubSrc1 = ubSrcPong1;
                ubSrc2 = ubSrcPong2;
            }
            if (tail > 0) {
                SplitKVectorProcessL2cache<T>(ubSrc1, ubSrc2, gmSrc, ubDst, gmDst, vIndex, mIndex, repeat, tail,
                                              dataSize, coreSize, singleSize, singleCoreNum, singleCoreN,
                                              tiling.singleCoreM, n, mCnt, orderNMFlag, nIndex, nCnt);
            }
            if (mIndex < mCnt - 1 || nIndex < nCnt - 1) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatMulMultiCoreSplitKDivideL2cache(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                             GM_ADDR mmOffsetGM, uint64_t singleSize, bool isHf32,
                                                             TPipe *que, const TCubeTiling &tiling, bool isBias)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;

    SetAtomicNone();

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobalPing;
    GlobalTensor<C_T> cGlobalPong;
    GlobalTensor<C_T> cGlobal = cGlobalPing;
    GlobalTensor<BIAS_T> biasGlobal;

    uint64_t c0Size = 8;
    GetSizeC0<A_T>(c0Size);
    uint64_t alignedOriM = MMV3CeilAlign(tiling.M, ALIGNED_H);
    uint64_t alignedOriN = MMV3CeilAlign(tiling.N, c0Size);
    uint64_t alignedKaSize = MMV3CeilAlign(tiling.Ka, c0Size);
    uint64_t alignedKbSize = MMV3CeilAlign(tiling.Kb, ALIGNED_H);

    // A B矩阵都是对齐矩阵
    if constexpr (A_TYPE::isTrans) {
        alignedOriM = MMV3CeilAlign(tiling.M, c0Size);
        alignedKaSize = MMV3CeilAlign(tiling.Ka, ALIGNED_H);
    }
    if constexpr (B_TYPE::isTrans) {
        alignedOriN = MMV3CeilAlign(tiling.N, ALIGNED_H);
        alignedKbSize = MMV3CeilAlign(tiling.Kb, c0Size);
    }

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling.Kb * tiling.N);
    cGlobalPing.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(mmOffsetGM), singleSize * NUM_TWO);
    cGlobalPong = cGlobalPing[singleSize];
    if (isBias) {
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(biasGM), tiling.N);
    }
#ifdef __CCE_KT_TEST__
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_PRELOAD_MK> mmNM;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_PRELOAD_NK> mmMN;
#else
    constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_MDL;
    constexpr static MatmulShapeParams shapeParams = {0, 0, 0, 128, 128, 256 / sizeof(A_T)};
    constexpr static MatmulQuantParams quantParams = {false, false};
    constexpr static MatmulBatchParams batchParams = {false, BatchMode::NONE};
    constexpr static MatmulFuncParams funcParamsMK{
        false, false, false, false, 2, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true}; // 2 preload左矩阵
    constexpr static MatmulFuncParams funcParamsNK{
        false, false, false, false, 1, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true}; // 1 preload右矩阵
    constexpr MatmulConfig mmStaticConfigMK =
        GetMMConfig<configMode>(shapeParams, quantParams, batchParams, funcParamsMK);
    constexpr MatmulConfig mmStaticConfigNK =
        GetMMConfig<configMode>(shapeParams, quantParams, batchParams, funcParamsNK);
    constexpr static MatmulApiStaticTiling staticTilingMK =
        GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmStaticConfigMK);
    constexpr static MatmulApiStaticTiling staticTilingNK =
        GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmStaticConfigNK);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, staticTilingMK> mmNM;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, staticTilingNK> mmMN;
#endif

    // mmMN对应tiling的33算法N取384，外循环顺序是for mCnt再 for nCnt
    // mmNM对应tiling的33算法M取384，外循环顺序是for nCnt再 for mCnt
    bool orderNMFlag = tiling.iterateOrder;

    if (orderNMFlag) {
        mmNM.SetSubBlockIdx(0);
        PRELOAD(4); // preload commands
        mmNM.Init(&tiling, que);
        if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
            mmNM.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, tiling.singleCoreN);
        } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
            mmNM.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, tiling.singleCoreN);
        } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
            mmNM.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, tiling.singleCoreN);
        } else {
            mmNM.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.singleCoreN);
        }
        isHf32 ? mmNM.SetHF32(true, 1) : mmNM.SetHF32(false, 0);
    } else {
        mmMN.SetSubBlockIdx(0);
        PRELOAD(4); // preload commands
        mmMN.Init(&tiling, que);
        if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
            mmMN.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, tiling.singleCoreN);
        } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
            mmMN.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, tiling.singleCoreN);
        } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
            mmMN.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, tiling.singleCoreN);
        } else {
            mmMN.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.singleCoreN);
        }
        isHf32 ? mmMN.SetHF32(true, 1) : mmMN.SetHF32(false, 0);
    }

    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t nCnt = MMV3DivCeil(tiling.N, tiling.singleCoreN);
    uint64_t mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
    uint64_t kCnt =
        orderNMFlag ? MMV3DivCeil(tiling.Kb, tiling.singleCoreK) : MMV3DivCeil(tiling.Ka, tiling.singleCoreK);
    uint64_t nCoreTail = tiling.N - (nCnt - 1) * tiling.singleCoreN;
    uint64_t mCoreTail = tiling.M - (mCnt - 1) * tiling.singleCoreM;
    uint64_t kCoreTail =
        orderNMFlag ? tiling.Kb - (kCnt - 1) * tiling.singleCoreK : tiling.Ka - (kCnt - 1) * tiling.singleCoreK;
    uint64_t preCoreNum = kCnt % tiling.usedCoreNum;

    if (preCoreNum == 0) {
        preCoreNum = tiling.usedCoreNum;
    }
    uint64_t round = MMV3DivCeil(kCnt, tiling.usedCoreNum);
    uint64_t index = GetCurrentBlockIdx() * round;
    uint64_t realRound = round;
    if (GetCurrentBlockIdx() >= preCoreNum) {
        index = GetCurrentBlockIdx() * (round - 1) + preCoreNum;
        realRound = round - 1;
    }
    uint64_t mCoreUse = tiling.singleCoreM;
    uint64_t nCoreUse = tiling.singleCoreN;
    uint64_t kCoreUse = tiling.singleCoreK;
    uint64_t mOffset = 0;
    uint64_t nOffset = 0;
    uint64_t kOffset = 0;
    uint8_t pingpongEventId = 1;

    uint64_t outCnt = orderNMFlag ? nCnt : mCnt;
    uint64_t inCnt = orderNMFlag ? mCnt : nCnt;
    uint64_t count = 0;
    for (uint64_t outIndex = 0; outIndex < outCnt; ++outIndex) {
        nOffset = outIndex * tiling.singleCoreN;
        mOffset = outIndex * tiling.singleCoreM;
        mCoreUse = tiling.singleCoreM; // 恢复
        nCoreUse = tiling.singleCoreN; // 恢复
        if (!orderNMFlag) {
            if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
                mmMN.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, tiling.singleCoreN);
            } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
                mmMN.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, tiling.singleCoreN);
            } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
                mmMN.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, tiling.singleCoreN);
            } else {
                mmMN.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.singleCoreN);
            }
        }

        if (!orderNMFlag && outIndex == (outCnt - 1)) {
            mCoreUse = mCoreTail;
        }
        if (orderNMFlag && outIndex == (outCnt - 1)) {
            nCoreUse = nCoreTail;
            if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
                mmNM.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, AlignTo256B(nCoreTail));
            } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
                mmNM.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, AlignTo256B(nCoreTail));
            } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
                mmNM.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, AlignTo256B(nCoreTail));
            } else {
                mmNM.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, AlignTo256B(nCoreTail));
            }
        }

        for (uint64_t inIndex = 0; inIndex < inCnt; ++inIndex) {
            pingpongEventId = (pingpongEventId + 1) & 1;
            cGlobal = pingpongEventId ? cGlobalPong : cGlobalPing;
            if (!orderNMFlag && inIndex == (inCnt - 1)) {
                nCoreUse = nCoreTail;
                if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
                    mmMN.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, AlignTo256B(nCoreTail));
                } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
                    mmMN.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, AlignTo256B(nCoreTail));
                } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
                    mmMN.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, AlignTo256B(nCoreTail));
                } else {
                    mmMN.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, AlignTo256B(nCoreTail));
                }
            }
            if (orderNMFlag && inIndex == (inCnt - 1)) {
                mCoreUse = mCoreTail;
            }

            if (orderNMFlag) {
                mOffset = inIndex * tiling.singleCoreM;
            } else {
                nOffset = inIndex * tiling.singleCoreN; // 如果是NK就需要对mOffset重新计算
            }
            if (count > 1) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                WaitEvent<PIPE_S>(AIV_SYNC_AIC_FLAG + pingpongEventId);
                WaitEvent<PIPE_S>(AIV_SYNC_AIC_FLAG + pingpongEventId + FLAG_ID_MAX);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
                CrossCoreWaitFlag(AIV_SYNC_AIC_FLAG + pingpongEventId);
#endif
                SyncAll();
                PipeBarrier<PIPE_ALL>();
            }
            count++;

            for (uint64_t kIndex = index; kIndex < (index + realRound); ++kIndex) {
                kOffset = kIndex * tiling.singleCoreK;
                kCoreUse = tiling.singleCoreK;
                if (kIndex == (kCnt - 1)) {
                    kCoreUse = kCoreTail;
                }
                SetOffset<A_TYPE, B_TYPE>(offsetA, offsetB, mOffset, nOffset, kOffset, c0Size, alignedOriM, alignedOriN,
                                          alignedKaSize, alignedKbSize, tiling);
                if (orderNMFlag) {
                    mmNM.SetSingleShape(mCoreUse, nCoreUse, kCoreUse);
                    mmNM.SetTensorA(aGlobal[offsetA], A_TYPE::isTrans);
                    mmNM.SetTensorB(bGlobal[offsetB], B_TYPE::isTrans);
                    isBias &&kIndex == 0 ?
                        mmNM.SetBias(biasGlobal[outIndex * tiling.singleCoreN]) :
                        mmNM.ClearBias(); // set bias at the first k loop and clear bias tag in the following loop
                    mmNM.IterateAll(cGlobal[offsetC], kIndex != index);
                } else {
                    mmMN.SetSingleShape(mCoreUse, nCoreUse, kCoreUse);
                    mmMN.SetTensorA(aGlobal[offsetA], A_TYPE::isTrans);
                    mmMN.SetTensorB(bGlobal[offsetB], B_TYPE::isTrans);
                    isBias &&kIndex == 0 ?
                        mmMN.SetBias(biasGlobal[inIndex * tiling.singleCoreN]) :
                        mmMN.ClearBias(); // set bias at the first k loop and clear bias tag in the following loop
                    mmMN.IterateAll(cGlobal[offsetC], kIndex != index);
                }
            }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId + FLAG_ID_MAX);
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            CrossCoreSetFlag<0x2, PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId);
#endif
            PipeBarrier<PIPE_ALL>();
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    orderNMFlag ? mmNM.SetHF32(false, 0) : mmMN.SetHF32(false, 0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE>
__aicore__ inline void Mc2MatMulKernelDeterministicSplitK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                          const Mc2MatmulV3TilingData &matmulTilingData,
                                                          GM_ADDR workspaceGM)
{
    const TCubeTiling &tiling = matmulTilingData.matmulTiling;
    TPipe que;
    bool orderNMFlag = tiling.iterateOrder; // orderNMFlag表示L2cache切分后循环遍历方式是for N for M  for K
    bool orderFlag = !tiling.iterateOrder;  // tiling侧拿到遍历方向来确认MK或NK，规避知识库
    bool isL2cacheSplit = orderFlag ? tiling.M != tiling.singleCoreM : tiling.N != tiling.singleCoreN;

    uint64_t singleSize = 0;
    uint64_t coreSize = 0;
    uint64_t cnt = 0;
    uint64_t mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
    uint64_t nCnt = MMV3DivCeil(tiling.N, tiling.singleCoreN);

    uint64_t alignedSingleCoreM = MMV3CeilAlign(tiling.singleCoreM, 16); // 384
    uint64_t alignedM = MMV3CeilAlign(tiling.M, 16);                     // 2000
    uint64_t alignedN = MMV3CeilAlign(tiling.N, 16);                     // 32
    alignedM =
        alignedM > static_cast<uint64_t>(tiling.singleCoreM) ? alignedM : static_cast<uint64_t>(tiling.singleCoreM);
    alignedN =
        alignedN > static_cast<uint64_t>(tiling.singleCoreN) ? alignedN : static_cast<uint64_t>(tiling.singleCoreN);

    uint64_t vIndex = GetBlockIdx();
    singleSize = alignedM * alignedN;
    if (isL2cacheSplit) {
        if (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::BASE) {
            singleSize = static_cast<uint64_t>(tiling.singleCoreM) * static_cast<uint64_t>(tiling.singleCoreN);
        }
        coreSize = MMV3DivCeil(tiling.singleCoreM, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) *
                   tiling.singleCoreN; // 无论MK还是NK都按照M方向进行分AIV核
    } else {                           // 不切L2cache
        if (orderFlag) {
            if (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::BASE) {
                singleSize = static_cast<uint64_t>(tiling.singleCoreN) * static_cast<uint64_t>(tiling.M);
            }
            coreSize = MMV3DivCeil(tiling.M, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) *
                       tiling.singleCoreN;
            cnt = nCnt;
        } else {
            if (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::BASE) {
                singleSize = static_cast<uint64_t>(tiling.singleCoreM) * static_cast<uint64_t>(tiling.N);
            }
            coreSize = MMV3DivCeil(singleSize, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO);
            cnt = mCnt;
        }
    }

    GM_ADDR mmGM = workspaceGM;

    if ASCEND_IS_AIV {
        // step3: k reduce and cast in UB
        if (GetBlockIdx() >= (tiling.usedCoreNum * NUM_AIV_TO_AIC_RATIO)) {
            return;
        }
        uint64_t totalSize = singleSize * static_cast<uint64_t>(tiling.usedCoreNum);
        uint64_t outSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
        TBuf<TPosition::VECCALC> tmpBuf;
        que.InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        if (isL2cacheSplit) {
            if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
                ReduceKInUbNzL2cache<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, mCnt, nCnt,
                                             tiling.singleCoreN, tiling.N, tmpBuf, orderNMFlag, tiling, tiling.M);
            } else {
                ReduceKInUbL2cache<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, mCnt, nCnt,
                                           tiling.singleCoreN, tiling.N, tmpBuf, orderNMFlag, tiling);
            }
        } else {
            if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
                ReduceKNzInUb<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, cnt, tiling.singleCoreN,
                                      tiling.N, tmpBuf, orderFlag, tiling, mCnt, nCnt, tiling.M);
            } else {
                ReduceKInUb<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, cnt, tiling.singleCoreN,
                                    tiling.N, tmpBuf, orderFlag, tiling);
            }
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= tiling.usedCoreNum) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            CrossCoreSetFlag<0x2, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }
        GM_ADDR mmOffsetGM = reinterpret_cast<GM_ADDR>(mmGM + GetBlockIdx() * singleSize * NUM_TWO * sizeof(float));
        using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
        if (isL2cacheSplit) {
            if (!matmulTilingData.matmulRunInfo.isNzA && !matmulTilingData.matmulRunInfo.isNzB) {
                Mc2MatMulMultiCoreSplitKDivideL2cache<A_TYPE, B_TYPE, cType, BIAS_TYPE>(
                    aGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling,
                    tiling.isBias);
            } else if (matmulTilingData.matmulRunInfo.isNzA && !matmulTilingData.matmulRunInfo.isNzB) {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                Mc2MatMulMultiCoreSplitKDivideL2cache<aType, B_TYPE, cType, BIAS_TYPE>(
                    aGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling,
                    tiling.isBias);
            } else if (!matmulTilingData.matmulRunInfo.isNzA && matmulTilingData.matmulRunInfo.isNzB) {
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                Mc2MatMulMultiCoreSplitKDivideL2cache<A_TYPE, bType, cType, BIAS_TYPE>(
                    aGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling,
                    tiling.isBias);
            } else {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                Mc2MatMulMultiCoreSplitKDivideL2cache<aType, bType, cType, BIAS_TYPE>(
                    aGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling,
                    tiling.isBias);
            }
        } else {
            if (!matmulTilingData.matmulRunInfo.isNzA && !matmulTilingData.matmulRunInfo.isNzB) {
                Mc2MatMulMultiCoreSplitKDivide<A_TYPE, B_TYPE, cType, BIAS_TYPE>(
                    aGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling,
                    tiling.isBias);
            } else if (matmulTilingData.matmulRunInfo.isNzA && !matmulTilingData.matmulRunInfo.isNzB) {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                Mc2MatMulMultiCoreSplitKDivide<aType, B_TYPE, cType, BIAS_TYPE>(
                    aGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling,
                    tiling.isBias);
            } else if (!matmulTilingData.matmulRunInfo.isNzA && matmulTilingData.matmulRunInfo.isNzB) {
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                Mc2MatMulMultiCoreSplitKDivide<A_TYPE, bType, cType, BIAS_TYPE>(
                    aGM, bGM, biasGM, mmOffsetGM, singleSize, matmulTilingData.matmulRunInfo.isHf32, &que, tiling,
                    tiling.isBias);
            } else {
                using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
                using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
                Mc2MatMulMultiCoreSplitKDivide<aType, bType, cType, BIAS_TYPE>(aGM, bGM, biasGM, mmOffsetGM, singleSize,
                                                                               matmulTilingData.matmulRunInfo.isHf32,
                                                                               &que, tiling, tiling.isBias);
            }
        }
        return;
    }
}

#endif // __OP_KERNEL_MATMUL_V3_H__
