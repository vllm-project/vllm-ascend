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
 * \file mat_mul_nd2nz.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_ND2NZ_H__
#define __OP_KERNEL_MATMUL_V3_ND2NZ_H__

#include "mat_mul_nd2nz_util.h"
#include "mat_mul_nd2nz_kernel.h"
#include "batch_mat_mul_nd2nz.h"

using namespace AscendC;
using namespace matmul;
#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

template <class T>
__aicore__ inline void CopyPadNd2Nz(const GlobalTensor<T> &dstGlobal, const GlobalTensor<T> &srcGlobal, uint32_t baseH,
                                    uint32_t baseW, uint32_t orgHeight, uint32_t orgWidth, LocalTensor<T> ubLocal1,
                                    LocalTensor<T> ubLocal2, uint8_t padH, uint8_t padW)
{
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);

    uint32_t width = baseW - padW;
    uint32_t height = baseH - padH;

    // Process gm->ub
    uint16_t blockLen = width * sizeof(T);
    uint32_t srcStride = (orgWidth - width) * sizeof(T);
    uint32_t numIter = height / BLOCK_COUNT_MAX;
    for (uint32_t i = 0; i < numIter; i++) {
        DataCopyPad(ubLocal1[BLOCK_COUNT_MAX * i * baseW],
                    srcGlobal[static_cast<uint64_t>(BLOCK_COUNT_MAX) * i * orgWidth],
                    {BLOCK_COUNT_MAX, blockLen, srcStride, 0, 0}, {false, 0, padW, 0});
    }
    uint16_t blockCountTail = height % BLOCK_COUNT_MAX;

    if (blockCountTail) {
        DataCopyPad(ubLocal1[BLOCK_COUNT_MAX * numIter * baseW],
                    srcGlobal[static_cast<uint64_t>(BLOCK_COUNT_MAX) * numIter * orgWidth],
                    {blockCountTail, blockLen, srcStride, 0, 0}, {false, 0, padW, 0});
    }

    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));

    // padding
    if (padH) {
        Duplicate(ubLocal1[height * baseW], (T)0, padH * baseW);
        PipeBarrier<PIPE_V>();
    }

    // Process ub->ub
    uint32_t nRepeat = SINGLE_COPY_SIZE / sizeof(T);
    uint16_t nRowBlock = baseW / c0Size;
    uint32_t numIterI = baseH / REPEAT_TIMES_MAX;
    uint32_t heightTail = baseH % REPEAT_TIMES_MAX;
    uint32_t numIterJ = baseW / nRepeat;
    uint32_t widthTail = baseW % nRepeat;

    for (uint32_t i = 0; i < numIterI; i++) {
        for (uint32_t j = 0; j < numIterJ; j++) {
            Copy(ubLocal2[baseH * nRepeat * j + i * REPEAT_TIMES_MAX * c0Size],
                 ubLocal1[nRepeat * j + i * REPEAT_TIMES_MAX * baseW], nRepeat, REPEAT_TIMES_MAX,
                 {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
        }
        if (widthTail) {
            Copy(ubLocal2[baseH * nRepeat * numIterJ + i * REPEAT_TIMES_MAX * c0Size],
                 ubLocal1[nRepeat * numIterJ + i * REPEAT_TIMES_MAX * baseW], widthTail, REPEAT_TIMES_MAX,
                 {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
        }
    }
    for (uint32_t j = 0; j < numIterJ; j++) {
        Copy(ubLocal2[baseH * nRepeat * j + numIterI * REPEAT_TIMES_MAX * c0Size],
             ubLocal1[nRepeat * j + numIterI * REPEAT_TIMES_MAX * baseW], nRepeat, heightTail,
             {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
    }
    if (widthTail) {
        Copy(ubLocal2[baseH * nRepeat * numIterJ + numIterI * REPEAT_TIMES_MAX * c0Size],
             ubLocal1[nRepeat * numIterJ + numIterI * REPEAT_TIMES_MAX * baseW], widthTail, heightTail,
             {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
    }

    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));

    uint64_t orgHeightRound = MMV3DivCeil(orgHeight, ALIGNED_H) * ALIGNED_H;
    // Process ub->gm
    if (orgHeightRound - baseH <= UINT16_MAX) {
        DataCopy(dstGlobal, ubLocal2, {nRowBlock, static_cast<uint16_t>(baseH), 0, uint16_t(orgHeightRound - baseH)});
    } else {
        for (uint16_t i = 0; i < nRowBlock; i++) {
            DataCopy(dstGlobal[orgHeightRound * c0Size * i], ubLocal2[baseH * c0Size * i],
                     {1, static_cast<uint16_t>(baseH), 0, 0});
        }
    }
}

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ > 200)
template <>
__aicore__ inline void
CopyPadNd2Nz<bfloat16_t>(const GlobalTensor<bfloat16_t> &dstGlobal, const GlobalTensor<bfloat16_t> &srcGlobal,
                         uint32_t baseH, uint32_t baseW, uint32_t orgHeight, uint32_t orgWidth,
                         LocalTensor<bfloat16_t> ubLocal1, LocalTensor<bfloat16_t> ubLocal2, uint8_t padH, uint8_t padW)
{
    GlobalTensor<half> dstGlobalTrans;
    GlobalTensor<half> srcGlobalTrans;
    dstGlobalTrans.SetGlobalBuffer((__gm__ half *)dstGlobal.GetPhyAddr(0));
    srcGlobalTrans.SetGlobalBuffer((__gm__ half *)srcGlobal.GetPhyAddr(0));
    CopyPadNd2Nz<half>(dstGlobalTrans, srcGlobalTrans, baseH, baseW, orgHeight, orgWidth,
                       ubLocal1.ReinterpretCast<half>(), ubLocal2.ReinterpretCast<half>(), padH, padW);
}
#endif

#if defined(__DAV_C220_VEC__)
template <class T, Nd2NzMode mode = Nd2NzMode::MULTI_CORE>
__aicore__ inline bool Nd2nzVnchwMM(GlobalTensor<T> &dst, GlobalTensor<T> &src, uint32_t height, uint32_t width,
                                    uint32_t batch, TBuf<TPosition::VECCALC> &ubBuffer, uint32_t usedCoreNum)
{
    Mc2KernelND2NZMM<T> op;
    op.Init((GM_ADDR)dst[0].GetPhyAddr(), (GM_ADDR)src[0].GetPhyAddr(), height, width, batch, ubBuffer, usedCoreNum);
    return op.template ProcessMM<mode>();
}

template <Nd2NzMode mode = Nd2NzMode::MULTI_CORE>
__aicore__ inline bool Nd2nzVnchwMM(GlobalTensor<bfloat16_t> &dst, GlobalTensor<bfloat16_t> &src, uint32_t height,
                                    uint32_t width, uint32_t batch, TBuf<TPosition::VECCALC> &ubBuffer,
                                    uint32_t usedCoreNum)
{
    GlobalTensor<half> dstGlobalTrans;
    GlobalTensor<half> srcGlobalTrans;
    dstGlobalTrans.SetGlobalBuffer((__gm__ half *)dst.GetPhyAddr(0));
    srcGlobalTrans.SetGlobalBuffer((__gm__ half *)src.GetPhyAddr(0));
    return Nd2nzVnchwMM<half, mode>(dstGlobalTrans, srcGlobalTrans, height, width, batch, ubBuffer, usedCoreNum);
}
#endif

template <class T>
__aicore__ inline void MatrixtoNZ(uint64_t oriN, uint64_t oriD, uint64_t nValue, uint64_t dValue, uint32_t baseN,
                                  uint32_t baseD, uint32_t usedCoreNum, GlobalTensor<T> &tempSrcGlobal,
                                  GlobalTensor<T> &tempDstGlobal, TBuf<TPosition::VECCALC> &tmpBuf)
{
    if ASCEND_IS_AIV {
        uint32_t vBlockIndex = GetBlockIdx();
        uint64_t c0Size;
        GetSizeC0<T>(c0Size);
        LocalTensor<T> tempUb = tmpBuf.Get<T>();
        LocalTensor<T> transBuf = tempUb[(TOTAL_UB_SIZE / NUM_TWO) / sizeof(T)];
        uint64_t nCnt = MMV3DivCeil(oriN, baseN);
        uint64_t dCnt = MMV3DivCeil(dValue, baseD);
        uint64_t nBaseTil = nValue - (nCnt - 1) * baseN;  // n方向上的baseN尾块
        uint64_t dBaseTail = dValue - (dCnt - 1) * baseD; // m方向上的baseM尾块
        uint64_t totalCnt = nCnt * dCnt;
        uint32_t round = (totalCnt + usedCoreNum - 1) / usedCoreNum; // 每一个core最大做base块计算的次数
        uint32_t realRound = 0;                                      // 单核做多少次base块计算
        uint32_t preCoreNum = totalCnt % usedCoreNum; // 从0core开始有多少个core会多做一次base块
        uint32_t preTotalBlock = 0;
        uint32_t index = 0; // 当前block_idx的起始基本块Index，这个idex是按照先循环D，再循环N的次序
        if (preCoreNum == 0) {
            preCoreNum = usedCoreNum;
        }
        // ND
        if (vBlockIndex < preCoreNum) {
            index = vBlockIndex * round;
            // 前面preCoreNum个core会多做一次
            realRound = round;
        } else {
            index = vBlockIndex * (round - 1) + preCoreNum;
            // 后面的core会少做一次
            realRound = round - 1;
        }
        uint32_t nCalcLen = 0;
        uint32_t dCalcLen = 0;
        uint32_t padN = 0;
        uint32_t padD = 0;
        uint32_t nIndx = 0;
        uint32_t dIndx = 0;
        uint32_t lastD = oriD % baseD;
        for (uint32_t j = 0; j < realRound; ++j) {
            if (index < totalCnt) {
                if ((index + 1) % (nCnt * dCnt) == 0) {
                    // 最后一块是尾块
                    nCalcLen = nBaseTil;
                    dCalcLen = dBaseTail;
                } else if ((index + 1) % (nCnt * dCnt) > (nCnt - 1) * dCnt) {
                    // n方向尾块
                    nCalcLen = nBaseTil;
                    dCalcLen = baseD;
                } else if ((index + 1) % dCnt == 0) {
                    // d方向尾块
                    nCalcLen = baseN;
                    dCalcLen = dBaseTail;
                } else {
                    // 对齐整块
                    nCalcLen = baseN;
                    dCalcLen = baseD;
                }
            }
            // calc pad_value
            nIndx = index / dCnt;
            dIndx = index % dCnt;
            padN = (nIndx == nCnt - 1) ? nValue - oriN : 0;
            padD = (dIndx == dCnt - 1) ? dValue - oriD : 0; // will be used ???
            auto srcGmIdx = (dIndx * baseD + nIndx * baseN * oriD);
            auto dstGmIdx = (dIndx * nValue * baseD + nIndx * baseN * c0Size);
            CopyPadNd2Nz(tempDstGlobal[dstGmIdx], tempSrcGlobal[srcGmIdx], nCalcLen, dCalcLen, oriN, oriD, tempUb,
                         transBuf, padN, padD);
            event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            index += 1;
        }
    }
}

template <class T>
__aicore__ inline void MatrixAtoNZV2(GM_ADDR workspace, GM_ADDR src, const TCubeTiling &cfg, bool isTransposeA,
                                     TBuf<TPosition::VECCALC> &tmpBuf, uint32_t baseAN, uint32_t baseAD,
                                     uint32_t batch = 1)
{
    uint64_t c0Size = 16;
    GetSizeC0<T>(c0Size);
    uint32_t usedCoreNum = cfg.usedCoreNum * GetTaskRation(); // 使用最大的核数
    uint64_t alignedMSize = 0;
    uint64_t alignedKSize = 0;
    alignedMSize =
        isTransposeA ? MMV3DivCeil(cfg.M, c0Size) * c0Size : MMV3DivCeil(cfg.M, ALIGNED_H) * ALIGNED_H; // N轴转换成分型
    alignedKSize = isTransposeA ? MMV3DivCeil(cfg.Ka, ALIGNED_H) * ALIGNED_H :
                                  MMV3DivCeil(cfg.Ka, c0Size) * c0Size; // K轴转换成分型
    uint64_t oriN = isTransposeA ? cfg.Ka : cfg.M;
    uint64_t oriD = isTransposeA ? cfg.M : cfg.Ka;
    uint64_t nValue = isTransposeA ? alignedKSize : alignedMSize;
    uint64_t dValue = isTransposeA ? alignedMSize : alignedKSize;
    GlobalTensor<T> tempSrcGlobal;
    GlobalTensor<T> tempDstGlobal;
    tempDstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace), alignedMSize * alignedKSize);
    tempSrcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), oriD * oriN);
#if defined(__DAV_C220_VEC__)
    if (batch > 1) {
        Mc2Nd2nzVnchwBMM(tempDstGlobal, tempSrcGlobal, oriN, oriD, batch, tmpBuf, usedCoreNum);
    } else {
        // outersize over 8192 and (innersize below 384B when it is even or below 192B when it is odd) for vnchw.
        if ((oriN > 8192) && (oriD > 1) && (oriD != c0Size) &&
            (oriD * sizeof(T) <= 192 || (oriD * sizeof(T) <= 384 && oriD % 2 == 0) ||
             (oriD * sizeof(T) <= 512 && oriD % 4 == 0))) { // 32 is blocksize, 512 is cacheline
            Nd2nzVnchwMM(tempDstGlobal, tempSrcGlobal, oriN, oriD, batch, tmpBuf, usedCoreNum);
        } else {
            MatrixtoNZ(oriN, oriD, nValue, dValue, baseAN, baseAD, usedCoreNum, tempSrcGlobal, tempDstGlobal, tmpBuf);
        }
    }
#else
    MatrixtoNZ(oriN, oriD, nValue, dValue, baseAN, baseAD, usedCoreNum, tempSrcGlobal, tempDstGlobal, tmpBuf);
#endif
}

template <class T>
__aicore__ inline void MatrixBtoNZV2(GM_ADDR workspace, GM_ADDR src, const TCubeTiling &cfg, bool isTransposeB,
                                     TBuf<TPosition::VECCALC> &tmpBuf, uint32_t baseBN, uint32_t baseBD,
                                     uint32_t batch = 1)
{
    uint64_t c0Size = 16;
    GetSizeC0<T>(c0Size);
    uint32_t usedCoreNum = cfg.usedCoreNum * GetTaskRation(); // 使用最大的核数
    uint64_t alignedNSize = 0;
    uint64_t alignedKSize = 0;
    alignedNSize = isTransposeB ? MMV3DivCeil(cfg.N, ALIGNED_H) * ALIGNED_H : MMV3DivCeil(cfg.N, c0Size) * c0Size;
    alignedKSize = isTransposeB ? MMV3DivCeil(cfg.Kb, c0Size) * c0Size : MMV3DivCeil(cfg.Kb, ALIGNED_H) * ALIGNED_H;
    uint64_t oriN = isTransposeB ? cfg.N : cfg.Kb;
    uint64_t oriD = isTransposeB ? cfg.Kb : cfg.N;
    uint64_t nValue = isTransposeB ? alignedNSize : alignedKSize;
    uint64_t dValue = isTransposeB ? alignedKSize : alignedNSize;
    GlobalTensor<T> tempSrcGlobal1;
    GlobalTensor<T> tempDstGlobal1;
    tempDstGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace), alignedNSize * alignedKSize);
    tempSrcGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), oriD * oriN);
#if defined(__DAV_C220_VEC__)
    if (batch > 1) {
        Mc2Nd2nzVnchwBMM(tempDstGlobal1, tempSrcGlobal1, oriN, oriD, batch, tmpBuf, usedCoreNum);
    } else {
        // outersize over 8192 and (innersize below 384B when it is even or below 192B when it is odd) for vnchw.
        if ((oriN > 8192) && (oriD > 1) && (oriD != c0Size) &&
            (oriD * sizeof(T) <= 192 || (oriD * sizeof(T) <= 384 && oriD % 2 == 0) ||
             (oriD * sizeof(T) <= 512 && oriD % 4 == 0))) { // 32 is blocksize, 512 is cacheline
            Nd2nzVnchwMM(tempDstGlobal1, tempSrcGlobal1, oriN, oriD, batch, tmpBuf, usedCoreNum);
        } else {
            MatrixtoNZ(oriN, oriD, nValue, dValue, baseBN, baseBD, usedCoreNum, tempSrcGlobal1, tempDstGlobal1, tmpBuf);
        }
    }
#else
    MatrixtoNZ(oriN, oriD, nValue, dValue, baseBN, baseBD, usedCoreNum, tempSrcGlobal1, tempDstGlobal1, tmpBuf);
#endif
}

#endif
