/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hc_pre_m_split_core_premix_arch35.h
 * \brief
 */

/*
 * Tiling key 1003 is dedicated to premix M split. Each kUb window is visited
 * for every hc segment while the cast results remain in UB; y is reduced and
 * written during phase 1. Phase 2 therefore does not read x again.
 */

#ifndef HC_PRE_M_SPLIT_CORE_PREMIX_ARCH35_H
#define HC_PRE_M_SPLIT_CORE_PREMIX_ARCH35_H

#include "hc_pre_base_arch35.h"
#include "hc_pre_cube_compute_arch35.h"
#include "kernel_operator.h"

namespace HcPreNs {
using namespace AscendC;

// The public shape check fixes hcMult to four, so one cast slice per segment is sufficient.
constexpr uint32_t PREMIX_YFUSE_CAST_SEG_NUM = 4;

// Reduce one kUb window in the same j order as VFProcessY, then cast it once.
template <typename T>
__aicore__ inline void
VFProcessPremixYFromCastSegments(const LocalTensor<T> &yOutLocal, const LocalTensor<float> &preMixLocal,
                                 const LocalTensor<float> &xCastSegLocal, const uint16_t rowNum, const uint16_t colNum,
                                 const uint16_t hcMult, const uint16_t pmRowStride, const uint16_t segRows) {
    __local_mem__ T *yOutAddr = (__local_mem__ T *)yOutLocal.GetPhyAddr();
    __local_mem__ float *pmAddr = (__local_mem__ float *)preMixLocal.GetPhyAddr();
    __local_mem__ float *xcSegBase = (__local_mem__ float *)xCastSegLocal.GetPhyAddr();
    uint16_t yOutRowStride = RoundUp<T>(colNum);
    uint16_t winBlocks = CeilDiv(colNum, VL_FP32);
    uint32_t xCastRowStride = RoundUp<float>(colNum) + BLOCK_SIZE / sizeof(float);
    uint32_t segStride = static_cast<uint32_t>(segRows) * xCastRowStride;
    __VEC_SCOPE__ {
        RegTensor<float> x;
        RegTensor<float> mix;
        RegTensor<float> sum;
        MaskReg pregFull = CreateMask<float>();
        for (uint16_t i = 0; i < rowNum; i++) {
            for (uint16_t w = 0; w < winBlocks; w++) {
                Duplicate(sum, static_cast<float>(0), pregFull);
                for (uint16_t j = 0; j < hcMult; j++) {
                    LoadInputDataWithBrc<float>(mix, pmAddr, pregFull, i * pmRowStride + j);
                    LoadInputData<float>(x, xcSegBase, pregFull, j * segStride + i * xCastRowStride + w * VL_FP32);
                    Mul(x, mix, x, pregFull);
                    Add(sum, sum, x, pregFull);
                }
                StoreOutputData<T>(yOutAddr, sum, pregFull, i * yOutRowStride + w * VL_FP32);
            }
        }
    }
}

template <typename T> class HcPreMSplitCorePremixArch35 {
public:
    __aicore__ inline HcPreMSplitCorePremixArch35() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcFn, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR preMix, GM_ADDR y,
                                GM_ADDR post, GM_ADDR combFrag, GM_ADDR pre, const HcPreTilingData *tilingDataPtr,
                                TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcFnGm.SetGlobalBuffer((__gm__ float *)hcFn);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);
        hasPreOut_ = (pre != nullptr);
        preMixGm.SetGlobalBuffer((__gm__ float *)preMix);
        if (hasPreOut_) {
            preGm.SetGlobalBuffer((__gm__ float *)pre);
        }
        ubRowGapBlocks_ = UbRowGapBlocks(tilingData->hcMult, tilingData->hcMix);

        TBuf<TPosition::A1> l1Buffer;
        pipe->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        wL1_ = l1Buffer.Get<float>()[L1_BUF_NUM * L1_BUF_OFFSET];

        pipe->InitBufPool(tbufPool0, tilingData->bufferPool0Size);
        tbufPool0.InitBuffer(mmXBuf, CeilDiv(tilingData->mL1Size, DOUBLE_BUFFER) * RoundUp<float>(tilingData->hcMix) *
                                         sizeof(float));
        mmXLocal = mmXBuf.Get<float>();

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        } else {
            tbufPool0.InitBuffer(rmsNormBuf, RoundUp<float>(CeilDiv(tilingData->mL1Size, 2)) * sizeof(float));
            tbufPool0.InitBufPool(tbufPool1, tilingData->bufferPool1Size);

            tbufPool0.InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

            hcBase0Local = hcBaseBuf0.Get<float>();
            hcBase1Local = hcBaseBuf1.Get<float>();
            hcBase2Local = hcBaseBuf2.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t logicalBlockIdx = curBlockIdx;
        if ASCEND_IS_AIV {
            logicalBlockIdx = curBlockIdx / 2;
        }
        if (logicalBlockIdx >= tilingData->cubeBlockDimM) {
            if ASCEND_IS_AIV {
                // Drain the two double-buffer credits seeded by the paired AIC.
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            } else {
                mmService_.End();
            }
            return;
        }

        if ASCEND_IS_AIV {
            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }

        int64_t totalBlockNum = GetBlockNum();

        uint64_t mBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimM;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;

        // todo 移到tiling计算
        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t singleCoreMaxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t mainCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t singleCoreRound =
            (mainCoreCount == 0 || logicalBlockIdx < mainCoreCount) ? singleCoreMaxRound : singleCoreMaxRound - 1;
        uint64_t mGmOffset = 0;
        if ASCEND_IS_AIC {
            if (mainCoreCount == 0 || curBlockIdx <= mainCoreCount) {
                mGmOffset = curBlockIdx * singleCoreMaxRound * tilingData->mL1Size;
            } else {
                mGmOffset =
                    (mainCoreCount * singleCoreMaxRound + (curBlockIdx - mainCoreCount) * (singleCoreMaxRound - 1)) *
                    tilingData->mL1Size;
            }
        } else {
            if (mainCoreCount == 0 || (curBlockIdx / 2) <= mainCoreCount) {
                mGmOffset = curBlockIdx / 2 * singleCoreMaxRound * tilingData->mL1Size;
            } else {
                mGmOffset = (mainCoreCount * singleCoreMaxRound +
                             (curBlockIdx / 2 - mainCoreCount) * (singleCoreMaxRound - 1)) *
                            tilingData->mL1Size;
            }
        }
        int64_t xGmBaseOffset = 0;
        int64_t yGmBaseOffset = 0;
        int64_t postGmBaseOffset = 0;
        int64_t combFragGmBaseOffset = 0;
        if ASCEND_IS_AIV {
            xGmBaseOffset = mGmOffset * tilingData->hcMult * tilingData->d;
            yGmBaseOffset = mGmOffset * tilingData->d;
            postGmBaseOffset = mGmOffset * tilingData->hcMult;
            combFragGmBaseOffset = mGmOffset * tilingData->hcMult * tilingData->hcMult;
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        }

        int64_t xSplitOffset = 0;
        int64_t ySplitOffset = 0;
        int64_t postSplitOffset = 0;
        int64_t combFragSplitOffset = 0;
        // Tiling key 1003 is emitted only when premix y fusion is feasible.
        const uint64_t kChunkWidth = tilingData->kUbSize;
        // m轴切分 按照0 0 1 1..分核
        for (uint64_t roundIdx = 0; roundIdx < singleCoreRound; mGmOffset += tilingData->mL1Size, ++roundIdx) {
            uint64_t mL1RealSize = AscendC::Std::min(tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
            uint64_t kGmStartOffset = 0;
            uint64_t kGmEndOffset = tilingData->multCoreSplitKSize;
            uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
            if ASCEND_IS_AIV {
                tbufPool1.Reset();
                tbufPool1.InitBuffer(xQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
                tbufPool1.InitBuffer(castSegBuf,
                                     PREMIX_YFUSE_CAST_SEG_NUM * tilingData->mUbSize *
                                         (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
                xCastSegLocal = castSegBuf.Get<float>();
                tbufPool1.InitBuffer(yQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
                tbufPool1.InitBuffer(preMixQue, 1,
                                     tilingData->mUbSize * RoundUp<float>(tilingData->hcMix) * sizeof(float));
                tbufPool1.InitBuffer(nd2NzBuf, nd2NzBufSize * sizeof(float) * DOUBLE_BUFFER);

                xNd2NzLocal = nd2NzBuf.Get<float>();
                rmsNormLocal = rmsNormBuf.Get<float>();
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                if (GetBlockIdx() % 2 != 0) {
                    xSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->d;
                    ySplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->d;
                    postSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult;
                    combFragSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->hcMult;
                }
                int64_t preMixRowFactor = CeilDiv(mL1RealSize, 2);
                if (curBlockIdx % 2 == 1) {
                    preMixRowFactor = mL1RealSize - preMixRowFactor;
                }
                preMixLocal = preMixQue.AllocTensor<float>();
                CopyInWithUbStride(
                    preMixGm[postGmBaseOffset + postSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult],
                    preMixLocal, preMixRowFactor, tilingData->hcMult, 0, ubRowGapBlocks_);
                preMixQue.EnQue(preMixLocal);
                preMixLocal = preMixQue.DeQue<float>();
            }
            // k轴切分（kCoreDim=1）
            int64_t bufferIdx = 0;
            if ASCEND_IS_AIV {
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            }
            // Group the same d-window from every hc segment so all cast slices needed by y
            // are resident together.
            const uint64_t kChunkTotal = CeilDiv(kGmEndOffset - kGmStartOffset, kChunkWidth);
            for (uint64_t kChunkIdx = 0; kChunkIdx < kChunkTotal; kChunkIdx++) {
                const uint64_t segIdx = kChunkIdx % (uint64_t)tilingData->hcMult;
                const uint64_t chunkInSeg = kChunkIdx / (uint64_t)tilingData->hcMult;
                const int64_t kGmOffset =
                    kGmStartOffset + (int64_t)(segIdx * (uint64_t)tilingData->d + chunkInSeg * kChunkWidth);
                if ASCEND_IS_AIC {
                    bool isFirstKL1 = kGmOffset == kGmStartOffset;
                    bool isLastKL1 = (kGmOffset + kChunkWidth) >= kGmEndOffset;
                    uint64_t kL1RealSize = AscendC::Std::min(kGmEndOffset - kGmOffset, kChunkWidth);
                    mmService_.CopyInB1Nd2Nz(tilingData->multCoreSplitKSize, kL1RealSize, tilingData->hcMix,
                                             hcFnGm[kGmOffset], wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                    uint64_t mL1AlignSize = Align(mL1RealSize, AscendC::BLOCK_CUBE);
                    uint64_t nL1AlignSize = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);
                    mmService_.Process(tilingData->bs, tilingData->hcMix, mL1RealSize,
                                       (256 / AscendC::Std::max(mL1AlignSize, nL1AlignSize)) * 32, isFirstKL1,
                                       isLastKL1, xL1_[aL1BufferID_ * L1_BUF_OFFSET],
                                       wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    if (isLastKL1) {
                        mmService_.CopyOut(mmXLocal);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG + FLAG_ID_MAX);
                    }
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(
                        SYNC_AIC_AIV_FLAG); // 写出ub搬出，cv流水同步比较复杂，暂不讨论
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                    // 偶数核取前半段(CeilDiv，多处理一行)、奇数核取后半段，须与 sinkhorn/输出阶段的行划分一致，
                    // 否则 mL1RealSize 为奇数时后半段错位一行。
                    int64_t rowFactor = CeilDiv(mL1RealSize, 2);
                    int64_t tailRowFactor = mL1RealSize - rowFactor;
                    int64_t curRowFactor = rowFactor;
                    int64_t mL1SizeAlign = CeilAlign(mL1RealSize, AscendC::BLOCK_CUBE);
                    if (curBlockIdx % 2 == 1) {
                        curRowFactor = tailRowFactor;
                    }
                    float coeff = 1 / static_cast<float>(tilingData->hcMult * tilingData->d);
                    int64_t castSegStride = static_cast<int64_t>(tilingData->mUbSize) *
                                            (RoundUp<float>(tilingData->kUbSize) + BLOCK_SIZE / sizeof(float));
                    LocalTensor<float> xCastDst =
                        xCastSegLocal[static_cast<uint32_t>(segIdx) * static_cast<uint32_t>(castSegStride)];
                    for (int64_t cvLoopIdx = 0; cvLoopIdx < 1; cvLoopIdx++) {
                        xLocal = xQue.template AllocTensor<T>();
                        CopyIn(xGm[xGmBaseOffset + xSplitOffset +
                                   roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->d + kGmOffset +
                                   cvLoopIdx * tilingData->kUbSize],
                               xLocal, curRowFactor, tilingData->kUbSize,
                               tilingData->hcMult * tilingData->d - tilingData->kUbSize);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();
                        if (kGmOffset == kGmStartOffset && cvLoopIdx == 0) {
                            VFProcessCastAndInvRmsPart1<T, false>(rmsNormLocal, xCastDst, xLocal, coeff, curRowFactor,
                                                                  tilingData->kUbSize);
                        } else {
                            VFProcessCastAndInvRmsPart1<T, true>(rmsNormLocal, xCastDst, xLocal, coeff, curRowFactor,
                                                                 tilingData->kUbSize);
                        }
                        xQue.template FreeTensor(xLocal);

                        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        VFTransND2NZ(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastDst, curRowFactor,
                                     tilingData->kUbSize);
                        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));

                        if (curBlockIdx % 2 == 0) {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
                            dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                            dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) - curRowFactor;
                            dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curRowFactor;
                            CopyToL1(
                                xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)],
                                xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign],
                                dataCopyXParams);
                        } else {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
                            dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                            dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) - curRowFactor;
                            dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curRowFactor;
                            CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)],
                                     xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + rowFactor * (BLOCK_SIZE / sizeof(float)) +
                                          cvLoopIdx * tilingData->kUbSize * mL1SizeAlign],
                                     dataCopyXParams);
                        }
                        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        bufferIdx++;
                    }
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                    if (segIdx == (uint64_t)tilingData->hcMult - 1) {
                        // F2(b)-②: 本窗口 c 组全部 hcMult 段 chunk 完成 → 4 cast slice 恰好
                        // 全驻留，每行做 j 内层寄存器链累加（sum 寄存器驻留、全 lane 常量
                        // mask、brc 每行每段每窗口一次）+ 单次 cast BF16 写出；置于
                        // CrossCoreSetFlag 之后，不阻塞配对 AIC 取下一 chunk。跨窗口的
                        // slice 复用冒险由 VEC 管内按序程序序覆盖（F2(b)-⑦：下一窗口段 0
                        // 的 cast store 在本 pass 的 load 之后，同管 store→load 既定可靠序）。
                        // 窗口 GM 偏移 = chunkInSeg*kUbSize（kUb-chunk 粒度下与契约
                        // chunkInSeg*kL1Size + window*kUbSize 的线性偏移等价）。
                        yLocal = yQue.template AllocTensor<T>();
                        VFProcessPremixYFromCastSegments<T>(yLocal, preMixLocal, xCastSegLocal, (uint16_t)curRowFactor,
                                                            (uint16_t)tilingData->kUbSize, (uint16_t)tilingData->hcMult,
                                                            (uint16_t)RoundUp<float>(tilingData->hcMix),
                                                            (uint16_t)tilingData->mUbSize);
                        yQue.template EnQue(yLocal);
                        yLocal = yQue.template DeQue<T>();
                        CopyOut(yLocal,
                                yGm[yGmBaseOffset + ySplitOffset + roundIdx * tilingData->mL1Size * tilingData->d +
                                    (int64_t)chunkInSeg * tilingData->kUbSize],
                                curRowFactor, tilingData->kUbSize, tilingData->d - tilingData->kUbSize);
                        yQue.template FreeTensor(yLocal);
                    }
                }
                aL1BufferID_ ^= 1;
            }

            if ASCEND_IS_AIV {
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
                preMixQue.FreeTensor(preMixLocal);
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(SYNC_AIC_AIV_PRE_POST_FLAG);
                // mm计算结果存入mmXLocal，mmXLocal每轮循环需要累加;
                tbufPool1.Reset();
                tbufPool1.InitBuffer(postQue, 2, tilingData->rowInnerFactor * tilingData->hcMultAlign * sizeof(float));

                // TBuf
                tbufPool1.InitBuffer(mixesBuf,
                                     tilingData->rowInnerFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));

                // 可选输出pre使用独立UB空间（hcMultAlign行距紧凑布局），TQue double buffer
                // 管理V->MTE3同步，MTE3搬出与下一轮V计算重叠；未请求输出时跳过分配
                if (hasPreOut_) {
                    tbufPool1.InitBuffer(preQue, DOUBLE_BUFFER,
                                         tilingData->rowInnerFactor * tilingData->hcMultAlign * sizeof(float));
                }

                mixesLocal = mixesBuf.Get<float>();

                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

                // m内层循环
                int64_t currentRow = mL1RealSize / 2;
                if (mL1RealSize % 2 == 1 && curBlockIdx % 2 == 0) {
                    // m不整除时偶数核多处理一行
                    currentRow += 1;
                }
                for (int64_t innerRowIdx = 0; innerRowIdx < currentRow; innerRowIdx += tilingData->rowInnerFactor) {
                    int64_t currentInnerRowFactor = innerRowIdx + tilingData->rowInnerFactor >= currentRow
                                                        ? currentRow - innerRowIdx
                                                        : tilingData->rowInnerFactor;
                    VFProcessInvRmsPart3(mixesLocal, mmXLocal[innerRowIdx * tilingData->hcMix],
                                         rmsNormLocal[innerRowIdx], tilingData->normEps, currentInnerRowFactor,
                                         tilingData->hcMix);

                    if (hasPreOut_) {
                        VFProcessPre(mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps,
                                     currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);
                    }
                    if (hasPreOut_) {
                        // pre与post同为[bs, hcMult]，复用post的GM偏移；先将mixesLocal行首的hcMult个
                        // 元素按hcMixAlign行距聚拢到preLocal(hcMultAlign行距)，再经TQue异步搬出
                        preLocal = preQue.AllocTensor<float>();
                        CopyOut(mixesLocal, preLocal, currentInnerRowFactor, tilingData->hcMult, 0, ubRowGapBlocks_);
                        preQue.EnQue(preLocal);
                        preLocal = preQue.DeQue<float>();
                        CopyOut(preLocal,
                                preGm[postGmBaseOffset + postSplitOffset +
                                      roundIdx * tilingData->mL1Size * tilingData->hcMult +
                                      innerRowIdx * tilingData->hcMult],
                                currentInnerRowFactor, tilingData->hcMult);
                        preQue.FreeTensor(preLocal);
                    }
                    // post
                    postLocal = postQue.AllocTensor<float>();
                    VFProcessPost(postLocal, mixesLocal[tilingData->hcMult], hcBase1Local, hcScaleGm.GetValue(1),
                                  tilingData->hcEps, currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);

                    postQue.EnQue(postLocal);
                    postLocal = postQue.DeQue<float>();
                    CopyOut(
                        postLocal,
                        postGm[postGmBaseOffset + postSplitOffset +
                               roundIdx * tilingData->mL1Size * tilingData->hcMult + innerRowIdx * tilingData->hcMult],
                        currentInnerRowFactor, tilingData->hcMult);
                    postQue.FreeTensor(postLocal);

                    VFProcessCombFragPremixElementMajor(mixesLocal, hcBase2Local, hcScaleGm.GetValue(2),
                                                        tilingData->hcEps, tilingData->iterTimes - 1,
                                                        currentInnerRowFactor, tilingData->hcMix);
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                    CopyOut(mixesLocal,
                            combFragGm[combFragGmBaseOffset + combFragSplitOffset +
                                       roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->hcMult +
                                       innerRowIdx * tilingData->hcMult * tilingData->hcMult],
                            1, currentInnerRowFactor * tilingData->hcMult * tilingData->hcMult);
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                }
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            }
        }
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            // Drain the two double-buffer credits seeded by the paired AIC.
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        } else {
            mmService_.End();
        }
    }

private:
    TPipe *pipe;
    const HcPreTilingData *tilingData;
    // (M, K) * (N, K)

    GlobalTensor<T> xGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> invRmsGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;
    GlobalTensor<float> preMixGm;
    GlobalTensor<float> preGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECIN, 1> preMixQue;
    TQue<QuePosition::VECOUT, 1> preQue;

    TBuf<QuePosition::VECCALC> castSegBuf;
    TBuf<QuePosition::VECCALC> nd2NzBuf;

    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> rsqrtBuf;
    TBuf<QuePosition::VECCALC> squareReduceBuf;
    TBuf<QuePosition::VECCALC> mixes01ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> mixesBuf;
    TBuf<QuePosition::VECCALC> rmsNormBuf;
    TBuf<QuePosition::VECCALC> mmXBuf;

    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> mmXLocal;
    LocalTensor<float> rmsNormLocal;
    LocalTensor<float> xCastSegLocal;
    LocalTensor<float> xNd2NzLocal;

    LocalTensor<float> mixesLocal;
    LocalTensor<float> rmsAndmmLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> preMixLocal;
    LocalTensor<float> preLocal;
    bool hasPreOut_ = false;
    uint32_t ubRowGapBlocks_ = 0;

    HcPreCubeCompute mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> wL1_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_AIC_AIV_PRE_POST_FLAG = 10;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    uint64_t cvLoopIdx_ = 0;
    uint8_t aL1BufferID_{0};

    static constexpr uint32_t UB_POOL_BUFFER_COUNT = 16;
    TBufPool<QuePosition::VECCALC, UB_POOL_BUFFER_COUNT> tbufPool0;
    TBufPool<QuePosition::VECCALC, UB_POOL_BUFFER_COUNT> tbufPool1;
};

} // namespace HcPreNs

#endif
