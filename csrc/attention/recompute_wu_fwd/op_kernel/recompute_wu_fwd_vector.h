/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd.h
 * \brief
 */


#ifndef RECOMPUTE_WU_FWD_VECTOR_H
#define RECOMPUTE_WU_FWD_VECTOR_H

#include "recompute_wu_fwd_struct.h"
#include "catlass/arch/cross_core_sync.hpp"
using namespace AscendC;

template <typename kType, typename betaType>
class RecomputeWUFwdVectorProcess {
public:
    /** @brief constructor */
    __aicore__ inline RecomputeWUFwdVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
                                                        GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessVb();
    __aicore__ inline void ProcessKbgExp();
    __aicore__ inline void Init(const GDN::RecomputeWUFwdTilingData &tiling, AscendC::TPipe *pipe_);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t Hv = 1;
    uint64_t Hk = 1;
    uint64_t hvPerHk = 1;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t vbVecRow = 0;
    uint64_t kbgExpVecRow = 0;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;

private:
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_AIC_AIV_FLAG_5, SYNC_AIV_AIC_FLAG_3};
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<kType> workSpaceTensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> vInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECOUT, 1> vbOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> kBetagExpOutQue;

    TBuf<AscendC::TPosition::VECCALC> kFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> vFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> gFp32Buf;
    // TBuf<AscendC::TPosition::VECCALC> gFp32BrcbBuf;

};

template <typename kType, typename betaType>
__aicore__ inline RecomputeWUFwdVectorProcess<kType, betaType>::RecomputeWUFwdVectorProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType>::Init(
    const GDN::RecomputeWUFwdTilingData &tiling, AscendC::TPipe *pipe_)
{
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    B = tiling.B;
    T = tiling.T;
    Hv = tiling.Hv;
    Hk = tiling.Hk;
    hvPerHk = tiling.hvPerHk;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    vbVecRow = tiling.vbVecRow;
    kbgExpVecRow = tiling.kbgExpVecRow;
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType>::Process()
{
    //计算K * Beta[:None]
    ProcessVb();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessKbgExp();
    return;
}


template <typename kType, typename betaType>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType>::ProcessVb()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = vbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t wholeReduceSumCnt = CeilDiv(V, FP32_PER_REPEAT_64);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // // init
    pipe->InitBuffer(vInQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(vbOutQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(vFp32Buf, rowNum * V * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);


    auto tensorVFp32 = vFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < Hv; h++) {
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
                continue;
            }
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * V;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorVin = vInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();

                    DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});

                    vInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                }
                // compute
                {
                    auto tensorVin = vInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();

                    auto tensorVbOut = vbOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    Cast(tensorVFp32, tensorVin, RoundMode::CAST_NONE, V * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < V) {
                        Mul(tensorVFp32[perchannelResOffset], tensorVFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorVbOut, tensorVFp32, RoundMode::CAST_RINT, V * curRowNum);

                    vInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);

                    vbOutQue.EnQue(tensorVbOut);
                }
                // copyout
                {
                    auto tensorVbOut = vbOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[vOffset], tensorVbOut, V * curRowNum);
                    vbOutQue.FreeTensor(tensorVbOut);
                }
            }

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType>::ProcessKbgExp()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = vbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(kBetagExpOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(gFp32Buf, rowNum * sizeof(float32_t));

    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorGFP32 = gFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < Hv; h++) {
            ++vecTaskIdx;
            uint64_t hk = h / hvPerHk;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
                continue;
            }
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                // 注意：定长场景下 GetChunkOffset 返回的 bos 已含按 Hv 计的批次偏移 bIdx*Hv*T
                // （见 recompute_wu_fwd_common.h GetChunkOffset 的 bos += bIdx*H*T，此处 H 传入的是 Hv）。
                // k 只有 Hk 个 head，需把批次偏移换算成 bIdx*Hk*T，即 bos - bIdx*(Hv-Hk)*T。
                // 此换算强耦合于 GetChunkOffset 的批次偏移实现，若后者修改需同步更新此处。
                // coreLoopsInB 必须与 GetChunkOffset 内保持一致的算法。
                uint64_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
                uint64_t bIdx = cu_seqlens ? 0 : (loopIdx / coreLoopsInB);
                uint64_t bosK = cu_seqlens ? bos : (bos - bIdx * (Hv - Hk) * T);
                auto kSrcOffset = (hk * T + bosK + rowOffset) * K;
                auto kDstOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorGin = gInQue.AllocTensor<betaType>();
                    DataCopy(tensorKin, kTensor[kSrcOffset], K * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    DataCopyPad(tensorGin, gTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    gInQue.EnQue(tensorGin);
                }
                // compute
                {
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorGin = gInQue.DeQue<betaType>();
                    auto tensorOut = kBetagExpOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorGFP32, tensorGin, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                        DataCopy(tensorGFP32, tensorGin, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    Exp(tensorGFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    Mul(tensorBetaFP32, tensorBetaFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorKFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    Cast(tensorOut, tensorKFp32, RoundMode::CAST_RINT, K * curRowNum);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    gInQue.FreeTensor(tensorGin);
                    kBetagExpOutQue.EnQue(tensorOut);
                }
                // copyout
                {
                    auto tensorOut = kBetagExpOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[kDstOffset], tensorOut, K * curRowNum);
                    kBetagExpOutQue.FreeTensor(tensorOut);
                }
            }
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
        }
    }
    return;
}


#endif // RECOMPUTE_WU_FWD_VECTOR_H
