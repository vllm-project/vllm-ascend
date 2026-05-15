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
 * \file grouped_matmul_finalize_routing.h
 * \brief
 */

#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "recurrent_gated_delta_rule_tiling_data.h"

namespace RecurrentGatedDeltaRule {

using namespace matmul;
using namespace AscendC;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64; // 256Byte for float
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t ADD_FOLD_REDUCE_MIN_K = 128;

constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t REPEAT_BYTES = 256;
constexpr int64_t REPEAT_BLOCKS = 8;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

// CastOrCopy: when DST==SRC (e.g. float→float), Cast is unsupported; use Adds(x,0) as copy.
template <typename DST, typename SRC>
__aicore__ inline void CastOrCopy(LocalTensor<DST> &dst, const LocalTensor<SRC> &src,
                                   AscendC::RoundMode mode, uint32_t count)
{
    if constexpr (std::is_same_v<DST, SRC>) {
        Adds(dst, src, static_cast<DST>(0), count);
    } else {
        Cast(dst, src, mode, count);
    }
}

template <typename T>
__aicore__ inline void SwapTensor(LocalTensor<T> &a, LocalTensor<T> &b)
{
    LocalTensor<T> tmp = a;
    a = b;
    b = tmp;
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(LocalTensor<T> inLocal, GlobalTensor<T> srcGm,
                                          DataCopyExtParams tokenCopyParams, DataCopyPadExtParams<T> padParams)
{
    int64_t elem = tokenCopyParams.blockLen / sizeof(T);
    int64_t numPerBlock = BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);
    int64_t srcStrideElem = tokenCopyParams.srcStride / sizeof(T);
    int64_t gmStepPerRow = elem + srcStrideElem;

    if (likely(alignElem == elem && srcStrideElem == 0)) {
        DataCopyParams copyParams = {tokenCopyParams.blockCount,
                                     static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(inLocal, srcGm, copyParams);
    } else {
        DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        for (uint32_t i = 0; i < tokenCopyParams.blockCount; i++) {
            DataCopy(inLocal[i * alignElem], srcGm[i * gmStepPerRow], copyParams);
        }
    }
}

// DataCopyCustom: converts DataCopyParams (blockLen in bytes) to block units for DataCopy on 310P
template <typename DST, typename SRC>
__aicore__ inline void DataCopyCustom(DST dst, SRC src, DataCopyParams copyParams)
{
    int64_t alignBytes = AlignUp(static_cast<int64_t>(copyParams.blockLen), BLOCK_BYTES);
    int64_t blocks = alignBytes
     / BLOCK_BYTES;
    DataCopyParams aligned = {copyParams.blockCount, static_cast<uint16_t>(blocks), 0, 0};
    DataCopy(dst, src, aligned);
}

template <typename T, bool needBack = false, bool isAtomic = false>
__aicore__ inline void DataCopyCustom(GlobalTensor<T> dstGm, LocalTensor<T> inLocal,
                                       DataCopyExtParams copyParamsIn)
{
    int64_t elem = copyParamsIn.blockLen / sizeof(T);
    int64_t numPerBlock = sizeof(T) == 0 ? 1 : BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                     static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(dstGm, inLocal, copyParams);
    } else {
        if (copyParamsIn.blockCount == 1) {
            if constexpr (needBack) {
                int64_t elemAlignDown = numPerBlock == 0 ? 0 : elem / numPerBlock * numPerBlock;
                if (elemAlignDown != 0) {
                    DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                                 static_cast<uint16_t>(elemAlignDown / numPerBlock), 0, 0};
                    DataCopy(dstGm, inLocal, copyParams);
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    for (uint32_t i = 0; i < numPerBlock; i++) {
                        inLocal.SetValue(alignElem - 1 - i, inLocal.GetValue(elem - 1 - i));
                    }
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyParams copyParamslast = {1, 1, 0, 0};
                    DataCopy(dstGm[elem - numPerBlock], inLocal[elemAlignDown], copyParamslast);
                } else {
                    T tmp[BLOCK_BYTES];
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    for (uint32_t i = 0; i < elem; i++) {
                        tmp[i] = inLocal.GetValue(elem - 1 - i);
                    }
                    DataCopyParams copyParamslast = {1, 1, 0, 0};
                    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                    DataCopy(inLocal, dstGm[elem - numPerBlock], copyParamslast);
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    for (uint32_t i = 0; i < elem; i++) {
                        inLocal.SetValue(numPerBlock - 1 - i, tmp[i]);
                    }
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopy(dstGm[elem - numPerBlock], inLocal, copyParamslast);
                }
            } else if constexpr (isAtomic) {
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                for (uint32_t i = 0; i < alignElem - elem; i++) {
                    inLocal.SetValue(alignElem - 1 - i, T(0));
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                             static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
                DataCopy(dstGm, inLocal, copyParams);
            } else {
                DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                             static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
                DataCopy(dstGm, inLocal, copyParams);
            }
        } else {
            DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
            for (uint32_t i = 0; i < copyParamsIn.blockCount; i++) {
                DataCopy(dstGm[i * elem], inLocal[i * alignElem], copyParams);
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }
}

#ifndef RGDR_ENABLE_ADD_FOLD_REDUCE
#define RGDR_ENABLE_ADD_FOLD_REDUCE  1
#endif
struct RGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gama;
    GM_ADDR gamaK;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename inType, typename outType>
class RGDR {
public:
    __aicore__ inline RGDR(const RecurrentGatedDeltaRuleTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        NK_ = tilingData->nk;
        realK_ = tilingData->dk;
        NV_ = tilingData->nv;
        realV_ = tilingData->dv;
        scale_ = tilingData->scale;
        hasAcceptedTokens_ = (tilingData->hasAcceptedTokens == 1);
        hasGama_ = (tilingData->hasGama == 1);
        hasGamaK_ = (tilingData->hasGamaK == 1);
        useAddFoldReduce_ = (RGDR_ENABLE_ADD_FOLD_REDUCE != 0);
        vStep_ = tilingData->vStep;
        restUbSize_ = tilingData->ubRestBytes;
        alignK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        load = 0;
        usedblk = 0;
    }

    __aicore__ inline void Init(const RGDRInitParams &initParams, TPipe *pipe)
    {
        uint64_t blockDim = GetBlockNum();
        blockIdx = GetBlockIdx();
        if (blockIdx >= blockDim) {
            return;
        }
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const RGDRInitParams &initParams)
    {
        queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value);
        gamaGm_.SetGlobalBuffer((__gm__ float *)initParams.gama);
        gamaKGm_.SetGlobalBuffer((__gm__ float *)initParams.gamaK);
        betaGm_.SetGlobalBuffer((__gm__ inType *)initParams.beta);
        initStateGm_.SetGlobalBuffer((__gm__ inType *)initParams.initState);
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
        ssmStateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        numAcceptedTokensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
        finalStateGm_.SetGlobalBuffer((__gm__ outType *)initParams.finalState);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
        uint32_t singleVSize = vStep_ * sizeof(float);
        uint32_t vSize = MAX_MTP * alignV_ * sizeof(float);
        uint32_t kSize = MAX_MTP * alignK_ * sizeof(float);
        uint32_t betaUbSize =
            Ceil(MAX_MTP * NV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK * sizeof(float); //  8: 8 * 4 = 32B;
        pipe_->InitBuffer(qInBuf_, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(kInBuf_, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(vInBuf_, MAX_MTP * alignV_ * sizeof(inType));
        pipe_->InitBuffer(stateInBuf_, alignK_ * vStep_ * sizeof(inType));
        if (hasGama_) {
            pipe_->InitBuffer(gamaInBuf_, MAX_MTP * NV_ * sizeof(float));
        }
        if (hasGamaK_) {
            pipe_->InitBuffer(gamaKInBuf_, MAX_MTP * alignK_ * sizeof(float));
        }
        pipe_->InitBuffer(betaInBuf_, MAX_MTP * NV_ * sizeof(inType));
        pipe_->InitBuffer(stateOutBuf_, alignK_ * vStep_ * sizeof(outType));
        pipe_->InitBuffer(attnOutBuf_, vStep_ * sizeof(outType));
        pipe_->InitBuffer(tmpBuff, restUbSize_);
        uint32_t buffOffset = 0;
        deltaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        attnInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        vInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignV_), buffOffset);
        buffOffset += vSize;
        qInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        kInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize + REPEAT_BYTES;
        stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize + 128;
        broadTmpInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbSize), buffOffset);
        buffOffset += betaUbSize;
        uint32_t halfK_ = alignK_ >> 1;
        foldTmpUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(halfK_ * vStep_), buffOffset);
    }

    __aicore__ inline void ComputeAvgload()
    {
        uint64_t realT = 0;
        for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
            realT += cuSeqlensGm_.GetValue(batch_i);
        }
        avgload = Ceil(realT * NV_, GetBlockNum());
    }

    __aicore__ inline void Process()
    {
        ComputeAvgload();
        int32_t seq1 = 0;
        for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
            int32_t seqLen = cuSeqlensGm_.GetValue(batch_i);
            if (seqLen <= 0) {
                continue;
            }
            if (seqLen > static_cast<int32_t>(MAX_MTP)) {
                return;
            }
            if (seq1 < 0 || seq1 > static_cast<int32_t>(T_) || (seq1 + seqLen) > static_cast<int32_t>(T_)) {
                return;
            }
            int32_t seq0 = seq1;
            seq1 += seqLen;
            uint32_t copyFlag = 0;
            uint64_t stateOffset;
            for (uint64_t head_i = 0; head_i < NV_; head_i++) {
                if (!IsCurrentBlock(seq1 - seq0)) {
                    continue;
                }
                copyFlag++;
                if (copyFlag == 1) {
                    int32_t stateTokenIdx = seq0;
                    if (hasAcceptedTokens_) {
                        int32_t acceptedTokenNum = numAcceptedTokensGm_.GetValue(batch_i);
                        if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
                            return;
                        }
                        stateTokenIdx = seq0 + acceptedTokenNum - 1;
                    }
                    stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx);
                    CopyInGamaBeta(seq0, seq1);
                }
                ProcessHead(seq0, seq1, head_i, stateOffset);
            }
        }
    }

private:
    __aicore__ inline void CopyInQKV(uint64_t vOffset, uint64_t qkOffset, int32_t seqLen)
    {
        LocalTensor<inType> qLocal = qInBuf_.Get<inType>();
        LocalTensor<inType> kLocal = kInBuf_.Get<inType>();
        LocalTensor<inType> vLocal = vInBuf_.Get<inType>();
        DataCopyExtParams qkInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                     static_cast<uint32_t>((NK_ - 1) * realK_ * sizeof(inType)), 0, 0};
        DataCopyExtParams vInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                    static_cast<uint32_t>((NV_ - 1) * realV_ * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> qkPadParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPadExtParams<inType> vPadParams{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};
        if (hasGamaK_) {
            uint32_t alignKGamma = Ceil(realK_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
            uint32_t stride = alignKGamma < alignK_ ? 1 : 0;
            DataCopyExtParams gkInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(float)),
                                     static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(float)), stride, 0};
            DataCopyPadExtParams<float> gkPadParams{true, 0, static_cast<uint8_t>(alignKGamma - realK_), 0};
            gamaKInUb = gamaKInBuf_.Get<float>();
            Duplicate<float>(gamaKInUb, 0, alignK_ * seqLen);
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            DataCopyPadCustom(gamaKInUb, gamaKGm_[vOffset / realV_ * realK_], gkInParams, gkPadParams);
#else
            DataCopyPad(gamaKInUb, gamaKGm_[vOffset / realV_ * realK_], gkInParams, gkPadParams);
#endif
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Exp(gamaKInUb, gamaKInUb, alignK_ * seqLen);
            AscendC::PipeBarrier<PIPE_V>();
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        DataCopyPadCustom(qLocal, queryGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPadCustom(kLocal, keyGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPadCustom(vLocal, valueGm_[vOffset], vInParams, vPadParams);
#else
        DataCopyPad(qLocal, queryGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPad(kLocal, keyGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPad(vLocal, valueGm_[vOffset], vInParams, vPadParams);
#endif
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        CastOrCopy(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        CastOrCopy(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        CastOrCopy(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(qInUb, qInUb, scale_, seqLen * alignK_);
    }

    __aicore__ inline void PrefetchState(uint64_t stateOffest, uint32_t curSingleV)
    {
        LocalTensor<inType> stateLocal = stateInBuf_.Get<inType>();
        DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV),
                                        static_cast<uint16_t>(realK_ * sizeof(inType)), 0, 0, 0};
        DataCopyPadExtParams<inType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        DataCopyPadCustom(stateLocal, initStateGm_[stateOffest], stateInParams, padParams);
#else
        DataCopyPad(stateLocal, initStateGm_[stateOffest], stateInParams, padParams);
#endif
    }

    __aicore__ inline void LoadPrefetchedState(uint32_t curSingleV)
    {
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        LocalTensor<inType> stateLocal = stateInBuf_.Get<inType>();
        CastOrCopy(stateInUb, stateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
    }

    __aicore__ inline void MatVecMul(const LocalTensor<float> &cubeTensor, const LocalTensor<float> &vecTensor,
                                          LocalTensor<float> &dstTensor, uint32_t cols, bool isAdd)
    {
        uint8_t rowStride = alignK_ / FP32_NUM_PER_BLOCK;
        for (uint32_t i = 0; i < alignK_; i += REPEAT_LENTH) {
            uint64_t mask = Std::min(REPEAT_LENTH, alignK_ - i);
            for (uint32_t j = 0; j < cols; j += MAX_REPEAT_TIME) {
                uint64_t repeatTime = Std::min(MAX_REPEAT_TIME, cols - j);
                uint32_t off = j * alignK_ + i;
                if (isAdd) {
                    MulAddDst(dstTensor[off], cubeTensor[off], vecTensor[i],
                              mask, repeatTime, {1, 1, 1, rowStride, rowStride, 0});
                } else {
                    Mul(dstTensor[off], cubeTensor[off], vecTensor[i],
                        mask, repeatTime, {1, 1, 1, rowStride, rowStride, 0});
                }
            }
        }
    }

    __aicore__ inline void ReduceSumDispatch(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                             uint32_t rows)
    {
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 200)
        uint32_t stateShape[2] = {rows, alignK_};
        ReduceSum<float, Pattern::Reduce::AR, true>(dstTensor, srcTensor, stateShape, true);
        return;
#else
        uint32_t curK = alignK_;
        bool readFromSrc = true;

        while (curK > REPEAT_LENTH) {
            if (!readFromSrc) AscendC::PipeBarrier<PIPE_V>();
            uint32_t half = curK >> 1;
            uint8_t sStride = curK / FP32_NUM_PER_BLOCK;
            uint8_t dStride = half / FP32_NUM_PER_BLOCK;
            for (uint32_t j = 0; j < rows; j += MAX_REPEAT_TIME) {
                uint32_t batch = Std::min(static_cast<uint32_t>(MAX_REPEAT_TIME), rows - j);
                if (readFromSrc) {
                    Add(foldTmpUb[j * half], srcTensor[j * alignK_], srcTensor[j * alignK_ + half],
                        half, batch, {1, 1, 1, dStride, sStride, sStride});
                } else {
                    Add(srcTensor[j * half], foldTmpUb[j * curK], foldTmpUb[j * curK + half],
                        half, batch, {1, 1, 1, dStride, sStride, sStride});
                }
            }
            curK = half;
            readFromSrc = !readFromSrc;
        }

        AscendC::PipeBarrier<PIPE_V>();
        uint8_t foldStride = curK / FP32_NUM_PER_BLOCK;
        for (uint32_t j = 0; j < rows; j += MAX_REPEAT_TIME) {
            uint32_t batch = Std::min(static_cast<uint32_t>(MAX_REPEAT_TIME), rows - j);
            if (readFromSrc) {
                WholeReduceSum(dstTensor[j], srcTensor[j * alignK_],
                               REPEAT_LENTH, batch, 1, 1, foldStride);
            } else {
                WholeReduceSum(dstTensor[j], foldTmpUb[j * curK],
                               REPEAT_LENTH, batch, 1, 1, foldStride);
            }
        }
#endif
    }

    __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset, uint64_t curVOffset)
    {
        uint32_t stateShape[2] = {curSingleV, alignK_};
        uint32_t ktShape[2] = {1, alignK_};
        uint32_t deltaShape[2] = {curSingleV, 1};
        if (hasGama_) {
            Muls(broadTmpInUb, stateInUb, gama_, alignK_ * curSingleV);
            SwapTensor(stateInUb, broadTmpInUb);
        }
        if (hasGamaK_) {
            MatVecMul(stateInUb, gamaKInUb[curQKOffset], stateInUb, curSingleV, false);
        }
        if (hasGama_ || hasGamaK_) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        MatVecMul(stateInUb, kInUb[curQKOffset], broadTmpInUb, curSingleV, false);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumDispatch(deltaInUb, broadTmpInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(deltaInUb, vInUb[curVOffset], deltaInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(deltaInUb, deltaInUb, beta_, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Broadcast<float, 2, 1>(broadTmpInUb, deltaInUb, stateShape, deltaShape); //  2: Dim Number 1: Second Dim
        AscendC::PipeBarrier<PIPE_V>();
        MatVecMul(broadTmpInUb, kInUb[curQKOffset], stateInUb, curSingleV, true);
        AscendC::PipeBarrier<PIPE_V>();
        MatVecMul(stateInUb, qInUb[curQKOffset], broadTmpInUb, curSingleV, false);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumDispatch(attnInUb, broadTmpInUb, curSingleV);
        LocalTensor<outType> stateOutLocal = stateOutBuf_.Get<outType>();
        LocalTensor<outType> attnOutLocal = attnOutBuf_.Get<outType>();
        AscendC::PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(evtMte3V_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        CastOrCopy(stateOutLocal, stateInUb, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
        CastOrCopy(attnOutLocal, attnInUb, AscendC::RoundMode::CAST_NONE, curSingleV);
#else
        CastOrCopy(stateOutLocal, stateInUb, AscendC::RoundMode::CAST_RINT, alignK_ * curSingleV);
        CastOrCopy(attnOutLocal, attnInUb, AscendC::RoundMode::CAST_RINT, curSingleV);
#endif
        SetFlag<HardEvent::V_MTE3>(evtVMte3_);
    }

    __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t curSingleV)
    {
        LocalTensor<outType> attnLocal = attnOutBuf_.Get<outType>();
        WaitFlag<HardEvent::V_MTE3>(evtVMte3_);
        DataCopyParams attnOutParams{1, static_cast<uint16_t>(curSingleV * sizeof(outType)), 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        DataCopyCustom(attnOutGm_[attnOffset], attnLocal, attnOutParams);
#else
        DataCopyPad(attnOutGm_[attnOffset], attnLocal, attnOutParams);
#endif
    }

    __aicore__ inline void CopyOutState(uint64_t stateOffset, uint32_t curSingleV)
    {
        LocalTensor<outType> stateOutLocal = stateOutBuf_.Get<outType>();
        DataCopyParams stateOutParams{static_cast<uint16_t>(curSingleV),
                                      static_cast<uint16_t>(realK_ * sizeof(outType)), 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        DataCopyCustom(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
#else
        DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
#endif
        SetFlag<HardEvent::MTE3_V>(evtMte3V_);
    }

    __aicore__ inline void CopyInGamaBeta(int32_t seq0, int32_t seq1)
    {
        int32_t seqLen = seq1 - seq0;
        uint64_t bBatchSize = Ceil(seqLen * NV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        LocalTensor<inType> betaLocal = betaInBuf_.Get<inType>();
        DataCopyParams betaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(inType)), 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        DataCopyCustom(betaLocal, betaGm_[seq0 * NV_], betaInParams);
#else
        DataCopyPadParams padParams;
        DataCopyPad(betaLocal, betaGm_[seq0 * NV_], betaInParams, padParams);
#endif
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        CastOrCopy(betaInUb, betaLocal, AscendC::RoundMode::CAST_NONE, bBatchSize);
        if (hasGama_) {
            gamaInUb = gamaInBuf_.Get<float>();
            DataCopyParams gamaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(float)), 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            DataCopyCustom(gamaInUb, gamaGm_[seq0 * NV_], gamaInParams);
#else
            DataCopyPad(gamaInUb, gamaGm_[seq0 * NV_], gamaInParams, padParams);
#endif
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Exp(gamaInUb, gamaInUb, seqLen * NV_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t stateOffset)
    {
        uint64_t vOffset = (seq0 * NV_ + head_i) * realV_;
        uint64_t qkOffset = (seq0 * NK_ + head_i / (NV_ / NK_)) * realK_;
        CopyInQKV(vOffset, qkOffset, seq1 - seq0);
        if (realV_ == 0) {
            return;
        }
        uint64_t nextVOffset = 0;
        uint32_t nextSingleV = realV_ > vStep_ ? vStep_ : realV_;
        uint64_t nextStateOffset = ((stateOffset * NV_ + head_i) * realV_) * realK_;
        PrefetchState(nextStateOffset, nextSingleV);
        for (uint64_t v_i = 0; v_i < realV_; v_i += vStep_) {
            uint32_t curSingleV = v_i + vStep_ > realV_ ? realV_ - v_i : vStep_;
            LoadPrefetchedState(curSingleV);
            nextVOffset = v_i + vStep_;
            if (nextVOffset < realV_) {
                nextSingleV = nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
                nextStateOffset = ((stateOffset * NV_ + head_i) * realV_ + nextVOffset) * realK_;
                PrefetchState(nextStateOffset, nextSingleV);
            }
            evtMte3V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            evtVMte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::MTE3_V>(evtMte3V_);
            for (uint64_t seq_i = seq0; seq_i < seq1; seq_i++) {
                uint64_t gbOffset = head_i + (seq_i - seq0) * NV_;
                uint64_t curQKOffset = (seq_i - seq0) * alignK_;
                uint64_t curVOffset = (seq_i - seq0) * alignV_ + v_i;
                uint64_t attnOffset = (seq_i * NV_ + head_i) * realV_ + v_i;
                uint64_t curStateOutOffset =
                    ((ssmStateIndicesGm_.GetValue(seq_i) * NV_ + head_i) * realV_ + v_i) * realK_;
                gama_ = hasGama_ ? gamaInUb.GetValue(gbOffset) : 1;
                beta_ = betaInUb.GetValue(gbOffset);
                Compute(curSingleV, curQKOffset, curVOffset);
                CopyOutAttn(attnOffset, curSingleV);
                CopyOutState(curStateOutOffset, curSingleV);
            }
            WaitFlag<HardEvent::MTE3_V>(evtMte3V_);
        }
    }

    __aicore__ inline bool IsCurrentBlock(int32_t seqlen)
    {
        load += seqlen;
        bool ret = (blockIdx == usedblk && seqlen > 0);
        if (load >= avgload) {
            load = 0;
            usedblk++;
        }
        return ret;
    }

private:
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<inType> betaGm_;
    GlobalTensor<float> gamaGm_;
    GlobalTensor<float> gamaKGm_;
    GlobalTensor<inType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> ssmStateIndicesGm_;
    GlobalTensor<int32_t> numAcceptedTokensGm_;
    GlobalTensor<outType> finalStateGm_;
    GlobalTensor<outType> attnOutGm_;
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> qInBuf_;
    TBuf<TPosition::VECCALC> kInBuf_;
    TBuf<TPosition::VECCALC> vInBuf_;
    TBuf<TPosition::VECCALC> gamaInBuf_;
    TBuf<TPosition::VECCALC> gamaKInBuf_;
    TBuf<TPosition::VECCALC> betaInBuf_;
    TBuf<TPosition::VECCALC> stateInBuf_;
    TBuf<TPosition::VECCALC> attnOutBuf_;
    TBuf<TPosition::VECCALC> stateOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<float> qInUb;
    LocalTensor<float> kInUb;
    LocalTensor<float> vInUb;
    LocalTensor<float> gamaInUb;
    LocalTensor<float> gamaKInUb;
    LocalTensor<float> betaInUb;
    LocalTensor<float> deltaInUb;
    LocalTensor<float> broadTmpInUb;
    LocalTensor<float> attnInUb;
    LocalTensor<float> stateInUb;
    LocalTensor<float> foldTmpUb;
    uint32_t B_;
    uint32_t T_;
    uint32_t NK_;
    uint32_t alignK_;
    uint32_t realK_;
    uint32_t NV_;
    uint32_t alignV_;
    uint32_t realV_;
    uint32_t vStep_;
    uint32_t restUbSize_;
    uint32_t load;
    uint32_t usedblk;
    uint32_t avgload;
    bool hasAcceptedTokens_;
    bool hasGama_;
    bool hasGamaK_;
    bool useAddFoldReduce_;
    float gama_;
    float beta_;
    float scale_;
    uint64_t blockIdx;
    event_t evtMte3V_;
    event_t evtVMte3_;
};
} // namespace RecurrentGatedDeltaRule
#endif
