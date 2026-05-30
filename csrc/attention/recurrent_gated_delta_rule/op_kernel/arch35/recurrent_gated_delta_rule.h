/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule.h
 * \brief
 */

#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../recurrent_gated_delta_rule_tiling_data.h"

namespace RecurrentGatedDeltaRule {

using namespace matmul;
using namespace AscendC;
using namespace AscendC::MicroAPI;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64; // 256Byte for float
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t ADD_FOLD_REDUCE_MIN_K = 128;
constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint16_t TWO_V_LENGTH = 2 * V_LENGTH;

constexpr CastTrait castTraitB16ToB32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

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

template <typename inType, typename outType, typename stateType>
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
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
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
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.initState);
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
        ssmStateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        numAcceptedTokensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.finalState);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
        uint32_t singleVSize = vStep_ * sizeof(float);
        uint32_t vSize = MAX_MTP * alignV_ * sizeof(float);
        uint32_t kSize = MAX_MTP * alignK_ * sizeof(float);
        uint32_t betaNumAlign = Ceil(MAX_MTP * NV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        uint32_t betaUbSize = betaNumAlign * sizeof(float); //  8: 8 * 4 = 32B;
        pipe_->InitBuffer(qInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(kInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(vInQueue_, BUFFER_NUM, MAX_MTP * alignV_ * sizeof(inType));
        pipe_->InitBuffer(stateInQueue_, BUFFER_NUM, alignK_ * vStep_ * sizeof(stateType));
        if (hasGama_) {
            pipe_->InitBuffer(gamaInQueue_, BUFFER_NUM, MAX_MTP * NV_ * sizeof(float));
        }
        if (hasGamaK_) {
            pipe_->InitBuffer(gamaKInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(float));
        }
        pipe_->InitBuffer(betaInQueue_, BUFFER_NUM, MAX_MTP * NV_ * sizeof(inType));
        pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, alignK_ * vStep_ * sizeof(stateType));
        pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));
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
        buffOffset += kSize;
        stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        broadTmpInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaNumAlign), buffOffset);
        buffOffset += betaUbSize;
        gamaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaNumAlign), buffOffset);
    }

    __aicore__ inline void ComputeAvgload()
    {
        uint64_t realT = 0;
        for (uint64_t batch_i = 1; batch_i < B_ + 1; batch_i++) {
            realT += cuSeqlensGm_.GetValue(batch_i);
        }
        avgload = Ceil(realT * NV_, GetBlockNum());
    }

    __aicore__ inline void Process()
    {
        ComputeAvgload();
        int32_t seq1 = cuSeqlensGm_.GetValue(0);
        for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
            int32_t seqLen = cuSeqlensGm_.GetValue(batch_i+1);
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
        LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
        LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
        LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();
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
            LocalTensor<float> gamaKLocal = gamaKInQueue_.AllocTensor<float>();
            Duplicate<float>(gamaKLocal, 0, alignK_ * seqLen);
            TEventID evevtIdVtoMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(evevtIdVtoMte2);
            WaitFlag<HardEvent::V_MTE2>(evevtIdVtoMte2);
            DataCopyPad(gamaKLocal, gamaKGm_[vOffset / realV_ * realK_], gkInParams, gkPadParams);
            gamaKInQueue_.EnQue<float>(gamaKLocal);
            gamaKInUb = gamaKInQueue_.DeQue<float>();
            Exp(gamaKInUb, gamaKInUb, alignK_ * seqLen);
            AscendC::PipeBarrier<PIPE_V>();
        }
        DataCopyPad(qLocal, queryGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPad(kLocal, keyGm_[qkOffset], qkInParams, qkPadParams);
        DataCopyPad(vLocal, valueGm_[vOffset], vInParams, vPadParams);
        qInQueue_.EnQue<inType>(qLocal);
        kInQueue_.EnQue<inType>(kLocal);
        vInQueue_.EnQue<inType>(vLocal);
        qLocal = qInQueue_.DeQue<inType>();
        kLocal = kInQueue_.DeQue<inType>();
        vLocal = vInQueue_.DeQue<inType>();
        Cast(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(qInUb, qInUb, scale_, seqLen * alignK_);
        qInQueue_.FreeTensor(qLocal);
        kInQueue_.FreeTensor(kLocal);
        vInQueue_.FreeTensor(vLocal);
    }

    __aicore__ inline void PrefetchState(uint64_t stateOffest, uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.AllocTensor<stateType>();
        DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV),
                                        static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0, 0};
        DataCopyPadExtParams<stateType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPad(stateLocal, initStateGm_[stateOffest], stateInParams, padParams);
        stateInQueue_.EnQue<stateType>(stateLocal);
    }

    __aicore__ inline void LoadPrefetchedState(uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.DeQue<stateType>();
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(stateInUb, stateLocal, alignK_ * curSingleV);
        } else {
            Cast(stateInUb, stateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
        }
        stateInQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void MatVecMul(const LocalTensor<float> &cubeTensor, const LocalTensor<float> &vecTensor,
                                          LocalTensor<float> &dstTensor, uint32_t rows)
    {
        __ubuf__ float* cubeAddr = (__ubuf__ float*)cubeTensor.GetPhyAddr();
        __ubuf__ float* vecAddr = (__ubuf__ float*)vecTensor.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dstTensor.GetPhyAddr();

        uint16_t rowNum = static_cast<uint16_t>(rows);
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(alignK_, V_LENGTH));
        uint32_t colLength = alignK_;
        __VEC_SCOPE__
        {
            RegTensor<float> cube;
            RegTensor<float> vec;
            RegTensor<float> dst;
            MaskReg pregLoop;
            for (uint16_t j = 0; j < colLoopTimes; j++) {
                pregLoop = UpdateMask<float>(colLength);
                DataCopy(vec, vecAddr + j * V_LENGTH);
                for (uint16_t i = 0; i < rowNum; i ++) {
                    DataCopy(cube, cubeAddr + i * alignK_ + j * V_LENGTH);
                    Mul(dst, cube, vec, pregLoop);
                    DataCopy(dstAddr + i * alignK_ + j * V_LENGTH, dst, pregLoop);
                }
            }
        }
    }

    __aicore__ inline void ProcessKQ(const LocalTensor<float> &cubeTensor, const LocalTensor<float> &vec1Tensor,
                                          LocalTensor<float> &dst1Tensor, const LocalTensor<float> &vec2Tensor,
                                          LocalTensor<float> &dst2Tensor, uint32_t rows)
    {
        __ubuf__ float* cubeAddr = (__ubuf__ float*)cubeTensor.GetPhyAddr();
        __ubuf__ float* vec1Addr = (__ubuf__ float*)vec1Tensor.GetPhyAddr();
        __ubuf__ float* vec2Addr = (__ubuf__ float*)vec2Tensor.GetPhyAddr();
        __ubuf__ float* dst1Addr = (__ubuf__ float*)dst1Tensor.GetPhyAddr();
        __ubuf__ float* dst2Addr = (__ubuf__ float*)dst2Tensor.GetPhyAddr();

        uint16_t rowNum = static_cast<uint16_t>(rows);
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(alignK_, V_LENGTH));
        uint32_t colLength = alignK_;
        __VEC_SCOPE__
        {
            RegTensor<float> cube;
            RegTensor<float> vec1;
            RegTensor<float> vec2;
            RegTensor<float> dst1;
            RegTensor<float> dst2;
            MaskReg pregLoop;
            for (uint16_t j = 0; j < colLoopTimes; j++) {
                pregLoop = UpdateMask<float>(colLength);
                DataCopy(vec1, vec1Addr + j * V_LENGTH);
                DataCopy(vec2, vec2Addr + j * V_LENGTH);
                for (uint16_t i = 0; i < rowNum; i ++) {
                    DataCopy<float, LoadDist::DIST_BRC_B32>(cube, cubeAddr + i);
                    DataCopy(dst1, dst1Addr + i * alignK_ + j * V_LENGTH);
                    Mul(cube, cube, vec1, pregLoop);
                    Add(dst1, dst1, cube, pregLoop);
                    Mul(dst2, dst1, vec2, pregLoop);
                    DataCopy(dst1Addr + i * alignK_ + j * V_LENGTH, dst1, pregLoop);
                    DataCopy(dst2Addr + i * alignK_ + j * V_LENGTH, dst2, pregLoop);
                }
            }
        }
    }

    __aicore__ inline void ReduceSum64(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, uint16_t rowNum)
    {
        uint32_t colLength = alignK_;
        __VEC_SCOPE__
        {
            RegTensor<float> src;
            RegTensor<float> sum;
            MaskReg pregLoop = UpdateMask<float>(colLength);
            for (uint16_t i = 0;i < rowNum;i ++) {
                DataCopy(src, srcAddr + i * alignK_);
                ReduceSum(sum, src, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sum, pregLoop);
            }
        }
    }

    __aicore__ inline void ReduceSum128(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, uint16_t rowNum)
    {
        uint32_t colLength = alignK_ - V_LENGTH;
        __VEC_SCOPE__
        {
            RegTensor<float> src1;
            RegTensor<float> src2;
            RegTensor<float> sum;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop = UpdateMask<float>(colLength);
            for (uint16_t i = 0;i < rowNum;i ++) {
                DataCopy(src1, srcAddr + i * alignK_);
                DataCopy(src2, srcAddr + i * alignK_ + V_LENGTH);
                Add<float, MaskMergeMode::MERGING>(src1, src1, src2, pregLoop);
                ReduceSum(sum, src1, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sum, pregFull);
            }
        }
    }

    __aicore__ inline void ReduceSumVF(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, uint16_t rowNum)
    {
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(alignK_, V_LENGTH));
        __VEC_SCOPE__
        {
            RegTensor<float> src;
            RegTensor<float> tmp;
            RegTensor<float> sum;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            for (uint16_t i = 0;i < rowNum;i ++) {
                uint32_t colLength = alignK_;
                Duplicate(tmp, 0.0f);
                for (uint16_t j = 0; j < colLoopTimes; j++) {
                    pregLoop = UpdateMask<float>(colLength);
                    DataCopy(src, srcAddr + i * alignK_ + j * V_LENGTH);
                    Add<float, MaskMergeMode::MERGING>(tmp, tmp, src, pregLoop);
                }
                ReduceSum(sum, tmp, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sum, pregFull);
            }
        }
    }

    __aicore__ inline void ReduceSumDispatch(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                             uint32_t rows)
    {
        __ubuf__ float* srcAddr = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dstTensor.GetPhyAddr();
        uint16_t rowNum = static_cast<uint16_t>(rows);
        if (alignK_ <= V_LENGTH) {
            ReduceSum64(dstAddr, srcAddr, rowNum);
        } else if (alignK_ <= TWO_V_LENGTH) {
            ReduceSum128(dstAddr, srcAddr, rowNum);
        } else {
            ReduceSumVF(dstAddr, srcAddr, rowNum);
        }
    }

    __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset, uint64_t curVOffset)
    {
        if (hasGama_) {
            Muls(stateInUb, stateInUb, gama_, alignK_ * curSingleV);
        }
        if (hasGamaK_) {
            MatVecMul(stateInUb, gamaKInUb[curQKOffset], stateInUb, curSingleV);
        }
        if (hasGama_ || hasGamaK_) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        MatVecMul(stateInUb, kInUb[curQKOffset], broadTmpInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumDispatch(deltaInUb, broadTmpInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(deltaInUb, vInUb[curVOffset], deltaInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(deltaInUb, deltaInUb, beta_, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        ProcessKQ(deltaInUb, kInUb[curQKOffset], stateInUb, qInUb[curQKOffset], broadTmpInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumDispatch(attnInUb, broadTmpInUb, curSingleV);
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.AllocTensor<stateType>();
        LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
        if constexpr (std::is_same<stateType, float32_t>()) {
            DataCopy(stateOutLocal, stateInUb, alignK_ * curSingleV);
        } else {
            Cast(stateOutLocal, stateInUb, AscendC::RoundMode::CAST_RINT, alignK_ * curSingleV);
        }
        stateOutQueue_.EnQue<stateType>(stateOutLocal);
        Cast(attnOutLocal, attnInUb, AscendC::RoundMode::CAST_RINT, curSingleV);
        attnOutQueue_.EnQue<outType>(attnOutLocal);
    }

    __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t curSingleV)
    {
        LocalTensor<outType> attnLocal = attnOutQueue_.DeQue<outType>();
        DataCopyParams attnOutParams{1, static_cast<uint16_t>(curSingleV * sizeof(outType)), 0, 0};
        DataCopyPad(attnOutGm_[attnOffset], attnLocal, attnOutParams);
        attnOutQueue_.FreeTensor(attnLocal);
    }

    __aicore__ inline void CopyOutState(uint64_t stateOffset, uint32_t curSingleV)
    {
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.DeQue<stateType>();
        DataCopyParams stateOutParams{static_cast<uint16_t>(curSingleV),
                                      static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0};
        DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
        stateOutQueue_.FreeTensor(stateOutLocal);
    }

    __aicore__ inline void CopyInGamaBeta(int32_t seq0, int32_t seq1)
    {
        int32_t seqLen = seq1 - seq0;
        LocalTensor<inType> betaLocal = betaInQueue_.AllocTensor<inType>();
        DataCopyParams betaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(inType)), 0, 0};
        DataCopyPadParams padParams;
        DataCopyPad(betaLocal, betaGm_[seq0 * NV_], betaInParams, padParams);
        betaInQueue_.EnQue<inType>(betaLocal);
        betaLocal = betaInQueue_.DeQue<inType>();
        Cast(betaInUb, betaLocal, AscendC::RoundMode::CAST_NONE, seqLen * NV_);
        betaInQueue_.FreeTensor(betaLocal);
        if (hasGama_) {
            LocalTensor<float> gamaLocal = gamaInQueue_.AllocTensor<float>();
            DataCopyParams gamaInParams{1, static_cast<uint16_t>(seqLen * NV_ * sizeof(float)), 0, 0};
            DataCopyPad(gamaLocal, gamaGm_[seq0 * NV_], gamaInParams, padParams);
            gamaInQueue_.EnQue<float>(gamaLocal);
            gamaLocal = gamaInQueue_.DeQue<float>();
            Exp(gamaInUb, gamaLocal, seqLen * NV_);
            gamaInQueue_.FreeTensor(gamaLocal);
        }
    }

    __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t stateOffset)
    {
        uint64_t vOffset = (seq0 * NV_ + head_i) * realV_;
        uint64_t qkOffset = (seq0 * NK_ + head_i / (NV_ / NK_)) * realK_;
        CopyInQKV(vOffset, qkOffset, seq1 - seq0);
        if (realV_ == 0) {
            if (hasGamaK_) {
                gamaKInQueue_.FreeTensor(gamaKInUb);
            }
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
            uint64_t pendingAttnOffset = 0;
            uint64_t pendingStateOffset = 0;
            bool hasPendingAttn = false;
            bool hasPendingState = false;
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
                if (attnOutBufferNum_ == BUFFER_NUM) {
                    CopyOutAttn(attnOffset, curSingleV);
                } else {
                    if (hasPendingAttn) {
                        CopyOutAttn(pendingAttnOffset, curSingleV);
                    }
                    pendingAttnOffset = attnOffset;
                    hasPendingAttn = true;
                }
                if (stateOutBufferNum_ == BUFFER_NUM) {
                    CopyOutState(curStateOutOffset, curSingleV);
                } else {
                    if (hasPendingState) {
                        CopyOutState(pendingStateOffset, curSingleV);
                    }
                    pendingStateOffset = curStateOutOffset;
                    hasPendingState = true;
                }
            }
            if (hasPendingAttn) {
                CopyOutAttn(pendingAttnOffset, curSingleV);
            }
            if (hasPendingState) {
                CopyOutState(pendingStateOffset, curSingleV);
            }
        }
        if (hasGamaK_) {
            gamaKInQueue_.FreeTensor(gamaKInUb);
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
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> ssmStateIndicesGm_;
    GlobalTensor<int32_t> numAcceptedTokensGm_;
    GlobalTensor<stateType> finalStateGm_;
    GlobalTensor<outType> attnOutGm_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> qInQueue_;
    TQue<QuePosition::VECIN, 1> kInQueue_;
    TQue<QuePosition::VECIN, 1> vInQueue_;
    TQue<QuePosition::VECIN, 1> gamaInQueue_;
    TQue<QuePosition::VECIN, 1> gamaKInQueue_;
    TQue<QuePosition::VECIN, 1> betaInQueue_;
    TQue<QuePosition::VECIN, 1> stateInQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> attnOutQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> stateOutQueue_;
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
    uint32_t B_;
    uint32_t T_;
    uint32_t NK_;
    uint32_t alignK_;
    uint32_t realK_;
    uint32_t NV_;
    uint32_t alignV_;
    uint32_t realV_;
    uint32_t vStep_;
    uint32_t stateOutBufferNum_;
    uint32_t attnOutBufferNum_;
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
};
} // namespace RecurrentGatedDeltaRule
#endif
