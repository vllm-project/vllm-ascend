/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file recurrent_kda.h
 * \brief Single-kernel fused recurrent KDA implementation.
 */

#ifndef __RECURRENT_KDA_KERNEL_H_
#define __RECURRENT_KDA_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../recurrent_kda_tiling_data.h"

namespace RecurrentKda {

using namespace matmul;
using namespace AscendC;
using namespace AscendC::MicroAPI;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64; // 256 bytes for float.
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t ADD_FOLD_REDUCE_MIN_K = 128;
constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint16_t TWO_V_LENGTH = 2 * V_LENGTH;
constexpr uint64_t INVALID_STATE_SLOT = static_cast<uint64_t>(-1);

#ifndef RKDA_ENABLE_ADD_FOLD_REDUCE
#define RKDA_ENABLE_ADD_FOLD_REDUCE  1
#endif

struct RKDAInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename inType, typename outType, typename stateType>
class RKDA {
public:
    __aicore__ inline explicit RKDA(const RecurrentKdaTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        seqLen_ = tilingData->seqLen;
        NK_ = tilingData->nk;
        realK_ = tilingData->dk;
        NV_ = tilingData->nv;
        realV_ = tilingData->dv;
        stateCapacity_ = tilingData->sBlockNum;
        ssmStateStride_ = tilingData->ssmStateStride;
        stateInStride0_ = tilingData->stateInStride0;
        stateInStride1_ = tilingData->stateInStride1;
        stateInStride2_ = tilingData->stateInStride2;
        stateInStride3_ = tilingData->stateInStride3;
        stateOutStride0_ = tilingData->stateOutStride0;
        stateOutStride1_ = tilingData->stateOutStride1;
        stateOutStride2_ = tilingData->stateOutStride2;
        stateOutStride3_ = tilingData->stateOutStride3;
        queryTokenStride_ = tilingData->queryTokenStride;
        queryHeadStride_ = tilingData->queryHeadStride;
        keyTokenStride_ = tilingData->keyTokenStride;
        keyHeadStride_ = tilingData->keyHeadStride;
        valueTokenStride_ = tilingData->valueTokenStride;
        valueHeadStride_ = tilingData->valueHeadStride;
        scale_ = tilingData->scale;
        lowerBound_ = tilingData->lowerBound;
        hasCuSeqlens_ = (tilingData->hasCuSeqlens == 1);
        hasSsmStateIndices_ = (tilingData->hasSsmStateIndices == 1);
        hasAcceptedTokens_ = (tilingData->hasAcceptedTokens == 1);
        hasALog_ = (tilingData->hasALog == 1);
        hasDtBias_ = (tilingData->hasDtBias == 1);
        useQkL2norm_ = (tilingData->useQkL2norm == 1);
        useGateInKernel_ = (tilingData->useGateInKernel == 1);
        useBetaSigmoid_ = (tilingData->useBetaSigmoid == 1);
        allowNegEigval_ = (tilingData->allowNegEigval == 1);
        safeGate_ = (tilingData->safeGate == 1);
        stateVFirst_ = (tilingData->stateVFirst == 1);
        shouldStoreState_ = (tilingData->inplaceFinalState == 1 || tilingData->outputFinalState == 1);
        gateDtype_ = tilingData->gateDtype;
        betaDtype_ = tilingData->betaDtype;
        cuSeqlensDtype_ = tilingData->cuSeqlensDtype;
        ssmStateIndicesDtype_ = tilingData->ssmStateIndicesDtype;
        acceptedTokensDtype_ = tilingData->acceptedTokensDtype;
        useAddFoldReduce_ = (RKDA_ENABLE_ADD_FOLD_REDUCE != 0);
        vStep_ = tilingData->vStep;
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        restUbSize_ = tilingData->ubRestBytes;
        alignK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        eventMte2ToVInitialized_ = false;
        eventVToMte2Initialized_ = false;
        eventVToSInitialized_ = false;
    }

    __aicore__ inline void Init(const RKDAInitParams &initParams, TPipe *pipe)
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

    __aicore__ inline void SetGlobalTensors(const RKDAInitParams &initParams)
    {
        queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value);
        gateFloatGm_.SetGlobalBuffer((__gm__ float *)initParams.gate);
        gateBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)initParams.gate);
        gateFp16Gm_.SetGlobalBuffer((__gm__ half *)initParams.gate);
        betaFloatGm_.SetGlobalBuffer((__gm__ float *)initParams.beta);
        betaBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)initParams.beta);
        betaFp16Gm_.SetGlobalBuffer((__gm__ half *)initParams.beta);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.initState);
        cuSeqlensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
        cuSeqlensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.cuSeqlens);
        ssmStateIndicesInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        ssmStateIndicesInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.ssmStateIndices);
        aLogGm_.SetGlobalBuffer((__gm__ float *)initParams.aLog);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)initParams.dtBias);
        numAcceptedTokensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
        numAcceptedTokensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.numAcceptedTokens);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.finalState);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
        uint32_t singleVSize = vStep_ * sizeof(float);
        uint32_t vSize = MAX_MTP * alignV_ * sizeof(float);
        uint32_t kSize = MAX_MTP * alignK_ * sizeof(float);
        uint32_t betaUbSize =
            Ceil(MAX_MTP * NV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK * sizeof(float);
        pipe_->InitBuffer(qInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(kInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(vInQueue_, BUFFER_NUM, MAX_MTP * alignV_ * sizeof(inType));
        pipe_->InitBuffer(gateInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(float));
        pipe_->InitBuffer(betaInQueue_, BUFFER_NUM, betaUbSize);
        pipe_->InitBuffer(stateInQueue_, BUFFER_NUM, alignK_ * vStep_ * sizeof(stateType));
        pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, alignK_ * vStep_ * sizeof(stateType));
        pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));
        pipe_->InitBuffer(tmpBuff, restUbSize_);
        pipe_->InitBuffer(scalarBuf_, 64);

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
        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbSize / sizeof(float)), buffOffset);
        buffOffset += betaUbSize;
        gateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        if (!eventMte2ToVInitialized_) {
            eventIdMte2ToV_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            eventMte2ToVInitialized_ = true;
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
    }

    __aicore__ inline void SyncVToMte2()
    {
        if (!eventVToMte2Initialized_) {
            eventIdVToMte2_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            eventVToMte2Initialized_ = true;
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
    }

    __aicore__ inline void SyncVToS()
    {
        if (!eventVToSInitialized_) {
            eventIdVToS_ = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            eventVToSInitialized_ = true;
        }
        SetFlag<HardEvent::V_S>(eventIdVToS_);
        WaitFlag<HardEvent::V_S>(eventIdVToS_);
    }

    __aicore__ inline void ReleaseEvents()
    {
        if (eventMte2ToVInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_);
            eventMte2ToVInitialized_ = false;
        }
        if (eventVToMte2Initialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2_);
            eventVToMte2Initialized_ = false;
        }
        if (eventVToSInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(eventIdVToS_);
            eventVToSInitialized_ = false;
        }
    }

    __aicore__ inline void Process()
    {
        if (!ValidateCuSeqlens()) {
            ReleaseEvents();
            return;
        }
        uint64_t vectorCoreNum = GetBlockNum();
        uint64_t taskNum = B_ * NV_;
        for (uint64_t taskIdx = blockIdx; taskIdx < taskNum; taskIdx += vectorCoreNum) {
            uint64_t batch_i = taskIdx / NV_;
            uint64_t head_i = taskIdx % NV_;
            int64_t seq0 = SequenceStart(batch_i);
            int64_t seq1 = SequenceEnd(batch_i);
            int64_t seqLen64 = seq1 - seq0;
            if (seqLen64 == 0) {
                continue;
            }
            int32_t seqLen = static_cast<int32_t>(seqLen64);
            if (!ValidateStateSlots(batch_i, seq0, seqLen)) {
                ReleaseEvents();
                return;
            }

            uint64_t stateSlot = ResolveInitialStateSlot(batch_i, seq0, seqLen);
            if (stateSlot == INVALID_STATE_SLOT) {
                ReleaseEvents();
                return;
            }
            CopyInBeta(seq0, seq1);
            ProcessHead(batch_i, seq0, seq1, head_i, stateSlot);
        }
        ReleaseEvents();
    }

private:
    __aicore__ inline bool ValidateCuSeqlens() const
    {
        if (!hasCuSeqlens_) {
            return seqLen_ <= MAX_MTP;
        }
        int64_t seq0 = LoadCuSeqlens(0);
        if (seq0 != 0) {
            return false;
        }
        for (uint64_t i = 0; i < B_; i++) {
            int64_t seq1 = LoadCuSeqlens(i + 1);
            int64_t length = seq1 - seq0;
            if (seq1 < seq0 || seq1 > static_cast<int64_t>(T_) ||
                length > static_cast<int64_t>(MAX_MTP) ||
                (hasSsmStateIndices_ && ssmStateStride_ > 0 && length > ssmStateStride_)) {
                return false;
            }
            seq0 = seq1;
        }
        return seq0 <= static_cast<int64_t>(T_);
    }

    __aicore__ inline int64_t LoadCuSeqlens(uint64_t index) const
    {
        return cuSeqlensDtype_ == 0 ? static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(index)) :
                                     cuSeqlensInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t SequenceStart(uint64_t batchIdx) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx) : static_cast<int64_t>(batchIdx * seqLen_);
    }

    __aicore__ inline int64_t SequenceEnd(uint64_t batchIdx) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx + 1) :
                               static_cast<int64_t>((batchIdx + 1) * seqLen_);
    }

    __aicore__ inline int64_t LoadSsmStateIndex(uint64_t index) const
    {
        return ssmStateIndicesDtype_ == 0 ? static_cast<int64_t>(ssmStateIndicesInt32Gm_.GetValue(index)) :
                                           ssmStateIndicesInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t LoadAcceptedTokens(uint64_t index) const
    {
        return acceptedTokensDtype_ == 0 ? static_cast<int64_t>(numAcceptedTokensInt32Gm_.GetValue(index)) :
                                          numAcceptedTokensInt64Gm_.GetValue(index);
    }

    __aicore__ inline uint64_t StateMetadataOffset(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        if (ssmStateStride_ == 0) {
            return static_cast<uint64_t>(tokenIdx);
        }
        return batchIdx * ssmStateStride_ + static_cast<uint64_t>(tokenIdx - seq0);
    }

    __aicore__ inline uint64_t LoadStateSlot(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        int64_t stateSlot = LoadSsmStateIndex(StateMetadataOffset(batchIdx, seq0, tokenIdx));
        if (stateSlot < 0 || stateSlot >= static_cast<int64_t>(stateCapacity_)) {
            return INVALID_STATE_SLOT;
        }
        return static_cast<uint64_t>(stateSlot);
    }

    __aicore__ inline bool ValidateStateSlots(uint64_t batchIdx, int64_t seq0, int32_t seqLen) const
    {
        if (!hasSsmStateIndices_) {
            return batchIdx < stateCapacity_;
        }
        if (hasAcceptedTokens_) {
            int64_t acceptedTokenNum = LoadAcceptedTokens(batchIdx);
            if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
                return false;
            }
        }
        for (int32_t step = 0; step < seqLen; ++step) {
            if (LoadStateSlot(batchIdx, seq0, seq0 + step) == INVALID_STATE_SLOT) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t ResolveInitialStateSlot(uint64_t batchIdx, int64_t seq0, int32_t seqLen) const
    {
        if (!hasSsmStateIndices_) {
            return batchIdx;
        }
        int64_t tokenIdx = seq0;
        if (hasAcceptedTokens_) {
            tokenIdx = seq0 + LoadAcceptedTokens(batchIdx) - 1;
        }
        return LoadStateSlot(batchIdx, seq0, tokenIdx);
    }

    template <typename dataType>
    __aicore__ inline void CopyVectorIn(LocalTensor<dataType> &dst, GlobalTensor<dataType> &src, uint64_t offset,
                                        uint64_t count)
    {
        uint64_t rowBytes = count * sizeof(dataType);
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(dst, src[offset], params, padParams);
        }
    }

    __aicore__ inline float ReadFloat(GlobalTensor<float> &tensor, uint64_t offset)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalar, tensor[offset], params, padParams);
        SyncMte2ToV();
        Adds(scalar, scalar, 0.0f, 1);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        Duplicate(scalar, x, 1);
        PipeBarrier<PIPE_V>();
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float SigmoidScalar(float x)
    {
        float denom = 1.0f + ExpScalar(-x);
        return 1.0f / denom;
    }

    __aicore__ inline void NormalizeRows(LocalTensor<float> &tensor, int32_t seqLen)
    {
        for (int32_t row = 0; row < seqLen; ++row) {
            uint32_t rowOffset = static_cast<uint32_t>(row) * alignK_;
            Mul(broadTmpInUb, tensor[rowOffset], tensor[rowOffset], alignK_);
            PipeBarrier<PIPE_V>();
            ReduceSumDispatch(deltaInUb, broadTmpInUb, 1);
            PipeBarrier<PIPE_V>();
            Sqrt(deltaInUb, deltaInUb, 1);
            PipeBarrier<PIPE_V>();
            SyncVToS();
            float norm = deltaInUb.GetValue(0);
            if (norm > 0.0f) {
                Muls(tensor[rowOffset], tensor[rowOffset], 1.0f / norm, alignK_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ApplyGateInKernel(uint64_t head, int32_t seqLen)
    {
        uint32_t total = static_cast<uint32_t>(seqLen) * alignK_;
        float expA = hasALog_ ? ExpScalar(ReadFloat(aLogGm_, head)) : 1.0f;

        if (hasDtBias_) {
            CopyVectorIn(broadTmpInUb, dtBiasGm_, head * realK_, realK_);
            SyncMte2ToV();
            for (int32_t row = 0; row < seqLen; ++row) {
                Add(gateInUb[row * alignK_], gateInUb[row * alignK_], broadTmpInUb, alignK_);
                PipeBarrier<PIPE_V>();
            }
        }

        if (safeGate_) {
            Muls(gateInUb, gateInUb, expA, total);
            PipeBarrier<PIPE_V>();
            Muls(broadTmpInUb, gateInUb, -1.0f, total);
            PipeBarrier<PIPE_V>();
            Exp(broadTmpInUb, broadTmpInUb, total);
            PipeBarrier<PIPE_V>();
            Adds(broadTmpInUb, broadTmpInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Duplicate(gateInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Div(gateInUb, gateInUb, broadTmpInUb, total);
            PipeBarrier<PIPE_V>();
            Muls(gateInUb, gateInUb, lowerBound_, total);
            PipeBarrier<PIPE_V>();
        } else {
            Exp(gateInUb, gateInUb, total);
            PipeBarrier<PIPE_V>();
            Adds(gateInUb, gateInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Ln(gateInUb, gateInUb, total);
            PipeBarrier<PIPE_V>();
            Muls(gateInUb, gateInUb, -expA, total);
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename gateType>
    __aicore__ inline void CopyInGate(uint64_t gateOffset, int32_t seqLen)
    {
        LocalTensor<gateType> gateLocal = gateInQueue_.AllocTensor<gateType>();
        Duplicate<gateType>(gateLocal, static_cast<gateType>(0), alignK_ * static_cast<uint32_t>(seqLen));
        SyncVToMte2();
        DataCopyExtParams gateInParams{static_cast<uint16_t>(seqLen),
                                       static_cast<uint32_t>(realK_ * sizeof(gateType)),
                                       static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(gateType)), 0, 0};
        DataCopyPadExtParams<gateType> gatePadParams{
            true, 0, static_cast<uint8_t>(alignK_ - realK_), static_cast<gateType>(0)};
        if constexpr (std::is_same<gateType, float32_t>()) {
            DataCopyPad(gateLocal, gateFloatGm_[gateOffset], gateInParams, gatePadParams);
        } else if constexpr (std::is_same<gateType, bfloat16_t>()) {
            DataCopyPad(gateLocal, gateBf16Gm_[gateOffset], gateInParams, gatePadParams);
        } else {
            DataCopyPad(gateLocal, gateFp16Gm_[gateOffset], gateInParams, gatePadParams);
        }
        gateInQueue_.EnQue<gateType>(gateLocal);
        gateLocal = gateInQueue_.DeQue<gateType>();
        if constexpr (std::is_same<gateType, float32_t>()) {
            Adds(gateInUb, gateLocal, 0.0f, alignK_ * static_cast<uint32_t>(seqLen));
        } else {
            Cast(gateInUb, gateLocal, AscendC::RoundMode::CAST_NONE,
                 alignK_ * static_cast<uint32_t>(seqLen));
        }
        gateInQueue_.FreeTensor(gateLocal);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInQKVGate(uint64_t qOffset, uint64_t kOffset, uint64_t vOffset,
                                         uint64_t gateOffset, int32_t seqLen, uint64_t head)
    {
        LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
        LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
        LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();

        DataCopyExtParams qInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((queryTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams kInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((keyTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams vInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                    static_cast<uint32_t>((valueTokenStride_ - realV_) * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> qkPadParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPadExtParams<inType> vPadParams{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};

        DataCopyPad(qLocal, queryGm_[qOffset], qInParams, qkPadParams);
        DataCopyPad(kLocal, keyGm_[kOffset], kInParams, qkPadParams);
        DataCopyPad(vLocal, valueGm_[vOffset], vInParams, vPadParams);
        qInQueue_.EnQue<inType>(qLocal);
        kInQueue_.EnQue<inType>(kLocal);
        vInQueue_.EnQue<inType>(vLocal);
        if (gateDtype_ == 0) {
            CopyInGate<float>(gateOffset, seqLen);
        } else if (gateDtype_ == 1) {
            CopyInGate<bfloat16_t>(gateOffset, seqLen);
        } else {
            CopyInGate<half>(gateOffset, seqLen);
        }

        qLocal = qInQueue_.DeQue<inType>();
        kLocal = kInQueue_.DeQue<inType>();
        vLocal = vInQueue_.DeQue<inType>();
        Cast(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
        AscendC::PipeBarrier<PIPE_V>();
        if (useQkL2norm_) {
            NormalizeRows(qInUb, seqLen);
            NormalizeRows(kInUb, seqLen);
        }
        Muls(qInUb, qInUb, scale_, seqLen * alignK_);
        AscendC::PipeBarrier<PIPE_V>();
        if (useGateInKernel_) {
            ApplyGateInKernel(head, seqLen);
        }
        Exp(gateInUb, gateInUb, alignK_ * seqLen);
        AscendC::PipeBarrier<PIPE_V>();

        qInQueue_.FreeTensor(qLocal);
        kInQueue_.FreeTensor(kLocal);
        vInQueue_.FreeTensor(vLocal);
    }

    __aicore__ inline void PrefetchState(uint64_t stateSlot, uint64_t head, uint64_t vOffset,
                                          uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.AllocTensor<stateType>();
        if (stateVFirst_) {
            uint64_t stateOffset =
                stateInStride0_ * stateSlot + stateInStride1_ * head + stateInStride2_ * vOffset;
            DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV),
                                            static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0, 0};
            DataCopyPadExtParams<stateType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
            DataCopyPad(stateLocal, initStateGm_[stateOffset], stateInParams, padParams);
        } else {
            for (uint32_t v = 0; v < curSingleV; ++v) {
                for (uint32_t k = 0; k < realK_; ++k) {
                    uint64_t stateOffset = stateInStride0_ * stateSlot + stateInStride1_ * head +
                                           stateInStride2_ * k +
                                           stateInStride3_ * (vOffset + v);
                    stateLocal.SetValue(v * alignK_ + k, initStateGm_.GetValue(stateOffset));
                }
            }
        }
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
        MatVecMul(stateInUb, gateInUb[curQKOffset], stateInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
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
        LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
        if (shouldStoreState_) {
            LocalTensor<stateType> stateOutLocal = stateOutQueue_.AllocTensor<stateType>();
            if constexpr (std::is_same<stateType, float32_t>()) {
                DataCopy(stateOutLocal, stateInUb, alignK_ * curSingleV);
            } else {
                Cast(stateOutLocal, stateInUb, AscendC::RoundMode::CAST_RINT, alignK_ * curSingleV);
            }
            stateOutQueue_.EnQue<stateType>(stateOutLocal);
        }
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

    __aicore__ inline void CopyOutState(uint64_t stateSlot, uint64_t head, uint64_t vOffset,
                                         uint32_t curSingleV)
    {
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.DeQue<stateType>();
        if (stateVFirst_) {
            uint64_t stateOffset =
                stateOutStride0_ * stateSlot + stateOutStride1_ * head + stateOutStride2_ * vOffset;
            DataCopyParams stateOutParams{static_cast<uint16_t>(curSingleV),
                                          static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0};
            DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
        } else {
            SyncVToS();
            for (uint32_t v = 0; v < curSingleV; ++v) {
                for (uint32_t k = 0; k < realK_; ++k) {
                    uint64_t stateOffset = stateOutStride0_ * stateSlot + stateOutStride1_ * head +
                                           stateOutStride2_ * k +
                                           stateOutStride3_ * (vOffset + v);
                    finalStateGm_.SetValue(stateOffset, stateOutLocal.GetValue(v * alignK_ + k));
                }
            }
        }
        stateOutQueue_.FreeTensor(stateOutLocal);
    }

    template <typename betaType>
    __aicore__ inline void CopyInBetaTyped(int64_t seq0, int64_t seq1)
    {
        int64_t seqLen = seq1 - seq0;
        uint64_t betaCount = static_cast<uint64_t>(seqLen) * NV_;
        uint64_t betaBatchSize = Ceil(betaCount, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
        LocalTensor<betaType> betaLocal = betaInQueue_.AllocTensor<betaType>();
        if constexpr (std::is_same<betaType, float32_t>()) {
            CopyVectorIn(betaLocal, betaFloatGm_, static_cast<uint64_t>(seq0) * NV_, betaCount);
        } else if constexpr (std::is_same<betaType, bfloat16_t>()) {
            CopyVectorIn(betaLocal, betaBf16Gm_, static_cast<uint64_t>(seq0) * NV_, betaCount);
        } else {
            CopyVectorIn(betaLocal, betaFp16Gm_, static_cast<uint64_t>(seq0) * NV_, betaCount);
        }
        betaInQueue_.EnQue<betaType>(betaLocal);
        betaLocal = betaInQueue_.DeQue<betaType>();
        if constexpr (std::is_same<betaType, float32_t>()) {
            Adds(betaInUb, betaLocal, 0.0f, static_cast<uint32_t>(betaBatchSize));
        } else {
            Cast(betaInUb, betaLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(betaBatchSize));
        }
        betaInQueue_.FreeTensor(betaLocal);
        PipeBarrier<PIPE_V>();
        SyncVToS();
    }

    __aicore__ inline void CopyInBeta(int64_t seq0, int64_t seq1)
    {
        if (betaDtype_ == 0) {
            CopyInBetaTyped<float>(seq0, seq1);
        } else if (betaDtype_ == 1) {
            CopyInBetaTyped<bfloat16_t>(seq0, seq1);
        } else {
            CopyInBetaTyped<half>(seq0, seq1);
        }
    }

    __aicore__ inline uint64_t StateSlotForToken(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        if (hasSsmStateIndices_) {
            return LoadStateSlot(batchIdx, seq0, tokenIdx);
        }
        return batchIdx;
    }

    __aicore__ inline float LoadBeta(uint64_t gbOffset)
    {
        float beta = betaInUb.GetValue(gbOffset);
        if (useBetaSigmoid_) {
            beta = SigmoidScalar(beta);
            if (allowNegEigval_) {
                beta *= 2.0f;
            }
        }
        return beta;
    }

    __aicore__ inline void ProcessHead(uint64_t batchIdx, int64_t seq0, int64_t seq1,
                                      uint64_t head_i, uint64_t stateSlot)
    {
        uint64_t qkHead = head_i / (NV_ / NK_);
        uint64_t qOffset = static_cast<uint64_t>(seq0) * queryTokenStride_ + qkHead * queryHeadStride_;
        uint64_t kOffset = static_cast<uint64_t>(seq0) * keyTokenStride_ + qkHead * keyHeadStride_;
        uint64_t vOffset = static_cast<uint64_t>(seq0) * valueTokenStride_ + head_i * valueHeadStride_;
        uint64_t gateOffset = (static_cast<uint64_t>(seq0) * NV_ + head_i) * realK_;
        CopyInQKVGate(qOffset, kOffset, vOffset, gateOffset, static_cast<int32_t>(seq1 - seq0), head_i);
        if (realV_ == 0) {
            return;
        }
        uint64_t nextVOffset = 0;
        uint32_t nextSingleV = realV_ > vStep_ ? vStep_ : realV_;
        PrefetchState(stateSlot, head_i, 0, nextSingleV);
        for (uint64_t v_i = 0; v_i < realV_; v_i += vStep_) {
            uint32_t curSingleV = v_i + vStep_ > realV_ ? realV_ - v_i : vStep_;
            LoadPrefetchedState(curSingleV);
            nextVOffset = v_i + vStep_;
            if (nextVOffset < realV_) {
                nextSingleV = nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
                PrefetchState(stateSlot, head_i, nextVOffset, nextSingleV);
            }
            uint64_t pendingAttnOffset = 0;
            uint64_t pendingStateSlot = 0;
            bool hasPendingAttn = false;
            bool hasPendingState = false;
            for (int64_t seq_i = seq0; seq_i < seq1; seq_i++) {
                uint64_t gbOffset = head_i + static_cast<uint64_t>(seq_i - seq0) * NV_;
                uint64_t curQKOffset = static_cast<uint64_t>(seq_i - seq0) * alignK_;
                uint64_t curVOffset = static_cast<uint64_t>(seq_i - seq0) * alignV_ + v_i;
                uint64_t attnOffset = (static_cast<uint64_t>(seq_i) * NV_ + head_i) * realV_ + v_i;
                uint64_t curStateSlot = StateSlotForToken(batchIdx, seq0, seq_i);
                uint64_t curStateOutSlot = curStateSlot;
                beta_ = LoadBeta(gbOffset);
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
                if (shouldStoreState_) {
                    if (stateOutBufferNum_ == BUFFER_NUM) {
                        CopyOutState(curStateOutSlot, head_i, v_i, curSingleV);
                    } else {
                        if (hasPendingState) {
                            CopyOutState(pendingStateSlot, head_i, v_i, curSingleV);
                        }
                        pendingStateSlot = curStateOutSlot;
                        hasPendingState = true;
                    }
                }
            }
            if (hasPendingAttn) {
                CopyOutAttn(pendingAttnOffset, curSingleV);
            }
            if (hasPendingState) {
                CopyOutState(pendingStateSlot, head_i, v_i, curSingleV);
            }
        }
     }

private:
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<float> gateFloatGm_;
    GlobalTensor<bfloat16_t> gateBf16Gm_;
    GlobalTensor<half> gateFp16Gm_;
    GlobalTensor<float> betaFloatGm_;
    GlobalTensor<bfloat16_t> betaBf16Gm_;
    GlobalTensor<half> betaFp16Gm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensInt32Gm_;
    GlobalTensor<int64_t> cuSeqlensInt64Gm_;
    GlobalTensor<int32_t> ssmStateIndicesInt32Gm_;
    GlobalTensor<int64_t> ssmStateIndicesInt64Gm_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<int32_t> numAcceptedTokensInt32Gm_;
    GlobalTensor<int64_t> numAcceptedTokensInt64Gm_;
    GlobalTensor<stateType> finalStateGm_;
    GlobalTensor<outType> attnOutGm_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> qInQueue_;
    TQue<QuePosition::VECIN, 1> kInQueue_;
    TQue<QuePosition::VECIN, 1> vInQueue_;
    TQue<QuePosition::VECIN, 1> gateInQueue_;
    TQue<QuePosition::VECIN, 1> betaInQueue_;
    TQue<QuePosition::VECIN, 1> stateInQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> attnOutQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> stateOutQueue_;
    TBuf<TPosition::VECCALC> tmpBuff;
    TBuf<TPosition::VECCALC> scalarBuf_;
    LocalTensor<float> qInUb;
    LocalTensor<float> kInUb;
    LocalTensor<float> vInUb;
    LocalTensor<float> gateInUb;
    LocalTensor<float> betaInUb;
    LocalTensor<float> deltaInUb;
    LocalTensor<float> broadTmpInUb;
    LocalTensor<float> attnInUb;
    LocalTensor<float> stateInUb;
    TEventID eventIdMte2ToV_;
    TEventID eventIdVToMte2_;
    TEventID eventIdVToS_;
    bool eventMte2ToVInitialized_;
    bool eventVToMte2Initialized_;
    bool eventVToSInitialized_;
    uint32_t B_;
    uint32_t T_;
    uint32_t seqLen_;
    uint32_t NK_;
    uint32_t alignK_;
    uint32_t realK_;
    uint32_t NV_;
    uint32_t alignV_;
    uint32_t realV_;
    uint32_t stateCapacity_;
    uint32_t ssmStateStride_;
    uint64_t queryTokenStride_;
    uint64_t queryHeadStride_;
    uint64_t keyTokenStride_;
    uint64_t keyHeadStride_;
    uint64_t valueTokenStride_;
    uint64_t valueHeadStride_;
    uint64_t stateInStride0_;
    uint64_t stateInStride1_;
    uint64_t stateInStride2_;
    uint64_t stateInStride3_;
    uint64_t stateOutStride0_;
    uint64_t stateOutStride1_;
    uint64_t stateOutStride2_;
    uint64_t stateOutStride3_;
    uint32_t vStep_;
    uint32_t stateOutBufferNum_;
    uint32_t attnOutBufferNum_;
    uint32_t restUbSize_;
    uint32_t gateDtype_;
    uint32_t betaDtype_;
    uint32_t cuSeqlensDtype_;
    uint32_t ssmStateIndicesDtype_;
    uint32_t acceptedTokensDtype_;
    bool hasCuSeqlens_;
    bool hasSsmStateIndices_;
    bool hasAcceptedTokens_;
    bool hasALog_;
    bool hasDtBias_;
    bool useQkL2norm_;
    bool useGateInKernel_;
    bool useBetaSigmoid_;
    bool allowNegEigval_;
    bool safeGate_;
    bool stateVFirst_;
    bool shouldStoreState_;
    bool useAddFoldReduce_;
    float beta_;
    float scale_;
    float lowerBound_;
    uint64_t blockIdx;
};
} // namespace RecurrentKda
#endif
