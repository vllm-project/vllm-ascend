// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//

#include "kernel/common.h"
#include "kernel/iterator.h"
#include "kernel/mem.h"
#include "kernel/mma.h"
#include "kernel/utils.h"
#include "kernel/simd.h"
#include "kernel/kernel_utils.h"

#include "lib/matmul_intf.h"

#include "mla_preprocess.h"
#include "../op_host/tiling/mla_preprocess_tiling.h"

namespace MLAPO_FP16 {

template <typename QkDtype, typename CosDtype, typename QOutDtype, int8_t CacheMode>
class RopeFp16
{
public:
    __aicore__ inline RopeFp16() : blockIdx_(AscendC::GetBlockIdx()) {}

    __aicore__ inline void RopeInit(AscendC::GlobalTensor<QkDtype> &qGm, AscendC::GlobalTensor<CosDtype> &cosGm,
                                    AscendC::GlobalTensor<CosDtype> &sinGm,
                                    AscendC::GlobalTensor<QOutDtype> &outRopeConcatGm,
                                    AscendC::GlobalTensor<QkDtype> &outRopeConcatGm2,
                                    const MlaTilingData &ropeConcatParams)
    {
        this->qGm_ = qGm;
        this->cosGm_ = cosGm;
        this->sinGm_ = sinGm;
        this->outRopeConcatGm_ = outRopeConcatGm;
        this->outRopeConcatGm2_ = outRopeConcatGm2;

        headDim = ropeConcatParams.headDim;
        headNumQ = ropeConcatParams.headNumQ;
        rotaryCoeff = ropeConcatParams.rotaryCoeff;
        ntokens = ropeConcatParams.ntokens;
        realCore = ropeConcatParams.realCore;
        nlCoreRun = ropeConcatParams.nlCoreRun;
        lCoreRun = ropeConcatParams.lCoreRun;
        maxNPerLoopForUb = ropeConcatParams.maxNPerLoopForUb;
        preCoreLoopTime = ropeConcatParams.preCoreLoopTime;
        preCoreLoopNLast = ropeConcatParams.preCoreLoopNLast;
        lastCoreLoopTime = ropeConcatParams.lastCoreLoopTime;
        lastCoreLoopNLast = ropeConcatParams.lastCoreLoopNLast;
        concatSize = ropeConcatParams.concatSize;
        blockIdx_ = (blockIdx_ / 2) * 2 + static_cast<uint64_t>(GetSubBlockidx());
        loopTime = (blockIdx_ == realCore - 1) ? lastCoreLoopTime : preCoreLoopTime;
        lastLoopN = (blockIdx_ == realCore - 1) ? lastCoreLoopNLast : preCoreLoopNLast;
        this->repeatSize_ = 64;  // 128 = 256B / sizeof(fp32)
        this->rotateStride_ = this->headDim / this->rotaryCoeff;
        headBlockLen = static_cast<uint16_t>(this->headDim / ELE_NUM_FP16);
        headBlockLenFP32 = static_cast<uint16_t>(this->headDim / ELE_NUM_FP32);
        rotaryLen = static_cast<uint16_t>(this->rotateStride_ / ELE_NUM_FP32);
        concatBlockLen = static_cast<uint16_t>(this->concatSize / ELE_NUM_FP16);
        outLineOffset = this->headDim + this->concatSize;
        uint32_t dataNum = this->headDim * this->maxNPerLoopForUb;
        dataSizeFp16 = dataNum * sizeof(QkDtype);
        dataSizeFp32 = dataNum * sizeof(float);
        uint32_t concatDataSize = this->concatSize * sizeof(QkDtype) * this->maxNPerLoopForUb;
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= realCore) return;
        uint64_t startCoreLineIndex = this->blockIdx_ * this->nlCoreRun;
        // [maxNPerLoopForUb,head_dim] 的 neg
        AscendC::LocalTensor<float> negLocal =
            buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 * 4 + dataSizeFp16 * 3);
        ExpandNeg(negLocal, this->maxNPerLoopForUb);

        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        for (uint32_t zz = 0; zz < this->loopTime; ++zz) {
            uint16_t loopN = (zz == this->loopTime - 1) ? this->lastLoopN : this->maxNPerLoopForUb;
            uint64_t startHead = startCoreLineIndex + zz * this->maxNPerLoopForUb;
            uint64_t endHead = startHead + loopN;

            // move in Q
            AscendC::LocalTensor<QkDtype> inputQ = buf.GetBuffer<BufferType::ASCEND_UB, QkDtype>(0);
            AscendC::LocalTensor<float> inputQCastFP32 = buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp16);
            AscendC::LocalTensor<float> reverseQ =
                buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 + dataSizeFp16);
            uint64_t qOffset = startHead * 192 + 128;
            CopyQGenReverseQ(inputQ, inputQCastFP32, reverseQ, qOffset, loopN);

            // move in cos/sin
            AscendC::LocalTensor<QkDtype> inputCos =
                buf.GetBuffer<BufferType::ASCEND_UB, QkDtype>(dataSizeFp32 * 2 + dataSizeFp16);
            AscendC::LocalTensor<QkDtype> inputSin =
                buf.GetBuffer<BufferType::ASCEND_UB, QkDtype>(dataSizeFp32 * 2 + dataSizeFp16 * 2);
            uint64_t startSinCosHeadIndex = startHead;
            uint64_t headRemain = startHead % this->headNumQ;
            uint64_t localStartAddr = 0;
            if (headRemain != 0) {
                uint64_t preProcessHeadNum = this->headNumQ - headRemain;
                uint64_t needToProcesHead = preProcessHeadNum > loopN ? loopN : preProcessHeadNum;
                CopyCosSin(inputCos, inputSin, localStartAddr, (startSinCosHeadIndex / this->headNumQ) * this->headDim,
                           needToProcesHead);
                startSinCosHeadIndex += needToProcesHead;
                localStartAddr += needToProcesHead * this->headDim;
            }
            // Iterate through the remaining data.
            if (startSinCosHeadIndex < endHead) {
                uint64_t startSinCosIndex = startSinCosHeadIndex / this->headNumQ;
                uint64_t endSinCosIndex = (endHead + this->headNumQ - 1) / this->headNumQ;
                for (uint32_t index = startSinCosIndex; index < endSinCosIndex; ++index) {
                    // Mantissa
                    uint32_t repeatNum =
                        index == endSinCosIndex - 1 ? endHead - index * this->headNumQ : this->headNumQ;
                    CopyCosSin(inputCos, inputSin, localStartAddr, index * this->headDim, repeatNum);
                    localStartAddr += this->headDim * this->headNumQ;
                }
            }
            AscendC::LocalTensor<float> inputCosCastFP32 =
                buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 * 2 + dataSizeFp16 * 3);
            AscendC::LocalTensor<float> inputSinCastFP32 =
                buf.GetBuffer<BufferType::ASCEND_UB, float>(dataSizeFp32 * 3 + dataSizeFp16 * 3);
            AscendC::Cast(inputCosCastFP32, inputCos, AscendC::RoundMode::CAST_NONE, loopN * this->headDim);
            AscendC::Cast(inputSinCastFP32, inputSin, AscendC::RoundMode::CAST_NONE, loopN * this->headDim);
            AscendC::PipeBarrier<PIPE_V>();

            // rope result
            uint32_t repeatTime = this->headDim * loopN;
            AscendC::Mul(inputQCastFP32, inputCosCastFP32, inputQCastFP32, repeatTime);

            AscendC::Mul(reverseQ, negLocal, reverseQ, repeatTime);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(reverseQ, inputSinCastFP32, reverseQ, repeatTime);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Add(inputQCastFP32, reverseQ, inputQCastFP32, repeatTime);
            AscendC::PipeBarrier<PIPE_V>();

            // move out rope result
            // cast fp16/bf16
            AscendC::Cast(inputQ, inputQCastFP32, AscendC::RoundMode::CAST_RINT, loopN * this->headDim);
            AscendC::PipeBarrier<PIPE_V>();
            uint64_t outQOffset = startHead * outLineOffset + this->concatSize;
            uint64_t outQOffset2 = startHead * this->headDim;
            SET_FLAG(V, MTE3, EVENT_ID1);
            WAIT_FLAG(V, MTE3, EVENT_ID1);
            if constexpr (CacheMode == CACHE_MODE_KVCACHE) {
                AscendC::DataCopy(this->outRopeConcatGm_[outQOffset], inputQ, {loopN, headBlockLen, 0, concatBlockLen});
            } else {
                AscendC::DataCopy(this->outRopeConcatGm2_[outQOffset2], inputQ, loopN * this->headDim);
            }
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    }
    // tensor -1 -1 -1 1 1 1
    template <typename BUF_TYPE>
    __aicore__ inline void ExpandNeg(const AscendC::LocalTensor<BUF_TYPE> &tempBuf, uint32_t headNumTemp)
    {
        for (uint32_t i = 0; i < this->rotateStride_; ++i) {
            tempBuf.SetValue(i, (BUF_TYPE)-1);
            tempBuf.SetValue(i + this->rotateStride_, (BUF_TYPE)1);
        }
        SET_FLAG(S, V, EVENT_ID1);
        WAIT_FLAG(S, V, EVENT_ID1);
        AscendC::Copy(tempBuf[this->headDim], tempBuf, this->headDim, headNumTemp - 1, {1, 1, headBlockLenFP32, 0});
    }

    template <typename BUF_TYPE>
    __aicore__ inline void CopyQGenReverseQ(const AscendC::LocalTensor<BUF_TYPE> &tempBufQ,
                                            const AscendC::LocalTensor<float> &tempBufQCast,
                                            const AscendC::LocalTensor<float> &tempBufRverseQ, uint64_t qOffset,
                                            uint16_t loopN)
    {
        SET_FLAG(S, MTE2, EVENT_ID1);
        WAIT_FLAG(S, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        // move in Q
        AscendC::DataCopy(tempBufQ, this->qGm_[qOffset], {loopN, headBlockLen, 128 / 16, 0});
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        // cast fp32
        AscendC::Cast(tempBufQCast, tempBufQ, AscendC::RoundMode::CAST_NONE, loopN * this->headDim);
        AscendC::PipeBarrier<PIPE_V>();
        // move in reverseQ
        AscendC::DataCopy(tempBufRverseQ, tempBufQCast[this->rotateStride_], {loopN, rotaryLen, rotaryLen, rotaryLen});
        AscendC::DataCopy(tempBufRverseQ[this->rotateStride_], tempBufQCast, {loopN, rotaryLen, rotaryLen, rotaryLen});
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename BUF_TYPE>
    __aicore__ inline void CopyCosSin(const AscendC::LocalTensor<BUF_TYPE> &tempBufCos,
                                      const AscendC::LocalTensor<BUF_TYPE> &tempBufSin, uint64_t localStartAddr,
                                      uint64_t gmStartAddr, uint64_t repeatNum)
    {
        SET_FLAG(S, MTE2, EVENT_ID1);
        WAIT_FLAG(S, MTE2, EVENT_ID1);
        AscendC::DataCopy(tempBufCos[localStartAddr], this->cosGm_[gmStartAddr], {1, headBlockLen, 0, 0});
        AscendC::DataCopy(tempBufSin[localStartAddr], this->sinGm_[gmStartAddr], {1, headBlockLen, 0, 0});
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        AscendC::Copy(tempBufCos[localStartAddr + this->headDim], tempBufCos[localStartAddr], this->headDim,
                      repeatNum - 1, {1, 1, headBlockLen, 0});
        AscendC::Copy(tempBufSin[localStartAddr + this->headDim], tempBufSin[localStartAddr], this->headDim,
                      repeatNum - 1, {1, 1, headBlockLen, 0});
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::GlobalTensor<QkDtype> qGm_;
    AscendC::GlobalTensor<CosDtype> cosGm_;
    AscendC::GlobalTensor<CosDtype> sinGm_;
    AscendC::GlobalTensor<QOutDtype> outRopeConcatGm_;
    AscendC::GlobalTensor<QkDtype> outRopeConcatGm2_;

    uint32_t repeatSize_{0};
    uint32_t rotateStride_{0};  // this->headDim / rope_conf
    uint32_t headDim;
    uint32_t headNumQ;
    uint32_t rotaryCoeff;
    uint32_t ntokens;
    uint32_t realCore;
    uint32_t nlCoreRun;
    uint32_t lCoreRun;
    uint32_t maxNPerLoopForUb;
    uint32_t preCoreLoopTime;
    uint32_t preCoreLoopNLast;
    uint32_t lastCoreLoopTime;
    uint32_t lastCoreLoopNLast;
    uint32_t concatSize;
    uint32_t blockIdx_;
    uint32_t loopTime{0};   // The number of current data rounds
    uint32_t lastLoopN{0};  // The number of lines currently processed by tails kernel

    uint32_t dataSizeFp32;
    uint32_t dataSizeFp16;
    uint16_t headBlockLen{0};
    uint16_t headBlockLenFP32{0};
    uint16_t rotaryLen{0};
    uint16_t concatBlockLen{0};
    uint64_t outLineOffset{0};
};

__aicore__ inline void ReduceSumCustom(const AscendC::LocalTensor<float> &dst_local,
                                       const AscendC::LocalTensor<float> &src_local,
                                       const AscendC::LocalTensor<float> &work_local, int32_t count)
{
#ifdef __DAV_C220_VEC__
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    AscendC::BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = AscendC::ONE_REPEAT_BYTE_SIZE / AscendC::ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    AscendC::PipeBarrier<PIPE_V>();
    if (likely(repeatTimes > 0)) {
        Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
    cadd_v<ArchType::ASCEND_V220, float>(dst_local,   // dst
                                         work_local,  // src
                                         1,           // repeat
                                         0,           // dstRepeatStride
                                         1,           // srcBlockStride
                                         0);          // srcRepeatStride
    AscendC::PipeBarrier<PIPE_V>();
#endif
}

template <typename T, bool WITH_BETA, bool FastComputeMode = false>
class Quant
{
public:
    __aicore__ inline Quant() {}

    __aicore__ inline void Init(AscendC::GlobalTensor<T> quantScaleGmTensor,
                                AscendC::GlobalTensor<int8_t> quantOffsetGmTensor,
                                AscendC::GlobalTensor<T> inputGmTensor, AscendC::GlobalTensor<int8_t> outputGmTensor,
                                uint32_t stride, uint32_t num_col, float avg_factor, uint64_t gm_offset,
                                uint64_t gm_out_offset, uint32_t row_work_, const MlaTilingData &mlaParams_)
    {
        this->quantScaleGmTensor = quantScaleGmTensor;
        this->quantOffsetGmTensor = quantOffsetGmTensor;
        this->inputGmTensor = inputGmTensor;
        this->outputGmTensor = outputGmTensor;
        num_col_ = num_col;
        quantMin_ = -128;
        uint32_t num_row = mlaParams_.n;
        this->row_work = row_work;
        this->row_work_ = row_work_;
        gm_offset_ = gm_offset;
        gm_out_offset_ = gm_out_offset;
        num_col_align_int8 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        num_col_align_f16 = (num_col_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        num_col_align_f32 = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        input_stride_ = stride;

        num_col_align_withStride_int8 =
            (num_col_ - input_stride_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        num_col_align_withStride_fp16 =
            (num_col_ - input_stride_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        num_col_align_withStride_fp32 =
            (num_col_ - input_stride_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
    }

    __aicore__ inline void Launch(const AscendC::LocalTensor<int8_t> &dstTensor,
                                  const AscendC::LocalTensor<T> &srcTensor, const AscendC::LocalTensor<T> &gammaTensor,
                                  const AscendC::LocalTensor<T> &betaTensor,
                                  const AscendC::LocalTensor<T> &quantScaleTensor,
                                  const AscendC::LocalTensor<int8_t> &quantOffsetTensor,
                                  const AscendC::LocalTensor<float> &res1Tensor,
                                  const AscendC::LocalTensor<float> &res3Tensor)
    {
        this->dstTensor = dstTensor;
        this->srcTensor = srcTensor;
        this->fp32_xy = res1Tensor;
        this->buf = res3Tensor;

        AscendC::DataCopy(srcTensor, inputGmTensor[gm_offset_],
                          AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));
        SET_FLAG(MTE2, V, EVENT_ID0);

        SET_FLAG(MTE2, V, EVENT_ID1);
        AscendC::DataCopy(quantScaleTensor, quantScaleGmTensor,
                          AscendC::DataCopyParams(1, 1, 0, 0));  // 7168 * 2 + 7168 * 2 + 32
        AscendC::DataCopy(quantOffsetTensor, quantOffsetGmTensor,
                          AscendC::DataCopyParams(1, 1, 0, 0));  // 7168 * 2 + 7168 * 2 + 64
        SET_FLAG(MTE2, S, EVENT_ID0);

        uint64_t pid = 0;
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        while (pid < row_work_) {
            uint64_t offset = pid * num_col_;  // + offset
            uint64_t outOffset = pid * (num_col_ - input_stride_);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            if (pid > 0) {
                AscendC::DataCopy(srcTensor, inputGmTensor[gm_offset_ + offset],
                                  AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));  // 7168 * 2
                SET_FLAG(MTE2, V, EVENT_ID0);
            }
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            // modify input
            Cast(fp32_xy, srcTensor[input_stride_], AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64,
                 num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
            AscendC::PipeBarrier<PIPE_V>();

            if (pid == 0) {
                WAIT_FLAG(MTE2, V, EVENT_ID1);
                WAIT_FLAG(MTE2, S, EVENT_ID0);
                input_scale_ = 1 / (float)(quantScaleTensor.GetValue(0));
                input_offset_ = (float)(quantOffsetTensor.GetValue(0));
                SET_FLAG(S, V, EVENT_ID0);
                WAIT_FLAG(S, V, EVENT_ID0);
            }

            Muls(fp32_xy, fp32_xy, input_scale_, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            Adds(fp32_xy, fp32_xy, input_offset_, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> tmpfp16 =
                buf.ReinterpretCast<half>()[OFFSET_SUM * num_col_align_withStride_fp32 * 2];
            CastFrom32To16(tmpfp16, fp32_xy, num_col_align_withStride_fp32);
            AscendC::PipeBarrier<PIPE_V>();
            CastFromF16ToI8(dstTensor, tmpfp16, quantMin_, num_col_align_withStride_fp16);
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            AscendC::DataCopy(outputGmTensor[gm_out_offset_ + outOffset], dstTensor,
                              AscendC::DataCopyParams(1, (num_col_ - input_stride_) / 32, 0, 0));
            SET_FLAG(MTE3, V, EVENT_ID0);
            WAIT_FLAG(MTE3, V, EVENT_ID0);
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            ++pid;
            AscendC::PipeBarrier<PIPE_V>();
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }

private:
    AscendC::LocalTensor<int8_t> dstTensor;
    AscendC::LocalTensor<T> srcTensor;
    AscendC::LocalTensor<float> fp32_xy;
    AscendC::LocalTensor<float> buf;

    AscendC::GlobalTensor<T> quantScaleGmTensor;
    AscendC::GlobalTensor<int8_t> quantOffsetGmTensor;
    AscendC::GlobalTensor<T> inputGmTensor;
    AscendC::GlobalTensor<int8_t> outputGmTensor;

    uint32_t num_col_{0};
    uint32_t row_work{0};
    uint32_t row_work_{0};
    uint32_t row_step_{0};
    uint32_t row_tail_{0};
    uint64_t gm_offset_{0};
    uint64_t gm_out_offset_{0};
    float avg_factor_{1.0};  // 1/num_col_
    float input_scale_{1.0};
    float input_offset_{0};
    int32_t input_stride_{0};
    float epsilon_{1e-12f};
    uint32_t num_col_align_int8{0};
    uint32_t num_col_align_f16{0};
    uint32_t num_col_align_f32{0};
    uint32_t num_col_align_f32_long{0};
    uint32_t num_col_align_withStride_int8{0};
    uint32_t num_col_align_withStride_fp16{0};
    uint32_t num_col_align_withStride_fp32{0};
    uint32_t num_col_temp;
    half quantMin_{-128};
    uint32_t num_slice_{0};
    uint32_t tail_size_{0};
    uint32_t tail_copy_{0};
};

template <typename T, bool WITH_BETA, bool FastComputeMode = false>
class RmsNormQuant
{
public:
    __aicore__ inline RmsNormQuant() {}

    __aicore__ inline void Init(AscendC::GlobalTensor<T> gammaGmTensor, AscendC::GlobalTensor<T> betaGmTensor,
                                AscendC::GlobalTensor<T> quantScaleGmTensor,
                                AscendC::GlobalTensor<int8_t> quantOffsetGmTensor,
                                AscendC::GlobalTensor<T> inputGmTensor, AscendC::GlobalTensor<int8_t> outputGmTensor,
                                uint32_t stride, uint32_t num_col, float avg_factor, uint64_t gm_offset,
                                uint64_t gm_out_offset, uint32_t row_work_, const MlaTilingData &mlaParams_)
    {
        this->gammaGmTensor = gammaGmTensor;
        this->betaGmTensor = betaGmTensor;
        this->quantScaleGmTensor = quantScaleGmTensor;
        this->quantOffsetGmTensor = quantOffsetGmTensor;
        this->inputGmTensor = inputGmTensor;
        this->outputGmTensor = outputGmTensor;
        num_col_ = num_col;
        avg_factor_ = avg_factor;
        epsilon_ = 1e-6;
        quantMin_ = -128;
        uint32_t num_row = mlaParams_.n;
        this->row_work = row_work;
        this->row_work_ = row_work_;
        gm_offset_ = gm_offset;
        gm_out_offset_ = gm_out_offset;
        num_col_align_int8 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        num_col_align_f16 = (num_col_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        num_col_align_f32 = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        input_stride_ = stride;

        num_col_align_withStride_int8 =
            (num_col_ - input_stride_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        num_col_align_withStride_fp16 =
            (num_col_ - input_stride_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        num_col_align_withStride_fp32 =
            (num_col_ - input_stride_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
    }

    __aicore__ inline void Launch(const AscendC::LocalTensor<int8_t> &dstTensor,
                                  const AscendC::LocalTensor<T> &srcTensor, const AscendC::LocalTensor<T> &gammaTensor,
                                  const AscendC::LocalTensor<T> &betaTensor,
                                  const AscendC::LocalTensor<T> &quantScaleTensor,
                                  const AscendC::LocalTensor<int8_t> &quantOffsetTensor,
                                  const AscendC::LocalTensor<float> &res1Tensor,
                                  const AscendC::LocalTensor<float> &res3Tensor)
    {
        this->dstTensor = dstTensor;
        this->srcTensor = srcTensor;
        this->gammaTensor = gammaTensor;
        this->betaTensor = betaTensor;
        this->fp32_xy = res1Tensor;
        this->buf = res3Tensor;
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * num_col_align_withStride_fp32];        // 0
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * num_col_align_withStride_fp32];        // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * num_col_align_withStride_fp32];       // 2
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE * num_col_align_withStride_fp32];  // 4

        AscendC::DataCopy(srcTensor, inputGmTensor[gm_offset_],
                          AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));
        SET_FLAG(MTE2, V, EVENT_ID0);

        AscendC::DataCopy(
            gammaTensor, gammaGmTensor,
            AscendC::DataCopyParams(1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));  // 7168 * 2 + 7168 * 2
        AscendC::DataCopy(
            betaTensor, betaGmTensor,
            AscendC::DataCopyParams(1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));  // 7168 * 2 + 7168 * 2
        SET_FLAG(MTE2, V, EVENT_ID1);
        AscendC::DataCopy(quantScaleTensor, quantScaleGmTensor,
                          AscendC::DataCopyParams(1, 1, 0, 0));  // 7168 * 2 + 7168 * 2 + 32
        AscendC::DataCopy(quantOffsetTensor, quantOffsetGmTensor,
                          AscendC::DataCopyParams(1, 1, 0, 0));  // 7168 * 2 + 7168 * 2 + 64
        SET_FLAG(MTE2, S, EVENT_ID0);

        uint64_t pid = 0;
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        while (pid < row_work_) {
            uint64_t offset = pid * num_col_;  // + offset
            uint64_t outOffset = pid * (num_col_ - input_stride_);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            if (pid > 0) {
                AscendC::DataCopy(srcTensor, inputGmTensor[gm_offset_ + offset],
                                  AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));  // 7168 * 2
                SET_FLAG(MTE2, V, EVENT_ID0);
            }
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            // modify input
            Cast(fp32_xy, srcTensor[input_stride_], AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64,
                 num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
            AscendC::PipeBarrier<PIPE_V>();
            Mul(sqx, fp32_xy, fp32_xy, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                 AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            Muls(sqx, sqx, avg_factor_, num_col_ - input_stride_);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSumCustom(sum, sqx, work, num_col_ - input_stride_);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(sum, sum, epsilon_, 1);
            AscendC::PipeBarrier<PIPE_V>();
            Sqrt(sum, sum, 1);
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
            float factor = 1 / sum.GetValue(0);
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            Muls(fp32_xy, fp32_xy, factor, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();

            if (pid == 0) {
                WAIT_FLAG(MTE2, V, EVENT_ID1);
                Cast(buf[OFFSET_GAMMA * num_col_align_withStride_fp32], gammaTensor, AscendC::RoundMode::CAST_NONE,
                     REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                     {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
                AscendC::PipeBarrier<PIPE_V>();

                WAIT_FLAG(MTE2, S, EVENT_ID0);
                input_scale_ = 1 / (float)(quantScaleTensor.GetValue(0));
                input_offset_ = (float)(quantOffsetTensor.GetValue(0));
            }

            Mul(fp32_xy, fp32_xy, g, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                 AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (WITH_BETA) {  // quant beta is fp16 add
                AscendC::LocalTensor<T> b = this->betaTensor;
                Cast(work, b, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64,
                     num_col_align_withStride_fp32 / REPEAT_TIME_64,
                     {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
                AscendC::PipeBarrier<PIPE_V>();
                Add(fp32_xy, fp32_xy, work, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                    {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                     AscendC::DEFAULT_REPEAT_STRIDE});
                AscendC::PipeBarrier<PIPE_V>();
            }
            Muls(fp32_xy, fp32_xy, input_scale_, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            Adds(fp32_xy, fp32_xy, input_offset_, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<half> tmpfp16 =
                buf.ReinterpretCast<half>()[OFFSET_SUM * num_col_align_withStride_fp32 * 2];
            CastFrom32To16(tmpfp16, fp32_xy, num_col_align_withStride_fp32);
            AscendC::PipeBarrier<PIPE_V>();
            CastFromF16ToI8(dstTensor, tmpfp16, quantMin_, num_col_align_withStride_fp16);
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            AscendC::DataCopy(outputGmTensor[gm_out_offset_ + outOffset], dstTensor,
                              AscendC::DataCopyParams(1, (num_col_ - input_stride_) / 32, 0, 0));
            SET_FLAG(MTE3, V, EVENT_ID0);
            WAIT_FLAG(MTE3, V, EVENT_ID0);
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            ++pid;
            AscendC::PipeBarrier<PIPE_V>();
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }

private:
private:
    AscendC::LocalTensor<int8_t> dstTensor;
    AscendC::LocalTensor<T> srcTensor;
    AscendC::LocalTensor<T> gammaTensor;
    AscendC::LocalTensor<T> betaTensor;
    AscendC::LocalTensor<float> fp32_xy;
    AscendC::LocalTensor<float> buf;

    AscendC::GlobalTensor<T> gammaGmTensor;
    AscendC::GlobalTensor<T> betaGmTensor;
    AscendC::GlobalTensor<T> quantScaleGmTensor;
    AscendC::GlobalTensor<int8_t> quantOffsetGmTensor;
    AscendC::GlobalTensor<T> inputGmTensor;
    AscendC::GlobalTensor<int8_t> outputGmTensor;

    uint32_t num_col_{0};
    uint32_t row_work{0};
    uint32_t row_work_{0};
    uint32_t row_step_{0};
    uint32_t row_tail_{0};
    uint64_t gm_offset_{0};
    uint64_t gm_out_offset_{0};
    float avg_factor_{1.0};
    float input_scale_{1.0};
    float input_offset_{0};
    int32_t input_stride_{0};
    float epsilon_{1e-12f};
    uint32_t num_col_align_int8{0};
    uint32_t num_col_align_f16{0};
    uint32_t num_col_align_f32{0};
    uint32_t num_col_align_f32_long{0};
    uint32_t num_col_align_withStride_int8{0};
    uint32_t num_col_align_withStride_fp16{0};
    uint32_t num_col_align_withStride_fp32{0};
    uint32_t num_col_temp;
    half quantMin_{-128};
    uint32_t num_slice_{0};
    uint32_t tail_size_{0};
    uint32_t tail_copy_{0};
};

__aicore__ __force_inline__ uint64_t Min(const uint64_t a, const uint64_t b)
{
    return a < b ? a : b;
}

__aicore__ __force_inline__ uint64_t Max(const uint64_t a, const uint64_t b)
{
    return a > b ? a : b;
}

template <uint64_t Base>
__aicore__ __force_inline__ uint64_t RoundUp(const uint64_t val)
{
    return (val + Base - 1) / Base * Base;
}

template <uint64_t Divisor>
__aicore__ __force_inline__ uint64_t CeilDiv(const uint64_t dividend)
{
    return (dividend + Divisor - 1) / Divisor;
}

template <typename InDtype, typename ScaleDtype>
class EinSumQuant
{
public:
    __aicore__ explicit EinSumQuant() {}

    __aicore__ inline void Init(GM_ADDR einSumOutGm, GM_ADDR scaleGm, GM_ADDR quantOutGm,
                                const MlaTilingData &tilingData)
    {
        einSumOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(einSumOutGm));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ScaleDtype *>(scaleGm));
        quantOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(quantOutGm));

        headNum = tilingData.esqHeadNum;
        colNum = tilingData.esqColNum;
        ubHeadLoop = tilingData.esqUbHeadLoop;
        headPerLoop = tilingData.esqHeadPerLoop;
        headTail = tilingData.esqHeadTail;
        colLoop = tilingData.esqColLoop;
        colTail = tilingData.esqColTail;

        currentIdx = (AscendC::GetBlockIdx() / 2) * 2 + GetSubBlockidx();
        if (currentIdx < tilingData.esqFrontCore) {
            batchNum = tilingData.esqFrontCoreBatch;
            currentCoreStartOffset = currentIdx * tilingData.esqFrontCoreBatch * headNum * colNum;
        } else {
            batchNum = tilingData.esqTailCoreBatch;
            currentCoreStartOffset = (tilingData.esqFrontCore * tilingData.esqFrontCoreBatch +
                                      (currentIdx - tilingData.esqFrontCore) * tilingData.esqTailCoreBatch) *
                                     headNum * colNum;
        }

        // calc tensors' data size(bytes)
        inputDataSize = headPerLoop * colNum * sizeof(InDtype);
        scaleDataSize = headPerLoop * sizeof(ScaleDtype);
        scaleBrcbFp16DataSize = headPerLoop * ELE_NUM_FP16 * sizeof(half);
        tempQuantFp16DataSize = inputDataSize;
        int8OutDataSize = headPerLoop * colNum;
        headTailDataSize = headTail * colNum * sizeof(InDtype);
        int8TailOutDataSize = headTail * colNum;
    }

    __aicore__ inline void Process()
    {
        if (batchNum == 0) {
            return;
        }
        // init local tensor
        inputTensor_ = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        scaleTensor_ = buf.GetBuffer<BufferType::ASCEND_UB, ScaleDtype>(inputDataSize);
        scaleBrcbFp16_ = buf.GetBuffer<BufferType::ASCEND_UB, half>(inputDataSize + scaleDataSize);
        tempQuantFp16_ =
            buf.GetBuffer<BufferType::ASCEND_UB, half>(inputDataSize + scaleDataSize + scaleBrcbFp16DataSize);
        int8OutTensor_ = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(inputDataSize + scaleDataSize +
                                                                      scaleBrcbFp16DataSize + tempQuantFp16DataSize);

        uint64_t inputLoopOffset = 0;
        uint32_t scaleLoopOffset = 0;
        uint64_t batchOffset = 0;
        uint64_t calcStartOffset = 0;
        uint64_t colOffset = 0;
        uint8_t calcRepeatStride = static_cast<uint8_t>(colNum / ELE_NUM_FP16);

        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        for (uint32_t ubLoopIdx = 0; ubLoopIdx < ubHeadLoop; ubLoopIdx++) {
            // scale CopyIn
            scaleLoopOffset = ubLoopIdx * headPerLoop;
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
            AscendC::DataCopy(scaleTensor_, scaleGm_[scaleLoopOffset], headPerLoop);
            SET_FLAG(MTE2, V, EVENT_ID1);
            WAIT_FLAG(MTE2, V, EVENT_ID1);
            // scale broadcast [H', 1] --> [H', 16]
            AscendC::Brcb(scaleBrcbFp16_, scaleTensor_, headPerLoop / 8, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();

            inputLoopOffset = ubLoopIdx * headPerLoop * colNum;
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
            for (uint32_t batchIdx = 0; batchIdx < batchNum; batchIdx++) {
                batchOffset = batchIdx * headNum * colNum;
                calcStartOffset = currentCoreStartOffset + batchOffset + inputLoopOffset;
                // input CopyIn
                WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
                AscendC::DataCopy(inputTensor_, einSumOutGm_[calcStartOffset],
                                  {1, static_cast<uint16_t>(inputDataSize / BLOCK_SIZE_32), 0, 0});
                SET_FLAG(MTE2, V, EVENT_ID1);
                WAIT_FLAG(MTE2, V, EVENT_ID1);

                // quant calc
                for (uint32_t colIdx = 0; colIdx < colLoop; colIdx++) {
                    colOffset = colIdx * CONST_128;
                    AscendC::Mul(tempQuantFp16_[colOffset], inputTensor_[colOffset], scaleBrcbFp16_, CONST_128,
                                 headPerLoop, {1, 1, 0, calcRepeatStride, calcRepeatStride, 1});
                }
                AscendC::PipeBarrier<PIPE_V>();

                // quant fp16 --> int8
                CastFromF16ToI8(int8OutTensor_, tempQuantFp16_, quantMin_, headPerLoop * colNum);
                AscendC::PipeBarrier<PIPE_V>();
                SET_FLAG(V, MTE3, EVENT_ID1);
                WAIT_FLAG(V, MTE3, EVENT_ID1);

                // int8 CopyOut
                AscendC::DataCopy(quantOutGm_[calcStartOffset], int8OutTensor_,
                                  {1, static_cast<uint16_t>(int8OutDataSize / BLOCK_SIZE_32), 0, 0});
                SET_FLAG(MTE3, MTE2, EVENT_ID1);
            }
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);

        // deal with headTail
        padLen = (headTail + ELE_NUM_FP16 - 1) / ELE_NUM_FP16 * ELE_NUM_FP16;
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        if (headTail > 0) {
            // scale CopyIn
            scaleLoopOffset = ubHeadLoop * headPerLoop;
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
            if (headTail == padLen) {
                AscendC::DataCopy(scaleTensor_, scaleGm_[scaleLoopOffset], headTail);
            } else {
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(headTail * sizeof(half)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<half> padParams{true, 0, static_cast<uint8_t>(padLen - headTail), 0};
                AscendC::DataCopyPad(scaleTensor_, scaleGm_[scaleLoopOffset], copyParams, padParams);
            }
            SET_FLAG(MTE2, V, EVENT_ID1);
            WAIT_FLAG(MTE2, V, EVENT_ID1);
            // scale broadcast [H', 1] --> [H', 16]
            AscendC::Brcb(scaleBrcbFp16_, scaleTensor_, padLen / 8, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();

            inputLoopOffset = ubHeadLoop * headPerLoop * colNum;
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
            for (uint32_t batchIdx = 0; batchIdx < batchNum; batchIdx++) {
                batchOffset = batchIdx * headNum * colNum;
                calcStartOffset = currentCoreStartOffset + batchOffset + inputLoopOffset;
                // input CopyIn
                WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
                AscendC::DataCopy(inputTensor_, einSumOutGm_[calcStartOffset],
                                  {1, static_cast<uint16_t>(headTailDataSize / BLOCK_SIZE_32), 0, 0});
                SET_FLAG(MTE2, V, EVENT_ID1);
                WAIT_FLAG(MTE2, V, EVENT_ID1);

                // quant calc
                for (uint32_t colIdx = 0; colIdx < colLoop; colIdx++) {
                    colOffset = colIdx * CONST_128;
                    AscendC::Mul(tempQuantFp16_[colOffset], inputTensor_[colOffset], scaleBrcbFp16_, CONST_128,
                                 headTail, {1, 1, 0, calcRepeatStride, calcRepeatStride, 1});
                }
                AscendC::PipeBarrier<PIPE_V>();

                // quant fp16 --> int8
                CastFromF16ToI8(int8OutTensor_, tempQuantFp16_, quantMin_, headTail * colNum);
                AscendC::PipeBarrier<PIPE_V>();
                SET_FLAG(V, MTE3, EVENT_ID1);
                WAIT_FLAG(V, MTE3, EVENT_ID1);

                // int8 CopyOut
                AscendC::DataCopy(quantOutGm_[calcStartOffset], int8OutTensor_,
                                  {1, static_cast<uint16_t>(int8TailOutDataSize / BLOCK_SIZE_32), 0, 0});
                SET_FLAG(MTE3, MTE2, EVENT_ID1);
            }
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    }

private:
    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::GlobalTensor<InDtype> einSumOutGm_;
    AscendC::GlobalTensor<ScaleDtype> scaleGm_;
    AscendC::GlobalTensor<int8_t> quantOutGm_;

    AscendC::LocalTensor<InDtype> inputTensor_;
    AscendC::LocalTensor<ScaleDtype> scaleTensor_;
    AscendC::LocalTensor<half> scaleBrcbFp16_;
    AscendC::LocalTensor<half> tempQuantFp16_;
    AscendC::LocalTensor<int8_t> int8OutTensor_;

    // [batchNum, headNum, colNum]
    uint32_t batchNum;
    uint32_t headNum;
    uint32_t colNum;
    // ub loop
    uint32_t ubHeadLoop;
    uint32_t headPerLoop;
    uint32_t headTail;
    // col loop
    uint32_t colLoop;
    uint32_t colTail;

    uint32_t currentIdx;
    uint64_t currentCoreStartOffset;
    uint32_t inputDataSize;  // bytes
    uint32_t scaleDataSize;
    uint32_t scaleBrcbFp16DataSize;
    uint32_t tempQuantFp16DataSize;
    uint32_t int8OutDataSize;
    uint32_t headTailDataSize;
    uint32_t int8TailOutDataSize;

    half quantMin_{-128};
    uint32_t padLen;
};

#ifdef __DAV_C220_CUBE__

struct MatCoord {
    uint64_t m{0};
    uint64_t k{0};
    uint64_t n{0};
};

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
class PpMatmulEinSum
{
    using InDtype = half;
    using OutDtype = half;
    using AccumDtype = float;

    template <DataFormat srcFormat, DataFormat dstFormat>
    using CopyGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, InDtype, srcFormat, dstFormat>;
    using LoadCbufToCa = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::ZN, DataFormat::ZZ>;
    using LoadCbufToCb = l1_to_l0_b<ArchType::ASCEND_V220, InDtype, transB, DataFormat::ZN, DataFormat::NZ>;
    using Mad = mmad<ArchType::ASCEND_V220, InDtype, InDtype, float, false>;
    using CopyCcToGm = l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, OutDtype, float>;

    static constexpr uint32_t L0_PINGPONG_BUFFER_LEN = 16384;
    static constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 131072;
    static constexpr uint32_t CONST_16 = 16;
    static constexpr uint32_t CONST_256 = 256;

public:
    __aicore__ explicit PpMatmulEinSum(){};

    __aicore__ __force_inline__ void Init(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, const MlaTilingData &mlaParams);

    __aicore__ __force_inline__ void Process();
    __aicore__ __force_inline__ void PreloadB();

private:
    __aicore__ __force_inline__ void GetBaseBlockIdx(uint64_t index, MatCoord &tidx);
    __aicore__ __force_inline__ uint64_t GetOffsetB(const uint64_t bIdx, const uint64_t kIdx, const uint64_t nIdx);
    __aicore__ __force_inline__ void CopyTileA(AscendC::LocalTensor<InDtype> &dstTensor,
                                               const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t m_actual,
                                               const uint64_t m_round, const uint64_t k_actual, const uint64_t k_round);
    __aicore__ __force_inline__ void CopyTileB(AscendC::LocalTensor<InDtype> &dstTensor,
                                               const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t k_actual,
                                               const uint64_t k_round, const uint64_t n_actual, const uint64_t n_round);

private:
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<InDtype> gm_b;
    AscendC::GlobalTensor<OutDtype> gm_c;
    AscendC::LocalTensor<InDtype> l1_base_a;
    AscendC::LocalTensor<InDtype> l1_base_b;
    AscendC::LocalTensor<InDtype> l0a_base;
    AscendC::LocalTensor<InDtype> l0b_base;
    AscendC::LocalTensor<float> l0c_buf;

    uint32_t num_core{0};
    uint32_t batch_size{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    MatCoord tdim{0};
    MatCoord fdim{0};
    uint32_t core_loop{0};
    uint32_t swizzle_cnt{1};
    uint32_t core_idx{0};
    uint32_t en_shuffle_k = 0;
    uint32_t ping_flag{0};
};

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
__aicore__ __force_inline__ void PpMatmulEinSum<formatB, transB, swizzleDirect, splitGapA, splitGapC>::Init(
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, const MlaTilingData &mlaParams)
{
#ifdef __DAV_C220_CUBE__
    batch_size = mlaParams.mm3.numBatch;
    m = mlaParams.mm3.m;
    k = mlaParams.mm3.k;
    n = mlaParams.mm3.n;
    m0 = mlaParams.mm3.m0;
    k0 = mlaParams.mm3.k0;
    n0 = mlaParams.mm3.n0;
    tdim.m = mlaParams.mm3.mLoop;
    tdim.k = mlaParams.mm3.kLoop;
    tdim.n = mlaParams.mm3.nLoop;
    core_loop = mlaParams.mm3.coreLoop;
    swizzle_cnt = mlaParams.mm3.swizzleCount;
    num_core = mlaParams.mm3.blockDim;
    core_idx = AscendC::GetBlockIdx();
    ping_flag = 1;

    gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(gmA));
    gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(gmB));
    gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(gmC));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    l1_base_a = buf.template GetBuffer<BufferType::ASCEND_CB>(0);
    l1_base_b = buf.template GetBuffer<BufferType::ASCEND_CB>(RoundUp<CONST_256>(m0 * k0 * sizeof(InDtype)));
    l0a_base = buf.template GetBuffer<BufferType::ASCEND_L0A>(0);
    l0b_base = buf.template GetBuffer<BufferType::ASCEND_L0B>(0);
#endif
    return;
}

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
__aicore__ __force_inline__ void
PpMatmulEinSum<formatB, transB, swizzleDirect, splitGapA, splitGapC>::GetBaseBlockIdx(uint64_t index, MatCoord &tidx)
{
    uint64_t in_batch_idx = index % (tdim.m * tdim.n);
    if constexpr (swizzleDirect == 0) {  // Zn
        uint64_t tile_block_loop = (tdim.m + swizzle_cnt - 1) / swizzle_cnt;
        uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.n);
        uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.n);

        uint64_t n_row = swizzle_cnt;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = tdim.m - swizzle_cnt * tile_block_idx;
        }
        tidx.m = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
        tidx.n = in_tile_block_idx / n_row;
        if (tile_block_idx % 2 != 0) {
            tidx.n = tdim.n - tidx.n - 1;
        }
    } else if constexpr (swizzleDirect == 1) {  // Nz
        uint64_t tile_block_loop = (tdim.n + swizzle_cnt - 1) / swizzle_cnt;
        uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.m);
        uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.m);

        uint64_t n_col = swizzle_cnt;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = tdim.n - swizzle_cnt * tile_block_idx;
        }
        tidx.m = in_tile_block_idx / n_col;
        tidx.n = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
        if (tile_block_idx % 2 != 0) {
            tidx.m = tdim.m - tidx.m - 1;
        }
    }
    return;
}

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
__aicore__ __force_inline__ void PpMatmulEinSum<formatB, transB, swizzleDirect, splitGapA, splitGapC>::PreloadB()
{
#ifdef __DAV_C220_CUBE__
    uint64_t batch_idx = core_idx / tdim.n / tdim.m;
    uint64_t shuffle_k = en_shuffle_k ? (core_idx % tdim.k) : 0;
    MatCoord tidx{0};
    GetBaseBlockIdx(core_idx, tidx);
    uint64_t offset_b = GetOffsetB(batch_idx, shuffle_k, tidx.n);
    uint64_t n_actual = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
    uint64_t n_round = RoundUp<CONST_16>(n_actual);
    uint64_t k_actual = (shuffle_k == tdim.k - 1) ? k - shuffle_k * k0 : k0;
    uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    CopyTileB(l1_base_b, gm_b[offset_b], k_actual, k_round, n_actual, n_round);
#endif
}

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
__aicore__ __force_inline__ uint64_t PpMatmulEinSum<formatB, transB, swizzleDirect, splitGapA, splitGapC>::GetOffsetB(
    const uint64_t batchIdx, const uint64_t kIdx, const uint64_t nIdx)
{
    if constexpr (formatB == DataFormat::ND) {
        if constexpr (transB) {
            return batchIdx * k * n + nIdx * n0 * k + kIdx * k0;
        } else {
            return batchIdx * k * n + kIdx * k0 * n + nIdx * n0;
        }
    } else {
        if constexpr (transB) {
            return batchIdx * RoundUp<CONST_16>(n) * RoundUp<CONST_16>(k) + kIdx * k0 * RoundUp<CONST_16>(n) +
                   nIdx * n0 * CONST_16;
        } else {
            return batchIdx * RoundUp<CONST_16>(k) * RoundUp<CONST_16>(n) + nIdx * n0 * RoundUp<CONST_16>(k) +
                   kIdx * k0 * CONST_16;
        }
    }
}

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
__aicore__ __force_inline__ void PpMatmulEinSum<formatB, transB, swizzleDirect, splitGapA, splitGapC>::CopyTileA(
    AscendC::LocalTensor<InDtype> &dstTensor, const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t m_actual,
    const uint64_t m_round, const uint64_t k_actual, const uint64_t k_round)
{
    if ((m == 1) || (m_actual == 1)) {
        CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(dstTensor,  // dst
                                                     srcTensor,  // src
                                                     1,          // nTileActual
                                                     CONST_16,   // nTileCeil
                                                     1,          // nVal
                                                     k_actual,   // kTileActual
                                                     k_round,    // kTileCeil
                                                     k);         // dVal
    } else {
        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(dstTensor,                      // dst
                                                     srcTensor,                      // src
                                                     m_actual,                       // nTileActual
                                                     m_round,                        // nTileCeil
                                                     m,                              // nVal
                                                     k_actual,                       // dTileActual
                                                     k_round,                        // dTileCeil
                                                     (k + splitGapA) * batch_size);  // dVal
    }
}

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
__aicore__ __force_inline__ void PpMatmulEinSum<formatB, transB, swizzleDirect, splitGapA, splitGapC>::CopyTileB(
    AscendC::LocalTensor<InDtype> &dstTensor, const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t k_actual,
    const uint64_t k_round, const uint64_t n_actual, const uint64_t n_round)
{
    if constexpr (formatB == DataFormat::ND) {
        if constexpr (transB) {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                                  srcTensor,  // src
                                                  n_actual,   // nTileActual
                                                  n_round,    // nTileCeil
                                                  n,          // nVal
                                                  k_actual,   // dTileActual
                                                  k_round,    // dTileCeil
                                                  k);         // dVal
        } else {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                                  srcTensor,  // src
                                                  k_actual,   // nTileActual
                                                  k_round,    // nTileCeil
                                                  k,          // nVal
                                                  n_actual,   // dTileActual
                                                  n_round,    // dTileCeil
                                                  n);         // dVal
        }
    } else {
        if constexpr (transB) {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,              // dst
                                                  srcTensor,              // src
                                                  n_actual,               // nTileActual
                                                  n_round,                // nTileCeil
                                                  RoundUp<CONST_16>(n),   // nVal
                                                  k_actual,               // dTileActual
                                                  k_round,                // dTileCeil
                                                  RoundUp<CONST_16>(k));  // dVal
        } else {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,              // dst
                                                  srcTensor,              // src
                                                  k_actual,               // nTileActual
                                                  k_round,                // nTileCeil
                                                  RoundUp<CONST_16>(k),   // nVal
                                                  n_actual,               // dTileActual
                                                  n_round,                // dTileCeil
                                                  RoundUp<CONST_16>(n));  // dVal
        }
    }
}

template <DataFormat formatB, bool transB, uint32_t swizzleDirect, uint64_t splitGapA, uint64_t splitGapC>
__aicore__ __force_inline__ void PpMatmulEinSum<formatB, transB, swizzleDirect, splitGapA, splitGapC>::Process()
{
#ifdef __DAV_C220_CUBE__
    if (block_idx >= num_core) {
        WaitFlagDev(MM2OUT);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(BMM3SPLIT);
        return;
    }
    using LocalTensor = AscendC::LocalTensor<InDtype>;

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);

    for (uint64_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += num_core) {
        uint64_t batch_idx = loop_idx / tdim.n / tdim.m;
        MatCoord tidx{0};
        GetBaseBlockIdx(loop_idx, tidx);
        uint64_t offset_a = 0, offset_b = 0, offset_a_next = 0, offset_b_next = 0;
        uint64_t offset_c = tidx.m * m0 * batch_size * (n + splitGapC) + batch_idx * (n + splitGapC) + tidx.n * n0;
        uint64_t m_actual = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
        uint64_t n_actual = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
        uint64_t m_round = RoundUp<CONST_16>(m_actual);
        uint64_t n_round = RoundUp<CONST_16>(n_actual);
        uint64_t mn_max = m_round > n_round ? m_round : n_round;
        uint64_t k_part_len = L0_PINGPONG_BUFFER_LEN / mn_max / CONST_16 * CONST_16;
        uint64_t shuffle_k = en_shuffle_k ? (core_idx % tdim.k) : 0;
        offset_a = tidx.m * m0 * batch_size * (k + splitGapA) + batch_idx * (k + splitGapA) + shuffle_k * k0;
        uint64_t k_actual = (shuffle_k == tdim.k - 1) ? k - shuffle_k * k0 : k0;
        uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;

        LocalTensor l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
        LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
        event_t event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

        if (loop_idx == core_idx) {
            WaitFlagDev(MM2OUT);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(BMM3SPLIT);

            // Copy A from gm to l1 buffer
            WAIT_FLAG(MTE1, MTE2, event_id);
            CopyTileA(l1_buf_a, gm_a[offset_a], m_actual, m_round, k_actual, k_round);
            SET_FLAG(MTE2, MTE1, event_id);

            WAIT_FLAG(MTE1, MTE2, event_id + 2);
            SET_FLAG(MTE2, MTE1, event_id + 2);
        }

        for (tidx.k = 0; tidx.k < tdim.k; ++tidx.k) {
            shuffle_k = en_shuffle_k ? (tidx.k + core_idx) % tdim.k : tidx.k;
            uint64_t k_actual = (shuffle_k == (tdim.k - 1)) ? (k - shuffle_k * k0) : k0;
            uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;
            fdim.k = (k_actual + k_part_len - 1) / k_part_len;

            LocalTensor l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
            LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            if (tidx.k < tdim.k - 1) {
                uint64_t shuffle_k_next = en_shuffle_k ? (core_idx + tidx.k + 1) % tdim.k : (tidx.k + 1);
                offset_a_next =
                    tidx.m * m0 * batch_size * (k + splitGapA) + batch_idx * (k + splitGapA) + shuffle_k_next * k0;
                offset_b_next = GetOffsetB(batch_idx, shuffle_k_next, tidx.n);

                uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                uint64_t k_round_next = (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;

                LocalTensor l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
                LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
                event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                // Preload A from gm to l1 buffer.
                WAIT_FLAG(MTE1, MTE2, event_id_next);
                CopyTileA(l1_buf_a_next, gm_a[offset_a_next], m_actual, m_round, k_actual_next, k_round_next);
                SET_FLAG(MTE2, MTE1, event_id_next);

                // Preload B from gm to l1 buffer.
                WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                CopyTileB(l1_buf_b_next, gm_b[offset_b_next], k_actual_next, k_round_next, n_actual, n_round);
                SET_FLAG(MTE2, MTE1, event_id_next + 2);
            }

            if (tidx.k == tdim.k - 1 && loop_idx + num_core < core_loop) {
                uint64_t b_idx_next = (loop_idx + num_core) / tdim.n / tdim.m;
                MatCoord tidx{0};
                GetBaseBlockIdx(loop_idx + num_core, tidx);
                uint64_t shuffle_k_next = en_shuffle_k ? (core_idx % tdim.k) : 0;
                uint64_t m_actual_next = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
                uint64_t n_actual_next = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
                uint64_t m_round_next = (m_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                uint64_t n_round_next = (n_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                uint64_t k_round_next = (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                offset_a_next =
                    tidx.m * m0 * batch_size * (k + splitGapA) + b_idx_next * (k + splitGapA) + shuffle_k_next * k0;
                offset_b_next = GetOffsetB(b_idx_next, shuffle_k_next, tidx.n);

                LocalTensor l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
                LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
                event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                // Preload A from gm to l1 buffer.
                WAIT_FLAG(MTE1, MTE2, event_id_next);
                CopyTileA(l1_buf_a_next, gm_a[offset_a_next], m_actual_next, m_round_next, k_actual_next, k_round_next);
                SET_FLAG(MTE2, MTE1, event_id_next);

                // Preload B from gm to l1 buffer.
                WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                CopyTileB(l1_buf_b_next, gm_b[offset_b_next], k_actual_next, k_round_next, n_actual_next, n_round_next);
                SET_FLAG(MTE2, MTE1, event_id_next + 2);
            }

            MatCoord fidx{0};
            for (fidx.k = 0; fidx.k < fdim.k; ++fidx.k) {
                uint32_t k0_round = (fidx.k < fdim.k - 1) ? k_part_len : k_round - fidx.k * k_part_len;
                uint32_t k0_actual = (fidx.k < fdim.k - 1) ? k_part_len : k_actual - fidx.k * k_part_len;

                auto mte1_mad_ping_flag = 1 - fidx.k % 2;
                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                LocalTensor l0a_buf = l0a_base[(fidx.k & 0b1) * L0_PINGPONG_BUFFER_LEN];
                LocalTensor l0b_buf = l0b_base[(fidx.k & 0b1) * L0_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (fidx.k == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id);
                }
                WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                if ((m == 1) || (m_actual == 1)) {
                    l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                        l0a_buf,                        // dst
                        l1_buf_a[fidx.k * k_part_len],  // src
                        0,                              // mTileCeil
                        CeilDiv<CONST_256>(k0_round),   // kPartCeil
                        0,                              // mSrcStride
                        1,                              // kSrcStride
                        0,                              // mDstStride
                        0);                             // kDstStride
                } else {
                    LoadCbufToCa(l0a_buf,                                  // l0Tensor
                                 l1_buf_a[fidx.k * k_part_len * m_round],  // l1Tensor
                                 m_round,                                  // mTileCeil
                                 k0_round,                                 // kPartCeil
                                 1,                                        // mSrcStride
                                 m_round / CONST_16,                       // kSrcStride
                                 k0_round / CONST_16,                      // mDstStride
                                 1);                                       // kDstStride
                }
                if (fidx.k == fdim.k - 1) {
                    SET_FLAG(MTE1, MTE2, event_id);
                }

                // *** load matrix B from L1 to L0B
                if (fidx.k == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id + 2);
                }
                if constexpr (transB) {
                    LoadCbufToCb(l0b_buf,                                  // l0Tensor
                                 l1_buf_b[fidx.k * k_part_len * n_round],  // l1Tensor
                                 n_round,                                  // nTileCeil
                                 k0_round,                                 // kPartCeil
                                 1,                                        // nSrcStride
                                 n_round / CONST_16,                       // kSrcStride
                                 1,                                        // nDstStride
                                 k0_round / CONST_16);                     // kDstStride
                } else {
                    LoadCbufToCb(l0b_buf,                                   // l0Tensor
                                 l1_buf_b[fidx.k * k_part_len * CONST_16],  // l1Tensor
                                 n_round,                                   // nTileCeil
                                 k0_round,                                  // kPartCeil
                                 k_round / CONST_16,                        // nSrcStride
                                 1,                                         // kSrcStride
                                 1,                                         // nDstStride
                                 n_round / CONST_16);                       // kDstStride
                }
                if (fidx.k == fdim.k - 1) {
                    SET_FLAG(MTE1, MTE2, event_id + 2);
                }

                SET_FLAG(MTE1, M, mte1_mad_event_id);
                WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                bool init_c = (tidx.k == 0 && fidx.k == 0);
                if (init_c) {
                    WAIT_FLAG(FIX, M, EVENT_ID0);
                }

                Mad(l0c_buf,    // c
                    l0a_buf,    // a
                    l0b_buf,    // b
                    m_actual,   // mTileActual
                    n_actual,   // nTileActual
                    k0_actual,  // kTileActual
                    init_c);    // initC

                AscendC::PipeBarrier<PIPE_M>();
                SET_FLAG(M, MTE1, mte1_mad_event_id);
            }

            ping_flag = 1 - ping_flag;
        }

        SET_FLAG(M, FIX, EVENT_ID0);
        WAIT_FLAG(M, FIX, EVENT_ID0);

        // copy from L0C to gm
        CopyCcToGm(gm_c[offset_c],                 // dst
                   l0c_buf,                        // src
                   m_actual,                       // mTileActual
                   n_actual,                       // nTileActual
                   m_round,                        // mTileCeil
                   (n + splitGapC) * batch_size);  // nActual
        SET_FLAG(FIX, M, EVENT_ID0);
    }

    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(FIX, M, EVENT_ID0);
#endif
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA = DataFormat::ND,
          DataFormat formatB = DataFormat::NZ>
class PpMatmulW8a8
{
    using InDtype = int8_t;
    using OutDtype = half;
    using AccumDtype = int32_t;
    using BiasDtype = int32_t;
    using ScaleDtype = uint64_t;

    template <DataFormat srcFormat, DataFormat dstFormat>
    using CopyGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, InDtype, srcFormat, dstFormat>;
    using LoadCbufToCa = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, transA, DataFormat::ZN, DataFormat::ZZ>;
    using LoadCbufToCb = l1_to_l0_b<ArchType::ASCEND_V220, InDtype, transB, DataFormat::ZN, DataFormat::NZ>;
    using Mmad = mmad<ArchType::ASCEND_V220, InDtype, InDtype, AccumDtype, false>;
    using CopyCcToGm = l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, OutDtype, AccumDtype>;

    static constexpr uint64_t L0_PINGPONG_BUFFER_LEN = 32768;
    static constexpr uint64_t L1_PINGPONG_BUFFER_LEN = 262144;
    static constexpr uint64_t BLOCK_SIZE_16 = 16;
    static constexpr uint64_t BLOCK_SIZE_32 = 32;
    static constexpr uint64_t CUBE_MATRIX_SIZE_512 = 512;
    static constexpr uint64_t FB_BUFF_SIZE = 1024 * 7;
    static constexpr uint64_t SCALE_L1_LEN = 4096;
    static constexpr uint64_t BIAS_L1_LEN = 2048;
    static constexpr uint64_t CONST_4 = 4;
    static constexpr uint64_t CONST_32 = 32;
    static constexpr uint64_t CONST_64 = 64;
    static constexpr uint64_t CONST_128 = 128;

public:
    __aicore__ PpMatmulW8a8() {};

    __aicore__ __force_inline__ void Init(AscendC::GlobalTensor<InDtype> &gm_a, AscendC::GlobalTensor<InDtype> &gm_b,
                                          AscendC::GlobalTensor<BiasDtype> &gm_bias,
                                          AscendC::GlobalTensor<ScaleDtype> &gm_descale,
                                          AscendC::GlobalTensor<OutDtype> &gm_c, MlaTilingData &mlaParams,
                                          uint32_t mode);
    __aicore__ __force_inline__ uint64_t GetOffsetA(const uint64_t batchIdx, const uint64_t mIdx, uint64_t kIdx);
    __aicore__ __force_inline__ uint64_t GetOffsetB(const uint64_t batchIdx, const uint64_t kIdx, uint64_t nIdx);
    __aicore__ __force_inline__ void CopyTileA(const AscendC::LocalTensor<InDtype> &dstTensor,
                                               const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t m_actual,
                                               const uint64_t m_round, const uint64_t k_actual, const uint64_t k_round);
    __aicore__ __force_inline__ void CopyTileB(const AscendC::LocalTensor<InDtype> &dstTensor,
                                               const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t k_actual,
                                               const uint64_t k_round, const uint64_t n_actual, const uint64_t n_round);
    __aicore__ __force_inline__ void Process();
    __aicore__ __force_inline__ void PreloadDoubleWeight();

private:
    __aicore__ __force_inline__ void InitBuffer();
    __aicore__ __force_inline__ void GetBaseBlockIdx(uint64_t index, uint64_t &m_idx, uint64_t &n_idx);

private:
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<InDtype> gm_b;
    AscendC::GlobalTensor<BiasDtype> gm_bias;
    AscendC::GlobalTensor<ScaleDtype> gm_descale;
    AscendC::GlobalTensor<OutDtype> gm_c;

    AscendC::LocalTensor<InDtype> l1_base_a;
    AscendC::LocalTensor<InDtype> l1_base_b;
    AscendC::LocalTensor<InDtype> l0a_base;
    AscendC::LocalTensor<InDtype> l0b_base;
    AscendC::LocalTensor<AccumDtype> l0c_buf;
    AscendC::LocalTensor<BiasDtype> bias_l1;
    AscendC::LocalTensor<ScaleDtype> scale_l1;
    AscendC::LocalTensor<ScaleDtype> scale_fb;

    uint64_t bias_bt{0};
    uint32_t core_num{0};
    uint32_t batch_size{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t m_loop{0};
    uint32_t n_loop{0};
    uint32_t k_loop{0};
    uint32_t core_loop{0};
    uint32_t core_idx{0};
    uint32_t ping_flag{0};
    uint32_t swizzle_cnt{1};
    uint32_t en_shuffle_k{0};
    uint64_t b0mat_pingpong_buffer_len{0};
    bool load_all_Amat_flag{false};
    uint32_t MM1_MM2_mode{0};
};

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ void PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::Init(
    AscendC::GlobalTensor<InDtype> &gm_a, AscendC::GlobalTensor<InDtype> &gm_b,
    AscendC::GlobalTensor<BiasDtype> &gm_bias, AscendC::GlobalTensor<ScaleDtype> &gm_descale,
    AscendC::GlobalTensor<OutDtype> &gm_c, MlaTilingData &mlaParams, uint32_t mode)
{
    this->gm_a = gm_a;
    this->gm_b = gm_b;
    this->gm_bias = gm_bias;
    this->gm_descale = gm_descale;
    this->gm_c = gm_c;
    MM1_MM2_mode = mode;
    if (mode == 0) {
        batch_size = mlaParams.mm1.numBatch;
        m = mlaParams.mm1.m;
        k = mlaParams.mm1.k;
        n = mlaParams.mm1.n;
        m0 = mlaParams.mm1.m0;
        k0 = mlaParams.mm1.k0;
        n0 = mlaParams.mm1.n0;
        m_loop = mlaParams.mm1.mLoop;
        k_loop = mlaParams.mm1.kLoop;
        n_loop = mlaParams.mm1.nLoop;
        core_loop = mlaParams.mm1.coreLoop;
        swizzle_cnt = mlaParams.mm1.swizzleCount;
        en_shuffle_k = mlaParams.mm1.enShuffleK;
        core_num = mlaParams.mm1.blockDim;
        load_all_Amat_flag = mlaParams.mm1.enLoadAllAmat;
        b0mat_pingpong_buffer_len = mlaParams.mm1.b0matPingPongBufferLen;
    } else {
        batch_size = mlaParams.mm2.numBatch;
        m = mlaParams.mm2.m;
        k = mlaParams.mm2.k;
        n = mlaParams.mm2.n;
        m0 = mlaParams.mm2.m0;
        k0 = mlaParams.mm2.k0;
        n0 = mlaParams.mm2.n0;
        m_loop = mlaParams.mm2.mLoop;
        k_loop = mlaParams.mm2.kLoop;
        n_loop = mlaParams.mm2.nLoop;
        core_loop = mlaParams.mm2.coreLoop;
        swizzle_cnt = mlaParams.mm2.swizzleCount;
        en_shuffle_k = mlaParams.mm2.enShuffleK;
        core_num = mlaParams.mm2.blockDim;
        load_all_Amat_flag = mlaParams.mm2.enLoadAllAmat;
        b0mat_pingpong_buffer_len = mlaParams.mm2.b0matPingPongBufferLen;
    }

    core_idx = AscendC::GetBlockIdx();
    ping_flag = 1;

    InitBuffer();
    return;
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ uint64_t PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::GetOffsetA(
    const uint64_t batch_idx, const uint64_t m_idx, uint64_t k_idx)
{
    if constexpr (transA) {
        return batch_idx * m * k + k_idx * k0 * m + m_idx * m0;
    } else {
        return batch_idx * m * k + m_idx * m0 * k + k_idx * k0;
    }
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ uint64_t PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::GetOffsetB(
    const uint64_t batch_idx, const uint64_t k_idx, uint64_t n_idx)
{
    if constexpr (formatB == DataFormat::ND) {
        if constexpr (transB) {
            return batch_idx * k * n + n_idx * n0 * k + k_idx * k0;
        } else {
            return batch_idx * k * n + k_idx * k0 * n + n_idx * n0;
        }
    } else {
        if constexpr (transB) {
            return batch_idx * RoundUp<16>(n) * RoundUp<32>(k) + k_idx * k0 * RoundUp<16>(n) + n_idx * n0 * CONST_32;
        } else {
            return batch_idx * RoundUp<16>(k) * RoundUp<32>(n) + n_idx * n0 * RoundUp<16>(k) + k_idx * k0 * CONST_32;
        }
    }
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ void PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::CopyTileA(
    const AscendC::LocalTensor<InDtype> &dstTensor, const AscendC::GlobalTensor<InDtype> &srcTensor,
    const uint64_t m_actual, const uint64_t m_round, const uint64_t k_actual, const uint64_t k_round)
{
    if ((m == 1) || (m_actual == 1 && !transA)) {
        CopyGmToCbuf<formatA, DataFormat::ND>(dstTensor,  // dst
                                              srcTensor,  // src
                                              1, BLOCK_SIZE_16, 1, k_actual, k_round, k);
    } else {
        if constexpr (transA) {
            CopyGmToCbuf<formatA, DataFormat::NZ>(dstTensor,  // dst
                                                  srcTensor,  // src
                                                  k_actual,   // nTileActual
                                                  k_round,    // nTileCeil
                                                  k,          // nVal
                                                  m_actual,   // dTileActual
                                                  m_round,    // dTileCeil
                                                  m);         // dVal
        } else {
            CopyGmToCbuf<formatA, DataFormat::NZ>(dstTensor,  // dst
                                                  srcTensor,  // src
                                                  m_actual,   // nTileActual
                                                  m_round,    // nTileCeil
                                                  n,          // nVal
                                                  k_actual,   // dTileActual
                                                  k_round,    // dTileCeil
                                                  k);         // dVal
        }
    }
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ void PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::CopyTileB(
    const AscendC::LocalTensor<InDtype> &dstTensor, const AscendC::GlobalTensor<InDtype> &srcTensor,
    const uint64_t k_actual, const uint64_t k_round, const uint64_t n_actual, const uint64_t n_round)
{
    if constexpr (formatB == DataFormat::ND) {
        if constexpr (transB) {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                                  srcTensor,  // src
                                                  n_actual,   // nTileActual
                                                  n_round,    // nTileCeil
                                                  n,          // nVal
                                                  k_actual,   // dTileActual
                                                  k_round,    // dTileCeil
                                                  k);         // dVal
        } else {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,  // dst
                                                  srcTensor,  // src
                                                  k_actual,   // nTileActual
                                                  k_round,    // nTileCeil
                                                  k,          // nVal
                                                  n_actual,   // dTileActual
                                                  n_round,    // dTileCeil
                                                  n);         // dVal
        }
    } else {
        if constexpr (transB) {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,        // dst
                                                  srcTensor,        // src
                                                  n_actual,         // nTileActual
                                                  n_round,          // nTileCeil
                                                  RoundUp<16>(n),   // nVal
                                                  k_actual,         // dTileActual
                                                  k_round,          // dTileCeil
                                                  RoundUp<32>(k));  // dVal
        } else {
            CopyGmToCbuf<formatB, DataFormat::NZ>(dstTensor,        // dst
                                                  srcTensor,        // src
                                                  k_actual,         // nTileActual
                                                  k_round,          // nTileCeil
                                                  RoundUp<16>(k),   // nVal
                                                  n_actual,         // dTileActual
                                                  n_round,          // dTileCeil
                                                  RoundUp<32>(n));  // dVal
        }
    }
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ void PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::InitBuffer()
{
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    l1_base_a = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(SCALE_L1_LEN + BIAS_L1_LEN);

    // try load all A matrix
    uint32_t a_l1_size = RoundUp<BLOCK_SIZE_16>(m) * RoundUp<BLOCK_SIZE_32>(k);
    if (!load_all_Amat_flag) {
        a_l1_size = RoundUp<CUBE_MATRIX_SIZE_512>(m0 * k0);
        if constexpr (transA || !transB) {
            a_l1_size = RoundUp<CUBE_MATRIX_SIZE_512>(RoundUp<BLOCK_SIZE_32>(m0) * k0);
        }
    }

    l1_base_b = l1_base_a[a_l1_size];
    bias_l1 = buf.template GetBuffer<BufferType::ASCEND_CB, BiasDtype>(0);
    scale_l1 = buf.template GetBuffer<BufferType::ASCEND_CB, ScaleDtype>(BIAS_L1_LEN);
    scale_fb.InitBuffer(0, FB_BUFF_SIZE);

    l0a_base = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
    l0b_base = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
    l0c_buf = buf.template GetBuffer<BufferType::ASCEND_L0C, AccumDtype>(0);
    return;
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ void PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::GetBaseBlockIdx(
    uint64_t index, uint64_t &m_idx, uint64_t &n_idx)
{
    uint64_t in_batch_idx = index % (m_loop * n_loop);
    if constexpr (swizzleDir == 0) {  // Zn
        uint64_t tile_block_loop = (m_loop + swizzle_cnt - 1) / swizzle_cnt;
        uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * n_loop);
        uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * n_loop);

        uint64_t n_row = swizzle_cnt;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = m_loop - swizzle_cnt * tile_block_idx;
        }
        m_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
        n_idx = in_tile_block_idx / n_row;
        if ((tile_block_idx & 0b1) != 0) {
            n_idx = n_loop - n_idx - 1;
        }
    } else {  // Nz
        uint64_t tile_block_loop = (n_loop + swizzle_cnt - 1) / swizzle_cnt;
        uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * m_loop);
        uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * m_loop);

        uint64_t n_col = swizzle_cnt;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - swizzle_cnt * tile_block_idx;
        }
        m_idx = in_tile_block_idx / n_col;
        n_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
        if ((tile_block_idx & 0b1) != 0) {
            m_idx = m_loop - m_idx - 1;
        }
    }
    return;
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ void
PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::PreloadDoubleWeight()
{
#ifdef __DAV_C220_CUBE__
    if (core_idx < core_num) {
        uint64_t m_idx = 0;
        uint64_t n_idx = 0;
        GetBaseBlockIdx(core_idx, m_idx, n_idx);
        uint64_t shuffle_k = en_shuffle_k ? core_idx % k_loop : 0;
        uint64_t offset_b = GetOffsetB(0, shuffle_k, n_idx);
        uint64_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
        uint64_t n_round = RoundUp<BLOCK_SIZE_16>(n_actual);
        uint64_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
        uint64_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
        CopyTileB(l1_base_b, gm_b[offset_b], k_actual, k_round, n_actual, n_round);
        if (k_loop > 1) {
            uint64_t shuffle_k = en_shuffle_k ? (core_idx + 1) % k_loop : 1;
            uint64_t offset_b = GetOffsetB(0, shuffle_k, n_idx);
            uint64_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
            uint64_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
            CopyTileB(l1_base_b[b0mat_pingpong_buffer_len], gm_b[offset_b], k_actual, k_round, n_actual, n_round);
        }
    }
#endif
}

template <bool transA, bool transB, bool withBias, uint32_t swizzleDir, DataFormat formatA, DataFormat formatB>
__aicore__ __force_inline__ void PpMatmulW8a8<transA, transB, withBias, swizzleDir, formatA, formatB>::Process()
{
    using LocalTensor = AscendC::LocalTensor<InDtype>;
    if (core_idx >= core_num) {
        if (MM1_MM2_mode == 0) {
            WaitFlagDev(MM1);
        } else if (MM1_MM2_mode == 1) {
            WaitFlagDev(MM2QUANT);
        }
        return;
    }
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(FIX, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID7);
    for (uint64_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += core_num) {
        uint64_t batch_idx = loop_idx / n_loop / m_loop;
        uint64_t m_idx = 0;
        uint64_t n_idx = 0;
        GetBaseBlockIdx(loop_idx, m_idx, n_idx);
        uint64_t offset_a;
        uint64_t offset_b;
        uint64_t offset_bias;
        uint64_t offset_scalar;
        uint64_t offset_a_next;
        uint64_t offset_b_next;
        uint64_t offset_c = batch_idx * m * n + m_idx * m0 * n + n_idx * n0;
        uint64_t m_actual = (m_idx == (m_loop - 1)) ? (m - m_idx * m0) : m0;
        uint64_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
        uint64_t m_round = 0;
        uint64_t n_round = 0;
        uint64_t shuffle_k = en_shuffle_k ? core_idx % k_loop : 0;
        uint64_t m_round_16 = RoundUp<BLOCK_SIZE_16>(m_actual);
        uint64_t m_round_32 = RoundUp<BLOCK_SIZE_32>(m_actual);
        if constexpr (transA) {
            m_round = m_round_32;
        } else {
            m_round = m_round_16;
        }
        if constexpr (transB) {
            n_round = RoundUp<BLOCK_SIZE_16>(n_actual);
        } else {
            n_round = RoundUp<BLOCK_SIZE_32>(n_actual);
        }

        uint64_t mn_max = m_round > n_round ? m_round : n_round;
        uint64_t k_part_len = 0;
        k_part_len = L0_PINGPONG_BUFFER_LEN / mn_max / BLOCK_SIZE_32 * BLOCK_SIZE_32;

        offset_b = GetOffsetB(batch_idx, shuffle_k, n_idx);
        offset_bias = batch_idx * n + n_idx * n0;
        offset_scalar = batch_idx * n + n_idx * n0;

        uint64_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
        uint64_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
        if constexpr (withBias) {
            WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
            gm_to_l1<ArchType::ASCEND_V220, BiasDtype, DataFormat::ND, DataFormat::ND>(bias_l1,               // dst
                                                                                       gm_bias[offset_bias],  // src
                                                                                       1, BLOCK_SIZE_16, 1, n_actual,
                                                                                       n_round, n);
            SET_FLAG(MTE2, MTE1, EVENT_ID6);
        }

        // 3.13 Wait after Scalar
        if (loop_idx == core_idx) {
            if (MM1_MM2_mode == 0) {
                WaitFlagDev(MM1);
            } else if (MM1_MM2_mode == 1) {
                WaitFlagDev(MM2QUANT);
            }
        }

        WAIT_FLAG(MTE1, MTE2, event_id);
        LocalTensor l1_buf_a =
            load_all_Amat_flag ? l1_base_a : (ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN]);
        LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[b0mat_pingpong_buffer_len];
        if (load_all_Amat_flag) {
            if (loop_idx == core_idx) {
                offset_a = GetOffsetA(batch_idx, m_idx, 0);
                uint64_t k_actual_first = k;
                uint64_t k_round_first = RoundUp<BLOCK_SIZE_32>(k_actual_first);
                CopyTileA(l1_buf_a, gm_a[offset_a], m_actual, m_round, k_actual_first, k_round_first);
            }
        } else {
            offset_a = GetOffsetA(batch_idx, m_idx, shuffle_k);
            CopyTileA(l1_buf_a, gm_a[offset_a], m_actual, m_round, k_actual, k_round);
        }
        SET_FLAG(MTE2, MTE1, event_id);

        WAIT_FLAG(MTE1, MTE2, event_id + CONST_2);
        // The first weight matrix block is loaded in advance.
        if (loop_idx != core_idx) {
            CopyTileB(l1_buf_b, gm_b[offset_b], k_actual, k_round, n_actual, n_round);
        }
        SET_FLAG(MTE2, MTE1, event_id + CONST_2);

        WAIT_FLAG(FIX, MTE2, EVENT_ID0);
        gm_to_l1<ArchType::ASCEND_V220, ScaleDtype, DataFormat::ND, DataFormat::ND>(scale_l1,                   // dst
                                                                                    gm_descale[offset_scalar],  // src
                                                                                    1, BLOCK_SIZE_16, 1, n_actual,
                                                                                    n_round, n);
        SET_FLAG(MTE2, FIX, EVENT_ID0);
        WAIT_FLAG(MTE2, FIX, EVENT_ID0);
        l1_to_fb<ArchType::ASCEND_V220, ScaleDtype>(scale_fb,                                           // dst
                                                    scale_l1,                                           // src
                                                    1,                                                  // nBurst
                                                    CeilDiv<CONST_128>(n_actual * sizeof(ScaleDtype)),  // lenBurst
                                                    0,                                                  // srcGap
                                                    0);                                                 // dstGap
        // when move scalar form L1 to fifpipe end, can move A/B from gm to L1
        SET_FLAG(FIX, MTE2, EVENT_ID0);

        for (uint64_t k_idx = 0; k_idx < k_loop; k_idx++) {
            shuffle_k = en_shuffle_k ? (k_idx + core_idx) % k_loop : k_idx;
            uint32_t k_actual = (shuffle_k == (k_loop - 1)) ? (k - shuffle_k * k0) : k0;
            uint32_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
            uint32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            // --------- load whole A in l1a addr change -------------
            LocalTensor l1_buf_a = load_all_Amat_flag ? (l1_base_a[k_idx * m0 * k0 * sizeof(int8_t)])
                                                      : (ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN]);
            LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[b0mat_pingpong_buffer_len];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            if (k_idx < k_loop - 1) {
                uint64_t shuffle_k_next = en_shuffle_k ? (core_idx + k_idx + 1) % k_loop : k_idx + 1;

                offset_b_next = GetOffsetB(batch_idx, shuffle_k_next, n_idx);
                uint32_t k_actual_next = (shuffle_k_next == (k_loop - 1)) ? (k - shuffle_k_next * k0) : k0;
                uint32_t k_round_next = RoundUp<BLOCK_SIZE_32>(k_actual_next);

                LocalTensor l1_buf_a_next =
                    load_all_Amat_flag ? l1_base_a : ((1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN]);
                LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[b0mat_pingpong_buffer_len];
                auto event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                WAIT_FLAG(MTE1, MTE2, event_id_next);
                if (!load_all_Amat_flag) {
                    offset_a_next = GetOffsetA(batch_idx, m_idx, shuffle_k_next);
                    CopyTileA(l1_buf_a_next, gm_a[offset_a_next], m_actual, m_round, k_actual_next, k_round_next);
                }
                SET_FLAG(MTE2, MTE1, event_id_next);

                WAIT_FLAG(MTE1, MTE2, event_id_next + CONST_2);
                if (loop_idx != core_idx || k_idx != 0) {  // The second weight matrix is preloaded.
                    CopyTileB(l1_buf_b_next, gm_b[offset_b_next], k_actual_next, k_round_next, n_actual, n_round);
                }
                SET_FLAG(MTE2, MTE1, event_id_next + CONST_2);
            }

            for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                uint32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                uint32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<InDtype> l0a_buf = l0a_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_LEN];
                AscendC::LocalTensor<InDtype> l0b_buf = l0b_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id);
                }
                WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                if ((m == 1) || (m_actual == 1 && !transA)) {
                    l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                        l0a_buf, l1_buf_a[k_part_idx * k_part_len],
                        0,                                        // mTileCeil
                        CeilDiv<CUBE_MATRIX_SIZE_512>(k0_round),  // kPartCeil
                        0,                                        // mSrcStride
                        1,                                        // kSrcStride
                        0,                                        // mDstStride
                        0);                                       // kDstStride
                } else {
                    if constexpr (transA) {
                        LoadCbufToCa(l0a_buf,                                            // l0Tensor
                                     l1_buf_a[k_part_idx * k_part_len * BLOCK_SIZE_32],  // l1Tensor
                                     m_round,                                            // mTileCeil
                                     k0_round,                                           // kPartCeil
                                     k_round / BLOCK_SIZE_16,                            // mSrcStride
                                     1,                                                  // kSrcStride
                                     k0_round / BLOCK_SIZE_32,                           // mDstStride
                                     1);                                                 // kDstStride
                    } else {
                        LoadCbufToCa(l0a_buf,                                      // l0Tensor
                                     l1_buf_a[k_part_idx * k_part_len * m_round],  // l1Tensor
                                     m_round,                                      // mTileCeil
                                     k0_round,                                     // kPartCeil
                                     1,                                            // mSrcStride
                                     m_round / BLOCK_SIZE_16,                      // kSrcStride
                                     k0_round / BLOCK_SIZE_32,                     // mDstStride
                                     1);                                           // kDstStride
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id + CONST_2);
                }
                if constexpr (transB) {
                    LoadCbufToCb(l0b_buf,                                      // l0Tensor
                                 l1_buf_b[k_part_idx * k_part_len * n_round],  // l1Tensor
                                 n_round,                                      // nTileCeil
                                 k0_round,                                     // kPartCeil
                                 1,                                            // nSrcStride
                                 n_round / BLOCK_SIZE_16,                      // kSrcStride
                                 1,                                            // nDstStride
                                 k0_round / BLOCK_SIZE_32);                    // kDstStride
                } else {
                    LoadCbufToCb(l0b_buf,                                            // l0Tensor
                                 l1_buf_b[k_part_idx * k_part_len * BLOCK_SIZE_32],  // l1Tensor
                                 n_round,                                            // nTileCeil
                                 k0_round,                                           // kPartCeil
                                 k_round / BLOCK_SIZE_16,                            // nSrcStride
                                 1,                                                  // kSrcStride
                                 1,                                                  // nDstStride
                                 n_round / BLOCK_SIZE_16);                           // kDstStride
                }
                if (k_part_idx == k_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id + CONST_2);
                }

                SET_FLAG(MTE1, M, mte1_mad_event_id);
                WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                bool sp_flag = (m != 1 && m_actual == 1 && transA);
                if (init_c) {
                    WAIT_FLAG(FIX, M, EVENT_ID0);
                }
                if (init_c) {
                    if constexpr (withBias) {
                        WAIT_FLAG(MTE2, MTE1, EVENT_ID6);
                        l1_to_bt<ArchType::ASCEND_V220, BiasDtype>(
                            bias_bt,                                          // dst
                            bias_l1,                                          // src
                            0,                                                // convControl
                            1,                                                // nBurst
                            CeilDiv<CONST_64>(n_actual * sizeof(BiasDtype)),  // lenBurst
                            0,                                                // srcGap
                            0);                                               // dstGap
                        SET_FLAG(MTE1, MTE2, EVENT_ID7);  // bias ready, mte2 can begin move A/B or scale
                        SET_FLAG(MTE1, M, EVENT_ID7);     // bias ready, mmad can begin
                        WAIT_FLAG(MTE1, M, EVENT_ID7);    // wait move bias from L1 to BT
                        Mmad(l0c_buf, l0a_buf, l0b_buf, ((uint64_t)bias_bt),
                             sp_flag ? m_round_16 : m_actual,  // m
                             n_actual,                         // n
                             k0_actual,                        // k
                             0);                               // cmatrixInitVal
                    } else {
                        Mmad(l0c_buf, l0a_buf, l0b_buf,
                             sp_flag ? m_round_16 : m_actual,  // m
                             n_actual,                         // n
                             k0_actual,                        // k
                             1);                               // cmatrixInitVal
                    }
                } else {
                    Mmad(l0c_buf, l0a_buf, l0b_buf,
                         sp_flag ? m_round_16 : m_actual,  // m
                         n_actual,                         // n
                         k0_actual,                        // k
                         0);                               // cmatrixInitVal
                }
                AscendC::PipeBarrier<PIPE_M>();
                SET_FLAG(M, MTE1, mte1_mad_event_id);
            }

            ping_flag = 1 - ping_flag;
        }
        SET_FLAG(M, FIX, EVENT_ID0);
        WAIT_FLAG(M, FIX, EVENT_ID0);
        AscendC::PipeBarrier<PIPE_FIX>();
        SetFpc<ScaleDtype>(scale_fb, false);
        // copy from L0C to gm
        CopyCcToGm(gm_c[offset_c],  // dst
                   l0c_buf,         // src
                   m_actual,        // MSize
                   n_actual,        // NSize
                   m_round_16,      // srcStride
                   n);              // dstStride_dst_D
        SET_FLAG(FIX, M, EVENT_ID0);
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
}
#endif

template <int8_t cacheMode, DataFormat weightFormat1, DataFormat weightFormat2, DataFormat weightFormat3>
class MLAOperation
{
    using qOutDtype = typename std::conditional_t<cacheMode == CACHE_MODE_INT8_NZCACHE, int8_t, half>;
    using kNopeDtype = typename std::conditional_t<cacheMode == CACHE_MODE_INT8_NZCACHE, int8_t, half>;

public:
    __aicore__ inline MLAOperation(const MlaTilingData &mlaParams_, GM_ADDR tilingGm)
    {
        blockIdx = AscendC::GetBlockIdx();
#ifdef __DAV_C220_VEC__
        sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());
#endif
        vectorBlockIdx = (blockIdx / 2) * 2 + sub_block_idx;
        this->n = mlaParams_.n;
        this->num_core_ = mlaParams_.rmsNumCore1;
        this->num_col_1 = mlaParams_.rmsNumCol1;
        this->num_col_2 = mlaParams_.rmsNumCol2;
        this->num_row = mlaParams_.n;
        this->epsilon_ = 1e-6;
        this->mlaParams = mlaParams_;
    }

    __aicore__ inline void Init(GM_ADDR hiddenStateGm, GM_ADDR quantScale1Gm,
                                GM_ADDR quantOffset1Gm, GM_ADDR wdqkvGm, GM_ADDR bias1Gm, GM_ADDR gamma2Gm,
                                GM_ADDR beta2Gm, GM_ADDR quantScale2Gm, GM_ADDR quantOffset2Gm, GM_ADDR gamma3Gm,
                                GM_ADDR sin1Gm, GM_ADDR cos1Gm, GM_ADDR sin2Gm, GM_ADDR cos2Gm, GM_ADDR keycacheGm,
                                GM_ADDR slotMappingGm, GM_ADDR wuqGm, GM_ADDR bias2Gm, GM_ADDR wukGm,
                                GM_ADDR descale1Gm, GM_ADDR descale2Gm, GM_ADDR gmCtkvScale, GM_ADDR gmQnopeScale,
                                GM_ADDR qGm, GM_ADDR keycacheOutGm, GM_ADDR qGm2, GM_ADDR keycacheOutGm2, GM_ADDR s1Gm,
                                GM_ADDR s2Gm, GM_ADDR s3Gm)
    {
        s1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(s1Gm));
        wdqkvGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(wdqkvGm));
        bias1gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bias1Gm));
        descale1gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(descale1Gm));
        s3GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(s3Gm));

#ifdef __DAV_C220_CUBE__
        mm_w8a8_1.Init(s1GmTensor, wdqkvGmTensor, bias1gmTensor, descale1gmTensor, s3GmTensor, mlaParams, 0);
        mm_w8a8_1.PreloadDoubleWeight();
#endif
        hiddenStateGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(hiddenStateGm));
        quantScale1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(quantScale1Gm));
        quantOffset1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(quantOffset1Gm));

        gamma2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gamma2Gm));
        quantScale2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(quantScale2Gm));
        quantScale3GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmCtkvScale));
        quantOffset2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(quantOffset2Gm));
        gamma3GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gamma3Gm));
        sin1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(sin1Gm));
        cos1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cos1Gm));
        sin2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(sin2Gm));
        cos2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cos2Gm));
        keycacheGmTensor1.SetGlobalBuffer(reinterpret_cast<__gm__ kNopeDtype *>(keycacheOutGm));
        keycacheGmTensor2.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(keycacheOutGm2));
        slotMappingGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotMappingGm));
        wuqGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(wuqGm));
        wukGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wukGm));
        descale2gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(descale2Gm));
        s2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(s2Gm));
        qGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ qOutDtype *>(qGm));
        qGmTensor2.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qGm2));
        bias2gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bias2Gm));

        beta2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(beta2Gm));
#ifdef __DAV_C220_CUBE__
        mm_w8a8_2.Init(s1GmTensor, wuqGmTensor, bias2gmTensor, descale2gmTensor, s2GmTensor, mlaParams, 1);
        if constexpr (cacheMode == CACHE_MODE_INT8_NZCACHE) {
            mm_ein_sum.Init(s2Gm, wukGm, s1Gm, mlaParams);
        } else {
            mm_ein_sum.Init(s2Gm, wukGm, qGm, mlaParams);
        }
#endif

#ifdef __DAV_C220_VEC__
        // rmsnormQuant
        row_work = (num_row + num_core_ - 1) / num_core_;
        row_work_ = 0;
        uint32_t need_core = (num_row + row_work - 1) / row_work;
        if (vectorBlockIdx < need_core - 1) {
            row_work_ = row_work;
        } else if (vectorBlockIdx == need_core - 1) {
            row_work_ = num_row - (need_core - 1) * row_work;
        } else {
            row_work_ = 0;
        }
        this->splitN = mlaParams.perTaskNum;
        Quant1.Init(quantScale1GmTensor, quantOffset1GmTensor, hiddenStateGmTensor,
                    s1GmTensor, 0, num_col_1, 0.0001395089285,
                    vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_1,
                    vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_1, row_work_, mlaParams);

        rmsNormQuant2.Init(gamma2GmTensor, beta2GmTensor, quantScale2GmTensor, quantOffset2GmTensor, s3GmTensor,
                           s1GmTensor, SPLIT_SIZE_ONE, num_col_2, 0.000651041666,
                           vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2,
                           vectorBlockIdx * static_cast<uint64_t>(row_work) * SPLIT_SIZE_TWO, row_work_, mlaParams);
        ropeFp16.RopeInit(s2GmTensor, cos2GmTensor, sin2GmTensor, qGmTensor, qGmTensor2, mlaParams);
        einSumQuant.Init(s1Gm, gmQnopeScale, qGm, mlaParams);
        ubTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(0);
        ub8Tensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(0);
        ub32Tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
#endif
    }

    __aicore__ inline void ProcessCube();

    __aicore__ inline void ProcessVector();

private:
    constexpr static uint32_t C0_SIZE = 16;
    constexpr static uint32_t I8_C0_SIZE = 32;

    template <class T1>
    __aicore__ inline void RmsNormAndRopeConvergence1(
        const AscendC::LocalTensor<T1> &srcTensor, const AscendC::LocalTensor<T1> &gammaTensor,
        const AscendC::LocalTensor<T1> &sinTensor, const AscendC::LocalTensor<T1> &cosTensor,
        const AscendC::LocalTensor<int32_t> &slotMappingTensor, const uint32_t sN,
        const AscendC::LocalTensor<float> &rmsNormTensor, const AscendC::LocalTensor<float> &gammaFp32,
        const AscendC::LocalTensor<float> &ropeKTensor, const AscendC::LocalTensor<float> &ropeKRevertTensor,
        const AscendC::LocalTensor<float> &calTensor, const AscendC::LocalTensor<T1> &outTmpTensor,
        AscendC::LocalTensor<half> &tmpfp16, AscendC::LocalTensor<int8_t> &int8OutTensor, float quantScale3)
    {
        int64_t slotMapGmOffset = vectorBlockIdx * row_work;
        AscendC::DataCopy(gammaTensor, gamma3GmTensor, SPLIT_RMSNRORM_SIZE_ONE);
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Cast(gammaFp32, gammaTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_ONE);
        AscendC::DataCopyPad(slotMappingTensor, slotMappingGmTensor[slotMapGmOffset],
                             AscendC::DataCopyExtParams(1, sN * sizeof(int32_t), 0, 0, 0),
                             AscendC::DataCopyPadExtParams<int32_t>(false, 0, 8 - sN % 8, 0));
        SET_FLAG(MTE2, V, EVENT_ID2);
        WAIT_FLAG(MTE2, V, EVENT_ID2);
        SET_FLAG(MTE2, S, EVENT_ID2);
        WAIT_FLAG(MTE2, S, EVENT_ID2);
        for (uint64_t loop = 0; loop < sN; ++loop) {
            uint64_t offset = vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2 + loop * MM1_OUT_SIZE;
            int64_t slotValue = static_cast<int64_t>(slotMappingTensor.GetValue(loop));
            if (slotValue == -1) {
                continue;
            }
            AscendC::DataCopy(srcTensor, s3GmTensor[offset], SPLIT_SIZE_ONE);
            AscendC::DataCopy(sinTensor, sin1GmTensor[(row_work * vectorBlockIdx + loop) * SPLIT_RMSNRORM_SIZE_TWO],
                              SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::DataCopy(cosTensor, cos1GmTensor[(row_work * vectorBlockIdx + loop) * SPLIT_RMSNRORM_SIZE_TWO],
                              SPLIT_RMSNRORM_SIZE_TWO);
            SET_FLAG(MTE2, V, EVENT_ID0);
            // ND
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(SPLIT_SIZE_ONE);
            uint64_t cacheStart1 = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(SPLIT_RMSNRORM_SIZE_ONE);
            uint64_t cacheStart2 = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(SPLIT_RMSNRORM_SIZE_TWO);
            // NZ
            uint32_t outer_idx = slotValue / 128;
            uint32_t inner_idx = slotValue % 128;
            SET_FLAG(S, MTE3, EVENT_ID0);
            /* RmsNorm start */
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            Cast(rmsNormTensor, srcTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(calTensor, rmsNormTensor, rmsNormTensor, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSumCustom(calTensor[SPLIT_RMSNRORM_SIZE_ONE], calTensor, calTensor[SPLIT_RMSNRORM_SIZE_ONE * 2],
                            SPLIT_RMSNRORM_SIZE_ONE);
            SET_FLAG(V, S, EVENT_ID1);
            WAIT_FLAG(V, S, EVENT_ID1);
            float rms = sqrt(calTensor.GetValue(SPLIT_RMSNRORM_SIZE_ONE) / SPLIT_RMSNRORM_SIZE_ONE + epsilon_);
            SET_FLAG(S, V, EVENT_ID1);
            WAIT_FLAG(S, V, EVENT_ID1);
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(calTensor, rms, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            Div(calTensor, rmsNormTensor, calTensor, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(rmsNormTensor, gammaFp32, calTensor, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outTmpTensor, rmsNormTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (cacheMode == CACHE_MODE_INT8_NZCACHE) {
                // quant
                Muls(rmsNormTensor, rmsNormTensor, quantScale3, SPLIT_RMSNRORM_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
                CastFrom32To16(tmpfp16, rmsNormTensor, SPLIT_RMSNRORM_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
                CastFromF16ToI8(int8OutTensor, tmpfp16, -128, SPLIT_RMSNRORM_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::PipeBarrier<PIPE_V>();
                if (std::is_same<T1, __bf16>::value) {
                    Cast(outTmpTensor, rmsNormTensor, AscendC::RoundMode::CAST_RINT, SPLIT_RMSNRORM_SIZE_ONE);
                } else {
                    Cast(outTmpTensor, rmsNormTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_ONE);
                }
            }
            /* RmsNorm end */
            // /* Rope K start */
            uint64_t revertOffset = SPLIT_RMSNRORM_SIZE_TWO / 2;
            Cast(ropeKTensor, srcTensor[SPLIT_RMSNRORM_SIZE_ONE], AscendC::RoundMode::CAST_NONE,
                 SPLIT_RMSNRORM_SIZE_TWO);
            Cast(ropeKRevertTensor[revertOffset], srcTensor[SPLIT_RMSNRORM_SIZE_ONE], AscendC::RoundMode::CAST_NONE,
                 revertOffset);
            Cast(ropeKRevertTensor, srcTensor[SPLIT_RMSNRORM_SIZE_ONE + revertOffset], AscendC::RoundMode::CAST_NONE,
                 revertOffset);
            Duplicate(calTensor, static_cast<float>(-1), revertOffset);
            Duplicate(calTensor[revertOffset], static_cast<float>(1), revertOffset);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(calTensor[SPLIT_RMSNRORM_SIZE_TWO], cosTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_TWO);
            Cast(calTensor[SPLIT_RMSNRORM_SIZE_TWO * 2], sinTensor, AscendC::RoundMode::CAST_NONE,
                 SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(ropeKTensor, calTensor[SPLIT_RMSNRORM_SIZE_TWO], ropeKTensor, SPLIT_RMSNRORM_SIZE_TWO);
            Mul(ropeKRevertTensor, calTensor[SPLIT_RMSNRORM_SIZE_TWO * 2], ropeKRevertTensor, SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(ropeKRevertTensor, calTensor, ropeKRevertTensor, SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            Add(ropeKRevertTensor, ropeKTensor, ropeKRevertTensor, SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], ropeKRevertTensor, AscendC::RoundMode::CAST_NONE,
                 SPLIT_RMSNRORM_SIZE_TWO);
            /* Rope K end */
            // reshapeAndcache
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(S, MTE3, EVENT_ID0);
            if constexpr (cacheMode == CACHE_MODE_KVCACHE) {
                DataCopy(keycacheGmTensor1[cacheStart], outTmpTensor, SPLIT_SIZE_ONE);
            } else if constexpr (cacheMode == CACHE_MODE_INT8_NZCACHE) {
                // NZ
                int64_t cacheSatartI8Nz1 = outer_idx * 128 * 512 + inner_idx * I8_C0_SIZE;
                uint64_t cacheSatartNz2 = outer_idx * 128 * 64 + inner_idx * C0_SIZE;
                AscendC::DataCopyExtParams outExt;
                // nope:int8 nz
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_ONE / I8_C0_SIZE;
                outExt.blockLen = I8_C0_SIZE * sizeof(int8_t);
                outExt.srcStride = 0;
                outExt.dstStride = (128 * I8_C0_SIZE - I8_C0_SIZE) * sizeof(int8_t);
                DataCopyPad(keycacheGmTensor1[cacheSatartI8Nz1], int8OutTensor, outExt);
                // rope:T1 nz
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_TWO / C0_SIZE;
                outExt.blockLen = C0_SIZE * sizeof(T1);
                outExt.srcStride = 0;
                outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
                DataCopyPad(keycacheGmTensor2[cacheSatartNz2], outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], outExt);
            } else if constexpr (cacheMode == CACHE_MODE_NZCACHE) {
                uint64_t cacheSatartNz1 = outer_idx * 128 * 512 + inner_idx * C0_SIZE;
                uint64_t cacheSatartNz2 = outer_idx * 128 * 64 + inner_idx * C0_SIZE;
                // nope:T1 nz
                AscendC::DataCopyExtParams outExt;
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_ONE / C0_SIZE;
                outExt.blockLen = C0_SIZE * sizeof(T1);
                outExt.srcStride = 0;
                outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
                DataCopyPad(keycacheGmTensor1[cacheSatartNz1], outTmpTensor, outExt);
                // rope:T1 nz
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_TWO / C0_SIZE;
                outExt.blockLen = C0_SIZE * sizeof(T1);
                outExt.srcStride = 0;
                outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
                DataCopyPad(keycacheGmTensor2[cacheSatartNz2], outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], outExt);
            } else {
                // keycache1
                DataCopy(keycacheGmTensor1[cacheStart1], outTmpTensor, SPLIT_RMSNRORM_SIZE_ONE);
                // keycache2
                DataCopy(keycacheGmTensor2[cacheStart2], outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE],
                         SPLIT_RMSNRORM_SIZE_TWO);
            }
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        }
    }

private:
    uint32_t n;
    uint32_t splitN;
    uint32_t rotaryCoeff;
    uint32_t blockIdx;
    uint32_t sub_block_idx;
    uint32_t vectorBlockIdx;
    uint32_t blockOffset;
    uint32_t perTaskNum;
    uint32_t resTaskNum;
    MlaTilingData mlaParams;

    // rmsnormQuant
    uint32_t num_core_;
    uint32_t num_col_1;
    uint32_t num_col_2;
    float epsilon_;
    uint32_t num_row;
    uint32_t quantMin_;
    uint32_t row_work;
    uint32_t row_work_;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<half> ubTensor;
    AscendC::LocalTensor<int8_t> ub8Tensor;
    AscendC::LocalTensor<float> ub32Tensor;

    AscendC::GlobalTensor<half> hiddenStateGmTensor;

    AscendC::GlobalTensor<half> quantScale1GmTensor;
    AscendC::GlobalTensor<int8_t> quantOffset1GmTensor;

    AscendC::GlobalTensor<int8_t> wdqkvGmTensor;
    AscendC::GlobalTensor<half> gamma2GmTensor;
    AscendC::GlobalTensor<half> quantScale2GmTensor;
    AscendC::GlobalTensor<half> quantScale3GmTensor;
    AscendC::GlobalTensor<int8_t> quantOffset2GmTensor;
    AscendC::GlobalTensor<half> gamma3GmTensor;
    AscendC::GlobalTensor<half> sin1GmTensor;
    AscendC::GlobalTensor<half> cos1GmTensor;
    AscendC::GlobalTensor<half> sin2GmTensor;
    AscendC::GlobalTensor<half> cos2GmTensor;
    AscendC::GlobalTensor<kNopeDtype> keycacheGmTensor1;
    AscendC::GlobalTensor<half> keycacheGmTensor2;
    AscendC::GlobalTensor<int32_t> slotMappingGmTensor;
    AscendC::GlobalTensor<int8_t> wuqGmTensor;
    AscendC::GlobalTensor<half> wukGmTensor;

    AscendC::GlobalTensor<qOutDtype> qGmTensor;
    AscendC::GlobalTensor<half> qGmTensor2;
    AscendC::GlobalTensor<int8_t> s1GmTensor;
    AscendC::GlobalTensor<half> s2GmTensor;
    AscendC::GlobalTensor<half> s3GmTensor;
    AscendC::GlobalTensor<uint64_t> descale1gmTensor;
    AscendC::GlobalTensor<uint64_t> descale2gmTensor;
    AscendC::GlobalTensor<half> beta2GmTensor;

    AscendC::GlobalTensor<int32_t> bias1gmTensor;
    AscendC::GlobalTensor<int32_t> bias2gmTensor;

#ifdef __DAV_C220_CUBE__
    PpMatmulW8a8<false, true, true, 0, DataFormat::ND, weightFormat1> mm_w8a8_1;
    PpMatmulW8a8<false, true, true, 1, DataFormat::ND, weightFormat2> mm_w8a8_2;
    static constexpr uint64_t splitGapC = cacheMode == CACHE_MODE_KVCACHE ? CONST_64 : CONST_0;
    PpMatmulEinSum<weightFormat3, false, 0, CONST_64, splitGapC> mm_ein_sum;
#endif

#ifdef __DAV_C220_VEC__
    Quant<half, true, false> Quant1;
    RmsNormQuant<half, true, false> rmsNormQuant2;
    RopeFp16<half, half, qOutDtype, cacheMode> ropeFp16;
    EinSumQuant<half, half> einSumQuant;
#endif
};

template <int8_t cacheMode, DataFormat weightFormat1, DataFormat weightFormat2, DataFormat weightFormat3>
__aicore__ inline void MLAOperation<cacheMode, weightFormat1, weightFormat2, weightFormat3>::ProcessCube()
{
#ifdef __DAV_C220_CUBE__
    mm_w8a8_1.Process();
    FftsCrossCoreSync<PIPE_FIX, 0>(RMSNORMQUANT2);
    WaitFlagDev(RMSNORMQUANT2);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(MM1QUANT);

    mm_w8a8_2.PreloadDoubleWeight();
    mm_w8a8_2.Process();
    FftsCrossCoreSync<PIPE_FIX, 0>(MM2OUT);
    mm_ein_sum.PreloadB();
    mm_ein_sum.Process();
    if constexpr (cacheMode == CACHE_MODE_INT8_NZCACHE) {
        FftsCrossCoreSync<PIPE_FIX, 0>(EINSUMOUT);
        WaitFlagDev(EINSUMOUT);
        FftsCrossCoreSync<PIPE_FIX, 0x2>(EINSUMQUANT);
    }
#endif
}

template <int8_t cacheMode, DataFormat weightFormat1, DataFormat weightFormat2, DataFormat weightFormat3>
__aicore__ inline void MLAOperation<cacheMode, weightFormat1, weightFormat2, weightFormat3>::ProcessVector()
{
#ifdef __DAV_C220_VEC__
    if (row_work_ != 0) {
        uint32_t num_col_align_int8 = (num_col_1 + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        uint32_t num_col_align_f16 = (num_col_1 + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        uint32_t num_col_align_f32 = (num_col_1 + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        AscendC::LocalTensor<half> input_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(0);
        AscendC::LocalTensor<half> gamma_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(HIDDTEN_STATE * 2);
        AscendC::LocalTensor<half> beta_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, half>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2);
        AscendC::LocalTensor<half> scale_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, half>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2);
        AscendC::LocalTensor<int8_t> offset_tensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 32);
        AscendC::LocalTensor<float> res1_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, float>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 64);
        AscendC::LocalTensor<float> res3_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(
            HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 64 + num_col_align_f32 * 4);
        AscendC::LocalTensor<int8_t> output_tensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 64 + num_col_align_f32 * 4 +
            BUF_FACTOR * num_col_align_f32 * 4 + 32);
        Quant1.Launch(output_tensor, input_tensor, gamma_tensor, beta_tensor, scale_tensor, offset_tensor, res1_tensor,
                      res3_tensor);
    }
    FftsCrossCoreSync<PIPE_MTE3, 0>(QUANT1);
    WaitFlagDev(QUANT1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(MM1);

    WaitFlagDev(MM1QUANT);
    if (row_work_ != 0) {
        uint32_t num_col_align_int8 = (num_col_2 + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        uint32_t num_col_align_f16 = (num_col_2 + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        uint32_t num_col_align_f32 = (num_col_2 + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        AscendC::LocalTensor<half> input_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(0);
        AscendC::LocalTensor<half> gamma_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(MM1_OUT_SIZE * 2);
        AscendC::LocalTensor<half> beta_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, half>(MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2);
        AscendC::LocalTensor<half> scale_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, half>(MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2);
        AscendC::LocalTensor<int8_t> offset_tensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 32);
        AscendC::LocalTensor<float> res1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64);
        AscendC::LocalTensor<float> res3_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64 + num_col_align_f32 * 4);
        AscendC::LocalTensor<int8_t> output_tensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64 + num_col_align_f32 * 4 +
            BUF_FACTOR * num_col_align_f32 * 4 + 32);
        rmsNormQuant2.Launch(output_tensor, input_tensor, gamma_tensor, beta_tensor, scale_tensor, offset_tensor,
                             res1_tensor, res3_tensor);
    }
    FftsCrossCoreSync<PIPE_MTE3, 0>(MM2);
    WaitFlagDev(MM2);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(MM2QUANT);

    if (row_work_ != 0) {
        AscendC::LocalTensor<half> input_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(0);
        AscendC::LocalTensor<half> gamma_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(MM1_OUT_SIZE * 2);
        AscendC::LocalTensor<half> sin_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, half>(MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2);
        AscendC::LocalTensor<half> cos_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 2);
        AscendC::LocalTensor<int32_t> slotMapping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4);
        int32_t rms3_ub_offset =
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4 + 4096 * 32;
        AscendC::LocalTensor<float> tmp32_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(rms3_ub_offset);

        int32_t out_ub_offset = MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4 +
                                4096 * 32 + SPLIT_RMSNRORM_SIZE_ONE * 3 * 4 + SPLIT_RMSNRORM_SIZE_TWO * 2 * 4;
        AscendC::LocalTensor<half> temp_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(out_ub_offset);

        AscendC::LocalTensor<half> tmpfp16;
        AscendC::LocalTensor<int8_t> int8OutTensor;
        float scale3 = 0;
        if constexpr (cacheMode == CACHE_MODE_INT8_NZCACHE) {
            // quantScale3
            AscendC::LocalTensor<half> quantScaleTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(rms3_ub_offset);
            AscendC::LocalTensor<float> floatQuantScaleTensor =
                buf.GetBuffer<BufferType::ASCEND_UB, float>(rms3_ub_offset + 32);
            // int8out
            tmpfp16 = buf.GetBuffer<BufferType::ASCEND_UB, half>(rms3_ub_offset +
                                                                 SPLIT_RMSNRORM_SIZE_ONE * sizeof(float) * 2);
            int8OutTensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(out_ub_offset);
            AscendC::DataCopy(quantScaleTensor, quantScale3GmTensor, AscendC::DataCopyParams(1, 1, 0, 0));
            SET_FLAG(MTE2, V, EVENT_ID1);
            WAIT_FLAG(MTE2, V, EVENT_ID1);
            Cast(floatQuantScaleTensor, quantScaleTensor, AscendC::RoundMode::CAST_NONE, 1);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID1);
            scale3 = 1 / (float)(floatQuantScaleTensor.GetValue(0));
        }

        RmsNormAndRopeConvergence1<half>(
            input_tensor,        // n * 576
            gamma_tensor,        // gamma
            sin_tensor,          // sin
            cos_tensor,          // cons
            slotMapping_tensor,  // slotMapping
            row_work_, tmp32_tensor, tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE],
            tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_ONE],
            tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_TWO],
            tmp32_tensor[SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_ONE + SPLIT_RMSNRORM_SIZE_TWO +
                         SPLIT_RMSNRORM_SIZE_TWO],
            temp_tensor, tmpfp16, int8OutTensor, scale3);
    }
    WaitFlagDev(BMM3SPLIT);
    ropeFp16.Process();

    if constexpr (cacheMode == CACHE_MODE_INT8_NZCACHE) {
        WaitFlagDev(EINSUMQUANT);
        einSumQuant.Process();
        PIPE_BARRIER(ALL);
    }
#endif
}

}  // namespace MLAPO_FP16
