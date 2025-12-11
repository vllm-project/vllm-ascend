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

namespace MLAPO_BF16_NQ {
template <typename QkDtype, typename CosDtype, typename QOutDtype, int8_t CacheMode>
class RopeFp16
{
public:
    __aicore__ inline RopeFp16() : blockIdx_(AscendC::GetBlockIdx()) {}

    __aicore__ inline void RopeInit(GM_ADDR qGm, AscendC::GlobalTensor<CosDtype> &cosGm,
                                    AscendC::GlobalTensor<CosDtype> &sinGm,
                                    AscendC::GlobalTensor<QOutDtype> &outRopeConcatGm,
                                    AscendC::GlobalTensor<QkDtype> &outRopeConcatGm2, MlaTilingData &ropeConcatParams)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype *>(qGm));
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
        if (blockIdx_ >= realCore) {
            return;
        }
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

            if (startSinCosHeadIndex < endHead) {
                uint64_t startSinCosIndex = startSinCosHeadIndex / this->headNumQ;
                uint64_t endSinCosIndex = (endHead + this->headNumQ - 1) / this->headNumQ;
                for (uint32_t index = startSinCosIndex; index < endSinCosIndex; ++index) {
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

            uint32_t repeatTime = this->headDim * loopN;
            AscendC::Mul(inputQCastFP32, inputCosCastFP32, inputQCastFP32, repeatTime);
            AscendC::Mul(reverseQ, negLocal, reverseQ, repeatTime);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(reverseQ, inputSinCastFP32, reverseQ, repeatTime);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Add(inputQCastFP32, reverseQ, inputQCastFP32, repeatTime);
            AscendC::PipeBarrier<PIPE_V>();

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
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID1);
        AscendC::Copy(tempBuf[this->headDim], tempBuf, this->headDim, headNumTemp - 1, {1, 1, headBlockLenFP32, 0});
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename BUF_TYPE>
    __aicore__ inline void CopyQGenReverseQ(const AscendC::LocalTensor<BUF_TYPE> &tempBufQ,
                                            const AscendC::LocalTensor<float> &tempBufQCast,
                                            const AscendC::LocalTensor<float> &tempBufRverseQ, uint64_t qOffset,
                                            uint16_t loopN)
    {
        // move in Q
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        AscendC::DataCopy(tempBufQ, this->qGm_[qOffset], {loopN, headBlockLen, 128 / 16, 0});
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        // cast fp32
        AscendC::Cast(tempBufQCast, tempBufQ, AscendC::RoundMode::CAST_NONE, loopN * this->headDim);
        AscendC::PipeBarrier<PIPE_V>();
        // move out reverseQ
        AscendC::DataCopy(tempBufRverseQ, tempBufQCast[this->rotateStride_], {loopN, rotaryLen, rotaryLen, rotaryLen});
        AscendC::DataCopy(tempBufRverseQ[this->rotateStride_], tempBufQCast, {loopN, rotaryLen, rotaryLen, rotaryLen});
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename BUF_TYPE>
    __aicore__ inline void CopyCosSin(const AscendC::LocalTensor<BUF_TYPE> &tempBufCos,
                                      const AscendC::LocalTensor<BUF_TYPE> &tempBufSin, uint64_t localStartAddr,
                                      uint64_t gmStartAddr, uint64_t repeatNum)
    {
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
    uint32_t rotateStride_{0};  // this->headDim / rope conf
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
    uint32_t loopTime{0};
    uint32_t lastLoopN{0};

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
class RmsNormQuant
{
public:
    __aicore__ inline RmsNormQuant() {}

    __aicore__ inline void Init(AscendC::GlobalTensor<T> &gammaGmTensor, AscendC::GlobalTensor<T> &betaGmTensor,
                                GM_ADDR gmInput, GM_ADDR gmOutput, uint32_t stride,
                                uint32_t num_col, float avg_factor, uint64_t gm_offset, uint64_t gm_out_offset,
                                uint32_t row_work_, const MlaTilingData &mlaParams_)
    {
        this->gammaGmTensor = gammaGmTensor;
        this->betaGmTensor = betaGmTensor;
        inputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmInput));
        outputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmOutput));

        num_col_ = num_col;
        avg_factor_ = avg_factor;
        epsilon_ = 1e-6;
        this->num_row_ = mlaParams_.n;
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
                                  const AscendC::LocalTensor<float> &res1Tensor,
                                  const AscendC::LocalTensor<float> &res3Tensor)
    {
        this->dstTensor = dstTensor;
        this->srcTensor = srcTensor;
        this->gammaTensor = gammaTensor;
        this->betaTensor = betaTensor;
        this->fp32_xy = res1Tensor;
        this->buf = res3Tensor;

        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * num_col_align_withStride_fp32];                 // 0
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * num_col_align_withStride_fp32];                 // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * num_col_align_withStride_fp32];                // 2
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE_BF16 * num_col_align_withStride_fp32];      // 4

        AscendC::DataCopy(gammaTensor, gammaGmTensor,
                          AscendC::DataCopyParams(1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));
        AscendC::DataCopy(betaTensor, betaGmTensor,
                          AscendC::DataCopyParams(1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));
        SET_FLAG(MTE2, V, EVENT_ID1);

        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);

        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Cast(buf[OFFSET_GAMMA * num_col_align_withStride_fp32], gammaTensor, AscendC::RoundMode::CAST_NONE,
             REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t pid = 0;
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        while (pid < row_work_) {
            uint64_t offset = pid * num_col_;
            uint64_t outOffset = pid * (num_col_ - input_stride_);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            AscendC::DataCopy(srcTensor, inputGmTensor[gm_offset_ + offset],
                                AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

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
            Mul(fp32_xy, fp32_xy, g, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                 AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            
            AscendC::LocalTensor<T> tmpfp16 =
                buf.ReinterpretCast<T>()[OFFSET_SUM * num_col_align_withStride_fp32 * 2];
            CastFrom32To16(tmpfp16, fp32_xy, num_col_align_withStride_fp32);
            AscendC::PipeBarrier<PIPE_V>();
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            AscendC::DataCopy(outputGmTensor[gm_out_offset_ + outOffset], tmpfp16,
                              AscendC::DataCopyParams(1, (num_col_ - input_stride_) / 16, 0, 0));
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            ++pid;
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }

private:
    AscendC::LocalTensor<int8_t> dstTensor;
    AscendC::LocalTensor<T> srcTensor;
    AscendC::LocalTensor<T> gammaTensor;
    AscendC::LocalTensor<T> betaTensor;
    AscendC::LocalTensor<float> fp32_xy;
    AscendC::LocalTensor<float> buf;

    AscendC::GlobalTensor<T> gammaGmTensor;
    AscendC::GlobalTensor<T> betaGmTensor;
    AscendC::GlobalTensor<T> inputGmTensor;
    AscendC::GlobalTensor<T> outputGmTensor;

    uint32_t num_col_{0};
    uint32_t num_row_{0};
    uint32_t row_work_{0};
    uint32_t row_work{0};
    uint32_t row_step_{0};
    uint32_t row_tail_{0};
    uint64_t gm_offset_{0};
    uint64_t gm_out_offset_{0};
    float avg_factor_{1.0};
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
    uint32_t num_slice_{0};
    uint32_t tail_size_{0};
    uint32_t tail_copy_{0};
};

#ifdef __DAV_C220_CUBE__
struct MatCoord {
    uint64_t m{0};
    uint64_t k{0};
    uint64_t n{0};
};

template <typename InDtype, typename OutDtype, DataFormat formatB, bool transB, uint32_t swizzleDirect,
          uint64_t splitGapA, uint64_t splitGapC, bool ein>
class PpMatmulEinSum
{
    using AccumDtype = float;

    template <DataFormat srcFormat, DataFormat dstFormat>
    using CopyGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, InDtype, srcFormat, dstFormat>;
    using LoadCbufToCa = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::ZN, DataFormat::ZZ>;
    using LoadCbufToCb = l1_to_l0_b<ArchType::ASCEND_V220, InDtype, transB, DataFormat::ZN, DataFormat::NZ>;
    using Mad = mmad<ArchType::ASCEND_V220, InDtype, InDtype, AccumDtype, false>;
    using CopyCcToGm = l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, OutDtype, AccumDtype>;

    static constexpr uint32_t L0_PINGPONG_BUFFER_LEN = 16384;
    static constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 131072;
    static constexpr uint32_t CONST_16 = 16;
    static constexpr uint32_t CONST_256 = 256;

public:
    __aicore__ explicit PpMatmulEinSum(){};

    __aicore__ __force_inline__ void Init(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, PpMatmulTilingData &tilingdata)
    {
#ifdef __DAV_C220_CUBE__
        batch_size = tilingdata.numBatch;
        m = tilingdata.m;
        k = tilingdata.k;
        n = tilingdata.n;
        m0 = tilingdata.m0;
        k0 = tilingdata.k0;
        n0 = tilingdata.n0;
        tdim.m = tilingdata.mLoop;
        tdim.k = tilingdata.kLoop;
        tdim.n = tilingdata.nLoop;
        core_loop = tilingdata.coreLoop;
        swizzle_cnt = tilingdata.swizzleCount;
        num_core = tilingdata.blockDim;
        core_idx = AscendC::GetBlockIdx();
        ping_flag = 1;

        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(gmA));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(gmB));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(gmC));

        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        l1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_base_b = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(RoundUp<CONST_256>(m0 * k0 * sizeof(InDtype)));
        l0a_base = buf.GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0b_base = buf.GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
#endif
        return;
    }

    __aicore__ __force_inline__ void Process()
    {
#ifdef __DAV_C220_CUBE__
        if (block_idx >= num_core) {
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
            uint64_t offset_c = 0;
            if constexpr (ein) {
                offset_c = tidx.m * m0 * batch_size * (n + splitGapC) + batch_idx * (n + splitGapC) + tidx.n * n0;
            } else {
                offset_c = batch_idx * m * n + tidx.m * m0 * n + tidx.n * n0;
            }
            uint64_t m_actual = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
            uint64_t n_actual = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
            uint64_t m_round = RoundUp<CONST_16>(m_actual);
            uint64_t n_round = RoundUp<CONST_16>(n_actual);
            uint64_t mn_max = m_round > n_round ? m_round : n_round;
            uint64_t k_part_len = L0_PINGPONG_BUFFER_LEN / mn_max / CONST_16 * CONST_16;
            uint64_t shuffle_k = en_shuffle_k ? (core_idx % tdim.k) : 0;

            uint64_t k_actual = (shuffle_k == tdim.k - 1) ? k - shuffle_k * k0 : k0;
            uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;

            LocalTensor l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
            LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
            event_t event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            if (loop_idx == core_idx) {
                // Copy A from gm to l1 buffer
                uint64_t offset_a = GetOffsetA(batch_idx, tidx.m, shuffle_k);
                WAIT_FLAG(MTE1, MTE2, event_id);
                CopyTileA(l1_buf_a, gm_a[offset_a], m_actual, m_round, k_actual, k_round);
                SET_FLAG(MTE2, MTE1, event_id);

                // Copy B from gm to l1 buffer
                uint64_t offset_b = GetOffsetB(batch_idx, shuffle_k, tidx.n);
                WAIT_FLAG(MTE1, MTE2, event_id + 2);
                CopyTileB(l1_buf_b, gm_b[offset_b], k_actual, k_round, n_actual, n_round);
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
                    uint64_t offset_a_next = GetOffsetA(batch_idx, tidx.m, shuffle_k_next);
                    uint64_t offset_b_next = GetOffsetB(batch_idx, shuffle_k_next, tidx.n);

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
                    uint64_t offset_a_next = GetOffsetA(b_idx_next, tidx.m, shuffle_k_next);
                    uint64_t offset_b_next = GetOffsetB(b_idx_next, shuffle_k_next, tidx.n);

                    LocalTensor l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
                    LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    // Preload A from gm to l1 buffer.
                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    CopyTileA(l1_buf_a_next, gm_a[offset_a_next], m_actual_next, m_round_next, k_actual_next,
                              k_round_next);
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // Preload B from gm to l1 buffer.
                    WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                    CopyTileB(l1_buf_b_next, gm_b[offset_b_next], k_actual_next, k_round_next, n_actual_next,
                              n_round_next);
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

                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, mte1_mad_event_id);
                }

                ping_flag = 1 - ping_flag;
            }

            SET_FLAG(M, FIX, EVENT_ID0);
            WAIT_FLAG(M, FIX, EVENT_ID0);

            // copy from L0C to gm
            if constexpr (ein) {
                CopyCcToGm(gm_c[offset_c],                 // dst
                           l0c_buf,                        // src
                           m_actual,                       // mTileActual
                           n_actual,                       // nTileActual
                           m_round,                        // mTileCeil
                           (n + splitGapC) * batch_size);  // nActual

            } else {
                CopyCcToGm(gm_c[offset_c],  // dst
                           l0c_buf,         // src
                           m_actual,        // MSize
                           n_actual,        // NSize
                           m_round,      // srcStride
                           n);              // dstStride_dst_D
            }
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

private:
    __aicore__ __force_inline__ void GetBaseBlockIdx(uint64_t index, MatCoord &tidx)
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

    __aicore__ __force_inline__ uint64_t GetOffsetA(const uint64_t bIdx, const uint64_t mIdx, const uint64_t kIdx)
    {
        if constexpr (ein) {
            return mIdx * m0 * batch_size * (k + splitGapA) + bIdx * (k + splitGapA) + kIdx * k0;
        } else {
            return bIdx * m * k + mIdx * m0 * k + kIdx * k0;
        }
    }

    __aicore__ __force_inline__ uint64_t GetOffsetB(const uint64_t bIdx, const uint64_t kIdx, const uint64_t nIdx)
    {
        if constexpr (formatB == DataFormat::ND) {
            if constexpr (transB) {
                return bIdx * k * n + nIdx * n0 * k + kIdx * k0;
            } else {
                return bIdx * k * n + kIdx * k0 * n + nIdx * n0;
            }
        } else {
            if constexpr (transB) {
                return bIdx * RoundUp<CONST_16>(n) * RoundUp<CONST_16>(k) + kIdx * k0 * RoundUp<CONST_16>(n) +
                       nIdx * n0 * CONST_16;
            } else {
                return bIdx * RoundUp<CONST_16>(k) * RoundUp<CONST_16>(n) + nIdx * n0 * RoundUp<CONST_16>(k) +
                       kIdx * k0 * CONST_16;
            }
        }
    }

    __aicore__ __force_inline__ void CopyTileA(AscendC::LocalTensor<InDtype> &dstTensor,
                                               const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t m_actual,
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
            if constexpr (ein) {
                CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(dstTensor,                      // dst
                                                             srcTensor,                      // src
                                                             m_actual,                       // nTileActual
                                                             m_round,                        // nTileCeil
                                                             m,                              // nVal
                                                             k_actual,                       // dTileActual
                                                             k_round,                        // dTileCeil
                                                             (k + splitGapA) * batch_size);  // dVal
            } else {
                CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(dstTensor,  // dst
                                                             srcTensor,  // src
                                                             m_actual,   // nTileActual
                                                             m_round,    // nTileCeil
                                                             m,          // nVal
                                                             k_actual,   // dTileActual
                                                             k_round,    // dTileCeil
                                                             k);         // dVal
            }
        }
    }

    __aicore__ __force_inline__ void CopyTileB(AscendC::LocalTensor<InDtype> &dstTensor,
                                               const AscendC::GlobalTensor<InDtype> &srcTensor, const uint64_t k_actual,
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
    uint32_t en_shuffle_k{0};
    uint32_t ping_flag{0};
};
#endif

template <typename InDtype, int8_t CACHE_MODE, DataFormat weightFormat1 = DataFormat::NZ,
          DataFormat weightFormat2 = DataFormat::NZ, DataFormat weightFormat3 = DataFormat::ND>
class MLAOperation
{
    static constexpr uint64_t splitGapC = CACHE_MODE == CACHE_MODE_KVCACHE ? CONST_64 : CONST_0;

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

    __aicore__ inline void Init(GM_ADDR hiddenStateGm, GM_ADDR wdqkvGm, GM_ADDR gamma2Gm,
                                GM_ADDR beta2Gm, GM_ADDR gamma3Gm,
                                GM_ADDR sin1Gm, GM_ADDR cos1Gm, GM_ADDR sin2Gm, GM_ADDR cos2Gm, GM_ADDR keycacheGm,
                                GM_ADDR slotMappingGm, GM_ADDR wuqGm, GM_ADDR wukGm,
                                GM_ADDR qGm, GM_ADDR keycacheOutGm, GM_ADDR qGm2, GM_ADDR keycacheOutGm2, GM_ADDR s1Gm,
                                GM_ADDR s2Gm, GM_ADDR s3Gm)
    {
        gamma3GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(gamma3Gm));
        sin1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(sin1Gm));
        cos1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(cos1Gm));
        keycacheGmTensor1.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(keycacheOutGm));
        keycacheGmTensor2.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(keycacheOutGm2));
        slotMappingGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotMappingGm));
        s2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(s2Gm));
        s3GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(s3Gm));

#ifdef __DAV_C220_CUBE__
        mm_1.Init(hiddenStateGm, wdqkvGm, s3Gm, mlaParams.mm1);
        mm_2.Init(s1Gm, wuqGm, s2Gm, mlaParams.mm2);
        mm_ein_sum.Init(s2Gm, wukGm, qGm, mlaParams.mm3);
#endif

        hiddenStateGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(hiddenStateGm));
        wdqkvGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(wdqkvGm));
        gamma2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(gamma2Gm));
        sin2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(sin2Gm));
        cos2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(cos2Gm));
        wuqGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(wuqGm));
        wukGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(wukGm));
        s1GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(s1Gm));
        qGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(qGm));
        qGmTensor2.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(qGm2));
        beta2GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(beta2Gm));

#ifdef __DAV_C220_VEC__
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
        rmsNormQuant2.Init(gamma2GmTensor, beta2GmTensor, s3Gm, s1Gm, SPLIT_SIZE_ONE,
                           num_col_2, 0.000651041666, vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2,
                           vectorBlockIdx * static_cast<uint64_t>(row_work) * SPLIT_SIZE_TWO, row_work_, mlaParams);
        ropeFp16.RopeInit(s2Gm, cos2GmTensor, sin2GmTensor, qGmTensor, qGmTensor2, mlaParams);
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
        const AscendC::LocalTensor<float> &calTensor, const AscendC::LocalTensor<T1> &outTmpTensor)
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
            AscendC::DataCopy(srcTensor, s3GmTensor[offset],
                              AscendC::DataCopyParams(1, MM1_OUT_SIZE / BLOCK_SIZE_16, 0, 0));
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
            Cast(outTmpTensor, rmsNormTensor, AscendC::RoundMode::CAST_RINT, SPLIT_RMSNRORM_SIZE_ONE);
            /* RmsNorm end */
            /* Rope K start */
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
            Cast(outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], ropeKRevertTensor, AscendC::RoundMode::CAST_RINT,
                 SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            /* Rope K end */
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(S, MTE3, EVENT_ID0);
            if constexpr (CACHE_MODE == CACHE_MODE_KVCACHE) {
                DataCopy(keycacheGmTensor1[cacheStart], outTmpTensor, SPLIT_SIZE_ONE);
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

    uint32_t num_core_;
    uint32_t num_col_1;
    uint32_t num_col_2;
    float epsilon_;
    uint32_t num_row;
    uint32_t row_work;
    uint32_t row_work_;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::GlobalTensor<InDtype> hiddenStateGmTensor;

    AscendC::GlobalTensor<InDtype> wdqkvGmTensor;
    AscendC::GlobalTensor<InDtype> gamma2GmTensor;
    AscendC::GlobalTensor<InDtype> gamma3GmTensor;
    AscendC::GlobalTensor<InDtype> sin1GmTensor;
    AscendC::GlobalTensor<InDtype> cos1GmTensor;
    AscendC::GlobalTensor<InDtype> sin2GmTensor;
    AscendC::GlobalTensor<InDtype> cos2GmTensor;
    AscendC::GlobalTensor<InDtype> keycacheGmTensor1;
    AscendC::GlobalTensor<InDtype> keycacheGmTensor2;
    AscendC::GlobalTensor<int32_t> slotMappingGmTensor;
    AscendC::GlobalTensor<InDtype> wuqGmTensor;
    AscendC::GlobalTensor<InDtype> wukGmTensor;

    AscendC::GlobalTensor<InDtype> qGmTensor;
    AscendC::GlobalTensor<InDtype> qGmTensor2;
    AscendC::GlobalTensor<InDtype> s1GmTensor;
    AscendC::GlobalTensor<InDtype> s2GmTensor;
    AscendC::GlobalTensor<InDtype> s3GmTensor;
    AscendC::GlobalTensor<InDtype> beta2GmTensor;

#ifdef __DAV_C220_CUBE__
    PpMatmulEinSum<InDtype, InDtype, weightFormat1, true, 0, 0, 0, false> mm_1;
    PpMatmulEinSum<InDtype, InDtype, weightFormat2, true, 0, 0, 0, false> mm_2;
    PpMatmulEinSum<InDtype, InDtype, weightFormat3, false, 0, CONST_64, splitGapC, true> mm_ein_sum;
#endif

#ifdef __DAV_C220_VEC__
    RmsNormQuant<InDtype, true, false> rmsNormQuant2;
    RopeFp16<InDtype, InDtype, InDtype, CACHE_MODE> ropeFp16;
#endif
};

template <typename InDtype, int8_t CACHE_MODE, DataFormat weightFormat1, DataFormat weightFormat2,
          DataFormat weightFormat3>
__aicore__ inline void
MLAOperation<InDtype, CACHE_MODE, weightFormat1, weightFormat2, weightFormat3>::ProcessCube()
{
#ifdef __DAV_C220_CUBE__
    mm_1.Process();
    FftsCrossCoreSync<PIPE_FIX, 0>(MM1);
    WaitFlagDev(MM1);
    FftsCrossCoreSync<PIPE_FIX, 0x2>(QUANT1);

    WaitFlagDev(AIC_MM2_START);
    mm_2.Process();
    FftsCrossCoreSync<PIPE_FIX, 0>(MM2);
    WaitFlagDev(MM2);

    FftsCrossCoreSync<PIPE_FIX, 0x2>(EINSUMOUT);
    mm_ein_sum.Process();
#endif
}

template <typename InDtype, int8_t CACHE_MODE, DataFormat weightFormat1, DataFormat weightFormat2,
          DataFormat weightFormat3>
__aicore__ inline void
MLAOperation<InDtype, CACHE_MODE, weightFormat1, weightFormat2, weightFormat3>::ProcessVector()
{
#ifdef __DAV_C220_VEC__
    WaitFlagDev(QUANT1);
    if (row_work_ != 0) {
        uint32_t num_col_align_int8 = (num_col_2 + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        uint32_t num_col_align_f16 = (num_col_2 + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        uint32_t num_col_align_f32 = (num_col_2 + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        AscendC::LocalTensor<InDtype> input_tensor = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        AscendC::LocalTensor<InDtype> gamma_tensor = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(MM1_OUT_SIZE * 2);
        AscendC::LocalTensor<InDtype> beta_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2);
        AscendC::LocalTensor<float> res1_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64);
        AscendC::LocalTensor<float> res3_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64 + num_col_align_f32 * 4);
        AscendC::LocalTensor<int8_t> output_tensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(
            MM1_OUT_SIZE * 2 + SPLIT_SIZE_TWO * 2 + SPLIT_SIZE_TWO * 2 + 64 + num_col_align_f32 * 4 +
            BUF_FACTOR * num_col_align_f32 * 4 + 64 + MM1_OUT_SIZE * 4 * 2 + 32);
        rmsNormQuant2.Launch(output_tensor, input_tensor, gamma_tensor, beta_tensor, res1_tensor, res3_tensor);
    }
    FftsCrossCoreSync<PIPE_MTE3, 0>(RMSNORMQUANT2);
    WaitFlagDev(RMSNORMQUANT2);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(AIC_MM2_START);

    if (row_work_ != 0) {
        AscendC::LocalTensor<InDtype> input_tensor = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        AscendC::LocalTensor<InDtype> gamma_tensor = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(MM1_OUT_SIZE * 2);
        AscendC::LocalTensor<InDtype> sin_tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2);
        AscendC::LocalTensor<InDtype> cos_tensor = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 2);
        AscendC::LocalTensor<int32_t> slotMapping_tensor = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4);
        int32_t rms3_ub_offset =
            MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4 + 4096 * 32;
        AscendC::LocalTensor<float> tmp32_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(rms3_ub_offset);

        int32_t out_ub_offset = MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4 +
                                4096 * 32 + SPLIT_RMSNRORM_SIZE_ONE * 3 * 4 + SPLIT_RMSNRORM_SIZE_TWO * 2 * 4 +
                                MM1_OUT_SIZE * 4 * 2 + 32;
        AscendC::LocalTensor<InDtype> temp_tensor = buf.GetBuffer<BufferType::ASCEND_UB, InDtype>(out_ub_offset);

        RmsNormAndRopeConvergence1<InDtype>(
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
            temp_tensor);
    }

    WaitFlagDev(EINSUMOUT);
    ropeFp16.Process();
#endif
}

}  // namespace MLAPO_BF16_NQ
