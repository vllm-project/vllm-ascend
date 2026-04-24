/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file inplace_partial_rotary_mul.h
 * \brief
 */
#ifndef INPLACE_PARITAL_ROTARY_MUL_H
#define INPLACE_PARITAL_ROTARY_MUL_H

#include "kernel_operator.h"

namespace InplacePartialRotaryMul {
using namespace AscendC;

template <typename T, bool isBrc>
class InplacePartialRotaryMulABA {
public:
    __aicore__ inline InplacePartialRotaryMulABA() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR r1, GM_ADDR r2, GM_ADDR y, GM_ADDR workspace,
        const RopeRegbaseTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInData(LocalTensor<T> &xUb, GlobalTensor<T> &xGm, int64_t blockCout, int64_t blockLen, int64_t gmOffset, int64_t ubOffset);
    __aicore__ inline void CopyInDataR(LocalTensor<T> &xUb, GlobalTensor<T> &xGm, int64_t blockCout, int64_t blockLen, int64_t gmOffset, int64_t ubOffset);
    __aicore__ inline void SetGatherSrcOffset(LocalTensor<int32_t> &idsUb, int64_t count);
    __aicore__ inline void ComputeMul(LocalTensor<float> &dtsUb, LocalTensor<float> & src0Ub, LocalTensor<float> &src1Ub, int64_t onceA, int64_t numHead,int64_t headDim);
    __aicore__ inline void  InterleavedInversion(int64_t count,LocalTensor<float> &ub);
    __aicore__ inline void DataCopyOut(LocalTensor<T> &yUb, GlobalTensor<T> &xGm, int64_t blockCout, int64_t blockLen, int64_t gmOffset, int64_t ubOffset);
private:
    TPipe* pipe_;
    const RopeRegbaseTilingData* tiling_;
    int32_t blockIdx_ = 0;

    //需要的tilingdata
    int64_t halfNumx_ = 0;
    int64_t halfNumr1_ = 0;
    int64_t xNum_= 0;
    int64_t r1Num_= 0;
    int64_t count_ = 0;
    
    static constexpr int32_t ONE_BLOCK_SIZE = 32;
    int32_t perBlock32 = ONE_BLOCK_SIZE / sizeof(float);

    GlobalTensor<T> xGm_;
    GlobalTensor<T> r1Gm_;
    GlobalTensor<T> r2Gm_;
    GlobalTensor<T> yGm_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECIN, 1> r1Que_;
    TQue<QuePosition::VECIN, 1> r2Que_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TBuf<QuePosition::VECCALC> idsBuf_;
    BinaryRepeatParams repeatParams_{1, 1, 1, 0, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadParams_{false, 0, 0, 0};

    int64_t usedCoreNum_=0;
    int64_t numHead_ =0;
    int64_t headDim_ =0;
    int64_t allHeadDim_ = 0;
    int64_t coreTUbLoopTime_ =0;
    int64_t coreBUbLoopTime_ = 0; //b分核，每个核处理多少个b，b就是shape0
    int64_t coreTUbLoopTail_ =0;
    int64_t coreBUbLoopTail_ = 0;
    int64_t ubFactor_ = 0;
    int64_t start_ = 0;
    int64_t blockFactor_=0;

};

template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::Init(GM_ADDR x, GM_ADDR r1, GM_ADDR r2, GM_ADDR y, GM_ADDR workspace,
    const RopeRegbaseTilingData *tilingData, TPipe *pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    pipe_ = pipe;
    usedCoreNum_ =tiling_->usedCoreNum;
    numHead_ = tiling_->numHead;
    headDim_ = tiling_->headDim;
    allHeadDim_ = tiling_->allHeadDim;
    coreTUbLoopTime_ = tiling_->coreTUbLoopTime;
    coreBUbLoopTime_ = tiling_->coreBUbLoopTime; //b分核，每个核处理多少个b，b就是shape0
    coreTUbLoopTail_ = tiling_->coreTUbLoopTail;
    coreBUbLoopTail_ = tiling_->coreBUbLoopTail;
    ubFactor_ = tiling_->ubFactor;
    blockFactor_ =  tiling_->blockFactor;
    start_ = tiling_->start;
    
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    r1Gm_.SetGlobalBuffer((__gm__ T *)r1);
    r2Gm_.SetGlobalBuffer((__gm__ T *)r2);
    yGm_.SetGlobalBuffer((__gm__ T *)y);
    count_ = numHead_ * headDim_;
    xNum_ = ubFactor_ * count_;
    r1Num_ = ubFactor_ * headDim_;

    pipe_->InitBuffer(xQue_, 2, xNum_ * sizeof(float));
    pipe_->InitBuffer(r1Que_, 2, r1Num_ * sizeof(float));
    pipe_->InitBuffer(r2Que_, 2, r1Num_ * sizeof(float));
    pipe_->InitBuffer(yQue_, 2, xNum_ * sizeof(float));
    pipe_->InitBuffer(idsBuf_, count_ *sizeof(uint32_t));
    if constexpr(sizeof(T) != sizeof(float)) {
        halfNumx_ = xNum_;
        halfNumr1_ = r1Num_;
    }
}
template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::CopyInData(LocalTensor<T> &xUb, GlobalTensor<T> &xGm, int64_t blockCout, int64_t blockLen, int64_t gmOffset, int64_t ubOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCout;
    copyParams.blockLen = blockLen * sizeof(T);
    copyParams.srcStride = (allHeadDim_ - headDim_)*sizeof(T); //整个输入的大小
    copyParams.dstStride = 0;
    DataCopyPad(xUb[ubOffset], xGm[gmOffset], copyParams, dataCopyPadParams_);
}
template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::CopyInDataR(LocalTensor<T> &xUb, GlobalTensor<T> &xGm, int64_t blockCout, int64_t blockLen, int64_t gmOffset, int64_t ubOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCout;
    copyParams.blockLen = blockLen * sizeof(T);
    copyParams.srcStride = 0; //整个输入的大小
    copyParams.dstStride = 0;
    DataCopyPad(xUb[ubOffset], xGm[gmOffset], copyParams, dataCopyPadParams_);
}
template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::SetGatherSrcOffset(LocalTensor<int32_t> &idsUb, int64_t count)
{
    for (int32_t i = 0; i < 8; ++i) {
        idsUb.SetValue(i, i ^ 1); // XOR with 1 to swap even and odd indices
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    int32_t scalarValue = 8;
    int32_t onceNum = 8;
    
    while (scalarValue < count) {
        int32_t nextValue = scalarValue * 2;
        if (nextValue < count) {
            Adds(idsUb[scalarValue], idsUb, scalarValue, scalarValue);
        } else {
            Adds(idsUb[scalarValue], idsUb, scalarValue, count - scalarValue);
            break;
        }
        scalarValue = nextValue;
    }
    Muls(idsUb, idsUb, 4, count);
}
template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::ComputeMul(LocalTensor<float> &dtsUb, LocalTensor<float> & src0Ub, LocalTensor<float> &src1Ub, int64_t onceA, int64_t numHead,int64_t headDim)
{
    int64_t count = numHead *headDim;
    if constexpr(!isBrc) {
        int64_t xtotalNum = onceA * count;
        Mul(dtsUb, src0Ub, src1Ub, xtotalNum); //x*cos 非brc elewise乘
    } else {
        if (headDim <= 64) {
            int32_t mask = headDim;
            repeatParams_.dstBlkStride = 1;
            repeatParams_.src0BlkStride = 1;
            repeatParams_.src1BlkStride = 1;
            repeatParams_.dstRepStride = 8;
            repeatParams_.src0RepStride = 8;
            repeatParams_.src1RepStride = 0;
            for (int64_t j =0; j<onceA; j++) {
                Mul(dtsUb[j*count], src0Ub[j*count], src1Ub[j*headDim], mask, numHead, repeatParams_); //x*cos 非brc elewise乘
            }
        } else {
            for (int64_t j =0; j < onceA; j++) {
                for (int64_t jj =0; jj < numHead; jj++) {
                    Mul(dtsUb[j*count + jj * headDim], src0Ub[j*count + jj * headDim], src1Ub[j*headDim], headDim); //x*cos 非brc elewise乘
            }
            }
        }
    }
    PipeBarrier<PIPE_V>();
}
template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::InterleavedInversion(int64_t count,LocalTensor<float> &ub)
{
    // 做奇数位的*-1
    SetMaskNorm();
    int64_t fp32Mask = 64;
    int64_t repeatTimes = count / 64;
    int64_t remain = count % 64;
    uint64_t fullMask = 0x5555555555555555; //0101010101010101  
    uint64_t tailMask = 0x55;
    SetVectorMask<float, MaskMode::NORMAL>(0, fullMask);
    Muls<float, false>(ub, ub, -1.0f, MASK_PLACEHOLDER, repeatTimes, {1,1,8,8});
    if (remain != 0) {
        int32_t tailTimes = count % 64 / 8;
        SetVectorMask<float, MaskMode::NORMAL>(0, tailMask);
        Muls<float, false>(ub[repeatTimes * 64], ub[repeatTimes * 64], -1.0f, MASK_PLACEHOLDER, tailTimes, {1,1,1,1});
    }
    ResetMask();
}
template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::DataCopyOut(LocalTensor<T> &yUb, GlobalTensor<T> &xGm, int64_t blockCout, int64_t blockLen, int64_t gmOffset, int64_t ubOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCout;
    copyParams.blockLen = blockLen * sizeof(T);
    copyParams.srcStride =  0;
    copyParams.dstStride = (allHeadDim_ - headDim_)*sizeof(T);
    DataCopyPad(yGm_[gmOffset], yUb[ubOffset], copyParams);
}

template <typename T, bool isBrc>
__aicore__ inline void InplacePartialRotaryMulABA<T, isBrc>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    LocalTensor<int32_t> idsUb = idsBuf_.Get<int32_t>();
    int32_t count = numHead_ * headDim_;
    SetGatherSrcOffset(idsUb, count);
    LocalTensor<uint32_t> idsUbUint32 =  idsBuf_.Get<uint32_t>();
    // A分核分ub, 一次搬入x个 cout, x就是ubFactor_
    int64_t ubLoopTimes = blockIdx_ == tiling_->usedCoreNum - 1 ? coreTUbLoopTime_ :coreBUbLoopTime_; //b分核，每个核处理多少个b，b就是shape0
    int64_t ubLoopTailNum = blockIdx_ == tiling_->usedCoreNum - 1 ? coreTUbLoopTail_ :coreBUbLoopTail_;
    int64_t ysCount = numHead_ *allHeadDim_;
    for(int64_t i=0; i< ubLoopTimes; i++) {
        int64_t ubSize = i == ubLoopTimes -1 ? ubLoopTailNum : ubFactor_;
        int64_t xtotalNum = ubSize * count;
        int64_t r1totalNum = ubSize * headDim_;
        LocalTensor<T> xUb = xQue_.AllocTensor<T>();
        int64_t gmOffset = blockIdx_ * blockFactor_ * ysCount + i * ubFactor_ * ysCount + start_;
        int64_t blockCout = ubSize * numHead_;
        CopyInData(xUb, xGm_, blockCout , headDim_, gmOffset, halfNumx_);
        xQue_.EnQue<T>(xUb);

        LocalTensor<T> r1Ub = r1Que_.AllocTensor<T>();
        int64_t r1Offset = blockIdx_ * blockFactor_ * headDim_ + i * ubFactor_ * headDim_;
        CopyInDataR(r1Ub, r1Gm_, ubSize , headDim_, r1Offset, halfNumr1_);
        r1Que_.EnQue<T>(r1Ub);

        LocalTensor<T> r2Ub = r2Que_.AllocTensor<T>();
        CopyInDataR(r2Ub, r2Gm_, ubSize , headDim_, r1Offset, halfNumr1_);
        r2Que_.EnQue<T>(r2Ub);

        xUb = xQue_.DeQue<T>();
        r1Ub = r1Que_.DeQue<T>();
        r2Ub = r2Que_.DeQue<T>();
        LocalTensor<float> xUbFp32 = xUb.template ReinterpretCast<float>();
        LocalTensor<float> r1UbFp32 = r1Ub.template ReinterpretCast<float>();
        LocalTensor<float> r2UbFp32 = r2Ub.template ReinterpretCast<float>();
        if constexpr(sizeof(T) != sizeof(float)) {
            // 非fp32时需要做cast
            Cast(xUbFp32, xUb[halfNumx_], RoundMode::CAST_NONE, xtotalNum);
            Cast(r1UbFp32, r1Ub[halfNumr1_], RoundMode::CAST_NONE, r1totalNum);
            Cast(r2UbFp32, r2Ub[halfNumr1_], RoundMode::CAST_NONE, r1totalNum);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<T> yUb = yQue_.AllocTensor<T>();
        LocalTensor<float> yUbFp32 = yUb.template ReinterpretCast<float>();
        ComputeMul(yUbFp32 , xUbFp32, r1UbFp32, ubSize, numHead_,headDim_);
        // 开始做选择的取数，基偶取数
        for (int64_t k =0; k < ubSize; k++) {
            Gather(xUbFp32[k*count], xUbFp32[k*count], idsUbUint32, uint32_t(0), uint32_t(count)); //奇偶交换完成
        }
        PipeBarrier<PIPE_V>();
        ComputeMul(xUbFp32 , xUbFp32, r2UbFp32, ubSize, numHead_,headDim_);
        r1Que_.FreeTensor(r1Ub);
        r2Que_.FreeTensor(r2Ub);
        InterleavedInversion(xtotalNum, xUbFp32);
        // 做最后的Add
        Add(yUbFp32, yUbFp32, xUbFp32, xtotalNum);
        xQue_.FreeTensor(xUb);
        PipeBarrier<PIPE_V>();
        if constexpr(sizeof(T) != sizeof(float)) {
            Cast(yUb, yUbFp32, RoundMode::CAST_RINT, xtotalNum);
        }
        yQue_.EnQue<T>(yUb);
        yUb = yQue_.DeQue<T>();
        DataCopyOut(yUb, yGm_, blockCout, headDim_, gmOffset, 0);
        yQue_.FreeTensor(yUb);
    }
}

}
#endif
