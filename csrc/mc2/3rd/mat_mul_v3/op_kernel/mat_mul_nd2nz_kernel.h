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
 * \file mat_mul_nd2nz_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_ND2NZ_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_ND2NZ_KERNEL_H__

#include "mat_mul_nd2nz_util.h"

using namespace AscendC;
using namespace matmul;
#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

template <class T>
class Mc2KernelND2NZMM {
public:
    __aicore__ inline Mc2KernelND2NZMM(){};
    __aicore__ inline void CopyIn(uint64_t progress, LocalTensor<T> &dstLocal);
    __aicore__ inline void CopyOutMM(uint64_t progress, LocalTensor<T> &srcLocal);
    __aicore__ inline void PadD(uint64_t progress, LocalTensor<T> &dstLocal, LocalTensor<T> &srcLocal, int eventIn,
                                int eventOut);
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src, uint32_t height, uint32_t width, uint32_t batch,
                                TBuf<TPosition::VECCALC> &ubBuffer, uint32_t usedCoreNum);
    template <ND2NZ_DB_TYPE TYPE, bool noZero = false>
    __aicore__ inline bool SetBufMM();
    template <Nd2NzMode mode = Nd2NzMode::MULTI_CORE>
    __aicore__ inline bool ProcessMM();
    template <Nd2NzMode mode>
    __aicore__ inline void ProcessInOutDB();
    template <Nd2NzMode mode>
    __aicore__ inline void ProcessNoDBReuse();
    template <Nd2NzMode mode>
    __aicore__ inline void ProcessOutDBReuse();

private:
    TBuf<TPosition::VECCALC> *ubPtr_;
    GlobalTensor<T> srcGM;
    GlobalTensor<T> dstGM;
    LocalTensor<T> inBuf_;
    LocalTensor<T> inBuf2_;
    LocalTensor<T> midBuf_;
    LocalTensor<T> outBuf_;
    LocalTensor<T> outBuf2_;
    LocalTensor<T> zeroBuf_;
    uint32_t padSize_;
    uint32_t height_;
    uint32_t hAligned_;
    uint32_t width_;
    uint32_t batch_;
    uint32_t wTail_;
    uint32_t hBuffer_;
    uint32_t nFullProgress_;
    uint32_t heightTotalTail_;
    uint16_t hPad_;
    uint32_t blockDim_;
    uint32_t blockIdx_;
    uint32_t hBlockNum_;
    uint32_t copyInSize_;
    uint64_t c0_;
    uint32_t copyInRepeat_;
    uint16_t widthBlockTotal_;
    bool noPadD_;
};

template <class T>
__aicore__ inline void Mc2KernelND2NZMM<T>::CopyIn(uint64_t progress, LocalTensor<T> &dstLocal)
{
    uint64_t curCopyInSize = progress == nFullProgress_ ? heightTotalTail_ * width_ : copyInSize_;
    uint64_t gmInOffset = copyInSize_ * progress;
    DataCopyExtParams copyParams{DEFAULT_DATA_COPY_NBURST, static_cast<uint32_t>(curCopyInSize * sizeof(T)),
                                 DEFAULT_DATA_COPY_STRIDE, DEFAULT_DATA_COPY_STRIDE, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dstLocal, srcGM[gmInOffset], copyParams, padParams);
}

template <class T>
__aicore__ inline void Mc2KernelND2NZMM<T>::CopyOutMM(uint64_t progress, LocalTensor<T> &srcLocal)
{
    uint64_t oneColSizeGM = Align2(height_ * batch_, ALIGNED_H) * c0_;
    uint64_t oneColSize = hBuffer_ * c0_;
    uint32_t copyOutSize = progress == nFullProgress_ ? Align2(heightTotalTail_, ALIGNED_H) * c0_ : oneColSize;
    if (progress == nFullProgress_) {
        for (uint32_t i = 0; i < widthBlockTotal_; i++) {
            Duplicate(srcLocal[oneColSize * i + heightTotalTail_ * c0_], T(0), padSize_);
        }
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    for (uint32_t i = 0; i < widthBlockTotal_; i++) {
        DataCopy(dstGM[oneColSizeGM * i + oneColSize * progress], srcLocal[oneColSize * i], copyOutSize);
    }
}

template <class T>
__aicore__ inline void Mc2KernelND2NZMM<T>::PadD(uint64_t progress, LocalTensor<T> &dstLocal, LocalTensor<T> &srcLocal,
                                                 int eventIn, int eventOut)
{
    if (wTail_ == 0) {
        PadDAligned<T>(progress, dstLocal, srcLocal, eventIn, eventOut, width_, c0_, hBlockNum_, true);
    } else {
        PadDMain<T>(progress, dstLocal, srcLocal, midBuf_, zeroBuf_, eventIn, eventOut, width_, c0_, hBlockNum_,
                    copyInRepeat_, hBuffer_, wTail_, true);
    }
}

template <class T>
__aicore__ inline void Mc2KernelND2NZMM<T>::Init(GM_ADDR dst, GM_ADDR src, uint32_t height, uint32_t width,
                                                 uint32_t batch, TBuf<TPosition::VECCALC> &ubBuffer,
                                                 uint32_t usedCoreNum)
{
    height_ = height;
    width_ = width;
    batch_ = batch;
    uint32_t hTotal = height_ * batch_;

    blockDim_ = usedCoreNum;
    blockIdx_ = GetBlockIdx();

    c0_ = BLOCK_SIZE_BYTE / sizeof(T);

    srcGM.SetGlobalBuffer((__gm__ T *)src);
    dstGM.SetGlobalBuffer((__gm__ T *)dst);
    ubPtr_ = &ubBuffer;

    noPadD_ = (width_ == c0_);

    uint32_t batchTail = height_ % ALIGNED_H;
    hPad_ = batchTail == 0 ? 0 : ALIGNED_H - batchTail;

    padSize_ = hPad_ * c0_;

    hAligned_ = Align2(height_, ALIGNED_H);

    uint32_t widthBlock = width_ / c0_;
    wTail_ = width_ & (c0_ - 1);

    widthBlockTotal_ = wTail_ ? widthBlock + 1 : widthBlock;
}

template <class T>
template <ND2NZ_DB_TYPE TYPE, bool noZero>
__aicore__ inline bool Mc2KernelND2NZMM<T>::SetBufMM()
{
    uint32_t hTotal = height_ * batch_;
    uint32_t wAligned = Align2(width_, c0_);

    uint32_t ubTotalWidth = 0;

    if constexpr (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT) {
        // If both of CopyIn and CopyOut with DB, total ub size calculates one midbuf + two inputbuf + two outbuf size.
        ubTotalWidth = 3 * width_ + 2 * wAligned;
    }
    if constexpr (TYPE == ND2NZ_DB_TYPE::OUTPUT) {
        // If only CopyOut with DB, total ub size calculates one midbuf + two outbuf size. (inputbuf can reuse midbuf).
        ubTotalWidth = width_ + 2 * wAligned;
    }
    if constexpr (TYPE == ND2NZ_DB_TYPE::NO_DB_REUSE_OUTPUT) {
        // If none of CopyIn or Copyout with DB, calculates one midbuf + one outbuf size. (inputbuf can reuse midbuf).
        ubTotalWidth = width_ + wAligned;
    }

    if (!(width_ > c0_) && !noZero) {
        ubTotalWidth += width_;
    }

    uint32_t hMax = TOTAL_UB_SIZE / sizeof(T) / ubTotalWidth;

    uint32_t hBlockNumEle = M_BLOCK_NUM_ELE_LIST[wTail_] * 2 / sizeof(T);
    hBlockNumEle = width_ == 1 ? 1 : hBlockNumEle;
    hBlockNumEle = hBlockNumEle == 0 ? 1 : hBlockNumEle;
    uint32_t gcd = GCD_LIST[wTail_];
    if constexpr (sizeof(T) == sizeof(float)) {
        gcd = wTail_ == 0 ? 8 : gcd; // if float, one Transdata5HD only need 8 aligned instead of 16 aligned.
    }

    uint32_t hEle = hBlockNumEle * ALIGNED_H;
    uint32_t eleNum = (hTotal + hEle - 1) / hEle;
    uint32_t eleNumTmp = hMax / hEle;
    eleNum = min(eleNumTmp, eleNum);
    eleNum = eleNum * hBlockNumEle > REPEAT_TIMES_MAX ? (REPEAT_TIMES_MAX / hBlockNumEle) : eleNum;

    if (eleNum == 0) {
        return false;
    }

    copyInRepeat_ = eleNum * width_ / gcd;
    hBuffer_ = eleNum * hEle;
    copyInSize_ = hBuffer_ * width_;
    hBlockNum_ = eleNum * hBlockNumEle;
    nFullProgress_ = hTotal / hBuffer_;
    heightTotalTail_ = hTotal % hBuffer_;

    uint32_t ubTail = 0;

    midBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];

    ubTail += copyInSize_ * sizeof(T);

    if ((width_ < c0_) && !noZero) {
        zeroBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += copyInSize_ * sizeof(T);
    }
    if (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT) {
        inBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += copyInSize_ * sizeof(T);
        inBuf2_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += copyInSize_ * sizeof(T);
    }

    if (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT || TYPE == ND2NZ_DB_TYPE::OUTPUT ||
        TYPE == ND2NZ_DB_TYPE::NO_DB_REUSE_OUTPUT) {
        outBuf_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += hBuffer_ * wAligned * sizeof(T);
    }
    if (TYPE == ND2NZ_DB_TYPE::IN_OUTPUT || TYPE == ND2NZ_DB_TYPE::OUTPUT) {
        outBuf2_ = ubPtr_->Get<T>()[ubTail / sizeof(T)];
        ubTail += hBuffer_ * wAligned * sizeof(T);
    }
    if ((width_ < c0_) && !noZero) {
        Duplicate(zeroBuf_, T(0), copyInSize_);
    }

    // PipeBarrier<PIPE_ALL>();
    return true;
}

template <class T>
template <Nd2NzMode mode>
__aicore__ inline bool Mc2KernelND2NZMM<T>::ProcessMM()
{
    if (width_ % c0_ == 0) {
        if (SetBufMM<ND2NZ_DB_TYPE::IN_OUTPUT, true>()) { // issue:when innersize > 49152B, will return false and break.
            ProcessInOutDB<mode>();
            return true;
        }
        return false;
    } else if (SetBufMM<ND2NZ_DB_TYPE::OUTPUT>()) {
        ProcessOutDBReuse<mode>();
        return true;
    } else if (SetBufMM<ND2NZ_DB_TYPE::NO_DB_REUSE_OUTPUT>()) {
        // now max totalwidth = 2 * width + widthalign = 3 * 512B <<< ubsize, so this branch will never be used.
        ProcessNoDBReuse<mode>();
        return true;
    }
    return false;
}

template <class T>
template <Nd2NzMode mode>
__aicore__ inline void Mc2KernelND2NZMM<T>::ProcessInOutDB()
{
    uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;
    uint32_t j = 0;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    int32_t start_idx = (mode == Nd2NzMode::MULTI_CORE ? blockIdx_ : GetSubBlockIdx());
    for (int32_t i = start_idx; i < nLoop; i += blockDim_, j++) {
        if (j % 2 == 1) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyIn(i, inBuf_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadD(i, outBuf_, inBuf_, EVENT_ID0, EVENT_ID0);

            CopyOutMM(i, outBuf_);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            CopyIn(i, inBuf2_);

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadD(i, outBuf2_, inBuf2_, EVENT_ID1, EVENT_ID1);

            CopyOutMM(i, outBuf2_);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    if constexpr (mode == Nd2NzMode::MULTI_CORE) {
        PipeBarrier<PIPE_ALL>();
    }
}

template <class T>
template <Nd2NzMode mode>
__aicore__ inline void Mc2KernelND2NZMM<T>::ProcessOutDBReuse()
{
    uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;
    uint32_t j = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t start_idx = (mode == Nd2NzMode::MULTI_CORE ? blockIdx_ : GetSubBlockIdx());
    for (int32_t i = start_idx; i < nLoop; i += blockDim_, j++) {
        if (j % 2 == 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            CopyIn(i, outBuf_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadDMain<T>(i, outBuf_, outBuf_, midBuf_, zeroBuf_, 0, 0, width_, c0_, hBlockNum_, copyInRepeat_, hBuffer_,
                        wTail_, false);
            CopyOutMM(i, outBuf_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            CopyIn(i, outBuf2_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadDMain<T>(i, outBuf2_, outBuf2_, midBuf_, zeroBuf_, 0, 0, width_, c0_, hBlockNum_, copyInRepeat_,
                        hBuffer_, wTail_, false);
            CopyOutMM(i, outBuf2_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    if constexpr (mode == Nd2NzMode::MULTI_CORE) {
        PipeBarrier<PIPE_ALL>();
    }
}

template <class T>
template <Nd2NzMode mode>
__aicore__ inline void Mc2KernelND2NZMM<T>::ProcessNoDBReuse()
{
    uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;
    int32_t start_idx = (mode == Nd2NzMode::MULTI_CORE ? blockIdx_ : GetSubBlockIdx());
    for (int32_t i = start_idx; i < nLoop; i += blockDim_) {
        CopyIn(i, outBuf_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        PadDMain<T>(i, outBuf_, outBuf_, midBuf_, zeroBuf_, 0, 0, width_, c0_, hBlockNum_, copyInRepeat_, hBuffer_,
                    wTail_, false);

        CopyOutMM(i, outBuf_);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if constexpr (mode == Nd2NzMode::MULTI_CORE) {
        PipeBarrier<PIPE_ALL>();
    }
}


#endif
