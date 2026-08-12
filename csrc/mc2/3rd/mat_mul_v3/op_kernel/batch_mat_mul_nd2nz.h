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
 * \file batch_mat_mul_nd2nz.h
 * \brief
 */
#ifndef __OP_BATCH_MATMUL_ND2NZ_H__
#define __OP_BATCH_MATMUL_ND2NZ_H__

#include "../../mat_mul_v3/op_kernel/mat_mul_nd2nz_util.h"

using namespace AscendC;
using namespace matmul;
#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

template <class T>
class Mc2KernelND2NZBMM {
public:
    __aicore__ inline Mc2KernelND2NZBMM(){};
    __aicore__ inline void CopyIn(uint64_t progress, LocalTensor<T> &dstLocal);
    __aicore__ inline bool SetBufBMM();
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src, uint32_t height, uint32_t width, uint32_t batch,
                                TBuf<TPosition::VECCALC> &ubBuffer, uint32_t usedCoreNum);

    __aicore__ inline bool ProcessBMM();

private:
    __aicore__ inline void CopyOutDirect(uint64_t gmOutOffset, uint32_t startPad, uint16_t total, uint64_t progress);
    __aicore__ inline void CopyOutPageInit(uint64_t &gmOutOffset, uint32_t startPad, uint32_t &bufOffset);
    __aicore__ inline void CopyOutMakePage(uint32_t nLoop, uint32_t &bufOffset);
    __aicore__ inline void CopyOutPageMainImp(uint64_t &gmOutOffset, uint32_t nLoop, uint32_t &bufOffset);
    __aicore__ inline void CopyOutPageMain(uint64_t &gmOutOffset, uint32_t mPage, uint32_t startPad, uint32_t total,
                                           uint32_t &bufOffset, uint64_t progress);
    __aicore__ inline void CopyOutPageEnd(uint64_t gmOutOffset, uint32_t res, uint32_t &bufOffset);
    __aicore__ inline void CopyOutPage(uint64_t gmOutOffset, uint32_t mPage, uint32_t total, uint32_t startPad,
                                       uint64_t progress);
    __aicore__ inline void CopyOutBatchReform(uint64_t gmOutOffset, uint32_t mPage, uint32_t total, uint32_t startPad,
                                              uint64_t progress);
    __aicore__ inline void ComputeBMM(uint64_t progress);

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
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyIn(uint64_t progress, LocalTensor<T> &dstLocal)
{
    uint64_t curCopyInSize = progress == nFullProgress_ ? heightTotalTail_ * width_ : copyInSize_;
    uint64_t gmInOffset = copyInSize_ * progress;
    DataCopyExtParams copyParams{DEFAULT_DATA_COPY_NBURST, static_cast<uint32_t>(curCopyInSize * sizeof(T)),
                                 DEFAULT_DATA_COPY_STRIDE, DEFAULT_DATA_COPY_STRIDE, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dstLocal, srcGM[gmInOffset], copyParams, padParams);
}

template <class T>
__aicore__ inline bool Mc2KernelND2NZBMM<T>::SetBufBMM()
{
    uint32_t hTotal = height_ * batch_;
    uint32_t wAligned = Align2(width_, c0_);

    uint32_t hMax = TOTAL_UB_SIZE / sizeof(T) / (width_ + width_ + width_ + wAligned);
    // hBlockNumEle表示最少要几行连续数据才能32B对齐
    uint32_t hBlockNumEle = M_BLOCK_NUM_ELE_LIST[wTail_] * 2 / sizeof(T);
    hBlockNumEle = width_ == 1 ? 1 : hBlockNumEle;
    hBlockNumEle = hBlockNumEle == 0 ? 1 : hBlockNumEle;
    // gcd是c0_和width_的最大公约数
    uint32_t gcd = GCD_LIST[wTail_];
    if constexpr (sizeof(T) == sizeof(float)) {
        gcd = wTail_ == 0 ? 8 : gcd;
    }
    // hEle表示最小载入行数，为满足vnchwconv的要求，要乘个16
    uint32_t hEle = hBlockNumEle * ALIGNED_H;
    // eleNum是在ub_buffer和外轴限制的基础上，最多可载入几倍的hEle
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
    // 16 * (eleNum * hBlockNumEle)*width_，计算地址偏移时使用
    hBlockNum_ = eleNum * hBlockNumEle;
    nFullProgress_ = hTotal / hBuffer_;
    heightTotalTail_ = hTotal % hBuffer_;

    midBuf_ = ubPtr_->Get<T>()[0];
    zeroBuf_ = ubPtr_->Get<T>()[copyInSize_];
    inBuf_ = ubPtr_->Get<T>()[copyInSize_ * 2];
    outBuf_ = ubPtr_->Get<T>()[copyInSize_ * 3];
    // 清零可以去掉，mad使用实际的大小计算，就不需要清零
    Duplicate(zeroBuf_, T(0), copyInSize_);

    PipeBarrier<PIPE_ALL>();
    return true;
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::Init(GM_ADDR dst, GM_ADDR src, uint32_t height, uint32_t width,
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
__aicore__ inline bool Mc2KernelND2NZBMM<T>::ProcessBMM()
{
    if (SetBufBMM()) {
        uint32_t nLoop = heightTotalTail_ ? nFullProgress_ + 1 : nFullProgress_;
        for (uint32_t i = blockIdx_; i < nLoop; i += blockDim_) {
            ComputeBMM(i);

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        PipeBarrier<PIPE_ALL>();
        return true;
    }
    return false;
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutDirect(uint64_t gmOutOffset, uint32_t startPad, uint16_t total,
                                                           uint64_t progress)
{
    uint32_t start = startPad - hPad_;
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // 处理上个核心的尾部没完成的batch
    if (start > total) {
        if (hAligned_ - total <= UINT16_MAX) {
            DataCopy(dstGM[gmOutOffset], outBuf_,
                     {widthBlockTotal_, total, 0, static_cast<uint16_t>(hAligned_ - total)});
        } else {
            for (uint16_t i = 0; i < widthBlockTotal_; i++) {
                DataCopy(dstGM[gmOutOffset + hAligned_ * c0_ * i], outBuf_[total * c0_ * i], {1, total, 0, 0});
            }
        }
        return;
    } else if (start == total) {
        DataCopy(dstGM[gmOutOffset], outBuf_,
                 {widthBlockTotal_, uint16_t(start), uint16_t(hBuffer_ - start), uint16_t(hAligned_ - start)});
        DataCopy(dstGM[gmOutOffset + start * c0_], zeroBuf_, {widthBlockTotal_, hPad_, 0, uint16_t(height_)});
        return;
    }

    if (startPad != hAligned_) {
        DataCopy(dstGM[gmOutOffset], outBuf_,
                 {widthBlockTotal_, uint16_t(start), uint16_t(hBuffer_ - start), uint16_t(hAligned_ - start)});
        DataCopy(dstGM[gmOutOffset + start * c0_], zeroBuf_, {widthBlockTotal_, hPad_, 0, uint16_t(height_)});
        gmOutOffset += startPad * c0_ + (widthBlockTotal_ - 1) * hAligned_ * c0_;

    } else {
        start = 0;
    }
    // 处理完整的batch
    uint32_t nLoop = (total - start) / height_;
    uint16_t res = (total - start) % height_;

    if (height_ <= total - start) {
        for (int i = 0; i < nLoop; i++) {
            DataCopy(dstGM[gmOutOffset], outBuf_[start * c0_ + height_ * c0_ * i],
                     {widthBlockTotal_, uint16_t(height_), uint16_t(hBuffer_ - height_), hPad_});
            DataCopy(dstGM[gmOutOffset + height_ * c0_], zeroBuf_, {widthBlockTotal_, hPad_, 0, uint16_t(height_)});
            gmOutOffset += hAligned_ * c0_ * widthBlockTotal_;
        }
    }
    // 处理尾部余下的batch
    if (res) {
        if (hAligned_ - total <= UINT16_MAX) {
            DataCopy(dstGM[gmOutOffset], outBuf_[start * c0_ + height_ * c0_ * nLoop],
                     {widthBlockTotal_, res, uint16_t(hBuffer_ - res), uint16_t(hAligned_ - res)});
        } else {
            for (uint16_t i = 0; i < widthBlockTotal_; i++) {
                DataCopy(dstGM[gmOutOffset + hAligned_ * c0_ * i],
                         outBuf_[start * c0_ + height_ * c0_ * nLoop + total * c0_ * i], {1, res, 0, 0});
            }
        }
    }
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutPageInit(uint64_t &gmOutOffset, uint32_t startPad,
                                                             uint32_t &bufOffset)
{
    uint32_t start = startPad - hPad_;
    uint32_t startSize = start * c0_;

    for (int k = 0; k < widthBlockTotal_; k++) {
        if (start > 0) {
            Copy(midBuf_[startPad * c0_ * k], outBuf_[hBuffer_ * c0_ * k], startSize);
            Duplicate(midBuf_[startPad * c0_ * k + startSize], T(0), padSize_);
        } else {
            Duplicate(midBuf_[startPad * c0_ * k], T(0), startPad * c0_);
        }
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(dstGM[gmOutOffset], midBuf_, {widthBlockTotal_, uint16_t(startPad), 0, uint16_t(hAligned_ - startPad)});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

    bufOffset = startSize;
    gmOutOffset += startPad * c0_ + (widthBlockTotal_ - 1) * hAligned_ * c0_;
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutMakePage(uint32_t nLoop, uint32_t &bufOffset)
{
    for (int j = 0; j < nLoop; j++) {
        for (int k = 0; k < widthBlockTotal_; k++) {
            Copy(midBuf_[c0_ * hAligned_ * (k + widthBlockTotal_ * j)],
                 outBuf_[bufOffset + hBuffer_ * c0_ * k + c0_ * height_ * j], height_ * c0_);
            Duplicate(midBuf_[height_ * c0_ + c0_ * hAligned_ * (k + widthBlockTotal_ * j)], T(0), padSize_);
        }
    }
    bufOffset += c0_ * height_ * nLoop;
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutPageMainImp(uint64_t &gmOutOffset, uint32_t nLoop,
                                                                uint32_t &bufOffset)
{
    CopyOutMakePage(nLoop, bufOffset);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(dstGM[gmOutOffset], midBuf_, hAligned_ * widthBlockTotal_ * c0_ * nLoop);

    gmOutOffset += hAligned_ * widthBlockTotal_ * c0_ * nLoop;
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutPageMain(uint64_t &gmOutOffset, uint32_t mPage, uint32_t startPad,
                                                             uint32_t total, uint32_t &bufOffset, uint64_t progress)
{
    uint32_t mPage2 = mPage / widthBlockTotal_;
    uint32_t nLoopIn = mPage2 / hAligned_;
    uint32_t mFinal = ((total - startPad + hPad_) / height_ + 1) * hPad_ + total;
    uint32_t nFull = (startPad == hAligned_) ? mFinal / hAligned_ : (mFinal - startPad) / hAligned_;
    uint32_t nLoopOut = nFull / nLoopIn;

    for (int i = 0; i < nLoopOut; i++) {
        CopyOutPageMainImp(gmOutOffset, nLoopIn, bufOffset);
    }
    uint32_t nLoopTail = nFull % nLoopIn;

    CopyOutPageMainImp(gmOutOffset, nLoopTail, bufOffset);
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutPageEnd(uint64_t gmOutOffset, uint32_t res, uint32_t &bufOffset)
{
    for (int k = 0; k < widthBlockTotal_; k++) {
        Copy(midBuf_[c0_ * res * k], outBuf_[bufOffset + hBuffer_ * c0_ * k], res * c0_);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(dstGM[gmOutOffset], midBuf_, {uint16_t(widthBlockTotal_), uint16_t(res), 0, uint16_t(hAligned_ - res)});
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutPage(uint64_t gmOutOffset, uint32_t mPage, uint32_t total,
                                                         uint32_t startPad, uint64_t progress)
{
    uint32_t bufOffset = 0;
    uint32_t start = startPad - hPad_;

    if (startPad != hAligned_) {
        CopyOutPageInit(gmOutOffset, startPad, bufOffset);
    } else {
        start = 0;
    }

    uint32_t res = (total - start) % height_;

    CopyOutPageMain(gmOutOffset, mPage, startPad, total, bufOffset, progress);

    if (res) {
        CopyOutPageEnd(gmOutOffset, res, bufOffset);
    }
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::CopyOutBatchReform(uint64_t gmOutOffset, uint32_t mPage, uint32_t total,
                                                                uint32_t startPad, uint64_t progress)
{
    uint32_t mPage2 = mPage / widthBlockTotal_;

    if (hAligned_ > mPage2) {
        CopyOutDirect(gmOutOffset, startPad, total, progress);
        return;
    }
    CopyOutPage(gmOutOffset, mPage, total, startPad, progress);
}

template <class T>
__aicore__ inline void Mc2KernelND2NZBMM<T>::ComputeBMM(uint64_t progress)
{
    if (noPadD_) {
        CopyIn(progress, outBuf_);
    } else {
        CopyIn(progress, inBuf_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (wTail_ == 0) { // 内轴32B对齐，大块搬入，再重排，当前实现可能有问题
            PadDAligned<T>(progress, outBuf_, inBuf_, 0, 0, width_, c0_, hBlockNum_, false);
        } else {
            PadDMain<T>(progress, outBuf_, inBuf_, midBuf_, zeroBuf_, 0, 0, width_, c0_, hBlockNum_, copyInRepeat_,
                        hBuffer_, wTail_, false);
        }
    }
    PipeBarrier<PIPE_ALL>();

    uint64_t gmOutOffset =
        (hBuffer_ * progress) / height_ * hAligned_ * widthBlockTotal_ * c0_ + ((hBuffer_ * progress) % height_) * c0_;
    uint32_t total = (progress == nFullProgress_) ? heightTotalTail_ : hBuffer_;
    uint32_t startPad = hAligned_ - (progress * hBuffer_) % height_;
    uint32_t mPage = (hBuffer_ * width_) / c0_ / ALIGNED_H * ALIGNED_H;

    CopyOutBatchReform(gmOutOffset, mPage, total, startPad, progress);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

#if defined(__DAV_C220_VEC__)
template <class T>
__aicore__ inline bool Mc2Nd2nzVnchwBMM(GlobalTensor<T> &dst, GlobalTensor<T> &src, uint32_t height, uint32_t width,
                                        uint32_t batch, TBuf<TPosition::VECCALC> &ubBuffer, uint32_t usedCoreNum)
{
    Mc2KernelND2NZBMM<T> op;
    op.Init((GM_ADDR)dst[0].GetPhyAddr(), (GM_ADDR)src[0].GetPhyAddr(), height, width, batch, ubBuffer, usedCoreNum);
    return op.ProcessBMM();
}

template <>
__aicore__ inline bool Mc2Nd2nzVnchwBMM(GlobalTensor<bfloat16_t> &dst, GlobalTensor<bfloat16_t> &src, uint32_t height,
                                        uint32_t width, uint32_t batch, TBuf<TPosition::VECCALC> &ubBuffer,
                                        uint32_t usedCoreNum)
{
    GlobalTensor<half> dstGlobalTrans;
    GlobalTensor<half> srcGlobalTrans;
    dstGlobalTrans.SetGlobalBuffer((__gm__ half *)dst.GetPhyAddr(0));
    srcGlobalTrans.SetGlobalBuffer((__gm__ half *)src.GetPhyAddr(0));
    return Mc2Nd2nzVnchwBMM(dstGlobalTrans, srcGlobalTrans, height, width, batch, ubBuffer, usedCoreNum);
}
#endif

#endif
