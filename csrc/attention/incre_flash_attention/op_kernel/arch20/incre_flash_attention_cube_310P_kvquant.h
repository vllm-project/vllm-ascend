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
 * \brief
 */
#ifndef INCRE_FLASH_ATTENTION_CUBE_310P_KVQUANT
#define INCRE_FLASH_ATTENTION_CUBE_310P_KVQUANT

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../ifa_public_define.h"

using namespace AscendC;

template <typename Q_T, typename SrcTensor>
__aicore__ inline void  CopyND2NZOnTheFly(
    const LocalTensor<Q_T>& dst,  const SrcTensor& src, const int height,
    const int width, const int gCol) {
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t calcWidth = width / BLOCK_CUBE;  
    int32_t calcHeightAlign = (height + BLOCK_CUBE - 1) / BLOCK_CUBE;
    constexpr uint32_t UB_ALIGN_NZ = 32U;
    if (height % BLOCK_CUBE != 0) {
        int32_t repeat = calcWidth * calcHeightAlign;
        Duplicate<Q_T>(dst, static_cast<Q_T>(0), repeat);
        PipeBarrier<PIPE_MTE2>();
    }
    int src_gap = gCol * sizeof(Q_T) / UB_ALIGN_NZ - 1;
    for (int i = 0; i < calcWidth; i++) {
        dstOffset = i * calcHeightAlign * CUBE_MAX_SIZE;
        srcOffset = i * BLOCK_CUBE;
        DataCopy<Q_T>(dst[dstOffset], src[srcOffset],
                 { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(src_gap), 0});
    }
}

template <typename C_T, typename A_T, typename B_T>
class MacroMatmulIFAGEMM {
public:
    inline __aicore__ MacroMatmulIFAGEMM();
    inline __aicore__ ~MacroMatmulIFAGEMM();
    // args
    uint64_t useL0PingPong_ = 1U;
    uint16_t sAL1M_;
    uint16_t sAL1K_;
    uint16_t sAL1MOffset_ = 0;
    uint16_t sAL1KOffset_ = 0;
    uint16_t sBL1N_;
    uint16_t sBL1K_;
    uint16_t sBL1NOffset_ = 0;
    uint16_t sBL1KOffset_ = 0;
    uint16_t sMadM_;
    uint16_t sMadN_;
    uint16_t sMadK_;
    uint16_t sMad0K_;
    uint16_t sL0cInit_; // 0; normal  1:init
    uint16_t sL0cLast_; // 0; normal  1:last
    // feature map
    constexpr static uint16_t sFmH_ = 1;
    // state
    uint16_t ssAl0PingPongFlag_;
    uint16_t ssBl0PingPongFlag_;
    // instance args
    // 0:format(M, K)
    // 1:format(K, M), need set transpose
    uint16_t ssAmatrixTranspose_;
    // 0:format(K, N), use load3dv2 carry
    // 1:format(N, K), use load2d carry
    uint16_t ssBmatrixTranspose_;
    // 0: bias
    // 1: no bias
    uint16_t biasType_;
    constexpr static uint16_t typeSize_ = sizeof(A_T);
    uint16_t isGemv_ = 0U;
    event_t eventIdMToMte1Ping_;
    event_t eventIdMToMte1Pong_;
    // tpipe
    TBuf<TPosition::A2> l0aBuf_;
    TBuf<TPosition::B2> l0bBuf_;

    constexpr static uint16_t HW_N0 = 16;
    constexpr static uint16_t HW_M0 = 16;
    constexpr static uint16_t ALIGN_NUM = 16;
    constexpr static uint16_t NUM_ELEMENTS_ONE_BLK_FP16 = 16;
    constexpr static uint16_t NUM_ELEMENTS_ONE_BLK_FP32 = 8; 
    constexpr static uint16_t NUM_ELEMENTS_ONE_BLK_INT8 = 32; 

    inline __aicore__ void Init();
    inline __aicore__ void ComputeNz(const LocalTensor<A_T> &l1AMatrix, const LocalTensor<B_T> &l1BMatrix,
        const LocalTensor<C_T> &cMatrix);
    inline __aicore__ void SetAccmulate(bool accmu=true);
    inline __aicore__ void  SetShapeNz(const uint16_t m, const uint16_t n, const uint16_t k, const uint16_t isATrans=0, const uint16_t isBTrans=0);
    inline __aicore__ void CopyNz2Ub(const LocalTensor<C_T> &ubMatrix, const LocalTensor<C_T> &cMatrix, uint32_t mAligned, uint32_t nAligned, uint32_t dstStride=0);
private:
    inline __aicore__ void LoadL12L0A(uint64_t aPoskPtr, uint16_t usedK,
        const LocalTensor<A_T> &l1A, LocalTensor<A_T> &l0A);
    inline __aicore__ void LoadL12L0B(uint64_t kInner, uint16_t kC0, uint16_t kC0Tail,
        const LocalTensor<B_T> &l1B, LocalTensor<B_T> &l0B);
    inline __aicore__ void MmadMacro(const LocalTensor<A_T> &l0A, const LocalTensor<B_T> &l0B,
        const LocalTensor<C_T> &cMatrix, uint16_t mmadK, bool isBias);
    inline __aicore__ void MmadGemv(const LocalTensor<A_T> &l0A, const LocalTensor<B_T> &l0B,
        const LocalTensor<C_T> &cMatrix, uint16_t mmadK, bool isBias);
    inline __aicore__ constexpr static uint16_t GetHwK0()
    {
        if constexpr (IsSameType<C_T, float>::value && sizeof(A_T) == sizeof(half)) {
            return NUM_ELEMENTS_ONE_BLK_FP16;
        } else if constexpr (IsSameType<C_T, float>::value && IsSameType<A_T, float>::value) {
            return NUM_ELEMENTS_ONE_BLK_FP32;
        } else {
            return NUM_ELEMENTS_ONE_BLK_INT8;
        }
    }
    __aicore__ inline static uint16_t myCeilDiv(uint16_t num1, uint16_t num2)
    {
        return (num2 == 0) ? 0 : (num1 + num2 - 1) / num2;
    }

    __aicore__ inline static uint16_t myCeilAlign(uint16_t num1, uint16_t num2)
    {
        return myCeilDiv(num1, num2) * num2;
    }
};

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ MacroMatmulIFAGEMM<C_T, A_T, B_T>::MacroMatmulIFAGEMM()
{
    eventIdMToMte1Ping_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
    eventIdMToMte1Pong_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ MacroMatmulIFAGEMM<C_T, A_T, B_T>::~MacroMatmulIFAGEMM()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Ping_);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Pong_);
}

template <typename C_T, typename A_T, typename B_T> inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::
CopyNz2Ub(const LocalTensor<C_T> &ubMatrix, const LocalTensor<C_T> &cMatrix, uint32_t mAligned, uint32_t nAligned, uint32_t dstStride){
    DataCopyParams repeatParams;
    DataCopyEnhancedParams enhancedParams;
    if (!isGemv_) {
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        repeatParams.blockLen = (mAligned / 16) * ( nAligned / 16);
        repeatParams.blockCount = 1;
    } else {
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        repeatParams.blockLen = nAligned / 16;
        repeatParams.blockCount = mAligned;
        repeatParams.dstStride = dstStride;
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_V));
    event_t eventIdVToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_M));
    SetFlag<HardEvent::M_V>(eventId);
    WaitFlag<HardEvent::M_V>(eventId);
    DataCopy(ubMatrix, cMatrix, repeatParams, enhancedParams);
    SetFlag<HardEvent::V_M>(eventIdVToM);
    WaitFlag<HardEvent::V_M>(eventIdVToM);
    PipeBarrier<PIPE_V>(); 
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::MmadMacro(
    const LocalTensor<A_T> &l0A, const LocalTensor<B_T> &l0B, const LocalTensor<C_T> &cMatrix,
    uint16_t mmadK, bool isBias)
{
    uint16_t madM = sMadM_;
    MmadParams mmadParams;
    mmadParams.m = madM;
    mmadParams.k = mmadK;
    mmadParams.n = sMadN_;
    mmadParams.cmatrixInitVal = isBias;
    Mmad(cMatrix, l0A, l0B, mmadParams);
    if ((madM / ALIGN_NUM) * (sMadN_ / ALIGN_NUM) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::MmadGemv(
const LocalTensor<A_T> &l0A, const LocalTensor<B_T> &l0B, const LocalTensor<C_T> &cMatrix, uint16_t mmadK, bool isBias)
{
    MmadParams mmadParams;
    mmadParams.m = 1;
    mmadParams.k = mmadK;
    mmadParams.n = sMadN_;
    mmadParams.cmatrixInitVal = isBias;
    for(uint16_t i=0; i < sMadM_; i++) {
        Mmad(cMatrix[i * 16 * sMadN_], l0A[i * BYTE_PER_FRACTAL / sizeof(A_T)], l0B, mmadParams);
    }
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::LoadL12L0A(uint64_t aPoskPtr, 
    uint16_t usedK, const LocalTensor<A_T> &l1A, LocalTensor<A_T> &l0A)
{
    constexpr uint8_t padList[4] = {0, 0, 0, 0};
    if (ssAmatrixTranspose_ > 0) {
        uint16_t wAlign = CeilAlign(sAL1K_, HW_M0);
        Load3DSetFMatrixCal(sFmH_, wAlign, padList);
    } else {
        // fmatrix w should be 16 aligned
        uint16_t wAlign = CeilAlign(sAL1M_, HW_M0);
        Load3DSetFMatrixCal(sFmH_, wAlign, padList);
    }
    if (isGemv_) {
        int32_t fracSize = BYTE_PER_FRACTAL / sizeof(A_T);
        int32_t repeat = myCeilDiv(usedK, fracSize) * sMadM_;
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = repeat;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = 0;
        LoadData(l0A[0], l1A[aPoskPtr], loadDataParams);
        return;
    }
    if (ssAmatrixTranspose_ > 0) {
        // format(K, M), K, M need to be 16 aligned for f32
        uint16_t madMAlign = CeilAlign(sMadM_, ALIGN_NUM);
        uint16_t usedKAlign = CeilAlign(usedK, HW_M0);
        uint16_t sAL1MAlign = CeilAlign(sAL1M_, ALIGN_NUM);
        // K_axis is m direction, and M_axis is k direction in load3d intrin
        LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = sAL1MAlign;
        loadData3DV2.extConfig = ((uint64_t)aPoskPtr << 48) | ((uint64_t)sAL1MOffset_ << 32) |
                                 ((uint64_t)usedKAlign << 16) | (uint64_t)madMAlign;
        loadData3DV2.enTranspose = true;
        LoadData<A_T>(l0A[0], l1A[0], loadData3DV2);
    } else {
        // format(M, K), K_axis is k direction, and M_axis is m direction in load3d intrin
        uint16_t madMAlign = CeilAlign(sMadM_, HW_M0);
        // k direction need to be 8 aligned for f32
        uint16_t usedKAlign = CeilAlign(usedK, GetHwK0());
        uint16_t sAL1KAlign = CeilAlign(sAL1K_, GetHwK0());
        LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = sAL1KAlign;
        loadData3DV2.extConfig = ((uint64_t)sAL1MOffset_ << 48) | ((uint64_t)aPoskPtr << 32) |
                                 ((uint64_t)madMAlign << 16) | (uint64_t)usedKAlign;
        LoadData<A_T>(l0A[0], l1A[0], loadData3DV2);
    }
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::LoadL12L0B(uint64_t kInner,
    uint16_t kC0, uint16_t kC0Tail, const LocalTensor<B_T> &l1B, LocalTensor<B_T> &l0B)
{
    constexpr uint8_t padList[4] = {0, 0, 0, 0};
    if (ssBmatrixTranspose_ < 1) {
        uint16_t wAlign = myCeilAlign(sBL1K_, HW_M0); // singlecore k
        Load3DSetFMatrixCal(sFmH_, wAlign, padList); // 1
    } else {
        uint16_t wAlign = myCeilAlign(sBL1N_, HW_M0);
        Load3DSetFMatrixCal(sFmH_, wAlign, padList);
    }
    bool isTail = kC0Tail != 0; // k % 16
    uint16_t nFraC0 = myCeilDiv(sMadN_, HW_N0); // sMadN_ =  baseN
    if (ssBmatrixTranspose_ > 0) {
        // SET LOAD2D parameters , loop axis: K or M, or 1
        // k is hwK0_ aligned for f32
        uint16_t l0bLoop = 1;
        uint64_t l0bSrcAddrStride = 0;
        uint64_t l0bDstAddrStride = 0;
        uint8_t l0bRepeat = kC0 * nFraC0;
        uint16_t l0bSrcstride = 1;
        uint16_t l0bDststride = 0;

        if (nFraC0 * HW_N0 == sBL1N_) {
            l0bLoop = 1;            // loop=1
            if (isTail) {
                l0bRepeat = kC0Tail * nFraC0;
            }
        } else if (nFraC0 >= kC0) { // LOOP is K  and repeat is n axis
            l0bLoop = isTail ? kC0Tail : kC0;
            l0bSrcAddrStride = sBL1N_ * GetHwK0() * typeSize_;
            l0bDstAddrStride = nFraC0 * HW_N0 * GetHwK0() * typeSize_;
            l0bRepeat = nFraC0;

            l0bSrcstride = 1;
            l0bDststride = 0;
        } else { // LOOP is N  and repeat is K axis
            l0bLoop = nFraC0;
            l0bSrcAddrStride = HW_N0 * GetHwK0() * typeSize_;
            l0bDstAddrStride = HW_N0 * GetHwK0() * typeSize_;
            l0bRepeat = isTail ? kC0Tail : kC0;

            l0bSrcstride = (sBL1N_ + HW_N0 - 1) / HW_N0;
            l0bDststride = nFraC0 - 1;
        }

        // use load2d for L1_2_L0B
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = l0bRepeat;
        loadDataParams.srcStride = l0bSrcstride;
        loadDataParams.dstGap = l0bDststride;
        loadDataParams.ifTranspose = 0;
        uint64_t l1bOffset = sBL1NOffset_ * GetHwK0() + sBL1KOffset_ * sBL1N_ +
            kInner * kC0 * GetHwK0() * sBL1N_;
        uint64_t l0bOffset = 0;
        for (uint64_t i = 0; i < l0bLoop; i++) {
            LoadData(l0B[l0bOffset], l1B[l1bOffset], loadDataParams);
            l1bOffset += (l0bSrcAddrStride / typeSize_);
            l0bOffset += (l0bDstAddrStride / typeSize_);
        }
    } else {
        // use load3dv2 for L1_2_L0B
        // n_axis is K direction, need to be 16 aligned
        uint16_t kAlign = isTail ? nFraC0 * HW_N0 : myCeilAlign(sMadN_, GetHwK0());
        uint16_t mPos = sBL1KOffset_ + kInner * sMad0K_;
        // channel size need to be 16 aligned
        uint16_t cAlign = isTail ? static_cast<uint16_t>(sBL1N_) : myCeilAlign(sBL1N_ + sBL1NOffset_, ALIGN_NUM);
        // k_axis is M direction, need to be HW_M0 aligned
        uint16_t mAlign = isTail ? kC0Tail * GetHwK0() : myCeilAlign(sMad0K_, HW_M0);

        // k direction need to be 8 aligned for f32
        // StepN need to be aligned
        LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = cAlign;
        loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) |
                                ((uint64_t)mAlign << 16) | (uint64_t)kAlign;
        LoadData<B_T>(l0B[0], l1B[0], loadData3DV2);
    }
}

// initialization
template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::Init()
{
    ssAl0PingPongFlag_ = 0;
    ssBl0PingPongFlag_ = 0;

    ssAmatrixTranspose_ = 0;
    biasType_ = 1;
    isGemv_ = 0U;

    sL0cInit_ = 0;
    sL0cLast_ = 0;
    sAL1M_ = 0U;
    sAL1K_ = 0U;
    sAL1MOffset_ = 0U;
    sAL1KOffset_ = 0U;
    sBL1N_ = 0U;
    sBL1K_= 0U;
    sBL1NOffset_ = 0U;
    sBL1KOffset_ = 0U;
    GetTPipePtr()->InitBuffer(l0aBuf_, L0AUF_SIZE);
    GetTPipePtr()->InitBuffer(l0bBuf_, L0BUF_SIZE);
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::SetAccmulate(bool accum){
    biasType_ = !accum;
    sL0cInit_ = accum;
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::
SetShapeNz(const uint16_t m, const uint16_t n, const uint16_t k, const uint16_t isATrans, const uint16_t isBTrans)
{
    isGemv_ = m < GetHwK0() ? 1U : 0U;
    ssBmatrixTranspose_ = isBTrans;
    ssAmatrixTranspose_ = isGemv_ ? 0 : isATrans;
    sMadM_ = m;
    sMadN_ = n;
    sMadK_ = k;
    sMad0K_= k;
    sBL1N_ = n;
    sBL1K_ = k;    
    sAL1M_ = m;
    sAL1K_ = k;
}

template <typename C_T, typename A_T, typename B_T>
inline __aicore__ void MacroMatmulIFAGEMM<C_T, A_T, B_T>::ComputeNz(
    const LocalTensor<A_T> &l1AMatrix, const LocalTensor<B_T> &l1BMatrix, const LocalTensor<C_T> &cMatrix)
{   
    SetFlag<HardEvent::M_MTE1>(eventIdMToMte1Ping_);
    SetFlag<HardEvent::M_MTE1>(eventIdMToMte1Pong_);
    uint64_t kC0 = sMad0K_ / GetHwK0();
    uint64_t kLoop = sMadK_ / sMad0K_;       // sMad0K_ loop times
    uint64_t kTail = sMadK_ - kLoop * sMad0K_;
    uint16_t sMad0KAlign = CeilAlign(sMad0K_, GetHwK0());
    uint16_t kC0Norm = sMad0KAlign / GetHwK0();

    LocalTensor<A_T> l0a;
    LocalTensor<B_T> l0b;
    event_t eventIDVToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_M));
    SetFlag<HardEvent::V_M>(eventIDVToM);
    WaitFlag<HardEvent::V_M>(eventIDVToM);
    for (uint64_t kInner = 0; kInner < kLoop; kInner++) {
        l0a = l0aBuf_.Get<A_T>();
        l0b = l0bBuf_.Get<B_T>();
        if ((ssAl0PingPongFlag_ & 0x1) != 0) {
            l0a = l0a[L0AUF_SIZE / 2 / sizeof(A_T)];
            l0b = l0b[L0BUF_SIZE / 2 / sizeof(B_T)];
        }
        event_t eventIdMToMte1PingPong = (ssAl0PingPongFlag_ & 0x1) ? eventIdMToMte1Pong_ : eventIdMToMte1Ping_;
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1PingPong);
        // load L0A
        uint64_t aPoskPtr = kInner * kC0 * GetHwK0() + sAL1KOffset_;
        LoadL12L0A(aPoskPtr, sMad0K_, l1AMatrix, l0a);
        // load L0B
        LoadL12L0B(kInner, kC0Norm, 0, l1BMatrix, l0b);
        SetFlag<HardEvent::MTE1_M>(ssAl0PingPongFlag_ & 0x1);
        WaitFlag<HardEvent::MTE1_M>(ssAl0PingPongFlag_ & 0x1);
        // MAD
        bool biasType = (kInner == 0) && biasType_;
        if (isGemv_) {
            MmadGemv(l0a, l0b, cMatrix, sMad0K_, biasType);
        } else {
            MmadMacro(l0a, l0b, cMatrix, sMad0K_, biasType);
        }
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1PingPong);
        // update pingpong flag
        ssAl0PingPongFlag_ += useL0PingPong_;
        ssBl0PingPongFlag_ += useL0PingPong_;
    }
    // k  tail
    if (kTail != 0) {
        uint16_t madKC0 = CeilDiv(sMadK_, GetHwK0());
        uint64_t kC0Tail = madKC0 - kLoop * kC0; // lopp times of tail block, unit is 16

        l0a = l0aBuf_.Get<A_T>();
        l0b = l0bBuf_.Get<B_T>();
        if ((ssAl0PingPongFlag_ & 0x1) != 0) {
            l0a = l0a[L0AUF_SIZE / 2 / sizeof(A_T)];
            l0b = l0b[L0BUF_SIZE / 2 / sizeof(B_T)];
        }
        event_t eventIdPingPong = (ssAl0PingPongFlag_ & 0x1) ? eventIdMToMte1Pong_ : eventIdMToMte1Ping_;
        WaitFlag<HardEvent::M_MTE1>(eventIdPingPong);
        uint16_t tailK = kC0Tail * GetHwK0();
        uint64_t aPoskPtr = kLoop * kC0 * GetHwK0() + sAL1KOffset_;
        // load L0A
        LoadL12L0A(aPoskPtr, tailK, l1AMatrix, l0a);
        // load L0B
        LoadL12L0B(kLoop, kC0, kC0Tail, l1BMatrix, l0b);

        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        // MAD
        bool biasType = (kLoop == 0) && biasType_;
        if (isGemv_) {
            MmadGemv(l0a, l0b, cMatrix, kTail, biasType);
        } else {
            MmadMacro(l0a, l0b, cMatrix, kTail, biasType);
        }
        SetFlag<HardEvent::M_MTE1>(eventIdPingPong);
        ssAl0PingPongFlag_ += useL0PingPong_;
        ssBl0PingPongFlag_ += useL0PingPong_;
    }
    WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1Ping_);
    WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1Pong_);
}

template <typename IFAT> 
class IncreFlashAttentionMulAttenCube310P {
public:
    __aicore__ inline IncreFlashAttentionMulAttenCube310P(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const IncreFlashAttentionTilingData *__restrict tiling, TPipe *tPipe);
        __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    using MM_IN_T = half;
    static constexpr bool KV_CONTINUOUS = IFAT::kvContinuous;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化 K V per-channel
    static constexpr uint8_t PER_TOKEN_MODE = 1; // 伪量化 K V per-token
    static constexpr uint8_t ANTIQUANT_MODE = IFAT::antiquantMode;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr bool ANTIQUANT_PER_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using ANTIQ_PARAMS_T = typename AscendC::Conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;
    // define pse datetype
    using pseShiftType = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

protected:
    const IncreFlashAttentionTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int32_t> blockTableGm;
    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;
    // PSE
    GlobalTensor<pseShiftType> pseShiftGm;

    // antiquant
    GlobalTensor<ANTIQ_PARAMS_T> antiqOffsetGm;
    GlobalTensor<ANTIQ_PARAMS_T> antiqScaleGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // out quant
    GlobalTensor<float> quantScale2Gm;
    GlobalTensor<float> quantOffset2Gm;
    // workspace
    GlobalTensor<KV_T> queryPreProcessResGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> logSumExpGm;
#if (__CCE_AICORE__ == 200)
    GlobalTensor<int32_t> syncGlobal;
#endif
    // kv_left_padding
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // queue
    TBuf<> ubInBuff;
    TBuf<> ubOutBuff;
    TBuf<> ubInBuff2;
    TBuf<> scaleBuff;
    TQue<QuePosition::VECIN, 1> maskQue;
    TQue<QuePosition::VECIN, 1> pseQue;

    TBuf<> tmpBuff2; 
    TBuf<> tmpBuff3; 
    TBuf<> tmpBuff5; // 32K
    // 常驻tbuf
    TBuf<QuePosition::A1> queryBuff;// 2K
    TBuf<QuePosition::A1> l1bBuff;// 2K
    TBuf<QuePosition::A1> l1aBuff;// 2K
    TBuf<QuePosition::CO1> L0CBuff;       
    TBuf<> bmm1ResBuff;     // 32K // 20k enough
    TBuf<> bmm2ResBuff;     // 2K
    TBuf<> softmaxMaxBuff;  // 32B
    TBuf<> softmaxExpBuff;  // 32B
    TBuf<> softmaxSumBuff;  // 32B

    LocalTensor<Q_T>  l1a;
    LocalTensor<Q_T>  scaleUb;
    LocalTensor<Q_T> queryL1;
    LocalTensor<T> bmm1ResUb;
    LocalTensor<T> bmm2ResUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<T> l0c;
    LocalTensor<KV_T> ubTransIn;
    LocalTensor<KV_T> ubTransInV;
    LocalTensor<MM_IN_T> ubTransOut;
    LocalTensor<MM_IN_T> l1b;
    LocalTensor<bool> maskUb;
    LocalTensor <pseShiftType> pseShiftUb;
    bool selectMaskFlag = false;
    bool useGemv = false;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM = 512 / sizeof(KV_T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM_THRESHLOD = 128 / sizeof(KV_T);
    static constexpr T antiquantExpandCoeff = 254;
    static constexpr T antiqCoeff1 = 127;
    static constexpr T antiqCoeff2 = 1 / antiqCoeff1;
    static constexpr T SOFTMAX_MIN_NUM = -2e38f;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0f; // 用于mask为bool类型
    static constexpr T FP16_ATTEN_MASK_SCALAR_VALUE = -10000.0f;           // 用于mask为fp16类型
    static constexpr uint32_t MASK_SIZE = BUFFER_SIZE_BYTE_1K * 10;
    static constexpr uint32_t SPLIT_ALIGN = 32; 
    bool antiqOffsetExistFlag = false;
    uint32_t antiquantPerTensorFlag = 0U;
    uint64_t sUnitSize = 0;

    // copy kv param
    uint32_t copyKeySplitS = 0;
    uint32_t copyKeyActSplitS = 0;
    uint32_t copyKeyLoopCount = 0;
    uint32_t copyKeyTailStart = 0;
    uint32_t copyKeyTailSplitSize = 0;
    uint32_t copyValueSplitS = 0;
    uint32_t copyValueActSplitS = 0;
    uint32_t copyValueLoopCount = 0;
    uint32_t copyValueTailStart = 0;
    uint32_t copyValueTailSplitSize = 0;

    // kv_left_padding
    uint32_t kvPaddingFlag = 0;
    uint64_t kvPaddingBeginOffset = 0;
    uint64_t kvPaddingSPadDataNum = 0;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = 1;

    __gm__ uint8_t *key_ptr = nullptr;
    __gm__ uint8_t *value_ptr = nullptr;
    uint32_t tmpBlockIdx = 0U;

    // tilingdata
    uint64_t singleProcessSInnerSize = 0ULL;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0ULL;

    uint32_t blockSplitBn2Range = 0U;
    uint32_t tailBlockSplitBn2Range = 0U;
    uint32_t formerCoreNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;

    uint32_t batchContinuous = 0U;

    uint64_t batchSize = 0U;
    uint64_t qHeadNum = 0U;
    uint64_t kvHeadNum = 0U;
    uint64_t gSize = 1U;
    uint64_t gqa = 1U;
    uint64_t kvSeqSize = 0U;
    uint64_t qSeqSize = 1U; // for multi-token inputs
    uint64_t headDim = 0U;
    uint64_t headDimAlign = 0U;
    uint64_t headDimAlignFp32 = 0U;
    bool useDataCopyPad = false;
    bool padForBankConflict = false;

    // attention mask
    bool attenMaskFlag = false;
    uint32_t selectWithByteMaskTmpMinSize = 0U;
    uint32_t attenMaskSizeAlign = 0U;
    // pse mask
    bool pseShiftFlag = false;
    uint32_t pseShiftB = 0U;
    uint32_t pseShiftS = 0U;
    uint64_t pseShiftOffset = 0U;
    uint64_t pseShiftCoreOffset = 0ULL;
    uint32_t pseMaskSizeAlign = 0U;
    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t valueOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t antiqKeyParamOffset = 0ULL;
    uint64_t antiqValueParamOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t antiqKeyParamCoreOffsetPerToken = 0ULL;
    uint64_t antiqValueParamCoreOffsetPerToken = 0ULL;
    uint64_t antiqKeyParamOffsetPerToken = 0ULL;
    uint64_t antiqValueParamOffsetPerToken = 0ULL;
    uint64_t attentMaskSize = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint64_t sInnerLoopSize = 0ULL;
    uint32_t actualCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t curSingleProcessSInnerSizeAlign = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    const uint32_t *coreStartIdx = nullptr;
    uint32_t actualLenDims = 0U;

    bool curActSeqLenIsZero = false;

    // // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t s2BatchOffset = 0;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t blockTableBaseOffset = 0;

    // // softmaxlse
    bool softmaxLseFlag;

    MacroMatmulIFAGEMM<T, half, half> mm;
    bool toReadPingpong;
    event_t eventUbIn[2];

    uint64_t NDLocalList0[8][16];
    uint64_t NDLocalList1[8][16];
    uint64_t ZNLocalList[8][16];

    template <typename T> 
    __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParams();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void GetActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void CalculateSUnitSize();
    __aicore__ inline void CalculatekvPaddingSPadDataNum(int64_t &startPosition);
    __aicore__ inline bool ComputeKVPaddingBeginOffset();
    __aicore__ inline void CalculateCopyKvSplitS();
    __aicore__ inline void UpdateCopyKvParam();

    __aicore__ inline void GetBN2id(const uint32_t bn2Idx);
    __aicore__ inline void CalcBN2Offset();
    __aicore__ inline void CalcBN2Params();

    __aicore__ inline bool CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx);

    __aicore__ inline void QueryPreProcess();

    __aicore__ inline void Trans2L1(
    const LocalTensor<MM_IN_T>& dst, const int height, const int width, const int gCol, const bool pingpong);
    __aicore__ inline void CopyND2Ub(
    const GlobalTensor<KV_T>& src, const int height, const int width, const int gCol, const bool pingpong, const uint32_t startRow=0);
    __aicore__ inline void ProcessVec1(const uint32_t sInnerLoopIdx);
    __aicore__ inline void ProcessVec2(const uint32_t sInnerLoopIdx);
    __aicore__ inline void SInnerLoopFunc(const uint32_t sInnerLoopIdx);

    __aicore__ inline void SoftmaxFlashV2Compute(LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);

    __aicore__ inline void ElewiseCompute(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CopyQuery(LocalTensor<Q_T>& queryUb);
    __aicore__ inline void ElewiseComputeMask(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseComputeMaskMtp(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseComputePse(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOut(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                              uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx);
    __aicore__ inline void InitAllZeroInt8Output();
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);
    __aicore__ inline bool FastMaskMultiToken();
    __aicore__ inline bool FastMask();
    __aicore__ inline void CopyPse();
};

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1,
                                                   __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
                                                   __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale,
                                                   __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *workspace)
{
    if constexpr (ANTIQUANT) {
        antiqScaleGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)antiquantScale);
        antiqOffsetExistFlag = (antiquantOffset != nullptr);
        if (antiqOffsetExistFlag) {
            antiqOffsetGm.SetGlobalBuffer((__gm__ ANTIQ_PARAMS_T *)antiquantOffset);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSize;
    sInnerLoopTimes = tilingData->increFlashAttentionSingleCoreParams.sInnerLoopTimes;
    singleProcessSInnerSizeTail = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSizeTail;
    usedCoreNum = tilingData->increFlashAttentionSingleCoreParams.usedCoreNum;
    formerCoreNum = tilingData->increFlashAttentionSingleCoreParams.formerCoreNum;
    blockSplitBn2Range = tilingData->increFlashAttentionSingleCoreParams.blockSplitBn2Range;
    tailBlockSplitBn2Range = tilingData->increFlashAttentionSingleCoreParams.tailSplitedBatchRange;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    batchSize = tilingData->baseParams.batchSize;
    qSeqSize = tilingData->baseParams.qSeqSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    gqa = tilingData->baseParams.nNumOfQInOneGroup;
    gSize = gqa * qSeqSize;
    kvSeqSize = tilingData->baseParams.seqSize;
    headDim = tilingData->baseParams.headSize;
    batchContinuous = tilingData->baseParams.batchContinuousFlag;
    antiquantPerTensorFlag = tilingData->baseParams.antiquantPerTensorFlag;

    headDimAlign = Align(headDim, BYTE_BLOCK);
    headDimAlignFp32 = Align(headDim, BYTE_BLOCK / sizeof(T));
    padForBankConflict = (!ANTIQUANT && headDim == 256);

    attenMaskFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attentMaskSize = tilingData->baseParams.attenMaskSize;

    pseShiftFlag = (tilingData->baseParams.pseShiftFlag == 1) ? true : false;
    if (pseShiftFlag) {
        pseShiftB = tilingData->baseParams.pseShiftB;
        pseShiftS = tilingData->baseParams.pseShiftS;
    }

    kvPaddingFlag = tilingData->baseParams.kvPaddingFlag;

    // 是否输出lse
    softmaxLseFlag = tilingData->baseParams.softmaxLseFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;

    coreStartIdx = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
    useGemv = (gSize == 1) && (headDimAlign * sizeof(Q_T) <= BYTE_PER_FRACTAL);
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitBuffers()
{
    // 常驻buffer
    pipe->InitBuffer(bmm1ResBuff, BUFFER_SIZE_BYTE_1K * 40); // 20k 
    pipe->InitBuffer(ubInBuff, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(ubInBuff2, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff5, BUFFER_SIZE_BYTE_32K * 2);

    pipe->InitBuffer(bmm2ResBuff, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(tmpBuff3, BUFFER_SIZE_BYTE_4K); 
    pipe->InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(maskQue, 1, MASK_SIZE); // 40 / sizeof(T)
    pipe->InitBuffer(pseQue, 1, MASK_SIZE * sizeof(pseShiftType)); // 40 / sizeof(T)
    pipe->InitBuffer(scaleBuff, BUFFER_SIZE_BYTE_2K);

    pipe->InitBuffer(queryBuff, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(L0CBuff, BUFFER_SIZE_BYTE_32K * 8);
    pipe->InitBuffer(l1aBuff, BUFFER_SIZE_BYTE_32K * 8);
    pipe->InitBuffer(l1bBuff, BUFFER_SIZE_BYTE_32K * 8);

    queryL1 = queryBuff.Get<Q_T>();
    l1a = l1aBuff.Get<half>();
    l1b = l1bBuff.Get<half>();
    l0c = L0CBuff.Get<T>();
    bmm1ResUb = bmm1ResBuff.Get<T>();
    bmm2ResUb = bmm2ResBuff.Get<T>();
    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();
    ubTransIn = ubInBuff.Get<KV_T>();
    ubTransInV = ubInBuff2.Get<KV_T>();
    ubTransOut = tmpBuff5.Get<half>();
    mm.Init();
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CalculateCopyKvSplitS()
{
    // 1、非量化场景：KV_T=Q_T
    // 2、伪量化场景：需要CAST到Q_T，queue的大小与CAST后的buff均为32K
    uint32_t splitS = BUFFER_SIZE_BYTE_32K / sizeof(MM_IN_T) / headDimAlign;
    copyKeySplitS = (splitS / SPLIT_ALIGN) * SPLIT_ALIGN;
    if (copyKeySplitS > BYTE_PER_FRACTAL / sizeof(MM_IN_T) && useGemv) {
        copyKeySplitS = BYTE_PER_FRACTAL / sizeof(MM_IN_T);
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::UpdateCopyKvParam()
{
    // copy key param
    copyKeyActSplitS = copyKeySplitS;
    copyKeyLoopCount = (actualSingleProcessSInnerSize + copyKeyActSplitS - 1) / copyKeyActSplitS - 1;
    copyKeyTailStart = copyKeyLoopCount * copyKeyActSplitS;
    copyKeyTailSplitSize = actualSingleProcessSInnerSize - copyKeyTailStart;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitAllZeroInt8Output()
{
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitAllZeroOutput(uint32_t bIdx)
{
    uint32_t copySize = gSize * headDim;
    if constexpr (POST_QUANT) { 
        InitAllZeroInt8Output();
    } else {
        matmul::InitOutput<OUT_T>(attentionOutGm[(bIdx * kvHeadNum + n2Idx) * copySize], copySize, 0);
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::GetActualSeqLen()
{
    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::GetBN2id(const uint32_t bn2Idx)
{
    if constexpr (FLASH_DECODE) {
        bIdx = tmpBlockIdx / (kvHeadNum * splitKVNum);
        n2Idx = (tmpBlockIdx / splitKVNum) % kvHeadNum;
        s2Idx = tmpBlockIdx % splitKVNum;
    } else {
        bIdx = (beforeBlockSplitBn2Nums + bn2Idx) / kvHeadNum;
        n2Idx = (beforeBlockSplitBn2Nums + bn2Idx) % kvHeadNum;
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::UpdateInnerLoopCond()
{
    if (curActualSeqLen == 0) {
        InitAllZeroOutput(bIdx);
        curActSeqLenIsZero = true;
        return;
    } else {
        curActSeqLenIsZero = false;
    }

    int32_t remainSinnerSize = curActualSeqLen + kvPaddingSPadDataNum;
    int32_t computeSinnerSize = curActualSeqLen + kvPaddingSPadDataNum;
    if constexpr (FLASH_DECODE) {
        remainSinnerSize = (int32_t)curActualSeqLen + kvPaddingSPadDataNum - sInnerLoopSize * s2Idx;
        computeSinnerSize = remainSinnerSize >= sInnerLoopSize ? sInnerLoopSize : remainSinnerSize;
        if (tmpBlockIdx >= batchSize * kvHeadNum * splitKVNum) {
            remainSinnerSize = 0;
        }
    }
    if (remainSinnerSize > 0) {
        if (computeSinnerSize <= singleProcessSInnerSize) {
            singleProcessSInnerSizeTail = computeSinnerSize;
            sInnerLoopTimes = 1;
        } else {
            sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
        }
    } else {
        sInnerLoopTimes = 0;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionMulAttenCube310P<IFAT>::
SeqLenFromTensorList(uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)key_ptr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CalculateSUnitSize()
{
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        sUnitSize = headDim * kvHeadNum;
    } else {
        sUnitSize = headDim;
    }
    return;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CalculatekvPaddingSPadDataNum(int64_t &startPosition)
{
    if (!attenMaskFlag || startPosition <= 0) {
        return;
    }

    uint64_t kvPaddingBeginOffsetBit = startPosition * sUnitSize;
    // already align, early quit
    if (kvPaddingBeginOffsetBit % ADDRESS_ALIGN_NUM == 0) {
        return;
    }

    // pad bit need fill to 512B
    uint64_t kvPaddingSPadDataBit = kvPaddingBeginOffsetBit % ADDRESS_ALIGN_NUM;

    // reduce the size of padding data for GM address alignment.
    int64_t addrAlignNum = ADDRESS_ALIGN_NUM;
    // condition 1: the size of padding data less than 50% of the original data.
    // condition 2: the size of padding data can not exceeds 0 offset of this batch.
    while ((kvPaddingSPadDataBit < curActualSeqLen * sizeof(KV_T) >> 1) &&
           (kvPaddingBeginOffsetBit - kvPaddingSPadDataBit > 0)) {
        // if the conditions are not met, do not fill in data because there is no peak rate of GM/L2->L1 data
        // transmission.
        if (addrAlignNum < ADDRESS_ALIGN_NUM_THRESHLOD) {
            kvPaddingSPadDataBit = 0;
            break;
        }
        // kvPaddingUnit RemainSize should be a factor of sUnitSize, because remainSinnerSize and SinnerLoopTimes
        // will be recalculated.
        if (kvPaddingSPadDataBit / sizeof(KV_T) % sUnitSize == 0) {
            break;
        }
        addrAlignNum /= 2;
        kvPaddingSPadDataBit = kvPaddingBeginOffsetBit % addrAlignNum;
    }

    kvPaddingSPadDataNum = kvPaddingSPadDataBit / sizeof(KV_T) / sUnitSize;
    return;
}

template <typename IFAT> 
__aicore__ inline bool IncreFlashAttentionMulAttenCube310P<IFAT>::ComputeKVPaddingBeginOffset()
{
    if (kvPaddingFlag != 1) {
        return true;
    }
    int64_t paddingSize = kvPaddingSizeGm.GetValue(0);
    if (paddingSize < 0) {
        paddingSize = 0;
    }

    int64_t startPosition = kvSeqSize - paddingSize - curActualSeqLen;
    // Calculate GM address align dataNum
    CalculatekvPaddingSPadDataNum(startPosition);

    if (startPosition < 0) {
        InitAllZeroOutput(bIdx);
        return false;
    }

    kvPaddingBeginOffset = static_cast<uint64_t>(startPosition) - kvPaddingSPadDataNum;
    return true;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const IncreFlashAttentionTilingData *__restrict tiling, TPipe *tPipe)
{
    tmpBlockIdx = GetBlockIdx();

    pipe = tPipe;

    // init tiling data
    tilingData = tiling;
    InitTilingData();

    key_ptr = key;
    value_ptr = value;

    ListTensorDesc keyListTensorDesc((__gm__ void *)key_ptr);
    ListTensorDesc valueListTensorDesc((__gm__ void *)value_ptr);
    __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);

    // init global buffer
    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    if (softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
    if (tilingData->baseParams.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }

    // GM for pse
    if (pseShiftFlag) {
        pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType *)pseShift);
    }

    if (attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }

    InitActualSeqLen(actualSeqLengths);

    if (kvPaddingFlag == 1) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t *)kvPaddingSize);
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    InitBuffers();

    uint64_t offset = 0;
    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        logSumExpGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
#if (__CCE_AICORE__ == 200)
        syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + offset),
                                   GetBlockNum() * BYTE_BLOCK / sizeof(int32_t));
        offset = offset + GetBlockNum() * BYTE_BLOCK;
#endif
    }

    CalculateCopyKvSplitS();
}

template <typename IFAT>
 __aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitCalcParams()
{
    bn2LoopTimes = blockSplitBn2Range;
    beforeBlockSplitBn2Nums = tmpBlockIdx * blockSplitBn2Range;
    // tail core
    if (tmpBlockIdx >= formerCoreNum) {
        bn2LoopTimes = tailBlockSplitBn2Range;
        beforeBlockSplitBn2Nums =
            formerCoreNum * blockSplitBn2Range + (tmpBlockIdx - formerCoreNum) * tailBlockSplitBn2Range;
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::InitCalcParamsEach()
{
    bn2LoopTimes = coreStartIdx[tmpBlockIdx + 1] - coreStartIdx[tmpBlockIdx];
    beforeBlockSplitBn2Nums = coreStartIdx[tmpBlockIdx];
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CalcBN2Offset()
{
    if constexpr (LAYOUT_T == LAYOUT::BSND || LAYOUT_T == LAYOUT::BSH) {
        // B,1,N2,G,D
        tensorACoreOffset = bIdx * qHeadNum * qSeqSize * headDim + n2Idx * gqa * headDim;
        // B,S2,N2,D
        tensorBCoreOffset =
            bIdx * kvSeqSize * kvHeadNum * headDim + n2Idx * headDim + kvPaddingBeginOffset * kvHeadNum * headDim;

        if (!batchContinuous) {
            tensorBCoreOffset = n2Idx * headDim;
        }

        if constexpr (FLASH_DECODE) {
            tensorBCoreOffset += s2Idx * sInnerLoopSize * headDim * kvHeadNum;
        }
    } else {
        // B,N2,G,1,D
        tensorACoreOffset = bIdx * qHeadNum * qSeqSize * headDim + n2Idx * gSize * headDim;
        // B,N2,S2,D
        tensorBCoreOffset =
            bIdx * kvHeadNum * kvSeqSize * headDim + n2Idx * kvSeqSize * headDim + kvPaddingBeginOffset * headDim;

        if (!batchContinuous) {
            uint64_t seqSize = SeqLenFromTensorList(bIdx);
            tensorBCoreOffset = n2Idx * seqSize * headDim;
        }

        if constexpr (FLASH_DECODE) {
            tensorBCoreOffset += s2Idx * sInnerLoopSize * headDim;
        }
    }
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CalcBN2Params()
{
    blockTableBaseOffset = bIdx * maxBlockNumPerBatch;
    s2BatchBaseOffset = kvPaddingBeginOffset; // 需确认PA是否从左padding起始位置开始分片
    attenMaskCoreOffset = bIdx * attentMaskSize * qSeqSize + kvPaddingBeginOffset;
    if constexpr (FLASH_DECODE) {
        attenMaskCoreOffset += s2Idx * sInnerLoopSize;
        s2BatchBaseOffset += s2Idx * sInnerLoopSize;
    }
    // antiquant的offset和scale参数数据排列是先key后value
    if (antiquantPerTensorFlag == 1) {
        antiqKeyParamOffset = 0;
        antiqValueParamOffset = 1;
    } else {
        antiqKeyParamOffset = n2Idx * headDim;
        antiqValueParamOffset = kvHeadNum * headDim + n2Idx * headDim;
    }
    antiqKeyParamCoreOffsetPerToken = bIdx * kvSeqSize + kvPaddingBeginOffset;
    antiqValueParamCoreOffsetPerToken = batchSize * kvSeqSize + bIdx * kvSeqSize + kvPaddingBeginOffset;
    if constexpr (FLASH_DECODE) {
        antiqKeyParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
        antiqValueParamCoreOffsetPerToken += s2Idx * sInnerLoopSize;
    }

    if (!batchContinuous) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)key_ptr);
        ListTensorDesc valueListTensorDesc((__gm__ void *)value_ptr);
        __gm__ uint8_t *key = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        __gm__ uint8_t *value = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
        if (tilingData->baseParams.l2CacheOffFlag) {
            // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
            valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
    }
    // 更新actualSingleProcessSInnerSize，防止尾块值，影响第二次loop
    actualSingleProcessSInnerSize = singleProcessSInnerSize;
    actualSingleProcessSInnerSizeAlign = Align(singleProcessSInnerSize, BYTE_BLOCK);
    // key和value的Copy参数随actualSingleProcessSInnerSize更新
    UpdateCopyKvParam();

    if (pseShiftFlag) {
        pseShiftCoreOffset = (pseShiftB == 1) ? (n2Idx * pseShiftS * gSize) :
            (bIdx * qHeadNum * pseShiftS + n2Idx * gSize * pseShiftS);

        if constexpr (FLASH_DECODE) {
            pseShiftCoreOffset += s2Idx * sInnerLoopSize;
        }
        pseShiftCoreOffset += kvPaddingBeginOffset; // kv_padding_size
    }
}

template <typename IFAT>
__aicore__ inline bool IncreFlashAttentionMulAttenCube310P<IFAT>::FastMask() {
    bool maskAct = false;
    int numBlack = 0;
    selectMaskFlag = true;
    if(attenMaskFlag) {
        maskUb = maskQue.AllocTensor<bool>();
        DataCopy(maskUb, attenMaskBoolGm[attenMaskOffset], actualSingleProcessSInnerSizeAlign);
        maskQue.EnQue(maskUb);
        maskQue.DeQue<bool>();
        if ((actualLenDims != 0) || (kvPaddingFlag)) {
            selectMaskFlag = true;
        } else {
            LocalTensor<half> maskHalf = tmpBuff5.Get<half>();
            Cast(maskHalf, maskUb.template ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, actualSingleProcessSInnerSizeAlign);
            PipeBarrier<PIPE_V>();
            auto maskFloat = maskHalf[actualSingleProcessSInnerSizeAlign].template ReinterpretCast<float>();
            Cast(maskFloat, maskHalf, AscendC::RoundMode::CAST_NONE, actualSingleProcessSInnerSizeAlign);
            PipeBarrier<PIPE_V>();
            auto res = maskFloat[actualSingleProcessSInnerSizeAlign];
            auto tmp = res[16];
            ReduceSum(res, maskFloat, tmp, actualSingleProcessSInnerSize);
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            numBlack = static_cast<int>(res.GetValue(0));
            if (numBlack > 0) {
                ReduceSum(res, maskFloat[actualSingleProcessSInnerSize - numBlack], tmp, numBlack);
                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);
                auto cntTailOnes = static_cast<int>(res.GetValue(0));
                if (numBlack == cntTailOnes) {
                    actualSingleProcessSInnerSize -= numBlack;
                    actualSingleProcessSInnerSizeAlign = Align(actualSingleProcessSInnerSize, BYTE_BLOCK);
                    maskAct = true;
                    selectMaskFlag = false;
                }
            }
        }
    }
    return maskAct;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CopyPse() {
    auto startRow = 0;
    pseMaskSizeAlign = Align(actualSingleProcessSInnerSize, 16UL);
    pseShiftUb = pseQue.AllocTensor<pseShiftType>();
    for (uint32_t i = 0; i < gSize; ++i) {
        DataCopy(pseShiftUb[i * actualSingleProcessSInnerSizeAlign], pseShiftGm[pseShiftOffset + startRow * pseShiftS + i * pseShiftS],
                 pseMaskSizeAlign);
    }
    pseQue.EnQue(pseShiftUb);
    pseQue.DeQue<pseShiftType>();
}

template <typename IFAT>
__aicore__ inline bool IncreFlashAttentionMulAttenCube310P<IFAT>::FastMaskMultiToken() {
    bool maskAct = false;
    int numBlack = 0;
    selectMaskFlag = false;
    if(attenMaskFlag) {
        maskUb = maskQue.AllocTensor<bool>();
        for (auto i = 0; i < qSeqSize; i++) {
            DataCopy(maskUb[i * actualSingleProcessSInnerSizeAlign], attenMaskBoolGm[attenMaskOffset + i * attentMaskSize], actualSingleProcessSInnerSizeAlign);
        }
        maskQue.EnQue(maskUb);
        maskQue.DeQue<bool>();
    }
    return maskAct;
}

template <typename IFAT>
__aicore__ inline bool IncreFlashAttentionMulAttenCube310P<IFAT>::CalcSInnerOffsetAndParams(const uint32_t sInnerLoopIdx)
{
    uint64_t sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        // B,Si,N2,D
        tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * kvHeadNum * headDim;
    } else {
        tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * headDim; 
    }
    valueOffset = tensorBOffset;
    attenOutOffset = tensorACoreOffset;

    attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;
    s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;
    antiqKeyParamOffsetPerToken = antiqKeyParamCoreOffsetPerToken + sInnerOffsetDataSize;
    antiqValueParamOffsetPerToken = antiqValueParamCoreOffsetPerToken + sInnerOffsetDataSize;
    // Calc Params
    if (sInnerLoopIdx == sInnerLoopTimes - 1) {
        actualSingleProcessSInnerSize = singleProcessSInnerSizeTail;
        actualSingleProcessSInnerSizeAlign = Align(actualSingleProcessSInnerSize, BYTE_BLOCK);
    }

    bool maskAct = false;
    selectMaskFlag = true;
    if (qSeqSize == 1) {
        maskAct = FastMask();
    } else {
        maskAct = FastMaskMultiToken();
    }

    if(maskAct || (sInnerLoopIdx == sInnerLoopTimes - 1)) {
        UpdateCopyKvParam();
    }

    if (pseShiftFlag) {
        pseShiftOffset = pseShiftCoreOffset + sInnerOffsetDataSize;
        CopyPse(); // only for qseqlen = 1
    }
    return maskAct;
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CopyQuery(LocalTensor<Q_T>& queryUb){
    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        DataCopy(queryUb, queryGm[tensorACoreOffset], gSize * headDimAlign); 
    } else {
        for (auto i=0; i<qSeqSize; i++) {
            for (auto j=0; j<gqa; j++) {
                DataCopy(queryUb[(j * qSeqSize + i) * headDimAlign], queryGm[tensorACoreOffset + (i * qHeadNum + j) * headDimAlign], headDimAlign);
            }
        }
    }
    
    if constexpr (ANTIQUANT) {
        scaleUb  = queryUb[gSize * headDim];
        DataCopy(scaleUb, antiqScaleGm[antiqKeyParamOffset], {2, static_cast<uint16_t>(headDim / FP16_ONE_BLOCK_SIZE),  static_cast<uint16_t>((kvHeadNum-1)*headDim / FP16_ONE_BLOCK_SIZE), 0});
        if (antiqOffsetExistFlag) {
            DataCopy(scaleUb[headDimAlign*2], antiqOffsetGm[antiqValueParamOffset], headDim);
        }
    }
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::QueryPreProcess(){
    LocalTensor<Q_T> queryUb = tmpBuff5.Get<Q_T>();  
    CopyQuery(queryUb);
    Adds(queryUb, queryUb, static_cast<Q_T>(0.0), gSize * headDimAlign);
    PipeBarrier<PIPE_V>();
    if constexpr (ANTIQUANT) {
        for (auto k = 0; k < gSize; k++) {
            Mul(queryUb[k*headDimAlign], queryUb[k*headDimAlign], scaleUb, headDimAlign);
        }
        PipeBarrier<PIPE_V>();
        scaleUb = scaleBuff.Get<Q_T>();
        DataCopy(scaleUb[headDim], queryUb[headDim * gSize + headDim], antiqOffsetExistFlag ? headDim * 2 : headDim);
        PipeBarrier<PIPE_V>();
    }
    event_t eid1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3)); 
    SetFlag<HardEvent::V_MTE3>(eid1); 
    WaitFlag<HardEvent::V_MTE3>(eid1); 
    DataCopyParams params;
    params.blockCount = gSize;
    params.blockLen = headDimAlign * sizeof(Q_T) / BYTE_BLOCK;
    params.dstStride = (BYTE_PER_FRACTAL - headDimAlign * sizeof(Q_T)) / BYTE_BLOCK;
    if (useGemv) {
        DataCopy(queryL1, queryUb, params);
    } else {
        CopyND2NZOnTheFly(queryL1, queryUb, gSize, headDim, headDimAlign);
    }
    event_t eid2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1)); 
    SetFlag<HardEvent::MTE3_MTE1>(eid2); 
    WaitFlag<HardEvent::MTE3_MTE1>(eid2); 
    event_t eid3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V)); 
    SetFlag<HardEvent::MTE3_V>(eid3); 
    WaitFlag<HardEvent::MTE3_V>(eid3); 
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::CopyND2Ub(
    const GlobalTensor<KV_T>& src, const int height, const int width, const int gCol, const bool pingpong,  const uint32_t startRow) {
    LocalTensor<KV_T> ubIn;
    if constexpr (!ANTIQUANT) {
        ubIn = pingpong ? ubTransInV: ubTransIn;
    } else {
        ubIn = pingpong ? ubTransIn[BUFFER_SIZE_BYTE_16K] : ubTransIn;
    }

    if constexpr (PAGE_ATTENTION) {
        uint32_t dealRowCount = height,  step =  kvHeadNum * headDim;  
        uint32_t curSeqIdx = s2BatchOffset + startRow;
        uint32_t copyFinishRowCnt = 0;
        uint32_t dstOffset = 0;
        while (copyFinishRowCnt < dealRowCount) {
            uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
            uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
            uint64_t idInBlockTable =
                blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
            // 计算可以拷贝行数
            uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
            if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
                copyRowCnt = dealRowCount - copyFinishRowCnt;
            }
            uint64_t offset = (idInBlockTable * kvCacheBlockSize + reaminRowCnt) * step + (uint64_t)(n2Idx * headDim);
            if(kvHeadNum == 1) {
                DataCopy(ubIn[dstOffset], src[offset], width * copyRowCnt);
            } else {
                DataCopy(ubIn[dstOffset], src[offset], {(uint16_t)copyRowCnt, 
                                                        static_cast<uint16_t>(width * sizeof(KV_T) / BYTE_BLOCK), 
                                                        static_cast<uint16_t>((kvHeadNum-1) * width * sizeof(KV_T) / BYTE_BLOCK), 0});
            }
            dstOffset += width * copyRowCnt;
            copyFinishRowCnt += copyRowCnt;
            curSeqIdx += copyRowCnt;
        }
    } else {
        DataCopy(ubIn, src[startRow * headDim], width * height);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::Trans2L1(
    const LocalTensor<MM_IN_T>& dst, const int height, const int width, const int gCol, const bool pingpong){
    uint32_t copyTimes = (height + 16 - 1) / 16;
    uint32_t k = ((width + 16 - 1) / 16) * 16;
    uint32_t h = (copyTimes - 1) * 16;
    LocalTensor<KV_T> ubIn;
    if constexpr (!ANTIQUANT) {
        ubIn = pingpong ? ubTransInV: ubTransIn;
    } else {
        ubIn = pingpong ? ubTransIn[BUFFER_SIZE_BYTE_16K] : ubTransIn;
    }
    auto dstTran = ubTransOut.template ReinterpretCast<uint16_t>();
    auto srcTran = ubIn.template ReinterpretCast<uint16_t>();
    LocalTensor<half> ubPad;
    
    if constexpr (ANTIQUANT) {
        auto ubCast = ubTransInV.template ReinterpretCast<half>();
        Cast(ubCast, ubIn,  AscendC::RoundMode::CAST_NONE, k * copyTimes * 16);
        PipeBarrier<PIPE_V>(); 
        srcTran = ubCast.template ReinterpretCast<uint16_t>();
        ubPad = ubCast;
    } else {
        ubPad = ubIn;
    }

    if(height - h < 16){
        Duplicate(ubPad[height * k], static_cast<half>(0), (16 - height + h) * k);
        PipeBarrier<PIPE_V>(); 
    }
 
    TransposeParamsExt tpe = {(uint16_t)copyTimes, 16, 1, (uint16_t)width, TransposeType::TRANSPOSE_NCHW2NHWC};
    Transpose(dstTran, srcTran, tmpBuff5.Get<uint8_t>(), tpe);
    PipeBarrier<PIPE_V>(); 
    event_t eid1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eid1);
    WaitFlag<HardEvent::V_MTE3>(eid1);
    DataCopy(dst, ubTransOut, k * copyTimes * 16);
    event_t eid2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1));
    SetFlag<HardEvent::MTE3_MTE1>(eid2);
    WaitFlag<HardEvent::MTE3_MTE1>(eid2);
    event_t eid3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eid3);
    WaitFlag<HardEvent::MTE3_V>(eid3);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::ElewiseComputePse(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf,
                                                                               uint32_t startRow, uint32_t dealRowCount,
                                                                               uint32_t columnCount,
                                                                               uint32_t actualColumnCount)
{
    auto tmpUb = tmpBuff5.Get<T>();
    for (auto i=0; i<gSize; i++) {
        Cast(tmpUb[i*columnCount], pseShiftUb[i*columnCount], AscendC::RoundMode::CAST_NONE, columnCount);   
    }
    PipeBarrier<PIPE_V>();
    for (auto i=0; i<qSeqSize; i++) {
        for (auto j=0; j<gqa; j++) {
            uint32_t row = j * qSeqSize + i; 
            Add(bmm1ResUb[row * columnCount], bmm1ResUb[row * columnCount], tmpUb[row * columnCount], columnCount);
        }
    }
    PipeBarrier<PIPE_V>();
    pseQue.FreeTensor(pseShiftUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::ElewiseComputeMask(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf,
                                                                               uint32_t startRow, uint32_t dealRowCount,
                                                                               uint32_t columnCount,
                                                                               uint32_t actualColumnCount)
{
    SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
    selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
    selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
    selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
    LocalTensor<bool> attenMaskUb = tmpBuff5.Get<bool>();
    for(uint32_t i=0; i<dealRowCount; i++) {
        DataCopy(attenMaskUb[i*columnCount], maskUb, columnCount);
    }
    PipeBarrier<PIPE_V>();
    auto tmpUb = attenMaskUb[columnCount * dealRowCount].template ReinterpretCast<uint8_t>();
    SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskUb, tmpUb,
                            selectWithBytesMaskShapeInfo);
    PipeBarrier<PIPE_V>();
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::ElewiseComputeMaskMtp(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf,
                                                                               uint32_t startRow, uint32_t dealRowCount,
                                                                               uint32_t columnCount,
                                                                               uint32_t actualColumnCount)
{
    LocalTensor<half> maskHalf = tmpBuff5.Get<half>();
    auto maskFloat = maskHalf[columnCount].template ReinterpretCast<float>();
    for (auto i=0; i<qSeqSize; i++) {
        Cast(maskHalf, maskUb[i * columnCount].template ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, columnCount);
        PipeBarrier<PIPE_V>();
        Cast(maskFloat, maskHalf, AscendC::RoundMode::CAST_NONE, columnCount);
        PipeBarrier<PIPE_V>();
        Muls(maskFloat, maskFloat, FP16_ATTEN_MASK_SCALAR_VALUE, columnCount);
        PipeBarrier<PIPE_V>();
        for (auto j=0; j<gqa; j++) {
            uint32_t row = j * qSeqSize + i; 
            Add(bmm1ResUb[row * columnCount], bmm1ResUb[row * columnCount], maskFloat, columnCount);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::ElewiseCompute(LocalTensor<T> &mmResUb, TBuf<> &tmpBuf,
                                                                               uint32_t startRow, uint32_t dealRowCount,
                                                                               uint32_t columnCount,
                                                                               uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
    if (pseShiftFlag) {
        ElewiseComputePse(bmm1ResUb, tmpBuff5, 0, gSize, actualSingleProcessSInnerSizeAlign, actualSingleProcessSInnerSize); 
    }
    if (selectMaskFlag && attenMaskFlag && qSeqSize==1) {
        ElewiseComputeMask(bmm1ResUb, tmpBuff5, 0, gSize, actualSingleProcessSInnerSizeAlign, actualSingleProcessSInnerSize); 
    }
    if (attenMaskFlag && qSeqSize > 1) {
        ElewiseComputeMaskMtp(bmm1ResUb, tmpBuff5, 0, gSize, actualSingleProcessSInnerSizeAlign, actualSingleProcessSInnerSize);
    }
    if (attenMaskFlag) {
        maskQue.FreeTensor(maskUb);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::SoftmaxFlashV2Compute(
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    SoftMaxShapeInfo srcShape = {dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb, softmaxSumUb[baseOffset], softmaxMaxUb[baseOffset], mmResUb, softmaxExpUb[baseOffset],
        softmaxSumUb[baseOffset], softmaxMaxUb[baseOffset], softmaxTmpUb, newTiling, srcShape);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionMulAttenCube310P<IFAT>::Bmm2DataCopyOut(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(OUT_T);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount / typeElementSize;
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / typeElementSize;
    dataCopyParams.dstStride = 0;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        dataCopyParams.blockCount = gqa;
        for (auto i = 0; i < qSeqSize; i++) {
            DataCopy(attentionOutGm[attenOutOffset + startRow * actualColumnCount  + i * gqa * kvHeadNum * headDim], attenOutUb[i * gqa * headDim], dataCopyParams);
        }
    } else {
        DataCopy(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionMulAttenCube310P<IFAT>::Bmm2CastAndCopyOut(LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                            uint32_t dealRowCount, uint32_t columnCount,
                                                            uint32_t actualColumnCount)
{ 
    auto tmpAntiqParamUb = tmpBuff5.Get<T>();
    scaleUb = scaleBuff.Get<Q_T>();
    if constexpr (ANTIQUANT) {
        if (antiqOffsetExistFlag) {
            // bmm2Res + offsetV
            auto antiqOffsetUb = scaleUb[headDim * 2];
            Cast(tmpAntiqParamUb, antiqOffsetUb, AscendC::RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();
            for(auto k=0; k<gSize; k++) {
                Add(bmm2ResUb[k*headDimAlign], bmm2ResUb[k*headDimAlign], tmpAntiqParamUb, headDimAlign);
            }
            PipeBarrier<PIPE_V>();
        }
        Cast(tmpAntiqParamUb, scaleUb[headDim], AscendC::RoundMode::CAST_NONE, headDim);
        PipeBarrier<PIPE_V>();
        for(auto k=0; k<gSize; k++) {
            Mul(bmm2ResUb[k*headDimAlign], bmm2ResUb[k*headDimAlign], tmpAntiqParamUb, headDimAlign);
        }
        PipeBarrier<PIPE_V>();
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventID);
        WaitFlag<HardEvent::V_MTE2>(eventID);
    }
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = bmm2ResUb.template ReinterpretCast<OUT_T>(); 
    Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_NONE, dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
    if (LAYOUT_T != LAYOUT::BNSD && (qSeqSize > 1)) { 
        auto tmpUb = tmpBmm2ResCastTensor[dealRowCount * columnCount];
        for (auto i=0; i<qSeqSize; i++) {
            for (auto j=0; j<gqa; j++) {
                DataCopy(tmpUb[(i * gqa + j) * columnCount], tmpBmm2ResCastTensor[(j * qSeqSize + i) * columnCount], columnCount);
            }
        }
        PipeBarrier<PIPE_V>();
        tmpBmm2ResCastTensor = tmpUb;
    }

    event_t eid = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eid);
    WaitFlag<HardEvent::V_MTE3>(eid);
    Bmm2DataCopyOut(tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    eid = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eid);
    WaitFlag<HardEvent::MTE3_V>(eid);
    eid = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eid);
    WaitFlag<HardEvent::MTE3_MTE2>(eid);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::ProcessVec1(const uint32_t sInnerLoopIdx)
{
    ElewiseCompute(bmm1ResUb, tmpBuff5, 0, gSize, actualSingleProcessSInnerSizeAlign, actualSingleProcessSInnerSize);
    LocalTensor<uint8_t> softmaxTmpUb = tmpBuff5.Get<uint8_t>();
    SoftmaxFlashV2Compute(bmm1ResUb, softmaxTmpUb, 0, gSize, actualSingleProcessSInnerSizeAlign,
                          actualSingleProcessSInnerSize);
    PipeBarrier<PIPE_V>();
    if (sInnerLoopIdx > 0) {
        RowMuls(bmm2ResUb, bmm2ResUb, softmaxExpUb, gSize, headDimAlign, headDim);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::ProcessVec2(const uint32_t sInnerLoopIdx)
{
    // 最后一次输出计算结果
    if (sInnerLoopIdx + 1 == sInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb, gSize, headDimAlign, headDim);
        PipeBarrier<PIPE_V>();
        Bmm2CastAndCopyOut(bmm2ResUb, 0, gSize, headDimAlign, headDim);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::SInnerLoopFunc(const uint32_t sInnerLoopIdx)
{
    uint32_t innerSize = copyKeyLoopCount == 0 ? copyKeyTailSplitSize : copyKeyActSplitS;
    uint32_t firstSize = innerSize;
    uint64_t tmpOffset = 0ULL;
    if constexpr (!PAGE_ATTENTION) {
        tmpOffset =  tensorBOffset;
    }
    DataCopyParams params;
    params.blockCount = gSize;
    params.blockLen = copyKeyActSplitS * sizeof(Q_T) / BYTE_BLOCK;
    params.dstStride = (BYTE_PER_FRACTAL - copyKeyActSplitS * sizeof(Q_T)) / BYTE_BLOCK;

    mm.SetAccmulate(false);
    bool pingpong = this->toReadPingpong;
    uint32_t m = useGemv ? gSize : Align(gSize, 16UL);
    LocalTensor<T> tmpUb = tmpBuff5.Get<T>(); 
    LocalTensor<T> mm2Res = tmpUb[m * headDimAlign];
    LocalTensor<Q_T> mmResUb = tmpBuff5.Get<Q_T>();

    this->CopyND2Ub(keyGm[tmpOffset], innerSize, headDim, headDim, pingpong);
    this->eventUbIn[0] = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    this->eventUbIn[1] = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(this->eventUbIn[pingpong]);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID);
    for (uint32_t i = 0; i < copyKeyLoopCount + 1; i++, pingpong = !pingpong) {
        innerSize = i == copyKeyLoopCount ?  copyKeyTailSplitSize : copyKeyActSplitS;
        uint32_t innerSizeAlign = Align(innerSize, 16U);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        if(i < copyKeyLoopCount){ // has next
            uint32_t nextSize = (i+1 == copyKeyLoopCount ?  copyKeyTailSplitSize : copyKeyActSplitS);
            CopyND2Ub(keyGm[tmpOffset], nextSize, headDim, headDim, !pingpong, (i+1) * copyKeyActSplitS); 
            SetFlag<HardEvent::MTE2_V>(this->eventUbIn[!pingpong]);
        } else {
            CopyND2Ub(valueGm[tmpOffset], firstSize, headDim, headDim, !pingpong); 
            this->toReadPingpong = !pingpong;
            SetFlag<HardEvent::MTE2_V>(this->eventUbIn[!pingpong]);
        }
        WaitFlag<HardEvent::MTE2_V>(this->eventUbIn[pingpong]);
        Trans2L1(l1b, innerSize, headDim, headDim, pingpong);
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        mm.SetShapeNz(static_cast<uint16_t>(m), static_cast<uint16_t>(innerSizeAlign),  static_cast<uint16_t>(headDim));
        mm.ComputeNz(queryL1, l1b, l0c);
        SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
        uint32_t dstStride = (actualSingleProcessSInnerSizeAlign - innerSizeAlign) / BLOCK_ELEMENT_NUM;
        if (useGemv) {
            mm.CopyNz2Ub(bmm1ResUb[i * copyKeyActSplitS], l0c, m, innerSizeAlign, dstStride);
        } else {
            mm.CopyNz2Ub(tmpUb, l0c, m, innerSizeAlign);
            for(uint32_t k=0; k < innerSizeAlign / 16; k++){
                DataCopy(bmm1ResUb[k*16 + i * copyKeyActSplitS], tmpUb[k*m*16], {(uint16_t)gSize, 2, 0, static_cast<uint16_t>((actualSingleProcessSInnerSizeAlign - 16UL) / BLOCK_ELEMENT_NUM )});
            }
        }
        PipeBarrier<PIPE_V>();
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);

    ProcessVec1(sInnerLoopIdx);

    mm.SetAccmulate(false);
    pingpong = this->toReadPingpong;
    event_t eventL1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventL0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1));
    SetFlag<HardEvent::MTE3_MTE2>(eventID);
    for (uint32_t i = 0; i < copyKeyLoopCount + 1; i++, pingpong=!pingpong) {
        innerSize = i == copyKeyLoopCount ?  copyKeyTailSplitSize : copyKeyActSplitS;
        uint32_t innerSizeAlign = Align(innerSize, 16U);
        LocalTensor<Q_T> mmResZn = mmResUb[m * innerSizeAlign];
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        if(i < copyKeyLoopCount){ // has next value
            uint32_t nextSize = i+1 == copyKeyLoopCount ?  copyKeyTailSplitSize : copyKeyActSplitS;
            CopyND2Ub(valueGm[tmpOffset], nextSize, headDim, headDim, !pingpong, (i+1) * copyKeyActSplitS); 
            SetFlag<HardEvent::MTE2_V>(this->eventUbIn[!pingpong]);
        } 
        WaitFlag<HardEvent::MTE2_V>(this->eventUbIn[pingpong]);
        Trans2L1(l1b, innerSize, headDim, headDim, pingpong);
        SetFlag<HardEvent::MTE3_MTE2>(eventID);

        for(uint32_t k=0; k<gSize; k++){
            uint32_t offset = i * copyKeyActSplitS + k * actualSingleProcessSInnerSizeAlign;
            Cast(mmResUb[k * innerSizeAlign], bmm1ResUb[offset], RoundMode::CAST_NONE, innerSizeAlign);
            PipeBarrier<PIPE_V>(); 
            if(innerSize < innerSizeAlign){
                Duplicate(mmResUb[k * innerSizeAlign + innerSize], static_cast<Q_T>(0), innerSizeAlign - innerSize);
                PipeBarrier<PIPE_V>(); 
            }
        }

        if (!useGemv) {
            TransposeParamsExt tpe = {(uint16_t)(m / 16), 16, 1, (uint16_t)innerSizeAlign, TransposeType::TRANSPOSE_NCHW2NHWC};
            Transpose(mmResZn.template ReinterpretCast<uint16_t>(), mmResUb.template ReinterpretCast<uint16_t>(), tmpBuff5.Get<uint8_t>(), tpe);
            PipeBarrier<PIPE_V>(); 
        }

        SetFlag<HardEvent::V_MTE3>(eventL1);
        WaitFlag<HardEvent::V_MTE3>(eventL1);
        if (useGemv) {
            DataCopy(l1a, mmResUb, params); 
        } else {
            DataCopy(l1a, mmResZn, m * innerSizeAlign);
        }

        SetFlag<HardEvent::MTE3_MTE1>(eventL0);
        WaitFlag<HardEvent::MTE3_MTE1>(eventL0);
        event_t eid3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eid3);
        WaitFlag<HardEvent::MTE3_V>(eid3);

        mm.SetShapeNz(static_cast<uint16_t>(m), static_cast<uint16_t>(headDim), static_cast<uint16_t>(innerSizeAlign), 1, 1);
        mm.ComputeNz(l1a, l1b, l0c);
        SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
        if(i == 0) {
            mm.SetAccmulate(true);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    auto dstUb = useGemv ? mm2Res : tmpUb;
    mm.CopyNz2Ub(dstUb, l0c, m, headDimAlign);
    if (!useGemv) {
        for(uint32_t k=0; k < headDimAlign / 16; k++){
            DataCopy(mm2Res[k*16], tmpUb[k*m*16], {(uint16_t)gSize, 2, 0, static_cast<uint16_t>((headDimAlign -16) / BLOCK_ELEMENT_NUM )});
        }
    }
    PipeBarrier<PIPE_V>(); 
    Add(bmm2ResUb, bmm2ResUb, mm2Res, gSize * headDim);
    PipeBarrier<PIPE_V>(); 
}

template <typename IFAT> 
__aicore__ inline void IncreFlashAttentionMulAttenCube310P<IFAT>::Process()
{
    if (g_coreType == AIV && tmpBlockIdx >= usedCoreNum) {
        return; 
    }
    InitCalcParamsEach();

    for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
        GetBN2id(bn2Idx);
        GetActualSeqLen();
        CalculateSUnitSize();
        if (!ComputeKVPaddingBeginOffset()) {
            continue;
        }
        // 计算BN2方向的offset
        CalcBN2Offset();
        CalcBN2Params();
        UpdateInnerLoopCond();
        PipeBarrier<PIPE_V>();
        if (curActSeqLenIsZero) {
            continue;
        }

        // softmax不区分首次
        Duplicate(softmaxMaxUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_2K / sizeof(T));
        Duplicate(softmaxSumUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_2K / sizeof(T));
        Duplicate(bmm2ResUb, FLOAT_ZERO, headDimAlign * Align(gSize, 16UL));

        QueryPreProcess();
        toReadPingpong = false;
        for (uint32_t sInnerLoopIdx = 0; sInnerLoopIdx < sInnerLoopTimes; sInnerLoopIdx++) {
            // 计算s2方向的offset
            bool hasBlack = CalcSInnerOffsetAndParams(sInnerLoopIdx);
            if(actualSingleProcessSInnerSize > 0) {
                SInnerLoopFunc(sInnerLoopIdx);
            } 
            if (hasBlack) {
                break;
            }
        }
        ProcessVec2(sInnerLoopTimes-1);
    }
    PipeBarrier<PIPE_ALL>();
}
#endif 
