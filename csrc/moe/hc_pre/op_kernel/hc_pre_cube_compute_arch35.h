/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hc_pre_cube_compute_arch35.h
 * \brief
 */
#ifndef HC_PRE_CUBE_COMPUTE_ARCH35_H
#define HC_PRE_CUBE_COMPUTE_ARCH35_H

#include "kernel_operator.h"
#include "hc_pre_base_arch35.h"

namespace HcPreNs
{
    using namespace AscendC;
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
    // constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<float>();
    constexpr static bool splitM_ = true;
    constexpr static uint64_t SPLIT_M_ALIGN = 2;
    static constexpr uint64_t L1_ALLOC_SIZE = 512 * 1024;
    static constexpr uint64_t L1_BUF_NUM = 2;
    static constexpr uint64_t L1_BUF_OFFSET = 128 * 256;
    constexpr static int SYNC_MODE4 = 4;

    class HcPreCubeCompute
    {
    public:
        uint64_t m_{0};
        uint64_t n_{0};
        uint64_t k_{0};
        uint64_t baseM_{0};
        uint64_t baseN_{0};
        uint64_t baseK_{0};
        uint64_t kL1_{0};

    public:
        AscendC::LocalTensor<float> aL0Ping_;
        AscendC::LocalTensor<float> aL0Pong_;
        AscendC::LocalTensor<float> bL0Ping_;
        AscendC::LocalTensor<float> bL0Pong_;
        AscendC::LocalTensor<float> cL0Ping_;
        AscendC::LocalTensor<float> cL0Pong_;
        uint8_t bL1BufferID_{0};
        uint8_t l0PingPongID_{0};
        uint8_t crossPingPongID_{0};
        uint8_t cl0PingPongID_{0};

        __aicore__ inline HcPreCubeCompute()
        {
        }

        __aicore__ inline uint8_t GetBL1BufferId()
        {
            return bL1BufferID_;
        }

        __aicore__ inline void Init()
        {
            // L0 空间的分配
            uint32_t aL0OneBuffer = 256 * 32;
            uint32_t bL0OneBuffer = 256 * 32;
            uint32_t cL0OneBuffer = 256 * 128;

            aL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::A2, 0, aL0OneBuffer);
            aL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::A2, aL0OneBuffer * sizeof(float), aL0OneBuffer);
            bL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::B2, 0, bL0OneBuffer);
            bL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::B2, bL0OneBuffer * sizeof(float), bL0OneBuffer);
            cL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, 0, cL0OneBuffer);
            cL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, cL0OneBuffer * sizeof(float), cL0OneBuffer);
            // 同步
            // B 的 gm2L1 的 pingpong id
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            // l12l0a & l12l0b 的 pingpong id (共用)
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);

            // hf32 compute
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }

        __aicore__ inline void CopyInB1Nd2Nz(uint64_t k, uint64_t currentK, uint64_t baseN, const AscendC::GlobalTensor<float> &bGlobal,
                                             const AscendC::LocalTensor<float> &bl1Local)
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_);
            k_ = k;
            kL1_ = currentK;
            baseN_ = baseN;
            AscendC::Nd2NzParams nd2nzParam;
            nd2nzParam.ndNum = 1;
            nd2nzParam.srcNdMatrixStride = 1;
            nd2nzParam.dstNzNStride = 1;
            nd2nzParam.dstNzMatrixStride = 1;
            nd2nzParam.nValue = baseN_;
            nd2nzParam.dValue = kL1_;
            nd2nzParam.srcDValue = k_;
            nd2nzParam.dstNzC0Stride = (baseN_ + AscendC::BLOCK_CUBE - 1) / AscendC::BLOCK_CUBE * AscendC::BLOCK_CUBE;
            AscendC::DataCopy(bl1Local, bGlobal, nd2nzParam);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_);
        }

        // note: baseM * baseK must <= 256 * 32; baseN * baseK must <= 256 * 32; baseM * baseN must <= 256 * 128
        __aicore__ inline void Process(uint64_t m, uint64_t n, uint64_t baseM, uint64_t baseK,
                                       bool isFirstKL1, bool isLastKL1, const AscendC::LocalTensor<float> &al1Local,
                                       const AscendC::LocalTensor<float> &bl1Local)
        {
            m_ = m;
            n_ = n;
            baseM_ = baseM;
            baseK_ = baseK;
            uint64_t kL1Offset = 0;
            for (uint64_t kb = 0; kb < kL1_; kb += baseK_)
            {
                bool isLastKL0 = (kb + baseK_) >= kL1_;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + 3);
                CopyInA2(kb, kL1Offset, al1Local);
                CopyInB2(kb, kL1Offset, bl1Local);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
                MmadBase(kb, isFirstKL1, isLastKL1, isLastKL0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + 3);
                l0PingPongID_ = l0PingPongID_ ^ 1;
                kL1Offset += baseK_;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_);
            bL1BufferID_ ^= 1;
        }

        __aicore__ inline void CopyInA2(uint64_t kOffset, uint64_t kAL1Offset, const AscendC::LocalTensor<float> &al1Local_)
        {
            uint64_t mAL1 = Align(baseM_, AscendC::BLOCK_CUBE);
            uint64_t offsetAL1 = Align(kAL1Offset, C0_SIZE) * mAL1;
            AscendC::LoadData2DParamsV2 loadData2dParams;

            uint64_t currM = baseM_;
            uint64_t currK = AscendC::Std::min(baseK_, kL1_ - kOffset);
            loadData2dParams.mStartPosition = 0;
            loadData2dParams.kStartPosition = 0;
            loadData2dParams.mStep = CeilDiv(currM, AscendC::BLOCK_CUBE);
            loadData2dParams.kStep = CeilDiv(currK, C0_SIZE);
            loadData2dParams.srcStride = CeilDiv(currM, AscendC::BLOCK_CUBE);
            loadData2dParams.dstStride = loadData2dParams.mStep;
            loadData2dParams.ifTranspose = false;
            AscendC::LoadData(l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_, al1Local_[offsetAL1], loadData2dParams);
        }

        __aicore__ inline void CopyInB2(uint64_t kOffset, uint64_t kBL1Offset, const AscendC::LocalTensor<float> &bl1Local_)
        {
            uint64_t nBL1 = Align(baseN_, AscendC::BLOCK_CUBE);
            uint64_t offsetBL1 = Align(kBL1Offset, C0_SIZE) * nBL1;
            AscendC::LoadData2DParamsV2 loadData2dParams;

            uint64_t currN = baseN_;
            uint64_t currK = AscendC::Std::min(baseK_, kL1_ - kOffset);
            loadData2dParams.mStartPosition = 0;
            loadData2dParams.kStartPosition = 0;
            loadData2dParams.mStep = CeilDiv(currN, AscendC::BLOCK_CUBE);
            loadData2dParams.kStep = CeilDiv(currK, C0_SIZE);
            loadData2dParams.srcStride = CeilDiv(currN, AscendC::BLOCK_CUBE);
            loadData2dParams.dstStride = loadData2dParams.mStep;
            loadData2dParams.ifTranspose = false;
            AscendC::LoadData(l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, bl1Local_[offsetBL1], loadData2dParams);
        }

        __aicore__ inline void MmadBase(uint64_t kOffset, bool isFirstKL1, bool isLastKL1, bool isLastKL0)
        {
            uint32_t mmadK = AscendC::Std::min(baseK_, kL1_ - kOffset);
            AscendC::MmadParams mmadParams;
            mmadParams.m = baseM_;
            mmadParams.n = baseN_;

            mmadParams.k = mmadK;
            mmadParams.disableGemv = true;
            mmadParams.cmatrixInitVal = (isFirstKL1 && kOffset == 0); // kOffset == 0: isFirstKL0
            mmadParams.cmatrixSource = false;
            mmadParams.unitFlag = (isLastKL1 && isLastKL0) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            AscendC::Mmad(cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_,
                          l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, mmadParams);
        }

        // fixpipe CopyOut实现c01拷贝到UB
        __aicore__ inline void CopyOut(const AscendC::LocalTensor<float>& dstLocal)
        {
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams; // ROW_MAJOR默认使能NZ2ND
            uint64_t c0 = AscendC::AuxGetC0Size<float>();
            fixpipeParams.nSize = Align(baseN_, c0);
            fixpipeParams.mSize = splitM_ ? Align(baseM_, SPLIT_M_ALIGN) : baseM_; // 切m需要m是2对齐
            fixpipeParams.dstStride = fixpipeParams.nSize;
            fixpipeParams.srcStride = Align(baseM_, AscendC::BLOCK_CUBE); // 单位CO_SIZE (16*sizeof(C_T))

            fixpipeParams.quantPre = QuantMode_t::NoQuant;
            // fixpipeParams.quantPre = 0;
            // set cvRatio=1:2 默认splitM
            fixpipeParams.dualDstCtl = splitM_ ? static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_M) : 0;
            fixpipeParams.unitFlag = FINAL_ACCUMULATION; // 3 unitflag
            fixpipeParams.params.ndNum = 1;              // ndNum
            fixpipeParams.params.srcNdStride = 1;        // srcNdStride
            fixpipeParams.params.dstNdStride = 1;        // dstNdStride
            AscendC::Fixpipe<float, float, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal, cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, fixpipeParams);
            cl0PingPongID_ ^= 1;
        }

        // fixpipe CopyOut实现c01拷贝到GM
        __aicore__ inline void CopyOut(const AscendC::GlobalTensor<float> &cGlobal)
        {
            AscendC::DataCopyCO12DstParams intriParams;
            intriParams.nSize = baseN_;
            intriParams.mSize = baseM_;
            intriParams.dstStride = n_;
            intriParams.srcStride = Align(baseM_, AscendC::BLOCK_CUBE);
            // set mode according to dtype
            intriParams.quantPre = QuantMode_t::NoQuant;
            intriParams.nz2ndEn = true;
            intriParams.unitFlag = FINAL_ACCUMULATION; // 3 unitflag
            AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
            AscendC::DataCopy(cGlobal, cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, intriParams);
            cl0PingPongID_ ^= 1;
        }

        __aicore__ inline void End()
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
            AscendC::SetHF32Mode(0);
        }
    };
}
#endif