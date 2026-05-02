/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_finalize_routing.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_FINALIZE_ROUTING_H
#define BLOCK_EPILOGUE_FINALIZE_ROUTING_H
#if defined(__DAV_C310__)
#include "kernel_operator.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Block {
namespace {
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t LOGIT_INDEX = 4;
constexpr uint32_t MAX_SINGLE_MN = 128 * 256;
constexpr uint32_t HALF_DB_MAX_SINGLE_MN = 32 * 256;
constexpr uint32_t BLOCK_BYTES = 256;
constexpr uint64_t MAX_OUTPUT_M_UB = 32;
constexpr uint64_t BLOCK_ELEMENTS_FP32 = 8;
}

using namespace AscendC;

#define GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS                                                                  \
    template <typename DataTypeOut_>
#define GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS                                                                   \
    DataTypeOut_

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
class BlockEpilogueFinalizeRouting {
public:
    __aicore__ inline BlockEpilogueFinalizeRouting() {}

    struct Arguments {
	    GM_ADDR yGMAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR logitGmAddr{nullptr};
        GM_ADDR rowIndexGmAddr{nullptr};
        int32_t baseM = 256;
        int32_t baseN = 256;
        Arguments() = default;
    };

    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // blk_m, blk_n, blk_k, _
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>; // y, _, _, _, logit, rowIndex
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // m, n, k, _

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    // static init
    __host_aicore__ static Params InitParams(Arguments const& args)
    {
        Params params = {args.yGMAddr, args.x2ScaleGmAddr, args.x1ScaleGmAddr, args.biasGmAddr,
            args.logitGmAddr, args.rowIndexGmAddr, args.baseM, args.baseN};
        return params;
    }

private:
    __aicore__ inline void CopyInLogit(uint32_t curBaseM, uint64_t offsetM, LocalTensor<float> logitUb);
    __aicore__ inline void VFDoLogitMuls(uint32_t offsetRe, uint32_t offsetLogit,
        uint16_t repeatTimesLogit, uint16_t repeatTimesRe, __ubuf__ DataTypeOut* outUbAddr, LocalTensor<float> logitUb);
    __aicore__ inline void VectorAtomicProcess(uint32_t curBaseN,
        uint32_t curVecBaseM, uint64_t offsetM, uint64_t yOffset, LocalTensor<DataTypeOut> yLocal);
    
    // GM ADDR
    AscendC::GlobalTensor<float> logitGlobal_;
    AscendC::GlobalTensor<int64_t> rowIndexGlobal_;
    AscendC::GlobalTensor<DataTypeOut> yGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeOut> l0cOutUb_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<float> logitUbPing_;
    AscendC::LocalTensor<float> logitUbPong_;
    AscendC::LocalTensor<DataTypeOut> outUbPing_;
    AscendC::LocalTensor<DataTypeOut> outUbPong_;

    const Params *params_;
    int64_t n_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint64_t singleM_;
    uint64_t singleN_;
    uint64_t alignN_;
    uint64_t vlForFloatNumber_;
    uint32_t repeatTimesLine_;
    uint16_t yCrossPingPongID_ = 0;
    uint16_t logitCrossPingPongID_ = 0;
};

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::Init(Params const &params)
{
    if ASCEND_IS_AIC
    {
        return;
    }
    params_ = &params;
    yGlobal_.SetGlobalBuffer((__gm__ DataTypeOut *)params_->yGMAddr);
    // input
    uint32_t afterFirstIn = MAX_SINGLE_MN * sizeof(DataTypeOut);
    logitUbPing_= AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, afterFirstIn, BLOCK_BYTES);
    uint32_t afterSecondIn = afterFirstIn  + BLOCK_BYTES * sizeof(DataTypeOut);
    logitUbPong_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, afterSecondIn, BLOCK_BYTES);
    outUbPing_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT,
        afterSecondIn + BLOCK_BYTES * sizeof(DataTypeOut), HALF_DB_MAX_SINGLE_MN);
    outUbPong_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT,
        afterSecondIn + (BLOCK_BYTES + HALF_DB_MAX_SINGLE_MN) * sizeof(DataTypeOut), HALF_DB_MAX_SINGLE_MN);
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor()
{
    return l0cOutUb_;
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = Get<MNK_N>(problemShape);
}


GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        logitGlobal_.SetGlobalBuffer((__gm__ float *)params_->logitGmAddr + Get<LOGIT_INDEX>(baseOffset));
        rowIndexGlobal_.SetGlobalBuffer((__gm__ int64_t *)params_->rowIndexGmAddr + Get<LOGIT_INDEX>(baseOffset));
    }
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyInLogit(
    uint32_t curBaseM, uint64_t offsetM, LocalTensor<float> logitUb)
    {
        DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams;
        DataCopyPad(logitUb, logitGlobal_[offsetM], perTokenScaleParams, padParams);
    }

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VectorAtomicProcess(
    uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM, uint64_t yOffset, LocalTensor<DataTypeOut> yLocal)
{
    SetAtomicAdd<float>();
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
    for (uint32_t i = 0; i < curVecBaseM; i++)
    {
        auto outRow = static_cast<uint64_t>(rowIndexGlobal_.GetValue(offsetM + i));
        DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN_], paramsOut);
    }

    SetAtomicNone();
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VFDoLogitMuls(
    uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit,uint16_t repeatTimesRe, __ubuf__ DataTypeOut* outUbAddr,
    LocalTensor<float> logitUb)
{
    __ubuf__ DataTypeOut* l0cOutUbAddr = (__ubuf__ DataTypeOut*)l0cOutUb_.GetPhyAddr();
    __ubuf__ DataTypeOut* logitUbAddr = (__ubuf__ DataTypeOut*)logitUb.GetPhyAddr();
    l0cOutUbAddr += offsetRe;
    logitUbAddr += offsetLogit;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregLogit;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vregRe, vDstReg;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
            DataCopy<DataTypeOut, MicroAPI::LoadDist::DIST_BRC_B32>(vregLogit, logitUbAddr + i);
            uint32_t elementNum = singleN_;
            for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                mask = AscendC::MicroAPI::UpdateMask<DataTypeOut>(elementNum);
                AscendC::MicroAPI::DataCopy(vregRe, l0cOutUbAddr + i * alignN_ + j * vlForFloatNumber_);
                AscendC::MicroAPI::Mul(vDstReg, vregLogit, vregRe, mask);
                AscendC::MicroAPI::DataCopy(outUbAddr + i * alignN_ + j * vlForFloatNumber_, vDstReg, mask);
            }
        }
    }
}

GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<GMM_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape& blockShape, const BlockCoord& blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    uint32_t halfSingleM = CeilDiv(singleM_,
        static_cast<uint64_t>(AscendC::GetTaskRation()));
    alignN_ = CeilDiv(singleN_, BLOCK_ELEMENTS_FP32) * BLOCK_ELEMENTS_FP32;
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint32_t mOffset = subBlockIdx_ * halfSingleM;
    vlForFloatNumber_ = BLOCK_BYTES / sizeof(DataTypeOut);
    auto repeatTimesRe = CeilDiv(singleN_, vlForFloatNumber_); 
    uint64_t logitOffset = Get<LOGIT_INDEX>(blockCoord) + mOffset;
    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint16_t remainRepeatTimesLogit = (singleMInVec % MAX_OUTPUT_M_UB != 0)
                                    ? singleMInVec % MAX_OUTPUT_M_UB : MAX_OUTPUT_M_UB;
    auto logitUb = logitCrossPingPongID_ == 0 ? logitUbPing_ : logitUbPong_;
    CopyInLogit(singleMInVec, logitOffset, logitUb);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(logitCrossPingPongID_ );
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(logitCrossPingPongID_ );
    logitCrossPingPongID_  = (logitCrossPingPongID_ + 1) & 1;
    uint32_t loopNumY = CeilDiv(singleMInVec, MAX_OUTPUT_M_UB);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    for (int32_t i = 0;  i < loopNumY;  i++)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongID_);
        repeatTimesLine_ = (i == loopNumY - 1) ? remainRepeatTimesLogit : MAX_OUTPUT_M_UB;
        __ubuf__ DataTypeOut* outUbAddr =
            yCrossPingPongID_ == 0 ? (__ubuf__ DataTypeOut*)outUbPing_.GetPhyAddr() : (__ubuf__ DataTypeOut*)outUbPong_.GetPhyAddr();
        VFDoLogitMuls(i * MAX_OUTPUT_M_UB * alignN_, i * MAX_OUTPUT_M_UB, repeatTimesLine_, repeatTimesRe, outUbAddr, logitUb);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongID_ );
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongID_);
        auto yLocal = yCrossPingPongID_ == 0 ? outUbPing_ : outUbPong_;
        VectorAtomicProcess(singleN_, repeatTimesLine_, logitOffset + i * MAX_OUTPUT_M_UB, yOffset, yLocal);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongID_);
        yCrossPingPongID_= (yCrossPingPongID_+ 1) & 1;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(logitCrossPingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(logitCrossPingPongID_);
    return;
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
#endif