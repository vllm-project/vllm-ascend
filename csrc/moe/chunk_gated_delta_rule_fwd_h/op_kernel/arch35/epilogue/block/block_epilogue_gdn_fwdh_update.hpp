/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHUpdate,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHUpdate;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using HElementInput = typename HInputType_::Element;
    using HUpdateElementInput = typename HUpdateInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t PING_BUF_0_OFFSET = 32 * 1024;
        constexpr uint32_t PING_BUF_1_OFFSET = 48 * 1024;
        constexpr uint32_t PING_BUF_2_OFFSET = 64 * 1024;
        constexpr uint32_t PING_BUF_3_OFFSET = 80 * 1024;
        constexpr uint32_t PONG_BUF_0_OFFSET = 96 * 1024;
        constexpr uint32_t PONG_BUF_1_OFFSET = 112 * 1024;
        constexpr uint32_t PONG_BUF_2_OFFSET = 128 * 1024;
        constexpr uint32_t PONG_BUF_3_OFFSET = 144 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = 161 * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = 162 * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = 163 * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = 164 * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = 165 * 1024;
        constexpr uint32_t SHARE_BUF_OFFSET = 166 * 1024;


        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        hUpdateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        hUbTensor_ping = resource.ubBuf.template GetBufferByByte<HElementOutput>(PING_BUF_3_OFFSET);
        glastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_INPUT_BUF_OFFSET);

        hUpdateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        hUbTensor_pong = resource.ubBuf.template GetBufferByByte<HElementOutput>(PONG_BUF_3_OFFSET);
        glastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_INPUT_BUF_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<float> hUpdateInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool isPing
    )
    {
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetH = mOffset * nActual + nOffset;

        AscendC::ResetMask();

        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<float> hUpdateInputThisSubBlock = hUpdateInput[offsetH];
        AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[offsetH];

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<float> hUpdateUbTensor = isPing ? hUpdateUbTensor_ping : hUpdateUbTensor_pong;
        AscendC::LocalTensor<HElementOutput> hUbTensor = isPing ? hUbTensor_ping : hUbTensor_pong;
        AscendC::LocalTensor<float> glastUbTensor = isPing ? glastUbTensor_ping : glastUbTensor_pong;

        if (storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        }
        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);

        AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();
        if (storeFinalState && isFinalState && std::is_same<FinalStateElement, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        }

        GElementInput gLastVal = gInputThisSubBlock.GetValue(chunkSize-1);
        float gLastFloat = 0.0f;
        if constexpr(std::is_same<GElementInput, float>::value) {
            gLastFloat = gLastVal;
        } else if constexpr(std::is_same<GElementInput, half>::value) {
            gLastFloat = (float)gLastVal;
        } else if constexpr(std::is_same<GElementInput, bfloat16_t>::value) {
            gLastFloat = AscendC::ToFloat(gLastVal);
        }
        glastUbTensor.SetValue(0, gLastFloat);

        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::Exp(glastUbTensor, glastUbTensor, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        float muls = glastUbTensor.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::Muls(calcUbTensor, calcUbTensor, muls, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube2Done);

        AscendC::Add<float>(hUpdateUbTensor, calcUbTensor, hUpdateUbTensor, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr(std::is_same<FinalStateElement, float>::value) {
            if (storeFinalState && isFinalState) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::DataCopy(finalStateThisSubBlock, hUpdateUbTensor, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
            } else {
                AscendC::Cast(hUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                AscendC::DataCopy(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
        } else {
            AscendC::Cast(hUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
            if (storeFinalState && isFinalState) {
                AscendC::DataCopy(finalStateThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
            } else {
                AscendC::DataCopy(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        }

    }

private:
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<float> hUpdateUbTensor_ping;
    AscendC::LocalTensor<HElementOutput> hUbTensor_ping;
    AscendC::LocalTensor<float> glastUbTensor_ping;

    AscendC::LocalTensor<float> hUpdateUbTensor_pong;
    AscendC::LocalTensor<HElementOutput> hUbTensor_pong;
    AscendC::LocalTensor<float> glastUbTensor_pong;

};
}

#endif