/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"



namespace Catlass::Epilogue::Block {

template <
    class VOutputType_,
    class GInputType_,
    class UInputType_,
    class WSInputType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHVnew,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHVnew;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using WSElementInput = typename WSInputType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t PING_BUF_0_OFFSET = 32 * 1024;
        constexpr uint32_t PING_BUF_1_OFFSET = 64 * 1024;
        constexpr uint32_t PONG_BUF_0_OFFSET = 96 * 1024;
        constexpr uint32_t PONG_BUF_1_OFFSET = 128 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = 161 * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = 162 * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = 163 * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = 164 * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = 165 * 1024;
        constexpr uint32_t SHARE_BUF_OFFSET = 166 * 1024;

        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        uUbTensor_ping = resource.ubBuf.template GetBufferByByte<UElementInput>(PING_BUF_1_OFFSET);
        uUbFloatTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        wsUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_1_OFFSET);
        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_BUF_OFFSET);
        gLastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_SUB_BUF_OFFSET);
        gInputUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(PING_G_INPUT_BUF_OFFSET);
        vNewOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_1_OFFSET);
        vNewDecayUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_0_OFFSET);

        uUbTensor_pong = resource.ubBuf.template GetBufferByByte<UElementInput>(PONG_BUF_1_OFFSET);
        uUbFloatTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        wsUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_1_OFFSET);
        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_BUF_OFFSET);
        gLastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_SUB_BUF_OFFSET);
        gInputUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_G_INPUT_BUF_OFFSET);
        vNewOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_1_OFFSET);
        vNewDecayUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_0_OFFSET);

        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_BUF_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput> vnewOutput,
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<UElementInput> uInput,
        AscendC::GlobalTensor<float> wsInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done
        // const LayoutOutput &layoutOutput,
        // const LayoutInput &LayoutInput    
    )
    {
        uint32_t mActual = chunkSize;
        uint32_t nkActual = kHeadDim;
        uint32_t nvActual = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        // 当前场景内部一定连续
        // k [B, H, T, D]
        // g [B, H, T]
        // 在外部offset的基础上进一步offset
        // 当前asset kdim == vHeadDim
        int64_t offsetK = mOffset * nvActual + nOffset;
        int64_t offsetD = 0; // 因为要用最后一个数减去之前所有，所以全部读入

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0)
        {
            gbrcStart = 0;
            gbrcRealStart = 0;
            gbrcReptime = (mActualThisSubBlock + 8 - 1) / 8;

        }
        else
        {
            gbrcStart = mActualPerSubBlock;
            gbrcRealStart = gbrcStart & ~15;
            gbrcReptime = (mActual - gbrcRealStart + 8 - 1) / 8;
        }
        gbrcEffStart = gbrcStart-gbrcRealStart;
        gbrcEffEnd = gbrcEffStart + mActualThisSubBlock;

        AscendC::ResetMask();

        AscendC::GlobalTensor<VElementOutput> vnewOutputThisSubBlock = vnewOutput[offsetK];
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutputThisSubBlock = vnewdecayOutput[offsetK];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<UElementInput> uInputThisSubBlock = uInput[offsetK];
        AscendC::GlobalTensor<float> wsInputThisSubBlock = wsInput[offsetK];

        pingpongFlag = isFirst ? 0 : 4;
        AscendC::LocalTensor<UElementInput> uUbTensor = isFirst ? uUbTensor_ping : uUbTensor_pong;
        AscendC::LocalTensor<float> uUbFloatTensor = isFirst ? uUbFloatTensor_ping : uUbFloatTensor_pong;
        AscendC::LocalTensor<float> wsUbTensor = isFirst ? wsUbTensor_ping : wsUbTensor_pong;
        AscendC::LocalTensor<float> gUbTensor = isFirst ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<float> gLastUbTensor = isFirst ? gLastUbTensor_ping : gLastUbTensor_pong;
        AscendC::LocalTensor<GElementInput> gInputUbTensor = isFirst ? gInputUbTensor_ping : gInputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor = isFirst ? vNewOutputUbTensor_ping : vNewOutputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor = isFirst ? vNewDecayUbTensor_ping : vNewDecayUbTensor_pong;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        if constexpr(std::is_same<GElementInput, float>::value) {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        } else {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gInputUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        if constexpr(!std::is_same<GElementInput, float>::value) {
            AscendC::Cast(gUbTensor, gInputUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2 + pingpongFlag);
        float inputVal = gUbTensor.GetValue(mActual-1);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2 + pingpongFlag);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<float>(gLastUbTensor, inputVal, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub<float>(gUbTensor, gLastUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t dstShape_[2] = {gbrcReptime*8, nvActual};
        uint32_t srcShape_[2] = {gbrcReptime*8, 1};
        AscendC::Broadcast<float, 2, 1>(calcUbTensor, gUbTensor[gbrcRealStart], dstShape_, srcShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube1Done);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
        
        AscendC::DataCopy(uUbTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::Cast(uUbFloatTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        AscendC::DataCopy(wsUbTensor, wsInputThisSubBlock, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
        
        AscendC::Sub<float>(uUbFloatTensor, uUbFloatTensor, wsUbTensor, mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(vNewOutputUbTensor, uUbFloatTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
        AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock * nvActual);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(calcUbTensor[gbrcEffStart*nvActual], uUbFloatTensor, calcUbTensor[gbrcEffStart*nvActual], mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(vNewDecayUbTensor, calcUbTensor[gbrcEffStart*nvActual], AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
        AscendC::DataCopy(vnewdecayOutputThisSubBlock, vNewDecayUbTensor, mActualThisSubBlock * nvActual);

        if (isFirst) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        isFirst = false;
    }

private:
    uint32_t pingpongFlag = 0;
    bool isFirst = true;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<UElementInput> uUbTensor_ping;
    AscendC::LocalTensor<float> uUbFloatTensor_ping;
    AscendC::LocalTensor<float> wsUbTensor_ping;
    AscendC::LocalTensor<float> gUbTensor_ping;
    AscendC::LocalTensor<float> gLastUbTensor_ping;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_ping;

    AscendC::LocalTensor<UElementInput> uUbTensor_pong;
    AscendC::LocalTensor<float> uUbFloatTensor_pong;
    AscendC::LocalTensor<float> wsUbTensor_pong;
    AscendC::LocalTensor<float> gUbTensor_pong;
    AscendC::LocalTensor<float> gLastUbTensor_pong;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_pong;

    AscendC::LocalTensor<uint8_t> shareBuffer_;

};
}

#endif