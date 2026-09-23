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

        // Fused per-m-tile layout (m <= 128 per call): h_work arrives UB-RESIDENT
        // at the hUpdUbOffset passed per call (the hand-mmad deformat window at
        // 64K..128K) -- no GM round-trip. calc reuses the cube staging region
        // [0,64K): the next tile's L0C->UB copy is a V op queued after this
        // call's V work. hOut/final cast IN PLACE over the resident f32 tile
        // (f16 write trails the f32 read). hUb sits above the resident window.
        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t H_BUF_OFFSET = 128 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;

        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);
        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementInput>(H_BUF_OFFSET);
        glastUbTensor = resource.ubBuf.template GetBufferByByte<float>(PING_G_BUF_OFFSET);
        resource_ = &resource;
    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        uint32_t hUpdUbOffset,
        float hDecayScale,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool isFinalState
    )
    {
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vHeadDim;
        // Single working instance (the kernel gates subblock 1 out): full m.
        uint32_t subBlockIdx = 0;
        uint32_t subBlockNum = 1;
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetH = mOffset * nActual + nOffset;

        AscendC::ResetMask();

        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[offsetH];
        // Resident h_work (f32, ND) and its in-place f16/f32 output views.
        AscendC::LocalTensor<float> hUpdateUbTensor =
            resource_->ubBuf.template GetBufferByByte<float>(hUpdUbOffset + offsetH * sizeof(float));
        AscendC::LocalTensor<HElementOutput> hOutputUbTensor =
            resource_->ubBuf.template GetBufferByByte<HElementOutput>(hUpdUbOffset + offsetH * sizeof(float));
        AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor =
            resource_->ubBuf.template GetBufferByByte<FinalStateElement>(hUpdUbOffset + offsetH * sizeof(float));

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        // The hand-mmad path's ND copy-out (MTE3) reads UB[64K..128K) right before
        // this call; hUbTensor lives at 64K, so drain MTE3 before overwriting.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();
        
        // exp(g_last) is hoisted to the caller (once per chunk, not per tile).
        AscendC::Muls(calcUbTensor, calcUbTensor, hDecayScale, mActualThisSubBlock * nActual);


        // h_work is already in UB (V-written by the deformat): V program order
        // covers the RAW; no GM load, no flags.
        AscendC::Add<float>(hUpdateUbTensor, calcUbTensor, hUpdateUbTensor, mActualThisSubBlock * nActual);

        if (isFinalState) {
            if constexpr(!std::is_same<FinalStateElement, float>::value) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(finalOutputUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                AscendC::DataCopy(finalStateThisSubBlock, finalOutputUbTensor, mActualThisSubBlock * nActual);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                AscendC::DataCopy(finalStateThisSubBlock, hUpdateUbTensor, mActualThisSubBlock * nActual);
            }
        } else {
            // Prev chunk's hOutput GM store still reads hOutputUbTensor: drain MTE3
            // before the V rewrite; Add -> Cast is V -> V.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(hOutputUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::DataCopy(hOutputThisSubBlock, hOutputUbTensor, mActualThisSubBlock * nActual);
        }
    }

private:
    Arch::Resource<ArchTag> *resource_ = nullptr;
    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<HElementInput> hUbTensor;


    AscendC::LocalTensor<float> glastUbTensor;

};
}

#endif