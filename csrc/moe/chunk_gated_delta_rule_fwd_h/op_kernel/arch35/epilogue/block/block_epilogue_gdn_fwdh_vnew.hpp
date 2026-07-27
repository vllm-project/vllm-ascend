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
    class WSInputType_,
    class VUpdateType_,
    class FinalStateType_,
    class KGatedTag
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHVnew,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_,
    VUpdateType_,
    FinalStateType_,
    KGatedTag
> {
    static constexpr bool kGated = KGatedTag::value;
public:
    using DispatchPolicy = EpilogueAtlasGDNFwdHVnew;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using WSElementInput = typename WSInputType_::Element;
    using VElementUpdate = typename VUpdateType_::Element;
    using VLayoutUpdate = typename VUpdateType_::Layout;
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
        constexpr uint32_t PING_WIDE_IO_BUF_OFFSET = 16 * 1024;
        constexpr uint32_t PONG_WIDE_IO_BUF_OFFSET = 24 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = 161 * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = 162 * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = 163 * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = 164 * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = 165 * 1024;
        constexpr uint32_t SHARE_BUF_OFFSET = 166 * 1024;

        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        uUbTensor_ping = resource.ubBuf.template GetBufferByByte<UElementInput>(PING_BUF_2_OFFSET);
        wsUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_BUF_OFFSET);
        gLastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_SUB_BUF_OFFSET);
        gInputUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(PING_G_SUB_BUF_OFFSET);
        vNewOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);
        vNewDecayUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);
        wideIoUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_WIDE_IO_BUF_OFFSET);

        uUbTensor_pong = resource.ubBuf.template GetBufferByByte<UElementInput>(PONG_BUF_2_OFFSET);
        wsUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_BUF_OFFSET);
        gLastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_SUB_BUF_OFFSET);
        gInputUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_G_SUB_BUF_OFFSET);
        vNewOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_2_OFFSET);
        vNewDecayUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_2_OFFSET);
        wideIoUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_WIDE_IO_BUF_OFFSET);

        gBrcbUbTensor_ = resource.ubBuf.template GetBufferByByte<float>(SHARE_BUF_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    template <typename Element>
    CATLASS_DEVICE
    void CopyGmToUb(
        AscendC::LocalTensor<Element> dst,
        AscendC::GlobalTensor<Element> src,
        uint32_t rows,
        uint32_t cols,
        uint32_t srcStride)
    {
        if (cols == srcStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)),
            static_cast<uint32_t>((srcStride - cols) * sizeof(Element)),
            0,
            0};
        AscendC::DataCopyPadExtParams<Element> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename Element>
    CATLASS_DEVICE
    void CopyUbToGm(
        AscendC::GlobalTensor<Element> dst,
        AscendC::LocalTensor<Element> src,
        uint32_t rows,
        uint32_t cols,
        uint32_t dstStride)
    {
        if (cols == dstStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)),
            0,
            static_cast<uint32_t>((dstStride - cols) * sizeof(Element)),
            0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    CATLASS_DEVICE
    void PrepareG(
        AscendC::LocalTensor<float> gUbTensor,
        AscendC::LocalTensor<float> gLastUbTensor,
        AscendC::LocalTensor<GElementInput> gInputUbTensor,
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock,
        uint32_t mActual,
        uint32_t pingpongFlag)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);
        if (mActual == 1) {
            AscendC::Duplicate<float>(gUbTensor, 1.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        if constexpr(std::is_same<GElementInput, float>::value) {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        } else {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(GElementInput)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gInputUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + pingpongFlag);
        if constexpr(!std::is_same<GElementInput, float>::value) {
            AscendC::Cast(gUbTensor, gInputUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        float inputVal = gUbTensor.GetValue(mActual - 1);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<float>(gLastUbTensor, inputVal, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub<float>(gUbTensor, gLastUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void ApplyRowScale(
        AscendC::LocalTensor<float> matrix,
        AscendC::LocalTensor<float> rowScale,
        uint32_t rowScaleOffset,
        uint32_t rows,
        uint32_t cols)
    {
        constexpr uint32_t FP32_PER_BLOCK = 8;
        constexpr uint32_t FP32_PER_REPEAT = 64;
        uint8_t rowStride = static_cast<uint8_t>(cols / FP32_PER_BLOCK);
        AscendC::BinaryRepeatParams params(1, 1, 0, rowStride, rowStride, 1);
        uint32_t localRow = 0;
        while (localRow < rows) {
            uint32_t scaleRow = rowScaleOffset + localRow;
            uint32_t alignedScaleRow = scaleRow & ~(FP32_PER_BLOCK - 1);
            uint32_t firstScaleLane = scaleRow - alignedScaleRow;
            uint32_t rowsThisBlock = Min(FP32_PER_BLOCK - firstScaleLane, rows - localRow);
            AscendC::Brcb(gBrcbUbTensor_, rowScale[alignedScaleRow], 1, {1, FP32_PER_BLOCK});
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < cols; col += FP32_PER_REPEAT) {
                uint32_t count = Min(FP32_PER_REPEAT, cols - col);
                uint32_t matrixOffset = localRow * cols + col;
                AscendC::Mul(matrix[matrixOffset], matrix[matrixOffset],
                             gBrcbUbTensor_[firstScaleLane * FP32_PER_BLOCK], count,
                             rowsThisBlock, params);
            }
            AscendC::PipeBarrier<PIPE_V>();
            localRow += rowsThisBlock;
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput> vnewOutput,
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutput,
        AscendC::LocalTensor<VElementUpdate> l1VUpdate,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<UElementInput> uInput,
        AscendC::GlobalTensor<float> wsInput,
        AscendC::GlobalTensor<GElementInput> gkInput,
        AscendC::GlobalTensor<VElementOutput> kInput,
        AscendC::GlobalTensor<VElementOutput> kDecayWorkspace,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vBlockDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done,
        Arch::CrossCoreFlag vec1Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool waitWsFromMte3,
        bool isPing
    )
    {
        static constexpr uint32_t ROW_TILE = 16;
        uint32_t mActual = chunkSize;
        uint32_t nvActual = vBlockDim;
        uint32_t nkActual = kHeadDim;
        uint32_t inputStride = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = rowBegin + rowsPerSubBlock;
        if (rowEnd > mActual) {
            rowEnd = mActual;
        }
        if (rowBegin >= mActual) {
            Arch::CrossCoreWaitFlag(cube1Done);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);
            return;
        }
        AscendC::ResetMask();

        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<UElementInput> uUbTensor = isPing ? uUbTensor_ping : uUbTensor_pong;
        AscendC::LocalTensor<float> wsUbTensor = isPing ? wsUbTensor_ping : wsUbTensor_pong;
        AscendC::LocalTensor<float> gUbTensor = isPing ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<float> gLastUbTensor = isPing ? gLastUbTensor_ping : gLastUbTensor_pong;
        AscendC::LocalTensor<GElementInput> gInputUbTensor = isPing ? gInputUbTensor_ping : gInputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor = isPing ? vNewOutputUbTensor_ping : vNewOutputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor = isPing ? vNewDecayUbTensor_ping : vNewDecayUbTensor_pong;
        if (nvActual > 128) {
            uUbTensor = isPing ? wideIoUbTensor_ping : wideIoUbTensor_pong;
            vNewOutputUbTensor = isPing ? wideIoUbTensor_ping : wideIoUbTensor_pong;
            vNewDecayUbTensor = isPing ? wideIoUbTensor_ping : wideIoUbTensor_pong;
        }

        if (rowBegin < rowEnd && nvActual <= 128 && nvActual == inputStride) {
            uint32_t mActualThisSubBlock = rowEnd - rowBegin;
            AscendC::GlobalTensor<VElementOutput> vnewOutputThisSubBlock = vnewOutput[rowBegin * inputStride];
            AscendC::GlobalTensor<UElementInput> uInputThisSubBlock = uInput[rowBegin * inputStride];
            AscendC::GlobalTensor<float> wsInputThisSubBlock = wsInput[rowBegin * nvActual];

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
            AscendC::DataCopy(uUbTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::Cast(calcUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);

            PrepareG(gUbTensor, gLastUbTensor, gInputUbTensor, gInputThisSubBlock, mActual, pingpongFlag);
            Arch::CrossCoreWaitFlag(cube1Done);

            if (waitWsFromMte3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            }
            AscendC::DataCopy(wsUbTensor, wsInputThisSubBlock, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);

            AscendC::Sub<float>(wsUbTensor, calcUbTensor, wsUbTensor, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Copy(calcUbTensor, wsUbTensor, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            ApplyRowScale(calcUbTensor, gUbTensor, rowBegin, mActualThisSubBlock, nvActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);

            uint32_t nvLoops = nvActual / FLOAT_NUM_PER_REPEAT;
            for (uint32_t nLoop = 0; nLoop < nvLoops; nLoop++) {
                uint32_t castSrcOffset = nLoop * FLOAT_NUM_PER_REPEAT;
                uint32_t castDstOffset = nLoop * mActualThisSubBlock * FLOAT_NUM_PER_REPEAT;
                AscendC::Cast(vNewDecayUbTensor[castDstOffset], calcUbTensor[castSrcOffset], AscendC::RoundMode::CAST_RINT, FLOAT_NUM_PER_REPEAT, mActualThisSubBlock, {(uint16_t)mActualThisSubBlock, 1, 1, (uint8_t)(nvLoops * 8)});
            }

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);

            uint32_t ubToL1Loops = nvActual / SIZE_16_NUM_PER_C0;
            uint32_t mActualPadded = (mActual + NZ_BLOCK_SIZE - 1) / NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
            AscendC::DataCopyParams intriParams;
            intriParams.blockCount = ubToL1Loops;
            intriParams.blockLen = mActualThisSubBlock;
            intriParams.srcGap = 0;
            intriParams.dstGap = mActualPadded - mActualThisSubBlock;
            uint32_t l1Addr = rowBegin * SIZE_16_NUM_PER_C0;
            AscendC::DataCopy(vnewdecayOutput[l1Addr], vNewDecayUbTensor, intriParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
            if constexpr (!kGated) {
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);
            }

            AscendC::Cast(vNewOutputUbTensor, wsUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);

            if constexpr (kGated) {
                AscendC::GlobalTensor<VElementOutput> kInputThisSubBlock = kInput[rowBegin * nkActual];
                AscendC::GlobalTensor<VElementOutput> kDecayWorkspaceThisSubBlock = kDecayWorkspace[rowBegin * nkActual];
                // KDA passes kg = k * exp2(g_last - gk). Keep that decay exactly once.
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
                AscendC::DataCopy(vNewOutputUbTensor, kInputThisSubBlock,
                                  mActualThisSubBlock * nkActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1 + pingpongFlag);
                AscendC::DataCopy(kDecayWorkspaceThisSubBlock, vNewOutputUbTensor,
                                  mActualThisSubBlock * nkActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);

                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);
            }
            return;
        }

        PrepareG(gUbTensor, gLastUbTensor, gInputUbTensor, gInputThisSubBlock, mActual, pingpongFlag);
        Arch::CrossCoreWaitFlag(cube1Done);

        uint32_t mActualPadded = (mActual + NZ_BLOCK_SIZE - 1) / NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
        bool waitWsThisTileFromMte3 = waitWsFromMte3;
        for (uint32_t rowStart = rowBegin; rowStart < rowEnd;) {
            uint32_t alignExtra = rowStart & 7;
            uint32_t maxRowsThisTile = ROW_TILE - alignExtra;
            uint32_t rowsThisTile = rowEnd - rowStart;
            if (rowsThisTile > maxRowsThisTile) {
                rowsThisTile = maxRowsThisTile;
            }
            uint32_t localRowStart = rowStart - rowBegin;

            AscendC::GlobalTensor<VElementOutput> vnewOutputThisTile = vnewOutput[rowStart * inputStride];
            AscendC::GlobalTensor<UElementInput> uInputThisTile = uInput[rowStart * inputStride];
            AscendC::GlobalTensor<float> wsInputThisTile = wsInput[rowStart * nvActual];
            AscendC::LocalTensor<float> wsUbTensorThisTile = wsUbTensor[localRowStart * nvActual];

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
            CopyGmToUb(uUbTensor, uInputThisTile, rowsThisTile, nvActual, inputStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::Cast(calcUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();

            if (waitWsThisTileFromMte3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
                waitWsThisTileFromMte3 = false;
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            }
            CopyGmToUb(wsUbTensorThisTile, wsInputThisTile, rowsThisTile, nvActual, nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::Sub<float>(wsUbTensorThisTile, calcUbTensor, wsUbTensorThisTile, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Copy(calcUbTensor, wsUbTensorThisTile, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            ApplyRowScale(calcUbTensor, gUbTensor, rowStart, rowsThisTile, nvActual);

            uint32_t nvLoops = nvActual / FLOAT_NUM_PER_REPEAT;
            for (uint32_t nLoop = 0; nLoop < nvLoops; nLoop++) {
                uint32_t castSrcOffset = nLoop * FLOAT_NUM_PER_REPEAT;
                uint32_t castDstOffset = nLoop * rowsThisTile * FLOAT_NUM_PER_REPEAT;
                AscendC::Cast(vNewDecayUbTensor[castDstOffset], calcUbTensor[castSrcOffset], AscendC::RoundMode::CAST_RINT, FLOAT_NUM_PER_REPEAT, rowsThisTile, {(uint16_t)rowsThisTile, 1, 1, (uint8_t)(nvLoops * 8)});
            }

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);

            uint32_t ubToL1Loops = nvActual / SIZE_16_NUM_PER_C0;
            AscendC::DataCopyParams intriParams;
            intriParams.blockCount = ubToL1Loops;
            intriParams.blockLen = rowsThisTile;
            intriParams.srcGap = 0;
            intriParams.dstGap = mActualPadded - rowsThisTile;
            uint32_t l1Addr = rowStart * SIZE_16_NUM_PER_C0;
            AscendC::DataCopy(vnewdecayOutput[l1Addr], vNewDecayUbTensor, intriParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
            if constexpr (!kGated) {
                if (rowStart + rowsThisTile >= rowEnd) {
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);
                }
            }

            AscendC::Cast(vNewOutputUbTensor, wsUbTensorThisTile, AscendC::RoundMode::CAST_RINT, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            CopyUbToGm(vnewOutputThisTile, vNewOutputUbTensor, rowsThisTile, nvActual, inputStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
            rowStart += rowsThisTile;
        }

        if constexpr (kGated) {
            uint32_t mActualThisSubBlock = rowEnd - rowBegin;
            for (uint32_t rowOffset = 0; rowOffset < mActualThisSubBlock; rowOffset += ROW_TILE) {
                uint32_t rowsThisTile = Min(ROW_TILE, mActualThisSubBlock - rowOffset);
                uint32_t row = rowBegin + rowOffset;
                AscendC::GlobalTensor<VElementOutput> kInputThisTile = kInput[row * nkActual];
                AscendC::GlobalTensor<VElementOutput> kDecayWorkspaceThisTile =
                    kDecayWorkspace[row * nkActual];
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
                CopyGmToUb(vNewOutputUbTensor, kInputThisTile, rowsThisTile, nkActual, nkActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1 + pingpongFlag);
                CopyUbToGm(kDecayWorkspaceThisTile, vNewOutputUbTensor,
                           rowsThisTile, nkActual, nkActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);
        }

        if (rowBegin < rowEnd) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);

    }

private:
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<UElementInput> uUbTensor_ping;
    AscendC::LocalTensor<float> wsUbTensor_ping;
    AscendC::LocalTensor<float> gUbTensor_ping;
    AscendC::LocalTensor<float> gLastUbTensor_ping;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_ping;

    AscendC::LocalTensor<UElementInput> uUbTensor_pong;
    AscendC::LocalTensor<float> wsUbTensor_pong;
    AscendC::LocalTensor<float> gUbTensor_pong;
    AscendC::LocalTensor<float> gLastUbTensor_pong;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> wideIoUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> wideIoUbTensor_pong;

    AscendC::LocalTensor<float> gBrcbUbTensor_;

};
}

#endif
