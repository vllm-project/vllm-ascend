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
#include "block_epilogue_gdn_fwdh_regbase.hpp"

namespace Catlass::Epilogue::Block {

template <
    class HOutputType_,
    class GInputType_,
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_,
    class KGatedTag
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHUpdate,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_,
    KGatedTag
> {
    static constexpr bool kGated = KGatedTag::value;
    static constexpr bool scalarGated = KGatedTag::scalarGated;
    static constexpr bool useExp2 = KGatedTag::useExp2;
    static constexpr float LN2 = 0.6931471805599453f;
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
        constexpr uint32_t PING_G_BUF_OFFSET = 168 * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = 169 * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = 170 * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = 171 * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = 172 * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = 173 * 1024;
        constexpr uint32_t UPDATE_SCRATCH_BUF_OFFSET = 160 * 1024;
        constexpr uint32_t UPDATE_G_BUF_OFFSET = 176 * 1024;


        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        hUpdateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        hUbTensor_ping = resource.ubBuf.template GetBufferByByte<HElementOutput>(UPDATE_SCRATCH_BUF_OFFSET);
        finalOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<FinalStateElement>(UPDATE_SCRATCH_BUF_OFFSET);
        glastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(UPDATE_G_BUF_OFFSET);

        hUpdateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        hUbTensor_pong = resource.ubBuf.template GetBufferByByte<HElementOutput>(UPDATE_SCRATCH_BUF_OFFSET);
        finalOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<FinalStateElement>(UPDATE_SCRATCH_BUF_OFFSET);
        glastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(UPDATE_G_BUF_OFFSET);

        if constexpr (kGated) {
            gkLastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_SUB_BUF_OFFSET);
            gkLastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_SUB_BUF_OFFSET);
            gkInputUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(PING_G_INPUT_BUF_OFFSET);
            gkInputUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_G_INPUT_BUF_OFFSET);
            gkBrcbUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_INPUT_BUF_OFFSET);
            gkBrcbUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_INPUT_BUF_OFFSET);
        }

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
    void ApplyRowScale(
        AscendC::LocalTensor<float> matrix,
        AscendC::LocalTensor<float> rowScale,
        uint32_t rows,
        uint32_t cols)
    {
        __ubuf__ float *matrixAddr = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        __ubuf__ float *rowScaleAddr = reinterpret_cast<__ubuf__ float *>(rowScale.GetPhyAddr());
        AscendC::VF_CALL<detail::ApplyRowScaleDualIssue>(
            matrixAddr, rowScaleAddr, 0,
            static_cast<uint16_t>(rows), static_cast<uint16_t>(cols));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void PrepareKGate(
        AscendC::LocalTensor<float> gateOutput,
        AscendC::LocalTensor<GElementInput> gateInput,
        uint32_t count)
    {
        __ubuf__ float *gateOutputAddr =
            reinterpret_cast<__ubuf__ float *>(gateOutput.GetPhyAddr());
        __ubuf__ GElementInput *gateInputAddr =
            reinterpret_cast<__ubuf__ GElementInput *>(gateInput.GetPhyAddr());
        AscendC::VF_CALL<detail::PrepareKGateRegbase<GElementInput, true>>(
            gateOutputAddr, gateInputAddr, static_cast<uint16_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename StateElement>
    CATLASS_DEVICE
    void ApplyKGateUpdate(
        AscendC::LocalTensor<float> update,
        AscendC::LocalTensor<StateElement> state,
        AscendC::LocalTensor<float> rowScale,
        uint32_t rows,
        uint32_t cols)
    {
        __ubuf__ float *updateAddr = reinterpret_cast<__ubuf__ float *>(update.GetPhyAddr());
        __ubuf__ StateElement *stateAddr =
            reinterpret_cast<__ubuf__ StateElement *>(state.GetPhyAddr());
        __ubuf__ float *rowScaleAddr = reinterpret_cast<__ubuf__ float *>(rowScale.GetPhyAddr());
        AscendC::VF_CALL<detail::ApplyKGateUpdateRegbaseDualIssue<StateElement>>(
            updateAddr, stateAddr, rowScaleAddr,
            static_cast<uint16_t>(rows), static_cast<uint16_t>(cols));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<float> hUpdateInput,
        AscendC::GlobalTensor<GElementInput> gkInput,
        AscendC::GlobalTensor<FinalStateElement> initialState,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vBlockDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool useInitialState,
        bool isPing,
        bool cube2AlreadyWaited,
        bool useDirectFp32Ub,
        uint64_t directUbFreeFlagBegin,
        uint64_t directUbReadyFlagBegin
    )
    {
        static constexpr uint32_t ROW_TILE = 16;
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vBlockDim;
        uint32_t outputStride = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = rowBegin + rowsPerSubBlock;
        if (rowEnd > mActual) {
            rowEnd = mActual;
        }
        if (rowBegin >= mActual) {
            if (useDirectFp32Ub) {
                uint32_t directUbSlot = isPing ? 0 : 1;
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    directUbReadyFlagBegin + directUbSlot);
                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(
                    directUbFreeFlagBegin + directUbSlot);
            } else if (!cube2AlreadyWaited) {
                Arch::CrossCoreWaitFlag(cube2Done);
            }
            return;
        }

        AscendC::ResetMask();

        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<float> hUpdateUbTensor = isPing ? hUpdateUbTensor_ping : hUpdateUbTensor_pong;
        AscendC::LocalTensor<HElementOutput> hUbTensor = isPing ? hUbTensor_ping : hUbTensor_pong;
        AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor = isPing ? finalOutputUbTensor_ping : finalOutputUbTensor_pong;
        AscendC::LocalTensor<float> glastUbTensor = isPing ? glastUbTensor_ping : glastUbTensor_pong;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pingpongFlag);
        bool useFp32StateUpdate = storeFinalState && std::is_same<FinalStateElement, float>::value &&
                                  (!isInitialState || useInitialState);
        float muls = 1.0f;
        if constexpr (scalarGated) {
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
            if constexpr (useExp2) {
                AscendC::Muls(glastUbTensor, glastUbTensor, LN2, 1);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(glastUbTensor, glastUbTensor, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
            muls = glastUbTensor.GetValue(0);
        }
        if constexpr (kGated) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1 + pingpongFlag);
        }
        if constexpr (scalarGated) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        }

        if (useDirectFp32Ub) {
            uint32_t directUbSlot = isPing ? 0 : 1;
            AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                directUbReadyFlagBegin + directUbSlot);
        } else if (!cube2AlreadyWaited) {
            Arch::CrossCoreWaitFlag(cube2Done);
        }
        // fix: need to adapt kGated. issue: A5 do not have vdim128 branch.
        bool waitHFromV = storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value;
        bool waitUpdateFromMte3 = false;
        uint32_t updateReadyEvent = EVENT_ID3 + pingpongFlag;
        for (uint32_t rowStart = rowBegin; rowStart < rowEnd; rowStart += ROW_TILE) {
            uint32_t rowsThisTile = rowEnd - rowStart;
            if (rowsThisTile > ROW_TILE) {
                rowsThisTile = ROW_TILE;
            }

            AscendC::GlobalTensor<HElementOutput> hOutputThisTile = hOutput[rowStart * outputStride];
            AscendC::GlobalTensor<HElementInput> hInputThisTile = hInput[rowStart * outputStride];
            AscendC::GlobalTensor<float> hUpdateInputThisTile = hUpdateInput[rowStart * nActual];
            AscendC::GlobalTensor<FinalStateElement> finalStateThisTile = finalState[rowStart * outputStride];
            AscendC::LocalTensor<float> hUpdateUbTensorThisTile = useDirectFp32Ub
                ? hUpdateUbTensor[(rowStart - rowBegin) * nActual]
                : hUpdateUbTensor;

            if (waitHFromV) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
            if constexpr (std::is_same<FinalStateElement, float>::value) {
                if (useFp32StateUpdate) {
                    if (isInitialState) {
                        CopyGmToUb(calcUbTensor, initialState[rowStart * outputStride],
                                   rowsThisTile, nActual, outputStride);
                    } else {
                        CopyGmToUb(calcUbTensor, finalStateThisTile,
                                   rowsThisTile, nActual, outputStride);
                    }
                } else {
                    CopyGmToUb(hUbTensor, hInputThisTile, rowsThisTile, nActual, outputStride);
                }
            } else {
                CopyGmToUb(hUbTensor, hInputThisTile, rowsThisTile, nActual, outputStride);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);

            if (!useFp32StateUpdate) {
                AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE,
                              rowsThisTile * nActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (scalarGated) {
                AscendC::Muls(calcUbTensor, calcUbTensor, muls, rowsThisTile * nActual);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if constexpr (kGated) {
                AscendC::GlobalTensor<GElementInput> gkLastInput =
                    gkInput[(chunkSize - 1) * kHeadDim + rowStart];
                AscendC::LocalTensor<float> gkLastUbTensor =
                    isPing ? gkLastUbTensor_ping : gkLastUbTensor_pong;
                AscendC::LocalTensor<GElementInput> gkInputUbTensor =
                    isPing ? gkInputUbTensor_ping : gkInputUbTensor_pong;
                if (rowStart == rowBegin) {
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1 + pingpongFlag);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                }
                if constexpr(std::is_same<GElementInput, float>::value) {
                    AscendC::DataCopy(gkLastUbTensor, gkLastInput, rowsThisTile);
                } else {
                    AscendC::DataCopy(gkInputUbTensor, gkLastInput, rowsThisTile);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                if constexpr(std::is_same<GElementInput, float>::value) {
                    PrepareKGate(gkLastUbTensor, gkLastUbTensor, rowsThisTile);
                } else {
                    PrepareKGate(gkLastUbTensor, gkInputUbTensor, rowsThisTile);
                }
                ApplyRowScale(calcUbTensor, gkLastUbTensor, rowsThisTile, nActual);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            }

            if (waitUpdateFromMte3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            }
            if (!useDirectFp32Ub) {
                CopyGmToUb(hUpdateUbTensorThisTile, hUpdateInputThisTile, rowsThisTile, nActual, nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            }
            AscendC::Add<float>(
                hUpdateUbTensorThisTile, calcUbTensor, hUpdateUbTensorThisTile,
                rowsThisTile * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            if (storeFinalState && isFinalState && std::is_same<FinalStateElement, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
                waitHFromV = true;
            } else {
                waitHFromV = false;
            }

            if constexpr(std::is_same<FinalStateElement, float>::value) {
                if (storeFinalState) {
                    if (!isFinalState) {
                        AscendC::Cast(hUbTensor, hUpdateUbTensorThisTile,
                                      AscendC::RoundMode::CAST_RINT,
                                      rowsThisTile * nActual);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    CopyUbToGm(finalStateThisTile, hUpdateUbTensorThisTile,
                               rowsThisTile, nActual, outputStride);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                    waitUpdateFromMte3 = true;
                    if (!isFinalState) {
                        CopyUbToGm(hOutputThisTile, hUbTensor,
                                   rowsThisTile, nActual, outputStride);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                            EVENT_ID2 + pingpongFlag);
                    }
                } else {
                    AscendC::Cast(hUbTensor, hUpdateUbTensorThisTile,
                                  AscendC::RoundMode::CAST_RINT,
                                  rowsThisTile * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    CopyUbToGm(hOutputThisTile, hUbTensor,
                               rowsThisTile, nActual, outputStride);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                    waitUpdateFromMte3 = false;
                }
            } else {
                if (storeFinalState && isFinalState) {
                    AscendC::Cast(finalOutputUbTensor, hUpdateUbTensorThisTile, AscendC::RoundMode::CAST_RINT, rowsThisTile * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    CopyUbToGm(finalStateThisTile, finalOutputUbTensor, rowsThisTile, nActual, outputStride);
                } else {
                    AscendC::Cast(hUbTensor, hUpdateUbTensorThisTile, AscendC::RoundMode::CAST_RINT, rowsThisTile * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    CopyUbToGm(hOutputThisTile, hUbTensor, rowsThisTile, nActual, outputStride);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                waitUpdateFromMte3 = false;
            }
        }

        if (storeFinalState && std::is_same<FinalStateElement, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
            if (!isFinalState) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        }
        if constexpr (kGated) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        }
        if (useDirectFp32Ub) {
            uint32_t directUbSlot = isPing ? 0 : 1;
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                directUbFreeFlagBegin + directUbSlot);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pingpongFlag);

    }

private:
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<float> hUpdateUbTensor_ping;
    AscendC::LocalTensor<HElementOutput> hUbTensor_ping;
    AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor_ping;
    AscendC::LocalTensor<float> glastUbTensor_ping;

    AscendC::LocalTensor<float> hUpdateUbTensor_pong;
    AscendC::LocalTensor<HElementOutput> hUbTensor_pong;
    AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor_pong;
    AscendC::LocalTensor<float> glastUbTensor_pong;

    AscendC::LocalTensor<float> gkLastUbTensor_ping;
    AscendC::LocalTensor<float> gkLastUbTensor_pong;
    AscendC::LocalTensor<GElementInput> gkInputUbTensor_ping;
    AscendC::LocalTensor<GElementInput> gkInputUbTensor_pong;
    AscendC::LocalTensor<float> gkBrcbUbTensor_ping;
    AscendC::LocalTensor<float> gkBrcbUbTensor_pong;
};
}

#endif
