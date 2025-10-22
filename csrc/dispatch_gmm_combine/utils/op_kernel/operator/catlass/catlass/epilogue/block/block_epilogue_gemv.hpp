/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class CType_,    
    class YType_,
    class ZType_,
    class TileElemWiseEpilogueAdd_,
    class TileElemWiseEpilogueMuls_,
    class TileCopy_
>
class BlockEpilogue<
    EpilogueAtlasA2Gemv,
    CType_,
    YType_,
    ZType_,
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMuls_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2Gemv;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
    using TileElemWiseEpilogueMuls = TileElemWiseEpilogueMuls_;

    using CopyGmToUbY = typename TileCopy_::CopyGmToUbC;
    using CopyGmToubC = typename TileCopy_::CopyGmToUbX;      
    using CopyUbToGmZ = typename TileCopy_::CopyUbToGmD;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueMuls::COMPUTE_LENGTH;

    static constexpr bool isNeedCast = !std::is_same<ElementC, ElementY>::value;

    using ElementCompute = typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementY, ElementZ>::ElementAccumulator;
    using ElementScalar = ElementCompute;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    // check the layout of Y, C and Z
    static_assert(std::is_same_v<LayoutY, layout::VectorLayout> && std::is_same_v<LayoutC, layout::VectorLayout> &&
        std::is_same_v<LayoutZ, layout::VectorLayout>,"Layout type of Y, C and Z must be VectorLayout");

    using LayoutComputeInUb = layout::VectorLayout;

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogueMuls::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");

    struct Params {
        ElementScalar alpha;
        ElementScalar beta;
        GM_ADDR ptrY;
        LayoutY layoutY;
        GM_ADDR ptrZ;
        LayoutZ layoutZ;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrY_, LayoutC layoutY_, GM_ADDR ptrZ_, LayoutZ layoutZ_)
            : alpha(alpha_), beta(beta_), ptrY(ptrY_), layoutY(layoutY_), ptrZ(ptrZ_), layoutZ(layoutZ_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params): params(params) 
    {
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        ubY = resource.ubBuf.template GetBufferByByte<ElementY>(COMPUTE_LENGTH * sizeof(ElementC));
        ubYCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(COMPUTE_LENGTH * sizeof(ElementC));
        ubZ = resource.ubBuf.template GetBufferByByte<ElementZ>(
            COMPUTE_LENGTH * sizeof(ElementY) + COMPUTE_LENGTH * sizeof(ElementC));
        ubZCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(
            COMPUTE_LENGTH * sizeof(ElementY) + COMPUTE_LENGTH * sizeof(ElementC));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue() 
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(
        TensorCoord const& blockOffsetMN,
        TensorCoord const& actualBlockShapeMN,
        AscendC::GlobalTensor<ElementCompute> const& gmBlockC,
        LayoutC const& layoutBlockC) 
    {
        TensorCoord actualBlockShape = actualBlockShapeMN;
        TensorCoord blockOffset = blockOffsetMN;

        TensorCoord subblockShape{
            CeilDiv(actualBlockShape[0], static_cast<uint32_t>(AscendC::GetSubBlockNum()))
        };
        TensorCoord subblockCoord{static_cast<uint32_t>(AscendC::GetSubBlockIdx())};

        TensorCoord actualSubblockShape = TensorCoord::Min(subblockShape, actualBlockShape - subblockCoord * subblockShape);
        TensorCoord subblockOffset = subblockCoord * subblockShape;

        // Get the data and layout of C
        auto gmSubblockC = gmBlockC[layoutBlockC.GetOffset(subblockOffset)];
        auto layoutSubblockC = layoutBlockC.GetTileLayout(actualSubblockShape);

        // Get the data and layout of y
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params.ptrY));
        auto gmSubblockY = gmY[params.layoutY.GetOffset(blockOffset + subblockOffset)];
        auto layoutSubblockY = params.layoutY.GetTileLayout(actualSubblockShape);

        // Get the data and layout of Z
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer(reinterpret_cast<__gm__ ElementZ*>(params.ptrZ));
        auto gmSubblockZ = gmZ[params.layoutZ.GetOffset(blockOffset + subblockOffset)];
        auto layoutSubblockZ = params.layoutZ.GetTileLayout(actualSubblockShape);

        // get the layout on UB
        auto layoutComputeInUb = LayoutComputeInUb::template MakeLayoutInUb<ElementCompute>(actualSubblockShape);

        // load C(A*x) from gm to ub
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        copyGmToubC(ubC, gmSubblockC, layoutComputeInUb, layoutSubblockC);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // compute C * alpha
        tileEpilogueMul(ubC, ubC, params.alpha);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        // load Y from gm to ub
        copyGmToUbY(ubY, gmSubblockY, layoutComputeInUb, layoutSubblockY);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // compute Y * beta
        if constexpr (isNeedCast) {
            AscendC::Cast<ElementCompute, ElementY>(ubYCast, ubY, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
            tileEpilogueMul(ubYCast, ubYCast, params.beta);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            tileEpilogueMul(ubY, ubY, params.beta);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if constexpr (isNeedCast) {
            tileEpilogueAdd(ubZCast, ubC, ubYCast);
        } else {
            tileEpilogueAdd(ubZ, ubC, ubY);
        }

        if constexpr (isNeedCast) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<ElementZ, ElementCompute>(ubZ, ubZCast, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        copyUbToGmZ(gmSubblockZ, ubZ, layoutSubblockZ, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    };

private:
    Params params;

    AscendC::LocalTensor<ElementY> ubY;
    AscendC::LocalTensor<ElementCompute> ubYCast;
    AscendC::LocalTensor<ElementC> ubC;
    AscendC::LocalTensor<ElementZ> ubZ;
    AscendC::LocalTensor<ElementCompute> ubZCast;

    TileElemWiseEpilogueAdd tileEpilogueAdd;
    TileElemWiseEpilogueMuls tileEpilogueMul;

    CopyGmToUbY copyGmToUbY;
    CopyGmToubC copyGmToubC;
    CopyUbToGmZ copyUbToGmZ;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP