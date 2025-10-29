/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP

#include "catlass/catlass.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Epilogue::Block {
template<
    class CType_,
    class XType_, 
    class DType_, 
    class TileElemWiseEpilogueAdd_, 
    class TileElemWiseEpilogueMuls_, 
    class TileElemWiseCastD_,
    class TileCopy_
>
class BlockEpilogue<
    EpilogueAtlasA2Gemm,
    CType_,
    XType_, 
    DType_, 
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMuls_,
    TileElemWiseCastD_,
    TileCopy_
>{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2Gemm;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
    using TileElemWiseEpilogueMuls = TileElemWiseEpilogueMuls_;
    using TileElemWiseCastD = TileElemWiseCastD_;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbX = typename TileCopy_::CopyGmToUbX;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    
    const uint32_t SubNum = AscendC::GetSubBlockNum();
    const uint32_t UBSize = ArchTag::UB_SIZE;
    static constexpr bool isNeedCast = !std::is_same<ElementC, ElementX>::value;
    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueAdd::COMPUTE_LENGTH;

    using ElementCompute = typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementX, ElementD>::ElementAccumulator;
    using ElementScalar = ElementCompute;

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogueAdd::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(std::is_same_v<typename TileElemWiseEpilogueMuls::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    struct Params{
        ElementScalar alpha;
        ElementScalar beta;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrD;
        LayoutD layoutD;

        CATLASS_HOST_DEVICE
        Params(){}
        
        CATLASS_HOST_DEVICE
        Params(ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrX_, LayoutX layoutX_, GM_ADDR ptrD_, LayoutD layoutD_)
            : alpha(alpha_), beta(beta_), ptrX(ptrX_), layoutX(layoutX_), ptrD(ptrD_), layoutD(layoutD_){}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, GemmCoord blockShape_, Params const& params_ ,uint32_t ubByteStart = 0) : blockShapeMNK(blockShape_), params(params_)
    {
        uint32_t maxMPerBlock = blockShapeMNK.m();
        uint32_t maxNPerBlock = blockShapeMNK.n();
        uint32_t tileSize = maxMPerBlock * maxNPerBlock / SubNum;
        uint32_t ubCSize = tileSize * sizeof(ElementC);
        uint32_t ubXSize = tileSize * sizeof(ElementX);
        uint32_t ubDSize = tileSize * sizeof(ElementD);
        uint32_t ubXCastSize = tileSize * sizeof(ElementCompute);
        uint32_t ubDCastSize = tileSize * sizeof(ElementCompute);
        ubCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubByteStart);
        ubByteStart += ubCSize;
        ubXTensor = resource.ubBuf.template GetBufferByByte<ElementX>(ubByteStart); 
        ubByteStart += ubXSize;
        ubDTensor = resource.ubBuf.template GetBufferByByte<ElementD>(ubByteStart);
        ubByteStart += ubDSize;
        if constexpr (isNeedCast)
        {
            ubXTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);
            ubByteStart += ubXCastSize;
            ubDTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);;
            ubByteStart += ubDCastSize;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue(){
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(
        GemmCoord const &actualShapeMNK, GemmCoord const &blockCoordMNK,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        uint64_t const &offset)
    {
        AscendC::GlobalTensor<ElementX> gmBlockX;
        gmBlockX.SetGlobalBuffer(reinterpret_cast<__gm__ ElementX*>(params.ptrX));
        AscendC::GlobalTensor<ElementD> gmBlockD;
        gmBlockD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params.ptrD));
        MatrixCoord blockShapeMN = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoordMN = blockCoordMNK.GetCoordMN();
        MatrixCoord actualShapeMN = actualShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoordMN * blockShapeMN;
        MatrixCoord subblockShape{
            CeilDiv(actualShapeMN.row(), SubNum),
            actualShapeMN.column()
        };
        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualShapeMN - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;
        LayoutC layoutInUb{blockShapeMN.row() / SubNum, blockShapeMN.column()};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        auto layoutTileX = params.layoutX.GetTileLayout(actualSubblockShape);
        auto layoutXInUb = layoutInUb.GetTileLayout(actualSubblockShape);
        auto gmTileX = gmBlockX[offset + params.layoutX.GetOffset(blockOffset + subblockOffset)];
        copyGmToUbX(ubXTensor, gmTileX, layoutXInUb, layoutTileX);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr (isNeedCast)
        {
            AscendC::Cast<ElementCompute, ElementX>(ubXTensorCast, ubXTensor, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
            tileElemWiseEpilogueMuls(ubXTensorCast, ubXTensorCast, (ElementCompute)params.beta);
        } else {
            tileElemWiseEpilogueMuls(ubXTensor, ubXTensor, (ElementX)params.beta);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        auto layoutTileC = layoutC.GetTileLayout(actualSubblockShape);
        auto layoutCInUb = layoutInUb.GetTileLayout(actualSubblockShape);
        auto gmTileC = gmBlockC[offset + layoutC.GetOffset(blockOffset + subblockOffset)];
        copyGmToUbC(ubCTensor, gmTileC, layoutCInUb, layoutTileC);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        tileElemWiseEpilogueMuls(ubCTensor, ubCTensor, (ElementC)params.alpha);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (isNeedCast)
        {
            tileElemWiseEpilogueAdd(ubDTensorCast, ubCTensor, ubXTensorCast);
        } else {
            tileElemWiseEpilogueAdd(ubDTensor, ubCTensor, ubXTensor);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (isNeedCast)
        {
            tileElemWiseCastD(ubDTensor, ubDTensorCast);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); 
        auto layoutDInGm = params.layoutD.GetTileLayout(actualSubblockShape);
        auto layoutTileD = layoutInUb.GetTileLayout(actualSubblockShape);
        auto gmTileD = gmBlockD[offset + params.layoutD.GetOffset(blockOffset + subblockOffset)];
        copyUbToGmD(gmTileD, ubDTensor, layoutDInGm, layoutTileD);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
private:
    GemmCoord blockShapeMNK;
    Params params;

    AscendC::LocalTensor<ElementC> ubCTensor;
    AscendC::LocalTensor<ElementX> ubXTensor;
    AscendC::LocalTensor<ElementD> ubDTensor;
    AscendC::LocalTensor<ElementCompute> ubXTensorCast;
    AscendC::LocalTensor<ElementCompute> ubDTensorCast;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbX copyGmToUbX;
    CopyUbToGmD copyUbToGmD;

    TileElemWiseEpilogueAdd tileElemWiseEpilogueAdd;
    TileElemWiseEpilogueMuls tileElemWiseEpilogueMuls;
    TileElemWiseCastD tileElemWiseCastD;
};
}

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP