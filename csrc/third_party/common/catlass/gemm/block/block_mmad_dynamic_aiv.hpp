/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_DYNAMIC_AIV_HPP
#define CATLASS_GEMM_BLOCK_MMAD_DYNAMIC_AIV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_muls.hpp"

namespace Catlass::Gemm::Block {
template <
    uint32_t SCALAR_BUFFER_ELE_NUM_,
    uint32_t STAGES_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileVmuls_
>
struct BlockMmadAiv <
    MmadAtlasA2DynamicAiv<SCALAR_BUFFER_ELE_NUM_, STAGES_>,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2DynamicAiv<SCALAR_BUFFER_ELE_NUM_, STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using CopyGmToUbA = typename TileCopy_::CopyGmToUbA;
    using CopyGmToUbB = typename TileCopy_::CopyGmToUbB;
    using CopyUbToGmC = typename TileCopy_::CopyUbToGmC;
    using TileVmuls = TileVmuls_;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t UBA_ALIGN = BYTE_PER_BLK / sizeof(ElementA);
    static constexpr uint32_t UBB_ALIGN = BYTE_PER_BLK / sizeof(ElementB);
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = DispatchPolicy::SCALAR_BUFFER_ELE_NUM;
    
    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmadAiv(Arch::Resource<ArchTag> &resource, MatrixCoord taskShape, uint32_t ubBufAddrStart = 0)
    {
        uint32_t ubASize = SCALAR_BUFFER_ELE_NUM * sizeof(ElementA);
        uint32_t ubBSize = taskShape.column() * sizeof(ElementB);
        uint32_t ubCSize = ArchTag::UB_SIZE / STAGES - ubASize - ubBSize;

        ubTileNum = ubCSize / ubBSize;
        ubTileSize = ubBSize / sizeof(ElementB);

        uint32_t ubAOffset = ubBufAddrStart;
        uint32_t ubBOffset = ubBufAddrStart + ubASize * STAGES;
        uint32_t ubCOffset = ubBOffset + ubBSize * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            ubATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(ubAOffset + ubASize * i);
            ubBTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementB>(ubBOffset + ubBSize * i);
            ubCTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubCOffset + ubCSize * i);
            
            inEventIds[i] = i;
            outEventIds[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inEventIds[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outEventIds[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadAiv()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inEventIds[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outEventIds[i]);
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(inEventIds[UbInStageId]));

        uint32_t mRound = RoundUp<UBA_ALIGN>(actualShape.m());
        ElementA scalarA[SCALAR_BUFFER_ELE_NUM];
        LayoutA layoutAInUb{mRound};
        LayoutB layoutBInUb{ubTileSize};
        LayoutC layoutCInUb{ubTileNum, ubTileSize};

        // load B
        auto layoutTileB = layoutB.GetTileLayout(TensorCoord(actualShape.n()));
        copyGmToUbB(ubBTensorList[UbInStageId], gmBlockB[0], layoutBInUb, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(inEventIds[UbInStageId]));

        // load A
        auto layoutTileA = layoutA.GetTileLayout(TensorCoord(actualShape.m()));
        copyGmToUbA(ubATensorList[UbInStageId], gmBlockA[0], layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((event_t)(inEventIds[UbInStageId]));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((event_t)(inEventIds[UbInStageId]));
        for (uint32_t i = 0; i < actualShape.m(); i++) {
            scalarA[i] = ubATensorList[UbInStageId].GetValue(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(inEventIds[UbInStageId]));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(inEventIds[UbInStageId]));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(inEventIds[UbInStageId]));

        auto mTiles = CeilDiv(actualShape.m(), ubTileNum);
        uint32_t lastTileM = actualShape.m() - (mTiles - 1) * ubTileNum;
        // tile vmuls && write c to gm
        for (uint32_t i = 0; i < mTiles; i++) {
            auto actualTileM = (i == mTiles - 1) ? lastTileM : ubTileNum;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(outEventIds[UbOutStageId]));
            for (uint32_t j = 0; j < actualTileM; j++) {
                tileVmuls(ubCTensorList[UbOutStageId][j * ubTileSize], ubBTensorList[UbInStageId],
                    scalarA[i * ubTileNum + j], actualShape.n());
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(outEventIds[UbOutStageId]));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(outEventIds[UbOutStageId]));
            MatrixCoord gmTileCOffset{i * ubTileNum, 0};
            auto gmTileC = gmBlockC[layoutC.GetOffset(gmTileCOffset)];
            LayoutC layoutCBlock = layoutC.GetTileLayout(MakeCoord(actualTileM, actualShape.n()));
            LayoutC lastLayoutCInUb = layoutCInUb.GetTileLayout(MakeCoord(actualTileM, actualShape.n()));
            copyUbToGmC(gmTileC, ubCTensorList[UbOutStageId], layoutCBlock, lastLayoutCInUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(outEventIds[UbOutStageId]));
            UbOutStageId = (UbOutStageId + 1 < STAGES) ? (UbOutStageId + 1) : 0;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(inEventIds[UbInStageId]));
        UbInStageId = (UbInStageId + 1 < STAGES) ? (UbInStageId + 1) : 0;
    }
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> ubATensorList[STAGES];
    AscendC::LocalTensor<ElementB> ubBTensorList[STAGES];
    AscendC::LocalTensor<ElementC> ubCTensorList[STAGES];

    CopyGmToUbA copyGmToUbA;
    CopyGmToUbB copyGmToUbB;
    CopyUbToGmC copyUbToGmC;
    TileVmuls tileVmuls;

    int32_t inEventIds[STAGES];
    int32_t outEventIds[STAGES];

    uint32_t UbOutStageId{0};
    uint32_t UbInStageId{0};

    uint32_t ubTileNum{0};
    uint32_t ubTileSize{0};
};

template <
    uint32_t SCALAR_BUFFER_ELE_NUM_,
    bool IS_TILE_M_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileVmuls_
>
struct BlockMmadAiv <
    MmadAtlasA2DynamicAivSimple<SCALAR_BUFFER_ELE_NUM_, IS_TILE_M_>,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2DynamicAivSimple<SCALAR_BUFFER_ELE_NUM_, IS_TILE_M_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using CopyGmToUbA = typename TileCopy_::CopyGmToUbA;
    using CopyGmToUbB = typename TileCopy_::CopyGmToUbB;
    using CopyUbToGmC = typename TileCopy_::CopyUbToGmC;
    using TileVmuls = TileVmuls_;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    static constexpr uint32_t UBA_ALIGN = BYTE_PER_BLK / sizeof(ElementA);
    static constexpr uint32_t UBB_ALIGN = BYTE_PER_BLK / sizeof(ElementB);
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = DispatchPolicy::SCALAR_BUFFER_ELE_NUM;
    static constexpr bool IS_TILE_M = DispatchPolicy::IS_TILE_M;
    
    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmadAiv(Arch::Resource<ArchTag> &resource, MatrixCoord taskShape, uint32_t ubBufAddrStart = 0)
    {
        uint32_t ubASize = SCALAR_BUFFER_ELE_NUM * sizeof(ElementA);
        uint32_t ubBSize = taskShape.column() * sizeof(ElementB);
        uint32_t ubCSize = ArchTag::UB_SIZE - ubASize - ubBSize;

        ubTileNum = ubCSize / ubBSize;
        ubTileSize = ubBSize / sizeof(ElementB);

        uint32_t ubAOffset = ubBufAddrStart;
        uint32_t ubBOffset = ubBufAddrStart + ubASize;
        uint32_t ubCOffset = ubBOffset + ubBSize;
        // Init buffers
        ubATensor = resource.ubBuf.template GetBufferByByte<ElementA>(ubAOffset);
        ubBTensor = resource.ubBuf.template GetBufferByByte<ElementB>(ubBOffset);
        ubCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubCOffset);
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadAiv()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape)
    {
        uint32_t mRound = RoundUp<UBA_ALIGN>(actualShape.m());

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(EVENT_ID0));

        ElementA scalarA[SCALAR_BUFFER_ELE_NUM];
        LayoutA layoutAInUb{mRound};
        LayoutB layoutBInUb{ubTileSize};
        LayoutC layoutCInUb{ubTileNum, ubTileSize};

        // load B
        auto layoutTileB = layoutB.GetTileLayout(TensorCoord(actualShape.n()));
        copyGmToUbB(ubBTensor, gmBlockB[0], layoutBInUb, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID1));
        // load A
        auto layoutTileA = layoutA.GetTileLayout(TensorCoord(actualShape.m()));
        copyGmToUbA(ubATensor, gmBlockA[0], layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((event_t)(EVENT_ID0));
        if constexpr (IS_TILE_M) {
            for (uint32_t i = 0; i < actualShape.m(); i++) {
                scalarA[i] = ubATensor.GetValue(i);
            }
        } else {
            scalarA[0] = ubATensor.GetValue(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(EVENT_ID0));

        // compute && write c to gm
        LayoutC layoutCBlock = layoutC.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        LayoutC lastLayoutCInUb = layoutCInUb.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        if constexpr (IS_TILE_M) {
            for (uint32_t j = 0; j < actualShape.m(); j++) {
                tileVmuls(ubCTensor[j * ubTileSize], ubBTensor, scalarA[j], actualShape.n());
            }
        } else {
            tileVmuls(ubCTensor[0], ubBTensor, scalarA[0], actualShape.n());
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(EVENT_ID0));
        copyUbToGmC(gmBlockC[0], ubCTensor, layoutCBlock, lastLayoutCInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(EVENT_ID0));
    }
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> ubATensor;
    AscendC::LocalTensor<ElementB> ubBTensor;
    AscendC::LocalTensor<ElementC> ubCTensor;

    CopyGmToUbA copyGmToUbA;
    CopyGmToUbB copyGmToUbB;
    CopyUbToGmC copyUbToGmC;
    TileVmuls tileVmuls;

    uint32_t ubTileNum{0};
    uint32_t ubTileSize{0};
};


template <
    uint32_t SCALAR_BUFFER_ELE_NUM_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileVmuls_
>
struct BlockMmadAiv <
    MmadAtlasA2DynamicAivTrans<SCALAR_BUFFER_ELE_NUM_>,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileVmuls_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2DynamicAivTrans<SCALAR_BUFFER_ELE_NUM_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileVmuls = TileVmuls_;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    using CopyGmToUbB = typename TileCopy_::CopyGmToUbB;
    using CopyGmToUbA = typename TileCopy_::CopyGmToUbA;
    using CopyUbToGmC = typename TileCopy_::CopyUbToGmC;

    static constexpr uint32_t UBA_ALIGN = BYTE_PER_BLK / sizeof(ElementA);
    static constexpr uint32_t UBB_ALIGN = BYTE_PER_BLK / sizeof(ElementB);
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = DispatchPolicy::SCALAR_BUFFER_ELE_NUM;
    static constexpr uint16_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementC);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementC);
    static constexpr uint32_t VNCHW_SIZE = 16;
    
    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmadAiv(Arch::Resource<ArchTag> &resource, MatrixCoord taskShape, uint32_t ubBufAddrStart = 0)
    {
        ubTileSize = RoundUp<ELE_NUM_PER_C0>(taskShape.row());
        ubTileNum = RoundUp<C0_NUM_PER_FRACTAL>(taskShape.column());
        uint32_t ubBSize = SCALAR_BUFFER_ELE_NUM * sizeof(ElementB);
        uint32_t ubASize = ubTileSize * sizeof(ElementA);
        uint32_t ubCSize = ubTileNum * ubTileSize * sizeof(ElementA);
        uint32_t ubTransposeCSize = ubTileNum * ubTileSize * sizeof(ElementA);

        uint32_t ubBOffset = ubBufAddrStart;
        uint32_t ubAOffset = ubBufAddrStart + ubBSize;
        uint32_t ubTransposeCOffset = ubAOffset + ubASize;
        uint32_t ubCOffset = ubTransposeCOffset + ubTransposeCSize;
        // Init buffers
        ubBTensor = resource.ubBuf.template GetBufferByByte<ElementB>(ubBOffset);
        ubATensor = resource.ubBuf.template GetBufferByByte<ElementA>(ubAOffset);
        ubTransposeCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubTransposeCOffset);
        ubCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubCOffset);
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadAiv()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID1));

        ElementA scalarB[SCALAR_BUFFER_ELE_NUM];
        LayoutA layoutBInUb{SCALAR_BUFFER_ELE_NUM};
        LayoutB layoutAInUb{ubTileSize};

        // load A
        auto layoutTileA = layoutA.GetTileLayout(TensorCoord(actualShape.m()));
        copyGmToUbA(ubATensor, gmBlockA[0], layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID1));
        // load B
        auto layoutTileB = layoutB.GetTileLayout(TensorCoord(actualShape.n()));
        copyGmToUbB(ubBTensor, gmBlockB[0], layoutBInUb, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((event_t)(EVENT_ID0));
        for (uint32_t i = 0; i < actualShape.n(); i++) {
            scalarB[i] = ubBTensor.GetValue(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(EVENT_ID0));

        // compute && write c to ub
        for (uint32_t i = 0; i < actualShape.n(); i++) {
            tileVmuls(ubCTensor[i * ubTileSize], ubATensor, scalarB[i], actualShape.m());
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(EVENT_ID1));

        LayoutC layoutCInUb{ubTileNum, ubTileSize};
        LayoutC layoutTranposeCInUb{ubTileSize, ubTileNum};
        auto actualLayoutC = layoutC.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        auto actualTransLayoutC = layoutTranposeCInUb.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));

        // transpose 16*16 block
        auto mTiles = CeilDiv<C0_NUM_PER_FRACTAL>(actualShape.n());
        auto usedBufferNum = CeilDiv<ELE_NUM_PER_C0>(actualShape.m());
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)(EVENT_ID2));
        uint64_t dstLocalList[VNCHW_SIZE];
        uint64_t srcLocalList[VNCHW_SIZE];
        for (uint32_t i = 0; i < mTiles; ++i) {
            for (uint32_t j = 0; j < VNCHW_SIZE; ++j) {
                srcLocalList[j] = ubCTensor.GetPhyAddr() + i * C0_NUM_PER_FRACTAL * ubTileSize * sizeof(ElementC)
                    + j * ubTileSize * sizeof(ElementC);
                if constexpr (std::is_same_v<ElementC, half>) {
                    dstLocalList[j] = ubTransposeCTensor.GetPhyAddr() + i * C0_NUM_PER_FRACTAL * sizeof(ElementC)
                        + j * ubTileNum * sizeof(ElementC);
                }
                if constexpr (std::is_same_v<ElementC, float>) {
                    dstLocalList[j] = ubTransposeCTensor.GetPhyAddr() + i * C0_NUM_PER_FRACTAL * sizeof(ElementC)
                        + j / 2 * ubTileNum * sizeof(ElementC) + j % 2 * BYTE_PER_C0;
                }
            }
            if (usedBufferNum > 1) {
                AscendC::TransDataTo5HDParams params(false, false, usedBufferNum, ubTileNum, 1);
                AscendC::TransDataTo5HD<ElementC>(dstLocalList, srcLocalList, params);
            } else {
                AscendC::TransDataTo5HDParams paramsRepeatOne{false, false, 1, 0, 0};
                AscendC::TransDataTo5HD<ElementC>(dstLocalList, srcLocalList, paramsRepeatOne);
            }

        }
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(EVENT_ID0));
        // write to gmC
        copyUbToGmC(gmBlockC[0], ubTransposeCTensor, actualLayoutC, actualTransLayoutC);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)(EVENT_ID2));
    }
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> ubATensor;
    AscendC::LocalTensor<ElementB> ubBTensor;
    AscendC::LocalTensor<ElementC> ubCTensor;
    AscendC::LocalTensor<ElementC> ubTransposeCTensor;

    CopyGmToUbA copyGmToUbA;
    CopyGmToUbB copyGmToUbB;
    CopyUbToGmC copyUbToGmC;
    TileVmuls tileVmuls;

    uint32_t stageId{0};
    uint32_t ubTileNum{0};
    uint32_t ubTileSize{0};
};

}
#endif // CATLASS_GEMM_BLOCK_MMAD_DYNAMIC_AIV_HPP