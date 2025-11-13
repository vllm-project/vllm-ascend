/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
 #define CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
 
 #include "catlass/catlass.hpp"
 #include "catlass/arch/resource.hpp"
 #include "catlass/coord.hpp"
 #include "catlass/gemv_coord.hpp"
 #include "catlass/gemv/helper.hpp"
 #include "catlass/gemm/helper.hpp"
 #include "catlass/layout/layout.hpp"
 #include "catlass/detail/alignment.hpp"
 #include "catlass/gemm/dispatch_policy.hpp"

 namespace Catlass::Gemv::Block {
 template <
     class UBTileShape_,
     class AType_,
     class XType_,
     class YType_,
     class BiasType_,
     class TileCopy_,
     class TileVmad_,
     class TileVmuls_
 >
 struct BlockGemv <
     Gemm::GemvAtlasA2,
     UBTileShape_,
     AType_,
     XType_,
     YType_,
     BiasType_,
     TileCopy_,
     TileVmad_,
     TileVmuls_
 > {
 public:
     // Type Aliases
     using DispatchPolicy = Gemm::GemvAtlasA2;
     using ArchTag = typename DispatchPolicy::ArchTag;
     using UBTileShape = UBTileShape_;
     using ElementA = typename AType_::Element;
     using LayoutA = typename AType_::Layout;
     using ElementX = typename XType_::Element;
     using LayoutX = typename XType_::Layout;
     using ElementY = typename YType_::Element;
     using LayoutY = typename YType_::Layout;
     using TileVmad = TileVmad_;
     using TileVmuls = TileVmuls_;
     using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
     using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
     using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
     using ElementAccumulator =
         typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    
     using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA>;
     using TensorCoord = layout::VectorLayout::TensorCoord;
     static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
     static constexpr uint32_t Abuf_SIZE_ = 128 * 1024; 
     static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;  
     static constexpr uint32_t Ybuf_SIZE_ = 16 * 1024;  
     static constexpr uint32_t workspace_SIZE_ = 32 * 1024; 
     CATLASS_DEVICE
     BlockGemv(){}
     /// Construct
     CATLASS_DEVICE
     BlockGemv(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
     {
        
        
         uint32_t UbAOffset = UBufAddrStart;
         uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
         uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_+Xbuf_SIZE_;
         uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_+Xbuf_SIZE_+Ybuf_SIZE_;
         // Init buffers
         for (uint32_t i = 0; i < STAGES; i++) {
             // Assign L1/L0A/L0B space for each stages
             UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset+i * (Abuf_SIZE_/2));
             UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset+i * (Xbuf_SIZE_/2));
             UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset+i * (Ybuf_SIZE_/2));
             UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset+i * (workspace_SIZE_/2));
 
             // Assign event ID for each stages
             UbInAEventList[i] = i;
             UbInXEventList[i] = i + STAGES;
             UbOutEventList[i] = i;
             
             // The event id that needs to be set before the loop
             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
             AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
         }
     }
 
     /// Destructor
     CATLASS_DEVICE
     ~BlockGemv()
     {
         for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
         }
         
     }
 
     CATLASS_DEVICE
     void operator()(
         AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
         AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
         AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
         AscendC::GlobalTensor<ElementY> const &gmZ,
         GemvCoord const &actualShape,
         float alpha,
         float beta
        )
     {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        vecCopyGmToUb(UbYTensorList[UbOutListId], gmY,actualShape.m());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));  
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
        tileVmuls(
            UbYTensorList[UbOutListId], 
            UbYTensorList[UbOutListId], 
            (ElementY)beta,
            actualShape.m()
        );
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));  
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        
        TileMRound = RoundUp(UBTileShape::M,UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N,UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_catlassual = (actualShape.m() < TileMRound) ? actualShape.m():TileMRound;
        n_catlassual = (actualShape.n() < TileNRound) ? actualShape.n():TileNRound;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
        vecCopyGmToUb(UbXTensorList[UbInListId], gmX,n_catlassual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
        
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_catlassual, n_catlassual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
         // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(),TileNRound);
        for(uint32_t LoopIdx = 0;LoopIdx < Nloop; LoopIdx++){
            m_catlassual = (actualShape.m() < TileMRound) ? actualShape.m():TileMRound;
            n_catlassual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_catlassual = m_catlassual;
            x_catlassual = n_catlassual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if(LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_catlassual_next = m_catlassual;
                uint32_t n_catlassual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_catlassual_next = m_catlassual_next;
                uint32_t x_catlassual_next = n_catlassual_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbATensorList[UbInListIdNext];
                auto vecTensor = UbXTensorList[UbInListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                vecCopyGmToUb(vecTensor, gmX[LoopIdxNext * TileNRound],x_catlassual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_catlassual_next, n_catlassual_next));
                matrixCopyGmToUb(matrixTensor, gmA[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                 
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
            tileVmuls(
                UbXTensorList[UbInListId], 
                UbXTensorList[UbInListId], 
                (ElementA)alpha,
                x_catlassual
            );
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId])); 
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_catlassual, n_catlassual));
            tileVmad(
                UbYTensorList[UbOutListId],
                UbXTensorList[UbInListId],
                UbATensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute
            );
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_catlassual));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_catlassual));
        vecCopyUbToGm(gmZ, UbYTensorList[UbOutListId],layoutDstY, layoutComputeInUb); 
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
     }
 
 protected:
     // Multi-stage tensors list
     AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
     AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
     AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
     AscendC::LocalTensor<ElementAccumulator> UbWTensorList[STAGES];
 
     // Multi-stage event id list
     int32_t UbInAEventList[STAGES];
     int32_t UbInXEventList[STAGES];
     int32_t UbOutEventList[STAGES];
 
     // The id of current stage
     uint32_t UbOutListId{0};
     uint32_t UbInListId{0};

     uint32_t m_catlassual, n_catlassual,x_catlassual, y_catlassual;
     uint32_t TileMRound,TileNRound;
     uint32_t strideA;

     TileVmad tileVmad;
     TileVmuls tileVmuls;
     MatrixCopyGmToUb matrixCopyGmToUb;
     VecCopyGmToUb vecCopyGmToUb;
     VecCopyUbToGm vecCopyUbToGm;
     
 };
 
 } // namespace Catlass::Gemv::Block
 
 #endif // CATLASS_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
 