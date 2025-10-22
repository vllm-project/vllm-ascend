/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_QUANT_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_QUANT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

namespace Catlass::Epilogue::Block {

template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantQuant<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantQuant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using ElementS = half;
    using LayoutS = layout::VectorLayout;

    // Check data infos
    static_assert(
        std::is_same_v<ElementC, half> && ( std::is_same_v<ElementD, int8_t>),
        "The element type template parameters of BlockEpilogue are wrong"
    );
    static_assert(
        std::is_same_v<LayoutC, layout::RowMajor> && 
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::RowMajor>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyUbToGmALL = Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementD, LayoutS>>;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, __gm__ int32_t *ptrTokenPerExpert_) : ptrTokenPerExpert(ptrTokenPerExpert_), EP(EP_), expertPerRank(expertPerRank_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        constexpr int32_t blockN = 7168;

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += blockN * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += (blockN * sizeof(ElementD) + blockN * sizeof(ElementS) / 8);
            ubCFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += blockN * sizeof(float);
            ubCAbsList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += blockN * sizeof(float);
            reduceMaxList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += blockN * sizeof(float) / 8;

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDSMTE3VList[i] = eventMTE3V++;
            eventUbDSVMTE3List[i] = eventVMTE3++;


            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDSMTE3VList[i]);
        }
    }

        CATLASS_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDSMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
      
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        MatrixCoord const &shapeC,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale,
        AscendC::GlobalTensor<ElementD> const &gmDS
    )
    {

        uint32_t blockM = shapeC.row();
        uint32_t blockN = shapeC.column();

        uint32_t tileLoops = blockM;
      
        uint32_t repeatNum = blockN * sizeof(float) / BYTE_PER_VECTOR_FRACTAL;
        uint32_t scaleNum = repeatNum * BLK_NUM_PER_VECTOR_FRACTAL;
        uint32_t stepN = blockN * sizeof(ElementD) + scaleNum * sizeof(ElementS);

        for (uint32_t loopIdx = 0; loopIdx < tileLoops; loopIdx ++) {

            auto gmTileC = gmC[loopIdx * blockN];
            auto ubC = ubCList[ubListId];
            auto &ubCFp32 = ubCFp32List[ubListId];
            auto &ubCAbs = ubCAbsList[ubListId];

            auto &reduceMax = reduceMaxList[ubListId];
            auto &scaleDupLocal = ubCAbsList[ubListId];
            auto ubCFp16 = ubCFp32List[ubListId].template ReinterpretCast<half>();
            
            //前一半放数据，后一半放量化的scale值
            auto &ubD = ubDList[ubListId];
            auto scaleFp16 = ubDList[ubListId][blockN].template ReinterpretCast<half>();

            auto gmTileDS = gmDS[loopIdx * stepN];

            LayoutC layoutUbC{1, blockN};

            // 把C从GM workspace搬到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // 在UB上做把C cast成FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, blockN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            // 获取pertoken scale值，gmPerTokenScale的第loopIdx行
            ElementPerTokenScale perTokenScale = gmPerTokenScale(loopIdx);

            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            // pertoken scale值与FP32的C做Muls乘法
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32, ubCFp32, perTokenScale, blockN);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(ubCAbs, ubCFp32, blockN);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax(reduceMax, ubCAbs, repeatNum, 64, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(reduceMax, reduceMax, 1.0f / 127, scaleNum); // 有效个数
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDSMTE3VList[ubListId]);
            //scale的量化
            AscendC::Cast(scaleFp16, reduceMax, AscendC::RoundMode::CAST_RINT, scaleNum); // 有效个数
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(scaleDupLocal, reduceMax, repeatNum, {1, 8}); // 一次256
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(ubCFp32, ubCFp32, scaleDupLocal, blockN); // 有效个数
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubCFp16, ubCFp32, AscendC::RoundMode::CAST_RINT, blockN);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ubD, ubCFp16, AscendC::RoundMode::CAST_RINT, blockN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDSVMTE3List[ubListId]);

            LayoutS layoutOut{ stepN };
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDSVMTE3List[ubListId]);
            copyUbToGmALL(gmTileDS, ubD, layoutOut, layoutOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDSMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDSMTE3VList[UB_STAGES];
    int32_t eventUbDSVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubCAbsList[UB_STAGES];
    AscendC::LocalTensor<float> reduceMaxList[UB_STAGES];

    

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    CopyUbToGmALL copyUbToGmALL;
};

}  

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_QUANT_HPP
