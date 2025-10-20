/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_UNQUANT_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_UNQUANT_HPP

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
    class DType_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2UnQuant<UB_STAGES_>,
    CType_,
    DType_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2UnQuant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(
        (std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t>) && (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>),
        "The element type template parameters of BlockEpilogue are wrong"
    );


    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    // using TileShape = typename TileRowBroadcastMul::TileShape;

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
        size_t ubOffset = 4096;

        int32_t eventMTE3MTE2 = 0;
        int32_t eventMTE2MTE3 = 0;
        constexpr int32_t blockN = 12000;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += blockN * sizeof(ElementC);

            eventUbCMTE3MTE2List[i] = eventMTE3MTE2++;
            eventUbCMTE2MTE3List[i] = eventMTE2MTE3++;

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbCMTE3MTE2List[i]);
        }
    }
    CATLASS_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbCMTE3MTE2List[i]);
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
    // 每个tile就是1*7168，每个block是一个expert的所有token=[group[i], 7168]
    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        MatrixCoord const &shapeC,
        AscendC::GlobalTensor<ElementD> const &gmD
    )
    {
        uint32_t blockM = shapeC.row();
        uint32_t blockN = shapeC.column();

        uint32_t tileLoops = blockM;
        uint32_t subblockIdx = get_block_idx() * 2 + get_subblockid();
        uint32_t subblockNum = get_block_num() * 2;
        for (uint32_t loopIdx = 0; loopIdx < tileLoops; loopIdx ++) {
            auto gmTileC = gmC[loopIdx * blockN];
            auto &ubC = ubCList[ubListId];
            auto gmTileD = gmD[loopIdx * blockN];
            LayoutC layoutUbC{1, blockN};

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbCMTE3MTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventUbCMTE2MTE3List[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventUbCMTE2MTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubC, layoutUbC, layoutUbC);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbCMTE3MTE2List[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];

    int32_t eventUbCMTE3MTE2List[UB_STAGES];
    int32_t eventUbCMTE2MTE3List[UB_STAGES];

    uint32_t ubListId{0};

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_ONLY_HPP
