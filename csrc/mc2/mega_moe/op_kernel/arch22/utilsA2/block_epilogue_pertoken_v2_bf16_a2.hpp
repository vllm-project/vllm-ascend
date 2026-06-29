/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_BF16_A2_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_BF16_A2_HPP

#include "../template_linear_algebra_v2/catlass.hpp"
#include "../template_linear_algebra_v2/arch/resource.hpp"
#include "../template_linear_algebra_v2/epilogue/dispatch_policy.hpp"
#include "../template_linear_algebra_v2/gemm_coord.hpp"
#include "../template_linear_algebra_v2/matrix_coord.hpp"
#include "../template_linear_algebra_v2/layout/layout.hpp"
#include "../template_linear_algebra_v2/detail/callback.hpp"

#include "../utils/hccl_shmem.hpp"
#include "../utils/layout3d.hpp"

namespace Catlass::Epilogue::Block {

// A2-only: IS_A2 is always true, no IS_A2_T parameter needed.
template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using CopyScaleGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, layout::VectorLayout>>;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        HcclShmem<true> shmem;
        int32_t offsetD;
        int32_t serverId;

        CATLASS_DEVICE
        Params() {};
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_,
        LayoutC layoutC_, int32_t n2_, int32_t n0_, HcclShmem<true>& shmem_, int32_t offsetD_,
        int32_t serverId_ = 0) :
        ptrTokenPerExpert(ptrTokenPerExpert_), EP(EP_),
        expertPerRank(expertPerRank_),rank(rank_), layoutC(layoutC_), n2(n2_), n0(n0_),
        shmem(shmem_), offsetD(offsetD_), serverId(serverId_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        n0 = params.n0;
        size_t ubOffset = 0;
        for(int32_t i = 0; i < 2; i++) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += max_len * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += max_len * sizeof(ElementD);
            ubFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += max_len * sizeof(float);
            scaleUbList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += (max_len / n0) * sizeof(float);
            source_scale_offset[i] = -1;
        }
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert));
        tokenPerExpertLayout = Layout3D(AlignUp(params.EP * params.expertPerRank, ALIGN_128), params.expertPerRank);
        is_ping = true;
    }

    CATLASS_DEVICE
    void Finalize()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale,
        GemmCoord& blockCoord,
        GemmCoord& actualBlockShape,
        int32_t groupIdx,
        int32_t preSrcExpertSum,
        AscendC::GlobalTensor<int32_t> preSumBeforeRank
    ){
        // BF16 path: per-token scale is not used (no dequant needed).
        // Delegate to the 6-arg copy-only operator.
        (void)gmPerTokenScale;
        this->operator()(gmC, blockCoord, actualBlockShape, groupIdx, preSrcExpertSum, preSumBeforeRank);
    }

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        GemmCoord& blockCoord,
        GemmCoord& actualBlockShape,
        int32_t groupIdx,
        int32_t preSrcExpertSum,
        AscendC::GlobalTensor<int32_t> preSumBeforeRank
    ){
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;

        auto &ubC = ubCList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id); // for debug
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id); // for debug

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx ++) {
            int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx);
            int32_t stRankInExpert = preSumRankInExpert;
            int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
            preSumRankInExpert += lenRankInExpert;
            if (stRankInExpert >= edTile) {
                break;
            }
            else if (edRankInExpert <= stTile) {
                continue;
            }
            int32_t stData = max(stRankInExpert, stTile);
            int32_t edData = min(edRankInExpert, edTile);
            uint32_t lenData = edData - stData;
            if (lenData <= 0){
                continue;
            }

            uint32_t dstOffsetInExpert = 0;
            if (stTile > stRankInExpert) {
                dstOffsetInExpert = stTile - stRankInExpert;
            }
            AscendC::GlobalTensor<ElementD> gmRemotePeer;
            __gm__ void* dstPeermemPtr = params.shmem(params.offsetD, dstEpIdx);
            gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr));
            MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset, blockCoord.n()};
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            auto gmTileD = gmRemotePeer[gmDstOffset];
            LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
            LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
            bool isCrossServer = (dstEpIdx / SERVER_RANK_SIZE_A2) != params.serverId;
            if (isCrossServer) {
                AscendC::GlobalTensor<ElementD> gmLocalWindowsOut;
 	            gmLocalWindowsOut.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(
 	                params.shmem.windowsOutAddr() + params.offsetD));
                MatrixCoord srcOffset{(uint32_t)(stData + preSrcExpertSum), blockCoord.n()};
                int64_t gmSrcOffset = params.layoutC.GetOffset(srcOffset);
                auto gmTileLocal = gmLocalWindowsOut[gmSrcOffset];
                copyUbToGmD(gmTileLocal, ubC[tileOffset *  n0], layoutGM2, layoutUB2);
            } else {
                copyUbToGmD(gmTileD, ubC[tileOffset *  n0], layoutGM2, layoutUB2);
            }
            tileOffset += lenData;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }

private:
    Params params;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32List[UB_STAGES];
    AscendC::LocalTensor<float> scaleUbList[UB_STAGES];
    int32_t source_scale_offset[UB_STAGES];

    int32_t max_len = 8 * 32 / 4 * 128;
    int32_t n0;
    bool is_ping = false;

    int32_t repeat = 128;

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;

    CopyScaleGmToUb copyScaleGmToUb;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    Layout3D tokenPerExpertLayout;
};
}
#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_BF16_A2_HPP