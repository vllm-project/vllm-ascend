#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "dispatch_policy_custom.hpp"
#include "get_tensor_addr.hpp"
#include "hccl_shmem.hpp"
#include "layout3d.hpp"

#define NO_POST_PROCESS

namespace Catlass::Epilogue::Block {
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2W4A8PostPerTokenDequantV2<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA2W4A8PostPerTokenDequantV2<UB_STAGES_>;
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
    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using TileCopy2 =
        Epilogue::Tile::TileCopy<Arch::AtlasA2, Gemm::GemmType<float32_t, layout::RowMajor>,
                                 Gemm::GemmType<uint64_t, layout::VectorLayout>,
                                 Gemm::GemmType<float, layout::VectorLayout>, Gemm::GemmType<float, layout::RowMajor>>;
    using CopyUbToGmGMM2 = typename TileCopy2::CopyUbToGmD;
    using CopyGMToUBW = typename TileCopy2::CopyGmToUbC;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        HcclShmem shmem;
        int32_t offsetD;

        CATLASS_DEVICE
        Params(){};
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_, LayoutC layoutC_,
               int32_t n2_, int32_t n0_, HcclShmem &shmem_, int32_t offsetD_)
            : ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              rank(rank_),
              layoutC(layoutC_),
              n2(n2_),
              n0(n0_),
              shmem(shmem_),
              offsetD(offsetD_)
        {
        }
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        //ub:192KB
        n0 = params.n0;
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventVMTE3 = 0;
        int32_t eventMTE3V = 0;
        for (int32_t i = 0; i < UB_STAGES; ++i) {
            ubCHList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);  // Upper 4 bits
            ubOffset += max_len * sizeof(ElementC);
            ubCLList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);  // Lower 4 bits
            ubOffset += max_len * sizeof(ElementC);
            ubweighAuxList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += n0 * sizeof(float);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += max_len * sizeof(ElementD);

            eventUbTileCVMTE2List[i] = eventVMTE2++;
            eventUbTileCMTE2VList[i] = eventMTE2V++;
            eventUbTileWMTE2VList[i] = eventMTE2V++;
            eventUbTileDVMTE3List[i] = eventVMTE3++;
            eventUbTileDMTE3VList[i] = eventMTE3V++;
        }

        ubFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);  // Upper 4 bits to fp32
        ubOffset += max_len * sizeof(float);
        ubFp32L = resource.ubBuf.template GetBufferByByte<float>(ubOffset);  // Lower 4 bits to fp32
        ubOffset += max_len * sizeof(float);

        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert));
        tokenPerExpertLayout = Layout3D(AlignUp(params.EP * params.expertPerRank, 128), params.expertPerRank);
        is_ping = true;
        rankIdx = params.rank;
    }

    CATLASS_DEVICE
    void InitFlag()
    {
        for (int32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbTileCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbTileDVMTE3List[i]);
        }
#ifdef W4A8_DEBUG
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
#endif
    }

    CATLASS_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbTileCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbTileDVMTE3List[i]);
        }
#ifdef W4A8_DEBUG
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
#endif
    }
    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<float> const &gmGMM2, AscendC::GlobalTensor<ElementC> const &gmC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale,
                    __gm__ float *gmWeightAux, GemmCoord &blockCoord, GemmCoord &actualBlockShape,
                    int32_t groupIdx, int32_t preSrcExpertSum, AscendC::GlobalTensor<int32_t> preSumBeforeRank,
                    int32_t listLen)
    {
        auto &ubCH = ubCHList[ubListId];
        auto &ubCL = ubCLList[ubListId];
        auto &ubD = ubDList[ubListId];
        auto &ubweighAux = ubweighAuxList[ubListId];
        int32_t gmCOffsetH = (preSrcExpertSum * params.n2 + blockCoord.m() * params.n2) + blockCoord.n();
        int32_t gmCOffsetL = (preSrcExpertSum * params.n2 + blockCoord.m() * params.n2) + blockCoord.n() + params.n2;
        auto gmTileCH = gmC[gmCOffsetH];
        auto gmTileCL = gmC[gmCOffsetL];

        AscendC::GlobalTensor<float> weightAux;
        AscendC::GlobalTensor<float> gmWeighAux;

        int32_t arrayGroupIdx = listLen == 1 ? 0 : groupIdx;
        weightAux.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(GetTensorAddr<float>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(gmWeightAux))));
        int64_t gmOffsetwA = blockCoord.n() + (listLen == 1 ? groupIdx * params.n2 : 0);
        gmWeighAux = weightAux[gmOffsetwA];


#ifdef W4A8_DEBUG
        int32_t gmCOffset = (preSrcExpertSum * params.n2 + blockCoord.m() * params.n2) / 2 + blockCoord.n();
        auto gmTileGMM2 = gmGMM2[gmCOffset];
#endif

        constexpr float DEFAULT_MUL_SCALE = 16.0f;

        LayoutC layoutGM{actualBlockShape.m() / 2, actualBlockShape.n(), params.n2};
        LayoutC layoutGM2{actualBlockShape.m() / 2, actualBlockShape.n(), params.n2 * 2};
        LayoutC layoutUB{actualBlockShape.m() / 2, actualBlockShape.n(), n0};
        LayoutC layoutUBW{1, actualBlockShape.n(), n0};

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbTileCVMTE2List[ubListId]);
        copyGmToUbC(ubCH, gmTileCH, layoutUB, layoutGM2);
        copyGmToUbC(ubCL, gmTileCL, layoutUB, layoutGM2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbTileCMTE2VList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbTileCMTE2VList[ubListId]);
        DataCopyExtParams copyParamsW{1, static_cast<uint32_t>(actualBlockShape.n() * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParamsW{false, 0, 0, 0};
        AscendC::DataCopyPad(ubweighAux, gmWeighAux, copyParamsW, padParamsW);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbTileWMTE2VList[ubListId]);

#ifdef W4A8_DEBUG
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
#endif
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<float, ElementC, false>(ubFp32, ubCH, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
        PipeBarrier<PIPE_V>();
        AscendC::Cast<float, ElementC, false>(ubFp32L, ubCL, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
        PipeBarrier<PIPE_V>();
        AscendC::Muls(ubFp32, ubFp32, DEFAULT_MUL_SCALE, actualBlockShape.m() * actualBlockShape.n() / 2);
        PipeBarrier<PIPE_V>();
        AscendC::Add(ubFp32, ubFp32, ubFp32L, actualBlockShape.m() * actualBlockShape.n() / 2);
        // Add W4A8 auxiliary matrix on UB
        PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbTileWMTE2VList[ubListId]);
        for (uint32_t i = 0; i < actualBlockShape.m() / 2; ++i) {
            AscendC::Add(ubFp32[i * n0], ubFp32[i * n0], ubweighAux, n0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbTileCVMTE2List[ubListId]);
        AscendC::PipeBarrier<PIPE_V>();

        int32_t gmScaleOffset = (preSrcExpertSum + blockCoord.m()) / 2;
        for (int32_t row = 0; row < actualBlockShape.m() / 2; ++row) {
            float scale = gmPerTokenScale(gmScaleOffset + row);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            Muls<float, false>(ubFp32[n0 * row], ubFp32[n0 * row], scale, -1, (actualBlockShape.n() + 127) / 128 * 2,
                               {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();

#ifdef W4A8_DEBUG
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        copyUbToGmGMM2(gmTileGMM2, ubFp32, layoutGM, layoutUB);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
#endif

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbTileDVMTE3List[ubListId]);
        AscendC::Cast<ElementD, float, false>(ubD, ubFp32, AscendC::RoundMode::CAST_RINT, -1, repeat, {1, 1, 4, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbTileDMTE3VList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbTileDMTE3VList[ubListId]);

        int32_t lenTile = actualBlockShape.m() / 2;
        int32_t stTile = blockCoord.m() / 2;
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx++) {
            int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx);
            int32_t stRankInExpert = preSumRankInExpert;
            int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
            preSumRankInExpert += lenRankInExpert;
            if (stRankInExpert >= edTile) {
                break;
            } else if (edRankInExpert <= stTile) {
                continue;
            }
            int32_t stData = max(stRankInExpert, stTile);
            int32_t edData = min(edRankInExpert, edTile);
            uint32_t lenData = edData - stData;
            if (lenData <= 0) {
                continue;
            }

            uint32_t dstOffsetInExpert = 0;
            if (stTile > stRankInExpert) {
                dstOffsetInExpert = stTile - stRankInExpert;
            }
            AscendC::GlobalTensor<ElementD> gmRemotePeer;
            __gm__ void *dstPeermemPtr = params.shmem(params.offsetD, dstEpIdx);
            gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(dstPeermemPtr));
            MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset, blockCoord.n()};
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            auto gmTileD = gmRemotePeer[gmDstOffset];
            LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
            LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
            copyUbToGmD(gmTileD, ubD[tileOffset * n0], layoutGM2, layoutUB2);
            tileOffset += lenData;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbTileDVMTE3List[ubListId]);

        ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
    }

private:
    Params params;
    AscendC::LocalTensor<ElementC> ubCHList[UB_STAGES];
    AscendC::LocalTensor<ElementC> ubCLList[UB_STAGES];
    AscendC::LocalTensor<float> ubweighAuxList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32;
    AscendC::LocalTensor<float> ubFp32L;

    int32_t max_len = 32 * 256;
    int32_t n0;
    bool is_ping = false;

    int32_t repeat = 128;

    int32_t eventUbTileCVMTE2List[UB_STAGES];
    int32_t eventUbTileCMTE2VList[UB_STAGES];
    int32_t eventUbTileWMTE2VList[UB_STAGES];
    int32_t eventUbTileDVMTE3List[UB_STAGES];
    int32_t eventUbTileDMTE3VList[UB_STAGES];

    uint32_t ubListId{0};

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    CopyUbToGmGMM2 copyUbToGmGMM2;
    CopyGMToUBW copyGMToUBW;

    CopyScaleGmToUb copyScaleGmToUb;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    Layout3D tokenPerExpertLayout;

    int32_t rankIdx;
};
}
#endif