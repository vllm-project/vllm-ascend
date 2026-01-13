#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "hccl_shmem.hpp"
#include "layout3d.hpp"

namespace Catlass::Epilogue::Block {
template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    //using CopyScaleGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, layout::RowMajor>>;
    using CopyScaleGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, layout::VectorLayout>>;
        // Tile copy
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
        HcclShmem shmem;
        int32_t offsetD;

        CATLASS_DEVICE
        Params() {};
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_,
        LayoutC layoutC_, int32_t n2_, int32_t n0_, HcclShmem& shmem_, int32_t offsetD_) :
        ptrTokenPerExpert(ptrTokenPerExpert_), EP(EP_),
        expertPerRank(expertPerRank_),rank(rank_), layoutC(layoutC_), n2(n2_), n0(n0_),
        shmem(shmem_), offsetD(offsetD_)
         {}
    };


    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);



        //ub:192KB
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
        tokenPerExpertLayout = Layout3D(params.EP * params.expertPerRank, params.expertPerRank);
        is_ping = true;
    }

    CATLASS_DEVICE
    void Finalize()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

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
        AscendC::GlobalTensor<int32_t> preSumBeforeRank,
        uint32_t *mPreSumBeforeRank
    ){
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
        auto event_id_2 = is_ping ? EVENT_ID2 : EVENT_ID3;

        auto &ubC = ubCList[is_ping];
        auto &ubD = ubDList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];
        auto &ubCFp32 = ubFp32List[is_ping];
        auto &scaleUb = scaleUbList[is_ping];
        // auto &ubOutFp32 = ubOutFp32List[is_ping];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};


        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id); //for debug
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id); //for debug

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::Cast<float, ElementC, false>(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);


        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event_id_2);

        int32_t gmScaleOffset = preSrcExpertSum + blockCoord.m();
        layout::VectorLayout scaleLauout{actualBlockShape.m()};
        if (source_scale_offset[event_id] != gmScaleOffset) {
                source_scale_offset[event_id] = gmScaleOffset;
                copyScaleGmToUb(scaleUb, gmPerTokenScale[gmScaleOffset],  scaleLauout, scaleLauout);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_2);




        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_2); // 注意必须是MTE2_S，不能是MTE2_V，否则会读到0，造成乱码
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < actualBlockShape.m(); ++row) {
                float scale = scaleUb(row);
                Muls<float, false>(ubCFp32[n0* row], ubCFp32[n0 * row] , scale, -1, (actualBlockShape.n() + 127) / 128 * 2, {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
        AscendC::Cast<ElementD, float, false>(ubD, ubCFp32, AscendC::RoundMode::CAST_RINT, -1, repeat, {1, 1, 4, 8});
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id); //for debug
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx ++) {
            int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * 16);
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
            MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset + mPreSumBeforeRank[dstEpIdx], blockCoord.n()};
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            auto gmTileD = gmRemotePeer[gmDstOffset];
            LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
            LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
            copyUbToGmD(gmTileD, ubD[tileOffset *  n0], layoutGM2, layoutUB2);
            tileOffset += lenData;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);

    }

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        GemmCoord& blockCoord,
        GemmCoord& actualBlockShape,
        int32_t groupIdx,
        int32_t preSrcExpertSum,
        AscendC::GlobalTensor<int32_t> preSumBeforeRank,
        uint32_t *mPreSumBeforeRank
    ){
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;

        auto &ubC = ubCList[is_ping];
        auto &ubD = ubDList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};


        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id); //for debug
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id); //for debug

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id); //for debug
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx ++) {
            int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * 16);
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
            MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset + mPreSumBeforeRank[dstEpIdx], blockCoord.n()};
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            auto gmTileD = gmRemotePeer[gmDstOffset];
            LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
            LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
            copyUbToGmD(gmTileD, ubC[tileOffset *  n0], layoutGM2, layoutUB2);
            tileOffset += lenData;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

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
#endif