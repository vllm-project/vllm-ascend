#ifndef CHUNK_KDA_FWD_ARCH35_FWD_H_H
#define CHUNK_KDA_FWD_ARCH35_FWD_H_H

#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"

namespace KdaForward::arch35 {

using namespace AscendC;

// The direct UB is shared by one AIC and its two AIV subblocks. Mode 4
// addresses each subblock explicitly: the second AIV uses flag + 16.
constexpr uint64_t KDA_FWD_H_DIRECT_FREE_FLAG = 6;
constexpr uint64_t KDA_FWD_H_DIRECT_READY_FLAG = 7;
constexpr uint64_t KDA_FWD_H_SUBBLOCK_FLAG_OFFSET = 16;
// Prepare is fully drained before FwdH starts. Keep FwdH mode-2 L1
// traffic outside Matmul's possible 0..7 range and SyncAll's 11..14 range.
constexpr uint64_t KDA_FWD_H_L1_FREE_FLAG = 9;
constexpr uint64_t KDA_FWD_H_L1_READY_FLAG = 9;
constexpr uint64_t KDA_FWD_H_STATE_FREE_FLAG = KDA_FWD_H_L1_FREE_FLAG;
constexpr uint64_t KDA_FWD_H_STATE_READY_FLAG = KDA_FWD_H_L1_READY_FLAG;
constexpr uint64_t KDA_FWD_H_VNEW_FREE_FLAG = 10;
constexpr uint64_t KDA_FWD_H_VNEW_READY_FLAG = 10;

constexpr TEventID KDA_FWD_H_MTE_W_EVENT = 0;
constexpr TEventID KDA_FWD_H_MTE_Q_EVENT = 1;
constexpr TEventID KDA_FWD_H_MTE_B_EVENT = 2;
constexpr TEventID KDA_FWD_H_MTE_A_EVENT = 3;
constexpr TEventID KDA_FWD_H_M_EVENT = 4;
constexpr TEventID KDA_FWD_H_FIX_EVENT = 5;
constexpr TEventID KDA_FWD_H_IO_REUSE_EVENT = 6;

constexpr uint32_t KDA_FWD_H_CHUNK = 64;
constexpr uint32_t KDA_FWD_H_DIM = 128;
constexpr uint32_t KDA_FWD_H_SUB_CHUNK = KDA_FWD_H_CHUNK / 2;
constexpr uint32_t KDA_FWD_H_SUB_DIM = KDA_FWD_H_DIM / 2;
constexpr uint32_t KDA_FWD_H_STATE_SUB_ELEMS = KDA_FWD_H_SUB_DIM * KDA_FWD_H_DIM;
constexpr uint32_t KDA_FWD_H_TOKEN_SUB_ELEMS = KDA_FWD_H_SUB_CHUNK * KDA_FWD_H_DIM;

constexpr uint32_t KDA_FWD_H_L1_W_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_L1_Q_OFFSET = 16 * 1024;
constexpr uint32_t KDA_FWD_H_L1_H_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG_OFFSET = 64 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK_OFFSET = 80 * 1024;
constexpr uint32_t KDA_FWD_H_L1_V_OFFSET = 96 * 1024;
constexpr uint32_t KDA_FWD_H_L1_W1_OFFSET = 112 * 1024;
constexpr uint32_t KDA_FWD_H_L1_Q1_OFFSET = 128 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG1_OFFSET = 144 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK1_OFFSET = 160 * 1024;
constexpr uint32_t KDA_FWD_H_L1_W2_OFFSET = 176 * 1024;
constexpr uint32_t KDA_FWD_H_L1_Q2_OFFSET = 192 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG2_OFFSET = 208 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK2_OFFSET = 224 * 1024;
constexpr uint32_t KDA_FWD_H_L1_W3_OFFSET = 240 * 1024;
constexpr uint32_t KDA_FWD_H_L1_Q3_OFFSET = 256 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG3_OFFSET = 272 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK3_OFFSET = 288 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AKK_OFFSET = 304 * 1024;
constexpr uint32_t KDA_FWD_H_L1_U_OFFSET = 336 * 1024;
constexpr uint32_t KDA_FWD_H_L1_STAGING_DEPTH = 4;
constexpr uint32_t KDA_FWD_H_L1_AKK_SLOT_BYTES = 8 * 1024;
constexpr uint32_t KDA_FWD_H_L1_U_SLOT_BYTES = 16 * 1024;

constexpr uint32_t KDA_FWD_H_L0A_STATE_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_L0A_VNEW_OFFSET = 16 * 1024;
constexpr uint32_t KDA_FWD_H_L0A_POST_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_L0B_STATE_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_L0B_VNEW_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_L0B_POST_OFFSET = 32 * 1024;

constexpr uint32_t KDA_FWD_H_UB_STATE_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_UB_STATE_TYPED_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_UB_DIRECT_OFFSET = 48 * 1024;
constexpr uint32_t KDA_FWD_H_UB_OUT1_OFFSET = 80 * 1024;
constexpr uint32_t KDA_FWD_H_UB_VNEW_OFFSET = 96 * 1024;
constexpr uint32_t KDA_FWD_H_UB_IO_OFFSET = 112 * 1024;
constexpr uint32_t KDA_FWD_H_UB_GATE_OFFSET = 128 * 1024;

template <typename T, typename GK_T, typename TilingData>
class ChunkKdaFwdFwdH {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using TileCopyRM = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, LayoutRM, T, LayoutRM, float, LayoutRM>;
    using DirectTileCopyRM = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, T, LayoutRM, T, LayoutRM, float, LayoutRM, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileCopyCM = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, LayoutCM, T, LayoutRM, float, LayoutRM>;
    using DirectTileCopyCM = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, T, LayoutCM, T, LayoutRM, float, LayoutRM, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;

    __aicore__ inline void Init(
        GM_ADDR gk, GM_ADDR initialState, GM_ADDR attnOut, GM_ADDR finalState,
        GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qgScaled, GM_ADDR kg,
        GM_ADDR vNew, GM_ADDR h, const TilingData &tiling)
    {
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ GK_T *>(gk));
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initialState));
        }
        attnOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(attnOut));
        finalState_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(finalState));
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aqk));
        akk_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(akk));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(w));
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(u));
        qgScaled_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(qgScaled));
        kg_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kg));
        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(vNew));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(h));
        batch_ = tiling.batch;
        heads_ = tiling.vHeadNum;
        seqlen_ = tiling.seqlen;
        totalChunks_ = tiling.totalChunks;
        hasInitialState_ = tiling.hasInitialState;
        storeFinalState_ = tiling.storeFinalState;
        storeVNew_ = tiling.storeVNew;
        storeH_ = tiling.storeH;
        fusePostWuIntoFwdH_ = tiling.fusePostWuIntoFwdH;
        coreNum_ = tiling.prepareUsedCoreNum;
        statePublishCount_[0] = 0;
        statePublishCount_[1] = 0;
        vnewPublishCount_[0] = 0;
        vnewPublishCount_[1] = 0;
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessAic();
        }
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
    }

private:
    __aicore__ inline void WaitL1SlotFreeMte3(
        uint64_t freeFlag, uint32_t publishCount)
    {
        if (publishCount != 0) {
            CrossCoreWaitFlag(freeFlag);
        }
    }

    template <pipe_t PIPE>
    __aicore__ inline void SetL1SlotFlagAicToAiv(uint64_t flag)
    {
        CrossCoreSetFlag<0x2, PIPE>(flag);
    }

    template <pipe_t PIPE>
    __aicore__ inline void SetL1SlotFlagAivToAic(uint64_t flag)
    {
        CrossCoreSetFlag<0x2, PIPE>(flag);
    }

    __aicore__ inline void WaitL1SlotReadyMte1(uint64_t readyFlag)
    {
        CrossCoreWaitFlag(readyFlag);
    }

    __aicore__ inline void WaitDirectFreeAic()
    {
        CrossCoreWaitFlag<0x4, PIPE_FIX>(KDA_FWD_H_DIRECT_FREE_FLAG);
        CrossCoreWaitFlag<0x4, PIPE_FIX>(
            KDA_FWD_H_DIRECT_FREE_FLAG + KDA_FWD_H_SUBBLOCK_FLAG_OFFSET);
    }

    __aicore__ inline void SetDirectReadyAic()
    {
        CrossCoreSetFlag<0x4, PIPE_FIX>(KDA_FWD_H_DIRECT_READY_FLAG);
        CrossCoreSetFlag<0x4, PIPE_FIX>(
            KDA_FWD_H_DIRECT_READY_FLAG + KDA_FWD_H_SUBBLOCK_FLAG_OFFSET);
    }

    __aicore__ inline void WaitDirectReadyAiv()
    {
        CrossCoreWaitFlag<0x4, PIPE_V>(KDA_FWD_H_DIRECT_READY_FLAG);
    }

    __aicore__ inline void SetDirectFreeAiv()
    {
        CrossCoreSetFlag<0x4, PIPE_V>(KDA_FWD_H_DIRECT_FREE_FLAG);
    }

    __aicore__ inline uint64_t MatrixOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return ((b * heads_ + hv) * seqlen_ + t) * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint64_t ChunkMatrixOffset(
        uint64_t b, uint64_t hv, uint64_t chunk, uint64_t row = 0) const
    {
        return ((b * heads_ + hv) * seqlen_ +
                chunk * KDA_FWD_H_CHUNK + row) * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint64_t ScoreOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return ((b * heads_ + hv) * seqlen_ + t) * KDA_FWD_H_CHUNK;
    }

    __aicore__ inline uint64_t ChunkScoreOffset(
        uint64_t b, uint64_t hv, uint64_t chunk) const
    {
        return ((b * heads_ + hv) * seqlen_ +
                chunk * KDA_FWD_H_CHUNK) * KDA_FWD_H_CHUNK;
    }

    __aicore__ inline uint64_t StateOffset(uint64_t b, uint64_t hv) const
    {
        return (b * heads_ + hv) * KDA_FWD_H_DIM * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint64_t HOffset(uint64_t b, uint64_t hv, uint64_t chunk) const
    {
        if (!storeH_) {
            return StateOffset(b, hv);
        }
        return ((b * heads_ + hv) * totalChunks_ + chunk) *
               KDA_FWD_H_DIM * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint64_t VNewOffset(
        uint64_t b, uint64_t hv, uint64_t chunk, uint64_t row = 0) const
    {
        if (!storeVNew_) {
            return ((b * heads_ + hv) * KDA_FWD_H_CHUNK + row) * KDA_FWD_H_DIM;
        }
        return ChunkMatrixOffset(b, hv, chunk, row);
    }

    __aicore__ inline uint64_t OutputOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return ((b * seqlen_ + t) * heads_ + hv) * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint32_t L1WOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_W_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_W1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_W2_OFFSET : KDA_FWD_H_L1_W3_OFFSET;
    }

    __aicore__ inline uint32_t L1QOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_Q_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_Q1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_Q2_OFFSET : KDA_FWD_H_L1_Q3_OFFSET;
    }

    __aicore__ inline uint32_t L1KgOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_KG_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_KG1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_KG2_OFFSET : KDA_FWD_H_L1_KG3_OFFSET;
    }

    __aicore__ inline uint32_t L1AqkOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_AQK_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_AQK1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_AQK2_OFFSET : KDA_FWD_H_L1_AQK3_OFFSET;
    }

    __aicore__ inline uint32_t L1AkkOffset(uint32_t slot) const
    {
        return KDA_FWD_H_L1_AKK_OFFSET + slot * KDA_FWD_H_L1_AKK_SLOT_BYTES;
    }

    __aicore__ inline uint32_t L1UOffset(uint32_t slot) const
    {
        return KDA_FWD_H_L1_U_OFFSET + slot * KDA_FWD_H_L1_U_SLOT_BYTES;
    }

    template <typename DirectTileCopy, typename TensorL0C>
    __aicore__ inline void PublishDirectTile(
        TensorL0C tensorL0C, uint32_t m, uint32_t n,
        TEventID mToFixEvent, TEventID fixToMEvent)
    {
        auto layoutUb = tla::MakeLayout<float, LayoutRM>(m, n);
        auto tensorUb = tla::MakeTensor(
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_DIRECT_OFFSET),
            layoutUb, Catlass::Arch::PositionUB{});
        using CopyL0CToDst =
            typename DirectTileCopy::template CopyL0CToDst<decltype(tensorUb)>;
        CopyL0CToDst copyL0CToDst;

        WaitDirectFreeAic();
        SetFlag<HardEvent::M_FIX>(mToFixEvent);
        WaitFlag<HardEvent::M_FIX>(mToFixEvent);
        copyL0CToDst(tensorUb, tensorL0C);
        SetDirectReadyAic();
        SetFlag<HardEvent::FIX_M>(fixToMEvent);
    }

    template <typename DirectTileCopy>
    __aicore__ inline void PublishDirect(
        LocalTensor<float> l0C, uint32_t m, uint32_t n,
        TEventID mToFixEvent, TEventID fixToMEvent)
    {
        auto layoutL0C = tla::MakeLayoutL0C(m, n);
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        PublishDirectTile<DirectTileCopy>(
            tensorL0C, m, n, mToFixEvent, fixToMEvent);
    }

    __aicore__ inline void PrefetchIndependentProductsAic(
        uint64_t b, uint64_t hv, uint64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 3);
        WaitFlag<HardEvent::MTE1_MTE2>(aicL1ReuseEvents_[slot]);
        using LayoutTagL1ARm = typename TileCopyRM::LayoutTagL1A;
        using LayoutTagL1BRm = typename TileCopyRM::LayoutTagL1B;
        using LayoutTagL1ACm = typename TileCopyCM::LayoutTagL1A;

        auto layoutToken = tla::MakeLayout<T, LayoutRM>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutKg = tla::MakeLayout<T, LayoutCM>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK);
        auto layoutAqk = tla::MakeLayout<T, LayoutRM>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK);
        auto layoutAkk = tla::MakeLayout<T, LayoutRM>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK);
        auto tensorW = tla::MakeTensor(
            w_[ChunkMatrixOffset(b, hv, chunk)], layoutToken, Catlass::Arch::PositionGM{});
        auto tensorQ = tla::MakeTensor(
            qgScaled_[ChunkMatrixOffset(b, hv, chunk)], layoutToken,
            Catlass::Arch::PositionGM{});
        auto tensorKg = tla::MakeTensor(
            kg_[ChunkMatrixOffset(b, hv, chunk)], layoutKg, Catlass::Arch::PositionGM{});
        auto tensorAqk = tla::MakeTensor(
            aqk_[ChunkScoreOffset(b, hv, chunk)], layoutAqk, Catlass::Arch::PositionGM{});
        auto tensorAkk = tla::MakeTensor(
            akk_[ChunkScoreOffset(b, hv, chunk)], layoutAkk, Catlass::Arch::PositionGM{});
        auto tensorU = tla::MakeTensor(
            u_[ChunkMatrixOffset(b, hv, chunk)], layoutToken, Catlass::Arch::PositionGM{});
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0),
                              tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                              tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto blockKg = GetTile(tensorKg, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0),
                                tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(0, 0),
                                tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK));
        auto blockU = GetTile(tensorU, tla::MakeCoord(0, 0),
                              tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        using CopyGmToL1ARmW = typename TileCopyRM::template CopyGmToL1A<decltype(blockW)>;
        using CopyGmToL1ARmQ = typename TileCopyRM::template CopyGmToL1A<decltype(blockQ)>;
        using CopyGmToL1ACm = typename TileCopyCM::template CopyGmToL1A<decltype(blockKg)>;
        using CopyGmToL1ARmAqk = typename TileCopyRM::template CopyGmToL1A<decltype(blockAqk)>;
        using CopyGmToL1ARmAkk = typename TileCopyRM::template CopyGmToL1A<decltype(blockAkk)>;
        using CopyGmToL1BRmU = typename TileCopyRM::template CopyGmToL1B<decltype(blockU)>;

        LocalTensor<T> l1W = resource_.l1Buf.template GetBufferByByte<T>(
            L1WOffset(slot));
        LocalTensor<T> l1Q = resource_.l1Buf.template GetBufferByByte<T>(
            L1QOffset(slot));
        LocalTensor<T> l1Kg = resource_.l1Buf.template GetBufferByByte<T>(
            L1KgOffset(slot));
        LocalTensor<T> l1Aqk = resource_.l1Buf.template GetBufferByByte<T>(
            L1AqkOffset(slot));
        LocalTensor<T> l1Akk = resource_.l1Buf.template GetBufferByByte<T>(
            L1AkkOffset(slot));
        LocalTensor<T> l1U = resource_.l1Buf.template GetBufferByByte<T>(
            L1UOffset(slot));
        auto tensorL1W = tla::MakeTensor(
            l1W, tla::MakeLayout<T, LayoutTagL1ARm>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL1{});
        auto tensorL1Q = tla::MakeTensor(
            l1Q, tla::MakeLayout<T, LayoutTagL1ARm>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL1{});
        auto tensorL1Kg = tla::MakeTensor(
            l1Kg, tla::MakeLayout<T, LayoutTagL1ACm>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK),
            Catlass::Arch::PositionL1{});
        auto tensorL1Aqk = tla::MakeTensor(
            l1Aqk, tla::MakeLayout<T, LayoutTagL1ARm>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK),
            Catlass::Arch::PositionL1{});
        auto tensorL1Akk = tla::MakeTensor(
            l1Akk, tla::MakeLayout<T, LayoutTagL1ARm>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK),
            Catlass::Arch::PositionL1{});
        auto tensorL1U = tla::MakeTensor(
            l1U, tla::MakeLayout<T, LayoutTagL1BRm>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL1{});

        CopyGmToL1ARmW{}(tensorL1W, blockW);
        CopyGmToL1ARmQ{}(tensorL1Q, blockQ);
        CopyGmToL1ACm{}(tensorL1Kg, blockKg);
        CopyGmToL1ARmAqk{}(tensorL1Aqk, blockAqk);
        if (fusePostWuIntoFwdH_) {
            CopyGmToL1ARmAkk{}(tensorL1Akk, blockAkk);
            CopyGmToL1BRmU{}(tensorL1U, blockU);
        }
        SetFlag<HardEvent::MTE2_MTE1>(aicMte2ToMte1Event_);
    }

    __aicore__ inline void ComputePostWuAic(uint64_t chunk)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 3);
        using LayoutTagL1A = typename TileCopyRM::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyRM::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyRM::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyRM::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyRM::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyRM::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, T, LayoutTagL1A>;

        LocalTensor<T> l1Akk = resource_.l1Buf.template GetBufferByByte<T>(
            L1AkkOffset(slot));
        LocalTensor<T> l1W = resource_.l1Buf.template GetBufferByByte<T>(
            L1WOffset(slot));
        LocalTensor<T> l1U = resource_.l1Buf.template GetBufferByByte<T>(
            L1UOffset(slot));
        LocalTensor<T> l0A = resource_.l0ABuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0A_POST_OFFSET);
        LocalTensor<T> l0B = resource_.l0BBuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0B_POST_OFFSET);
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

        auto tensorL1Akk = tla::MakeTensor(
            l1Akk, tla::MakeLayout<T, LayoutTagL1A>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK),
            Catlass::Arch::PositionL1{});
        auto tensorL1W = tla::MakeTensor(
            l1W, tla::MakeLayout<T, LayoutTagL1B>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL1{});
        auto tensorL1U = tla::MakeTensor(
            l1U, tla::MakeLayout<T, LayoutTagL1B>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<T, LayoutTagL0A>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<T, LayoutTagL0B>(KDA_FWD_H_CHUNK, 2 * KDA_FWD_H_DIM),
            Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(KDA_FWD_H_CHUNK, 2 * KDA_FWD_H_DIM),
            Catlass::Arch::PositionL0C{});
        auto tileL1Akk = GetTile(tensorL1Akk, tla::MakeCoord(0, 0),
                                 tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK));
        auto tileL1W = GetTile(tensorL1W, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL1U = GetTile(tensorL1U, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK));
        auto tileL0BW = GetTile(tensorL0B, tla::MakeCoord(0, 0),
                                tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL0BU = GetTile(tensorL0B, tla::MakeCoord(0, KDA_FWD_H_DIM),
                                tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, 2 * KDA_FWD_H_DIM));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, 2 * KDA_FWD_H_DIM));
        auto tileL0CW = GetTile(tensorL0C, tla::MakeCoord(0, 0),
                                tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL0CU = GetTile(tensorL0C, tla::MakeCoord(0, KDA_FWD_H_DIM),
                                tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));

        WaitFlag<HardEvent::MTE2_MTE1>(aicMte2ToMte1Event_);
        WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        CopyL1ToL0A{}(tileL0A, tileL1Akk);
        CopyL1ToL0B{}(tileL0BW, tileL1W);
        CopyL1ToL0B{}(tileL0BU, tileL1U);
        SetFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        WaitFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        TileMmad{}(tileL0C, tileL0A, tileL0B, KDA_FWD_H_CHUNK,
                   2 * KDA_FWD_H_DIM, KDA_FWD_H_CHUNK, true, 0);
        SetFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        PublishDirectTile<DirectTileCopyRM>(
            tileL0CW, KDA_FWD_H_CHUNK, KDA_FWD_H_DIM,
            aicMToFixEvent_, aicFixToMEvent_);
        WaitFlag<HardEvent::FIX_M>(aicFixToMEvent_);
        PublishDirectTile<DirectTileCopyRM>(
            tileL0CU, KDA_FWD_H_CHUNK, KDA_FWD_H_DIM,
            aicMToFixEvent_, aicFixToMEvent_);
        WaitFlag<HardEvent::FIX_M>(aicFixToMEvent_);
    }

    __aicore__ inline void ComputeStateProductsAic(
        uint64_t b, uint64_t hv, uint64_t chunk, bool inputsReady = false)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 3);
        using LayoutTagL1A = typename TileCopyRM::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyRM::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyRM::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyRM::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyRM::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyRM::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, T, LayoutTagL1A>;

        LocalTensor<T> l1A0 = resource_.l1Buf.template GetBufferByByte<T>(
            L1WOffset(slot));
        LocalTensor<T> l1A1 = resource_.l1Buf.template GetBufferByByte<T>(
            L1QOffset(slot));
        LocalTensor<T> l1B =
            resource_.l1Buf.template GetBufferByByte<T>(KDA_FWD_H_L1_H_OFFSET);
        LocalTensor<T> l0A = resource_.l0ABuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0A_STATE_OFFSET);
        LocalTensor<T> l0B = resource_.l0BBuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0B_STATE_OFFSET);
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

        auto layoutL1A = tla::MakeLayout<T, LayoutTagL1A>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(KDA_FWD_H_DIM, KDA_FWD_H_DIM);
        auto layoutL0A = tla::MakeLayout<T, LayoutTagL0A>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutL0B = tla::MakeLayout<T, LayoutTagL0B>(KDA_FWD_H_DIM, KDA_FWD_H_DIM);
        auto layoutL0C = tla::MakeLayoutL0C(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto tensorL1A0 = tla::MakeTensor(l1A0, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(l1A1, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_DIM));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL1W = GetTile(tensorL1A0, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL1Q = GetTile(tensorL1A1, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_DIM));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        if (!inputsReady) {
            WaitFlag<HardEvent::MTE2_MTE1>(aicMte2ToMte1Event_);
        }
        WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        copyL1ToL0A(tileL0A, tileL1W);
        copyL1ToL0B(tileL0B, tileL1B);
        SetFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        WaitFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        tileMmad(tileL0C, tileL0A, tileL0B, KDA_FWD_H_CHUNK,
                 KDA_FWD_H_DIM, KDA_FWD_H_DIM, true, 0);
        SetFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        PublishDirect<DirectTileCopyRM>(
            l0C, KDA_FWD_H_CHUNK, KDA_FWD_H_DIM,
            aicMToFixEvent_, aicFixToMEvent_);
        SetL1SlotFlagAicToAiv<PIPE_FIX>(KDA_FWD_H_STATE_FREE_FLAG);

        WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        WaitFlag<HardEvent::FIX_M>(aicFixToMEvent_);
        copyL1ToL0A(tileL0A, tileL1Q);
        SetFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        WaitFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        tileMmad(tileL0C, tileL0A, tileL0B, KDA_FWD_H_CHUNK,
                 KDA_FWD_H_DIM, KDA_FWD_H_DIM, true, 0);
        SetFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        PublishDirect<DirectTileCopyRM>(
            l0C, KDA_FWD_H_CHUNK, KDA_FWD_H_DIM,
            aicMToFixEvent_, aicFixToMEvent_);
        WaitFlag<HardEvent::FIX_M>(aicFixToMEvent_);
    }

    __aicore__ inline void ComputeVnewProductsAic(
        uint64_t b, uint64_t hv, uint64_t chunk, bool prefetchNext)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 3);
        using LayoutTagL1AK = typename TileCopyCM::LayoutTagL1A;
        using LayoutTagL1AA = typename TileCopyRM::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyRM::LayoutTagL1B;
        using LayoutTagL0AK = typename TileCopyCM::LayoutTagL0A;
        using LayoutTagL0AA = typename TileCopyRM::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyRM::LayoutTagL0B;
        using CopyL1ToL0AK = typename TileCopyCM::CopyL1ToL0A;
        using CopyL1ToL0AA = typename TileCopyRM::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyRM::CopyL1ToL0B;
        using TileMmadK = Catlass::Gemm::Tile::TileMmadTla<ArchTag, T, LayoutTagL1AK>;
        using TileMmadA = Catlass::Gemm::Tile::TileMmadTla<ArchTag, T, LayoutTagL1AA>;

        auto layoutKg = tla::MakeLayout<T, LayoutCM>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK);
        auto layoutAqk = tla::MakeLayout<T, LayoutRM>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK);
        LocalTensor<T> l1Kg = resource_.l1Buf.template GetBufferByByte<T>(
            L1KgOffset(slot));
        LocalTensor<T> l1Aqk = resource_.l1Buf.template GetBufferByByte<T>(
            L1AqkOffset(slot));
        LocalTensor<T> l1V =
            resource_.l1Buf.template GetBufferByByte<T>(KDA_FWD_H_L1_V_OFFSET);
        LocalTensor<T> l0A = resource_.l0ABuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0A_VNEW_OFFSET);
        LocalTensor<T> l0B = resource_.l0BBuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0B_VNEW_OFFSET);
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

        auto layoutL1Kg = tla::MakeLayout<T, LayoutTagL1AK>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK);
        auto layoutL1Aqk = tla::MakeLayout<T, LayoutTagL1AA>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK);
        auto layoutL1V = tla::MakeLayout<T, LayoutTagL1B>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutL0Kg = tla::MakeLayout<T, LayoutTagL0AK>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK);
        auto layoutL0Aqk = tla::MakeLayout<T, LayoutTagL0AA>(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK);
        auto layoutL0V = tla::MakeLayout<T, LayoutTagL0B>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto baseL1Kg = tla::MakeTensor(l1Kg, layoutL1Kg, Catlass::Arch::PositionL1{});
        auto baseL1Aqk = tla::MakeTensor(l1Aqk, layoutL1Aqk, Catlass::Arch::PositionL1{});
        auto baseL1V = tla::MakeTensor(l1V, layoutL1V, Catlass::Arch::PositionL1{});
        auto baseL0Kg = tla::MakeTensor(l0A, layoutL0Kg, Catlass::Arch::PositionL0A{});
        auto baseL0Aqk = tla::MakeTensor(l0A, layoutL0Aqk, Catlass::Arch::PositionL0A{});
        auto baseL0V = tla::MakeTensor(l0B, layoutL0V, Catlass::Arch::PositionL0B{});
        auto baseL0Update = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(KDA_FWD_H_DIM, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL0C{});
        auto baseL0Out = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL0C{});
        auto tensorL1Kg = GetTile(baseL1Kg, tla::MakeCoord(0, 0),
                                  tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK));
        auto tensorL1Aqk = GetTile(baseL1Aqk, tla::MakeCoord(0, 0),
                                   tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK));
        auto tensorL1V = GetTile(baseL1V, tla::MakeCoord(0, 0),
                                 tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tensorL0Kg = GetTile(baseL0Kg, tla::MakeCoord(0, 0),
                                  tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK));
        auto tensorL0Aqk = GetTile(baseL0Aqk, tla::MakeCoord(0, 0),
                                   tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_CHUNK));
        auto tensorL0V = GetTile(baseL0V, tla::MakeCoord(0, 0),
                                 tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tensorL0Update = GetTile(baseL0Update, tla::MakeCoord(0, 0),
                                      tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_DIM));
        auto tensorL0Out = GetTile(baseL0Out, tla::MakeCoord(0, 0),
                                   tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));

        CopyL1ToL0AK copyL1ToL0AK;
        CopyL1ToL0AA copyL1ToL0AA;
        CopyL1ToL0B copyL1ToL0B;
        TileMmadK tileMmadK;
        TileMmadA tileMmadA;

        WaitFlag<HardEvent::M_MTE1>(vnewL0FreeEvent_);
        copyL1ToL0AK(tensorL0Kg, tensorL1Kg);
        copyL1ToL0B(tensorL0V, tensorL1V);
        SetFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        WaitFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        tileMmadK(tensorL0Update, tensorL0Kg, tensorL0V,
                  KDA_FWD_H_DIM, KDA_FWD_H_DIM, KDA_FWD_H_CHUNK, true, 0);
        SetFlag<HardEvent::M_MTE1>(vnewL0FreeEvent_);
        PublishDirect<DirectTileCopyCM>(
            l0C, KDA_FWD_H_DIM, KDA_FWD_H_DIM,
            aicMToFixEvent_, aicFixToMEvent_);
        SetL1SlotFlagAicToAiv<PIPE_FIX>(KDA_FWD_H_VNEW_FREE_FLAG);

        WaitFlag<HardEvent::M_MTE1>(vnewL0FreeEvent_);
        WaitFlag<HardEvent::FIX_M>(aicFixToMEvent_);
        copyL1ToL0AA(tensorL0Aqk, tensorL1Aqk);
        SetFlag<HardEvent::MTE1_MTE2>(aicL1ReuseEvents_[slot]);
        if (prefetchNext) {
            PrefetchIndependentProductsAic(b, hv, chunk + 1);
        }
        SetFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        WaitFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        tileMmadA(tensorL0Out, tensorL0Aqk, tensorL0V,
                  KDA_FWD_H_CHUNK, KDA_FWD_H_DIM, KDA_FWD_H_CHUNK, true, 0);
        SetFlag<HardEvent::M_MTE1>(vnewL0FreeEvent_);
        PublishDirect<DirectTileCopyRM>(
            l0C, KDA_FWD_H_CHUNK, KDA_FWD_H_DIM,
            aicMToFixEvent_, aicFixToMEvent_);
        WaitFlag<HardEvent::FIX_M>(aicFixToMEvent_);
    }

    __aicore__ inline void ProcessAic()
    {
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        SetFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        for (uint32_t slot = 0; slot < KDA_FWD_H_L1_STAGING_DEPTH; ++slot) {
            SetFlag<HardEvent::MTE1_MTE2>(aicL1ReuseEvents_[slot]);
        }
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t coreNum = coreNum_ == 0 ? 1 : coreNum_;
        for (uint64_t task = coreIdx; task < batch_ * heads_; task += coreNum) {
            const uint64_t b = task / heads_;
            const uint64_t hv = task % heads_;
            PrefetchIndependentProductsAic(b, hv, 0);
            for (uint64_t chunk = 0; chunk < totalChunks_; ++chunk) {
                if (fusePostWuIntoFwdH_) {
                    ComputePostWuAic(chunk);
                }
                WaitL1SlotReadyMte1(KDA_FWD_H_STATE_READY_FLAG);
                ComputeStateProductsAic(b, hv, chunk, fusePostWuIntoFwdH_);
                WaitL1SlotReadyMte1(KDA_FWD_H_VNEW_READY_FLAG);
                ComputeVnewProductsAic(
                    b, hv, chunk, chunk + 1 < totalChunks_);
            }
            WaitDirectFreeAic();
        }
        WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        for (uint32_t slot = 0; slot < KDA_FWD_H_L1_STAGING_DEPTH; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(aicL1ReuseEvents_[slot]);
        }
    }

    __aicore__ inline void CopyOutputRows(
        uint64_t b, uint64_t hv, uint64_t tokenStart,
        LocalTensor<T> src, uint32_t rows)
    {
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(KDA_FWD_H_DIM * sizeof(T)),
            0,
            static_cast<uint32_t>((heads_ * KDA_FWD_H_DIM - KDA_FWD_H_DIM) * sizeof(T)),
            0};
        DataCopyPad(attnOut_[OutputOffset(b, hv, tokenStart)], src, params);
    }

    __aicore__ inline void InitializeStateAiv(
        uint64_t b, uint64_t hv, uint32_t rowBegin,
        LocalTensor<float> state)
    {
        if (hasInitialState_) {
            DataCopy(state, initialState_[StateOffset(b, hv) + rowBegin * KDA_FWD_H_DIM],
                     KDA_FWD_H_STATE_SUB_ELEMS);
            SetFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
        } else {
            Duplicate(state, 0.0f, KDA_FWD_H_STATE_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreCurrentStateAiv(
        uint64_t b, uint64_t hv, uint64_t chunk, uint32_t rowBegin,
        LocalTensor<float> state, LocalTensor<T> stateTyped)
    {
        if (storeH_) {
            Cast(stateTyped, state, RoundMode::CAST_RINT, KDA_FWD_H_STATE_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            DataCopy(h_[HOffset(b, hv, chunk) + rowBegin * KDA_FWD_H_DIM],
                     stateTyped, KDA_FWD_H_STATE_SUB_ELEMS);
            SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        }

        constexpr uint32_t columnGroup = 64;
        constexpr uint32_t columnGroups = KDA_FWD_H_DIM / columnGroup;
        for (uint32_t group = 0; group < columnGroups; ++group) {
            Cast(stateTyped[group * KDA_FWD_H_SUB_DIM * columnGroup],
                 state[group * columnGroup], RoundMode::CAST_RINT,
                 columnGroup, KDA_FWD_H_SUB_DIM,
                 {static_cast<uint16_t>(KDA_FWD_H_SUB_DIM), 1, 1,
                  static_cast<uint8_t>(columnGroups * 8)});
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        LocalTensor<T> l1State =
            resource_.l1Buf.template GetBufferByByte<T>(KDA_FWD_H_L1_H_OFFSET);
        const uint32_t subBlockIdx = rowBegin / KDA_FWD_H_SUB_DIM;
        WaitL1SlotFreeMte3(
            KDA_FWD_H_STATE_FREE_FLAG, statePublishCount_[subBlockIdx]);
        DataCopyParams stateCopyParams;
        stateCopyParams.blockCount = KDA_FWD_H_DIM / 16;
        stateCopyParams.blockLen = KDA_FWD_H_SUB_DIM;
        stateCopyParams.srcGap = 0;
        stateCopyParams.dstGap = KDA_FWD_H_DIM - KDA_FWD_H_SUB_DIM;
        DataCopy(l1State[rowBegin * 16], stateTyped, stateCopyParams);
        SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        if (!fusePostWuIntoFwdH_) {
            SetL1SlotFlagAivToAic<PIPE_MTE3>(KDA_FWD_H_STATE_READY_FLAG);
        }
        ++statePublishCount_[subBlockIdx];
    }

    __aicore__ inline void ProcessChunkAiv(
        uint64_t b, uint64_t hv, uint32_t chunk,
        uint32_t subBlockIdx, LocalTensor<float> state,
        LocalTensor<T> stateTyped, LocalTensor<float> direct,
        LocalTensor<float> out1, LocalTensor<float> vnew,
        LocalTensor<T> ioTyped, LocalTensor<float> gate)
    {
        const uint32_t tokenBegin = subBlockIdx * KDA_FWD_H_SUB_CHUNK;
        const uint32_t stateRowBegin = subBlockIdx * KDA_FWD_H_SUB_DIM;
        StoreCurrentStateAiv(b, hv, chunk, stateRowBegin, state, stateTyped);

        WaitDirectReadyAiv();
        WaitFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
        if (fusePostWuIntoFwdH_) {
            Cast(ioTyped, direct, RoundMode::CAST_RINT, KDA_FWD_H_TOKEN_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
            SetDirectFreeAiv();

            SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            LocalTensor<T> l1W = resource_.l1Buf.template GetBufferByByte<T>(
                L1WOffset(chunk & 3));
            DataCopyParams wCopyParams;
            wCopyParams.blockCount = KDA_FWD_H_SUB_CHUNK;
            wCopyParams.blockLen = 1;
            wCopyParams.srcGap = KDA_FWD_H_DIM / 16 - 1;
            wCopyParams.dstGap = 0;
            for (uint32_t colBlock = 0; colBlock < KDA_FWD_H_DIM / 16; ++colBlock) {
                const uint32_t dstOffset =
                    colBlock * KDA_FWD_H_CHUNK * 16 + tokenBegin * 16;
                DataCopy(l1W[dstOffset], ioTyped[colBlock * 16], wCopyParams);
            }
            SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
            SetL1SlotFlagAivToAic<PIPE_MTE3>(KDA_FWD_H_STATE_READY_FLAG);

            WaitDirectReadyAiv();
            Cast(ioTyped, direct, RoundMode::CAST_RINT, KDA_FWD_H_TOKEN_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
            Cast(vnew, ioTyped, RoundMode::CAST_NONE, KDA_FWD_H_TOKEN_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
            SetDirectFreeAiv();
            WaitDirectReadyAiv();
        } else {
            DataCopy(ioTyped, u_[ChunkMatrixOffset(b, hv, chunk, tokenBegin)],
                     KDA_FWD_H_TOKEN_SUB_ELEMS);
            SetFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
            Cast(vnew, ioTyped, RoundMode::CAST_NONE, KDA_FWD_H_TOKEN_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
        }
        Sub(vnew, vnew, direct, KDA_FWD_H_TOKEN_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        SetDirectFreeAiv();
        if (storeVNew_) {
            Cast(ioTyped, vnew, RoundMode::CAST_RINT, KDA_FWD_H_TOKEN_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            DataCopy(vNew_[VNewOffset(b, hv, chunk, tokenBegin)],
                     ioTyped, KDA_FWD_H_TOKEN_SUB_ELEMS);
            SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        }

        constexpr uint32_t columnGroup = 64;
        constexpr uint32_t columnGroups = KDA_FWD_H_DIM / columnGroup;
        for (uint32_t group = 0; group < columnGroups; ++group) {
            Cast(ioTyped[group * KDA_FWD_H_SUB_CHUNK * columnGroup],
                 vnew[group * columnGroup], RoundMode::CAST_RINT,
                 columnGroup, KDA_FWD_H_SUB_CHUNK,
                 {static_cast<uint16_t>(KDA_FWD_H_SUB_CHUNK), 1, 1,
                  static_cast<uint8_t>(columnGroups * 8)});
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        LocalTensor<T> l1Vnew =
            resource_.l1Buf.template GetBufferByByte<T>(KDA_FWD_H_L1_V_OFFSET);
        WaitL1SlotFreeMte3(
            KDA_FWD_H_VNEW_FREE_FLAG, vnewPublishCount_[subBlockIdx]);
        DataCopyParams vnewL1CopyParams;
        vnewL1CopyParams.blockCount = KDA_FWD_H_DIM / 16;
        vnewL1CopyParams.blockLen = KDA_FWD_H_SUB_CHUNK;
        vnewL1CopyParams.srcGap = 0;
        vnewL1CopyParams.dstGap = KDA_FWD_H_CHUNK - KDA_FWD_H_SUB_CHUNK;
        DataCopy(l1Vnew[tokenBegin * 16], ioTyped, vnewL1CopyParams);
        SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        SetL1SlotFlagAivToAic<PIPE_MTE3>(KDA_FWD_H_VNEW_READY_FLAG);
        ++vnewPublishCount_[subBlockIdx];

        WaitDirectReadyAiv();
        Adds(out1, direct, 0.0f, KDA_FWD_H_TOKEN_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        SetDirectFreeAiv();

        WaitDirectReadyAiv();
        DataCopy(gate, gk_[ChunkMatrixOffset(b, hv, chunk, KDA_FWD_H_CHUNK - 1) +
                           stateRowBegin], KDA_FWD_H_SUB_DIM);
        SetFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
        Muls(gate, gate, 0.6931471805599453f, KDA_FWD_H_SUB_DIM);
        PipeBarrier<PIPE_V>();
        Exp(gate, gate, KDA_FWD_H_SUB_DIM);
        PipeBarrier<PIPE_V>();
        AscendC::VF_CALL<Catlass::Epilogue::Block::detail::ApplyKGateUpdateRegbaseDualIssue<float>>(
            reinterpret_cast<__ubuf__ float *>(direct.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gate.GetPhyAddr()),
            static_cast<uint16_t>(KDA_FWD_H_SUB_DIM),
            static_cast<uint16_t>(KDA_FWD_H_DIM));
        PipeBarrier<PIPE_V>();
        Adds(state, direct, 0.0f, KDA_FWD_H_STATE_SUB_ELEMS);
        PipeBarrier<PIPE_V>();

        SetDirectFreeAiv();
        WaitDirectReadyAiv();
        Add(direct, direct, out1, KDA_FWD_H_TOKEN_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        Cast(ioTyped, direct, RoundMode::CAST_RINT, KDA_FWD_H_TOKEN_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        CopyOutputRows(
            b, hv, chunk * KDA_FWD_H_CHUNK + tokenBegin,
            ioTyped, KDA_FWD_H_SUB_CHUNK);
        SetDirectFreeAiv();
        SetFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
    }

    __aicore__ inline void ProcessAiv()
    {
        const uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        const uint32_t subBlockNum = static_cast<uint32_t>(GetSubBlockNum());
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        const uint64_t coreNum = coreNum_ == 0 ? 1 : coreNum_;
        LocalTensor<float> state =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_STATE_OFFSET);
        LocalTensor<T> stateTyped =
            resource_.ubBuf.template GetBufferByByte<T>(KDA_FWD_H_UB_STATE_TYPED_OFFSET);
        LocalTensor<float> direct =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_DIRECT_OFFSET);
        LocalTensor<float> out1 =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_OUT1_OFFSET);
        LocalTensor<float> vnew =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_VNEW_OFFSET);
        LocalTensor<T> ioTyped =
            resource_.ubBuf.template GetBufferByByte<T>(KDA_FWD_H_UB_IO_OFFSET);
        LocalTensor<float> gate =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_GATE_OFFSET);

        SetFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
        for (uint64_t task = coreIdx; task < batch_ * heads_; task += coreNum) {
            const uint64_t b = task / heads_;
            const uint64_t hv = task % heads_;
            const uint32_t stateRowBegin = subBlockIdx * KDA_FWD_H_SUB_DIM;
            InitializeStateAiv(b, hv, stateRowBegin, state);
            SetDirectFreeAiv();
            for (uint32_t chunk = 0;
                 chunk < static_cast<uint32_t>(totalChunks_); ++chunk) {
                ProcessChunkAiv(
                    b, hv, chunk, subBlockIdx,
                    state, stateTyped, direct, out1, vnew, ioTyped, gate);
            }
            if (storeFinalState_) {
                SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
                WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
                DataCopy(finalState_[StateOffset(b, hv) +
                         stateRowBegin * KDA_FWD_H_DIM],
                         state, KDA_FWD_H_STATE_SUB_ELEMS);
                SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
                WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
    }

private:
    GlobalTensor<GK_T> gk_;
    GlobalTensor<float> initialState_;
    GlobalTensor<T> attnOut_;
    GlobalTensor<float> finalState_;
    GlobalTensor<T> aqk_;
    GlobalTensor<T> akk_;
    GlobalTensor<T> w_;
    GlobalTensor<T> u_;
    GlobalTensor<T> qgScaled_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<T> h_;
    uint32_t statePublishCount_[2] = {0, 0};
    uint32_t vnewPublishCount_[2] = {0, 0};
    TEventID aicMte2ToMte1Event_ = KDA_FWD_H_MTE_W_EVENT;
    TEventID aicL1ReuseEvents_[KDA_FWD_H_L1_STAGING_DEPTH] = {0, 1, 2, 3};
    TEventID stateL0FreeEvent_ = KDA_FWD_H_M_EVENT;
    TEventID vnewL0FreeEvent_ = KDA_FWD_H_M_EVENT;
    TEventID aicMte1ToMEvent_ = KDA_FWD_H_M_EVENT;
    TEventID aicMToFixEvent_ = KDA_FWD_H_FIX_EVENT;
    TEventID aicFixToMEvent_ = KDA_FWD_H_FIX_EVENT;
    TEventID aivMte2ToVEvent_ = KDA_FWD_H_MTE_W_EVENT;
    TEventID aivVToMte3Event_ = KDA_FWD_H_MTE_Q_EVENT;
    TEventID aivMte3ToVEvent_ = KDA_FWD_H_MTE_B_EVENT;
    TEventID aivMte3ToMte2Event_ = KDA_FWD_H_IO_REUSE_EVENT;
    Catlass::Arch::Resource<ArchTag> resource_;
    uint64_t batch_ = 0;
    uint64_t heads_ = 0;
    uint64_t seqlen_ = 0;
    uint64_t totalChunks_ = 0;
    uint64_t coreNum_ = 1;
    bool hasInitialState_ = false;
    bool storeFinalState_ = false;
    bool storeVNew_ = false;
    bool storeH_ = false;
    bool fusePostWuIntoFwdH_ = false;
};

} // namespace KdaForward::arch35

#endif
