/*
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is licensed under CANN Open Software License Agreement
 * Version 2.0. See LICENSE in the repository root.
 */
#ifndef TRIANGLE_PAGED_ATTENTION_BLOCK_MMAD_H
#define TRIANGLE_PAGED_ATTENTION_BLOCK_MMAD_H

/*
 * The Cube loops below are deliberately derived from the CANN 9.0.1
 * BlockSparseAttention QK/PV blocks, while the GM->L1 address generation is
 * taken from FIA PageAttention.  The important difference from both upstream
 * implementations is the input contract:
 *
 *   - an absolute contiguous logical-token interval is supplied directly;
 *   - every page fragment is resolved through block_table;
 *   - no packed GM K/V, full mask, or selectIdx array is materialized.
 *
 * The Cube and online-softmax primitives are shared with the in-tree
 * SparseAttentionScore operator.  TrianglePagedSparseAttention only
 * specializes paged K/V address generation and the static Triangle schedule.
 *
 * Cache layout is fixed BSND:
 *   [physical_page, 128, 8, 128].
 */

#include "../../sparse_attention_score/op_kernel/arch22/kernel_utils.hpp"

namespace TrianglePaged {

template <class BaseMmad>
class DirectPagedQkMmad : public BaseMmad {
public:
    using ArchTag = typename BaseMmad::ArchTag;
    using L1TileShape = typename BaseMmad::L1TileShape;
    using L0TileShape = typename BaseMmad::L0TileShape;
    using ElementA = typename BaseMmad::ElementA;
    using ElementB = typename BaseMmad::ElementB;
    using ElementC = typename BaseMmad::ElementC;
    using LayoutA = typename BaseMmad::LayoutA;
    using LayoutB = typename BaseMmad::LayoutB;
    using LayoutC = typename BaseMmad::LayoutC;
    using LayoutAInL1 = typename BaseMmad::LayoutAInL1;
    using LayoutBInL1 = typename BaseMmad::LayoutBInL1;
    using LayoutAInL0 = typename BaseMmad::LayoutAInL0;
    using LayoutBInL0 = typename BaseMmad::LayoutBInL0;
    using LayoutCInL0 = typename BaseMmad::LayoutCInL0;
    using L1AAlignHelper = typename BaseMmad::L1AAlignHelper;

    __aicore__ inline explicit DirectPagedQkMmad(
        NpuArch::Arch::Resource<ArchTag>& resource,
        uint32_t l1Offset = 0)
        : BaseMmad(resource, l1Offset)
    {
    }

    __aicore__ inline void operator()(
        AscendC::GlobalTensor<ElementA> gQ,
        AscendC::GlobalTensor<ElementB> gK,
        AscendC::GlobalTensor<ElementC> gS,
        AscendC::GlobalTensor<int32_t> gBlockTable,
        LayoutA layoutQ,
        LayoutB layoutK,
        LayoutC layoutS,
        NpuArch::GemmCoord shape,
        uint32_t absoluteTokenStart,
        uint32_t pageSize,
        uint32_t strideKv)
    {
        const uint32_t rows = shape.m();
        const uint32_t tokenCount = shape.n();
        const uint32_t embed = shape.k();
        const uint32_t rowsRound =
            RoundUp<L1AAlignHelper::M_ALIGNED>(rows);
        LayoutAInL1 qL1Layout =
            LayoutAInL1::template MakeLayout<ElementA>(rows, embed);
        const uint32_t nLoops =
            CeilDiv<L1TileShape::N>(tokenCount);

        for (uint32_t nLoop = 0; nLoop < nLoops; ++nLoop) {
            const uint32_t nActual =
                nLoop + 1U < nLoops
                    ? L1TileShape::N
                    : tokenCount - nLoop * L1TileShape::N;
            LayoutBInL1 kL1Layout =
                LayoutBInL1::template MakeLayout<ElementB>(
                    embed, nActual);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                this->l1KvPingPongFlag);

            uint32_t copied = 0;
            while (copied < nActual) {
                const uint32_t logicalToken =
                    absoluteTokenStart + nLoop * L1TileShape::N + copied;
                const uint32_t logicalPage = logicalToken / pageSize;
                const uint32_t pageOffset = logicalToken % pageSize;
                const uint32_t physicalPage = static_cast<uint32_t>(
                    gBlockTable.GetValue(logicalPage));
                const uint32_t fragment =
                    AscendC::Std::min(
                        nActual - copied, pageSize - pageOffset);
                const uint64_t physicalElementOffset =
                    static_cast<uint64_t>(
                        physicalPage * pageSize + pageOffset) *
                    strideKv;

                auto sourceLayout =
                    layoutK.GetTileLayout(
                        NpuArch::MakeCoord(embed, fragment));
                NpuArch::MatrixCoord destinationCoord{0, copied};
                auto destination =
                    this->l1BTensor[this->l1KvPingPongFlag]
                        [kL1Layout.GetOffset(destinationCoord)];
                this->copyGmToL1B(
                    destination,
                    gK[physicalElementOffset],
                    kL1Layout,
                    sourceLayout);
                copied += fragment;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
                this->l1KvPingPongFlag);

            const uint32_t mLoops =
                CeilDiv<L0TileShape::M>(rows);
            const uint32_t kLoops =
                CeilDiv<L0TileShape::K>(embed);
            for (uint32_t mLoop = 0; mLoop < mLoops; ++mLoop) {
                const uint32_t mActual =
                    mLoop + 1U < mLoops
                        ? L0TileShape::M
                        : rows - mLoop * L0TileShape::M;
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(
                    this->l0CPingPongFlag);
                for (uint32_t kLoop = 0; kLoop < kLoops; ++kLoop) {
                    const uint32_t kActual =
                        kLoop + 1U < kLoops
                            ? L0TileShape::K
                            : embed - kLoop * L0TileShape::K;
                    LayoutAInL0 qL0Layout =
                        LayoutAInL0::template MakeLayout<ElementA>(
                            mActual, kActual);
                    NpuArch::MatrixCoord qCoord{
                        mLoop * L0TileShape::M,
                        kLoop * L0TileShape::K};
                    auto qL1 =
                        this->l1ATensor[qL1Layout.GetOffset(qCoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag);
                    this->copyL1ToL0A(
                        this->l0ATensor[this->l0ABPingPongFlag],
                        qL1,
                        qL0Layout,
                        qL1Layout);

                    LayoutBInL0 kL0Layout =
                        LayoutBInL0::template MakeLayout<ElementB>(
                            kActual, nActual);
                    NpuArch::MatrixCoord kCoord{
                        kLoop * L0TileShape::K, 0};
                    auto kL1 =
                        this->l1BTensor[this->l1KvPingPongFlag]
                            [kL1Layout.GetOffset(kCoord)];
                    if (mLoop == 0U && kLoop == 0U) {
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE2_MTE1>(
                            this->l1KvPingPongFlag);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag + 2U);
                    this->copyL1ToL0B(
                        this->l0BTensor[this->l0ABPingPongFlag],
                        kL1,
                        kL0Layout,
                        kL1Layout);
                    if (mLoop + 1U == mLoops &&
                        kLoop + 1U == kLoops) {
                        AscendC::SetFlag<
                            AscendC::HardEvent::MTE1_MTE2>(
                            this->l1KvPingPongFlag);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(
                        EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(
                        EVENT_ID0);
                    this->tileMmad(
                        this->l0CTensor[this->l0CPingPongFlag],
                        this->l0ATensor[this->l0ABPingPongFlag],
                        this->l0BTensor[this->l0ABPingPongFlag],
                        rowsRound,
                        nActual,
                        kActual,
                        kLoop == 0U);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag + 2U);
                    this->l0ABPingPongFlag =
                        1U - this->l0ABPingPongFlag;
                }

                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                NpuArch::MatrixCoord sCoord{
                    mLoop * L0TileShape::M,
                    nLoop * L1TileShape::N};
                auto sTileLayout =
                    layoutS.GetTileLayout(
                        NpuArch::MakeCoord(mActual, nActual));
                auto cLayout =
                    LayoutCInL0::MakeLayoutInL0C(
                        NpuArch::MakeCoord(mActual, nActual));
                this->copyL0CToGm(
                    gS[layoutS.GetOffset(sCoord)],
                    this->l0CTensor[this->l0CPingPongFlag],
                    sTileLayout,
                    cLayout);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(
                    this->l0CPingPongFlag);
                this->l0CPingPongFlag =
                    1U - this->l0CPingPongFlag;
            }
            this->l1KvPingPongFlag =
                1U - this->l1KvPingPongFlag;
        }
    }
};

template <class BaseMmad>
class DirectPagedPvMmad : public BaseMmad {
public:
    using ArchTag = typename BaseMmad::ArchTag;
    using L1TileShape = typename BaseMmad::L1TileShape;
    using L0TileShape = typename BaseMmad::L0TileShape;
    using ElementA = typename BaseMmad::ElementA;
    using ElementB = typename BaseMmad::ElementB;
    using ElementC = typename BaseMmad::ElementC;
    using LayoutA = typename BaseMmad::LayoutA;
    using LayoutB = typename BaseMmad::LayoutB;
    using LayoutC = typename BaseMmad::LayoutC;
    using LayoutAInL1 = typename BaseMmad::LayoutAInL1;
    using LayoutBInL1 = typename BaseMmad::LayoutBInL1;
    using LayoutAInL0 = typename BaseMmad::LayoutAInL0;
    using LayoutBInL0 = typename BaseMmad::LayoutBInL0;
    using LayoutCInL0 = typename BaseMmad::LayoutCInL0;
    using L1AAlignHelper = typename BaseMmad::L1AAlignHelper;

    __aicore__ inline explicit DirectPagedPvMmad(
        NpuArch::Arch::Resource<ArchTag>& resource,
        uint32_t l1Offset = 0)
        : BaseMmad(resource, l1Offset)
    {
    }

    __aicore__ inline void operator()(
        AscendC::GlobalTensor<ElementA> gP,
        AscendC::GlobalTensor<ElementB> gV,
        AscendC::GlobalTensor<ElementC> gOTmp,
        AscendC::GlobalTensor<int32_t> gBlockTable,
        LayoutA layoutP,
        LayoutB layoutV,
        LayoutC layoutOTmp,
        NpuArch::GemmCoord shape,
        uint32_t absoluteTokenStart,
        uint32_t pageSize,
        uint32_t strideKv,
        NpuArch::Arch::CrossCoreFlag softmaxReady)
    {
        const uint32_t rows = shape.m();
        const uint32_t embed = shape.n();
        const uint32_t tokenCount = shape.k();
        LayoutBInL1 vL1Layout =
            LayoutBInL1::template MakeLayout<ElementB>(
                tokenCount, embed);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        uint32_t copied = 0;
        while (copied < tokenCount) {
            const uint32_t logicalToken =
                absoluteTokenStart + copied;
            const uint32_t logicalPage = logicalToken / pageSize;
            const uint32_t pageOffset = logicalToken % pageSize;
            const uint32_t physicalPage = static_cast<uint32_t>(
                gBlockTable.GetValue(logicalPage));
            const uint32_t fragment =
                AscendC::Std::min(
                    tokenCount - copied, pageSize - pageOffset);
            const uint64_t physicalElementOffset =
                static_cast<uint64_t>(
                    physicalPage * pageSize + pageOffset) *
                strideKv;
            auto sourceLayout =
                layoutV.GetTileLayout(
                    NpuArch::MakeCoord(fragment, embed));
            NpuArch::MatrixCoord destinationCoord{copied, 0};
            auto destination =
                this->l1BTensor[
                    vL1Layout.GetOffset(destinationCoord)];
            this->copyGmToL1B(
                destination,
                gV[physicalElementOffset],
                vL1Layout,
                sourceLayout);
            copied += fragment;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        NpuArch::Arch::CrossCoreWaitFlag(softmaxReady);

        const uint32_t mLoops =
            CeilDiv<L1TileShape::M>(rows);
        const uint32_t kLoops =
            CeilDiv<L1TileShape::K>(tokenCount);
        for (uint32_t mLoop = 0; mLoop < mLoops; ++mLoop) {
            const uint32_t mActual =
                mLoop + 1U < mLoops
                    ? L1TileShape::M
                    : rows - mLoop * L1TileShape::M;
            const uint32_t mRound =
                RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(
                this->l0CPingPongFlag);

            for (uint32_t kLoop = 0; kLoop < kLoops; ++kLoop) {
                const uint32_t kActual =
                    kLoop + 1U < kLoops
                        ? L1TileShape::K
                        : tokenCount - kLoop * L1TileShape::K;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                    this->l1PPingPongFlag);
                NpuArch::MatrixCoord pCoord{
                    mLoop * L1TileShape::M,
                    kLoop * L1TileShape::K};
                auto pGm = gP[layoutP.GetOffset(pCoord)];
                auto pTileLayout =
                    layoutP.GetTileLayout(
                        NpuArch::MakeCoord(mActual, kActual));
                LayoutAInL1 pL1Layout =
                    LayoutAInL1::template MakeLayout<ElementA>(
                        mActual, kActual);
                this->copyGmToL1A(
                    this->l1ATensor[this->l1PPingPongFlag],
                    pGm,
                    pL1Layout,
                    pTileLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
                    this->l1PPingPongFlag);

                const uint32_t kL0Loops =
                    CeilDiv<L0TileShape::K>(kActual);
                for (uint32_t kL0Loop = 0;
                     kL0Loop < kL0Loops;
                     ++kL0Loop) {
                    const uint32_t kL0Actual =
                        kL0Loop + 1U < kL0Loops
                            ? L0TileShape::K
                            : kActual - kL0Loop * L0TileShape::K;
                    LayoutAInL0 pL0Layout =
                        LayoutAInL0::template MakeLayout<ElementA>(
                            mActual, kL0Actual);
                    NpuArch::MatrixCoord pL1Coord{
                        0, kL0Loop * L0TileShape::K};
                    auto pL1 =
                        this->l1ATensor[this->l1PPingPongFlag]
                            [pL1Layout.GetOffset(pL1Coord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag);
                    if (kL0Loop == 0U) {
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE2_MTE1>(
                            this->l1PPingPongFlag);
                    }
                    this->copyL1ToL0A(
                        this->l0ATensor[this->l0ABPingPongFlag],
                        pL1,
                        pL0Layout,
                        pL1Layout);
                    if (kL0Loop + 1U == kL0Loops) {
                        AscendC::SetFlag<
                            AscendC::HardEvent::MTE1_MTE2>(
                            this->l1PPingPongFlag);
                    }

                    LayoutBInL0 vL0Layout =
                        LayoutBInL0::template MakeLayout<ElementB>(
                            kL0Actual, embed);
                    NpuArch::MatrixCoord vL1Coord{
                        kLoop * L1TileShape::K +
                            kL0Loop * L0TileShape::K,
                        0};
                    auto vL1 =
                        this->l1BTensor[
                            vL1Layout.GetOffset(vL1Coord)];
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag + 2U);
                    this->copyL1ToL0B(
                        this->l0BTensor[this->l0ABPingPongFlag],
                        vL1,
                        vL0Layout,
                        vL1Layout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(
                        EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(
                        EVENT_ID0);
                    this->tileMmad(
                        this->l0CTensor[this->l0CPingPongFlag],
                        this->l0ATensor[this->l0ABPingPongFlag],
                        this->l0BTensor[this->l0ABPingPongFlag],
                        mRound,
                        embed,
                        kL0Actual,
                        kLoop == 0U && kL0Loop == 0U);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                        this->l0ABPingPongFlag + 2U);
                    this->l0ABPingPongFlag =
                        1U - this->l0ABPingPongFlag;
                }
                this->l1PPingPongFlag =
                    1U - this->l1PPingPongFlag;
            }

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            NpuArch::MatrixCoord oCoord{
                mLoop * L0TileShape::M, 0};
            auto oTileLayout =
                layoutOTmp.GetTileLayout(
                    NpuArch::MakeCoord(mActual, embed));
            auto cLayout =
                LayoutCInL0::MakeLayoutInL0C(
                    NpuArch::MakeCoord(mActual, embed));
            this->copyL0CToGm(
                gOTmp[layoutOTmp.GetOffset(oCoord)],
                this->l0CTensor[this->l0CPingPongFlag],
                oTileLayout,
                cLayout);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(
                this->l0CPingPongFlag);
            this->l0CPingPongFlag =
                1U - this->l0CPingPongFlag;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    }
};

}  // namespace TrianglePaged

#endif  // TRIANGLE_PAGED_ATTENTION_BLOCK_MMAD_H
