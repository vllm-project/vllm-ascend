/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msa_block_mmad.h
 * \brief Q 驻留 L1 + K L1 pingpong + L0C 双缓冲的 BlockMmad。
 *
 * 基于 Catlass FullLoadA：同一 M-tile 只搬一次 Q；相邻 K page 的 FIXPIPE 与 Cube 重叠。
 * Atlas A2 L0C=128KB，两块 128x128 fp32 C 各 64KB。ENABLE_UNIT_FLAG 保持 false。
 */

#ifndef MSA_BLOCK_MMAD_H
#define MSA_BLOCK_MMAD_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

#include "../msa_index_score_common.h"

namespace MsaIndexScoreNs {

template <class AType_, class BType_, class CType_, uint32_t STAGES_IN = 3, bool USE_UNIT_FLAG_ = false>
class MsaBlockMmad {
public:
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using BiasType = void;
    using TileCopy = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType>;
    using TileMmad = Catlass::Gemm::Tile::TileMmad<ArchTag, AType_, BType_, BiasType>;
    using CopyGmToL1A = typename TileCopy::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy::CopyL0CToGm;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = Catlass::layout::zN;
    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    // K (B) 的 L1 流水级数。非量化 3 级：fixpipe 减半（fp16 S）后 MTE2 的逐页延迟
    // （~390ns，L2 命中）会成为新的关键路径，加深一级让 MTE2 提前两页预取。
    // int8 保持 2 级：K 源是每 stile 复用的 per-core scratch，AIV 下一 stile 的 cast
    // 会重写该区，更深的预取与 cast 重写产生竞态（实测 STAGES=3 下 int8 崩溃）。
    // L1 占用：A 32KB + 3×32KB B = 128KB / 512KB。
    static constexpr uint32_t STAGES = STAGES_IN;
    // 非量化路径开 unit flag：mmad 与 fixpipe 的依赖交给硬件互锁，省掉每页
    // M_FIX/FIX_M 四次标量 set/wait。标量因此能在 cube 完成前就发出下一页的
    // MTE2/MTE1，把 GM→L1 的搬运藏到 cube/fixpipe 后面。
    static constexpr bool USE_UNIT_FLAG = USE_UNIT_FLAG_;
    static constexpr uint32_t L0C_STAGES = 2;
    static constexpr uint32_t L1_M = MSA_ROW_TILE_M;
    static constexpr uint32_t L1_N = MSA_BLOCK_SIZE;
    static constexpr uint32_t L1_K = MSA_K_TILE;
    static constexpr uint32_t L1A_SIZE = L1_M * L1_K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1_N * L1_K * sizeof(ElementB);
    // L0A/L0B 各仅 64KB，B tile 32KB → 乒乓固定 2 级（32KB/级）；与 L1B STAGES 解耦。
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0A_RESIDENT_ID = 0;
    static constexpr uint32_t L0A_FALLBACK_ID = 1;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0A_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0B_STAGES;
    static constexpr uint32_t L0C_TILE_SIZE = L1_M * L1_N * sizeof(ElementAccumulator);

    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "LayoutC must be RowMajor (fixpipe nz2nd)");
    // A2 的 L0A/L0B 各仅 64KB：A/B tile 32KB，乒乓最多 2 级（32KB/级）。
    static_assert(L0A_PINGPONG_BUF_SIZE >= L1A_SIZE, "L0A pingpong buf smaller than A tile");
    static_assert(L0B_PINGPONG_BUF_SIZE >= L1B_SIZE, "L0B pingpong buf smaller than B tile");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0C pingpong exceeds L0C");
    static_assert(L1A_SIZE + L1B_SIZE * STAGES <= ArchTag::L1_SIZE, "L1 A+B exceeds L1");

    __aicore__ inline MsaBlockMmad(Catlass::Arch::Resource<ArchTag> &resource)
    {
        l1ATensor_ = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        const uint32_t l1BOffset = L1A_SIZE;
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1BTensorList_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
            l1BEventList_[i] = static_cast<int32_t>(i + STAGES);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
        }
        // L0A 槽 0 专供常驻 Q：不预置 M_MTE1，改用 LoadA 内「Set+Wait 成对」自平衡，
        // 避免计数型 flag 让 MTE1 用构造时的余额提前放行、覆盖仍被 mmad 读取的 Q。
        // 槽 1 供 kTileCount>1 的回退路径（每 k-tile 重搬 A），沿用预置 + 析构等待。
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0AEventList_[i] = static_cast<int32_t>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0BEventList_[i] = static_cast<int32_t>(i + STAGES + L0A_STAGES);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
            l0CEventList_[i] = static_cast<int32_t>(i);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }
    }

    __aicore__ inline ~MsaBlockMmad()
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<ElementA> &gmA, const LayoutA &layoutA,
                                      const AscendC::GlobalTensor<ElementB> &gmB, const LayoutB &layoutB,
                                      const AscendC::GlobalTensor<ElementC> &gmC, const LayoutC &layoutC,
                                      const Catlass::GemmCoord &actualShape, bool needLoadL1)
    {
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1_M, actualShape.k());
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1_K, L1_N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mRound, nRound));
        uint32_t kActual = Min(actualShape.k(), L1_K);

        const uint32_t kTileCount = CeilDiv<L1_K>(actualShape.k());
        const uint32_t mPartLoop = CeilDiv<L1_M>(mRound);
        const uint32_t nPartLoop = CeilDiv<L1_N>(nRound);
        // headDim <= L1_K（实际场景恒成立）时整块 Q 一次进 L0A，可跨 page 复用；
        // 否则退回逐 page 重搬 A 的通用路径。
        const bool qResident = (kTileCount == 1U) && (mPartLoop == 1U) && (CeilDiv<L1_K>(kActual) == 1U);

        if (needLoadL1) {
            auto layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(), actualShape.k()));
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_);
            copyGmToL1A_(l1ATensor_, gmA, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_);
            if (qResident) {
                // 本 M-tile 的 Q 一次搬进 L0A 并驻留：省掉每页 32KB 的 L1→L0A
                // 搬运与两次标量同步。先让 MTE1 等上一 M-tile 的 mmad 读完。
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_RESIDENT_ID]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_RESIDENT_ID]);
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
                copyL1ToL0A_(l0ATensorList_[L0A_RESIDENT_ID], l1ATensor_, layoutAInL0, layoutAInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[L0A_RESIDENT_ID]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[L0A_RESIDENT_ID]);
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);
        auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActual, actualShape.n()));
        copyGmToL1B_(l1BTensorList_[l1ListId_], gmB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);

        // 等本槽位上一轮 FIXPIPE 完成后再写 L0C，从而与另一槽位的 FIXPIPE 重叠。
        // unit flag 下该依赖由硬件维护，省掉这次标量等待。
        if constexpr (!USE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            const uint32_t l1ListIdNext = (l1ListId_ + 1 < STAGES) ? (l1ListId_ + 1) : 0;
            uint32_t kActualNext = 0;
            if (kLoopIdx < kTileCount - 1) {
                const uint32_t kLoopIdxNext = kLoopIdx + 1;
                kActualNext = (kLoopIdxNext < kTileCount - 1) ? L1_K : (actualShape.k() - kLoopIdxNext * L1_K);
                Catlass::MatrixCoord gmTileBOffset{kLoopIdxNext * L1_K, 0};
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListIdNext]);
                layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B_(l1BTensorList_[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListIdNext]);
            }

            auto l1ATensor = l1ATensor_;
            auto l1BTensor = l1BTensorList_[l1ListId_];
            const uint32_t kPartLoop = CeilDiv<L1_K>(kActual);

            for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
                const uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L1_M : (mRound - mPartIdx * L1_M);
                for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                    const uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L1_K : (kActual - kPartIdx * L1_K);
                    auto l0ATile = l0ATensorList_[qResident ? L0A_RESIDENT_ID : L0A_FALLBACK_ID];
                    if (!qResident) {
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                        Catlass::MatrixCoord l1AOffset{mPartIdx * L1_M, kPartIdx * L1_K + kLoopIdx * L1_K};
                        auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
                        copyL1ToL0A_(l0ATile, l1ATile, layoutAInL0, layoutAInL1);
                    }

                    for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                        const uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L1_N : (nRound - nPartIdx * L1_N);
                        auto l0BTile = l0BTensorList_[l0BListId_];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        Catlass::MatrixCoord l1BOffset{kPartIdx * L1_K, nPartIdx * L1_N};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);
                        }
                        copyL1ToL0B_(l0BTile, l1BTile, layoutBInL0, layoutBInL1);
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        Catlass::MatrixCoord l0COffset{mPartIdx * L1_M, nPartIdx * L1_N};
                        auto l0CTile = l0CTensorList_[l0CListId_][layoutInL0C.GetOffset(l0COffset)];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        const bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        uint8_t unitFlag = 0b00;
                        if constexpr (USE_UNIT_FLAG) {
                            const bool last = (kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                              (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1);
                            unitFlag = last ? 0b11 : 0b10;
                        }
                        tileMmad_(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);
                        l0BListId_ = (l0BListId_ + 1 < L0B_STAGES) ? (l0BListId_ + 1) : 0;
                    }
                    if (!qResident) {
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
                    }
                }
            }
            l1ListId_ = l1ListIdNext;
            kActual = kActualNext;
        }

        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());
        if constexpr (USE_UNIT_FLAG) {
            copyL0CToGm_(gmC, l0CTensorList_[l0CListId_], layoutBlock, layoutInL0C, 0b11);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
            copyL0CToGm_(gmC, l0CTensorList_[l0CListId_], layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
        }
        l0CListId_ = (l0CListId_ + 1 < L0C_STAGES) ? (l0CListId_ + 1) : 0;
    }

private:
    AscendC::LocalTensor<ElementA> l1ATensor_;
    AscendC::LocalTensor<ElementB> l1BTensorList_[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList_[L0C_STAGES];
    int32_t l1AEvent_{0};
    int32_t l1BEventList_[STAGES]{};
    int32_t l0AEventList_[STAGES]{};
    int32_t l0BEventList_[STAGES]{};
    int32_t l0CEventList_[L0C_STAGES]{};
    uint32_t l1ListId_{0};
    uint32_t l0BListId_{0};
    uint32_t l0CListId_{0};
    TileMmad tileMmad_;
    CopyGmToL1A copyGmToL1A_;
    CopyGmToL1B copyGmToL1B_;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    CopyL0CToGm copyL0CToGm_;
};

} // namespace MsaIndexScoreNs

#endif // MSA_BLOCK_MMAD_H
