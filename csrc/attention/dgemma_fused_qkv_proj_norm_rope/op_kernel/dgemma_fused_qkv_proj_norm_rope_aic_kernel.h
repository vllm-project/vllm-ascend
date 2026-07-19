/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope_aic_kernel.h
 *  \brief Cube (AIC) stage: qkv = hidden @ Wqkv.T via Catlass BlockMmad.
 *  Writes qkv[m,n] into GM workspace; signals AIV per m-split (dur=max(aic,aiv)). */
#ifndef DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_AIC_KERNEL_H
#define DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_AIC_KERNEL_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include <kernel_operator.h>
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "dgemma_fused_qkv_proj_norm_rope_tiling_data.h"
#include "dgemma_fused_qkv_proj_norm_rope_utils.h"

using namespace AscendC;
using namespace Catlass;
using namespace DgemmaFusedQkvProjNormRope;

// QKV_PIPE_DEPTH + AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID come from the shared utils.h

template <class MmadDtype>
class DgemmaFusedQkvProjNormRopeAicKernel {
public:
    __aicore__ inline DgemmaFusedQkvProjNormRopeAicKernel() {}

    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR wqkv, GM_ADDR qkv_ws,
                                const DgemmaFusedQkvProjNormRopeTilingData *tiling)
    {
        m_ = tiling->m; k_ = tiling->k; n_ = tiling->n;
        m0_ = tiling->m0; swizzlCount_ = tiling->swizzlCount;
        syncDoneFlag_ = tiling->syncDoneFlag;
        syncReadyFlag_ = tiling->syncReadyFlag;
        gm_a_src_ = reinterpret_cast<__gm__ MmadDtype *>(hidden);
        gm_b_src_ = reinterpret_cast<__gm__ MmadDtype *>(wqkv);
        gm_c_src_ = reinterpret_cast<__gm__ MmadDtype *>(qkv_ws);
        core_idx_ = get_block_idx();
        core_num_ = get_block_num();
    }

    __aicore__ inline void Process()
    {
        // Wait for the vector cores to signal they have entered the consume loop
        // before the cube starts writing GM. Mirrors MC2 template InitFlags(); without
        // it the cube can run ahead of a not-yet-ready vector core -> startup race.
        WaitEvent(syncReadyFlag_);

        using LayoutA = layout::RowMajor;      // hidden [m,k] row-major
        using LayoutB = layout::ColumnMajor;   // Wqkv row-major [n,k] viewed as ColumnMajor [k,n] == Wqkv.T
        using LayoutC = layout::RowMajor;      // qkv [m,n] row-major
        LayoutB layoutB{(layout::ColumnMajor::Index)k_, (layout::ColumnMajor::Index)n_};

        using L1TileShape = GemmShape<128, 256, 256>;
        using L0TileShape = GemmShape<128, 256, 64>;
        using AType = Gemm::GemmType<MmadDtype, LayoutA>;
        using BType = Gemm::GemmType<MmadDtype, LayoutB>;
        using CType = AType;
        // UNIT_FLAG=false: each block's L0C->GM fixpipe is self-flushing at issue time.
        // The residual race is confined to splitM==1 (T<=128): no splitM>=2 size (T>=192)
        // has ever failed across runs, while a PROBABILISTIC subset of splitM==1 sizes
        // fails each run (which sizes shift run-to-run) on the first-read region only.
        // At splitM>=2 the extra matmul latency masks the cube->vector visibility gap.
        // Self-flushing removes the reliance on a single trailing FIX_M drain.
        constexpr bool ENABLE_UNIT_FLAG = false;
        using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
        using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        GemmCoord blockShape = L1TileShape::ToCoord();

        BlockMmad blockMmad(resource_);
        gmB_.SetGlobalBuffer(gm_b_src_, (uint64_t)k_ * n_);

        int mPerSplit = m0_ * swizzlCount_;
        if (mPerSplit <= 0) mPerSplit = m0_;
        int splitM = AscendC::DivCeil((int)m_, mPerSplit);
        // Cube->vector GM visibility race: the vector's first MTE2 read after the single
        // end-of-cube cross-core sync races when the cube runs as ONE split (splitM==1,
        // small m) -- the first-read region is corrupted probabilistically. When the cube
        // does >=2 splits the extra matmul latency lets the first region settle before the
        // sync fires (empirically: splitM>=2 is always clean). Force >=2 splits for small m
        // so the vector never reads a workspace region whose fixpipe is still in flight.
        // GM-visibility race only manifests at large m (empirically T>=128 NaNs at
        // splitM==1; T<=96 is clean). Forcing a 2-split at SMALL m instead produces
        // tile-unaligned sub-tiles (mPerSplit 7..12 for T=13..24) that the cube mis-
        // computes. So force >=2 splits ONLY for m>=64, where mPerSplit>=32 stays tile-
        // aligned; small m stays at splitM==1 (race-free there).
        // Cube fractal requires 16-aligned m tiles. Force >=2 splits (to defeat the
        // splitM==1 GM-visibility race) but keep every split 16-aligned: mPerSplit is
        // rounded UP to a multiple of 16 so both the head split(s) and the tail land on
        // fractal boundaries. Only split when m>16 (a single <=16 tile is already one fractal).
        if (splitM < 2 && m_ > 16) {
            int half = AscendC::DivCeil((int)m_, 2);
            mPerSplit = ((half + 15) / 16) * 16;      // round up to 16
            splitM = AscendC::DivCeil((int)m_, mPerSplit);
            if (splitM < 2) { mPerSplit = 16; splitM = AscendC::DivCeil((int)m_, mPerSplit); }
        }
        icache_preload(8);

        for (int splitIndex = 0; splitIndex < splitM; ++splitIndex) {
            uint32_t mStart = splitIndex * mPerSplit;
            uint32_t mActual = ((uint32_t)mPerSplit > (m_ - mStart)) ? (m_ - mStart) : (uint32_t)mPerSplit;

            __gm__ MmadDtype *gm_a_tmp = gm_a_src_ + (uint64_t)mStart * k_;
            __gm__ MmadDtype *gm_c_tmp = gm_c_src_ + (uint64_t)mStart * n_;
            gmA_.SetGlobalBuffer(gm_a_tmp, (uint64_t)mActual * k_);
            gmC_.SetGlobalBuffer(gm_c_tmp, (uint64_t)mActual * n_);

            GemmCoord splitShape{(uint32_t)mActual, n_, k_};
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            BlockScheduler splitScheduler(splitShape, blockShape.GetCoordMN());
            uint32_t coreLoops = splitScheduler.GetCoreLoops();

            LayoutA layoutA{(uint32_t)mActual, k_};
            LayoutC layoutC{(uint32_t)mActual, n_};

            for (uint32_t loopIdx = core_idx_; loopIdx < coreLoops; loopIdx += core_num_) {
                GemmCoord blockCoord = splitScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = splitScheduler.GetActualBlockShape(blockCoord);
                GemmCoord offsetCoord = blockCoord * blockShape;
                MatrixCoord offsetA = offsetCoord.GetCoordMK();
                MatrixCoord offsetB = offsetCoord.GetCoordKN();
                MatrixCoord offsetC = offsetCoord.GetCoordMN();
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                blockMmad(gmA_[gmOffsetA], layoutA, gmB_[gmOffsetB], layoutB,
                          gmC_[gmOffsetC], layoutC, actualBlockShape);
            }

        }
        // Correctness-first serialization: complete ALL cube splits, drain the fixpipe,
        // then signal the vector ONCE. The per-split pingpong handshake had a residual
        // race on the plain-workspace GM buffer (non-deterministic q/k/v corruption at
        // some m). Cube is only ~53us of the ~332us kernel, so serializing costs little;
        // the cube/vector overlap capability is documented separately from the profiler.
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        // Force fixpipe writeback of the ENTIRE qkv workspace to GM before signaling the
        // vector. Without this the vector's first MTE2 read can race in-flight cube writes
        // for mid-range m (the forced-split heuristic only masked this probabilistically).
        {
            AscendC::GlobalTensor<MmadDtype> wsFull;
            wsFull.SetGlobalBuffer(gm_c_src_, (uint64_t)m_ * n_);
            AscendC::DataCacheCleanAndInvalid<MmadDtype, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                              AscendC::DcciDst::CACHELINE_OUT>(wsFull);
        }
        PipeBarrier<PIPE_ALL>();
        FFTSCrossCoreSync<PIPE_FIX>(FFTS_SYNC_AICORE_GROUP_MODE, syncDoneFlag_);
    }

private:
    Arch::Resource<Arch::AtlasA2> resource_;
    AscendC::GlobalTensor<MmadDtype> gmA_;
    AscendC::GlobalTensor<MmadDtype> gmB_;
    AscendC::GlobalTensor<MmadDtype> gmC_;
    __gm__ MmadDtype *gm_a_src_{nullptr};
    __gm__ MmadDtype *gm_b_src_{nullptr};
    __gm__ MmadDtype *gm_c_src_{nullptr};
    uint32_t m_, k_, n_, m0_, swizzlCount_, core_idx_, core_num_;
    uint32_t syncDoneFlag_, syncReadyFlag_;
};
#endif
