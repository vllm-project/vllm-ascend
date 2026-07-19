/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front_aic_kernel.h
 *  \brief Cube (AIC) stage of the router MIX op, runs SECOND (consumer):
 *  waits for the vector's normed rows in norm_ws, then GEMM out = norm_ws @ proj_weight.T.
 *  out[m, n=num_experts] feeds MoE topk (host). No downstream on-device consumer, so the
 *  cube does not signal the vector at the end (unlike the qkv MIX, which is cube-first). */
#ifndef DGEMMA_FUSED_ROUTER_FRONT_AIC_KERNEL_H
#define DGEMMA_FUSED_ROUTER_FRONT_AIC_KERNEL_H

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
#include "dgemma_fused_router_front_tiling_data.h"
#include "dgemma_fused_router_front_utils.h"

using namespace AscendC;
using namespace Catlass;
using namespace DgemmaFusedRouterFront;

template <class MmadDtype>
class DgemmaFusedRouterFrontAicKernel {
public:
    __aicore__ inline DgemmaFusedRouterFrontAicKernel() {}

    __aicore__ inline void Init(GM_ADDR norm_ws, GM_ADDR proj_weight, GM_ADDR out, GM_ADDR sync_scratch,
                                const DgemmaFusedRouterFrontTilingData *tiling)
    {
        m_ = tiling->m; k_ = tiling->k; n_ = tiling->n;
        m0_ = tiling->m0; swizzlCount_ = tiling->swizzlCount;
        syncAlignFlag_ = tiling->syncReadyFlag;
        syncNormReadyFlag_ = tiling->syncDoneFlag + 1U;
        syncLogitsDoneFlag_ = tiling->syncDoneFlag + 2U;
        gm_a_src_ = reinterpret_cast<__gm__ MmadDtype *>(norm_ws);      // normed x [m,k]
        gm_b_src_ = reinterpret_cast<__gm__ MmadDtype *>(proj_weight);  // W [n,k] row-major
        gm_c_src_ = reinterpret_cast<__gm__ float *>(out);              // fp32 logits [m,n]
        syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sync_scratch), 128);
        core_idx_ = get_block_idx();
        core_num_ = get_block_num();
    }

    __aicore__ inline void Process()
    {
        // Two-phase consumer wait:
        // 1. align: AIV has entered this op instance (stale align alone is harmless).
        // 2. norm-ready: AIV has finished writing current norm_scratch.
        // Splitting these avoids consuming an old "ready" event as the data dependency.
        WaitEvent(syncAlignFlag_);
        WaitEvent(syncAlignFlag_);
        WaitEvent(syncNormReadyFlag_);
        WaitEvent(syncNormReadyFlag_);

        using LayoutA = layout::RowMajor;      // norm_ws [m,k] row-major
        using LayoutB = layout::ColumnMajor;   // W row-major [n,k] viewed as ColumnMajor [k,n] == W.T
        using LayoutC = layout::RowMajor;      // out [m,n] row-major
        LayoutB layoutB{(layout::ColumnMajor::Index)k_, (layout::ColumnMajor::Index)n_};

        using L1TileShape = GemmShape<128, 256, 256>;
        using L0TileShape = GemmShape<128, 256, 64>;
        using AType = Gemm::GemmType<MmadDtype, LayoutA>;
        using BType = Gemm::GemmType<MmadDtype, LayoutB>;
        using CType = Gemm::GemmType<float, LayoutC>;
        constexpr bool ENABLE_UNIT_FLAG = false;
        using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
        using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        GemmCoord blockShape = L1TileShape::ToCoord();

        BlockMmad blockMmad(resource_);
        gmB_.SetGlobalBuffer(gm_b_src_, (uint64_t)k_ * n_);

        int mPerSplit = m0_ * swizzlCount_;
        if (mPerSplit <= 0) mPerSplit = m0_;
        int splitM = AscendC::DivCeil((int)m_, mPerSplit);
        // Same GM-visibility guard as qkv MIX: when the cube runs as a single
        // split, the vector's first post-sync read can race the cube fixpipe
        // visibility on some shapes. Keep split boundaries 16-aligned for cube.
        if (splitM < 2 && m_ > 16) {
            int half = AscendC::DivCeil((int)m_, 2);
            mPerSplit = ((half + 15) / 16) * 16;
            splitM = AscendC::DivCeil((int)m_, mPerSplit);
            if (splitM < 2) {
                mPerSplit = 16;
                splitM = AscendC::DivCeil((int)m_, mPerSplit);
            }
        }
        icache_preload(8);

        for (int splitIndex = 0; splitIndex < splitM; ++splitIndex) {
            uint32_t mStart = splitIndex * mPerSplit;
            uint32_t mActual = ((uint32_t)mPerSplit > (m_ - mStart)) ? (m_ - mStart) : (uint32_t)mPerSplit;

            __gm__ MmadDtype *gm_a_tmp = gm_a_src_ + (uint64_t)mStart * k_;
            __gm__ float *gm_c_tmp = gm_c_src_ + (uint64_t)mStart * n_;
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
        // Drain the fixpipe so logits_scratch[m,n] is committed to GM before AIV consumes it.
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        {
            AscendC::GlobalTensor<float> logitsFull;
            logitsFull.SetGlobalBuffer(gm_c_src_, (uint64_t)m_ * n_);
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                              AscendC::DcciDst::CACHELINE_OUT>(logitsFull);
        }
        PipeBarrier<PIPE_ALL>();
        FFTSCrossCoreSync<PIPE_FIX>(FFTS_SYNC_AICORE_GROUP_MODE, syncLogitsDoneFlag_);
        FFTSCrossCoreSync<PIPE_FIX>(FFTS_SYNC_AICORE_GROUP_MODE, syncLogitsDoneFlag_);
    }

private:
    Arch::Resource<Arch::AtlasA2> resource_;
    AscendC::GlobalTensor<MmadDtype> gmA_;
    AscendC::GlobalTensor<MmadDtype> gmB_;
    AscendC::GlobalTensor<float> gmC_;
    AscendC::GlobalTensor<int32_t> syncGm_;
    __gm__ MmadDtype *gm_a_src_{nullptr};
    __gm__ MmadDtype *gm_b_src_{nullptr};
    __gm__ float *gm_c_src_{nullptr};
    uint32_t m_, k_, n_, m0_, swizzlCount_, core_idx_, core_num_;
    uint32_t syncAlignFlag_, syncNormReadyFlag_, syncLogitsDoneFlag_;
};
#endif
