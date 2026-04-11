/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_AIC_KERNEL_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_AIC_KERNEL_H

#define ASCENDC_CUBE_ONLY

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "matmul_allreduce_add_rmsnorm_utils.h"
#include "matmul_allreduce_add_rmsnorm_tiling.h"

constexpr int32_t SCALE_L1_SIZE_A = 256 * 8;
constexpr int32_t SCALE_L1_SIZE_B = 128 * 1024;
constexpr int32_t CUBE_MATRIX_SIZE_B16 = 256;                    // 16 * 16
constexpr int32_t CUBE_MATRIX_SIZE_B8 = 16 * 32;                 // 16 * 32
constexpr int32_t SCALE_L1_SIZE = 256 * 8;                       // 2 KB
constexpr int32_t BLOCK_SIZE_16 = 16;
constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int32_t DOUBLE_BUFFER_SIZE = 2;
constexpr uint32_t MM_L1_TILE_SHAPE_M = 128;
constexpr uint32_t MM_L1_TILE_SHAPE_N = 256;
constexpr uint32_t MM_L1_TILE_SHAPE_K = 256;
constexpr uint32_t MM_L0_TILE_SHAPE_M = MM_L1_TILE_SHAPE_M;
constexpr uint32_t MM_L0_TILE_SHAPE_N = MM_L1_TILE_SHAPE_N;
constexpr uint32_t MM_L0_TILE_SHAPE_K = 64;

using namespace Catlass;

template <typename T_INPUT>
struct GetAccumType {
    using T = float;
};

__aicore__ inline bool IsQuant(const QuantGranularity &granularity)
{
    return (granularity > QuantGranularity::QUANT_GRANULARITY_UNDEFINED) &&
           (granularity < QuantGranularity::QUANT_GRANULARITY_MAX);
}

template <typename MmadDtype, typename OutDtype>
class MatmulAllreduceAddRmsnormAicKernel {
    using T_ACCUM = typename GetAccumType<MmadDtype>::T;
public:
    int PIPE_DEPTH = 2;
    Arch::Resource<Arch::AtlasA2> resource;
    __aicore__ inline MatmulAllreduceAddRmsnormAicKernel<MmadDtype, OutDtype>() { }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR gamma, GM_ADDR y,
                                GM_ADDR workspace, const MatmulAllreduceAddRmsnormTilingData* tilingData,
                                Hccl<HCCL_SERVER_TYPE_AICPU> &hccl_)
    {
        this->hccl_ = hccl_;
        this->gm_c = reinterpret_cast<__gm__ OutDtype *>(y);

        this->gm_dequant_scale = nullptr;
        this->has_offset = false;

        auto ppTilingData = &tilingData->matmulAllreduceAddRmsnormInfo.ppTilingData;
        auto commTilingData = &tilingData->matmulAllreduceAddRmsnormInfo.commTilingData;
        auto quantInfo = &tilingData->matmulAllreduceAddRmsnormInfo.quantInfo;

        this->batch_size = ppTilingData->opShape.batchSize;
        this->m = ppTilingData->opShape.m;
        this->k = ppTilingData->opShape.k;
        this->n = ppTilingData->opShape.n;
        this->weight_nz = false;

        this->is_int8 = false;
        this->cube_matrix_size = this->is_int8 ? CUBE_MATRIX_SIZE_B8 : CUBE_MATRIX_SIZE_B16;

        this->m_align = Block512B<MmadDtype>::AlignUp(m);
        this->k_align = Block512B<MmadDtype>::AlignUp(k);
        this->n_align = Block512B<MmadDtype>::AlignUp(n);

        this->m0 = ppTilingData->m0;
        this->k0 = ppTilingData->k0;
        this->n0 = ppTilingData->n0;

        int32_t tiling_key = ppTilingData->tilingKey;
        this->trans_a = ppTilingData->isTransA;
        this->trans_b = ppTilingData->isTransB;

        int32_t aligned_a;
        int32_t aligned_b;
        this->dequant_granularity = quantInfo->dequantGranularity;
        AlignJudge(this->trans_a, this->trans_b, this->m, this->k, this->n,
            this->m_align, this->k_align, this->n_align, aligned_a, aligned_b);
        this->aligned_a = aligned_a;
        this->aligned_b = aligned_b;
        if (weight_nz) {
            this->k_align16 = Block32B<MmadDtype>::AlignUp(k);
            this->n_align16 = Block32B<MmadDtype>::AlignUp(n);
        }
        bool has_a_align = IsQuant(quantInfo->quantGranularity) || aligned_a;
        bool has_b_align = IsQuant(this->dequant_granularity) && !this->is_int8 || aligned_b;
        bool has_accum = IsQuant(this->dequant_granularity) &&
            this->is_int8 && std::is_same<OutDtype, bfloat16_t>::value;
        bool has_format_dequant_offset =
            (this->dequant_granularity == QuantGranularity::PER_TENSOR) && this->is_int8 && this->has_offset;
        auto workspace_info = GetWorkspaceInfo(workspace, this->batch_size, this->m, this->k, this->n,
            this->m_align, this->k_align, this->n_align, this->trans_a, this->trans_b,
            sizeof(MmadDtype), has_a_align, has_b_align, has_accum, has_format_dequant_offset);
        this->gm_a_src = reinterpret_cast<__gm__ MmadDtype *>(x1);
        this->gm_b_src = reinterpret_cast<__gm__ MmadDtype *>(x2);
        this->gm_format_dequant_offset = reinterpret_cast<__gm__ int32_t *>(has_format_dequant_offset ?
            workspace_info.gm_dequant_param : nullptr);
        this->gm_workspace_src = workspace;
        this->block_size = BLOCK_SIZE_32 / sizeof(MmadDtype);

        int32_t a_l1_size = this->m0 * this->k0 * sizeof(MmadDtype);
        int32_t a_l1_size_round = AscendC::DivCeil(a_l1_size, 512) * 512;
        int32_t b_l1_size = this->n0 * this->k0 * sizeof(MmadDtype);
        int32_t b_l1_size_round = AscendC::DivCeil(b_l1_size, 512) * 512;
        this->l1_base_a = reinterpret_cast<__cbuf__ MmadDtype *>((uintptr_t)(this->is_int8 ? SCALE_L1_SIZE : 0));
        this->l1_base_b =
            reinterpret_cast<__cbuf__ MmadDtype *>(a_l1_size_round * (this->is_int8 ? DOUBLE_BUFFER_SIZE : 1) +
            (uintptr_t) this->l1_base_a);

        this->core_num = get_block_num();
        this->core_idx = get_block_idx();

        this->m_loop = ppTilingData->mLoop;
        this->k_loop = ppTilingData->kLoop;
        this->n_loop = ppTilingData->nLoop;
        this->core_loop = ppTilingData->coreLoop;
        this->swizzl_count = ppTilingData->swizzlCount;
        this->swizzl_direct = ppTilingData->swizzlDirect;
        this->is_91093 = commTilingData->is91093;
        this->ping_flag = 1;
        this->rank = hccl_.GetRankId();
        this->rank_size = hccl_.GetRankDim();
        this->withSerialMode = commTilingData->withSerialMode;

        this->gm_peer_mem = (__gm__ OutDtype *)hccl_.GetWindowsInAddr(this->rank);
    }

    __aicore__ inline void MoveL0CToGM(__gm__ OutDtype *gm_dst, int64_t offset_c,
        int32_t m_actual, int32_t n_actual, int32_t src_stride, int32_t dst_stride) {
            if constexpr (std::is_same<OutDtype, __bf16>::value) {
                copy_matrix_cc_to_gm(
                    gm_dst + offset_c,
                    l0c_buf,
                    0,
                    n_actual,
                    m_actual,
                    dst_stride,
                    src_stride,
                    0,
                    F322BF16,
                    0,
                    false,
                    true
                );
            } else {
                copy_matrix_cc_to_gm(
                    gm_dst + offset_c,
                    l0c_buf,
                    0,
                    n_actual,
                    m_actual,
                    dst_stride,
                    src_stride,
                    0,
                    F322F16,
                    0,
                    false,
                    true
                );
            }
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void InitFlags()
    {
        WaitEvent(AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID);
    }

    __aicore__ inline void Endflags()
    {
    }

     __aicore__ inline void Process()
    {
        // AIC matmul func, waits for AIV to complete [AllReduce & Add & RMSNorm].
        InitFlags();
        uint32_t m = this->m;
        uint32_t k = this->k;
        uint32_t n = this->n;
        gmB.SetGlobalBuffer(gm_b_src, k * n);

        using LayoutA = layout::RowMajor;
        using LayoutB = layout::ColumnMajor;
        using LayoutC = layout::RowMajor;
        LayoutB layoutB {(layout::ColumnMajor::Index)k, (layout::ColumnMajor::Index)n};

        using L1TileShape = GemmShape<MM_L1_TILE_SHAPE_M, MM_L1_TILE_SHAPE_N, MM_L1_TILE_SHAPE_K>;
        using L0TileShape = GemmShape<MM_L0_TILE_SHAPE_M, MM_L0_TILE_SHAPE_N, MM_L0_TILE_SHAPE_K>;
        using AType = Gemm::GemmType<MmadDtype, LayoutA>;
        using BType = Gemm::GemmType<MmadDtype, LayoutB>;
        using CType = AType;
        constexpr bool ENABLE_UNIT_FLAG = true;
        using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
        using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        GemmCoord blockShape = L1TileShape::ToCoord();

        BlockMmad blockMmad(resource);
        int mPerSplit = this->m0 * this->swizzl_count;
        int mAvg = mPerSplit;
        int splitM = AscendC::DivCeil(m, mPerSplit);
        int flag_idx = 0;
        icache_preload(8); // 8 corresponding to 16k
        for (int splitIndex = 0; splitIndex < splitM; ++splitIndex) {
            uint32_t mStart = splitIndex * mAvg;
            uint32_t mActual = mAvg > (m - mStart) ? m - mStart:mAvg;
            flag_idx = splitIndex % PIPE_DEPTH;
            if (splitIndex >= PIPE_DEPTH) {
                WaitEvent(flag_idx);
            }

            __gm__ MmadDtype *gm_a_src_tmp = reinterpret_cast<__gm__ MmadDtype *>(gm_a_src) + mStart * k;
            __gm__ MmadDtype *gm_c_src_tmp = reinterpret_cast<__gm__ MmadDtype *>(gm_peer_mem) + mStart * n;
            gmA.SetGlobalBuffer(gm_a_src_tmp, mActual*k);
            gmC.SetGlobalBuffer(gm_c_src_tmp, mActual*n);

            GemmCoord splitShape{mActual, n, k};
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>; // SwizzleOffset=3
            BlockScheduler splitScheduler(splitShape, blockShape.GetCoordMN());
            uint32_t coreLoops = splitScheduler.GetCoreLoops();

            LayoutA layoutA{mActual, k};
            LayoutC layoutC{mActual, n};

            for (uint32_t loopIdx = core_idx; loopIdx < coreLoops; loopIdx += core_num) {
                GemmCoord blockCoord = splitScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape  = splitScheduler.GetActualBlockShape(blockCoord);
                GemmCoord offsetCoord = blockCoord * blockShape;

                MatrixCoord offsetA = offsetCoord.GetCoordMK();
                MatrixCoord offsetB = offsetCoord.GetCoordKN();
                MatrixCoord offsetC = offsetCoord.GetCoordMN();

                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                blockMmad (gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC, actualBlockShape);
            }

            FFTSCrossCoreSync<PIPE_FIX>(FFTS_SYNC_AICORE_GROUP_MODE, flag_idx);
        }

        Endflags();
        PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::GlobalTensor<MmadDtype> gmA;
    AscendC::GlobalTensor<MmadDtype> gmB;
    AscendC::GlobalTensor<MmadDtype> gmC;
    __gm__ MmadDtype *gm_a_src{nullptr};
    __gm__ MmadDtype *gm_b_src{nullptr};

    __gm__ OutDtype *gm_c{nullptr};
    __gm__ OutDtype *gm_peer_mem{nullptr};
    __gm__ int64_t *gm_dequant_scale{nullptr};
    __gm__ int32_t *gm_format_dequant_offset{nullptr};
    __gm__ int32_t *gm_accum{nullptr};
    __gm__ uint8_t *gm_workspace_src;

    __cbuf__ MmadDtype *l1_base_a = reinterpret_cast<__cbuf__ MmadDtype *>((uintptr_t) SCALE_L1_SIZE_A);
    __cbuf__ MmadDtype *l1_base_b = reinterpret_cast<__cbuf__ MmadDtype *>((uintptr_t) SCALE_L1_SIZE_B);

    __ca__ MmadDtype *l0a_base = reinterpret_cast<__ca__ MmadDtype *>((uintptr_t) 0);
    __cb__ MmadDtype *l0b_base = reinterpret_cast<__cb__ MmadDtype *>((uintptr_t) 0);

    __cc__ T_ACCUM *l0c_buf = reinterpret_cast<__cc__ T_ACCUM *>((uintptr_t) 0);

    __cbuf__ int64_t *scale_l1 = reinterpret_cast<__cbuf__ int64_t *>((uintptr_t) 0);
    __fbuf__ int64_t *scale_FB = (__fbuf__  int64_t *)(0);

    __cbuf__ int32_t *bias_l1  = reinterpret_cast<__cbuf__ int32_t *>((uintptr_t)0);
    uint16_t bias_bt = 0;
    bool has_offset{false};

    int32_t core_num;

    int32_t batch_size;
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t m_align;
    int32_t k_align;
    int32_t n_align;
    int32_t k_align16;
    int32_t n_align16;
    int32_t m0;
    int32_t k0;
    int32_t n0;

    int32_t m_loop;
    int32_t n_loop;
    int32_t k_loop;
    int32_t core_loop;
    int32_t core_idx;
    int32_t ping_flag;
    int32_t block_size;
    int32_t cube_matrix_size;

    int32_t aligned_a;
    int32_t aligned_b;

    int32_t swizzl_count;
    int32_t swizzl_direct;

    int32_t rank;
    int32_t rank_size;

    int32_t withSerialMode;

    int32_t ag_dim;
    int32_t rs_dim;
    bool inner_dim_is_Ag{false};
    int32_t ag_rank_idx;
    int32_t rs_rank_idx;
    bool weight_nz{false};

    bool is_91093{false};
    QuantGranularity dequant_granularity;

    bool is_int8;
    bool trans_a;
    bool trans_b;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};

#endif // MATMUL_ALLREDUCE_ADD_RMSNORM_AIC_KERNEL_H
