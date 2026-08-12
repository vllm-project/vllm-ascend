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
 * \file quant_matmul_mx_kernel.h
 * \brief mxfp8 场景 quantMatmul 实现。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "lib/hccl/hccl.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler, bool HcommPureCMode>
#define QBMM_MX_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler, HcommPureCMode

using namespace AscendC;
using AscendC::Te::Get;

/**
 * @brief SWAT MX 量化矩阵乘内核实现
 * 该类负责具体的矩阵乘块调度和计算，支持本地(LOCAL)和远程(REMOTE)两种切片模式
 */

template <class ProblemShape, class BlockMmad, class BlockScheduler, bool HcommPureCMode = false>
class QuantMatmulMxKernel {
public:
    __aicore__ inline QuantMatmulMxKernel()
    {}
    __aicore__ inline ~QuantMatmulMxKernel()
    {}

    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<SCALE_C0>>>;
    /**
     * @brief 算子模式：NORMAL (常规), LOCAL (仅本地计算), REMOTE (仅远程同步数据后的计算),
     *                 DEFERRED_SYNC (per-tile 本地先算驻留 L0C → wait_flag → 远程累加 → 单次 fixpipe)
     */
    enum class MatmulMode : uint32_t {
        REMOTE = 0,
        LOCAL = 1,
        DEFERRED_SYNC = 2
    };
    /**
     * @brief Tiling 配置
     */
    struct QBMMTiling {
        enum BiasMode : uint32_t {
            BIAS_DISABLED = 0,
            BIAS_ENABLED = 1
        };
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
        uint32_t isBias;
    };

    /**
     * @brief 本地 Rank 相关参数
     */
    struct LocalParams {
        uint32_t rankId;
        uint32_t rankSize;
        uint64_t originalM;    // 单卡负责的总 M 行数
        GM_ADDR  localAGmAddr;
        GM_ADDR  localScaleAGmAddr;
        uint32_t localMatmul; // 1 means local matmul
        uint32_t splitKNum;
        MatmulMode matmulMode; // 1: LOCAL, 2: REMOTE, 3: DEFERRED_SYNC
        uint32_t headTileSize;
        void* hcclPtr{nullptr};
        AscendC::HcclHandle dataHeadHandle{0};
        AscendC::HcclHandle dataTailHandle{0};
        uint32_t headTileCnt{0};
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        LocalParams localParams;
    };

public:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ResetGmAddr(const Params &params);
    __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler& bs,
                            uint64_t restBatch, bool isTailRound);
    __aicore__ inline uint32_t CalcDependTileIdx(int64_t mPos, uint32_t headTileSize, uint32_t totalTiles) const;
    __aicore__ inline void WaitCommTile(const Params &params, uint32_t tileIdx);

    template <typename TensorB, typename TensorScaleB, typename TensorC>
    __aicore__ inline void SetL2Cache(
        const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
        TensorScaleB &gmScaleB, TensorC& gmC);

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(
        const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB);

private:
    BlockMmad mmadOp_;

    __gm__ AType *aGmAddr_;               // 远程数据基址（通信缓冲区）
    __gm__ AType *localAGmAddr_;          // 本地数据基址
    __gm__ BType *bGmAddr_;
    __gm__ CType *cGmAddr_;               // 输出基址（已根据流水步偏移）
    __gm__ BiasType *biasGmAddr_ = nullptr;
    __gm__ ::fp8_e8m0_t *scaleAGmAddr_;     // 远程 Scale 基址
    __gm__ ::fp8_e8m0_t *localScaleAGmAddr_;
    __gm__ ::fp8_e8m0_t *scaleBGmAddr_;

    bool isAtomicAdd_{false}; // REMOTE 模式下需要原子累加到 C
    bool isBias_{false};
    bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Run(const Params &params)
{
    Init(params);

    if (isAtomicAdd_) {
        AscendC::SetAtomicAdd<CType>(); // 开启原子累加
    }
    BlockScheduler bs(params.problemShape, params.schParams);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, params.qbmmParams.dbL0C > 1,
        params.localParams.splitKNum);

    ProcessSingleBatch(params, bs, 0, true);

    if (isAtomicAdd_) {
        AscendC::SetAtomicNone();
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorScaleB>
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::SetScaleL2Cache(
    const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
{
    if (Te::Get<MNK_B>(problemShape) != 1) {
        return;
    }
    if constexpr (transB) {
        const int64_t scaleKRowBytes =
            Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0 &&
            (scaleKL1RowBytes & kCacheLineAlignMask) == 0;
        gmScaleB.SetL2CacheHint(
            scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const int64_t scaleNStrideBytes = Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
        const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream = (scaleNStrideBytes & kCacheLineAlignMask) == 0 &&
            (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
        gmScaleB.SetL2CacheHint(
            scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::SetL2Cache(
    const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
    TensorScaleB &gmScaleB, TensorC& gmC)
{
    if (isAtomicAdd_) {
        gmC.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    }

    const bool fullMTile = curBaseM >= Te::Get<MNK_M>(problemShape);
    if (!fullMTile) {
        return;
    }

    SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        if constexpr (transB) {
            bool bAlignForL2Stream = (Te::Get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            bool bAlignForL2Stream =
                (Te::Get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 && (baseN & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Init(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED) {
        isBias_ = true;
    }

    if ((params.localParams.matmulMode == MatmulMode::REMOTE) && (params.localParams.localMatmul == 1)) {
        isAtomicAdd_ = true; // REMOTE 模式需要原子累加
    }
    needUpdateTail_ = false;
    ResetGmAddr(params);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ResetGmAddr(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    localAGmAddr_ = reinterpret_cast<__gm__ AType*>(params.localParams.localAGmAddr);
    localScaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.localParams.localScaleAGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline uint32_t QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::CalcDependTileIdx(
    int64_t mPos, uint32_t headTileSize, uint32_t totalTiles) const
{
    uint32_t tileIdx = static_cast<uint32_t>(mPos / headTileSize);
    if (tileIdx >= totalTiles) {
        tileIdx = totalTiles - 1;
    }
    return tileIdx;
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::WaitCommTile(
    const Params &params, uint32_t tileIdx)
{
    if constexpr (HcommPureCMode) {
        auto* hccl = static_cast<AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU>*>(
            params.localParams.hcclPtr);
        if (tileIdx < params.localParams.headTileCnt) {
            hccl->Wait(params.localParams.dataHeadHandle);
        } else {
            hccl->Wait(params.localParams.dataTailHandle);
        }
    } else {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernel<QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ProcessSingleBatch(
    const Params &params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
    auto rankId = params.localParams.rankId;
    auto rankSize = params.localParams.rankSize;
    auto oriM = params.localParams.originalM;
    bool localFirst = ((params.localParams.matmulMode == MatmulMode::LOCAL) && (params.localParams.localMatmul == 1));
    // DEFERRED_SYNC: per-tile 内 local 先算驻留 L0C → wait_flag → remote 累加 → 单次 fixpipe。
    // wait_flag 必须由 kernel 内部发出（在 self rank 的 mmad 之后、其它 rank 的 mmad 之前）。
    bool deferredSync = (params.localParams.matmulMode == MatmulMode::DEFERRED_SYNC);
    auto scaleKLen =
        Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;

    // 构建各 Tensor 的全局布局
    auto layoutA = MakeLayoutA{}(rankSize * params.localParams.originalM, Te::Get<MNK_K>(params.problemShape));
    auto layoutALocal = MakeLayoutA{}(rankSize * oriM, Te::Get<MNK_K>(params.problemShape));
    auto layoutScaleA = MakeLayoutScaleA{}(rankSize * oriM, scaleKLen);

    auto layoutB = MakeLayoutB{}(rankSize * Te::Get<MNK_K>(params.problemShape), Te::Get<MNK_N>(params.problemShape));
    auto layoutScaleB = MakeLayoutScaleB{}(rankSize * scaleKLen, Te::Get<MNK_N>(params.problemShape));
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
    auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

    // 创建 Tensor 句柄
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
    auto gmALocal = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(localAGmAddr_), layoutALocal); // local输入
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmScaleALocal = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(localScaleAGmAddr_), layoutScaleA);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);

    // 尾块更新逻辑
    auto& mTailTile = params.schParams.mTailTile;
    auto& nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(
        params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB, gmScaleB,
        gmC);

    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    uint32_t totalTiles = (oriM + params.localParams.headTileSize - 1) / params.localParams.headTileSize;
    uint32_t waitedMask = 0U;
    // 遍历当前块的调度任务
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        if ((Te::Get<IDX_M_TILEIDX>(singleShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(singleShape) <= 0)) {
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        // 切分输出块：地址基址已在外部按流水步偏移，此处仅按调度器位置切局部块
        auto gmBlockC =
            gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
        auto gmBlockBias =
            gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));

        if (localFirst) {
            // LOCAL 模式：计算本 Rank 的 A 和 本 Rank 的 B 对应部分
            auto actualMPos = rankId * oriM + mPos;
            auto gmBlockA =
                gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                    AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockScaleA = gmScaleALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                    AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rankId * Get<MNK_K>(params.problemShape), nPos),
                AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
            auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(rankId * scaleKLen, nPos),
                AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));

            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, 0);
        } else if (deferredSync) {
            // DEFERRED_SYNC 模式：
            int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);
            uint32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, params.localParams.headTileSize, totalTiles);
            // Phase 1: 本 rank 的 local A × 本 rank 的 B 段 → L0C（reset，remoteRankCnt=0）
            //          此处读 GM 的 localAGmAddr_，不依赖通信，可与 AIV 的 UDMA put 并行。
            auto selfMPos = rankId * oriM + mPos;
            auto gmBlockA_self =
                gmALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos),
                    AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockScaleA_self = gmScaleALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos),
                    AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
            auto gmBlockB_self = gmB.Slice(AscendC::Te::MakeCoord(rankId * Get<MNK_K>(params.problemShape), nPos),
                AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
            auto gmBlockScaleB_self = gmScaleB.Slice(AscendC::Te::MakeCoord(rankId * scaleKLen, nPos),
                AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));
            mmadOp_(gmBlockA_self, gmBlockB_self, gmBlockScaleA_self, gmBlockScaleB_self,
                    gmBlockBias, gmBlockC, singleShape, 0);

            // Phase 2: 在 self rank mmad 之后 wait，阻塞后续 shmem 读（去重：同一 tile 只 wait 一次）
            if ((waitedMask & (1U << dependTileIdx)) == 0U) {
                WaitCommTile(params, dependTileIdx);
                waitedMask |= (1U << dependTileIdx);
            }

            // Phase 3: 遍历其它 rank，在 L0C 上累加（最后一个 rank 触发 fixpipe）
            uint32_t remoteRankCnt = 1;
            for (uint64_t rank = 0; rank < rankSize; rank++) {
                if (rank == rankId) continue;
                auto actualMPos = rank * oriM + mPos;
                auto gmBlockA_remote =
                    gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                auto gmBlockScaleA_remote = gmScaleA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
                auto gmBlockB_r = gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                    AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
                auto gmBlockScaleB_r = gmScaleB.Slice(AscendC::Te::MakeCoord(rank * scaleKLen, nPos),
                    AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));
                mmadOp_(gmBlockA_remote, gmBlockB_r, gmBlockScaleA_remote, gmBlockScaleB_r,
                        gmBlockBias, gmBlockC, singleShape, remoteRankCnt);
                remoteRankCnt++;
            }
        } else {
            // REMOTE 模式：低精度模式下遍历除本 Rank 外的所有其他卡发送过来的数据
            auto remoteRankCnt = 0UL;
            int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);
            uint32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, params.localParams.headTileSize, totalTiles);
            // 等待当前 block 依赖的通信 tile 完成（去重：同一 tile 只 wait 一次）
            if ((waitedMask & (1U << dependTileIdx)) == 0U) {
                WaitCommTile(params, dependTileIdx);
                waitedMask |= (1U << dependTileIdx);
            }
            for (uint64_t rank = 0; rank < rankSize; rank++) {
                auto actualMPos = rank * oriM + mPos;
                // 从通信buffer上切片
                auto gmBlockA =
                    gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                        AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));

                if (rank == rankId) {
                    if (params.localParams.localMatmul == 1) {
                        continue;
                    } else {
                        gmBlockA = gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                        gmBlockScaleA = gmScaleALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
                    }
                }

                auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                    AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
                auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(rank * scaleKLen, nPos),
                    AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));

                // L0C上累加
                mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC,
                    singleShape, remoteRankCnt);
                remoteRankCnt++;
            }
        }
    }
    if (!localFirst) {
        for (uint32_t t = 0; t < totalTiles; ++t) {
            if ((waitedMask & (1U << t)) == 0U) {
                WaitCommTile(params, t);
                waitedMask |= (1U << t);
            }
        }
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
