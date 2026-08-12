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
 * \file all_to_all_mx_matmul_udma_impl.h
 * \brief AlltoAll MX Quant Matmul — 通信+计算融合实现
 *
 * Run():
 *   AIV: RunAllToAll()  根据通信轮次下发通信任务和通知AIC通信已完成
 *   AIC: RunMatmul()    等待通信完成后开始cube计算
 */


#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "quant_matmul_mx_kernel.h"
#include "all_to_all_matmul_tiling_data.h"

#include "apace/block/aiv_comm/collective_comm_api.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/block/aiv_comm/barrier/barrier_ubmem.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using AscendC::Te::Get;
using namespace Apace::AivComm;

// 定义问题形状：[M, N, K, Batch]
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

/**
 * @brief All2All-Matmul 核心实现类 (Hcomm GET 版本，计算通信融合)
 * 实现思路：
 * 1. AIC 负责本地计算 和远程数据计算。
 * 2. AIV 负责背景通信，通过 Hcomm GET 接口从其他卡拉取 A 和 ScaleA 数据。
 * 3. 通过流水线掩盖通信开销。
 * 4. 计算与通信逻辑合并于此类，便于精细化流水线控制。
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
class AllToAllMxQuantMatmulUdmaImpl {
public:
    __aicore__ inline AllToAllMxQuantMatmulUdmaImpl() {};
    __aicore__ inline ~AllToAllMxQuantMatmulUdmaImpl() {}

    /**
     * @brief 初始化算子状态和参数
     */
    __aicore__ inline void Init(__gm__ CommContext *hcommCtx,
                    GM_ADDR aGM, GM_ADDR scaleAGM,
                    GM_ADDR bGM, GM_ADDR scaleBGM,
                    GM_ADDR cGM,
                    const allToAllMatmulTilingData *tilingData);
    /**
     * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
     */
    __aicore__ inline void Run();

    using TypeA = AType;
    using TypeB = BType;
    using TypeC = CType;
    using TypeScaleA = ::fp8_e8m0_t;
    using TypeScaleB = ::fp8_e8m0_t;
    static constexpr int32_t SCALE_C0 = 2;

    // Layout 定义
    using LayoutA =
        typename AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB =
        typename AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransA, AscendC::Te::ScaleADNLayoutPtn, AscendC::Te::ScaleANDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<
        AscendC::Std::conditional_t<TransB, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>,
        AscendC::Std::Int<SCALE_C0>>;

    // 组件定义
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape,
        NONE_FULL_LOAD_MODE, LayoutA, LayoutB, TypeA>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    // ---------------- 通信相关组件与参数 ----------------
    __gm__ CommUbmemContext* syncBuffer_{nullptr};
    __gm__ CommUdmaContext* udmaCtx_{nullptr};
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, AType, TeamBarrier> allToAllA_;
    CollectiveComm<
        CommCollectiveOp::AllToAll, CommMode::PUT, TypeScaleA, TeamBarrier> allToAllScaleA_;
    TeamBarrier teamBarrier_;

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr}; // 通信窗口地址
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR cGm{nullptr};

        uint32_t rankId{0};
        int32_t  commTurn{0};     // 总流水步数
        uint32_t rankSize{0};    // 卡数
        uint64_t axisM{0};     // M 轴总大小
        uint64_t axisKa{0};    // K 轴大小
        uint64_t headMSize{0};   // 长块 M 大小
        uint64_t scaleKaSize{0};   // Scale 的 K 轴字节长度
        uint64_t rankDataBytes{0};
    } baseParams_;

    const allToAllMatmulTilingData* tilingData_{nullptr};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const allToAllMatmulTilingData *tilingData);
    __aicore__ inline void SetupParams(const QuantMatmulTilingData* mmTile,
                        Params& out, MatmulMode matmulMode);

    __aicore__ inline void RunAllToAll();  // AIV 通信任务
    __aicore__ inline void RunLocalMatmul(); // AIC 本地计算任务
    __aicore__ inline void RunMatmul();    // AIC 远程计算任务
};

// =================================================================================
// 公有方法实现
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR cGM,
    const allToAllMatmulTilingData *tilingData)
{
    tilingData_ = tilingData;
    baseParams_.aGm = aGM;
    baseParams_.scaleAGm = scaleAGM;
    baseParams_.bGm = bGM;
    baseParams_.scaleBGm = scaleBGM;
    baseParams_.cGm = cGM;

    syncBuffer_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);
    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;

    InitBaseParams(tilingData);

    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset); // 用512B
    ubOffset += COMM_WORKSPACE_SIZE;
    auto commScaleBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset); // 用512B
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

    allToAllA_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData, baseParams_.aGm,
        commBuf.Get(), baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

    allToAllScaleA_.Init(udmaCtx_, teamBarrier_, tilingData->scaleCommTilingData, baseParams_.scaleAGm,
        commScaleBuf.Get(), baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()),
        baseParams_.rankSize * baseParams_.rankDataBytes);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>::Run()
{
    if ASCEND_IS_AIV {
        RunAllToAll(); // AIV 通信
    }

    if ASCEND_IS_AIC {
        if (tilingData_->localMatmul == 1) {
            RunLocalMatmul(); // AIC local计算前置
        }
        RunMatmul();    // AIC 通信块计算（带同步等待）
    }
}

// =================================================================================
// 通信与流水线控制实现 (AIV 侧)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>::RunAllToAll()
{
    for (uint32_t tid = 0; tid < baseParams_.commTurn; ++tid) {
        // 必选保证baseParams_.rankSize <= BlockNum
        if (AscendC::GetBlockIdx() < baseParams_.rankSize) {
            allToAllScaleA_.Commit();
            allToAllA_.Commit();
            allToAllA_.template Wait<BARRIER_DEVICE>(); // scale的通信和a矩阵的通信使用同一channel，因此只需要wait一次
        }

        AscendC::SyncAll<true>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(tid);
    }

    allToAllScaleA_.Finalize();
    allToAllA_.Finalize();
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
    const allToAllMatmulTilingData *tilingData)
{
    baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
    baseParams_.commTurn = tilingData->commTilingData.splitAxisTileCnt
                        + tilingData->commTilingData.splitAxisTailCnt;

    baseParams_.axisM = tilingData->tileQbmmTilingData.m;
    baseParams_.axisKa = tilingData->tileQbmmTilingData.k;
    // 计算 MXFP8 格式下 Scale 每一行的字节数
    baseParams_.scaleKaSize =
        CeilDiv(baseParams_.axisKa, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

    // 计算地址偏移所需的参数
    baseParams_.rankDataBytes = baseParams_.axisM * baseParams_.axisKa * sizeof(AType);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>::SetupParams(
    const QuantMatmulTilingData* mmTile, Params& out, MatmulMode matmulMode)
{
    ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
    BlockMmadParams mmadParams;
    LocalParams localParams{baseParams_.rankId, baseParams_.rankSize, baseParams_.axisM, baseParams_.aGm,
        baseParams_.scaleAGm, tilingData_->localMatmul, 1UL, matmulMode, static_cast<uint32_t>(baseParams_.headMSize)};
    L1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1,
        mmTile->nBufferNum};

    if (matmulMode == MatmulMode::LOCAL) {
        mmadParams.cGmAddr = baseParams_.cGm; // 本地模式直接写入输出
    } else {
        if (tilingData_->localMatmul == 1) {
            localParams.splitKNum = baseParams_.rankSize - 1; // 低精度模式通信块计算只有rankSize-1个远程卡参与
        } else {
            // localMatmul==0（融合）和==2（DEFERRED_SYNC）都需要 splitKNum=rankSize
            localParams.splitKNum = baseParams_.rankSize;
        }
    }

    mmadParams.bGmAddr = baseParams_.bGm;
    mmadParams.scaleBGmAddr = baseParams_.scaleBGm;

    // 调度器参数
    BlockSchedulerParams schedulerParams{
        mmTile->baseM, mmTile->baseN, mmTile->mTailTile, mmTile->nTailTile,
        mmTile->mBaseTailSplitCnt, mmTile->nBaseTailSplitCnt, mmTile->mTailMain, mmTile->nTailMain};
    // 基础 Tiling 参数
    QBMMTiling qbmmParams{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c, false};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>::RunLocalMatmul()
{
    Params localParams;
    SetupParams(&tilingData_->tileQbmmTilingData, localParams, MatmulMode::LOCAL);
    quantMatmulKernelImpl_(localParams);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>::RunMatmul()
{
    Params params;
    MatmulMode mode = (tilingData_->localMatmul == 2) ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE;
    SetupParams(&tilingData_->tileQbmmTilingData, params, mode);
    params.mmadParams.aGmAddr = baseParams_.selfWinAddr;
    params.mmadParams.scaleAGmAddr = baseParams_.selfWinAddr + baseParams_.rankSize * baseParams_.rankDataBytes;
    params.mmadParams.cGmAddr = baseParams_.cGm;
    params.localParams.localAGmAddr = baseParams_.aGm;
    params.localParams.localScaleAGmAddr = baseParams_.scaleAGm;
    quantMatmulKernelImpl_(params);
}

} // namespace Apace
