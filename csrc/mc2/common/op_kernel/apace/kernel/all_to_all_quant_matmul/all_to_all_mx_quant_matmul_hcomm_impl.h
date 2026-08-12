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
 * \file all_to_all_mx_matmul_hcomm_impl.h
 * \brief AlltoAll MX Quant Matmul — 通信+计算融合实现（纯C核通信）
 *
 * Init():
 *   AIC: 批量下发 AlltoAll<true> 通信任务（scale + dataHead + dataTail）
 *   AIV: 直接返回，不参与通信
 *
 * Run():
 *   AIC: hccl_.Wait(scaleHandle_) -> MatmulProcess() -> hccl_.Finalize()
 *        kernel 内逐 tile 通过 hccl->Wait(dataHeadHandle/dataTailHandle) 等待通信完成
 *
 * AIC side: 通信下发 + 计算 + per-tile wait 均在 AIC 核内完成，无 AIV↔AIC 跨核 flag 同步.
 *
 * \note 本实现仅启动 AIC 核（cube-only）。若需启动 AIV 核，须为 Init/Run 中的通信下发、
 *        Wait、Finalize 及 SyncAll 增加 AIC 守卫（if ASCEND_IS_AIC），否则 AIV 误参与 HCCL
 *        调用将导致 Prepare/Wait 失配或死锁。
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "quant_matmul_mx_kernel.h"
#include "include/tensor_api/tensor.h"

namespace Apace {
template <typename T>
constexpr uint64_t GetX1HcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
class AllToAllMxQuantMatmulHcommImpl {
public:
    __aicore__ inline AllToAllMxQuantMatmulHcommImpl(AlltoAllMatmulTilingDataType *tilingData,
                                                AscendC::TPipe *tPipe)
        : tilingData_(tilingData), tPipe_(tPipe) {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Blaze::Gemm::Kernel::QuantMatmulMxKernel<
        ProblemShape, BlockMmad, BlockScheduler, true>; // HcommPureCMode = true，hcomm纯C通信

    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    static constexpr AscendC::HcclServerType ServerType = AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU;
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHeadHandle_{0};
    AscendC::HcclHandle dataTailHandle_{0};

    AlltoAllMatmulTilingDataType *tilingData_;
    AscendC::TPipe *tPipe_;

    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR commX1ScaleGM1_;
    GM_ADDR commOutGM_;

    static constexpr uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t x1HcclDataType_ = GetX1HcclDataType<X1Type>();
    uint32_t rankId_{0};
    uint32_t rankDim_{0};
    uint64_t splitAxisSize_{0};

    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    __aicore__ inline void SetupParams(Params &out);
    __aicore__ inline void MatmulProcess();
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1,
                          AlltoAllMatmulTilingDataType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM)
{
    auto &&commTiling = tilingData_->commTilingData;
    splitAxisSize_ = commTiling.splitAxisTileSize * commTiling.splitAxisTileCnt +
                    commTiling.splitAxisTailSize * commTiling.splitAxisTailCnt;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    commX1ScaleGM1_ = workspaceGM;
    uint64_t x1ScaleLen = Blaze::Gemm::CeilDiv(commTiling.nonSplitAxisSize, MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE *
                            static_cast<uint64_t>(splitAxisSize_) * sizeof(AscendC::fp8_e8m0_t);
    x1ScaleLen = Blaze::Gemm::CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
    commOutGM_ = all2all_out;
    if (all2all_out == nullptr) {
        commOutGM_ = workspaceGM + x1ScaleLen;
    }
    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(AlltoAllMatmulTilingDataType, mc2CcTiling)));
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();

    uint64_t rankForComm = commTiling.nonSplitAxisSize / rankDim_;
    uint32_t scaleKPerRank =
        static_cast<uint32_t>(Blaze::Gemm::CeilDiv(rankForComm, MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE);

    uint64_t dataStrideCount = static_cast<uint64_t>(splitAxisSize_) * rankForComm;
    uint64_t scaleSendCount = static_cast<uint64_t>(splitAxisSize_) * scaleKPerRank;
    scaleHandle_ = hccl_.template AlltoAll<true>(
        x1Scale_, commX1ScaleGM1_,
        scaleSendCount, static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_),
        scaleSendCount, 1);

    if (commTiling.splitAxisTileCnt > 0) {
        uint64_t headSendCount = static_cast<uint64_t>(commTiling.splitAxisTileSize) * rankForComm;
        dataHeadHandle_ = hccl_.template AlltoAll<true>(
            x1_, commOutGM_,
            headSendCount, static_cast<AscendC::HcclDataType>(x1HcclDataType_),
            dataStrideCount, static_cast<uint8_t>(commTiling.splitAxisTileCnt));
    }

    if (commTiling.splitAxisTailCnt > 0) {
        uint64_t headOffset = static_cast<uint64_t>(commTiling.splitAxisTileCnt) *
            commTiling.splitAxisTileSize * rankForComm * sizeof(X1Type);
        uint64_t tailSendCount = static_cast<uint64_t>(commTiling.splitAxisTailSize) * rankForComm;
        dataTailHandle_ = hccl_.template AlltoAll<true>(
            x1_ + headOffset, commOutGM_ + headOffset,
            tailSendCount, static_cast<AscendC::HcclDataType>(x1HcclDataType_),
            dataStrideCount, static_cast<uint8_t>(commTiling.splitAxisTailCnt));
    }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1,
                          AlltoAllMatmulTilingDataType, IsMxFp4>::Run()
{
    hccl_.Wait(scaleHandle_);
    MatmulProcess();
    AscendC::SyncAll();
    hccl_.Finalize();
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1,
                          AlltoAllMatmulTilingDataType, IsMxFp4>::SetupParams(Params &out)
{
    auto &&commTiling = tilingData_->commTilingData;
    const auto &mmTile = tilingData_->tileQbmmTilingData;

    ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};

    BlockMmadParams mmadParams;
    mmadParams.aGmAddr = commOutGM_;
    mmadParams.scaleAGmAddr = commX1ScaleGM1_;
    mmadParams.bGmAddr = x2_;
    mmadParams.scaleBGmAddr = x2Scale_;
    mmadParams.cGmAddr = y_;
    mmadParams.biasGmAddr = bias_;

    LocalParams localParams{rankId_, rankDim_, splitAxisSize_, x1_, x1Scale_,
                            tilingData_->localMatmul, rankDim_, MatmulMode::REMOTE,
                            static_cast<uint32_t>(commTiling.splitAxisTileSize),
                            &hccl_, dataHeadHandle_, dataTailHandle_,
                            static_cast<uint32_t>(commTiling.splitAxisTileCnt)};

    L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1,
                        mmTile.nBufferNum};

    BlockSchedulerParams schedulerParams{
        mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};

    QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c, false};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1,
                          AlltoAllMatmulTilingDataType, IsMxFp4>::MatmulProcess()
{
    Params params;
    SetupParams(params);
    quantMatmulKernelImpl_(params);
}

} // namespace Apace
