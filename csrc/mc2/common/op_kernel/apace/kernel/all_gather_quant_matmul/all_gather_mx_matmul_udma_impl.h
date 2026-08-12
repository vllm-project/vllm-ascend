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
 * \file all_gather_mx_matmul_udma_impl.h
 * \brief AllGatherQuantMatmul 算子实现，基于 FragmentTensor + 通算解耦（UDMA 通信）。
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_tiling_data.h"
#include "apace/kernel/all_gather_quant_matmul/qmm_mx_kernel_ag_udma.h"
#include "adv_api/hcomm/hcomm.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "apace/block/aiv_comm/collective_comm_api.h"
#include "apace/block/aiv_comm/barrier/barrier_ubmem.h"
#include "apace/tiling/comm_tiling_data.h"
#include "include/tensor_api/tensor.h"

namespace AllGatherQuantMatmulImpl {

using namespace AscendC;
using namespace Apace::AivComm;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::DNExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

template <typename AType, typename BType, typename CType>
class AllGatherMxMatmulUdmaImpl {
public:
    __aicore__ inline AllGatherMxMatmulUdmaImpl() {}
    __aicore__ inline void Init(__gm__ CommContext *hcommCtx,
      GM_ADDR aGM, GM_ADDR aScaleGM,
      GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
      const AllGatherMxMatmulUdmaTilingData *tilingData);
    __aicore__ inline void Process();

    using QuantMatmulKernelImpl = QmmMxKernelAgUdma<AType, BType, CType>;
    using KernelParams = typename QuantMatmulKernelImpl::Params;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using FragmentParams = typename QuantMatmulKernelImpl::FragmentParams;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    __aicore__ inline void InitBaseParams(const AllGatherMxMatmulUdmaTilingData *td);

    __aicore__ inline void AllGatherProcess();
    __aicore__ inline void MatmulProcess();

    __aicore__ inline GM_ADDR GetWinDataRegionBase();
    __aicore__ inline GM_ADDR GetWinScaleRegionBase();

    // 通信
    using AllGatherCommData = Apace::AivComm::CollectiveComm<
      Apace::AivComm::CommCollectiveOp::AllGather,
      Apace::AivComm::CommMode::PUT, AType, TeamBarrier>;
    using AllGatherCommScale = Apace::AivComm::CollectiveComm<
      Apace::AivComm::CommCollectiveOp::AllGather,
      Apace::AivComm::CommMode::PUT, AscendC::fp8_e8m0_t, TeamBarrier>;
    AllGatherCommData all_gather_data_;
    AllGatherCommScale all_gather_scale_;
    CommTilingData commTilingData_{};
    CommTilingData commTilingScale_{};
    TeamBarrier teamBarrier_;

    static constexpr uint32_t kPaddingLength = 16;

    __gm__ CommContext *hcommCtx_{};
    __gm__ CommUdmaContext* udmaCtx_{};
    __gm__ CommUbmemContext* ubmemCtx_{};

    GM_ADDR aGM_{};
    GM_ADDR aScaleGM_{};
    GM_ADDR bGM_{};
    GM_ADDR bScaleGM_{};
    GM_ADDR cGM_{};

    uint32_t rankId_{};
    uint32_t rankSize_{};
    uint32_t m_{};
    uint32_t k_{};
    uint32_t n_{};
    uint32_t tileCnt_{};
    uint32_t tileM_{};
    uint32_t tailCnt_{};
    uint32_t tailM_{};
    uint32_t paddedTailM_{};
    uint32_t commTurn_{};
    uint64_t headRows_{};
    uint64_t scaleKLen_{};
    uint64_t dataBytesPerMRow_{};
    uint64_t scaleBytesPerMRow_{};
    uint64_t cBytesPerM_{};
    uint64_t scaleKGroups_{};
    uint64_t dataRegionBytes_{};
    uint64_t scaleRegionBytes_{};

    const AllGatherMxMatmulUdmaTilingData *tilingData_{};
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUdmaImpl<AType, BType, CType>::Init(
  __gm__ CommContext *hcommCtx,
  GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
  const AllGatherMxMatmulUdmaTilingData *tilingData)
{
    tilingData_ = tilingData;
    hcommCtx_ = hcommCtx;
    aGM_ = aGM;
    aScaleGM_ = aScaleGM;
    bGM_ = bGM;
    bScaleGM_ = bScaleGM;
    cGM_ = cGM;

    InitBaseParams(tilingData);

    udmaCtx_ = &(hcommCtx_->udmaCtx);
    ubmemCtx_ = &(hcommCtx_->ubmemCtx);
    rankId_ = udmaCtx_->rankId;
    rankSize_ = udmaCtx_->rankSize;

    // Rank-major 布局：每个 rank 占完整的 M 行。data + scale 两段连续存放。
    //   Win buffer 内存布局：
    // ┌──────────────────────────────────────────────────┐
    // │  rank0 M行 data │ ... │ rankR-1 M行 data │          ← data 段
    // │  rank0 M行 scale│ ... │ rankR-1 M行 scale│         ← scale 段 (offset = dataRegionBytes_)
    // └──────────────────────────────────────────────────┘
    dataRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * dataBytesPerMRow_;
    scaleRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * scaleBytesPerMRow_;

    // 静态 tensor 替代 tpipe buffer
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.Get(), ubmemCtx_, rankSize_,
                      static_cast<uint32_t>(GetBlockIdx()));

    commTilingData_.splitAxisTileSize = tileM_;   // 每个 tile 搬 tileM 行
    commTilingData_.splitAxisTileCnt = tileCnt_;   // head 段 tile 数量
    commTilingData_.splitAxisTailSize = tailM_;   // tail 行数
    commTilingData_.splitAxisTailCnt = tailCnt_;
    commTilingData_.nonSplitAxisSize = k_;    // 非切分轴k

    commTilingScale_.splitAxisTileSize = tileM_;
    commTilingScale_.splitAxisTileCnt = tileCnt_;
    commTilingScale_.splitAxisTailSize = tailM_;
    commTilingScale_.splitAxisTailCnt = tailCnt_;
    scaleKLen_ = scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
    commTilingScale_.nonSplitAxisSize = scaleKLen_;
    all_gather_data_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, commTilingData_,
                          aGM_, commBuf.Get(), rankSize_,
                          static_cast<uint32_t>(GetBlockIdx()));
    all_gather_scale_.Init(udmaCtx_, teamBarrier_, commTilingScale_,
                          aScaleGM_, commBuf.Get(), rankSize_,
                          static_cast<uint32_t>(GetBlockIdx()), dataRegionBytes_);
    AscendC::SyncAll<true>();
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUdmaImpl<AType, BType, CType>::InitBaseParams(
  const AllGatherMxMatmulUdmaTilingData *td)
{
    const auto &ct = td->commTile;
    tileCnt_ = static_cast<uint32_t>(ct.splitAxisTileCnt);
    tileM_ = static_cast<uint32_t>(ct.splitAxisTileSize);
    tailCnt_ = static_cast<uint32_t>(ct.splitAxisTailCnt);
    tailM_ = static_cast<uint32_t>(ct.splitAxisTailSize);
    k_ = td->mmTile.k;
    n_ = td->mmTile.n;
    m_ = static_cast<uint32_t>(ct.splitAxisTileSize * ct.splitAxisTileCnt +
                                ct.splitAxisTailSize * ct.splitAxisTailCnt);
    commTurn_ = tileCnt_ + tailCnt_;
    paddedTailM_ = (tailM_ > 0) ? ((tailM_ + kPaddingLength - 1) / kPaddingLength * kPaddingLength) : 0U;
    headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

    scaleKGroups_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(k_), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    dataBytesPerMRow_ = static_cast<uint64_t>(k_) * sizeof(AType);
    scaleBytesPerMRow_ = scaleKGroups_ * static_cast<uint64_t>(
      Blaze::Gemm::MXFP_MULTI_BASE_SIZE) * sizeof(AscendC::fp8_e8m0_t);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUdmaImpl<AType, BType, CType>::Process()
{
    if ASCEND_IS_AIV {
        AllGatherProcess();
    }
    if ASCEND_IS_AIC {
        MatmulProcess();
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUdmaImpl<AType, BType, CType>::MatmulProcess()
{
    KernelParams params;
    params.mmTile = &tilingData_->mmTile;
    params.qbmmParams = {tilingData_->mmTile.baseM, tilingData_->mmTile.baseN,
                    tilingData_->mmTile.baseK, tilingData_->mmTile.dbL0c};
    params.fragParams = {tileCnt_, tileM_, tailCnt_, tailM_, paddedTailM_, commTurn_,
                    headRows_, rankId_, rankSize_, m_,
                    static_cast<uint64_t>(k_), static_cast<uint64_t>(n_), scaleKLen_};
    params.aGM = aGM_;
    params.aScaleGM = aScaleGM_;
    params.bGM = bGM_;
    params.bScaleGM = bScaleGM_;
    params.cGM = cGM_;
    params.winDataBase = GetWinDataRegionBase();
    params.winScaleBase = GetWinScaleRegionBase();
    params.dataBytesPerMRow = dataBytesPerMRow_;
    params.scaleBytesPerMRow = scaleBytesPerMRow_;
    params.cBytesPerM = cBytesPerM_;
    quantMatmulKernelImpl_(params);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherMxMatmulUdmaImpl<AType, BType, CType>::GetWinDataRegionBase()
{
    return reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[rankId_]);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherMxMatmulUdmaImpl<AType, BType, CType>::GetWinScaleRegionBase()
{
    return GetWinDataRegionBase() + dataRegionBytes_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUdmaImpl<AType, BType, CType>::AllGatherProcess()
{
    // 预触发 dependId=0：自身数据始终就绪，AIC 可直接消费。
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

    for (uint32_t round = 0; round < commTurn_; ++round) {
        if (static_cast<uint32_t>(GetBlockIdx()) < rankSize_) {
            all_gather_scale_.Commit();
            all_gather_data_.Commit();
            all_gather_data_.template Wait<BARRIER_DEVICE>();
        }
        AscendC::SyncAll<true>();
        // round 0 对应 dependId=1，AIC 侧按 dependId 等待对应的远端 round 数据。
        CrossCoreSetFlag<0x2, PIPE_MTE3>(round + 1);
    }
    all_gather_scale_.Finalize();
    all_gather_data_.Finalize();
}

} // namespace AllGatherQuantMatmulImpl

__global__ __aicore__ void AllGatherQuantMatmulKernel(
  __gm__ Apace::AivComm::CommContext *hcommCtx,
  GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
  AllGatherMxMatmulUdmaTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AllGatherQuantMatmulImpl::AllGatherMxMatmulUdmaImpl<
      fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl;
    impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, &tilingData);
    impl.Process();
}
