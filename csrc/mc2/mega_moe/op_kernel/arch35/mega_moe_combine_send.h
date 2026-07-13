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
 * \file mega_moe_combine_send.h
 * \brief
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
#include "../../common_utils/mc2_kernel_utils.h"
#include "moe_distribute_dispatch_v2/quantize_functions.h"
#include "mega_moe_impl_base.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {
constexpr uint32_t COMBINE_SEND_ADDR = 140 * 1024U;  // triple tensor 在 UB 中的起始地址
constexpr uint32_t rankIdIndex = 0;
constexpr uint32_t tokenIdxIndex = 1;
constexpr uint32_t topkIdxIndex = 2;
constexpr uint32_t blockLenIndex = 3;
constexpr uint32_t tokenLenIndex = 4;
constexpr uint32_t tokenActualLenIndex = 5;
constexpr uint32_t flagIndex = 7;

template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2); // N_VALUE是当前tile块的n长度
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset],
            l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    }
}

template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokensLayered(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    // int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    // uint32_t actualDataLength = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    // uint32_t maxDataLengthPerBlock = Ops::Base::CeilAlign(static_cast<int64_t>(MegaMoeImpl::L1_TILE_N *
    //                                                       sizeof(ElementMMadOut2)), (int64_t)ALIGN_32);
    // // 每个 token 含数据+三元组(32B)
    // uint32_t maxDataSizePerToken = Ops::Base::CeilDiv(static_cast<int64_t>(params.tilingData->h),
    //     (int64_t)MegaMoeImpl::L1_TILE_N) * (ALIGN_32 + maxDataLengthPerBlock);
    // LocalTensor<int32_t> tripleTempTensor = LocalTensor<int32_t>(TPosition::VECCALC,
    //                                                              COMBINE_SEND_ADDR, ALIGN_32 / sizeof(int32_t));
    // AscendC::GlobalTensor<ElementMMadOut2> gmLocalD;
    // AscendC::GlobalTensor<int32_t> gmLocalDTriple;
    // AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    // ub2GmParams.blockCount = 1;
    // ub2GmParams.blockLen = actualDataLength;

    // AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    // AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    // for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
    //     uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
    //     uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
    //     uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
    //     if (toRankId == params.combineCommParams.rankId) {
    //         uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
    //         GM_ADDR localAddr = (GM_ADDR)(params.peermemInfo.combineSendPtr + gmDstOffset * sizeof(ElementMMadOut2));
    //         gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(localAddr));
    //         AscendC::DataCopyPad(gmLocalD,
    //             l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    //         continue;
    //     }
    //     tripleTempTensor.SetValue(rankIdIndex, toRankId);
    //     tripleTempTensor.SetValue(tokenIdxIndex, tokenIdx);
    //     tripleTempTensor.SetValue(topkIdxIndex, topkIdx);
    //     tripleTempTensor.SetValue(blockLenIndex, nLoc);
    //     tripleTempTensor.SetValue(tokenLenIndex, n);
    //     tripleTempTensor.SetValue(tokenActualLenIndex, actualDataLength);
    //     tripleTempTensor.SetValue(flagIndex, 1); // 标记为已处理
    //     __gm__  int32_t* notifyAddr = (__gm__ int32_t*)(params.workspaceInfo.combineCommNotifyPtr +
    //                                                     toRankId * sizeof(int32_t));
    //     GM_ADDR dataAddr = params.workspaceInfo.combineCommDataPtr +
    //                     static_cast<uint64_t>(toRankId) * maxDataSizePerToken * params.tilingData->bs *
    //                     params.tilingData->expertPerRank;
    //     int32_t cnt = AscendC::AtomicAdd(notifyAddr, 1);
    //     AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    //     AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    //     // 数据区: 每个 rank 独立空间, 每个槽位 = maxDataLength(数据) + 32B(三元组)
    //     gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
    //         dataAddr + cnt * (maxDataLengthPerBlock + ALIGN_32)));
    //     AscendC::DataCopyPad(gmLocalD,
    //         l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    //     AscendC::PipeBarrier<PIPE_MTE3>();
    //     // 三元组区: 紧跟数据之后
    //     gmLocalDTriple.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
    //         dataAddr + cnt * (maxDataLengthPerBlock + ALIGN_32) + maxDataLengthPerBlock));
    //     AscendC::DataCopy(gmLocalDTriple, tripleTempTensor, ALIGN_32 / sizeof(int32_t));
    //     AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    //     AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    // }
}

// =============================================
// ComputeRecvTokenCounts: 从 mask 统计每个 rank 的应收 token 总数
// 返回已完成 rank 数 (本卡 + 0-token)
// =============================================
// template <typename ElementMMadOut2>
// __aicore__ inline uint32_t ComputeRecvTokenCounts(
//     uint32_t startRankId, uint32_t endRankId, uint32_t rankId,
//     uint32_t expertPerRank, uint32_t worldSize,
//     int64_t maskSlotSize, uint32_t slotCopyStride, uint32_t countWordIdx,
//     LocalTensor<uint32_t>& slotReadTensor, LocalTensor<uint8_t>& slotCopyTensor,
//     AscendC::GlobalTensor<uint8_t>& maskSrcGlobal,
//     LocalTensor<int32_t>& totalTokenCnt, const Params& params)
// {
//     uint32_t completedRank = 0;
//     for (uint32_t index = startRankId; index < endRankId; index++) {
//         if (index == rankId) {
//             completedRank++;
//             continue;
//         }
//         uint32_t cnt = 0;
//         for (uint32_t expertIdx = 0; expertIdx < expertPerRank; expertIdx++) {
//             maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params.peermemInfo.maskRecvPtr +
//                 static_cast<uint64_t>(expertIdx) * worldSize * maskSlotSize +
//                 static_cast<uint64_t>(index) * maskSlotSize));
//             AscendC::DataCopy(slotCopyTensor, maskSrcGlobal, slotCopyStride);
//             SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
//             cnt += slotReadTensor.GetValue(countWordIdx);
//             SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID2>();
//         }
//         uint32_t totalDataBlockCnt = cnt * static_cast<uint32_t>(Ops::Base::CeilDiv(
//             static_cast<int64_t>(params.tilingData->h), static_cast<int64_t>(MegaMoeImpl::L1_TILE_N)));
//         totalTokenCnt.SetValue(index - startRankId, totalDataBlockCnt);
//         if (totalDataBlockCnt == 0) {
//             completedRank++;
//         }
//     }
//     return completedRank;
// }

// =============================================
// PollNotifyAndSendData: 轮询 notify 计数, 读取 triple 并 URMA 发送数据
// initialCompletedRank: Phase 1 已完成的 rank 数 (本卡 + 0-token)
// =============================================
// template <typename ElementMMadOut2>
// __aicore__ inline void PollNotifyAndSendData(
//     uint32_t startRankId, uint32_t initialCompletedRank, uint32_t processRankNum, uint32_t rankId,
//     uint32_t maxDataLengthPerBlock, uint32_t maxDataSizePerToken,
//     LocalTensor<int32_t>& sendTokenCnt, LocalTensor<int32_t>& totalTokenCnt,
//     LocalTensor<int32_t>& sendTensor, const Params& params)
// {
//     uint32_t completedRank = initialCompletedRank;
//     uint32_t rankIndex = 0;
//     AscendC::GlobalTensor<int32_t> gmLocalDTriple;
//     uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;

//     while (completedRank < processRankNum) {
//         // 跳过本卡 rank 或已完成的 rank
//         while ((rankIndex + startRankId == rankId) ||
//                sendTokenCnt.GetValue(rankIndex) == totalTokenCnt.GetValue(rankIndex)) {
//             rankIndex = (rankIndex + 1) % processRankNum;
//         }
//         // 读 notify count
//         __gm__ int32_t* notifyAddr = (__gm__ int32_t*)(params.workspaceInfo.combineCommNotifyPtr +
//             (rankIndex + startRankId) * sizeof(int32_t));
//         uint32_t curCnt = ReadGmByPassDCache(notifyAddr);
//         uint32_t completeCnt = sendTokenCnt.GetValue(rankIndex);

//         if (curCnt > completeCnt) {
//             uint32_t needCnt = curCnt - completeCnt;
//             uint32_t slotIdx = completeCnt;
//             while (needCnt > 0) {
//                 GM_ADDR dataAddr = params.workspaceInfo.combineCommDataPtr +
//                     static_cast<uint64_t>(rankIndex + startRankId) * maxDataSizePerToken *
//                     params.tilingData->bs * params.tilingData->expertPerRank +
//                     slotIdx * (maxDataLengthPerBlock + ALIGN_32);
//                 gmLocalDTriple.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dataAddr + maxDataLengthPerBlock));
//                 AscendC::DataCopy(sendTensor, gmLocalDTriple, 8);
//                 SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
//                 if (sendTensor.GetValue(flagIndex) == 1) {
//                     uint32_t toRankId = sendTensor.GetValue(rankIdIndex);
//                     uint32_t tokenIdx = sendTensor.GetValue(tokenIdxIndex);
//                     uint32_t topkIdx = sendTensor.GetValue(topkIdxIndex);
//                     uint32_t nLoc = sendTensor.GetValue(blockLenIndex);
//                     uint32_t n = sendTensor.GetValue(tokenLenIndex);
//                     uint32_t dataLen = sendTensor.GetValue(tokenActualLenIndex);
//                     uint64_t channelHandle = GetUrmaCommHandle(params.combineCommParams.mc2Context, toRankId, rankId);
//                     uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
//                     GM_ADDR remoteAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset) +
//                         gmDstOffset * sizeof(ElementMMadOut2);
//                     params.combineCommParams.hcomm->WriteNbi(channelHandle, remoteAddr, dataAddr, dataLen);
//                     // 标记已处理, 避免重复发送
//                     sendTensor.SetValue(flagIndex, 0);
//                     SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
//                     AscendC::DataCopy(gmLocalDTriple, sendTensor, 8);
//                     SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
//                     needCnt--;
//                 }
//                 slotIdx = (slotIdx + 1 < curCnt) ? slotIdx + 1 : completeCnt;
//             }
//             sendTokenCnt.SetValue(rankIndex, curCnt);
//             if (curCnt == totalTokenCnt.GetValue(rankIndex)) {
//                 completedRank++;
//             }
//         }
//         rankIndex = (rankIndex + 1) % processRankNum;
//     }
// }

// =============================================
// DrainUrmaConnections: drain 本核负责的 rank 的 urma 发送
// =============================================
// __aicore__ inline void DrainUrmaConnections(
//     uint32_t startRankId, uint32_t endRankId, uint32_t rankId, const Params& params)
// {
//     for (uint32_t dr = startRankId; dr < endRankId; ++dr) {
//         if (dr == rankId) {
//             continue;
//         }
//         params.combineCommParams.hcomm->Drain(GetUrmaCommHandle(params.combineCommParams.mc2Context, dr, rankId));
//     }
// }

// =============================================
// CombineSendTokenToRemote: 编排器 — 分核、VECCALC 分配、调用三阶段子函数
// =============================================
// template <typename ElementMMadOut2>
// __aicore__ inline void CombineSendTokenToRemote(const Params& params)
// {
//     uint32_t maxDataLengthPerBlock = Ops::Base::CeilAlign(static_cast<int64_t>(MegaMoeImpl::L1_TILE_N *
//         sizeof(ElementMMadOut2)), (int64_t)ALIGN_32);
//     uint32_t maxDataSizePerToken = Ops::Base::CeilDiv(static_cast<int64_t>(params.tilingData->h),
//         (int64_t)MegaMoeImpl::L1_TILE_N) * (ALIGN_32 + maxDataLengthPerBlock);
//     LocalTensor<int32_t> sendTensor = LocalTensor<int32_t>(TPosition::VECCALC, ALIGN_512, ALIGN_32 / sizeof(int32_t));

//     // 分核: 仅 subBlockIdx_==1 的核参与, 核数 = GetBlockNum() (每个 block 的核1)
//     uint32_t aivIdx = GetBlockIdx() / GetTaskRation();
//     uint32_t aivNum = GetBlockNum();
//     uint32_t rankId = params.combineCommParams.rankId;
//     uint32_t worldSize = params.tilingData->epWorldSize;
//     uint32_t rankPerCore = worldSize / aivNum;
//     uint32_t remainder = worldSize % aivNum;
//     uint32_t startRankId = rankPerCore * aivIdx + (aivIdx < remainder ? aivIdx : remainder);
//     uint32_t processRankNum = rankPerCore + (aivIdx < remainder ? 1 : 0);
//     uint32_t endRankId = startRankId + processRankNum;
//     uint32_t cntTensorLength = Ops::Base::CeilAlign(static_cast<uint64_t>(processRankNum) * sizeof(uint32_t),
//         (uint64_t)ALIGN_32);
//     if (startRankId >= endRankId || processRankNum == 0) {
//         return;
//     }

//     LocalTensor<int32_t> sendTokenCnt = LocalTensor<int32_t>(TPosition::VECCALC, ALIGN_32 + ALIGN_512,
//         cntTensorLength / sizeof(int32_t));
//     LocalTensor<int32_t> totalTokenCnt = LocalTensor<int32_t>(TPosition::VECCALC, ALIGN_32 + ALIGN_512 +
//         cntTensorLength, cntTensorLength / sizeof(int32_t));
//     AscendC::Duplicate<int32_t>(sendTokenCnt, 0, processRankNum);
//     AscendC::Duplicate<int32_t>(totalTokenCnt, 0, processRankNum);
//     SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
//     SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID3>();

//     // mask slot 参数
//     int64_t sendTotalNum = static_cast<int64_t>(params.tilingData->bs) * params.tilingData->topK;
//     int64_t compareCount = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum) * (int64_t)sizeof(int32_t),
//         (int64_t)ALIGN_256) / (int64_t)sizeof(int32_t);
//     int64_t maskAlignSize = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount) / 8, (int64_t)ALIGN_32);
//     int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32;
//     uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize / sizeof(int32_t));
//     uint32_t slotCopyStride = static_cast<uint32_t>(maskSlotSize / sizeof(uint8_t));
//     uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize / sizeof(int32_t));
//     uint32_t expertPerRank = params.tilingData->expertPerRank;
//     LocalTensor<uint32_t> slotReadTensor = LocalTensor<uint32_t>(TPosition::VECCALC,
//         ALIGN_32 + ALIGN_512 + cntTensorLength * 2, slotWordStride);
//     LocalTensor<uint8_t> slotCopyTensor = LocalTensor<uint8_t>(TPosition::VECCALC,
//         ALIGN_32 + ALIGN_512 + cntTensorLength * 2, slotCopyStride);
//     AscendC::GlobalTensor<uint8_t> maskSrcGlobal;

//     // Phase 1: 从 mask 统计每个 rank 的应收 token 总数
//     uint32_t completedRank = ComputeRecvTokenCounts<ElementMMadOut2>(
//         startRankId, endRankId, rankId, expertPerRank, worldSize,
//         maskSlotSize, slotCopyStride, countWordIdx,
//         slotReadTensor, slotCopyTensor, maskSrcGlobal,
//         totalTokenCnt, params);
//     if (completedRank == processRankNum) {
//         return;
//     }

//     // Phase 2: 轮询 notify 计数并 URMA 发送数据
//     PollNotifyAndSendData<ElementMMadOut2>(
//         startRankId, completedRank, processRankNum, rankId,
//         maxDataLengthPerBlock, maxDataSizePerToken,
//         sendTokenCnt, totalTokenCnt, sendTensor, params);

//     // Phase 3: drain 本核负责的 rank 的 urma 发送
//     DrainUrmaConnections(startRankId, endRankId, rankId, params);
// }


// =============================================
// QuantMxFp8：将 bf16 数据量化为 MXFP8 格式
// =============================================
template<uint8_t QuantMode, typename ExpandXType>
__aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType>& outLocal, LocalTensor<ExpandXType>& inLocal,
    LocalTensor<float>& floatTemp, int32_t processLen)
{
    PipeBarrier<PIPE_V>();
    uint32_t mxScaleNum = Align2(Ceil32(processLen));
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
    LocalTensor<Fp8Type> castFp8LocalTensor = outLocal.template ReinterpretCast<Fp8Type>();
    __ubuf__ ExpandXType* srcAddr = (__ubuf__ ExpandXType*)inLocal.GetPhyAddr();
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatTemp.GetPhyAddr();
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)floatTemp[Align32(mxScaleNum)].GetPhyAddr();
    __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)castFp8LocalTensor.GetPhyAddr();
    __ubuf__ uint16_t* mxScaleLocalAddr =
        (__ubuf__ uint16_t*)castFp8LocalTensor[processLen].GetPhyAddr();
    Quant::ComputeMaxExp(srcAddr, maxExpAddr, processLen); // 计算最大Exp
    // 计算scales并填充
    Quant::ComputeScale<Fp8Type>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
    Quant::ComputeFp8Data<ExpandXType, Fp8Type,
        AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
        srcAddr, halfScaleLocalAddr, outLocalAddr, processLen); // 计算量化后的expandx并填充
}

// =============================================
// DeQuantMxFp8：FP8 反量化，将 FP8 数据转换回 BF16/FP32
// =============================================
template <typename T, typename XType>
__aicore__ inline void DeQuantMxFp8(LocalTensor<XType>& inLocal, LocalTensor<float>& sumTensor,
    LocalTensor<bfloat16_t>& scaleBf16Tensor, LocalTensor<float>& scaleFP32Tensor,
    uint32_t scaleLen, uint32_t tokenLen)
{
    LocalTensor<T> castFp8LocalTensor_ = inLocal.template ReinterpretCast<T>();
    LocalTensor<fp8_e8m0_t> scaleDivFp8Tensor_ =
        inLocal[Align256<uint32_t>(tokenLen) / 2].template ReinterpretCast<fp8_e8m0_t>();
    __ubuf__ bfloat16_t *dyScaleBf16Ptr = (__ubuf__ bfloat16_t *)scaleBf16Tensor.GetPhyAddr();
    __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleFP32Tensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *srcPtr0 = (__ubuf__ fp8_e8m0_t *)scaleDivFp8Tensor_.GetPhyAddr();
    __ubuf__ T *tokenPtr0 = (__ubuf__ T *)castFp8LocalTensor_.GetPhyAddr();
    __ubuf__ float *sumDstPtr = (__ubuf__ float *)sumTensor.GetPhyAddr();
    uint32_t bf16RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(bfloat16_t);
    uint32_t fp32RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(float);
    uint16_t repeatTimes = Ceil(scaleLen, bf16RepeatSize);
    uint16_t fp32RepeatTimes = Ceil(tokenLen, fp32RepeatSize);
    uint16_t repeatTimes2 = Ceil(scaleLen * 2, fp32RepeatSize);
    uint32_t quantCount2 = scaleLen * 2;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<fp8_e8m0_t> vSrcReg;
        AscendC::MicroAPI::RegTensor<T> tokenSrcReg;
        AscendC::MicroAPI::RegTensor<float> tokenFp32SrcReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vDstReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> dyScaleBf16Reg;
        AscendC::MicroAPI::RegTensor<float> dyScaleFp32Reg;
        AscendC::MicroAPI::RegTensor<float> sumDstReg;
        AscendC::MicroAPI::RegTensor<float> sumLocalDstReg;
        static constexpr AscendC::MicroAPI::CastTrait FP82BF16CastTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait FP162FP32CastTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg maskReg1;
        AscendC::MicroAPI::MaskReg maskReg2;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(scaleLen);
            MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vSrcReg,
                srcPtr0 + i * bf16RepeatSize);
            MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                dyScaleBf16Ptr + i * bf16RepeatSize * 2, vDstReg, vDstReg, maskReg);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < repeatTimes2; i++) {
            maskReg1 = AscendC::MicroAPI::UpdateMask<float>(quantCount2);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg,
                dyScaleBf16Ptr + i * fp32RepeatSize);
            MicroAPI::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
                dyScaleFp32Ptr + i * fp32RepeatSize * 2, dyScaleFp32Reg, dyScaleFp32Reg, maskReg1);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < fp32RepeatTimes; i++) {
            maskReg2 = AscendC::MicroAPI::UpdateMask<float>(tokenLen);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg,
                tokenPtr0 + i * fp32RepeatSize);
            MicroAPI::Cast<float, T, FP82BF16CastTraitZero>(tokenFp32SrcReg, tokenSrcReg, maskReg2);
            MicroAPI::Mul(sumLocalDstReg, dyScaleFp32Reg, tokenFp32SrcReg, maskReg2);
            MicroAPI::DataCopy(sumDstPtr + i * fp32RepeatSize, sumLocalDstReg, maskReg2);
        }
    }
}

// =============================================
// CombineQuantizedTokens：将量化后的 token 发送到目标 rank
// =============================================
template <typename QuantOutType>
__aicore__ inline void CombineQuantizedTokens(
    uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
    uint32_t groupIdx, uint32_t rankId, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<QuantOutType>& ubQuant, const Params& params)
{
    int64_t quantTokenSize = n + nScale;
    uint32_t toRankId = tripleTensor.GetValue(batchStart * TRIPLE_SIZE + RANK_ID);
    uint32_t tokenIdx = tripleTensor.GetValue(batchStart * TRIPLE_SIZE + TOKEN_ID);
    uint32_t topkIdx = tripleTensor.GetValue(batchStart * TRIPLE_SIZE + TOPK_INDEX);

    AscendC::GlobalTensor<QuantOutType> gmRemoteD;
    uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dstPeermemPtr));

    uint64_t dstBaseOffset = (tokenIdx * params.tilingData->topK + topkIdx) * quantTokenSize;

    AscendC::DataCopyExtParams singleCopyParams{
        1, static_cast<uint32_t>(quantTokenSize), 0, 0, 0};
    AscendC::DataCopyPad(gmRemoteD[dstBaseOffset], ubQuant, singleCopyParams);
}

// =============================================
// CombineTokenGroup：处理一个 token group 的 Combine 操作，从 GMM2 输出读取数据，量化后发送到目标 rank
// =============================================
template <uint8_t QuantMode, typename T>
__aicore__ inline void CombineTokenGroup(
    uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx, uint32_t rankId,
    GM_ADDR gmm2OutAddr, const Params& params, LocalTensor<int32_t>& tripleTensor,
    int64_t ubTensorSize, int64_t offset, uint32_t quantTokenSizeBytes)
{
    LocalTensor<T> combineUbTensor(TPosition::VECIN, offset, ubTensorSize);
    offset += ubTensorSize * sizeof(T);
    
    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t floatTempSize = nScale + nScale / 2;
    LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offset, floatTempSize);
    
    GlobalTensor<T> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmm2OutAddr));

    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;

    uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantTokenSizeBytes) / sizeof(T);
    DataCopyPadExtParams<T> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};

    for (uint32_t i = 0; i < tokenCount; i++) {
        uint32_t pingPong = i % 2;
        LocalTensor<T> ubBf16 = combineUbTensor[pingPong * singleTokenElems];
        LocalTensor<T> ubQuantData = ubBf16[nAlign32];
        
        // MTE2: read from GM
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
        AscendC::DataCopyPad(ubBf16, gmm2OutGm[(tokenStart + i) * n], gm2UbParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();
        
        // V: quantize
        QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();
        
        // MTE3: send to GM
        LocalTensor<Fp8Type> ubQuantDataFp8 = ubQuantData.template ReinterpretCast<Fp8Type>();
        CombineQuantizedTokens<Fp8Type>(i, 1, n, nScale, groupIdx, rankId, tripleTensor,
            ubQuantDataFp8, params);
    }
    
    // Wait for all MTE3 operations to complete
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H