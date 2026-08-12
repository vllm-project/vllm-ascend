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
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

using namespace AscendC;

namespace MegaMoeCombineImpl {

template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(uint32_t nLoc, uint32_t n, LocalTensor<int32_t> &metaInfoTensor,
                                     LocalTensor<ElementMMadOut2> &l0cOutUbGMM2, BlockShape &actualBlockShape,
                                     uint32_t ubTileN, const Params &params)
{
    // The caller owns the GM-to-UB lifetime of metaInfoTensor and must make it
    // scalar-visible once after the bulk preload.  Keeping that dependency at
    // the preload boundary avoids one MTE2_S round trip for every token row.
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    // 尾块只发送 actualN 个有效元素，但 UB 中相邻两行仍按物理 tile 宽度 ubTileN 排布。
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = metaInfoTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = metaInfoTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = metaInfoTensor.GetValue(tileIdx * 8 + 2);
        gmRemoteD.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementMMadOut2 *>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[tileIdx * ubTileN], ub2GmParams);
    }
}

// Wave combine has one complete row resident in UB. Resolve the route and
// destination row once, then issue fixed-size remote copies for both BF16 and
// FP8 rows.
template <typename Element>
__aicore__ inline void SendCombineTokenRow(uint32_t rowElements, uint64_t gmRemoteBaseOffset,
                                           LocalTensor<int32_t> &metaInfoTensor, LocalTensor<Element> &rowTensor,
                                           const Params &params)
{
    uint32_t toRankId = metaInfoTensor.GetValue(RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(TOPK_INDEX);

    AscendC::GlobalTensor<Element> gmRemoteD;
    gmRemoteD.SetGlobalBuffer(
        reinterpret_cast<__gm__ Element *>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
    uint64_t gmDstRowOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * rowElements;

    constexpr uint32_t TRANSFER_BYTES = 512U;
    constexpr uint32_t TILE_ELEMENTS = TRANSFER_BYTES / sizeof(Element);
    AscendC::DataCopyExtParams ub2GmParams{1U, 0U, 0U, 0U, 0U};
    for (uint32_t elementOffset = 0U; elementOffset < rowElements; elementOffset += TILE_ELEMENTS) {
        // 每轮最多下发 512B；最后一轮不足 512B 时只发送有效尾块。
        uint32_t remainingElements = rowElements - elementOffset;
        uint32_t currentElements = remainingElements < TILE_ELEMENTS ? remainingElements : TILE_ELEMENTS;
        ub2GmParams.blockLen = currentElements * sizeof(Element);
        AscendC::DataCopyPad(
            gmRemoteD[gmDstRowOffset + elementOffset], rowTensor[elementOffset], ub2GmParams);
    }
}

template <typename DataType, bool IsQuantized = true>
__aicore__ inline void CombineSendTokenToRemote(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                                uint32_t groupIdx, uint32_t rankId,
                                                LocalTensor<int32_t> &metaInfoTensor, LocalTensor<DataType> &ubQuant,
                                                const Params &params, GM_ADDR localSrcPtr)
{
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID3>();
    int64_t quantTokenSize = IsQuantized ? (n + nScale) : n;
    uint32_t toRankId = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);

    AscendC::GlobalTensor<DataType> gmLocalD;
    uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    GM_ADDR srcAddr = localSrcPtr;

    if (toRankId == rankId) {
        srcAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    }

    gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(srcAddr));
    uint64_t dstBaseOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSize; // 元素个数
    AscendC::DataCopyExtParams singleCopyParams{1, static_cast<uint32_t>(quantTokenSize * sizeof(DataType)), 0, 0, 0};

    if constexpr (!IsQuantized) {
        DataCopyPadExtParams<DataType> copyPadParams{false, 0U, 0U, 0U};
        if (toRankId == rankId) {
            AscendC::GlobalTensor<DataType> gmm2OutGm;
            gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(localSrcPtr));
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
            AscendC::DataCopyPad(ubQuant, gmm2OutGm, singleCopyParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_MTE3, SYNC_EVENT_ID4>();
        }
    }

    if (IsQuantized || toRankId == rankId) {
        uint64_t dstOffset = toRankId == rankId ? dstBaseOffset : 0;
        AscendC::DataCopyPad(gmLocalD[dstOffset], ubQuant, singleCopyParams);
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID4>();
    }

    if (toRankId != rankId) {
        uint64_t channelHandle = GetUrmaCommHandle(params.combineCommParams.mc2Context, toRankId, rankId);
        GM_ADDR remoteAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset) + dstBaseOffset * sizeof(DataType);
        params.combineCommParams.hcomm->WriteNbi(channelHandle, remoteAddr, srcAddr, quantTokenSize * sizeof(DataType));
    }
}

// =============================================
// QuantMxFp8：将 bf16 数据量化为 MXFP8 格式
// =============================================
template <uint8_t QuantMode, typename ExpandXType>
__aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal,
                                  LocalTensor<float> &floatTemp, int32_t processLen)
{
    PipeBarrier<PIPE_V>();
    uint32_t mxScaleNum = Align2(Ceil32(processLen));
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
    LocalTensor<Fp8Type> castFp8LocalTensor = outLocal.template ReinterpretCast<Fp8Type>();
    __ubuf__ ExpandXType *srcAddr = (__ubuf__ ExpandXType *)inLocal.GetPhyAddr();
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)floatTemp.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)floatTemp[Align32(mxScaleNum)].GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)castFp8LocalTensor.GetPhyAddr();
    // The MTE record stores scales after the 256B-aligned FP8 token region.
    uint32_t tokenStorageElementCount = Align256<uint32_t>(static_cast<uint32_t>(processLen));
    __ubuf__ uint16_t *mxScaleLocalAddr =
        (__ubuf__ uint16_t *)castFp8LocalTensor[tokenStorageElementCount].GetPhyAddr();
    Quant::ComputeMaxExp(srcAddr, maxExpAddr, processLen); // 计算最大Exp
    // 计算scales并填充
    Quant::ComputeScale<Fp8Type>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
    Quant::ComputeFp8Data<ExpandXType, Fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
        srcAddr, halfScaleLocalAddr, outLocalAddr, processLen); // 计算量化后的expandx并填充
}

// =============================================
// DeQuantMxFp8：FP8 反量化，将 FP8 数据转换回 BF16/FP32
// =============================================
template <typename T, typename XType>
__aicore__ inline void DeQuantMxFp8(LocalTensor<XType> &inLocal, LocalTensor<float> &sumTensor,
                                    LocalTensor<bfloat16_t> &scaleBf16Tensor, LocalTensor<float> &scaleFP32Tensor,
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
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait FP162FP32CastTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg maskReg1;
        AscendC::MicroAPI::MaskReg maskReg2;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(scaleLen);
            MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vSrcReg, srcPtr0 + i * bf16RepeatSize);
            MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(dyScaleBf16Ptr + i * bf16RepeatSize * 2,
                                                                                vDstReg, vDstReg, maskReg);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < repeatTimes2; i++) {
            maskReg1 = AscendC::MicroAPI::UpdateMask<float>(quantCount2);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg,
                                                                                dyScaleBf16Ptr + i * fp32RepeatSize);
            MicroAPI::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(dyScaleFp32Ptr + i * fp32RepeatSize * 2,
                                                                           dyScaleFp32Reg, dyScaleFp32Reg, maskReg1);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < fp32RepeatTimes; i++) {
            maskReg2 = AscendC::MicroAPI::UpdateMask<float>(tokenLen);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg, tokenPtr0 + i * fp32RepeatSize);
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
__aicore__ inline void CombineQuantizedTokens(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                              uint32_t groupIdx, uint32_t rankId, LocalTensor<int32_t> &metaInfoTensor,
                                              LocalTensor<QuantOutType> &ubQuant, const Params &params,
                                              uint32_t quantTokenSizeBytes)
{
    uint32_t toRankId = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);

    AscendC::GlobalTensor<QuantOutType> gmRemoteD;
    uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    __gm__ void *dstPeermemPtr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(dstPeermemPtr));

    uint64_t dstBaseOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSizeBytes;

    AscendC::DataCopyExtParams singleCopyParams{1, quantTokenSizeBytes, 0, 0, 0};
    AscendC::DataCopyPad(gmRemoteD[dstBaseOffset], ubQuant, singleCopyParams);
}

// =============================================
// CombineTokenGroup：处理一个 token group 的 Combine 操作，从 GMM2 输出读取数据，量化后发送到目标 rank
// =============================================
template <uint8_t QuantMode, typename T, bool IsLayered = false, bool IsQuantized = true>
__aicore__ inline void CombineTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx,
                                         uint32_t rankId, GM_ADDR gmm2OutAddr, const Params &params,
                                         LocalTensor<int32_t> &metaInfoTensor, int64_t ubTensorSize, int64_t offset,
                                         uint32_t quantTokenSizeBytes)
{
    LocalTensor<T> combineUbTensor(TPosition::VECIN, offset, ubTensorSize);
    offset += ubTensorSize * sizeof(T);

    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t mxScaleNum = Align2(nScale);
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    // floatTemp stores aligned maxExp followed by one BF16 halfScale for each stored scale.
    uint32_t floatTempSize = Align32(mxScaleNum) + mxScaleNum / 2;
    LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offset, floatTempSize);

    GlobalTensor<T> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmm2OutAddr));

    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;

    uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantTokenSizeBytes) / sizeof(T);
    DataCopyPadExtParams<T> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};

    for (uint32_t i = 0; i < tokenCount; i++) {
        uint32_t pingPong = i % 2;
        LocalTensor<T> ubBf16 = combineUbTensor[pingPong * singleTokenElems];
        LocalTensor<T> ubQuantData = ubBf16[nAlign32];

        if constexpr (IsQuantized) {
            // MTE2: read from GM
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
            AscendC::DataCopyPad(ubBf16, gmm2OutGm[(tokenStart + i) * n], gm2UbParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();

            // V: quantize
            QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();
        }

        using SendType = typename std::conditional<IsQuantized, Fp8Type, T>::type;
        LocalTensor<SendType> ubQuantSend = ubQuantData.template ReinterpretCast<SendType>();
        if constexpr (IsLayered) {
            if constexpr (!IsQuantized) {
                ubQuantSend = ubBf16;
            }
            GM_ADDR localSrcPtr = gmm2OutAddr + (tokenStart + i) * n * sizeof(T);
            CombineSendTokenToRemote<SendType, IsQuantized>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor,
                                                            ubQuantSend, params, localSrcPtr);
        } else {
            // Only the MTE path uses the padded H=32 quantized-record layout.
            CombineQuantizedTokens<SendType>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor, ubQuantSend, params,
                                             quantTokenSizeBytes);
        }
    }

    // Wait for all MTE3 operations to complete
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeCombineImpl

#endif // MEGA_MOE_COMBINE_SEND_H
