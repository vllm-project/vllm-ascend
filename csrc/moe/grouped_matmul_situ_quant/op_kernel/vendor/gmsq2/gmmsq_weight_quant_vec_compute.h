/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the License for the specific text of the License.
 */

/*!
 * \file gmmsq_weight_quant_vec_compute.h
 * \brief GMMSQ SiTU variant — AIV-side vector compute.
 *
 * Adapted from the CANN OSL 2.0 production op grouped_matmul_swiglu_quant_v2
 * (arch35/weight_quant_basic_block/gmmsq_weight_quant_vec_compute.h): the
 * weight FP4->FP8 widen datapath is kept verbatim; the SwiGLU epilogue is
 * replaced by the SiTU + dynamic MXQuant epilogue (bit-exact regbase formulas
 * from op_kernel/situ_epilogue.h), consuming the L0C->LCM F32 relay.
 *
 * LCM budget per AIV (variant A "mmRes/输出槽让位"):
 *   [0, 64KB)    FP4-in MTE2 ring (4 x 16KB)          — untouched
 *   [64, 192KB)  FP8-out widen ring (4 x 32KB)        — between blocks this
 *                region is YIELDED to the mmRes F32 relay ([64,128KB)) and
 *                the SiTU epilogue scratch ([128,192KB)); ring depth stays 4.
 */

#ifndef GMMSQ_SITU_VEC_COMPUTE_H
#define GMMSQ_SITU_VEC_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "basic_block_config.h"
#include "basic_block_vf_mx.h"
#include "../wqbmm/weight_quant_tool.h"
#include "gmmsq_quant_mx_vf.h"
#include "gmmsq_weight_quant_cube_compute_tools.h"
#include "../../situ_epilogue.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::PaddingMode;
using AscendC::SetFlag;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
using namespace WeightQuantBatchMatmulV2::Arch35;

namespace GMMSQWeightQuant {

// SiTU epilogue scratch — all inside the FP8-out widen ring region (yielded
// between blocks). Max m-half rows = 64 (mL1Size <= 128), nHalf = 64 cols.
struct GmmsqSituUbBufferInfo {
    uint64_t weightHighBitBufferNum;
    uint64_t weightHighBitTotalSize;
    uint64_t weightLowbitTotalSize;
    uint64_t weightLowBitSingleBufferSize;
    uint64_t situGateTotalSize;   // bf16 elems: 64 rows x 64
    uint64_t situUpTotalSize;     // bf16 elems
    uint64_t situOutTotalSize;    // bf16 elems
    uint64_t yFp8TotalSize;       // fp8 elems: 64 x 64
    uint64_t maxExpTotalSize;     // u16 elems
    uint64_t halfScaleTotalSize;  // u16 elems
    uint64_t scaleOutTotalSize;   // u16 elems
};

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr GmmsqSituUbBufferInfo GetGmmsqSituBufferInfo()
{
    return {
        .weightHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
        .weightHighBitTotalSize = 128 * GetKBUnit<int8_t>(),
        .weightLowbitTotalSize = 64 * GetKBUnit<int8_t>(),
        .weightLowBitSingleBufferSize = 64 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
        .situGateTotalSize = 64 * 64,
        .situUpTotalSize = 64 * 64,
        .situOutTotalSize = 64 * 64,
        .yFp8TotalSize = 64 * 64,
        .maxExpTotalSize = 1024,
        .halfScaleTotalSize = 1024,
        .scaleOutTotalSize = 1024,
    };
}

#define GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM                                                       \
    template <typename xType, typename wType, typename yType, typename yScaleType,                  \
              const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>

#define GMMSQ_SITU_VEC_COMPUTE_CLASS \
    GMMSQSituVecCompute<xType, wType, yType, yScaleType, wqmmConfig, vecConfig>

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
class GMMSQSituVecCompute {
public:
    __aicore__ inline GMMSQSituVecCompute(){};

    __aicore__ inline void Init(__gm__ yType *yAddr, __gm__ yScaleType *yScaleAddr, float beta, float invBeta,
                                float linearBeta, float invLinearBeta);
    __aicore__ inline void UpdateGlobalAddr(__gm__ wType *weight, const bool weightL2Cacheable);
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset,
                                      const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WeightAntiQuantComputeNzNk(uint64_t kRealSize, uint64_t kGmOffset,
                                                      const LocalTensor<xType> &weightHighBitL1,
                                                      const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SituEpilogue(const LocalTensor<float> &yF32, const BasicBlockOffsetParam &param);
    __aicore__ inline void End();

private:
    __aicore__ inline void AntiQuantProcessNzMxA8W4(uint64_t ubMte2KSize, uint64_t kGmOffset,
                                                    const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t antiQuantRealN, uint64_t antiQuantRealK,
                                                       const LocalTensor<xType> &weightHighBitL1);

    constexpr static GmmsqSituUbBufferInfo UB_BUFFER_INFO = GetGmmsqSituBufferInfo<vecConfig>();

    uint64_t weightMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;

    static constexpr uint32_t EVENT_ID_V_TO_MTE2 = 0;
    static constexpr uint32_t EVENT_ID_MTE2_TO_V = 0;
    static constexpr uint32_t EVENT_ID_MTE3_TO_V = 0;
    static constexpr uint32_t EVENT_ID_V_TO_MTE3 = 0;

    static constexpr uint64_t SITU_DEQ_FACTOR = 64UL; // GMM缩小64倍在此处补齐 (与生产 quantPre deqScalar 等价)

    GlobalTensor<wType> wGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<uint8_t> yScaleGlobal_;

    LocalTensor<int8_t> weightLowBit_;
    LocalTensor<xType> weightHighBit_;
    // SiTU scratch — carved out of the FP8-out ring region (offset 64KB in)
    LocalTensor<bfloat16_t> situGate_;
    LocalTensor<bfloat16_t> situUp_;
    LocalTensor<bfloat16_t> situOut_;
    LocalTensor<int8_t> yFp8_;
    LocalTensor<uint16_t> maxExpBuf_;
    LocalTensor<uint16_t> halfScaleBuf_;
    LocalTensor<uint16_t> scaleOutBuf_;

    float beta_ = 4.0f;
    float invBeta_ = 0.25f;
    float linearBeta_ = 25.0f;
    float invLinearBeta_ = 0.04f;

    constexpr static uint32_t C0_SIZE = C0_SIZE_B8;
    constexpr static uint64_t VEC_REG_ELEM = VECTOR_REG_WIDTH;
    constexpr static uint64_t UB_AVAILABLE_SIZE = 248 * GetKBUnit<int8_t>();
    constexpr static uint64_t DIVISOR_FACTOR_TWO = 2;
};

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::Init(__gm__ yType *yAddr, __gm__ yScaleType *yScaleAddr,
                                                          float beta, float invBeta, float linearBeta,
                                                          float invLinearBeta)
{
    beta_ = beta;
    invBeta_ = invBeta;
    linearBeta_ = linearBeta;
    invLinearBeta_ = invLinearBeta;
    yGlobal_.SetGlobalBuffer(yAddr);
    yScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(yScaleAddr));

    weightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightLowbitTotalSize);
    uint64_t ubOffset = UB_BUFFER_INFO.weightLowbitTotalSize;

    weightHighBit_ = LocalTensor<xType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.weightHighBitTotalSize);
    // Variant-A yield: the FP8-out ring region [64,192KB) is handed to the mmRes
    // F32 relay between blocks ([64,128KB), flag-guarded by the production
    // FIXFLAG/VFFLAG handshake). The SiTU scratch lives OUTSIDE the ring
    // ([192,226KB) fresh LCM) so the cycling widen ring slots are never touched
    // by the epilogue (removes the scratch-vs-ring V/MTE3 interleaving).
    uint64_t reuseOffset = ubOffset + UB_BUFFER_INFO.weightHighBitTotalSize;
    situGate_ = LocalTensor<bfloat16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.situGateTotalSize);
    reuseOffset += UB_BUFFER_INFO.situGateTotalSize * sizeof(bfloat16_t);
    situUp_ = LocalTensor<bfloat16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.situUpTotalSize);
    reuseOffset += UB_BUFFER_INFO.situUpTotalSize * sizeof(bfloat16_t);
    situOut_ = LocalTensor<bfloat16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.situOutTotalSize);
    reuseOffset += UB_BUFFER_INFO.situOutTotalSize * sizeof(bfloat16_t);
    yFp8_ = LocalTensor<int8_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.yFp8TotalSize);
    reuseOffset += UB_BUFFER_INFO.yFp8TotalSize * sizeof(int8_t);
    maxExpBuf_ = LocalTensor<uint16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.maxExpTotalSize);
    reuseOffset += UB_BUFFER_INFO.maxExpTotalSize * sizeof(uint16_t);
    halfScaleBuf_ = LocalTensor<uint16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.halfScaleTotalSize);
    reuseOffset += UB_BUFFER_INFO.halfScaleTotalSize * sizeof(uint16_t);
    scaleOutBuf_ = LocalTensor<uint16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.scaleOutTotalSize);

    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::UpdateGlobalAddr(__gm__ wType *weight,
                                                                      const bool weightL2Cacheable)
{
    wGlobal_.SetGlobalBuffer(weight);
    if (!weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::WaitVToMTE2()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::SetVToMTE2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
    weightMte2LoopIdx_++;
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset,
                                                                uint64_t kL1Offset,
                                                                const BasicBlockOffsetParam &offsetParam)
{
    if (offsetParam.nL1Size == 0 || ubMte2KSize == 0) {
        return;
    }

    // gate/up N轴对半切分：前半 fetch = gate（weight cols [nOffset, nOffset+nHalf)），
    // 后半 fetch = up（weight cols [nOffset+N/2, ...)），拼入同一 LCM 槽。
    uint64_t gateNOffset = offsetParam.nOffset * static_cast<uint64_t>(C0_SIZE);
    uint64_t upNOffset = (offsetParam.nOffset + offsetParam.nSize / 2) * static_cast<uint64_t>(C0_SIZE);
    uint64_t kOffset = (kGmOffset + kL1Offset) * offsetParam.nAlign;

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize]
            .template ReinterpretCast<wType>(),
        wGlobal_[kOffset + gateNOffset], CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        CeilAlign(offsetParam.nL1Size / 2, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE, offsetParam.nAlign * C0_SIZE);

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize +
                      (offsetParam.nL1Size / DIVISOR_FACTOR_TWO * C0_SIZE / 2)]
            .template ReinterpretCast<wType>(),
        wGlobal_[kOffset + upNOffset], CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        CeilAlign(offsetParam.nL1Size / 2, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE, offsetParam.nAlign * C0_SIZE);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::WeightAntiQuantComputeNzNk(
    uint64_t kRealSize, uint64_t kGmOffset, const LocalTensor<xType> &weightHighBitL1,
    const BasicBlockOffsetParam &offsetParam)
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));

    AntiQuantProcessNzMxA8W4(kRealSize, kGmOffset, offsetParam);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    if (likely(kRealSize > 0)) {
        CopyWeightHighBitForAligned(offsetParam.nL1Size, kRealSize, weightHighBitL1);
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));
    ubComputeLoopIdx_++;
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::AntiQuantProcessNzMxA8W4(
    uint64_t ubMte2KSize, uint64_t kGmOffset, const BasicBlockOffsetParam &offsetParam)
{
    MxA8W4NzParams<xType, wType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1);
    mxA8W4NzParams.nRealSizeAlign = CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType *)weightLowBit_[ubMte2BufferIdx * UB_BUFFER_INFO.weightLowBitSingleBufferSize].GetPhyAddr();

    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType *)
            weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VECTOR_REG_WIDTH]
                .GetPhyAddr();
    mxA8W4NzParams.loopKNum = CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE));
    mxA8W4NzParams.innerLoopNum =
        CeilDivide(CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                   static_cast<uint64_t>(VECTOR_REG_WIDTH));
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.weightHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;

    GmmsqAntiQuantMxA8W4NzNkVf<xType, wType>(mxA8W4NzParams);
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::CopyWeightHighBitForAligned(
    uint64_t antiQuantRealN, uint64_t antiQuantRealK, const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(C0_SIZE)) *
                        CeilAlign(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE)) / VEC_REG_ELEM;

    params.blockLen = VEC_REG_ELEM / ONE_BLK_SIZE;
    params.srcStride = (UB_BUFFER_INFO.weightHighBitBufferNum - 1) * params.blockLen;
    params.dstStride = 0;
    DataCopy(weightHighBitL1,
             weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VEC_REG_ELEM], params);
}

/*
 * SiTU + dynamic MXQuant epilogue on the L0C->LCM F32 relay.
 *
 * Relay layout (per basic block, written by AIC fixpipe NoQuant F32):
 *   row-major [mL1Size x nL1Size] f32, row stride nL1Size;
 *   cols [0, nHalf) = gate half, cols [nHalf, nL1) = up half.
 * Numerics contract (bit-exact vs golden npu_grouped_matmul+situ_mx_quant):
 *   bf16_gate/up = CastRINT(relay_f32 * 64.0f)   (== fixpipe QF322BF16_PRE
 *   deqScalar=64, production-precedent: grouped_matmul_swiglu_quant_v2
 *   GmmSwigluVf applies the same vector-side x64), then the verbatim
 *   situ_epilogue.h SiTU/MXQuant regbase sequence.
 */
GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::SituEpilogue(const LocalTensor<float> &yF32,
                                                                  const BasicBlockOffsetParam &param)
{
    uint64_t mHalf = CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    uint64_t mReal = AscendC::GetSubBlockIdx() ? (param.mL1Size - mHalf) : mHalf;
    if (mReal == 0 || param.mL1Size == 0) {
        return;
    }
    uint64_t mBase = static_cast<uint64_t>(AscendC::GetSubBlockIdx()) * mHalf;
    uint64_t nHalf = param.nL1Size / DIVISOR_FACTOR_TWO; // 2：gate/up 半区
    if (nHalf == 0) {
        return;
    }

    // ---- Step 1: dequant x64 + cast bf16 (gate & up halves) into scratch ----
    {
        __ubuf__ float *relayAddr = (__ubuf__ float *)yF32.GetPhyAddr();
        __ubuf__ bfloat16_t *gateAddr = (__ubuf__ bfloat16_t *)situGate_.GetPhyAddr();
        __ubuf__ bfloat16_t *upAddr = (__ubuf__ bfloat16_t *)situUp_.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vF;
            AscendC::MicroAPI::RegTensor<bfloat16_t> vB;
            uint32_t elemNum = static_cast<uint32_t>(nHalf);
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(elemNum);
            uint16_t rowCnt = static_cast<uint16_t>(mReal);
            // fixpipe dualDstCtl=1：F32 中继按 M 对半分投到两颗 AIV 各自的 LCM，
            // 本地 LCM 从偏移 0 起即为本 sub 的行段（生产 GmmSwigluVf 同语义）。
            for (uint16_t r = 0; r < rowCnt; ++r) {
                uint64_t srcOff = r * param.nL1Size;
                AscendC::MicroAPI::LoadAlign(vF, relayAddr + srcOff);
                AscendC::MicroAPI::Muls(vF, vF, static_cast<float>(SITU_DEQ_FACTOR), mask);
                AscendC::MicroAPI::Cast<bfloat16_t, float, GmsqEpilogue::CAST_FP32_TO_BF16>(vB, vF, mask);
                AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    gateAddr + r * nHalf, vB, mask);

                AscendC::MicroAPI::LoadAlign(vF, relayAddr + srcOff + nHalf);
                AscendC::MicroAPI::Muls(vF, vF, static_cast<float>(SITU_DEQ_FACTOR), mask);
                AscendC::MicroAPI::Cast<bfloat16_t, float, GmsqEpilogue::CAST_FP32_TO_BF16>(vB, vF, mask);
                AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    upAddr + r * nHalf, vB, mask);
            }
        }
    }

    // ---- Step 2: SiTU activation (bit-exact regbase formula) ----
    GmsqEpilogue::ComputeVfSitu<bfloat16_t, true>(
        (__ubuf__ bfloat16_t *)situGate_.GetPhyAddr(), (__ubuf__ bfloat16_t *)situUp_.GetPhyAddr(),
        (__ubuf__ bfloat16_t *)situOut_.GetPhyAddr(), static_cast<int64_t>(mReal), static_cast<int64_t>(nHalf),
        static_cast<int64_t>(nHalf), beta_, invBeta_, linearBeta_, invLinearBeta_);
    AscendC::PipeBarrier<PIPE_V>();

    // ---- Step 3-5: dynamic MX quant (E8M0 scale, 32-elem groups, FP8 out) ----
    GmsqEpilogue::ComputeVfMaxExpVfLast<bfloat16_t>(
        (__ubuf__ bfloat16_t *)situOut_.GetPhyAddr(), (__ubuf__ uint16_t *)maxExpBuf_.GetPhyAddr(),
        static_cast<int64_t>(mReal), static_cast<int64_t>(nHalf));
    AscendC::PipeBarrier<PIPE_V>();

    GmsqEpilogue::ComputeScaleLast<bfloat16_t>(
        GmsqEpilogue::FP8_E4M3_MAX_EXP, (__ubuf__ uint16_t *)maxExpBuf_.GetPhyAddr(),
        (__ubuf__ uint16_t *)scaleOutBuf_.GetPhyAddr(), (__ubuf__ uint16_t *)halfScaleBuf_.GetPhyAddr(),
        static_cast<int64_t>(mReal), static_cast<int64_t>(nHalf));
    AscendC::PipeBarrier<PIPE_V>();

    GmsqEpilogue::ComputeDataF8Last<bfloat16_t, fp8_e4m3fn_t>(
        (__ubuf__ bfloat16_t *)situOut_.GetPhyAddr(), (__ubuf__ uint16_t *)halfScaleBuf_.GetPhyAddr(),
        (__ubuf__ int8_t *)yFp8_.GetPhyAddr(), static_cast<int64_t>(mReal), static_cast<int64_t>(nHalf));

    // ---- Step 6: y / y_scale -> GM ----
    uint64_t n2Total = param.nSize / DIVISOR_FACTOR_TWO;                       // 输出列总数
    uint64_t scaleRowGroups = CeilDivide(n2Total, static_cast<uint64_t>(MX_GROUPSIZE)); // 每行 scale 字节数
    uint64_t yGmOffset = (param.mOffset + mBase) * n2Total + param.nOffset;
    uint64_t scaleGmOffset = (param.mOffset + mBase) * scaleRowGroups + param.nOffset / MX_GROUPSIZE;

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    {
        // 块级 GM 基址（控制器每 group 更新，生产 CopyOutputFromUb2Gm 同模式）
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(param.yGmAddr));
        yScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(param.yScaleGmAddr));
        DataCopyExtParams oy;
        oy.blockCount = static_cast<uint16_t>(mReal);
        oy.blockLen = static_cast<uint32_t>(nHalf);
        oy.srcStride = 0;
        oy.dstStride = static_cast<uint32_t>(n2Total - nHalf);
        oy.rsv = 0;
        auto yOutView = yFp8_.template ReinterpretCast<yType>();
        AscendC::DataCopyPad(yGlobal_[yGmOffset], yOutView, oy);

        DataCopyExtParams os;
        os.blockCount = static_cast<uint16_t>(mReal);
        os.blockLen = static_cast<uint32_t>(nHalf / MX_GROUPSIZE);
        os.srcStride = 0;
        os.dstStride = static_cast<uint32_t>(scaleRowGroups - nHalf / MX_GROUPSIZE);
        os.rsv = 0;
        auto sOutView = scaleOutBuf_.template ReinterpretCast<uint8_t>();
        AscendC::DataCopyPad<uint8_t, PaddingMode::Compact>(yScaleGlobal_[scaleGmOffset], sOutView, os);
    }

    // ring 释放：等 MTE3 排空后，FP8-out 环槽（含 relay 与 scratch 区域）复归展宽使用
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
}

GMMSQ_SITU_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_SITU_VEC_COMPUTE_CLASS::End()
{
    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

} // namespace GMMSQWeightQuant

#endif // GMMSQ_SITU_VEC_COMPUTE_H
