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
 * \file sparse_flash_attention_service_vector_mla_arch35.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_ARCH35_H
#define SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_ARCH35_H

#include "common/util_regbase.h"
#include "sparse_flash_attention_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "cann_common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#include "cann_common/arch35/vf/vf_flashupdate_new.h"
#include "cann_common/buffers_policy.h"
#include "cann_common/attn_buffer_manager.h"
#include "cann_common/attn_buffer.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {

TEMPLATES_DEF
class SFAVectorService {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t sfaDTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = 576;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;

    // ==================== Functions ======================
    __aicore__ inline SFAVectorService(){};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionTilingDataMla *__restrict tiling,
                                        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                        __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            if (actualSeqLengthsQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ);
            }
            if (actualSeqLengths != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths);
            }
            this->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    __aicore__ inline void UninitLocalBuffer();
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                       __gm__ uint8_t *softmaxSum, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *keyRope,
                                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable,
                                            __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                            __gm__ uint8_t *sinks);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);

private:
    __aicore__ inline void ProcessVec1SoftmaxDispatchSFA(LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes,
                                                         LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
                                                         LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo,
                                                         ConstInfo &constInfo);
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                           Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
                                           const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
                                             int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
                                              int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          ConstInfo &constInfo);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                           int64_t vec2S1Idx, int64_t sfaVec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                               LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                               int64_t sfaVec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void CopyFALseToGm(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void ComputeNeedInitSFA(CVSharedParams &sharedParams) const;
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &constInfo);

    TPipe *tPipe;
    const SparseFlashAttentionTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<T> sinksGm;
    LocalTensor<float> lseUb;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TBuf<> stage0OutBuf[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVAttnOutId; // 存放MTE3_V的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID vToMte3AttnOutId; // 存放V_MTE3的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID mte3ToVLseOutId;  // 存放MTE3_V的eventId, 用于V1 LSE拷出阶段的同步
    TEventID vToMte3LseOutId;  // 存放V_MTE3的eventId, 用于V1 LSE拷出阶段的同步
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> dequantScaleBuff;
    TBuf<> lseBuf;
    TBuf<> vselrIndexesBuf[2];

    TEventID mte2ToV;
    TEventID mte2ToMte3[2];
    TEventID mte3ToMte2[2];

    bool isSinks = false;
    T negativeFloatScalar;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;

    int64_t sfaSparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;

    TEventID initOutputEventId;
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(
    int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = runInfo.boIdx == 0 ? 0 : cuSeqlensQGm.GetValue(runInfo.boIdx - 1);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.sparseBlockCount +
                      runInfo.s1oIdx * constInfo.sparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount;
    int64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= constInfo.sparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
    topkKIdx += 1;
    if (unlikely((topkKIdx >= constInfo.sparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline int64_t SFAVectorService<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx,
                                                                                                 const RunInfo &runInfo,
                                                                                                 ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        realkeyOffset =
            blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * constInfo.keyStride0 +
            blkTableOffset; // BlockNum, BlockSize, N(1), D
    } else {
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
            realkeyOffset = (runInfo.boIdx * constInfo.s2Size + s2Idx); // BSN(1)D
        } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            int64_t batchKvStart = (runInfo.boIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(runInfo.boIdx - 1);
            realkeyOffset = (batchKvStart + s2Idx);
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyInSingleKv(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset, ConstInfo &constInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;

    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    uint32_t combineBytes = 512 * sizeof(KV_T); // 512: Key特征维度
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * 512], intriParams,
                padParams); // 512: Key特征维度; 576: 局部Buffer行跨度

    intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeRope * sizeof(KV_T);
    intriParams.dstStride =
        // 512: 模型特征维度(dSize)
        512 / BUFFER_SIZE_BYTE_32B;
    // 576: 局部Buffer行跨度 512: Key特征维度 64：RoPE索引数据每Token/块的行跨度（Stride）
    DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * 64], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline uint32_t SFAVectorService<TEMPLATE_ARGS>::CopyInKvSparse(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo,
    ConstInfo &constInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    int64_t blkTableSrcStride =
        ((keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - constInfo.sparseBlockSize);
    int64_t keySrcStride = blkTableSrcStride * constInfo.dSizeNope * sizeof(KV_T);
    int64_t keyRopeSrcStride = blkTableSrcStride * constInfo.dSizeRope * sizeof(KV_T);
    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
    } else if (keySrcStride >= INT32_MAX || keySrcStride < 0 || constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, constInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeNope * sizeof(KV_T);
        intriParams.dstStride = constInfo.dSizeRope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * constInfo.dSizeNope], // 576: 局部Buffer行跨度
                    intriParams, padParams);                                        // combineDimAlign

        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeRope * sizeof(KV_T);
        intriParams.dstStride = constInfo.dSizeNope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        intriParams.srcStride = keyRopeSrcStride;
        // 512: Key特征维度; 576: 局部Buffer行跨度
        DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * constInfo.dSizeRope], intriParams,
                    padParams); // combineDimAlign
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_1 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_2 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_4 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float *ubDstAddr, __ubuf__ int8_t *ubSrcAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<fp8_e8m0_t> vScale0;
    Reg::RegTensor<fp8_e8m0_t> vScale1;
    Reg::RegTensor<bfloat16_t> vScalebf16Res0;
    Reg::RegTensor<bfloat16_t> vScalebf16Res1;
    Reg::RegTensor<float> vScalefp32Res0;
    Reg::RegTensor<float> vScalefp32Res1;
    __ubuf__ int8_t *ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float *ubDstAddrTmp = ubDstAddr;
    Reg::MaskReg bf16TypeMaskAll = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg fp32MaskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<int8_t> &)vScale0, ubScaleSrcAddrTemp, 640);

        Reg::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        Reg::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vScalefp32Res0, 64, bf16TypeMaskAll);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb, int64_t dealRow,
    int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * 576], kvOutUb, dealRow * 576); // 576: KV Cache 每 Token 的特征宽度
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                sfaSparseCalSize = CeilDiv(v0S2SizeFirstCore, 2); // 2: Vector split size for first core (first half)
                sparseS2Start = 0;
            } else {
                // 2: Vector split size for first core (second half)
                sfaSparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2); // 2: Start offset for second half of first core
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sfaSparseCalSize = CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sfaSparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);     // 2: Same as above
            }
        }
        sparseS2End = sparseS2Start + sfaSparseCalSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
        sparseS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        sfaSparseCalSize = sparseS2End - sparseS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo,
    int32_t startPos)
{
    blockSize = constInfo.blockSize;
    maxBlockNumPerBatch = constInfo.maxBlockNumPerBatch;

    CalSparseCalSize(runInfo, constInfo);
    ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE3>(15);  // 15: 跨核同步标志位值
        CrossCoreWaitFlag<0, PIPE_MTE3>(15); // 15: 跨核同步标志位值
    }
    outputL1.SetCrossCore();
    v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo,
    int32_t startPos)
{
    if (sfaSparseCalSize == 0) {
        return;
    }
    bool meetEnd = false;
    int64_t s2Start = sparseS2Start;
    int64_t s2 = sparseS2Start;
    int64_t token0Idx;
    int64_t token1Idx; // 拷贝进入的两个token的index
    // 处理一个s2的base块
    uint32_t pingPong = 0;
    while ((s2 < sparseS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        LocalTensor<Q_T> stage0OutUb = this->stage0OutBuf[pingPong].template Get<Q_T>();
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
        while (dealRow < Min(16, sfaSparseCalSize) && s2 < sparseS2End) { // 拷贝满16行或者遇到-1
            GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
            s2 += 2; // 每次搬运2行
            if (token0Idx == -1 && token1Idx == -1) {
                meetEnd = true;
                break;
            }
            dealRow += CopyInKvSparse(stage0OutUb, dealRow, token0Idx, token1Idx, runInfo, constInfo);
            if (token1Idx == -1) {
                meetEnd = true;
                break;
            }
        }
        if (dealRow == 0) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            pingPong ^= 1;
            return;
        }
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
        // 2、copy kv out, ub -> l1
        CopyOutKvUb2Gm(v0ResGm, stage0OutUb, dealRow, s2Start, runInfo, constInfo);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
        s2Start += dealRow;
        pingPong ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> sfaExpUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    ProcessVec1SoftmaxDispatchSFA(stage1CastTensor, mmRes, sumUb, maxUb, apiTmpBuffer, runInfo, constInfo);

    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeV);
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize, (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, sfaExpUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if (constInfo.returnSoftmaxLse && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        CopyFALseToGm(runInfo, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec1SoftmaxDispatchSFA(
    LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
    LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo, ConstInfo &constInfo)
{
    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (runInfo.s2LoopCount == 0 && !isSinks) { // sink 丢失首token信息，sink会增加首token信息，维度是n1
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, // 实际的计算有效元素，
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (runInfo.s2LoopCount == 0 && isSinks) {
            // vec0: 0 ~ halfMRealSize - 1, vec1: firstHalfMRealSize ~ gSize
            int64_t sinksOffset = runInfo.goIdx;
            if (constInfo.subBlockIdx == 1) {
                sinksOffset += runInfo.firstHalfMRealSize;
            }
            LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
            InitSoftmaxFromSinks<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, runInfo.halfMRealSize);
        }
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t sfaVec2CalcSize = runInfo.vec2MRealSize * sfaDTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

    WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, sfaVec2CalcSize);
    } else {
        LocalTensor<T> sfaExpUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, sfaDTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, sfaExpUb, sfaExpUb, runInfo.vec2MRealSize, sfaDTemplateAlign64, 1.0, 1.0);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, sfaDTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, sfaExpUb, sfaExpUb, sumUb, runInfo.vec2MRealSize, sfaDTemplateAlign64, 1.0,
                1.0);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, sfaDTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize,
                                                                     sfaDTemplateAlign64, 1.0);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, sfaVec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                        LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                        int64_t vec2S1Idx, int64_t sfaVec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)sfaDTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, sfaVec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                            LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                            int64_t vec2S1Idx, int64_t sfaVec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, sfaVec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitOutputSingleCore(
    ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;
    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (constInfo.aivIdx * singleCoreSize < totalOutputSize && singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    if (constInfo.returnSoftmaxLse) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
            totalReturnSoftmaxSize = constInfo.bSize * constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            totalReturnSoftmaxSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize; // (N2,T1,G)
        }
        if (coreNum != 0 && totalReturnSoftmaxSize > 0) {
            uint64_t singleCoreSoftmaxSize = (totalReturnSoftmaxSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            uint64_t tailSoftmaxSize = totalReturnSoftmaxSize - constInfo.aivIdx * singleCoreSoftmaxSize;
            uint64_t singleInitSoftmaxSize =
                tailSoftmaxSize < singleCoreSoftmaxSize ? tailSoftmaxSize : singleCoreSoftmaxSize;
            if (constInfo.aivIdx * singleCoreSoftmaxSize < totalReturnSoftmaxSize && singleInitSoftmaxSize > 0) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                matmul::InitOutput<float>(this->softmaxSumGm[constInfo.aivIdx * singleCoreSoftmaxSize],
                                          singleInitSoftmaxSize, 0);
                SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                matmul::InitOutput<float>(this->softmaxMaxGm[constInfo.aivIdx * singleCoreSoftmaxSize],
                                          singleInitSoftmaxSize, 0);
                SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            }
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CleanOutput(
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)(softmaxSum));
        this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)(softmaxMax));
        if (constInfo.needInit == 1) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore(constInfo);
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *sinks)
{
    keyGm.SetGlobalBuffer((__gm__ KV_T *)(key));
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        ;
    }
    sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    keyRopeGm.SetGlobalBuffer((__gm__ KV_T *)(keyRope));
    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyFALseToGm(RunInfo &runInfo,
                                                                                               ConstInfo &constInfo)
{
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();

    size_t alignedSize = (sizeof(float) * runInfo.halfMRealSize + 31) / 32 * 32 / sizeof(float);

    int64_t lseOffset = runInfo.softmaxLseOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(float) * runInfo.halfMRealSize;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    // 拷贝 softmaxMaxUb -> GM
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, maxUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxMaxGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    // 拷贝 softmaxSumUb -> GM
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, sumUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxSumGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &constInfo)
{
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe,
                                                                                                 ConstInfo &constInfo)
{
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B
    tPipe->InitBuffer(lseBuf, 512);     // lseBuf内存申请512B
    lseUb = this->lseBuf.template Get<float>();

    tPipe->InitBuffer(stage0OutBuf[0],
                      576 * 16 * sizeof(KV_T)); // 576: 模型特征维度(dSize); 16 : 注意力头数 (num_heads)
    tPipe->InitBuffer(stage0OutBuf[1],
                      576 * 16 * sizeof(KV_T)); // 576: 模型特征维度(dSize); 16 : 注意力头数 (num_heads)

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * sfaDTemplateAlign64 * sizeof(T));

    mte3ToVAttnOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    vToMte3AttnOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();

    mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte3ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    mte3ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    mte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }

    initOutputEventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::UninitLocalBuffer()
{
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVAttnOutId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVLseOutId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3AttnOutId);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3LseOutId);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(initOutputEventId);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sharedParams.bSize = sparseAttnSharedkvBaseParams.batchSize;
    sharedParams.n2Size = 1;
    sharedParams.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup;
    sharedParams.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sharedParams.s2Size = sparseAttnSharedkvBaseParams.seqSize;
    sharedParams.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    sharedParams.maskMode = sparseAttnSharedkvBaseParams.sparseMode;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout;
    sharedParams.dSizeRope = 64; // 64: 编码维度
    sharedParams.softmaxScale = sparseAttnSharedkvBaseParams.scaleValue;
    sharedParams.dSize = 512;       // 512: 模型特征维度(dSize)
    sharedParams.dSizeVInput = 512; // 512: 模型特征维度(dSize)
    sharedParams.usedCoreNum = this->tilingData->singleCoreParams.usedCoreNum;

    // pageAttention, rope在C侧搬运时使用
    if constexpr (isPa) {
        sharedParams.blockSize = sparseAttnSharedkvBaseParams.blockSize;
        sharedParams.maxBlockNumPerBatch = sparseAttnSharedkvBaseParams.maxBlockNumPerBatch;
    }

    // actQ->TND, actKV pa场景任意layout均有
    sharedParams.isActualSeqLengthsNull = sparseAttnSharedkvBaseParams.isActualLenDimsNull;
    sharedParams.isActualSeqLengthsKVNull = sparseAttnSharedkvBaseParams.isActualLenDimsKVNull;
    sharedParams.returnSoftmaxLse = sparseAttnSharedkvBaseParams.returnSoftmaxLse;

    ComputeNeedInitSFA(sharedParams);

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t *>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ComputeNeedInitSFA(
    CVSharedParams &sharedParams) const
{
    // Sparse mode 0 can have no selected local keys even when the KV cache is nonempty.
    // Skipped queries must retain zero output and zero max/sum (LSE = -inf).
    sharedParams.needInit = sharedParams.returnSoftmaxLse && sharedParams.maskMode == 0;
    for (uint32_t bIdx = 0; bIdx < sharedParams.bSize; bIdx++) {
        int64_t s2Size;
        if constexpr (KV_LAYOUT_T == SFA_LAYOUT::TND) {
            s2Size = bIdx == 0 ? actualSeqLengthsKVGm.GetValue(bIdx) :
                                 actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsKVNull) {
                s2Size = sharedParams.s2Size;
            } else {
                s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            }
        }
        int64_t s1Size;
        if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            s1Size =
                bIdx == 0 ? cuSeqlensQGm.GetValue(bIdx) : cuSeqlensQGm.GetValue(bIdx) - cuSeqlensQGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsNull) {
                s1Size = sharedParams.s1Size;
            } else {
                s1Size = cuSeqlensQGm.GetValue(bIdx);
            }
        }
        if (s1Size > s2Size || (LAYOUT_T == SFA_LAYOUT::BSND && s1Size < sharedParams.s1Size)) {
            sharedParams.needInit = 1;
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF class SFAVectorServiceDummy {
public:
    __aicore__ inline SFAVectorServiceDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                       __gm__ uint8_t *softmaxSum, ConstInfo &constInfo)
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *keyRope,
                                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable,
                                            __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                            __gm__ uint8_t *sinks)
    {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionTilingDataMla *__restrict tiling,
                                        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                        __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void UninitLocalBuffer() {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo)
    {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo) {}
};
} // namespace BaseApi
#endif // SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_ARCH35_H
