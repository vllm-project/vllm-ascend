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
 * \file kv_quant_sparse_flash_attention_vllm_service_vector_mla.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_SERVICE_VECTOR_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_SERVICE_VECTOR_MLA_H

#include "kv_quant_sparse_flash_attention_vllm_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lib/matmul_intf.h"
#if __has_include("../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h")
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#else
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#endif
#if __has_include("../../common/op_kernel/arch35/vf/vf_flashupdate_new.h")
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#else
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {

TEMPLATES_DEF
class QSFAVectorService {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t qsfaDTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = NO_ROPE ? 544 : 672;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;

    // ==================== Functions ======================
    __aicore__ inline QSFAVectorService() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseFlashAttentionVllmTilingDataMla *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths)
    {
        if ASCEND_IS_AIV {
            tilingData = tiling;
            tPipe = pipe;
            if (actualSeqLengths != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths);
            }
            if (actualSeqLengthsQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ);
            }

            this->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

    __aicore__ inline void GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx,
                                        uint32_t gS1StartIdx, uint32_t nextGs1Idx, __gm__ uint8_t* workspace,
                                        ConstInfo& qsfaConstInfo);

private:
    __aicore__ inline void CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx, int64_t s1Idx,
                                           int64_t validS2, int64_t alignNum, ConstInfo& qsfaConstInfo);
    __aicore__ inline void CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx, ConstInfo& qsfaConstInfo);
    __aicore__ inline void CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx,
                                             int64_t validS2, ConstInfo& qsfaConstInfo);
    __aicore__ inline int32_t GetActualS1Size(uint32_t bIdx, ConstInfo& qsfaConstInfo);
    __aicore__ inline int32_t GetActualS2Size(uint32_t bIdx, ConstInfo& qsfaConstInfo);
    __aicore__ inline void GetRealS2Addr(int64_t& token0Idx, int64_t& token1Idx, int64_t s2IdxInBase,
                                         int64_t topkBS1Idx, int64_t s2LoopOffset, const RunInfo& qsfaRunInfo,
                                         ConstInfo& qsfaConstInfo);
    __aicore__ inline uint32_t CopyInKvHoisted(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
                                               int64_t token1Idx, const RunInfo& info, ConstInfo& ci,
                                               DataCopyExtParams& params, const DataCopyPadExtParams<int8_t>& padding);
    __aicore__ inline void ProcessVec1SoftmaxDispatchQSFA(LocalTensor<Q_T> &stage1CastTensor,
        LocalTensor<T> &mmRes, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
        LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        int64_t topkBS1Idx, int64_t s2LoopOffset, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
        int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb , int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
        ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void CopyOutMrgeResult(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        int64_t mte2Size, int64_t mte3Size, int64_t s2keyOffset, int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
        int64_t keyOffset, uint32_t combineBytes);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t qsfaVec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(
        RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
        int64_t qsfaVec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void ComputeNeedInitQSFA(CVSharedParams &sharedParams) const;
    __aicore__ inline void GetExtremeValue(T &negativeScalar);

    TPipe *tPipe;
    const KvQuantSparseFlashAttentionVllmTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    bool returnSoftmaxLse = false;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> SparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<uint32_t> kvPhyAddrGm;

    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TQue<QuePosition::VECIN, 2> stage0InQue; // for v0 input, 2份表示可能存在pingpong
    TQue<QuePosition::VECOUT, 2> stage0OutQue; // for v0 output, 2份表示可能存在pingpong
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];

    T negativeFloatScalar;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t qsfaSparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};


TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(
    int64_t& token0Idx, int64_t& token1Idx, int64_t s2IdxInBase, int64_t topkBS1Idx, int64_t s2LoopOffset,
    const RunInfo& qsfaRunInfo, ConstInfo& qsfaConstInfo)
{
    int64_t topkKIdx = s2IdxInBase + s2LoopOffset;
    if (unlikely(topkKIdx >= qsfaConstInfo.sparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = SparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + qsfaRunInfo.s2StartIdx;
    }
    topkKIdx += 1;
    if (unlikely((topkKIdx >= qsfaConstInfo.sparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = SparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + qsfaRunInfo.s2StartIdx;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::GetRealS2Addr(
    int64_t& token0Idx, int64_t& token1Idx, int64_t s2IdxInBase, int64_t topkBS1Idx, int64_t s2LoopOffset,
    const RunInfo& qsfaRunInfo, ConstInfo& qsfaConstInfo)
{
    int64_t topkKIdx = s2IdxInBase + s2LoopOffset;
    GlobalTensor<int64_t> kvPhyAddrGm64 = kvPhyAddrGm.template ReinterpretCast<int64_t>();
    if (unlikely(topkKIdx >= qsfaConstInfo.sparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = kvPhyAddrGm64.GetValue(topkBS1Idx + topkKIdx);
    }
    topkKIdx += 1;
    if (unlikely((topkKIdx >= qsfaConstInfo.sparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = kvPhyAddrGm64.GetValue(topkBS1Idx + topkKIdx);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline int64_t QSFAVectorService<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) *
            static_cast<int64_t>(blockSize) * (NO_ROPE ? 528 : 656) +
            blkTableOffset * (NO_ROPE ? 528 : 656); // BlockNum, BlockSize, N(1), D
    } else {
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
            realkeyOffset = (runInfo.boIdx * constInfo.s2Size + s2Idx) * (NO_ROPE ? 528 : 656); // BSN(1)D
        } else if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            int64_t batchKvStart = (runInfo.boIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(runInfo.boIdx - 1);
            realkeyOffset = (batchKvStart + s2Idx) * (NO_ROPE ? 528 : 656);
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
QSFAVectorService<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
    int64_t keyOffset, uint32_t combineBytes)
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
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    // Match the aligned packed-row pitch used by the dequantizer.
    DataCopyPad(kvInUb[startRow * dVTemplateTypeInput], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline uint32_t QSFAVectorService<TEMPLATE_ARGS>::CopyInKvHoisted(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx, int64_t token1Idx, const RunInfo& info,
    ConstInfo& ci, DataCopyExtParams& params, const DataCopyPadExtParams<int8_t>& padding)
{
    constexpr uint32_t packedBytes = NO_ROPE ? 528 : 656;
    constexpr uint32_t packedStride = NO_ROPE ? 544 : 672;
    int64_t p0 = token0Idx;
    int64_t p1 = token1Idx;
    if constexpr (!IS_VEC_S2PHYADDR) {
        p0 = GetkeyOffset(token0Idx, info, ci);
        p1 = GetkeyOffset(token1Idx, info, ci);
    }
    if (p0 < 0 || p1 < 0) {
        return CopyInKvSparse(kvInUb, startRow, token0Idx, token1Idx, info, ci);
    }
    int64_t stride = (p0 > p1 ? p0 - p1 : p1 - p0) - packedBytes;
    bool mergePair = !IS_VEC_S2PHYADDR;
    if (mergePair && p0 >= 0 && p1 >= 0 && stride >= 0 && stride < INT32_MAX) {
        params.blockCount = 2;
        params.srcStride = stride;
        int64_t low = p0 < p1 ? p0 : p1;
        DataCopyPad(kvInUb[startRow * packedStride], keyGm[low], params, padding);
    } else {
        params.blockCount = 1;
        params.srcStride = 0;
        if (p0 >= 0)
            DataCopyPad(kvInUb[startRow * packedStride], keyGm[p0], params, padding);
        if (p1 >= 0)
            DataCopyPad(kvInUb[(startRow + (p0 >= 0)) * packedStride], keyGm[p1], params, padding);
    }
    return (p0 >= 0) + (p1 >= 0);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline uint32_t QSFAVectorService<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb , int64_t startRow,
    int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_VEC_S2PHYADDR) {
        uint32_t combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
        CopyInSingleKv(kvInUb, startRow, token0Idx, combineBytes);
        CopyInSingleKv(kvInUb, startRow + 1, token1Idx, combineBytes);
        return (token1Idx > -1) + (token0Idx > -1);
    }

    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    constexpr uint32_t combineBytes = NO_ROPE ? 528 : 656;
    int64_t keySrcStride = (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) * sizeof(KV_T):
        (keyOffset1 - keyOffset0)) * sizeof(KV_T) - combineBytes;
    if (keySrcStride >= INT32_MAX || keySrcStride < 0 || constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0, combineBytes);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, combineBytes);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }

        // 当前仅支持COMBINE模式
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        intriParams.dstStride = (dVTemplateTypeInput - combineDimAlign) * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvInUb[startRow * dVTemplateTypeInput], keyGm[keyOffset], intriParams, padParams);
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// int8->half
static constexpr MicroAPI::CastTrait castTraitint8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

// half->fp32
static constexpr MicroAPI::CastTrait castTraithalf_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename Q_T, typename KV_T, uint32_t packedRowStride>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ int8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, // output first
    __ubuf__ float* ubScaleSrcAddr, uint32_t dealRowCount)
{
    constexpr uint32_t combineDim = packedRowStride;
    MicroAPI::RegTensor<KV_T> vKvData0;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<half> vKvDataHalf0;
    MicroAPI::RegTensor<half> vKvDataHalf1;
    MicroAPI::RegTensor<half> vCastHalfRes0;
    MicroAPI::RegTensor<half> vCastHalfRes1;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<float> vMulRes0;
    MicroAPI::RegTensor<float> vMulRes1;
    MicroAPI::RegTensor<float> vScale0;
    MicroAPI::RegTensor<float> vScale1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vCastResPack0;
    MicroAPI::RegTensor<Q_T> vCastResPack1;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg int8MaskAll = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t blockStride = 17; // +1 to solve bank conflict
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 512;  // 448->512 64
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 1;
    const uint32_t tileSize = 128;
    static constexpr bool isKvInt8 = (IsSameType<KV_T, int8_t>::value);
    // tilesize is 128, deal 128 b8 kv, deal 1 fp32 scale
    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            // load scale
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData0, ubSrcTemp, tileSize / 2);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData1, ubSrcTemp, combineDim - tileSize / 2);

            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float>&)vScale0, ubScaleSrcAddrTemp, combineDim / 4);

            if constexpr (isKvInt8) {
                // int8 -> half
                MicroAPI::Cast<half, KV_T, castTraitint8_1>(vCastHalfRes0, vKvData0, int8MaskAll);
                MicroAPI::Cast<half, KV_T, castTraitint8_1>(vCastHalfRes1, vKvData1, int8MaskAll);
                // half -> float
                MicroAPI::Cast<float, half, castTraithalf_1>(vCastFp32Res0, vCastHalfRes0, fp32MaskAll);
                MicroAPI::Cast<float, half, castTraithalf_1>(vCastFp32Res1, vCastHalfRes1, fp32MaskAll);
            } else {
                MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
                MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);
            }

            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale0, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vMulRes1, fp32MaskAll);

            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, bool NO_ROPE>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& outputUb, LocalTensor<KV_T>& inputUb,
    uint32_t dealRowCount)
{
    constexpr uint32_t scaleOffset = NO_ROPE ? 512 : 640;
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb.GetPhyAddr()); // nope改成在左，所以起始位置是0
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(inputUb[scaleOffset].GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T, NO_ROPE ? 544 : 672>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, ConstInfo &constInfo)
{
    // External rows contain NoPE, optional RoPE, then four FP32 scales.
    AntiquantVFFp8D448<Q_T, KV_T, NO_ROPE>(antiKvTensorAsB16, srcTensor, dealRow);

    LocalTensor<Q_T> kRopeUb = srcTensor[constInfo.dSizeNope].template ReinterpretCast<Q_T>();
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)]; // V0单次处理16行数据
    if constexpr (NO_ROPE) {
        // Only the NoPE columns are consumed when the effective head dimension is 512.
        return;
    }
    Copy(kRopeUbNz, kRopeUb,
        constInfo.dSizeRope, // mask 处理多少列数据
        static_cast<uint8_t>(dealRow), // repeatTime, 每次处理多少个block
        {
            17, // dst stride
            1, // src stride
            1, // dst repeat stride
            21 // src repeat stride, 640 / 32   // 640 -> 672 : 20 -> 21
        });
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (NO_ROPE ? 512 : 576) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> antiKvTensorAsB16,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (NO_ROPE ? 512 : 576) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;
    DataCopy(v0ResGmTensor[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                qsfaSparseCalSize = CeilDiv(v0S2SizeFirstCore, 2); // 2: Vector split size for first core (first half)
                sparseS2Start = 0;
            } else {
                // 2: Vector split size for first core (second half)
                qsfaSparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2); // 2: Start offset for second half of first core
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                qsfaSparseCalSize = CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                qsfaSparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
            }
        }
        sparseS2End = sparseS2Start + qsfaSparseCalSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : s2Loops * s2PerVecLoop;
        sparseS2End = GetSubBlockIdx() == 0 ? s2Loops * s2PerVecLoop : runInfo.s2RealSize;
        qsfaSparseCalSize = sparseS2End - sparseS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    outputL1.WaitCrossCore(); // 核间同步
    blockSize = constInfo.oriBlockSize;
    maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;

    CalSparseCalSize(runInfo, constInfo);
    ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo);

    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15); // 15: 跨核同步标志位值
        CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE3>(15); // 15: 跨核同步标志位值
    }

    outputL1.SetCrossCore(); // 核间同步
    if constexpr (IS_SPLIT_G) {
        v0ResGm.SetCrossCore();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, const RunInfo& qsfaRunInfo,
    ConstInfo& qsfaConstInfo)
{
    if (qsfaSparseCalSize == 0) {
        return;
    }
    // Left-closed, right-open interval
    // 4x = 2x + 2x
    // 4x + 1 = (2x + 2) + (2x - 1)
    // 4x + 2 = (2x + 2) + (2x)
    // 4x + 3 = (2x + 2) + (2x + 1)
    int64_t s2Start = sparseS2Start;
    int64_t s2 = sparseS2Start;
    bool meetEnd = false;
    int64_t token0Idx, token1Idx; // 拷贝进入的两个token的index
    // 循环不变量外提：topkBS1Idx / s2LoopOffset / maxDealRow 在整个 s2 循环内不变
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = qsfaRunInfo.boIdx == 0 ? 0 : cuSeqlensQGm.GetValue(qsfaRunInfo.boIdx - 1);
        topkBS1Idx = (actualSeqQPrefixSum + qsfaRunInfo.s1oIdx) * qsfaConstInfo.sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx = qsfaRunInfo.boIdx * qsfaConstInfo.s1Size * qsfaConstInfo.sparseBlockCount +
                     qsfaRunInfo.s1oIdx * qsfaConstInfo.sparseBlockCount; // B, S1, N2(1), K
    }
    int64_t s2LoopOffset = qsfaRunInfo.s2LoopCount * qsfaConstInfo.s2BaseSize;
    int64_t maxDealRow = Min(16, qsfaSparseCalSize);
    DataCopyExtParams kvParams{1, NO_ROPE ? 528U : 656U, 0, 0, 0};
    DataCopyPadExtParams<int8_t> kvPadding{true, 0, 16, 0};

    // 处理一个s2的base块
    while ((s2 < sparseS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        LocalTensor<KV_T> kvInUb = stage0InQue.AllocTensor<KV_T>();
        while (dealRow < maxDealRow && s2 < sparseS2End) { // 拷贝满16行或者遇到-1
            if constexpr (IS_VEC_S2PHYADDR) {
                GetRealS2Addr(token0Idx, token1Idx, s2, topkBS1Idx, s2LoopOffset, qsfaRunInfo, qsfaConstInfo);
            } else {
                GetRealCmpS2Idx(token0Idx, token1Idx, s2, topkBS1Idx, s2LoopOffset, qsfaRunInfo, qsfaConstInfo);
            }
            s2 += 2; // 每次搬运2行
            if (token0Idx == -1 && token1Idx == -1) {
                meetEnd = true;
                break;
            }
            if constexpr (IsSameType<KV_T, int8_t>::value && !IS_VEC_S2PHYADDR) {
                if (qsfaConstInfo.sparseBlockSize == 1) {
                    dealRow += CopyInKvHoisted(kvInUb, dealRow, token0Idx, token1Idx, qsfaRunInfo, qsfaConstInfo,
                                               kvParams, kvPadding);
                } else {
                    dealRow += CopyInKvSparse(kvInUb, dealRow, token0Idx, token1Idx, qsfaRunInfo, qsfaConstInfo);
                }
            } else {
                dealRow += CopyInKvSparse(kvInUb, dealRow, token0Idx, token1Idx, qsfaRunInfo, qsfaConstInfo);
            }
            if (token1Idx == -1) {
                meetEnd = true;
                break;
            }
        }
        if (dealRow == 0) {
            stage0InQue.FreeTensor(kvInUb);
            return;
        }
        stage0InQue.EnQue(kvInUb);
        kvInUb = stage0InQue.DeQue<KV_T>();

        // 2、dequant by vf
        LocalTensor<Q_T> kvDequantOutUb = stage0OutQue.AllocTensor<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRow, qsfaConstInfo);
        stage0InQue.FreeTensor(kvInUb);
        stage0OutQue.EnQue(kvDequantOutUb);
        kvDequantOutUb = stage0OutQue.DeQue<Q_T>();

        // 3、copy kv out, ub -> l1
        if constexpr (IS_SPLIT_G) {
            CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRow, s2Start, qsfaRunInfo, qsfaConstInfo);
        } else {
            CopyOutKvUb2L1(outputL1, kvDequantOutUb, dealRow, s2Start, qsfaRunInfo, qsfaConstInfo);
        }
        s2Start += dealRow;
        stage0OutQue.FreeTensor(kvDequantOutUb);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> qsfaExpUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    ProcessVec1SoftmaxDispatchQSFA(stage1CastTensor, mmRes, sumUb, maxUb, apiTmpBuffer, runInfo, constInfo);

    bmm1ResBuf.SetCrossCore();
    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor =
        outputBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeV);

    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.mRealSize - runInfo.halfMRealSize)],
            stage1CastTensor, {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize,
            (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
            (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, qsfaExpUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec1SoftmaxDispatchQSFA(
    LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes, LocalTensor<float> &sumUb,
    LocalTensor<float> &maxUb, LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    if (runInfo.s2LoopCount == 0) {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();

    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    int64_t qsfaVec2CalcSize = runInfo.vec2MRealSize * qsfaDTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, qsfaVec2CalcSize);
    } else {
        LocalTensor<T> qsfaExpUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, qsfaDTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, qsfaExpUb, qsfaExpUb, runInfo.vec2MRealSize,
                qsfaDTemplateAlign64, 1.0, 1.0);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, qsfaDTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, qsfaExpUb, qsfaExpUb, sumUb, runInfo.vec2MRealSize,
                qsfaDTemplateAlign64, 1.0, 1.0);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, qsfaDTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, qsfaDTemplateAlign64, 1.0);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, qsfaVec2CalcSize);
        if (returnSoftmaxLse) {
            // A5 MLA has one KV head. LSE is [N2=1, T, query_heads].
            const uint64_t offset = runInfo.attentionOutOffset / constInfo.dSizeV;
            LocalTensor<float> maxUb = softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LocalTensor<float> sumUb = softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[1]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[1]);
            DataCopyExtParams statsCopy{1, static_cast<uint32_t>(runInfo.vec2MRealSize * sizeof(float)), 0, 0, 0};
            DataCopyPad(softmaxMaxGm[offset], maxUb, statsCopy);
            DataCopyPad(softmaxSumGm[offset], sumUb, statsCopy);
            SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
        }
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::Bmm2DataCopyOut (RunInfo &runInfo, ConstInfo &constInfo,
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t qsfaVec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)qsfaDTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, qsfaVec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, int64_t qsfaVec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, qsfaVec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        returnSoftmaxLse = this->tilingData->baseParams.returnSoftmaxLse;
        if (returnSoftmaxLse) {
            softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
            softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
            uint64_t total = constInfo.s1Size * constInfo.gSize;
            if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) { total *= constInfo.bSize; }
            const uint64_t count = (total + GetBlockNum() * CV_RATIO - 1) / (GetBlockNum() * CV_RATIO);
            const uint64_t begin = constInfo.aivIdx * count;
            if (begin < total) {
                const uint64_t size = (total - begin < count) ? total - begin : count;
                matmul::InitOutput<float>(softmaxMaxGm[begin], size, -3.402823466e38F);
                matmul::InitOutput<float>(softmaxSumGm[begin], size, 0);
                matmul::InitOutput<OUTPUT_T>(attentionOutGm[begin * constInfo.dSizeV], size * constInfo.dSizeV, 0);
            }
            SyncAll();
        }
        if (constInfo.needInit == 1 && !returnSoftmaxLse) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *key,
__gm__ uint8_t *value, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable)
{
    keyGm.SetGlobalBuffer((__gm__ KV_T *)(key));
    SparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(stage0InQue, 2, dVTemplateTypeInput * 16 * sizeof(KV_T)); // V0阶段每次处理16个seq, 开2 buffer
    // The dequantized NZ output contains only the effective QK columns.
    tPipe->InitBuffer(stage0OutQue, 2, (NO_ROPE ? 512 : 576) * (16 + 1) * sizeof(Q_T));

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * qsfaDTemplateAlign64 * sizeof(T));

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sharedParams.bSize = sparseAttnSharedkvBaseParams.batchSize;
    sharedParams.n2Size = 1;
    sharedParams.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sharedParams.s2Size = sparseAttnSharedkvBaseParams.seqSize;
    sharedParams.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup;

    sharedParams.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    sharedParams.oriMaskMode = sparseAttnSharedkvBaseParams.sparseMode;
    sharedParams.returnSoftmaxLse = sparseAttnSharedkvBaseParams.returnSoftmaxLse;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout;
    sharedParams.dSizeRope = NO_ROPE ? 0 : 64;
    sharedParams.softmaxScale = sparseAttnSharedkvBaseParams.scaleValue;
    sharedParams.dSize = NO_ROPE ? 512 : 576;
    sharedParams.dSizeVInput = sparseAttnSharedkvBaseParams.dSizeVInput;
    sharedParams.usedCoreNum = this->tilingData->singleCoreParams.usedCoreNum;
    if constexpr (isPa) {
        sharedParams.oriBlockSize = sparseAttnSharedkvBaseParams.blockSize;
        sharedParams.oriMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.maxBlockNumPerBatch;
    }

    sharedParams.isActualSeqLengthsNull = sparseAttnSharedkvBaseParams.isActualLenDimsNull;
    sharedParams.isActualSeqLengthsKVNull = sparseAttnSharedkvBaseParams.isActualLenDimsKVNull;

    ComputeNeedInitQSFA(sharedParams);

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto qsfaTempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);

            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++qsfaTempTilingSSbuf, ++tempTiling) {
                *qsfaTempTilingSSbuf = *tempTiling;
            }

            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ComputeNeedInitQSFA(
    CVSharedParams &sharedParams) const
{
    sharedParams.needInit = 0;
    for (uint32_t bIdx = 0; bIdx < sharedParams.bSize; bIdx++) {
        int64_t s2Size;
        if constexpr (KV_LAYOUT_T == QSFA_LAYOUT::TND) {
            s2Size = (bIdx == 0) ? actualSeqLengthsKVGm.GetValue(bIdx) : \
                (actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1));
        } else {
            if (sharedParams.isActualSeqLengthsKVNull) {
                s2Size = sharedParams.s2Size;
            } else {
                s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            }
        }

        int64_t s1Size;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            s1Size = (bIdx == 0) ? cuSeqlensQGm.GetValue(bIdx) : \
                (cuSeqlensQGm.GetValue(bIdx) - cuSeqlensQGm.GetValue(bIdx - 1));
        } else {
            if (sharedParams.isActualSeqLengthsNull) {
                s1Size = sharedParams.s1Size;
            } else {
                s1Size = cuSeqlensQGm.GetValue(bIdx);
            }
        }
        if (s1Size > s2Size || (LAYOUT_T == QSFA_LAYOUT::BSND && s1Size < sharedParams.s1Size)) {
            sharedParams.needInit = 1;
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t QSFAVectorService<TEMPLATE_ARGS>::GetActualS1Size(uint32_t bIdx, ConstInfo& qsfaConstInfo)
{
    int32_t actualS1Size = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        actualS1Size =
            (bIdx == 0) ? cuSeqlensQGm.GetValue(0) : (cuSeqlensQGm.GetValue(bIdx) - cuSeqlensQGm.GetValue(bIdx - 1));
    } else {
        actualS1Size = qsfaConstInfo.isActualLenDimsNull ? static_cast<int32_t>(qsfaConstInfo.s1Size) :
                                                           cuSeqlensQGm.GetValue(bIdx);
    }
    return actualS1Size;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t QSFAVectorService<TEMPLATE_ARGS>::GetActualS2Size(uint32_t bIdx, ConstInfo& qsfaConstInfo)
{
    int32_t actualS2Size = 0;
    if (qsfaConstInfo.isActualLenDimsKVNull) {
        actualS2Size = static_cast<int32_t>(qsfaConstInfo.s2Size);
    } else {
        actualS2Size = actualSeqLengthsKVGm.GetValue(bIdx);
    }
    return actualS2Size;
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFImpl(__ubuf__ uint32_t* kvPhyAddrUb, __ubuf__ int32_t* sparseIdxUb,
                                    __ubuf__ int32_t* blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                    const uint32_t blockSize, const int16_t shiftRightNum,
                                    const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t s2NumPerLoop = 128;
    static const uint16_t s2NumPerReg = 64;
    static const uint16_t outOffsetPerLoop = 256;
    static const uint16_t outOffsetPerReg = 128;
    static const uint32_t inValidValue = 0xFFFFFFFF;
    MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg add_carry_l_1;
    MicroAPI::MaskReg add_carry_h_1;
    MicroAPI::MaskReg add_carry_l_2;
    MicroAPI::MaskReg add_carry_h_2;
    MicroAPI::MaskReg preg_tail_neg_1_b32;
    MicroAPI::MaskReg preg_tail_neg_2_b32;
    MicroAPI::MaskReg preg_neg_idx_1;
    MicroAPI::MaskReg preg_neg_idx_2;
    MicroAPI::MaskReg preg_gather_1;
    MicroAPI::MaskReg preg_gather_2;

    MicroAPI::RegTensor<uint32_t> vreg_kvStride;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_2;
    MicroAPI::RegTensor<uint32_t> vreg_blockSize;
    MicroAPI::RegTensor<uint32_t> vreg_shift_rights_num;
    MicroAPI::RegTensor<uint32_t> vreg_pa_blk_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_blk_idx_2;
    MicroAPI::RegTensor<uint32_t> vreg_pa_tmp_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_tmp_2;
    MicroAPI::RegTensor<uint32_t> vreg_pa_offset_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_offset_2;
    MicroAPI::RegTensor<uint32_t> vreg_phy_offset_1;
    MicroAPI::RegTensor<uint32_t> vreg_phy_offset_2;
    MicroAPI::RegTensor<uint32_t> vreg_phy_blk_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_phy_blk_idx_2;

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_H_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_H_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_H_1;

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_H_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_H_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_H_2;

    MicroAPI::RegTensor<uint32_t> vreg_zero;
    MicroAPI::RegTensor<uint32_t> vreg_invalid;
    MicroAPI::Duplicate(vreg_zero, 0);
    MicroAPI::Duplicate(vreg_invalid, inValidValue);
    MicroAPI::Duplicate(vreg_kvStride, kvStride);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_1,
                                                              sparseIdxUb + i * s2NumPerLoop);
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_2,
                                                              sparseIdxUb + s2NumPerReg + i * s2NumPerLoop);
            // 原始 sparse 下标为 -1 的车道，物理地址高低 32 位都保持 0xFFFFFFFF
            MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg_neg_idx_1, vreg_sparse_idx_1, vreg_invalid, preg_all_b32);
            MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg_neg_idx_2, vreg_sparse_idx_2, vreg_invalid, preg_all_b32);
            // * sparseBlockSize
            MicroAPI::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            MicroAPI::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
            MicroAPI::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_all_b32);
            MicroAPI::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_all_b32);

            MicroAPI::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_all_b32);
            MicroAPI::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_all_b32);
            // offset
            MicroAPI::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_all_b32);
            MicroAPI::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_all_b32);
            // 物理页内offset
            MicroAPI::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_all_b32);
            MicroAPI::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_all_b32);

            // 原始下标为 -1 的车道不参与页表寻址，避免用 0xFFFFFFFF 右移后的页号越界读 block table
            MicroAPI::Not(preg_gather_1, preg_neg_idx_1, preg_all_b32);
            MicroAPI::Not(preg_gather_2, preg_neg_idx_2, preg_all_b32);
            MicroAPI::Gather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_gather_1);
            MicroAPI::Gather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_gather_2);

            // 分高低32位计算int64物理地址 -- 乘 stride
            // 低位乘 带进位
            MicroAPI::Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride,
                      preg_all_b32);
            MicroAPI::Mull(vreg_blk_id_mul_stride_L_2, vreg_mul_overflow_L_2, vreg_phy_blk_idx_2, vreg_kvStride,
                      preg_all_b32);

            // 分高低32位计算int64物理地址 -- 加 offset
            MicroAPI::Add(add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1, preg_all_b32);
            MicroAPI::Add(add_carry_l_2, vreg_total_offset_L_2, vreg_blk_id_mul_stride_L_2, vreg_phy_offset_2, preg_all_b32);

            MicroAPI::AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1,
                      preg_all_b32);
            MicroAPI::AddC(add_carry_h_2, vreg_total_offset_H_2, vreg_mul_overflow_L_2, vreg_zero, add_carry_l_2,
                      preg_all_b32);
            MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_1, inValidValue, preg_neg_idx_1);
            MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_1, inValidValue, preg_neg_idx_1);
            MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_2, inValidValue, preg_neg_idx_2);
            MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_2, inValidValue, preg_neg_idx_2);

            // 搬出 由于拆分为了int32类型，元素个数翻倍
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * outOffsetPerLoop, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + outOffsetPerReg + i * outOffsetPerLoop, vreg_total_offset_L_2, vreg_total_offset_H_2,
                preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        uint32_t tailCount1 = (s2Tail < s2NumPerReg) ? s2Tail : s2NumPerReg;
        uint32_t tailCount2 = (s2Tail > s2NumPerReg) ? (s2Tail - s2NumPerReg) : 0;
        MicroAPI::MaskReg preg_tail_1_b32 = MicroAPI::UpdateMask<int32_t>(tailCount1);
        MicroAPI::MaskReg preg_tail_2_b32 = MicroAPI::UpdateMask<int32_t>(tailCount2);
        MicroAPI::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        MicroAPI::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_1,
                                                          sparseIdxUb + i * s2NumPerLoop);
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_2,
                                                          sparseIdxUb + s2NumPerReg + i * s2NumPerLoop);
        // 只在本拍有效尾数里识别原始 -1，尾部之外仍由下面的 tail mask 填 -1
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg_neg_idx_1, vreg_sparse_idx_1, vreg_invalid, preg_tail_1_b32);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg_neg_idx_2, vreg_sparse_idx_2, vreg_invalid, preg_tail_2_b32);
        // * sparseBlockSize
        MicroAPI::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        MicroAPI::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
        MicroAPI::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_tail_1_b32);
        MicroAPI::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_tail_2_b32);

        MicroAPI::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_tail_1_b32);
        MicroAPI::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_tail_2_b32);
        // offset
        MicroAPI::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_tail_1_b32);
        MicroAPI::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_tail_2_b32);
        // 物理页内offset
        MicroAPI::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_tail_1_b32);
        MicroAPI::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_tail_2_b32);

        // 有效尾数里去掉原始 -1，避免用越界页号读 block table
        MicroAPI::Not(preg_gather_1, preg_neg_idx_1, preg_all_b32);
        MicroAPI::Not(preg_gather_2, preg_neg_idx_2, preg_all_b32);
        MicroAPI::And(preg_gather_1, preg_gather_1, preg_tail_1_b32, preg_all_b32);
        MicroAPI::And(preg_gather_2, preg_gather_2, preg_tail_2_b32, preg_all_b32);
        MicroAPI::Gather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_gather_1);
        MicroAPI::Gather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_gather_2);

        // 分高低32位计算int64物理地址 -- 乘 stride
        // 低位乘 带进位
        MicroAPI::Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride,
                  preg_tail_1_b32);
        MicroAPI::Mull(vreg_blk_id_mul_stride_L_2, vreg_mul_overflow_L_2, vreg_phy_blk_idx_2, vreg_kvStride,
                  preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 加 offset
        MicroAPI::Add(add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1, preg_tail_1_b32);
        MicroAPI::Add(add_carry_l_2, vreg_total_offset_L_2, vreg_blk_id_mul_stride_L_2, vreg_phy_offset_2, preg_tail_2_b32);

        MicroAPI::AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1,
                  preg_tail_1_b32);
        MicroAPI::AddC(add_carry_h_2, vreg_total_offset_H_2, vreg_mul_overflow_L_2, vreg_zero, add_carry_l_2,
                  preg_tail_2_b32);

        // 原始 sparse 下标为 -1 的位置，以及有效尾数之外的车道，都输出 int64 -1
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_1, inValidValue, preg_neg_idx_1);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_1, inValidValue, preg_neg_idx_1);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_2, inValidValue, preg_neg_idx_2);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_2, inValidValue, preg_neg_idx_2);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_1, inValidValue, preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_1, inValidValue, preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_2, inValidValue, preg_tail_neg_2_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_2, inValidValue, preg_tail_neg_2_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * outOffsetPerLoop, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(kvPhyAddrUb + outOffsetPerReg + i * outOffsetPerLoop,
                                                                  vreg_total_offset_L_2, vreg_total_offset_H_2,
                                                                  preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVF(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                      LocalTensor<int32_t> blkTableTensor, const uint16_t s2Loop, const uint32_t s2Tail,
                                      const uint32_t blockSize, const int16_t shiftRightNum,
                                      const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t* kvPhyAddrUb = (__ubuf__ uint32_t*)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t* sparseIdxUb = (__ubuf__ int32_t*)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t* blkTableUb = (__ubuf__ int32_t*)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFImpl<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize, shiftRightNum,
                                 sparseBlockSize, kvDim, kvStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb,
                                                                         int64_t bS1Idx, int64_t s1Idx, int64_t validS2,
                                                                         int64_t alignNum, ConstInfo& qsfaConstInfo)
{
    if (NO_ROPE || (qsfaConstInfo.sparseBlockCount & 127) != 0) {
        // Copy exact bytes: K need not be 128-aligned and rows must not overlap.
        int64_t stopCount = Min(static_cast<int64_t>(4), qsfaConstInfo.sparseBlockCount - validS2);
        DataCopyExtParams params{1, static_cast<uint32_t>((validS2 + stopCount) * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(this->kvPhyAddrGm[(bS1Idx + s1Idx) * qsfaConstInfo.sparseBlockCount * 2], kvPhyAddrUb, params);
        return;
    }
    static constexpr int64_t bytesPerBlock = 32;
    static constexpr int64_t phyAddrPerBlock = bytesPerBlock / sizeof(int64_t);
    bool appendSentinelBlock = validS2 % alignNum == 0 && validS2 + phyAddrPerBlock <= qsfaConstInfo.sparseBlockCount;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1U; // 每次处理1行
    dataCopyParams.blockLen =
        ((validS2 + alignNum - 1) / alignNum * alignNum) * sizeof(int64_t) / bytesPerBlock + appendSentinelBlock;
    dataCopyParams.srcGap = 0U;
    dataCopyParams.dstGap = 0U;
    // 2：每个位置对应key和value两个地址
    DataCopy(this->kvPhyAddrGm[(bS1Idx + s1Idx) * qsfaConstInfo.sparseBlockCount * 2], kvPhyAddrUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                                                         ConstInfo& qsfaConstInfo)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U; // 每次处理1行
    dataCopyParams.blockLen = qsfaConstInfo.oriMaxBlockNumPerBatch * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(blkTableUb, this->blockTableGm[bIdx * qsfaConstInfo.oriMaxBlockNumPerBatch], dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb,
                                                                           int64_t bS1Idx, int64_t s1Idx,
                                                                           int64_t validS2, ConstInfo& qsfaConstInfo)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = validS2 * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(sparseIdxUb, this->SparseIndicesGm[(bS1Idx + s1Idx) * qsfaConstInfo.sparseBlockCount], dataCopyParams,
                padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx,
                                                                      uint32_t bN2EndIdx, uint32_t gS1StartIdx,
                                                                      uint32_t nextGs1Idx, __gm__ uint8_t* workspace,
                                                                      ConstInfo& qsfaConstInfo)
{
    if (hasLoad == 0) {
        SyncAll();
        tPipe->Reset();
        return;
    }
    static constexpr uint16_t s2NumPerLoop = 128;
    int32_t blkSize = static_cast<int32_t>(qsfaConstInfo.oriBlockSize);
    // N1=128时相邻4个v核分核数据一致，N1=64时两个v核分核数据一致
    static constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 4 : 2;
    uint32_t vecCoreIdx = IS_SPLIT_G ? qsfaConstInfo.aivIdx % 4 : qsfaConstInfo.aivIdx % 2;

    // 计算右移位数
    int16_t shiftRightNum = 0;
    while (blkSize > 1) {
        blkSize >>= 1;
        shiftRightNum++;
    }

    // GM分配
    int64_t v0TotalOffset = 0;
    if constexpr (IS_SPLIT_G) {
        uint32_t v0ResSize = qsfaConstInfo.s2BaseSize * (NO_ROPE ? 512U : 576U) * sizeof(Q_T);
        v0TotalOffset = v0ResSize * 3 * (GetBlockNum() >> 1U); // 3buffer
    }
    this->kvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t*)(workspace + v0TotalOffset));

    const uint32_t kvStride = qsfaConstInfo.oriBlockSize * qsfaConstInfo.dSizeVInput;

    TBuf<> blkTableBuf;
    TBuf<> sparseIdxBuf;
    TBuf<> kvPhyAddrBuf;
    // blkTableBuf 对齐到 512B，保证后续 sparseIdxBuf 的 UB 地址满足 LoadAlign 512B 对齐要求
    tPipe->InitBuffer(blkTableBuf, ((qsfaConstInfo.oriMaxBlockNumPerBatch * sizeof(int32_t) + 511U) / 512U) * 512U);
    tPipe->InitBuffer(sparseIdxBuf, ((qsfaConstInfo.sparseBlockCount + s2NumPerLoop - 1) / s2NumPerLoop) *
                                        s2NumPerLoop * sizeof(int32_t));
    const uint32_t alignedSparseBlockCount =
        ((qsfaConstInfo.sparseBlockCount + s2NumPerLoop - 1) / s2NumPerLoop) * s2NumPerLoop;
    tPipe->InitBuffer(kvPhyAddrBuf, alignedSparseBlockCount * sizeof(int64_t));
    LocalTensor<int32_t> blkTableUb = blkTableBuf.template Get<int32_t>();
    LocalTensor<int32_t> sparseIdxUb = sparseIdxBuf.template Get<int32_t>(); // 逐行处理
    LocalTensor<uint32_t> kvPhyAddrUb = kvPhyAddrBuf.template Get<uint32_t>();

    int64_t totalValidS1 = 0;
    uint32_t tmpGS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);
        int32_t actualS1Size = GetActualS1Size(bIdx, qsfaConstInfo);

        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        int32_t actualS2Size = GetActualS2Size(bIdx, qsfaConstInfo);

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t numerator = actualS2Size - actualS1Size + 1 + s1Idx;
            int32_t curValidS2 = (numerator > 0) ? Min((int32_t)qsfaConstInfo.sparseBlockCount, numerator) : 0;
            if (curValidS2 > 0) {
                totalValidS1++;
            }
        }
        tmpGS1Start = 0;
    }

    int64_t s1PerVecCore = totalValidS1 / vecCoreNum;
    int64_t s1Tail = totalValidS1 % vecCoreNum;
    int64_t curStart = s1PerVecCore * vecCoreIdx + Min((int64_t)vecCoreIdx, s1Tail);
    int64_t curCount = s1PerVecCore + (vecCoreIdx < (uint32_t)s1Tail ? 1 : 0); // 尾块均分到前几个vec

    if (curCount == 0) {
        SyncAll();
        tPipe->Reset();
        return;
    }

    int64_t validCounter = 0;
    int64_t processedCount = 0;
    tmpGS1Start = gS1StartIdx;
    bool done = false;

    SetFlag<AscendC::HardEvent::V_MTE2>(3);
    SetFlag<AscendC::HardEvent::V_MTE2>(4);
    SetFlag<AscendC::HardEvent::MTE3_V>(7);

    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !done; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);

        int32_t actualS1Size = GetActualS1Size(bIdx, qsfaConstInfo);
        int64_t bS1Idx = 0;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            bS1Idx = (bIdx == 0) ? 0 : cuSeqlensQGm.GetValue(bIdx - 1);
        } else {
            bS1Idx = qsfaConstInfo.s1Size * bIdx;
        }

        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        int32_t actualS2Size = GetActualS2Size(bIdx, qsfaConstInfo);

        WaitFlag<AscendC::HardEvent::V_MTE2>(3);
        CopyPaTableToUb(blkTableUb, bIdx, qsfaConstInfo);
        SetFlag<AscendC::HardEvent::MTE2_V>(8);
        WaitFlag<AscendC::HardEvent::MTE2_V>(8);

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t numerator = actualS2Size - actualS1Size + 1 + s1Idx;
            int32_t curValidS2 = (numerator > 0) ? Min((int32_t)qsfaConstInfo.sparseBlockCount, numerator) : 0;
            if (curValidS2 <= 0) {
                continue;
            }

            if (validCounter < curStart || validCounter >= curStart + curCount) {
                validCounter++;
                continue;
            }
            validCounter++;

            uint16_t s2Loop = (curValidS2 + s2NumPerLoop - 1) / s2NumPerLoop;
            int32_t s2Tail = curValidS2 - (s2Loop - 1) * s2NumPerLoop;
            WaitFlag<AscendC::HardEvent::V_MTE2>(4);
            CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, curValidS2, qsfaConstInfo);
            SetFlag<AscendC::HardEvent::MTE2_V>(6);

            WaitFlag<AscendC::HardEvent::MTE2_V>(6);
            WaitFlag<AscendC::HardEvent::MTE3_V>(7);
            GetKVPhyAddrVF<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, qsfaConstInfo.oriBlockSize,
                                     shiftRightNum, qsfaConstInfo.sparseBlockSize, qsfaConstInfo.dSizeVInput, kvStride);
            static constexpr int32_t phyAddrPerBlock = 4;
            if (curValidS2 % s2NumPerLoop == 0 &&
                ((NO_ROPE || (qsfaConstInfo.sparseBlockCount & 127) != 0) ?
                     curValidS2 < static_cast<int32_t>(qsfaConstInfo.sparseBlockCount) :
                     curValidS2 + phyAddrPerBlock <= static_cast<int32_t>(qsfaConstInfo.sparseBlockCount))) {
                // A full VF tail has no invalid lanes. Append one 32-B block of -1 as the CopyIn stop sentinel.
                Duplicate(kvPhyAddrUb[curValidS2 * 2], UINT32_MAX, phyAddrPerBlock * 2);
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(4);
            SetFlag<AscendC::HardEvent::V_MTE3>(5);
            WaitFlag<AscendC::HardEvent::V_MTE3>(5);
            CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, curValidS2, s2NumPerLoop, qsfaConstInfo);
            SetFlag<AscendC::HardEvent::MTE3_V>(7);

            processedCount++;
            if (processedCount >= curCount) {
                done = true;
                break;
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(3);
        tmpGS1Start = 0;
    }

    WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    WaitFlag<AscendC::HardEvent::V_MTE2>(4);
    WaitFlag<AscendC::HardEvent::MTE3_V>(7);
    SyncAll();
    tPipe->Reset();
}

TEMPLATES_DEF class QSFAVectorServiceDummy {
public:
    __aicore__ inline QSFAVectorServiceDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseFlashAttentionVllmTilingDataMla *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
        RunInfo &runInfo,
        ConstInfo &constInfo) {}
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
};
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_SERVICE_VECTOR_MLA_H
