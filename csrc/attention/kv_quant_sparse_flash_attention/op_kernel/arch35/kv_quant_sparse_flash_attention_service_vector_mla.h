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
 * \file kv_quant_sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

#include "kv_quant_sparse_flash_attention_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lib/matmul_intf.h"
#if __has_include("../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h")
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#else
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#endif
#if __has_include("../../common/op_kernel/arch35/vf/vf_flashupdate_new.h")
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#else
#include "../../common/arch35/vf/vf_flashupdate_new.h"
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
    static constexpr uint32_t dVTemplateTypeInput = 672;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;

    // ==================== Functions ======================
    __aicore__ inline QSFAVectorService() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling,
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
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo);
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

private:
    __aicore__ inline void ProcessVec1SoftmaxDispatchQSFA(LocalTensor<Q_T> &stage1CastTensor,
        LocalTensor<T> &mmRes, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
        LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &runInfo, ConstInfo &constInfo);
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
    const KvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> SparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

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


TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = runInfo.boIdx == 0 ? 0 : cuSeqlensQGm.GetValue(runInfo.boIdx - 1);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.sparseBlockCount +
            runInfo.s1oIdx * constInfo.sparseBlockCount; // B, S1, N2(1), K
    }

    int64_t qsfaCmpS2LoopCnt = runInfo.s2LoopCount;
    int64_t qsfaTopkIdx = s2IdxInBase + qsfaCmpS2LoopCnt * constInfo.s2BaseSize;

    if (unlikely(qsfaTopkIdx >= constInfo.sparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = SparseIndicesGm.GetValue(topkBS1Idx + qsfaTopkIdx) + runInfo.s2StartIdx;
    }
    qsfaTopkIdx += 1;
    if (unlikely((qsfaTopkIdx >= constInfo.sparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = SparseIndicesGm.GetValue(topkBS1Idx + qsfaTopkIdx) + runInfo.s2StartIdx;
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
            static_cast<int64_t>(blockSize) * constInfo.dSizeVInput +
            blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else {
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
            realkeyOffset = (runInfo.boIdx * constInfo.s2Size + s2Idx) * constInfo.dSizeVInput; // BSN(1)D
        } else if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            int64_t batchKvStart = (runInfo.boIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(runInfo.boIdx - 1);
            realkeyOffset = (batchKvStart + s2Idx) * constInfo.dSizeVInput;
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
    DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline uint32_t QSFAVectorService<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb , int64_t startRow,
    int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    uint32_t combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
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
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvInUb[startRow *  combineDimAlign], keyGm[keyOffset], intriParams, padParams);
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

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ int8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, // output first
    __ubuf__ float* ubScaleSrcAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 672; // 128对齐 640->672
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

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& outputUb,  LocalTensor<KV_T>& inputUb, uint32_t dealRowCount)
{
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb.GetPhyAddr()); // nope改成在左，所以起始位置是0
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(inputUb[512 + 64 * 2].GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, ConstInfo &constInfo)
{
    // srcTensor是nope(512) + nope(64) + scale + pad, dstTensor是nope(512) + rope(64)
    AntiquantVFFp8D448<Q_T, KV_T>(antiKvTensorAsB16, srcTensor, dealRow);

    LocalTensor<Q_T> kRopeUb = srcTensor[constInfo.dSizeNope].template ReinterpretCast<Q_T>();
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)]; // V0单次处理16行数据
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
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
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
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
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
    blockSize = constInfo.blockSize;
    maxBlockNumPerBatch = constInfo.maxBlockNumPerBatch;

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
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
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
    // 处理一个s2的base块
    while ((s2 < sparseS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        LocalTensor<KV_T> kvInUb = stage0InQue.AllocTensor<KV_T>();
        while (dealRow < Min(16, qsfaSparseCalSize) && s2<sparseS2End) { // 拷贝满16行或者遇到-1
            GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
            s2 += 2; // 每次搬运2行
            if (token0Idx== -1 && token1Idx == -1) {
                meetEnd = true;
                break;
            }
            dealRow += CopyInKvSparse(kvInUb, dealRow, token0Idx, token1Idx, runInfo, constInfo);
            if (token1Idx == -1) {
                meetEnd = true;
                break;
            }
        }
        if (dealRow  == 0) {
            stage0InQue.FreeTensor(kvInUb);
            return;
        }
        stage0InQue.EnQue(kvInUb);
        kvInUb = stage0InQue.DeQue<KV_T>();

        // 2、dequant by vf
        LocalTensor<Q_T> kvDequantOutUb = stage0OutQue.AllocTensor<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRow, constInfo);
        stage0InQue.FreeTensor(kvInUb);
        stage0OutQue.EnQue(kvDequantOutUb);
        kvDequantOutUb = stage0OutQue.DeQue<Q_T>();

        // 3、copy kv out, ub -> l1
        if constexpr (IS_SPLIT_G) {
            CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRow, s2Start, runInfo, constInfo);
        } else {
            CopyOutKvUb2L1(outputL1, kvDequantOutUb, dealRow, s2Start, runInfo, constInfo);
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
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo.needInit == 1) {
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
    // 576: 模型特征维度(dSize)
    tPipe->InitBuffer(stage0OutQue, 2, 576 * (16 + 1) * sizeof(Q_T)); // kv输入D轴640, V0阶段每次处理16个seq, 开2 buffer

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
    sharedParams.maskMode = sparseAttnSharedkvBaseParams.sparseMode;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout;
    sharedParams.dSizeRope = 64; // 64: 编码维度
    sharedParams.softmaxScale = sparseAttnSharedkvBaseParams.scaleValue;
    sharedParams.dSize = 576; // 576: 模型特征维度(dSize)
    sharedParams.dSizeVInput = sparseAttnSharedkvBaseParams.dSizeVInput;
    sharedParams.usedCoreNum = this->tilingData->singleCoreParams.usedCoreNum;
    if constexpr (isPa) {
        sharedParams.blockSize = sparseAttnSharedkvBaseParams.blockSize;
        sharedParams.maxBlockNumPerBatch = sparseAttnSharedkvBaseParams.maxBlockNumPerBatch;
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

TEMPLATES_DEF class QSFAVectorServiceDummy {
public:
    __aicore__ inline QSFAVectorServiceDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseFlashAttentionTilingDataMla *__restrict tiling,
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
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
