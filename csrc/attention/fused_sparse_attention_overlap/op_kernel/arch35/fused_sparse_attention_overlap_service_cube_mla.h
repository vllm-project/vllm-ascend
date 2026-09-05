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
 * \file fused_sparse_attention_overlap_service_cube_mla.h
 * \brief
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_CUBE_MLA_ARCH35_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_CUBE_MLA_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_sparse_attention_overlap_common_arch35.h"
#include "util_regbase.h"

#if __has_include("../../common/op_kernel/offset_calculator.h")
#include "../../common/op_kernel/offset_calculator.h"
#else
#include "../common/offset_calculator.h"
#endif
#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {

template <FusedSparseAttentionOverlapLayoutArch35 LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == FusedSparseAttentionOverlapLayoutArch35::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class FusedSparseAttentionOverlapMatmulService {
public:
    /* =================Compile-time base-block constants================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 576;
    static constexpr uint32_t dBaseMatmulSize = 128;

    __aicore__ inline FusedSparseAttentionOverlapMatmulService() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> &l1BuffMgr,
                                         __gm__ uint8_t *query, __gm__ uint8_t *queryRope);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                        __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, const ConstInfo& constInfo);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer(BufferManager<BufferType::L1> &l1BuffMgr);
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);

    __aicore__ inline void IterateBmm1SFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2SFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM variables==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT, int32_t> queryGm;
    FaGmTensor<Q_T, Q_FORMAT, int32_t> queryRopeGm;

    FaGmTensor<KV_T, GmFormat::PA_BnBsND> keyGm;
    GlobalTensor<int32_t> blockTableGm;
    FaGmTensor<KV_T, GmFormat::PA_BnBsND> curKvGm;
    GlobalTensor<int32_t> cuSeqlensQGm;

    /* =====================Runtime variables==================== */
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    TEventID mte1ToMte2Id[3];
    TEventID mte2ToMte1Id[3];

    /* =====================Local-buffer variables==================== */
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    // When D <= 256, reuse the MM1 left matrix Q within a GS1 loop and ping-pong across GS1 loops.
    // When D > 256, use one buffer and keep it resident across S1 loops; one FP32 buffer is not kept resident.
    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers;

    // L0A
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    // L0B
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    // L0C
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, BufferManager<BufferType::L1> &l1BuffMgr, __gm__ uint8_t *query, __gm__ uint8_t *queryRope)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        this->queryRopeGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)queryRope);
        InitLocalBuffer(l1BuffMgr);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ,
    const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        mte1ToMte2Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        mte1ToMte2Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        mte1ToMte2Id[2] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        mte2ToMte1Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        mte2ToMte1Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        mte2ToMte1Id[2] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        InitGmTensor(actualSeqLengthsQ, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::InitLocalBuffer(BufferManager<BufferType::L1> &l1BuffMgr)
{
    constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(Q_T);
    l1QBuffers.Init(l1BuffMgr, mm1LeftSize);

    // L0A, L0B, and L0C are currently hard-coded; determine whether the base API can provide them.
    l0aBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0bBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0cBufferManager.Init(tPipe, L0C_SHARED_SIZE_256K);

    mmL0ABuffers.Init(l0aBufferManager, BUFFER_SIZE_16K); // Double-buffered; the value is half of the total size
    mmL0BBuffers.Init(l0bBufferManager, BUFFER_SIZE_32K);
    mmL0CBuffers.Init(l0cBufferManager, BUFFER_SIZE_128K);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *actualSeqLengthsQ, const ConstInfo& constInfo)
{
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayoutArch35::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
        this->queryRopeGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSizeRope);
    } else {  // FusedSparseAttentionOverlapLayoutArch35::TND
        GlobalTensor<int32_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            actualSeqQLen, constInfo.actualSeqLenSize);
        this->queryRopeGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeRope,
            actualSeqQLen, constInfo.actualSeqLenSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,  RunInfo &runInfo,
    ConstInfo &constInfo)
{
    IterateBmm1SFA(outputBuf, inputRightBuf, v0ResGm, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::IterateBmm2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    IterateBmm2SFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::IterateBmm1SFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    Buffer<BufferType::L1> inputLeftBuf;
    // Reuse the left matrix and load it on the first S2 iteration.
    // Load the entire left matrix into L1.
    if (unlikely(runInfo.s2LoopCount == 0)) { // First base block in the sOuter loop: copy Q
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>(); // Acquire L1A
        LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();
        uint32_t s1Coord = runInfo.s1oIdx * runInfo.qSNumInOneBlock;
        uint64_t queryGmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx,
            runInfo.goIdx, s1Coord, 0);
        uint64_t queryRopeGmOffset = this->queryRopeGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx,
            runInfo.goIdx, s1Coord, 0);
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[queryGmOffset],
            runInfo.mRealSize, 512, 512); // 64 constInfo.dSize constInfo.mm1Ka
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor[Align16Func(runInfo.mRealSize) * 512],
            this->queryRopeGm.gmTensor[queryRopeGmOffset], runInfo.mRealSize,
            64, 64); // constInfo.dSize constInfo.mm1Ka
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // Notify
    } else { // Reuse Q directly after the first S2 iteration
        inputLeftBuf = l1QBuffers.GetPre();
        // Reusing the left matrix removes the need for MTE2 synchronization waits within the sInner loop.
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // Notify
    }

    inputRightBuf.WaitCrossCore();
    SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
    WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
    LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
    v0ResGm.WaitCrossCore();
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    CopyToL1Nd2Nz<Q_T>(dst, v0ResGmTensor, runInfo.s2RealSize, 576, 576);
    SetFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
    WaitFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);

    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>(); // Wait for L1A
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // Acquire
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                     static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                     static_cast<uint32_t>(constInfo.dSizeNope + constInfo.dSizeRope),   // singleK
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(), // MM1 B uses tensor data directly
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        inputLeftBuf.Set<HardEvent::MTE1_MTE2>(); // Release L1A
    }

    mm1ResL0C.Set<HardEvent::M_FIX>();    // Notify
    mm1ResL0C.Wait<HardEvent::M_FIX>();   // Wait for L0C

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C to UB
    // N size of the BMM1 result matrix in L0C, equal to mmadParams.n; align to eight elements (32 bytes). // 128
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // If fewer than 16 rows are valid, output only those rows; M size of the BMM1 result matrix in L0C (must be even). // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // Stride between adjacent contiguous BMM1 result fragments in L0C, from one block start to the next,
    // in units of 16 * sizeof(T).
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    // Gap between rows in mmResUb, in elements. // 128: derived by comparing dumps; masks remove invalid data for ND (S1 * S2)
    fixpipeParams.dstStride = s2BaseSize;
    fixpipeParams.dualDstCtl = 1; // Dual-destination mode splits M; each UB receives M / 2 * N, and M must be even
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), \
        mm1ResL0C.GetTensor<T>(), fixpipeParams); // Copy the matmul result from L0C to UB
    mm1ResL0C.Set<HardEvent::FIX_M>(); // Release L0C
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<TEMPLATE_ARGS>::IterateBmm2SFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    inputRightBuf.WaitCrossCore();

    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // Acquire
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),   // singleM
                     static_cast<uint32_t>(constInfo.dSizeNope), // singleN
                     static_cast<uint32_t>(runInfo.s2RealSize),  // singleK
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputRightBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeNope), // Left matrix P starts at the RoPE position
        inputRightBuf.GetTensor<Q_T>(), // Right matrix V NoPE
        mmL0ABuffers,
        mmL0BBuffers,
        mm2ResL0C.GetTensor<T>(),
        param);

    inputRightBuf.SetCrossCore();   // Release KV here because it remains in use until BMM2

    mm2ResL0C.Set<HardEvent::M_FIX>();  // Notify
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // Wait

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;      // L0C to UB; FixpipeParamsM300 also copies L0C to UB
    fixpipeParams.nSize = Align8Func(constInfo.dSizeNope);      // N size of the BMM1 result matrix in L0C; use buckets and mask valid values in Vector2
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);        // If fewer than 16 rows are valid, output only those rows; M size of the BMM1 result matrix in L0C
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize); // Stride between adjacent contiguous BMM1 result fragments in L0C, from one block start to the next
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeNope);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mm2ResL0C.GetTensor<T>(), fixpipeParams); // Copy the matmul result from L0C to UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // Release

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class FusedSparseAttentionOverlapMatmulServiceDummy {
public:
    __aicore__ inline FusedSparseAttentionOverlapMatmulServiceDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe,
        BufferManager<BufferType::L1> &l1BuffMgr, __gm__ uint8_t *query, __gm__ uint8_t *queryRope) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable,
        __gm__ uint8_t *actualSeqLengthsQ, const ConstInfo& constInfo) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        RunInfo &runInfo, ConstInfo &constInfo) {}
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
};


template <typename T>
struct CubeBlockTraits;  // Declaration

/* Generate CubeBlockTraits. */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(FusedSparseAttentionOverlapMatmulService);
DEFINE_CUBE_BLOCK_TRAITS(FusedSparseAttentionOverlapMatmulServiceDummy);

// /* Generate argument traits; the kernel can use ARGS_TRAITS to obtain all CubeBlock template parameters. */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_CUBE_MLA_ARCH35_H
