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
 * \file flash_mla_with_kvcache_block_cube_noquant_mla.h
 * \brief arch35 flash_mla_with_kvcache block cube 非量化 MLA（由 fia_block_cube_noquant_mla.h 复制改名）；
 *        静态 tensor buffer + Mutex/cross-core 显式同步（镜像 flash_attn_block_cube_nd.h）。
 *        L0C 例外（表 2 注 3）：bmm2 单次 Mmad 足迹 128K，槽保持 2×128K 静态 + 2 Mutex（不得拆 4×64K）。
 */
#ifndef FLASH_MLA_WITH_KVCACHE_BLOCK_CUBE_NOQUANT_MLA_H_
#define FLASH_MLA_WITH_KVCACHE_BLOCK_CUBE_NOQUANT_MLA_H_
#if __has_include("../../../common/op_kernel/offset_calculator.h")
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/const_def.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#else
#include "../../common/offset_calculator.h"
#include "../../common/matmul.h"
#include "../../common/const_def.h"
#include "../../common/FixpipeOut.h"
#endif
#include "memory_copy_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "flash_mla_with_kvcache_type.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace FlashAttnKernel {

template <typename FA_T>
class FlashMlaWithKvcacheNoQuantMlaBlockCube {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr LayOutTypeEnum LAYOUT = FA_T::qLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();
    using SeqLensToolType = FlashMlaSeqLensTool<Q_MODE, KV_MODE>;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true}; // true: bmm2Write2Ub
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<FA_T::qLayout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<FA_T::qLayout, FA_T::kvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using ConstInfoX = ConstInfo_t<FlashMlaWithKvcacheKernelType::NO_QUANT>;

    // ================== 核间同步 ID（与 vec 侧逐字节一致，参照 flash_attn_block_cube_nd.h:67-75）==================
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_BMM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_BMM1_1 = 1U;
    static constexpr uint32_t CROSSCORE_BMM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_BMM2_1 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;
    // AIV0_AIV1_OFFSET 使用全局宏（attention/common/op_kernel/buffer.h:30，=16）

    // ================== 核内 Mutex ID（AIC 核内 0-26 自洽，参照 flash_attn_block_cube_nd.h:77-91）==================
    static constexpr uint32_t L0C_BUFFER_ID0 = 0U;
    static constexpr uint32_t L0C_BUFFER_ID1 = 1U;
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 2U; // Q 单槽（表 1），仅用 base
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 4U;
    static constexpr uint32_t KV_L1_BUFFER_ID1 = 5U;
    static constexpr uint32_t KV_L1_BUFFER_ID2 = 6U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 7U;
    static constexpr uint32_t L0A_BUFFER_ID1 = 8U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 9U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 10U;

    // ================== 静态布局常量（表 1，数值 = 现状容量）==================
    // UB CV 区（VECIN，与 vec 侧同一偏移别名）：
    //   mm2Res 2×64K @0（mm2ResultSize）、mm1Res 2×16K @128K（mm1ResultSize）
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(MM_T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(MM_T);
    // L1（A1）：Q 1×72K @0（576宽），KV/P 3×144K @72K（mm12RightSize；k==v，P 复用）
    static constexpr uint32_t L1_Q_BUFCNT = 1U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * 576 * sizeof(Q_T);
    static constexpr uint32_t L1_KVP_BUFCNT = 3U;
    static constexpr uint32_t L1_KVP_BUF_BYTES = s2BaseSize * 576 * sizeof(INPUT_T);
    // L0C（CO1）：2×128K = 256K 静态（表 2 注 3）
    static constexpr uint32_t L0C_BUFCNT = 2U;
    static constexpr uint32_t L0C_BUF_BYTES = 128U * 1024U;
    // L0A/L0B：静态 manager 64K + DB policy 2×32K
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;

    // buffer位置+用途+Buffers；使用时命名: 用途+buffer位置+Tensor
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvpBuffers_; // K/V/P 共用 ring（vec 写 P 的目标偏移注释见 vec 块）
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;
    // L0A/B
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0A> l0aBufferManager_;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0B> l0bBufferManager_;
    using L0APolicyType =
        BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    using L0BPolicyType =
        BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    L0APolicyType mmL0APolicy_;
    L0BPolicyType mmL0BPolicy_;

    /* =====================GM变量(with layout)==================== */
    // q/k_cache 为 576 宽单张量（nope 512 + rope 64 合并），无独立 queryRope/keyRope 张量；
    // query 与 key 的 GM stride 布局按 576 宽初始化（偏移切分见 CopyQueryAndRopeTile/CopyKeyAndRopeTile）。
    // seq-lens（cu_seqlens_q/cache_seqlens）为 INT32，ACTLEN_T=uint32_t。
    // WITH_ZERO_HEAD 仅对 TND（cu_seqlens 首零头）成立；BSND/BNSD 走 GM_Q_OUT_BNGSD 3-参实现
    // （offset_calculator_v2.h，无 WITH_ZERO_HEAD 形参），与 flash_attn IS_TND 同构。
    FaGmTensor<Q_T, Q_FORMAT, uint32_t, LAYOUT == LayOutTypeEnum::LAYOUT_TND> queryGm;
    FaGmTensor<KV_T, KV_FORMAT> kCacheGm;
    GlobalTensor<int32_t> blockTableGm;
    // seq-lens GM 张量与解析器所有权在 kernel 侧 FlashMlaSeqLensTool，本 block 只读引用
    SeqLensToolType &seqLensTool_;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    const ConstInfoX &constInfo;

    /*============================================================================== */
    __aicore__ inline FlashMlaWithKvcacheNoQuantMlaBlockCube(ConstInfoX &constInfo, SeqLensToolType &seqLensTool)
        : constInfo(constInfo),
          seqLensTool_(seqLensTool){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *kCache, __gm__ uint8_t *blockTable)
    {
        InitCubeInput(query, kCache, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        static_assert(mBaseSize == 64 && s2BaseSize == 128, "mBaseSize != 64 or s2BaseSize != 128");
        /*--------------------------------------------UB--------------------------------------------*/
        // CV 区与 vec 侧同一偏移别名（mm2@0..128K、mm1@128K..160K）
        uint32_t addrUb = 0;
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES);
        addrUb = UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);

        /*--------------------------------------------L1--------------------------------------------*/
        struct L1Layout {
            uint8_t qBuffers[L1_Q_BUFCNT][L1_Q_BUF_BYTES];
            uint8_t kvpBuffers[L1_KVP_BUFCNT][L1_KVP_BUF_BYTES];
        };
        static_assert(sizeof(L1Layout) <= 512 * 1024, "L1 buffer too large");
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, SIZE_OF_MEMBER(L1Layout, qBuffers));
        // KV/P 区从 A1 72K 起；vec 写 P 的目标偏移必须与本布局逐字节一致（表 2 注 2）：
        //   l1KvpBuffers_[loop%3 * L1_KVP_BUF_BYTES] + s2BaseSize*dVBaseSize（rope 段）
        l1KvpBuffers_ =
            LocalTensor<uint8_t>(TPosition::A1, L1_Q_BUFCNT * L1_Q_BUF_BYTES, SIZE_OF_MEMBER(L1Layout, kvpBuffers));

        /*--------------------------------------------L0A/B--------------------------------------------*/
        l0aBufferManager_.Init(BUFFER_SIZE_BYTE_64K);
        l0bBufferManager_.Init(BUFFER_SIZE_BYTE_64K);

        /*--------------------------------------------L0C--------------------------------------------*/
        // 2×128K 静态 + M/FIX Mutex 对（表 2 注 3：bmm2 单次足迹 128K，不得拆 4×64K）
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *kCache, __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        // query 576 宽单张量（nope 512 + rope 64 合并）——GM stride 按 576 初始化，
        // 供 CopyQueryAndRopeTile 的 rope 段按 dIdx=512 偏移定位
        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                    constInfo.dSize + constInfo.dSizeRope, seqLensTool_.actualSeqLengthsGmQ, constInfo.actualSeqLenSize,
                    queryGm, query);

        // k_cache 为普通 ND 张量（非 TensorList）：直接取裸指针，
        // 不能用 ListTensorDesc 解码（会把 k_cache 数据首部误解析为 list 描述符 → GM 越界）

        // k_cache 576 宽单张量（nope 512 + rope 64 合并）——key 与 value 同源（k==v）：
        // bmm2 的 V 段 = L1 上 nope 区（512 宽，mm1 后仍留在 L1），无独立 value GM 张量/读取
        // 仅 PA 实例化（host 强制 PA 路由），非 PA 分支已删
        InitKVBuffer(constInfo.n2Size, constInfo.blockSize, constInfo.dSize + constInfo.dSizeRope, kCacheGm, kCache,
                     constInfo.keyStrides.bnStride, constInfo.keyStrides.n2Stride);
    }

    __aicore__ inline void InitQBuffer(
        uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize, uint32_t headDim,
        GlobalTensor<uint32_t> actualSeqLengthsGmQ, uint32_t actualLenQDims,
        FaGmTensor<Q_T, Q_FORMAT, uint32_t, LAYOUT == LayOutTypeEnum::LAYOUT_TND> &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, actualSeqLengthsGmQ,
                                            actualLenQDims);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, actualSeqLengthsGmQ, actualLenQDims);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                        FaGmTensor<KV_T, KV_FORMAT> &kvGmTensor, __gm__ uint8_t *gm, uint32_t bnStride,
                                        uint32_t n2Stride)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bnStride, n2Stride);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        // AIC 侧无需预置（AIV 的 InitCrossCoreSync 预置 c1v1/c2v2 四旗标）
    }

    __aicore__ inline void UnInitCrossCoreSync()
    {
        // 收尾 drain 8 个 c1v1/c2v2 旗标（4 标志 × {本 AIV0, +AIV0_AIV1_OFFSET}），防悬空旗标
        // 污染下一 kernel 的首次 Wait（flash_attn_block_cube_nd.h:221-231 逐一对齐）
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_1 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_1 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void AllocEventID()
    {
        // L0A/L0B 静态 policy 手填 EXTERNAL Mutex ID（表 2：L0A 7/8、L0B 9/10），与 FA IsAllocEventID 同构
        mmL0APolicy_.Init(l0aBufferManager_, BUFFER_SIZE_BYTE_32K, L0A_BUFFER_ID0, L0A_BUFFER_ID1);
        mmL0BPolicy_.Init(l0bBufferManager_, BUFFER_SIZE_BYTE_32K, L0B_BUFFER_ID0, L0B_BUFFER_ID1);
    }

    __aicore__ inline void FreeEventID()
    {
        mmL0APolicy_.Uninit(l0aBufferManager_);
        mmL0BPolicy_.Uninit(l0bBufferManager_);
    }

    // copy query（q 为 576 宽单张量，GM 上 nope+rope 合并存储，
    // L1 上按行切分为 [nope 512) + [rope 64) 两段拼出 mm1 K=576 的 S 矩阵；
    // rope 段 GM 偏移 = dIdx = dSize(=head_dim_v=512)，L1 偏移 = dSize*dstStride 与双张量版本一致）
    __aicore__ inline void CopyQueryAndRopeTile(const LocalTensor<Q_T> &dstTensor, FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        {
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = 0,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = (uint32_t)constInfo.dSize};
            copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
        }

        {
            uint32_t ropeL1Offset = constInfo.dSize * dstStride;
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor[ropeL1Offset], .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = (uint32_t)constInfo.dSize,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = (uint32_t)constInfo.dSizeRope};
            copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
        }
    }

    // copy key（k_cache 为 576 宽单张量，GM 上 nope+rope 合并存储，
    // L1 上按行切分为 [nope 512) + [rope 64) 两段；rope 段 GM 偏移 = dIdx = dSize(=head_dim_v=512)；
    // key and value 是同一份数据（k==v），bmm2 的 V 段 = L1 nope 区（512 宽），无独立 value GM 读取）
    __aicore__ inline void CopyKeyAndRopeTile(const LocalTensor<KV_T> &dstTensor, FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;
        {
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = 0,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = (uint32_t)constInfo.dSize};
            copyKvGmToL1(l1Tensor, kCacheGm, gmCoord);
        }

        {
            uint32_t ropeL1Offset = constInfo.dSize * dstStride;
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor[ropeL1Offset], .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = (uint32_t)constInfo.dSize,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = (uint32_t)constInfo.dSizeRope};
            copyKvGmToL1(l1Tensor, kCacheGm, gmCoord);
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C,
                                      FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm1(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t kvpL1BufId = runInfo.loop % L1_KVP_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CROSSCORE_BMM1_0 + mm1ResUbBufId;
        LocalTensor<MM_T> mm1ResUbTensor =
            ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<MM_T>();

        // c1v1 反向：AIV0/AIV1 消费完上一轮 mm1Res 槽后才能 FIXPIPE 覆写（表 2）
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx + AIV0_AIV1_OFFSET);

        LocalTensor<Q_T> qL1Tensor = l1QBuffers_.template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
            CopyQueryAndRopeTile(qL1Tensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0); // 数据就绪，交棒 MTE1 消费
        }

        LocalTensor<KV_T> kvpL1Tensor = l1KvpBuffers_[kvpL1BufId * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvpL1BufId); // 等待本槽上一轮 MatmulN 释放（MTE1_MTE2 语义）
        CopyKeyAndRopeTile(kvpL1Tensor, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvpL1BufId);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvpL1BufId); // 交棒 MTE1（K 消费 + 后续 bmm2 V/P 消费）
        {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();

            MMParam mmParam = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                          (uint32_t)(constInfo.dSize + constInfo.dSizeRope), false, true);

            MatmulK<Q_T, KV_T, T, mBaseSize, s2BaseSize, 128, ABLayout::MK, ABLayout::KN>(
                qL1Tensor, kvpL1Tensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor, mmParam);

            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_); // M 写完 → FIX 可消费
            FixpipeMm1(mm1ResUbTensor, l0CSubTensor, runInfo);
            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        // KV/P 槽 MTE1 锁保持至 bmm2 的 MatmulN 消费完后释放（= 旧 SetFlag<MTE1_MTE2> 语义）
        // 见 IterateBmm2 尾部 Mutex::Unlock<PIPE_MTE1>

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0); // S1G 内最后一个 S2 循环后释放 Q
        }

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = ((uint32_t)FA_T::dVBaseSize + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        uint32_t mm2ResUbBufId = runInfo.loop % UB_MM2_RES_BUFCNT;
        uint32_t kvpL1BufId = runInfo.loop % L1_KVP_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + kvpL1BufId;
        uint32_t c2v2CrossCoreSyncIdx = CROSSCORE_BMM2_0 + mm2ResUbBufId;
        LocalTensor<MM_T> mm2ResUbTensor =
            ubMm2ResBuffers_[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<MM_T>();

        // v1c2：等待 AIV0/AIV1 完成本槽 P 的 L1 写入；c2v2：等待 AIV0/AIV1 消费完 mm2Res 槽
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c2v2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c2v2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);
        {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            uint64_t VDsize = (uint32_t)FA_T::dVBaseSize;
            MMParam param =
                MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo.dSizeV, (uint32_t)runInfo.actSingleLoopS2Size,
                            false, false, true, true, 0, (uint32_t)runInfo.actMSize);

            LocalTensor<KV_T> kvpL1Tensor =
                l1KvpBuffers_[kvpL1BufId * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
            // A = P（本槽 rope 段，AIV vec1 写入，v1c2 旗标门控）；B = V（本槽 nope 段，即 K）
            MatmulN<Q_T, KV_T, T, mBaseSize, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                kvpL1Tensor[s2BaseSize * VDsize], kvpL1Tensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor, param);

            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            FixpipeMm2PartialN(mm2ResUbTensor, l0CSubTensor, constInfo.dSizeV, runInfo);
            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        // 本槽 MTE1 消费（MatmulN）完成 → 释放槽，下一轮 bmm1 可覆写（= 旧 SetFlag<MTE1_MTE2>）
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvpL1BufId);

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c2v2CrossCoreSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c2v2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);
    }
}; // FlashMlaWithKvcacheNoQuantMlaBlockCube

template <typename FA_T>
class FlashMlaWithKvcacheNoQuantMlaBlockCubeDummy {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr LayOutTypeEnum LAYOUT = FA_T::qLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using ConstInfoX = ConstInfo_t<FlashMlaWithKvcacheKernelType::NO_QUANT>;
    template <typename FlashMlaSeqLensToolT>
    __aicore__ inline FlashMlaWithKvcacheNoQuantMlaBlockCubeDummy(ConstInfoX &constInfo,
                                                                  FlashMlaSeqLensToolT &seqLensTool){};
};

} // namespace FlashAttnKernel

#endif // FLASH_MLA_WITH_KVCACHE_BLOCK_CUBE_NOQUANT_MLA_H_
