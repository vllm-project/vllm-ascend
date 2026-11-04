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
 * \file quant_lightning_indexer_v2_common_arch22.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_COMMON_H
#define QUANT_LIGHTNING_INDEXER_V2_COMMON_H

namespace QLIV2Common {

// candidate (two-level topk) 模式, 与 host 侧 tiling.h 定义保持一致
inline constexpr uint32_t CANDIDATE_MODE_SOURCE = 1;   // is_candidate_source: 输出候选块索引
inline constexpr uint32_t CANDIDATE_MODE_CONSUMER = 2; // use_candidate: 输入候选块索引, 候选内选topk
inline constexpr uint32_t CANDIDATE_MODE_OFF = 3;      // 关闭candidate功能(默认)

// D3b 候选 staging ring（AIV gather ↔ AIC cube 消费）。
// 910B 的跨核 flag 语义（同 sparse_flash_attention 的 910B 用法）：多个 AIV 对同一 id 的
// set 合并成一次可等待通知；AIC 的一次 set 广播给两个 AIV。故每个事件每侧只 set/wait 一次。
// **flag id 约束**：910B 只有 4 bit（`GetffstMsg` 里 `flagId & 0xf`），且 AscendC 保留
// 11=SYNC_AIC_FLAG、12=SYNC_AIV_FLAG、13=SYNC_AIC_AIV_FLAG、14=SYNC_AIV_ONLY_ALL；
// 本 kernel 已用 0/1/2（syncV1C1/syncV0C1/syncC1V0）。
//
// v4（对齐 DSL 的软件流水）：生产/消费固定错一拍 + **单 id 计数信号量**，深度不再受
// flag id 数量限制，只受 workspace 限制。槽号 = 全局 tile 序号 % CAND_RING_SLOTS。
inline constexpr uint32_t CAND_RING_SLOTS = 8; // staging 槽数（流水深度）
// AIV→AIC：每写完一个 tile 的半个 staging 槽各 set 一次（合并成 1 个令牌/tile）；
// AIC 每消费一个 tile wait 一次，并在进入任务的 tile 循环前先 wait 一次（错一拍）。
inline constexpr uint32_t CAND_READY_FLAG = 6;
// AIC→AIV：空槽回执。AIC 启动预置 CAND_RING_SLOTS 个令牌，之后每消费 1 tile 回补 1 个。
inline constexpr uint32_t CAND_FREE_FLAG = 8;
// AIC→AIV：W 槽回收。W 按候选任务交替使用 2 个槽（runInfo.loop%2），预置 2 个令牌，
// AIC 每完成一个候选任务回补 1 个 ⇒ AIV 最多领先 2 个任务，保证不覆盖 AIC 正在读的槽。
inline constexpr uint32_t CAND_WFREE_FLAG = 3;
// B4: 候选任务 W 就绪 —— AIV 先跑一次 ProcessVec0（w×qscale 广播到 vec0Out），
// 再放行 AIC 的候选 cube 循环。id 10（避开 0/1/2、ring 6/8、WFREE 3、保留 11..14）。
inline constexpr uint32_t CAND_WREADY_FLAG = 10;
// B5: AIC 每算完一个 token 的全部候选 tile 就 set 一次；AIV 消费该 token 分数前 wait。
// id 15（0/1/2 已用、3/6/8/10 被 W 与环占用、11..14 为 AscendC 保留）。
inline constexpr uint32_t CAND_SCORE_READY_FLAG = 15;
// v4: AIC 主循环结束令牌 —— pending 队列（各任务末尾两个 token）在核尾统一排空，
// 等 AIC 推进/退出后再读，保证 GM 写已落地。
inline constexpr uint32_t CAND_AIC_DONE_FLAG = 4;
// 候选分数区每任务的槽位数：s1Base 4 行 × 最大 64 tile = 256 个 256-float 行。
// 按任务序号奇偶双缓冲（跨任务延迟消费时，下一任务不会覆盖尚未消费的分数）。
inline constexpr uint32_t CAND_SCORE_TILES_PER_TASK = 256;
// 候选 tile 粒度：32 个候选块 × 8 token = 256 token（DSL 为 64 块×8=512；910B 上按
// 现有 mm1 骨架与 UB 容量取 256）
inline constexpr uint32_t CAND_TILE_BLOCKS = 32;
inline constexpr uint32_t CAND_TILE_TOKENS = CAND_TILE_BLOCKS * 8;
// 候选 kscale 辅助区（每 tile 32 块 × 16 half = 512 half = 1KB，块内前 8 half 有效、后 8 填充）：
// AIV 在 gather 时每 AIV 写本半（256 half = 512B，一次 DMA），消费时按 tile 一次 1KB 读回，
// 取代原先"每 tile 32 次 16B DMA + 32 次 block_table 标量读"的逐块读取；
// 与分数区同样按任务奇偶双缓冲，跨任务延迟消费时不会被覆盖。
inline constexpr uint32_t CAND_AUX_SCALE_HALVES_PER_TILE = CAND_TILE_BLOCKS * 16U;

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename QK_T, typename SCORE_T, typename OUT_T,
          const bool PAGE_ATTENTION = false, LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND,
          LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BBND, typename... Args>
struct QLIV2Type {
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t candidateOutOffset;
    uint64_t outputIdxOffsetCoreOffset;  // A15: output_idx_offset 的 batch 级前缀 (行 x kHeadNum 布局)

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isValid = false;
    bool isNeedLD = false;
    // B5: 候选驱动任务 —— 正常全量扫的输出写回被抑制（结果由候选分数路径产生）
    bool suppressOut = false;
    // v4: 本任务是否候选驱动任务（ProcessVec0 据此决定是否等待 W 槽回收令牌 WFREE）
    bool candTask = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;             // topK选取大小
    uint64_t kSeqSize = 0ULL;         // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t keyStride0 = 0;          // key 第0维真实 stride (0=非PA/旧调用, 紧凑兜底; 对齐 arch35 A11)
    uint32_t keyDequantScaleStride0 = 0; // k_scale 第0维真实 stride (同上)
    LI_LAYOUT outputLayout;           // 输出的格式
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1; // 压缩率

    uint32_t actualLenQDims = 0U;     // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;      // KV 的actualSeqLength 的维度
    uint32_t cmpResiduaKLenDims = 0U; // cmpResidualK的维度
    bool isAccumSeqS1 = false;        // 是否累加模式
    bool isAccumSeqS2 = false;        // 是否累加模式

    // candidate (two-level topk)
    uint32_t candidateMode = 3U;        // 1=source 2=consumer 3=off
    uint32_t candidateTopkBlocks = 2048U;
    uint32_t candidateBlockSize = 8U;

    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    uint32_t coreEnable = 0U;
};

struct LdSplitCoreInfo {
    bool isLdCoreEnable = false;    // 当前核是否参与规约任务
    uint32_t saveWorkSpaceIdx = 0U; // 存放LD参数的地址
    uint32_t bn2Idx = 0U;           // 归约任务
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U; // 当前AIV核上规约任务的索引
    uint32_t workspaceNum = 0U; // 当前AIV核上规约任务的S2切分数量
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0U; // 最终输出索引搬出Topk的初始偏移地址
    // mode=1（candidate source，候选块来源）块级输出 candidate_topk_index 的初始偏移地址；
    // 结构与 indiceOutCoreOffset 同构，只是每行的宽度从 sparseCount（最终 topk 数）换成
    // candidateTopkBlocks（块级 topk 数）
    uint64_t candidateOutCoreOffset = 0U;
};

template <typename T1, typename T2>
__aicore__ inline T1 Align(T1 num, T2 rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace QLIV2Common

#endif // QUANT_LIGHTNING_INDEXER_V2_COMMON_H
