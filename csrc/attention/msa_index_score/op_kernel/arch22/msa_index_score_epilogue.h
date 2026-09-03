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
 * \file msa_index_score_epilogue.h
 * \brief AIV 侧 Epilogue：可选反量化列乘 -> atten_mask -> 分段 RowMax -> local_mask -> 写回。
 *
 * 数值路径：
 *   score = Maxpool[ (scale·) Q@Kᵀ + atten_mask ] + local_mask
 * atten_mask（sparse_mode=3）按 rightDownCausal 解析，不加载 [2048,2048] 模板。
 * local_mask 由 start_loc（query 所在逻辑 block）+ init/local_blocks 生成强制 +∞。
 */

#ifndef MSA_INDEX_SCORE_EPILOGUE_H
#define MSA_INDEX_SCORE_EPILOGUE_H

#include "kernel_operator.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

#include "../msa_index_score_common.h"
#include "../msa_index_score_debug.h"
#include "msa_index_score_task.h"

namespace MsaIndexScoreNs {

template <bool IS_QUANT>
class MsaSegRowMaxEpilogue {
public:
    using ArchTag = Catlass::Arch::AtlasA2;
    // S workspace 元素类型：非量化 fp16，int8 fp32。
    using ElementS = std::conditional_t<IS_QUANT, float, half>;

    // score 暂存容量：1 AIC : 2 AIV 下每个 subcore 最多拿到半个 M-tile 的行；
    // 一次 flush 覆盖 MSA_STAGE_BLOCKS 个 block（= 每行 1KB 连续写回）。
    static constexpr uint32_t MSA_STAGE_ROWS = MSA_ROW_TILE_M / 2;
    static constexpr uint32_t MSA_STAGE_BLOCKS = 256;
    static_assert(MSA_STAGE_BLOCKS % MSA_BLOCKS_PER_STILE == 0, "stage window must hold whole S-tiles");

    // UB 手工布局（字节偏移），catlass 不提供 UB allocator。
    static constexpr uint32_t UB_OFF_S = 0;
    static constexpr uint32_t UB_SIZE_S = MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t UB_OFF_T64 = UB_OFF_S + UB_SIZE_S;
    static constexpr uint32_t UB_SIZE_T64 = MSA_REDUCE_ROWS * (MSA_BLOCK_SIZE / 2) * sizeof(float);
    static constexpr uint32_t UB_OFF_T8 = UB_OFF_T64 + UB_SIZE_T64;
    static constexpr uint32_t UB_SIZE_T8 = MSA_REDUCE_ROWS * MSA_FP32_PER_BLOCK * sizeof(float);
    static constexpr uint32_t UB_OFF_ROWMAX = UB_OFF_T8 + UB_SIZE_T8;
    static constexpr uint32_t UB_SIZE_ROWMAX = 512;
    // 反量化 scale 一页：block_size 个 fp32 = 512B。
    static constexpr uint32_t UB_OFF_DEQ = UB_OFF_ROWMAX + UB_SIZE_ROWMAX;
    static constexpr uint32_t UB_SIZE_DEQ = MSA_BLOCK_SIZE * sizeof(float);
    // 非量化：WholeReduceMax 的 fp16 输出（每 pass MSA_REDUCE_ROWS 个 half）。
    static constexpr uint32_t UB_OFF_RED16 = UB_OFF_DEQ + UB_SIZE_DEQ;
    static constexpr uint32_t UB_SIZE_RED16 = MSA_REDUCE_ROWS * sizeof(half);
    // score 暂存：本 subcore 的行 × 一次 flush 覆盖的 block 数。逐 pass 只写 32B 到 GM
    // 会把 MTE3 压到 ~9GB/s；改为在 UB 内按行累积、整行（≥1KB）一次写回。
    static constexpr uint32_t UB_OFF_STAGE = ((UB_OFF_RED16 + UB_SIZE_RED16 + 511U) / 512U) * 512U;
    static constexpr uint32_t UB_SIZE_STAGE = MSA_STAGE_ROWS * MSA_STAGE_BLOCKS * sizeof(float);
    // 非量化：S16（fp16 载入缓冲）复用量化路径的 fp32 S 区（64KB = 两级 32KB 乒乓）。
    // v0.7 PipeUtilization：aiv_mte2_wait_ratio≈0.84、aiv_mte2_ratio≈0.33、aiv_vec_ratio≈0.57，
    // 说明 GM→UB 与归约串行；两级缓冲让下一 pass 的 MTE2 叠在当前 pass 的 V 上。
    static constexpr uint32_t UB_OFF_S16 = UB_OFF_S;
    static constexpr uint32_t UB_SIZE_S16 = MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE * sizeof(half);
    static constexpr uint32_t S16_STAGES = 2;
    static constexpr uint32_t UB_TOTAL = UB_OFF_STAGE + UB_SIZE_STAGE;
    static_assert(UB_TOTAL <= ArchTag::UB_SIZE, "MsaSegRowMaxEpilogue UB out of bounds");
    static_assert(UB_OFF_S16 + S16_STAGES * UB_SIZE_S16 <= UB_OFF_T64, "S16 ping-pong overlaps T64");

    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    // fp16：一个 32B datablock 16 个元素，一个 repeat 128 个元素（掩码 128 位）。
    static constexpr uint32_t MSA_HALF_PER_BLOCK = 16;
    static constexpr uint32_t MSA_HALF_PER_MASK_WORD = 64;

    // -3.4e38 越界，float→half 按 IEEE 就近舍入得 -inf；归约后由 Maxs 抬回 MSA_FILL_VALUE。
    static constexpr half FILL_VALUE_H = static_cast<half>(MSA_FILL_VALUE);

    __aicore__ inline MsaSegRowMaxEpilogue() {}

    __aicore__ inline void Init(Catlass::Arch::Resource<ArchTag> &resource, uint32_t numQHeads, uint32_t strideOutHead,
                                uint32_t strideOutToken, uint32_t maxBlocksPerBatch, uint32_t strideScalePage,
                                uint32_t strideScaleHead, bool isQuant, uint32_t keyLayout, uint32_t totalK,
                                const AscendC::GlobalTensor<float> &gScore, const AscendC::GlobalTensor<float> &gScale,
                                const AscendC::GlobalTensor<int32_t> &gBlockTable)
    {
        numQHeads_ = numQHeads;
        strideOutHead_ = strideOutHead;
        strideOutToken_ = strideOutToken;
        maxBlocksPerBatch_ = maxBlocksPerBatch;
        strideScalePage_ = strideScalePage;
        strideScaleHead_ = strideScaleHead;
        isQuant_ = isQuant;
        keyLayout_ = keyLayout;
        totalK_ = totalK;
        if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
            // N2=1 packed [T2] / [T2,1]：token 步长为 1，避免误用 PA page 步长。
            strideScalePage_ = 1;
            strideScaleHead_ = 1;
        }
        gScore_ = gScore;
        gScale_ = gScale;
        gBlockTable_ = gBlockTable;

        ubS_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_S);
        ubS16Ping_[0] = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_S16);
        ubS16Ping_[1] = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_S16 + UB_SIZE_S16);
        ubS16_ = ubS16Ping_[0];
        ubT64_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T64);
        ubT8_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T8);
        ubRowMax_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_ROWMAX);
        ubDeqScale_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_DEQ);
        ubRed16_ = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_RED16);
        ubStage_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_STAGE);
    }

    /// 进入一个新任务（M-tile）：确定本 subcore 负责的行区间并复位 score 暂存窗口。
    __aicore__ inline void BeginTask(const MsaTask &task, uint32_t subIdx, uint32_t subBlockNum)
    {
        const uint32_t subM = MsaCeilDiv(task.mActual, subBlockNum);
        mOff_ = subIdx * subM;
        mSub_ = (mOff_ >= task.mActual) ? 0U : MsaMinU32(subM, task.mActual - mOff_);
        // 暂存区按半个 M-tile 静态分配；若 subcore 数变化导致行数超出则退回逐 pass 直写。
        stageOn_ = (mSub_ > 0U) && (mSub_ <= MSA_STAGE_ROWS);
        stageBlkBase_ = 0;
        stageBlkEnd_ = 0;
        rowInReqBase_ = task.mStart + mOff_;
        tokenBase0_ = task.cuQStart;
        if constexpr (!IS_QUANT) {
            // 本任务两级 S16 的 V_MTE2 余额；EndTask 对称 Wait，避免跨任务/跨启动泄漏。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
    }

    /// 任务结束：把暂存窗口里剩余的 score 写回 GM。
    __aicore__ inline void EndTask()
    {
        FlushStage();
        if constexpr (!IS_QUANT) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
    }

    /// 处理一个 S tile（MSA_BLOCKS_PER_STILE 个 sparse block）中属于本 subcore 的行。
    __aicore__ inline void ProcessSTile(const AscendC::GlobalTensor<ElementS> &gS, const MsaTask &task,
                                        uint32_t blkBase, uint32_t subIdx, uint32_t subBlockNum, bool dumpDebug = false)
    {
        if (mSub_ == 0U) {
            return;
        }
        const uint32_t mOff = mOff_;
        const uint32_t mSub = mSub_;
        if (stageOn_ && (blkBase >= stageBlkBase_ + MSA_STAGE_BLOCKS)) {
            FlushStage();
            stageBlkBase_ = blkBase;
            stageBlkEnd_ = blkBase;
        }

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            AscendC::printf("[MSA_DBG][AIV] ProcessSTile blkBase=%u subIdx=%u mOff=%u mSub=%u "
                            "fullEndBlk=%u visibleEndBlk=%u isQuant=%u\n",
                            blkBase, subIdx, mOff, mSub, task.fullEndBlk, task.visibleEndBlk,
                            static_cast<uint32_t>(isQuant_));
        }
#endif

        if constexpr (IS_QUANT) {
            for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
                const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
                ProcessPass(gS, task, blkBase, mOff + p0, rows, dumpDebug && (p0 == 0));
            }
        } else if (blkBase >= task.visibleEndBlk) {
            for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
                const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
                ProcessPass(gS, task, blkBase, mOff + p0, rows, false);
            }
        } else {
            // 非量化可见 tile：pass i 的 MTE2 与 pass i-1 的 mask/reduce/stage 重叠。
            uint32_t stage = 0;
            bool hasPrev = false;
            uint32_t prevRowOff = 0;
            uint32_t prevRows = 0;
            uint32_t prevStage = 0;
            for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
                const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
                const uint32_t rowOff = mOff + p0;
                IssueS16(gS, rowOff, rows, stage);
                if (hasPrev) {
                    WaitS16(prevStage);
                    ProcessPass(gS, task, blkBase, prevRowOff, prevRows, dumpDebug && (prevRowOff == mOff), true);
                    ReleaseS16(prevStage);
                }
                hasPrev = true;
                prevRowOff = rowOff;
                prevRows = rows;
                prevStage = stage;
                stage ^= 1U;
            }
            if (hasPrev) {
                WaitS16(prevStage);
                ProcessPass(gS, task, blkBase, prevRowOff, prevRows, false, true);
                ReleaseS16(prevStage);
            }
        }
        stageBlkEnd_ = blkBase + MSA_BLOCKS_PER_STILE;
    }

private:
    /// 非量化乒乓：把本 pass 的 S 搬进 ubS16Ping_[stage]，不阻塞 Vector。
    __aicore__ inline void IssueS16(const AscendC::GlobalTensor<ElementS> &gS, uint32_t rowOff, uint32_t rows,
                                    uint32_t stage)
    {
        if constexpr (IS_QUANT) {
            return;
        } else {
            const int32_t evt = static_cast<int32_t>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
            auto dst = ubS16Ping_[stage];
            if (rows < MSA_ROWS_PER_PASS) {
                AscendC::Duplicate(dst, static_cast<half>(MSA_FILL_VALUE), MSA_ROWS_PER_PASS * STILE_WIDTH);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::DataCopy(dst, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
        }
    }

    __aicore__ inline void WaitS16(uint32_t stage)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<int32_t>(stage));
        ubS16_ = ubS16Ping_[stage];
    }

    __aicore__ inline void ReleaseS16(uint32_t stage)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<int32_t>(stage));
    }

    /// 处理一个 pass（最多 MSA_ROWS_PER_PASS 行 × MSA_BLOCKS_PER_STILE 个 block）。
    /// sLoaded：非量化乒乓路径已把 S 搬进 ubS16_，跳过本函数内的 GM→UB。
    __aicore__ inline void ProcessPass(const AscendC::GlobalTensor<ElementS> &gS, const MsaTask &task, uint32_t blkBase,
                                       uint32_t rowOff, uint32_t rows, bool dumpDebug, bool sLoaded = false)
    {
        const bool allInvalid = (blkBase >= task.visibleEndBlk);

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            AscendC::printf("[MSA_DBG][AIV] ProcessPass rowOff=%u rows=%u allInvalid=%u STILE_WIDTH=%u\n", rowOff, rows,
                            static_cast<uint32_t>(allInvalid), STILE_WIDTH);
        }
#endif

#if MSA_INDEX_SCORE_DEBUG
        uint32_t sDumpN = MSA_DEBUG_DUMP_ELEMS;
        if (task.kvLen > 0 && task.kvLen <= 8) {
            sDumpN = static_cast<uint32_t>(task.kvLen) + 3U;
        }
#endif
        if (!allInvalid) {
            if (!sLoaded) {
                // 短 pass 只拷了部分行；SegRowMax 固定按 16 行归约，未写入区域必须是 -inf。
                // 满 16 行时 DataCopy 覆盖整块 ubS_，可跳过 Duplicate。
                if (rows < MSA_ROWS_PER_PASS) {
                    if constexpr (IS_QUANT) {
                        AscendC::Duplicate(ubS_, MSA_FILL_VALUE, MSA_ROWS_PER_PASS * STILE_WIDTH);
                    } else {
                        AscendC::Duplicate(ubS16_, static_cast<half>(MSA_FILL_VALUE), MSA_ROWS_PER_PASS * STILE_WIDTH);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                // 上一 pass 的归约还在读 ubS_/ubS16_（score 已改为暂存在 UB，不再有
                // 逐 pass 的 MTE3 顺带排空 V），MTE2 覆盖该缓冲前必须显式等 V。
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                if constexpr (IS_QUANT) {
                    AscendC::DataCopy(ubS_, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
                } else {
                    AscendC::DataCopy(ubS16_, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }

#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf("[MSA_DBG][AIV] after DOT load S ub (row0, blk0 first %u fp32):\n", sDumpN);
                MsaDumpUbSample(ubS_, MSA_DUMP_DESC_S_DOT, sDumpN);
                AscendC::printf("[MSA_DBG][AIV] S_after_dot:");
                MsaPrintUbFloats(ubS_, sDumpN);
            }
#endif

            if (isQuant_) {
                ApplyDequantScale(task, blkBase, MSA_ROWS_PER_PASS);
#if MSA_INDEX_SCORE_DEBUG
                if (dumpDebug) {
                    AscendC::printf("[MSA_DBG][AIV] S_after_dequant:");
                    MsaPrintUbFloats(ubS_, sDumpN);
                }
#endif
            }

            ApplyMask(task, blkBase, rowOff, rows);

#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf("[MSA_DBG][AIV] after MASK S ub (row0, blk0 first %u fp32), "
                                "maskRange=[fullEndBlk,visibleEndBlk)=[%u,%u):\n",
                                sDumpN, task.fullEndBlk, task.visibleEndBlk);
                MsaDumpUbSample(ubS_, MSA_DUMP_DESC_S_MASK, sDumpN);
                AscendC::printf("[MSA_DBG][AIV] S_after_mask:");
                MsaPrintUbFloats(ubS_, sDumpN);

                // 逐行打印本 pass 的 validCount，核对 MASK 切分点。
                for (uint32_t r = 0; r < rows && r < 8U; ++r) {
                    const uint32_t rowInReq = task.mStart + rowOff + r;
                    const uint32_t tOff = rowInReq / numQHeads_;
                    const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
                    const uint32_t blk = blkBase; // 小用例通常只有 blk0
                    int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
                    if (validCount < 0) {
                        validCount = 0;
                    }
                    if (validCount > static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                        validCount = static_cast<int32_t>(MSA_BLOCK_SIZE);
                    }
                    AscendC::printf("[MSA_DBG][AIV] row r=%u flat=%u tokenOff=%u visibleEnd=%d validCount=%d\n", r,
                                    rowInReq, tOff, visibleKeyEnd, validCount);
                    if (validCount < static_cast<int32_t>(MSA_BLOCK_SIZE) && validCount >= 0) {
                        const uint32_t base = (r * MSA_BLOCKS_PER_STILE) * MSA_BLOCK_SIZE;
                        const uint32_t lo = (validCount >= 2) ? static_cast<uint32_t>(validCount - 2) : 0U;
                        const uint32_t hi = static_cast<uint32_t>(validCount) + 3U;
                        AscendC::printf("[MSA_DBG][AIV]   S_after_mask[%u:%u]:", lo, hi);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for (uint32_t i = lo; i < hi && i < MSA_BLOCK_SIZE; ++i) {
                            AscendC::printf(" %f", ubS_.GetValue(base + i));
                        }
                        AscendC::printf("\n");
                    }
                }
            }
#endif

            if constexpr (IS_QUANT) {
                SegRowMax();
            } else {
                SegRowMaxWhole();
            }

#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf("[MSA_DBG][AIV] after MAX rowmax (rows[0:%u) x %u blocks):\n", rows,
                                MSA_BLOCKS_PER_STILE);
                MsaDumpUbSample(ubRowMax_, MSA_DUMP_DESC_SCORE_MAX, rows * MSA_BLOCKS_PER_STILE);
                for (uint32_t r = 0; r < rows && r < 8U; ++r) {
                    AscendC::printf("[MSA_DBG][AIV] score_after_max[r=%u]:", r);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
                        AscendC::printf(" %f", ubRowMax_.GetValue(r * MSA_BLOCKS_PER_STILE + j));
                    }
                    AscendC::printf("\n");
                }
            }
#endif
        } else {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf("[MSA_DBG][AIV] allInvalid stile -> fill score with -inf\n");
            }
#endif
        }

        FillInvalidBlocks(task, blkBase);
        ApplyLocalMaskUb(task, blkBase, rows);

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            AscendC::printf("[MSA_DBG][AIV] final score before store (after fillInvalid+localMask), "
                            "rows[0:%u) x %u blocks:\n",
                            rows, MSA_BLOCKS_PER_STILE);
            MsaDumpUbSample(ubRowMax_, MSA_DUMP_DESC_SCORE_FINAL, rows * MSA_BLOCKS_PER_STILE);
            for (uint32_t r = 0; r < rows && r < 8U; ++r) {
                AscendC::printf("[MSA_DBG][AIV] score_final[r=%u]:", r);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
                    AscendC::printf(" %f", ubRowMax_.GetValue(r * MSA_BLOCKS_PER_STILE + j));
                }
                AscendC::printf("\n");
            }
        }
#endif

        if (stageOn_) {
            StageScore(blkBase, rowOff - mOff_, rows);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            StoreScore(task, blkBase, rowOff, rows);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    /// 把本 pass 的 [rows, MSA_BLOCKS_PER_STILE] 归约结果搬到 score 暂存区对应的
    /// 行 × block 窗口（行内 stride 为 MSA_STAGE_BLOCKS），供整行 flush。
    __aicore__ inline void StageScore(uint32_t blkBase, uint32_t stageRow, uint32_t rows)
    {
        const uint32_t colOff = blkBase - stageBlkBase_;
        AscendC::Adds<float>(ubStage_[stageRow * MSA_STAGE_BLOCKS + colOff], ubRowMax_, 0.0F,
                             static_cast<int32_t>(MSA_BLOCKS_PER_STILE), static_cast<uint8_t>(rows),
                             AscendC::UnaryRepeatParams(1, 1, MSA_STAGE_BLOCKS / MSA_FP32_PER_BLOCK,
                                                        MSA_BLOCKS_PER_STILE / MSA_FP32_PER_BLOCK));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// 把 score 暂存窗口按行写回 GM：每行一次 ≥1KB 的连续 DataCopy。
    __aicore__ inline void FlushStage()
    {
        if (!stageOn_ || stageBlkEnd_ <= stageBlkBase_) {
            return;
        }
        const uint32_t count = stageBlkEnd_ - stageBlkBase_;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        uint32_t tOff = rowInReqBase_ / numQHeads_;
        uint32_t h = rowInReqBase_ - tOff * numQHeads_;
        uint64_t tokenBase = static_cast<uint64_t>(tokenBase0_ + tOff) * strideOutToken_ + stageBlkBase_;
        for (uint32_t r = 0; r < mSub_; ++r) {
            AscendC::DataCopy(gScore_[tokenBase + static_cast<uint64_t>(h) * strideOutHead_],
                              ubStage_[r * MSA_STAGE_BLOCKS], count);
            if (++h == numQHeads_) {
                h = 0;
                tokenBase += strideOutToken_;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    /// sparse_mode=3：rightDownCausal；否则仅 kv_len 截断。
    __aicore__ inline int32_t VisibleKeyEnd(const MsaTask &task, int32_t tOff) const
    {
        if (task.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
            return MsaClampI32(task.kvLen - task.qLen + tOff + 1, 0, task.kvLen < 0 ? 0 : task.kvLen);
        }
        return task.kvLen < 0 ? 0 : task.kvLen;
    }

    /// 只对 [fullEndBlk, visibleEndBlk) 的边界 block 逐行置 -inf。
    /// 全可见 block 走无 mask 快通道；全不可见 block 的结果在 FillInvalidBlocks 中整体覆盖。
    __aicore__ inline void ApplyMask(const MsaTask &task, uint32_t blkBase, uint32_t rowOff, uint32_t rows)
    {
        if (task.fullEndBlk >= task.visibleEndBlk) {
            return;
        }
        const uint32_t jLo = (task.fullEndBlk > blkBase) ? (task.fullEndBlk - blkBase) : 0;
        if (jLo >= MSA_BLOCKS_PER_STILE) {
            return;
        }
        const uint32_t jHi =
            MsaMinU32(MSA_BLOCKS_PER_STILE, (task.visibleEndBlk > blkBase) ? (task.visibleEndBlk - blkBase) : 0);

        bool touched = false;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t rowInReq = task.mStart + rowOff + r;
            const uint32_t tOff = rowInReq / numQHeads_;
            const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
            for (uint32_t j = jLo; j < jHi; ++j) {
                const uint32_t blk = blkBase + j;
                int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
                if (validCount < 0) {
                    validCount = 0;
                }
                if (validCount >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                    continue;
                }
                if constexpr (IS_QUANT) {
                    FillBlockTail((r * MSA_BLOCKS_PER_STILE + j) * MSA_BLOCK_SIZE, static_cast<uint32_t>(validCount));
                } else {
                    // 非量化归约在 fp16 域起步，mask 必须先落在 ubS16_ 上。
                    FillBlockTailFp16((r * MSA_BLOCKS_PER_STILE + j) * MSA_BLOCK_SIZE,
                                      static_cast<uint32_t>(validCount));
                }
                touched = true;
            }
        }
        if (touched) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// Maxpool 之后、Store 之前施加 local_mask。
    /// 用与 FillInvalidBlocks 相同的位掩码 Duplicate，保证 32B 对齐访问。
    __aicore__ inline void ApplyLocalMaskUb(const MsaTask &task, uint32_t blkBase, uint32_t rows)
    {
        if (task.initBlocks == 0 && task.localBlocks == 0) {
            return;
        }
        const int32_t qBlock = task.startLoc;
        const int32_t localStart = (qBlock + 1 > static_cast<int32_t>(task.localBlocks)) ?
                                       (qBlock + 1 - static_cast<int32_t>(task.localBlocks)) :
                                       0;

        uint64_t initBits = 0;
        uint64_t localBits = 0;
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const int32_t blk = static_cast<int32_t>(blkBase + j);
            if (blk < 0 || blk >= static_cast<int32_t>(task.visibleEndBlk)) {
                continue;
            }
            if (static_cast<uint32_t>(blk) < task.initBlocks) {
                initBits |= (1ULL << j);
            }
            if (blk >= localStart && blk <= qBlock) {
                localBits |= (1ULL << j);
            }
        }
        initBits &= ~localBits; // local 覆盖 init

        const uint8_t repeats =
            static_cast<uint8_t>((rows * MSA_BLOCKS_PER_STILE + MSA_FP32_PER_REPEAT - 1) / MSA_FP32_PER_REPEAT);
        if (initBits != 0) {
            uint64_t bits = 0;
            for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / MSA_BLOCKS_PER_STILE; ++g) {
                bits |= (initBits << (g * MSA_BLOCKS_PER_STILE));
            }
            uint64_t mask[2] = {bits, 0};
            AscendC::Duplicate<float>(ubRowMax_, MSA_LOCAL_SCORE_INIT, mask, repeats, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (localBits != 0) {
            uint64_t bits = 0;
            for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / MSA_BLOCKS_PER_STILE; ++g) {
                bits |= (localBits << (g * MSA_BLOCKS_PER_STILE));
            }
            uint64_t mask[2] = {bits, 0};
            AscendC::Duplicate<float>(ubRowMax_, MSA_LOCAL_SCORE_LOCAL, mask, repeats, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// 把 ubS_[base + validCount, base + MSA_BLOCK_SIZE) 置为 -inf。
    /// 拆成「整 64-fp32 repeat 尾段」+「单 repeat 内的位掩码」两步，保证每次 Duplicate 都是 256B 对齐。
    __aicore__ inline void FillBlockTail(uint32_t base, uint32_t validCount)
    {
        if (validCount >= MSA_BLOCK_SIZE) {
            return;
        }
        const uint32_t alignedStart = MsaCeilDiv(validCount, MSA_FP32_PER_REPEAT) * MSA_FP32_PER_REPEAT;
        if (alignedStart < MSA_BLOCK_SIZE) {
            AscendC::Duplicate(ubS_[base + alignedStart], MSA_FILL_VALUE, MSA_BLOCK_SIZE - alignedStart);
        }
        if (validCount < alignedStart) {
            const uint32_t repeatBase = base + (validCount / MSA_FP32_PER_REPEAT) * MSA_FP32_PER_REPEAT;
            const uint32_t lane = validCount % MSA_FP32_PER_REPEAT;
            uint64_t bits = (lane == 0) ? ~0ULL : (~0ULL << lane);
            uint64_t mask[2] = {bits, 0};
            AscendC::Duplicate<float>(ubS_[repeatBase], MSA_FILL_VALUE, mask, 1, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
        }
    }

    /// 把某一行在某个 block 内 [validCount, MSA_BLOCK_SIZE) 的列置为 -inf。
    /// fp16 下一个 repeat 恰好 128 lane = 一个 block 的 8 个 datablock，
    /// 故一条带 128 位掩码的 Duplicate 即可。
    __aicore__ inline void FillBlockTailFp16(uint32_t base, uint32_t validCount)
    {
        if (validCount >= MSA_BLOCK_SIZE) {
            return;
        }
        uint64_t lo = 0;
        uint64_t hi = 0;
        if (validCount < MSA_HALF_PER_MASK_WORD) {
            lo = ~0ULL << validCount;
            hi = ~0ULL;
        } else {
            hi = ~0ULL << (validCount - MSA_HALF_PER_MASK_WORD);
        }
        uint64_t mask[2] = {lo, hi};
        AscendC::Duplicate<half>(ubS16_[base], FILL_VALUE_H, mask, 1, 1, MSA_BLOCK_SIZE / MSA_HALF_PER_BLOCK);
    }

    /// int8 路径：S 已是 Q · cast(K_int8)，再按列乘 scale[page,0,n] 完成反量化。
    /// 数学上 Q·(K_int8*s) = (Q·cast(K_int8))*s（int8→fp 对整数精确）。
    __aicore__ inline void ApplyDequantScale(const MsaTask &task, uint32_t blkBase, uint32_t rows)
    {
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            uint64_t scaleOff = 0;
            uint32_t nValidScale = MSA_BLOCK_SIZE;
            if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
                const uint32_t tokenStart = task.cuKStart + blk * MSA_BLOCK_SIZE;
                // TND scale 为 packed [T2, N2]，N2=1 时按 token 连续存放，不能用 PA 的 page 步长。
                scaleOff = static_cast<uint64_t>(tokenStart);
                const uint32_t seqTok = blk * MSA_BLOCK_SIZE;
                nValidScale = 0;
                if (task.kvLen > 0 && seqTok < static_cast<uint32_t>(task.kvLen)) {
                    nValidScale = static_cast<uint32_t>(task.kvLen) - seqTok;
                    if (nValidScale > MSA_BLOCK_SIZE) {
                        nValidScale = MSA_BLOCK_SIZE;
                    }
                }
            } else {
                const int32_t page = gBlockTable_.GetValue(task.batchIdx * maxBlocksPerBatch_ + blk);
                scaleOff = static_cast<uint64_t>(page) * strideScalePage_;
            }
            // 上一轮 Mul 占用 ubDeqScale_，必须等 V 完成后再 MTE2 覆盖。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            if (nValidScale < MSA_BLOCK_SIZE) {
                AscendC::Duplicate(ubDeqScale_, 0.0f, MSA_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            if (nValidScale > 0) {
                const uint32_t bytes = nValidScale * static_cast<uint32_t>(sizeof(float));
                if ((bytes % 32U) == 0U) {
                    AscendC::DataCopy(ubDeqScale_, gScale_[scaleOff], nValidScale);
                } else {
                    AscendC::DataCopyExtParams params;
                    params.blockCount = 1;
                    params.blockLen = bytes;
                    params.srcStride = 0;
                    params.dstStride = 0;
                    AscendC::DataCopyPadExtParams<float> pad;
                    pad.isPad = false;
                    pad.leftPadding = 0;
                    pad.rightPadding = 0;
                    pad.paddingValue = 0;
                    AscendC::DataCopyPad(ubDeqScale_, gScale_[scaleOff], params, pad);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t base = r * STILE_WIDTH + j * MSA_BLOCK_SIZE;
                AscendC::Mul(ubS_[base], ubS_[base], ubDeqScale_, static_cast<int32_t>(MSA_BLOCK_SIZE));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// 把 ubS_ 视作 [MSA_REDUCE_ROWS, MSA_BLOCK_SIZE] 做行归约，结果落在 ubRowMax_[MSA_REDUCE_ROWS]，
    /// 其排布恰为 [MSA_ROWS_PER_PASS][MSA_BLOCKS_PER_STILE] 行主序。
    __aicore__ inline void SegRowMax()
    {
        constexpr uint32_t HALF_BLOCK = MSA_BLOCK_SIZE / 2;
        constexpr uint8_t SRC_REP_BLK = MSA_BLOCK_SIZE / MSA_FP32_PER_BLOCK; // 16
        constexpr uint8_t DST_REP_BLK = HALF_BLOCK / MSA_FP32_PER_BLOCK;     // 8

        AscendC::Max<float>(ubT64_, ubS_, ubS_[HALF_BLOCK], static_cast<int32_t>(MSA_FP32_PER_REPEAT),
                            static_cast<uint8_t>(MSA_REDUCE_ROWS),
                            AscendC::BinaryRepeatParams(1, 1, 1, DST_REP_BLK, SRC_REP_BLK, SRC_REP_BLK));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, true>(ubT8_, ubT64_, static_cast<uint8_t>(MSA_REDUCE_ROWS),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, true>(ubRowMax_, ubT8_,
                                             static_cast<uint8_t>(MSA_REDUCE_ROWS / MSA_FP32_PER_BLOCK),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// 非量化路径：一个 sparse block（128 个 fp16）恰好等于向量单元一个满 repeat，
    /// 故整个 pass 的分段 RowMax 可由单条 WholeReduceMax 完成——每 repeat 归约一个
    /// block，128 次迭代按 [row][block] 行主序密集写出，与 ubRowMax_ 的排布一致。
    /// 相比「配对 Max + Cast + 两级 BlockReduceMax」省掉 3 条全宽向量指令和 3 次 barrier。
    /// max 在 fp16 上无舍入、Cast fp16→fp32 精确且保序，结果与 fp32 归约逐 bit 一致。
    __aicore__ inline void SegRowMaxWhole()
    {
        constexpr uint8_t SRC_REP_BLK = MSA_BLOCK_SIZE / MSA_HALF_PER_BLOCK; // 8
        AscendC::WholeReduceMax<half>(ubRed16_, ubS16_, static_cast<int32_t>(MSA_BLOCK_SIZE),
                                      static_cast<int32_t>(MSA_REDUCE_ROWS), 1, 1, SRC_REP_BLK,
                                      AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(ubRowMax_, ubRed16_, AscendC::RoundMode::CAST_NONE, MSA_REDUCE_ROWS);
        AscendC::PipeBarrier<PIPE_V>();

        // mask/短 pass 填充在 fp16 域是 -inf，归约后抬回 fp32 的 MSA_FILL_VALUE，
        // 与 int8 路径及原实现的填充值保持一致（真实分值远大于 -3.4e38，不受影响）。
        AscendC::Maxs<float>(ubRowMax_, ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// blk >= visibleEndBlk 的位置整体置为 -inf。
    __aicore__ inline void FillInvalidBlocks(const MsaTask &task, uint32_t blkBase)
    {
        const uint32_t invStart = (task.visibleEndBlk > blkBase) ? (task.visibleEndBlk - blkBase) : 0;
        if (invStart >= MSA_BLOCKS_PER_STILE) {
            return;
        }
        if (invStart == 0) {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        uint64_t groupBits = 0;
        for (uint32_t k = invStart; k < MSA_BLOCKS_PER_STILE; ++k) {
            groupBits |= (1ULL << k);
        }
        uint64_t bits = 0;
        for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / MSA_BLOCKS_PER_STILE; ++g) {
            bits |= (groupBits << (g * MSA_BLOCKS_PER_STILE));
        }
        uint64_t mask[2] = {bits, 0};
        AscendC::Duplicate<float>(ubRowMax_, MSA_FILL_VALUE, mask,
                                  static_cast<uint8_t>(MSA_REDUCE_ROWS / MSA_FP32_PER_REPEAT), 1,
                                  MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreScore(const MsaTask &task, uint32_t blkBase, uint32_t rowOff, uint32_t rows)
    {
        // 行 = (token, head) 扁平索引、head 为低位，故 pass 内逐行递推即可，
        // 无需每行一次标量整除（整除在 AIV 上是数十 cycle 的长指令）。
        uint32_t tOff = (task.mStart + rowOff) / numQHeads_;
        uint32_t h = (task.mStart + rowOff) - tOff * numQHeads_;
        uint64_t tokenBase = static_cast<uint64_t>(task.cuQStart + tOff) * strideOutToken_ + blkBase;
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopy(gScore_[tokenBase + static_cast<uint64_t>(h) * strideOutHead_],
                              ubRowMax_[r * MSA_BLOCKS_PER_STILE], MSA_BLOCKS_PER_STILE);
            if (++h == numQHeads_) {
                h = 0;
                tokenBase += strideOutToken_;
            }
        }
    }

    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::LocalTensor<float> ubS_;
    AscendC::LocalTensor<half> ubS16_;
    AscendC::LocalTensor<half> ubS16Ping_[S16_STAGES];
    AscendC::LocalTensor<float> ubT64_;
    AscendC::LocalTensor<float> ubT8_;
    AscendC::LocalTensor<float> ubRowMax_;
    AscendC::LocalTensor<float> ubDeqScale_;
    AscendC::LocalTensor<half> ubRed16_;
    AscendC::LocalTensor<float> ubStage_;

    // 当前任务内本 subcore 的行区间与 score 暂存窗口。
    uint32_t mOff_ = 0;
    uint32_t mSub_ = 0;
    uint32_t rowInReqBase_ = 0;
    uint32_t tokenBase0_ = 0;
    uint32_t stageBlkBase_ = 0;
    uint32_t stageBlkEnd_ = 0;
    bool stageOn_ = false;

    uint32_t numQHeads_ = 1;
    uint32_t strideOutHead_ = 0;
    uint32_t strideOutToken_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t strideScalePage_ = 0;
    uint32_t strideScaleHead_ = 1;
    uint32_t keyLayout_ = MSA_KEY_LAYOUT_BBND;
    uint32_t totalK_ = 0;
    bool isQuant_ = false;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_EPILOGUE_H
