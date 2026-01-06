/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FUSE_DENSE_ALLGATHER_AIV_KERNEL_H
#define FUSE_DENSE_ALLGATHER_AIV_KERNEL_H

#include "kernel_operator.h"
#include "fuse_dense_allgather_tiling.h"

using namespace AscendC;

constexpr int32_t TQUE_DEPTH = 1;
constexpr uint32_t TBUF_POOL_MAX_BUFID_SIZE = 8;
constexpr int32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t BLOCK_COUNT_4 = 4;
enum CrossRankSyncFlagEnum {
    FLAG_ZERO_IDX,
    FLAG_GATHER_ADD_OUT_STEP1,
    FLAG_GATHER_ADD_OUT_STEP2,
    FLAG_NUM
};
constexpr int32_t FLAG_VALUE = 1;
constexpr int32_t NUM_PER_REP_FP32 = 64;

template <typename T>
__aicore__ void CopyUbufToGmAlignB16(__gm__ T *dst, __ubuf__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                         uint16_t srcSTride, uint16_t dstStride)
{
    DataCopyExtParams dataCopyParams(nBurst,
                                     lenBurst,
                                     srcSTride,
                                     dstStride,
                                     0);
    LocalTensor<uint8_t> ubTensor;
    TBuffAddr ubAddr;
    ubAddr.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubAddr.bufferAddr = reinterpret_cast<uint64_t>(src);
    ubTensor.SetAddr(ubAddr);
    GlobalTensor<uint8_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dst));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <typename T>
__aicore__ void CopyGmToUbufAlignB16(__ubuf__ T *dst, __gm__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                        uint16_t srcSTride, uint16_t dstStride)
{
    DataCopyExtParams dataCopyParams(nBurst,
                                     lenBurst,
                                     srcSTride,
                                     dstStride,
                                     0);
    LocalTensor<uint8_t> ubTensor;
    TBuffAddr ubAddr;
    ubAddr.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubAddr.bufferAddr = reinterpret_cast<uint64_t>(dst);
    ubTensor.SetAddr(ubAddr);
    GlobalTensor<uint8_t> gmTensor;
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template <typename MmadDtype, typename OutDtype>
class FuseDenseAllgather {
public:
    __aicore__ inline FuseDenseAllgather<MmadDtype, OutDtype>() { }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
        GM_ADDR workspace, const FuseDenseAllgatherTilingData *tilingData,
        Hccl<HCCL_SERVER_TYPE_AICPU> &hccl_)
    {
        this->hccl_ = hccl_;
        auto ppTilingData = &tilingData->fuseDenseAllgatherInfo.ppTilingData;
        auto commTilingData = &tilingData->fuseDenseAllgatherInfo.commTilingData;

        gm_out = reinterpret_cast<__gm__ MmadDtype *>(y);
        gm_input = reinterpret_cast<__gm__ MmadDtype *>(x);

        batch_size = ppTilingData->opShape.batchSize;
        m = ppTilingData->opShape.m;
        n = ppTilingData->opShape.n;

        m0 = ppTilingData->m0;
        n_loop = ppTilingData->nLoop;

        core_loop = ppTilingData->coreLoop;
        swizzl_count = ppTilingData->swizzlCount;
        tiling_key = ppTilingData->tilingKey;
        rank = hccl_.GetRankId();
        rank_size = hccl_.GetRankDim();
        is_91093 = false;
        max_ub_single_dma_size = commTilingData->ubMoveNum;
        max_ub_ping_pong_size = max_ub_single_dma_size / 2; // 2 - double buffer

        core_idx = get_block_idx();
        core_num = get_block_num();
        aiv_idx = get_subblockid();
        other_rank = (core_idx < rank_size) ? core_idx : -1;

        uint32_t step_ub_usage = AscendC::AlignUp(
            max_ub_ping_pong_size * sizeof(MmadDtype),
            AscendC::ONE_BLK_SIZE) * 2;

        pipe.InitBufPool(step1BufPool, step_ub_usage);
        pipe.InitBuffer(ctrlBuf, AscendC::ONE_BLK_SIZE);

        step1BufPool.InitBuffer(allgatherBuf[0], max_ub_ping_pong_size * sizeof(MmadDtype));
        step1BufPool.InitBuffer(allgatherBuf[1], max_ub_ping_pong_size * sizeof(MmadDtype));
        ub_ctrl_flag = reinterpret_cast<__ubuf__ int32_t *>(ctrlBuf.Get<int32_t>().GetPhyAddr());
    }

    __aicore__ inline void Process(const FuseDenseAllgatherTilingData *tilingData)
    {
        PipeBarrier<PIPE_ALL>();

        ResetIpcFlags(FLAG_NUM);
        CrossRankSyncEx(FLAG_NUM);
        constexpr int32_t allgather_used_core = 16;
        int32_t one_comm_count = swizzl_count;
        int32_t loop_num_per_comm = one_comm_count * n_loop;
        int32_t comm_count = DivCeil(core_loop, loop_num_per_comm);
        int32_t pipe_depth = is_91093 ? BLOCK_COUNT_4 : MAX_BLOCK_COUNT;

        for (int cal_idx = 0; cal_idx < comm_count; ++cal_idx) {
            uint64_t flag_idx = cal_idx % pipe_depth;
            int32_t m_total = (cal_idx == comm_count - 1) ?
                m - cal_idx * swizzl_count * m0 : swizzl_count * m0;
            int32_t m_per_rank = DivCeil(m_total, rank_size);
            int32_t loop_offset = cal_idx * swizzl_count * m0;

            {
                int32_t used_core_per_rank = allgather_used_core / rank_size;
                int32_t sub_core_idx = core_idx % used_core_per_rank;
                int32_t gather_rank_id = core_idx / used_core_per_rank;
                int32_t m_in_rank = LimitRange(m_total - gather_rank_id * m_per_rank, 0, m_per_rank);
                int32_t m_per_core = DivCeil(m_in_rank, used_core_per_rank);
                int32_t m_cur_core = LimitRange(m_in_rank - sub_core_idx * m_per_core, 0, m_per_core);
                int32_t core_offset_m = loop_offset + gather_rank_id * m_per_rank + sub_core_idx * m_per_core;
                auto gm_share_buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(gather_rank_id);

                bool filter_core_cond = aiv_idx == 0 && core_idx < allgather_used_core && m_cur_core > 0;

                if (filter_core_cond && gather_rank_id == rank) {
                    ParallelAllGather(gm_share_buff, gm_input, core_offset_m * n, m_cur_core * n);
                }

                SetAndWaitAivSync(flag_idx);
                CrossRankSyncV2(FLAG_GATHER_ADD_OUT_STEP1, cal_idx + 1);
                SetAndWaitAivSync(flag_idx);

                if (filter_core_cond) {
                    ParallelAllGather(gm_out, gm_share_buff, core_offset_m * n, m_cur_core * n);
                }

                SetAndWaitAivSync(flag_idx);
                CrossRankSyncV2(FLAG_GATHER_ADD_OUT_STEP2, cal_idx + 1);
                SetAndWaitAivSync(flag_idx);
            }
        }
        ResetIpcFlags(FLAG_NUM);
        if (aiv_idx == 0 && core_idx < rank_size) {
            __gm__ int32_t *state_buff = (__gm__ int32_t *)hccl_.GetWindowsOutAddr(other_rank);
            CheckBuffFlag(ub_ctrl_flag, state_buff + FLAG_ZERO_IDX, 0);
        }
    }

private:
    template <pipe_t pipe>
    __aicore__ inline void FFTSCrossCoreSync(uint64_t mode, uint64_t flag_id)
    {
        uint64_t config = 1 | (mode << 4) | (flag_id << 8);
        ffts_cross_core_sync(pipe, config);
    }
    __aicore__ void SetBuffFlag(__ubuf__ int32_t *ub_ctrl_flag, __gm__ int32_t *buff, int32_t flag)
    {
        *ub_ctrl_flag = flag;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        CopyUbufToGmAlignB16(buff, ub_ctrl_flag, 1, sizeof(int32_t), 0, 0);
    }

    __aicore__ void SetBuffFlagByAdd(__ubuf__ int32_t *ub_ctrl_flag, __gm__ int32_t *buff, int32_t flag)
    {
        PipeBarrier<PIPE_ALL>();
        *ub_ctrl_flag = flag;
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<int32_t>();
        PipeBarrier<PIPE_ALL>();
        CopyUbufToGmAlignB16(buff, ub_ctrl_flag, 1, sizeof(int32_t), 0, 0);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void CheckBuffFlag(__ubuf__ int32_t *ub_ctrl_flag, __gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (true) {
            CopyGmToUbufAlignB16(ub_ctrl_flag, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            if (*ub_ctrl_flag == flag) {
                break;
            }
        }
    }

    __aicore__ void ResetIpcFlags(int32_t num_flags)
    {
        for (int32_t idx = 0; idx <= num_flags; ++idx) {
            if (core_idx == 0 && aiv_idx == 0) {
                __gm__ int32_t *state_buff = (__gm__ int32_t *)hccl_.GetWindowsOutAddr(rank);
                SetBuffFlag(ub_ctrl_flag, state_buff + idx, 0);
            }
        }
    }

    __aicore__ void CrossRankSyncV2(int32_t flag_idx, int32_t flag_data)
    {
        if (aiv_idx == 0 && core_idx < rank_size) {
            __gm__ int32_t *state_buff = (__gm__ int32_t *)hccl_.GetWindowsOutAddr(core_idx);
            SetBuffFlagByAdd(ub_ctrl_flag, state_buff + flag_idx, FLAG_VALUE);
        }
        if (aiv_idx == 0 && core_idx == rank) {
            __gm__ int32_t *state_buff = (__gm__ int32_t *)hccl_.GetWindowsOutAddr(rank);
            CheckBuffFlag(ub_ctrl_flag, state_buff + flag_idx, FLAG_VALUE * rank_size * flag_data);
        }
    }

    __aicore__ void SetAndWaitAivSync(uint64_t flag_idx, int32_t pipe_depth = 2)
    {
        FFTSCrossCoreSync<PIPE_MTE3>(0, flag_idx + pipe_depth);
        WaitEvent(flag_idx + pipe_depth);
    }

    __aicore__ inline uint32_t GetGmU32(GM_ADDR gm_addr)
    {
        copy_gm_to_ubuf_align_b32(ub_ctrl_flag, gm_addr, 0, 1, sizeof(uint32_t), 0, 0, 0, 0);
        PipeSync<HardEvent::MTE2_S>();
        return *reinterpret_cast<__ubuf__ uint32_t *>(ub_ctrl_flag);
    }

    __aicore__ inline void SetGmU32(GM_ADDR gm_addr, uint32_t data)
    {
        *reinterpret_cast<__ubuf__ uint32_t *>(ub_ctrl_flag) = data;
        PipeSync<HardEvent::S_MTE3>();
        copy_ubuf_to_gm_align_b32(gm_addr, ub_ctrl_flag, 0, 1, sizeof(uint32_t), 0, 0, 0, 0);
    }

    __aicore__ inline void CrossRankSyncEx(uint32_t flag_idx)
    {
        AscendC::SyncAll<true>();
        __asm__ __volatile__("");
        if (aiv_idx == 0 && core_idx == 0) {
            auto flag_addr = (GM_ADDR)hccl_.GetWindowsOutAddr(0) + flag_idx * AscendC::ONE_BLK_SIZE;
            uint32_t old_flag_data = GetGmU32(flag_addr);
            __asm__ __volatile__("");
            SetAtomicAdd<int32_t>();
            SetGmU32(flag_addr, 1);
            PipeSync<HardEvent::MTE3_S>();
            SetAtomicNone();
            __asm__ __volatile__("");

            uint32_t new_flag_data;
            do {
                new_flag_data = GetGmU32(flag_addr);
                __asm__ __volatile__("");
            } while (new_flag_data - old_flag_data < rank_size);
            __asm__ __volatile__("");
            SetAtomicAdd<int32_t>();
            SetGmU32(flag_addr, 1);
            PipeSync<HardEvent::MTE3_S>();
            SetAtomicNone();
        }
        __asm__ __volatile__("");
        AscendC::SyncAll<true>();
    }

    template <typename T>
    __aicore__ inline T min(const T& a, const T& b) {
        return (a < b) ? a : b;
    }

    template <typename T>
    __aicore__ inline T max(const T& a, const T& b) {
        return (a > b) ? a : b;
    }

    template <typename T>
    __aicore__ inline T LimitRange(const T& val, const T& low, const T& high) {
        return min(max(val, low), high);
    }

    template <AscendC::HardEvent EVENT>
    __aicore__ inline void PipeSync()
    {
        AscendC::TEventID event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
        AscendC::SetFlag<EVENT>(event_id);
        AscendC::WaitFlag<EVENT>(event_id);
    }

    __aicore__ void ParallelAllGather(__gm__ MmadDtype *gm_dst, __gm__ MmadDtype *gm_src,
        uint32_t core_buf_offset, uint32_t data_len)
    {
        GlobalTensor<MmadDtype> src_global;
        GlobalTensor<MmadDtype> dst_global;
        src_global.SetGlobalBuffer(gm_src);
        dst_global.SetGlobalBuffer(gm_dst);

        constexpr uint32_t PIPELINE_COPY_NUM = sizeof(allgatherBuf) / sizeof(allgatherBuf[0]);
        TEventID ev_mte3_mte2[PIPELINE_COPY_NUM];
        TEventID ev_mte2_mte3[PIPELINE_COPY_NUM];
        LocalTensor<MmadDtype> local_tensors[PIPELINE_COPY_NUM];

        for (uint32_t i = 0; i < PIPELINE_COPY_NUM; i++) {
            ev_mte3_mte2[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            ev_mte2_mte3[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
            SetFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[i]);
            local_tensors[i] = allgatherBuf[i].Get<MmadDtype>();
        }

        uint32_t offset = core_buf_offset;
        uint32_t copy_len = max_ub_ping_pong_size; // num of MmadDtype, not the byte length
        uint32_t copy_count = DivCeil(data_len, copy_len);
        uint32_t pipe_id = 0;

        for (uint32_t i = 0; i < copy_count; i++) {
            uint32_t actual_copy_len =
                (i == copy_count - 1) ? (data_len - i * copy_len) : copy_len;

            auto &local_tensor = local_tensors[pipe_id];

            WaitFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[pipe_id]);
            DataCopy(local_tensor, src_global[offset], actual_copy_len);
            SetFlag<HardEvent::MTE2_MTE3>(ev_mte2_mte3[pipe_id]);
            WaitFlag<HardEvent::MTE2_MTE3>(ev_mte2_mte3[pipe_id]);
            DataCopy(dst_global[offset], local_tensor, actual_copy_len);
            SetFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[pipe_id]);

            offset += actual_copy_len;
            pipe_id = (pipe_id + 1) % PIPELINE_COPY_NUM;
        }

        for (uint32_t i = 0; i < PIPELINE_COPY_NUM; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(ev_mte3_mte2[i]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(ev_mte3_mte2[i]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(ev_mte2_mte3[i]);
        }

        PipeBarrier<PIPE_ALL>();
    }

    __gm__ MmadDtype *gm_out;
    __gm__ MmadDtype *gm_input;
    __ubuf__ int32_t *ub_ctrl_flag;

    int32_t batch_size;
    int32_t m;
    int32_t n;
    int32_t m0;
    int32_t n0;

    int32_t n_loop;
    int32_t core_loop;
    int32_t core_idx;

    int32_t rank;
    int32_t rank_size;
    int32_t tiling_key;
    int32_t swizzl_count;

    bool is_91093;

    int32_t aiv_idx;
    int32_t other_rank;
    int32_t core_num;
    int32_t max_ub_single_dma_size;
    int32_t max_ub_ping_pong_size;

    TPipe pipe;
    AscendC::TBufPool<TPosition::VECCALC, TBUF_POOL_MAX_BUFID_SIZE> step1BufPool;
    AscendC::TBuf<TPosition::VECCALC> allgatherBuf[2];
    AscendC::TBuf<TPosition::VECCALC> ctrlBuf;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};
#endif