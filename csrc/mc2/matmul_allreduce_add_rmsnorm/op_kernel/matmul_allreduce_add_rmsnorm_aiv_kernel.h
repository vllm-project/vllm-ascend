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

#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_AIV_KERNEL_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_AIV_KERNEL_H

#include "kernel_operator.h"
#include "matmul_allreduce_add_rmsnorm_tiling.h"
#include "matmul_allreduce_add_rmsnorm_utils.h"

using namespace AscendC;

constexpr int32_t DIFUSION_ADD_LEN = 512;
constexpr int32_t TQUE_DEPTH = 1;
constexpr uint32_t TBUF_POOL_MAX_BUFID_SIZE = 8;
enum CrossRankSyncFlagEnum {
    FLAG_ZERO_IDX,
    FLAG_ONE_IDX,
    FLAG_TWO_IDX,
    FLAG_ADD_IDX,
    FLAG_FOUR_IDX,
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
class MatmulAllreduceAddRmsnormAivKernel {

public:
    __aicore__ inline MatmulAllreduceAddRmsnormAivKernel<MmadDtype, OutDtype>() { }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR gamma, GM_ADDR y, GM_ADDR add_out,
        GM_ADDR workspace, const MatmulAllreduceAddRmsnormTilingData *tilingData,
        Hccl<HCCL_SERVER_TYPE_AICPU> &hccl_)
    {
        this->hccl_ = hccl_;
        is_deterministic = false;
        auto ppTilingData = &tilingData->matmulAllreduceAddRmsnormInfo.ppTilingData;
        auto commTilingData = &tilingData->matmulAllreduceAddRmsnormInfo.commTilingData;
        auto quantInfo = &tilingData->matmulAllreduceAddRmsnormInfo.quantInfo;

        gm_out = reinterpret_cast<__gm__ MmadDtype *>(y);
        gm_add_input = reinterpret_cast<__gm__ MmadDtype *>(residual);
        gm_add_output = reinterpret_cast<__gm__ MmadDtype *>(add_out);
        gm_gamma = reinterpret_cast<__gm__ MmadDtype *>(gamma);

        batch_size = ppTilingData->opShape.batchSize;
        m = ppTilingData->opShape.m;
        k = ppTilingData->opShape.k;
        n = ppTilingData->opShape.n;

        m0 = ppTilingData->m0;
        k0 = ppTilingData->k0;
        n0 = ppTilingData->n0;

        m_loop = ppTilingData->mLoop;
        k_loop = ppTilingData->kLoop;
        n_loop = ppTilingData->nLoop;

        core_loop = ppTilingData->coreLoop;
        swizzl_count = ppTilingData->swizzlCount;
        tiling_key = ppTilingData->tilingKey;
        rank = hccl_.GetRankId();
        rank_size = hccl_.GetRankDim();

        max_ub_single_dma_size = commTilingData->ubMoveNum;
        withSerialMode = false;
        tag = commTilingData->tag;
        comm_npu_split = commTilingData->commNpuSplit;
        comm_data_split = commTilingData->commDataSplit;
        comm_direct = commTilingData->commDirect;
        is_91093 = false;
        core_count = comm_npu_split * comm_data_split;
        dequant_granularity = static_cast<QuantGranularity>(quantInfo->dequantGranularity);
        dequant_group_size = quantInfo->dequantGroupSize;
        quant_granularity = static_cast<QuantGranularity>(quantInfo->quantGranularity);
        quant_group_size = quantInfo->quantGroupSize;
        epsilon = tilingData->matmulAllreduceAddRmsnormInfo.rmsnormTilingData.epsilon;
        is_gather_add_out = tilingData->matmulAllreduceAddRmsnormInfo.ppTilingData.isGatherAddOut;

        swizzl_direct = (tiling_key & SWIZZL_MASK) ? true : false;
        trans_a = ppTilingData->isTransA;
        trans_b = ppTilingData->isTransB;
        is_int8 = false;
        ag_dim = 0;
        rs_dim = 0;
        inner_dim_is_Ag = false;
        weight_nz = false;
        max_ub_ping_pong_size = max_ub_single_dma_size / 2; // 2 - double buffer

        core_idx = get_block_idx();
        core_num = get_block_num();
        aiv_idx = get_subblockid();
        other_rank = (core_idx < rank_size) ? core_idx : -1;

        // init ub usage
        pipe.InitBuffer(ctrlBuf, AscendC::ONE_BLK_SIZE);
        ub_ctrl_flag = reinterpret_cast<__ubuf__ int32_t *>(ctrlBuf.Get<int32_t>().GetPhyAddr());

        pipe.InitBuffer(gammaBuf, n * sizeof(MmadDtype));

        uint32_t step1_ub_usage = AscendC::AlignUp(
            n * sizeof(MmadDtype) +
            2 * (rank_size * DIFUSION_ADD_LEN * sizeof(MmadDtype)) +
            n * sizeof(MmadDtype) +
            n * sizeof(MmadDtype) +
            n * sizeof(float) +
            n * sizeof(float) +
            n * sizeof(float),
            AscendC::ONE_BLK_SIZE);

        uint32_t step2_ub_usage = AscendC::AlignUp(
            max_ub_ping_pong_size * sizeof(MmadDtype),
            AscendC::ONE_BLK_SIZE) * 2;
        uint32_t max_step_ub_usage = max(step1_ub_usage, step2_ub_usage);

        pipe.InitBufPool(step1BufPool, max_step_ub_usage);
        pipe.InitBufPool(step2BufPool, max_step_ub_usage, step1BufPool);

        step1BufPool.InitBuffer(inQueueX, 1, n * sizeof(MmadDtype));
        step1BufPool.InitBuffer(inQueueY, 2, rank_size * DIFUSION_ADD_LEN * sizeof(MmadDtype));
        step1BufPool.InitBuffer(addOutQueue, 1, n * sizeof(MmadDtype));
        step1BufPool.InitBuffer(outQueue, 1, n * sizeof(MmadDtype));
        step1BufPool.InitBuffer(xFp32Buf, n * sizeof(float));
        step1BufPool.InitBuffer(sqxBuf, n * sizeof(float));
        step1BufPool.InitBuffer(reduceFp32Buf, n * sizeof(float));

        step2BufPool.InitBuffer(allgatherBuf[0], max_ub_ping_pong_size * sizeof(MmadDtype));
        step2BufPool.InitBuffer(allgatherBuf[1], max_ub_ping_pong_size * sizeof(MmadDtype));

        CopyInGamma();
    }

    __aicore__ inline void Process(const MatmulAllreduceAddRmsnormTilingData *tilingData)
    {
        // AIV AllReduce & Add & RMSNorm func, waits for AIC to complete [Matmul].
        FFTSCrossCoreSync<PIPE_MTE3>(FFTS_SYNC_AICORE_GROUP_MODE, AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID);
        PipeBarrier<PIPE_ALL>();

        ResetIpcFlags(FLAG_NUM);
        CrossRankSyncEx(FLAG_NUM);
        constexpr int32_t allreduce_used_core = 16;
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

            WaitEvent(flag_idx);
            SetAndWaitAivSync(flag_idx, is_91093 ? BLOCK_COUNT_4 : MAX_BLOCK_COUNT);
            CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx, is_91093 ? BLOCK_COUNT_4 : MAX_BLOCK_COUNT);

            if (aiv_idx == 0 && core_idx < allreduce_used_core) {
                int32_t m_cur_rank = LimitRange(m_total - rank * m_per_rank, 0, m_per_rank);
                int32_t m_per_core = DivCeil(m_cur_rank, allreduce_used_core);
                int32_t m_cur_core = LimitRange(m_cur_rank - core_idx * m_per_core, 0, m_per_core);
                int32_t core_offset_m = loop_offset + rank * m_per_rank + core_idx * m_per_core;
                ParallelWithSplitStepOneAddNorm(core_offset_m * n, m_cur_core);
            }

            PipeBarrier<PIPE_ALL>();

            SetAndWaitAivSync(flag_idx, is_91093 ? BLOCK_COUNT_4 : MAX_BLOCK_COUNT);
            CrossRankSyncV1(FLAG_ADD_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx, is_91093 ? BLOCK_COUNT_4 : MAX_BLOCK_COUNT);

            { // ParallelWithSplitStepTwo
                int32_t used_core_per_rank = allreduce_used_core / rank_size;
                int32_t sub_core_idx = core_idx % used_core_per_rank;
                int32_t gather_rank_id = core_idx / used_core_per_rank;
                int32_t m_in_rank = LimitRange(m_total - gather_rank_id * m_per_rank, 0, m_per_rank);
                int32_t m_per_core = DivCeil(m_in_rank, used_core_per_rank);
                int32_t m_cur_core = LimitRange(m_in_rank - sub_core_idx * m_per_core, 0, m_per_core);
                int32_t core_offset_m = loop_offset + gather_rank_id * m_per_rank + sub_core_idx * m_per_core;
                auto gm_share_buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(gather_rank_id);

                bool filter_core_cond = aiv_idx == 0 && core_idx < allreduce_used_core && m_cur_core > 0;
                if (filter_core_cond) {
                    ParallelAllGather(gm_out, gm_share_buff, core_offset_m * n, m_cur_core * n);
                }

                SetAndWaitAivSync(flag_idx);
                CrossRankSyncV2(FLAG_TWO_IDX, cal_idx + 1);
                SetAndWaitAivSync(flag_idx);

                if (is_gather_add_out) {
                    if (filter_core_cond && gather_rank_id == rank) {
                        ParallelAllGather(gm_share_buff, gm_add_output, core_offset_m * n, m_cur_core * n);
                    }

                    SetAndWaitAivSync(flag_idx);
                    CrossRankSyncV2(FLAG_GATHER_ADD_OUT_STEP1, cal_idx + 1);
                    SetAndWaitAivSync(flag_idx);

                    if (filter_core_cond && gather_rank_id != rank) {
                        ParallelAllGather(gm_add_output, gm_share_buff, core_offset_m * n, m_cur_core * n);
                    }

                    SetAndWaitAivSync(flag_idx);
                    CrossRankSyncV2(FLAG_GATHER_ADD_OUT_STEP2, cal_idx + 1);
                    SetAndWaitAivSync(flag_idx);
                }
            }

            if (cal_idx <= comm_count - pipe_depth) {
                SetAicSync(flag_idx);
            }
        }
        ResetIpcFlags(FLAG_NUM);
        if (aiv_idx == 0 && core_idx < rank_size) {
            __gm__ int32_t *state_buff = (__gm__ int32_t *)hccl_.GetWindowsOutAddr(other_rank);
            CheckBuffFlag(ub_ctrl_flag, state_buff + FLAG_ZERO_IDX, 0);
        }
    }

private:
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

    __aicore__ void SetAicSync(uint64_t flag_idx)
    {
        FFTSCrossCoreSync<PIPE_MTE3>(FFTS_SYNC_AICORE_GROUP_MODE, flag_idx);
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

    __aicore__ void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
    {
        if (aiv_idx == 0 && core_idx == rank) {
            __gm__ int32_t *state_buff = (__gm__ int32_t *)hccl_.GetWindowsOutAddr(rank);
            SetBuffFlagByAdd(ub_ctrl_flag, state_buff + flag_idx, FLAG_VALUE);
        } else if (aiv_idx == 0 && core_idx < rank_size) {
            __gm__ int32_t *state_buff = (__gm__ int32_t *)hccl_.GetWindowsOutAddr(core_idx);
            CheckBuffFlag(ub_ctrl_flag, state_buff + flag_idx, FLAG_VALUE * flag_data);
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

    __aicore__ inline void CopyInGamma()
    {
        GlobalTensor<MmadDtype> gamma_global;
        gamma_global.SetGlobalBuffer((__gm__ MmadDtype *)gm_gamma, n);
        DataCopy(gammaBuf.Get<MmadDtype>(), gamma_global, n);
        PipeSync<HardEvent::MTE2_V>();
    }

    __aicore__ void ParallelWithSplitStepOneAddNorm(uint32_t core_buf_offset, uint32_t m_cur_core)
    {
        if (m_cur_core <= 0) {
            return;
        }

        auto buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(rank);

        GlobalTensor<MmadDtype> x_global;
        GlobalTensor<MmadDtype> y_global;
        GlobalTensor<MmadDtype> out_global;
        GlobalTensor<MmadDtype> add_out_global;

        x_global.SetGlobalBuffer(buff + core_buf_offset);
        out_global.SetGlobalBuffer(buff + core_buf_offset);
        add_out_global.SetGlobalBuffer(gm_add_output + core_buf_offset);

        uint32_t add_count = DivCeil(n, DIFUSION_ADD_LEN);

        LocalTensor<MmadDtype> x_local;
        LocalTensor<MmadDtype> y_local;

        for (uint32_t i = 0; i < m_cur_core; i++) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> sqx = sqxBuf.Get<float>();

            x_local = inQueueX.AllocTensor<MmadDtype>();
            for (uint32_t j = 0; j < add_count; j++) {
                uint32_t add_offset = j * DIFUSION_ADD_LEN;
                uint32_t add_len = min<uint32_t>(n - add_offset, DIFUSION_ADD_LEN);

                DataCopy(x_local[add_offset], x_global[i * n + add_offset], add_len);
                inQueueX.EnQue(x_local);

                uint32_t iterate_end = (rank + 1) % rank_size;
                y_local = inQueueY.AllocTensor<MmadDtype>();
                for (uint32_t k = 0; k < rank_size; ++k) {
                    uint32_t iterate_idx = iterate_end + k;
                    if (iterate_idx >= rank_size) {
                        iterate_idx -= rank_size;
                    }

                    if (iterate_idx == rank) {
                        y_global.SetGlobalBuffer(gm_add_input + core_buf_offset);
                    } else {
                        auto other_buff = (__gm__ MmadDtype *)hccl_.GetWindowsInAddr(iterate_idx);
                        y_global.SetGlobalBuffer(other_buff + core_buf_offset);
                    }
                    DataCopy(y_local[k * add_len], y_global[i * n + add_offset], add_len);
                }
                inQueueY.EnQue(y_local);
                x_local = inQueueX.DeQue<MmadDtype>();
                y_local = inQueueY.DeQue<MmadDtype>();

                Cast(x_fp32[add_offset], x_local[add_offset], RoundMode::CAST_NONE, add_len);
                PipeBarrier<PIPE_V>();
                for (uint32_t k = 0; k < rank_size; ++k) {
                    // use sqx as shared buf, required n >= add_len
                    Cast(sqx, y_local[k * add_len], RoundMode::CAST_NONE, add_len);
                    PipeBarrier<PIPE_V>();
                    Add(x_fp32[add_offset], x_fp32[add_offset], sqx, add_len);
                    PipeBarrier<PIPE_V>();
                }

                inQueueY.FreeTensor(y_local);
            }
            inQueueX.FreeTensor(x_local);

            // copy add result out
            LocalTensor<MmadDtype> add_out = addOutQueue.AllocTensor<MmadDtype>();
            Cast(add_out, x_fp32, RoundMode::CAST_RINT, n);
            addOutQueue.EnQue(add_out);
            add_out = addOutQueue.DeQue<MmadDtype>();
            DataCopy(add_out_global[i * n], add_out, n);
            addOutQueue.FreeTensor(add_out);

            LocalTensor<MmadDtype> gamma_local = gammaBuf.Get<MmadDtype>();
            LocalTensor<MmadDtype> out_local = outQueue.AllocTensor<MmadDtype>();
            LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

            // make sure precision is same in bf16 case
            Cast(out_local, x_fp32, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();

            Cast(x_fp32, out_local, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();

            Mul(sqx, x_fp32, x_fp32, n);
            PipeBarrier<PIPE_V>();

            Muls(sqx, sqx, (float)1.0 / n, n);
            PipeBarrier<PIPE_V>();

            ReduceSum(sqx, sqx, reduce_buf_local, n);
            PipeBarrier<PIPE_V>();

            Adds(sqx, sqx, epsilon, 1);
            PipeBarrier<PIPE_V>();

            Sqrt(sqx, sqx, 1);
            Duplicate(reduce_buf_local, (float)1.0, 1);
            PipeBarrier<PIPE_V>();

            Div(sqx, reduce_buf_local, sqx, 1);
            PipeBarrier<PIPE_V>();

            PipeSync<HardEvent::V_S>();
            float rstd_value = sqx.GetValue(0);
            PipeSync<HardEvent::S_V>();
            PipeBarrier<PIPE_V>();

            Muls(x_fp32, x_fp32, rstd_value, n);
            PipeBarrier<PIPE_V>();

            if constexpr (std::is_same<MmadDtype, half>::value) {
                Cast(out_local, x_fp32, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
                Mul(out_local, gamma_local, out_local, n);
                PipeBarrier<PIPE_V>();
            } else if constexpr (std::is_same<MmadDtype, bfloat16_t>::value) {
                Cast(out_local, x_fp32, RoundMode::CAST_RINT, n);
                PipeBarrier<PIPE_V>();
                Cast(x_fp32, out_local, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
                Cast(sqx, gamma_local, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();

                Mul(x_fp32, x_fp32, sqx, n);
                PipeBarrier<PIPE_V>();
                Cast(out_local, x_fp32, RoundMode::CAST_RINT, n);
                PipeBarrier<PIPE_V>();
                PipeSync<HardEvent::V_MTE2>();
            }

            outQueue.EnQue(out_local);
            out_local = outQueue.DeQue<MmadDtype>();
            DataCopy(out_global[i * n], out_local, n);
            outQueue.FreeTensor(out_local);
        }
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
    __gm__ MmadDtype *gm_add_input;
    __gm__ MmadDtype *gm_add_output;
    __gm__ MmadDtype *gm_gamma;
    __ubuf__ int32_t *ub_ctrl_flag;

    int32_t batch_size;
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t m0;
    int32_t k0;
    int32_t n0;

    int32_t m_loop;
    int32_t n_loop;
    int32_t k_loop;
    int32_t core_loop;
    int32_t core_idx;

    int32_t rank;
    int32_t rank_size;
    int32_t tiling_key;
    int32_t swizzl_count;
    bool swizzl_direct;

    bool trans_a;
    bool trans_b;
    bool is_int8;
    bool is_91093;
    bool is_gather_add_out;

    int32_t aiv_idx;
    int32_t other_rank;
    int32_t core_num;
    int32_t max_ub_single_dma_size;
    int32_t max_ub_ping_pong_size;

    int32_t gm_c_pingpong_size;
    int32_t withSerialMode;
    int32_t tag;
    int32_t comm_npu_split;
    int32_t comm_data_split;
    int32_t comm_direct;

    int32_t core_count;
    bool is_deterministic;

    QuantGranularity dequant_granularity;
    int32_t dequant_group_size;
    QuantGranularity quant_granularity;
    int32_t quant_group_size;

    WorkspaceInfo workspace_info;
    int32_t ag_dim;
    int32_t rs_dim;
    bool inner_dim_is_Ag;
    bool weight_nz{false};

    float epsilon;

    TPipe pipe;
    AscendC::TBufPool<TPosition::VECCALC, TBUF_POOL_MAX_BUFID_SIZE> step1BufPool;
    AscendC::TBufPool<TPosition::VECCALC, TBUF_POOL_MAX_BUFID_SIZE> step2BufPool;

    AscendC::TQue<AscendC::QuePosition::VECIN, TQUE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, TQUE_DEPTH> outQueueZ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, TQUE_DEPTH> addOutQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, TQUE_DEPTH> outQueue;

    AscendC::TBuf<TPosition::VECCALC> ctrlBuf;
    AscendC::TBuf<TPosition::VECCALC> gammaBuf;
    AscendC::TBuf<TPosition::VECCALC> xFp32Buf;
    AscendC::TBuf<TPosition::VECCALC> sqxBuf;
    AscendC::TBuf<TPosition::VECCALC> reduceFp32Buf;
    AscendC::TBuf<TPosition::VECCALC> allgatherBuf[2];

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};
#endif