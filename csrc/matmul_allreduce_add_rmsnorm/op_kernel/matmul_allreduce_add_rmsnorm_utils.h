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

#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_UTILS_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_UTILS_H

#include <type_traits>
#include "kernel_operator.h"
using namespace AscendC;

constexpr int64_t ND2NZ_STRIDE_LIMIT = 65536;
constexpr int32_t AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID = 12;
constexpr int32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t BLOCK_COUNT_3 = 3;
constexpr int32_t BLOCK_COUNT_4 = 4;
constexpr int32_t TILE_BLOCK_MOD = 2;

constexpr int32_t BLOCK_SIZE_32B = 32;
constexpr int32_t BLOCK_SIZE_256B = 256;
constexpr int32_t BLOCK_SIZE_512B = 512;

constexpr int32_t FFTS_SYNC_INTERNEL_MODE = 0;
constexpr int32_t FFTS_SYNC_AICORE_GROUP_MODE = 2;

constexpr int32_t SWIZZL_MASK = 0b100000;
constexpr int32_t TRANS_A_MASK = 0b010000;
constexpr int32_t TRANS_B_MASK = 0b001000;
constexpr int32_t INT8_MASK = 0b000100;
constexpr int32_t BIAS_MASK = 0b000010;

template <typename T, size_t SIZE>
struct BaseBlock {
    static_assert((SIZE & (SIZE - 1)) == 0, "Invalid block size");
    static constexpr size_t size = SIZE / sizeof(T);

    static __aicore__ inline size_t Count(size_t len)
    {
        return (len + size - 1) / size;
    }

    static __aicore__ inline bool IsAligned(size_t len)
    {
        return len % size == 0;
    }

    static __aicore__ inline size_t AlignUp(size_t len)
    {
        return (len + size - 1) & ~(size - 1);
    }

    static __aicore__ inline size_t AlignDown(size_t len)
    {
        return len & ~(size - 1);
    }
};

template <typename T>
using Block32B = BaseBlock<T, BLOCK_SIZE_32B>;

template <typename T>
using Block256B = BaseBlock<T, BLOCK_SIZE_256B>;

template <typename T>
using Block512B = BaseBlock<T, BLOCK_SIZE_512B>;

struct WorkspaceInfo {
    __gm__ uint8_t *gm_a_align{ nullptr };
    __gm__ uint8_t *gm_b_align{ nullptr };
    __gm__ uint8_t *gm_accum{ nullptr };
    __gm__ uint8_t *gm_dequant_param{ nullptr };
};

template <typename T>
__aicore__ inline LocalTensor<T> CreateLocalTensor(__ubuf__ T *addr)
{
    LocalTensor<T> tensor;
    TBuffAddr taddr;
    taddr.bufferAddr = reinterpret_cast<uint64_t>(addr);
    tensor.SetAddr(taddr);
    return tensor;
}

template <typename T>
__aicore__ inline LocalTensor<T> CreateLocalTensor(uint32_t buffer_offset)
{
    LocalTensor<T> tensor;
    tensor.address_.bufferAddr = buffer_offset;
    return tensor;
}

template <typename T>
__aicore__ inline LocalTensor<T> CreateLocalTensor(uint32_t buffer_offset, uint8_t logic_pos)
{
    LocalTensor<T> tensor;
    tensor.address_.logicPos = logic_pos;
    tensor.address_.bufferAddr = buffer_offset;
    return tensor;
}

template<typename T>
struct IntrinsicCopyGmToL1Nd2Nz {
    static __aicore__ inline void move(
        __cbuf__ T *dst, __gm__ T *src,
        uint8_t sid, uint16_t ndNum, uint16_t nValue, uint16_t dValue,
        uint16_t srcNdMatrixStride, uint16_t srcDValue, uint16_t dstNzC0Stride,
        uint16_t dstNzNStride, uint16_t dstNzMatrixStride) {
        Nd2NzParams nd2nzParams(
            ndNum, nValue, dValue,
            srcNdMatrixStride, srcDValue, dstNzC0Stride,
            dstNzNStride, dstNzMatrixStride
        );
        uint32_t dst_buffer_offset = reinterpret_cast<uint64_t>(dst);
        uint8_t dst_logicpos = static_cast<uint8_t>(TPosition::C1);
        LocalTensor<T> dstTensor;
        dstTensor = CreateLocalTensor<T>(dst_buffer_offset, dst_logicpos);
        GlobalTensor<T> srcTensor;
        srcTensor.SetGlobalBuffer(src);
        DataCopy(dstTensor, srcTensor, nd2nzParams);
        }
};

template <typename T>
struct CopyGmToL1Nd2zN {
    static __aicore__ inline void move(
            __cbuf__ T *dst, __gm__ T *src,
            uint16_t nValue, uint16_t dValue, uint32_t srcDValue, uint16_t dstNzC0Stride) {
        constexpr int BLOCK_LEN = 32 / sizeof(T);
        if (srcDValue < ND2NZ_STRIDE_LIMIT) {
            IntrinsicCopyGmToL1Nd2Nz<T>::move(
                dst,
                src,
                0,
                1,
                nValue,
                dValue,
                0,
                srcDValue,
                dstNzC0Stride,
                1,
                0
            );
        } else {
            for (int i = 0; i < nValue; i++) {
                IntrinsicCopyGmToL1Nd2Nz<T>::move(
                    dst + i * BLOCK_LEN,
                    src + i * srcDValue,
                    0,
                    1,
                    1,
                    dValue,
                    0,
                    0,
                    dstNzC0Stride,
                    0,
                    0
                );
            }
        }
    }
};

__aicore__ inline void AlignJudge(bool trans_a, bool trans_b, int32_t m, int32_t k, int32_t n, int32_t m_align,
                                  int32_t k_align, int32_t n_align, int32_t &aligned_a, int32_t &aligned_b)
{
    if (!trans_a) {
        aligned_a = k != k_align;
    } else {
        aligned_a = (m != m_align && m != 1);
    }

    if (!trans_b) {
        aligned_b = (n != n_align);
    } else {
        aligned_b = (k != k_align);
    }
}

__aicore__ inline WorkspaceInfo GetWorkspaceInfo(__gm__ uint8_t *gm_workspace, int32_t batch_size, int32_t m,
        int32_t k, int32_t n, int32_t m_align, int32_t k_align, int32_t n_align, bool trans_a, bool trans_b,
        int32_t mmad_dsize, bool has_a_align, bool has_b_align, bool has_accum = false, bool has_dequant_param = false)
{
    WorkspaceInfo workspace_info;
    uint64_t workspace_offset = 0;

    if (has_a_align) {
        workspace_info.gm_a_align = gm_workspace + workspace_offset;
        workspace_offset += static_cast<uint64_t>(batch_size) * (trans_a ? k * m_align : m * k_align) * mmad_dsize;
    }

    if (has_b_align) {
        workspace_info.gm_b_align = gm_workspace + workspace_offset;
        workspace_offset += static_cast<uint64_t>(batch_size) * (trans_b ? n * k_align : k * n_align) * mmad_dsize;
    }

    if (has_accum) {
        workspace_info.gm_accum = gm_workspace + workspace_offset;
        workspace_offset += static_cast<uint64_t>(batch_size) * m * n * sizeof(int32_t);
    }

    if (has_dequant_param) {
        workspace_info.gm_dequant_param = gm_workspace + workspace_offset;
        workspace_offset += n * sizeof(float32_t);
    }

    return workspace_info;
}


template<typename T>
__aicore__ inline void CopyCubfToBt(uint64_t dst, __cbuf__ T *src, uint16_t convControl, uint16_t nBurst,
                                    uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap)
{
    DataCopyParams intriParams(nBurst, lenBurst, sourceGap, dstGap);
    uint32_t src_buffer_offset = reinterpret_cast<uint64_t>(src);
    uint32_t dst_buffer_offset = reinterpret_cast<uint64_t>(dst);
    uint8_t src_logicpos = static_cast<uint8_t>(TPosition::C1);
    uint8_t dst_logicpos = static_cast<uint8_t>(TPosition::C2);
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    srcTensor = CreateLocalTensor<T>(src_buffer_offset, src_logicpos);
    dstTensor = CreateLocalTensor<T>(dst_buffer_offset, dst_logicpos);
    DataCopy(dstTensor, srcTensor, intriParams);
}

template<typename T>
__aicore__ inline void CopyGmToCbuf(__cbuf__ T *dst, __gm__ T *src, uint8_t sid, uint16_t nBurst,
                                    uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride, pad_t padMode)
{
    DataCopyParams intriParams(nBurst, lenBurst, srcStride, dstStride);
    GlobalTensor<T> srcTensor;
    srcTensor.SetGlobalBuffer(src);
    uint32_t dst_buffer_offset = reinterpret_cast<uint64_t>(dst);
    uint8_t logicpos = static_cast<uint8_t>(TPosition::C1);
    LocalTensor<T> dstTensor;
    dstTensor = CreateLocalTensor<T>(dst_buffer_offset, logicpos);
    DataCopy(dstTensor, srcTensor, intriParams);
}


template<typename T>
__aicore__ inline void SetFpc(__fbuf__ T *src)
{
    LocalTensor<T> tensor;
    uint32_t src_buffer_offset = reinterpret_cast<uint64_t>(src);
    tensor = CreateLocalTensor<T>(src_buffer_offset);
    SetFixPipeConfig(tensor);
}


template<typename T>
__aicore__ inline void LoadCbufToCaTranspose(__ca__ T *dst, __cbuf__ T *src, uint16_t indexID, uint8_t repeat,
                                            uint16_t srcStride, uint16_t dstStride, bool addrmode,
                                            uint16_t dstFracStride)
{
    LoadData2dTransposeParams params(
        indexID,
        repeat,
        srcStride,
        dstStride,
        dstFracStride,
        addrmode
    );
    uint32_t src_buffer_offset = reinterpret_cast<uint64_t>(src);
    uint32_t dst_buffer_offset = reinterpret_cast<uint64_t>(dst);
    uint8_t src_logicpos = static_cast<uint8_t>(TPosition::C1);
    uint8_t dst_logicpos = static_cast<uint8_t>(TPosition::A2);
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    srcTensor = CreateLocalTensor<T>(src_buffer_offset, src_logicpos);
    dstTensor = CreateLocalTensor<T>(dst_buffer_offset, dst_logicpos);
    LoadDataWithTranspose(dstTensor, srcTensor, params);
}

template<typename T>
__aicore__ inline void LoadCbufToCbTranspose(__cb__ T *dst, __cbuf__ T *src, uint16_t indexID, uint8_t repeat,
                                            uint16_t srcStride, uint16_t dstStride, bool addrmode,
                                            uint16_t dstFracStride)
{
    LoadData2dTransposeParams params(
        indexID,
        repeat,
        srcStride,
        dstStride,
        dstFracStride,
        addrmode
    );
    uint32_t src_buffer_offset = reinterpret_cast<uint64_t>(src);
    uint32_t dst_buffer_offset = reinterpret_cast<uint64_t>(dst);
    uint8_t src_logicpos = static_cast<uint8_t>(TPosition::C1);
    uint8_t dst_logicpos = static_cast<uint8_t>(TPosition::B2);
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    srcTensor = CreateLocalTensor<T>(src_buffer_offset, src_logicpos);
    dstTensor = CreateLocalTensor<T>(dst_buffer_offset, dst_logicpos);
    LoadDataWithTranspose(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void LoadCbufToCa(__ca__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
                                    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                                    uint8_t addr_cal_mode)
{
    LoadData2dParams params(
        baseIdx,
        repeat,
        srcStride,
        sid,
        dstStride,
        transpose,
        addr_cal_mode
    );
    uint32_t src_buffer_offset = reinterpret_cast<uint64_t>(src);
    uint32_t dst_buffer_offset = reinterpret_cast<uint64_t>(dst);
    uint8_t src_logicpos = static_cast<uint8_t>(TPosition::C1);
    uint8_t dst_logicpos = static_cast<uint8_t>(TPosition::A2);
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    srcTensor = CreateLocalTensor<T>(src_buffer_offset, src_logicpos);
    dstTensor = CreateLocalTensor<T>(dst_buffer_offset, dst_logicpos);
    LoadData(dstTensor, srcTensor, params);
}


template<typename T>
__aicore__ inline void LoadCbufToCb(__cb__ T *dst, __cbuf__ T *src, uint16_t baseIdx, uint8_t repeat,
                                    uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                                    uint8_t addr_cal_mode)
{
    LoadData2dParams params(
        baseIdx,
        repeat,
        srcStride,
        sid,
        dstStride,
        transpose,
        addr_cal_mode
    );
    uint32_t src_buffer_offset = reinterpret_cast<uint64_t>(src);
    uint32_t dst_buffer_offset = reinterpret_cast<uint64_t>(dst);
    uint8_t src_logicpos = static_cast<uint8_t>(TPosition::C1);
    uint8_t dst_logicpos = static_cast<uint8_t>(TPosition::B2);
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    srcTensor = CreateLocalTensor<T>(src_buffer_offset, src_logicpos);
    dstTensor = CreateLocalTensor<T>(dst_buffer_offset, dst_logicpos);
    LoadData(dstTensor, srcTensor, params);
}


__aicore__ inline void GetBlockIdx(int32_t loop_idx, int32_t m_loop, int32_t n_loop, int32_t swizzl_direction,
                                   int32_t swizzl_count, int64_t &m_idx, int64_t &n_idx)
{
    uint32_t in_batch_idx = loop_idx % (m_loop * n_loop);
    if (swizzl_direction == 0) {
        uint32_t tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);

        uint32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_row;
        n_idx = in_tile_block_idx / n_row;
        if (tile_block_idx % TILE_BLOCK_MOD != 0) {
            n_idx = n_loop - n_idx - 1;
        }
    } else if (swizzl_direction == 1) {
        uint32_t tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);

        uint32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        m_idx = in_tile_block_idx / n_col;
        n_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_col;
        if (tile_block_idx % TILE_BLOCK_MOD != 0) {
            m_idx = m_loop - m_idx - 1;
        }
    }
}

template <pipe_t pipe>
__aicore__ inline void FFTSCrossCoreSync(uint64_t mode, uint64_t flag_id)
{
    uint64_t config = 1 | (mode << 4) | (flag_id << 8);
    ffts_cross_core_sync(pipe, config);
}


template <typename T>
__aicore__ GlobalTensor<T> CreateGlobalTensor(__gm__ T *addr)
{
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(addr);
    return tensor;
}

#endif // MATMUL_ALLREDUCE_ADD_RMSNORM_H
