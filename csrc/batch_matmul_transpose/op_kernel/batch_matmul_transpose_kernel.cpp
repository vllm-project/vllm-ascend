// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//

#define __aicore__ [aicore]
#include "kernel_operator.h"
#include "../op_host/tiling/tiling_data.h"
#include "../../mla_preprocess/op_kernel/kernel/common.h"
#include "../../mla_preprocess/op_kernel/kernel/hardware.h"
#include "../../mla_preprocess/op_kernel/kernel/mma.h"
#include "../../mla_preprocess/op_kernel/kernel/utils.h"
#include "../../mla_preprocess/op_kernel/kernel/iterator.h"
#include "../../kernels/math_utils.h"

constexpr uint32_t L0_PINGPONG_BUFFER_LEN = 16384;
constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 131072;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_256 = 256;
constexpr uint64_t ND2NZ_STRIDE_LIMIT = 65536;
constexpr uint64_t BLOCK_SIZE_16 = 16;
constexpr uint64_t CONST_16UL = 16;
constexpr uint64_t CONST_256UL = 256;

struct MatCoord {
    uint64_t m{0};
    uint64_t k{0};
    uint64_t n{0};
};

using namespace device_utils;

template <uint32_t SwizzleDirect, bool TA, bool TB, typename InDtype = half, typename OutDtype = half,
          DataFormat FormatB = DataFormat::ND>
class PpMatmulEinSum
{
    using LocalTensor = AscendC::LocalTensor<InDtype>;
    template <DataFormat srcFormat = DataFormat::ND, DataFormat dstFormat = DataFormat::ND>
    using CopyGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, InDtype, srcFormat, dstFormat>;
    using LoadCbufToCa = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, TA, DataFormat::ZN, DataFormat::ZZ>;
    using LoadCbufToCb = l1_to_l0_b<ArchType::ASCEND_V220, InDtype, TB, DataFormat::ZN, DataFormat::NZ>;
    using Mad = mmad<ArchType::ASCEND_V220, InDtype, InDtype, float, TA>;
    using CopyCcToGm = l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, OutDtype, float>;

public:
    __aicore__ explicit PpMatmulEinSum(){};

    __aicore__ __force_inline__ void Init(__gm__ uint8_t *__restrict__ a, __gm__ uint8_t *__restrict__ b,
                                          __gm__ uint8_t *__restrict__ c, __gm__ uint8_t *__restrict__ tiling_data)
    {
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(a));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(b));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(c));
        auto gm_tiling_data = reinterpret_cast<__gm__ pp_matmul::PpMatmulTilingData *>(tiling_data);

        batch_size = gm_tiling_data->opShape.batchSize;
        m = gm_tiling_data->opShape.m;
        k = gm_tiling_data->opShape.k;
        n = gm_tiling_data->opShape.n;
        m0 = gm_tiling_data->opShape.m0;
        k0 = gm_tiling_data->opShape.k0;
        n0 = gm_tiling_data->opShape.n0;
        tdim.m = gm_tiling_data->mLoop;
        tdim.k = gm_tiling_data->kLoop;
        tdim.n = gm_tiling_data->nLoop;
        core_loop = gm_tiling_data->coreLoop;
        swizzle_cnt = gm_tiling_data->swizzlCount;
        en_shuffle_k = gm_tiling_data->enShuffleK;

        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        l1_base_a = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_base_b = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(
            RoundUp<uint64_t>(m0 * k0 * sizeof(InDtype), CONST_256UL));
        l0a_base = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0b_base = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        ping_flag = 1;
    }

    __aicore__ __force_inline__ void GetBlockIdx(uint64_t index, MatCoord &tidx)
    {
        uint64_t in_batch_idx = index % (tdim.m * tdim.n);
        if constexpr (SwizzleDirect == 0) {  // Zn
            uint64_t tile_block_loop = (tdim.m + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.n);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.n);

            uint64_t n_row = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_row = tdim.m - swizzle_cnt * tile_block_idx;
            }
            tidx.m = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
            tidx.n = in_tile_block_idx / n_row;
            if (tile_block_idx % 2 != 0) {
                tidx.n = tdim.n - tidx.n - 1;
            }
        } else if constexpr (SwizzleDirect == 1) {  // Nz
            uint64_t tile_block_loop = (tdim.n + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.m);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.m);

            uint64_t n_col = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_col = tdim.n - swizzle_cnt * tile_block_idx;
            }
            tidx.m = in_tile_block_idx / n_col;
            tidx.n = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
            if (tile_block_idx % 2 != 0) {
                tidx.m = tdim.m - tidx.m - 1;
            }
        }
    }

    __aicore__ __force_inline__ void Process()
    {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        for (uint64_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += num_core) {
            uint64_t batch_idx = loop_idx / tdim.n / tdim.m;
            MatCoord tidx{0};
            GetBlockIdx(loop_idx, tidx);
            uint64_t offset_a = 0, offset_b = 0, offset_a_next = 0, offset_b_next = 0;
            uint64_t offset_c = tidx.m * m0 * batch_size * n + batch_idx * n + tidx.n * n0;
            uint64_t m_actual = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
            uint64_t n_actual = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
            uint64_t m_round = RoundUp<uint64_t, CONST_16UL>(m_actual);
            uint64_t n_round = RoundUp<uint64_t, CONST_16UL>(n_actual);
            uint64_t mn_max = m_round > n_round ? m_round : n_round;
            uint64_t k_part_len = L0_PINGPONG_BUFFER_LEN / mn_max / CONST_16 * CONST_16;
            uint64_t shuffle_k = en_shuffle_k ? (core_idx % tdim.k) : 0;
            if (TA) {
                offset_a = shuffle_k * k0 * m * batch_size + batch_idx * m + tidx.m * m0;
            } else {
                offset_a = tidx.m * m0 * batch_size * k + batch_idx * k + shuffle_k * k0;
            }

            if (TB) {
                if constexpr (FormatB != DataFormat::NZ) {
                    offset_b = batch_idx * k * n + tidx.n * n0 * k + shuffle_k * k0;
                } else {
                    offset_b = batch_idx * RoundUp<uint64_t, CONST_16UL>(k) * RoundUp<uint64_t, CONST_16UL>(n) +
                               tidx.n * n0 * BLOCK_SIZE_16 + shuffle_k * k0 * RoundUp<uint64_t, CONST_16UL>(n);
                }
            } else {
                if constexpr (FormatB != DataFormat::NZ) {
                    offset_b = batch_idx * k * n + shuffle_k * k0 * n + tidx.n * n0;
                } else {
                    offset_b = batch_idx * RoundUp<uint64_t, CONST_16UL>(k) * RoundUp<uint64_t, CONST_16UL>(n) +
                               shuffle_k * k0 * BLOCK_SIZE_16 + tidx.n * n0 * RoundUp<uint64_t, CONST_16UL>(k);
                }
            }

            uint64_t k_actual = (shuffle_k == tdim.k - 1) ? k - shuffle_k * k0 : k0;
            uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;

            LocalTensor l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
            LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
            LocalTensor l0a_buf = ping_flag ? l0a_base : l0a_base[L0_PINGPONG_BUFFER_LEN];
            LocalTensor l0b_buf = ping_flag ? l0b_base : l0b_base[L0_PINGPONG_BUFFER_LEN];
            event_t event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            if (loop_idx == core_idx) {
                WAIT_FLAG(MTE1, MTE2, event_id);
                // *** load matrix A to L1
                if ((m == 1) || (m_actual == 1 && !TA)) {
                    CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(l1_buf_a,        // dst
                                                                 gm_a[offset_a],  // src
                                                                 1,               // nTileActual
                                                                 16,              // nTileCeil
                                                                 1,               // nVal
                                                                 k_actual,        // kTileActual
                                                                 k_round,         // kTileCeil
                                                                 k);              // dVal
                } else {
                    if (TA) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a,         // dst
                                                                     gm_a[offset_a],   // src
                                                                     k_actual,         // nTileActual
                                                                     k_round,          // nTileCeil
                                                                     k,                // nVal
                                                                     m_actual,         // dTileActual
                                                                     m_round,          // dTileCeil
                                                                     m * batch_size);  // dVal
                    } else {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a,         // dst
                                                                     gm_a[offset_a],   // src
                                                                     m_actual,         // nTileActual
                                                                     m_round,          // nTileCeil
                                                                     m,                // nVal
                                                                     k_actual,         // dTileActual
                                                                     k_round,          // dTileCeil
                                                                     k * batch_size);  // dVal
                    }
                }
                SET_FLAG(MTE2, MTE1, event_id);
                // *** load matrix B to L1
                wait_flag(PIPE_MTE1, PIPE_MTE2, event_id + 2);
                if constexpr (FormatB != DataFormat::NZ) {
                    if (TB) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b,        // dst
                                                                     gm_b[offset_b],  // src
                                                                     n_actual,        // nTileActual
                                                                     n_round,         // nTileCeil
                                                                     n,               // nVal
                                                                     k_actual,        // dTileActual
                                                                     k_round,         // dTileCeil
                                                                     k);              // dVal
                    } else {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b,        // dst
                                                                     gm_b[offset_b],  // src
                                                                     k_actual,        // nTileActual
                                                                     k_round,         // nTileCeil
                                                                     k,               // nVal
                                                                     n_actual,        // dTileActual
                                                                     n_round,         // dTileCeil
                                                                     n);              // dVal
                    }
                } else {
                    if (TB) {
                        CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b,                           // dst
                                                                     gm_b[offset_b],                     // src
                                                                     n_actual,                           // nTileActual
                                                                     n_round,                            // nTileCeil
                                                                     RoundUp<uint64_t, CONST_16UL>(n),   // nVal
                                                                     k_actual,                           // dTileActual
                                                                     k_round,                            // dTileCeil
                                                                     RoundUp<uint64_t, CONST_16UL>(k));  // dVal
                    } else {
                        CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b,                           // dst
                                                                     gm_b[offset_b],                     // src
                                                                     k_actual,                           // nTileActual
                                                                     k_round,                            // nTileCeil
                                                                     RoundUp<uint64_t, CONST_16UL>(k),   // nVal
                                                                     n_actual,                           // dTileActual
                                                                     n_round,                            // dTileCeil
                                                                     RoundUp<uint64_t, CONST_16UL>(n));  // dVal
                    }
                }
                SET_FLAG(MTE2, MTE1, event_id + 2);
            }

            for (tidx.k = 0; tidx.k < tdim.k; ++tidx.k) {
                shuffle_k = en_shuffle_k ? (tidx.k + core_idx) % tdim.k : tidx.k;
                uint64_t k_actual = (shuffle_k == (tdim.k - 1)) ? (k - shuffle_k * k0) : k0;
                uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;
                fdim.k = (k_actual + k_part_len - 1) / k_part_len;

                LocalTensor l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
                LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
                auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                if (tidx.k < tdim.k - 1) {
                    uint64_t shuffle_k_next = en_shuffle_k ? (core_idx + tidx.k + 1) % tdim.k : (tidx.k + 1);
                    if (TA) {
                        offset_a_next = shuffle_k_next * k0 * m * batch_size + batch_idx * m + tidx.m * m0;
                    } else {
                        offset_a_next = tidx.m * m0 * batch_size * k + batch_idx * k + shuffle_k_next * k0;
                    }

                    if (TB) {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = batch_idx * k * n + tidx.n * n0 * k + shuffle_k_next * k0;
                        } else {
                            offset_b_next =
                                batch_idx * RoundUp<uint64_t, CONST_16UL>(k) * RoundUp<uint64_t, CONST_16UL>(n) +
                                tidx.n * n0 * BLOCK_SIZE_16 + shuffle_k_next * k0 * RoundUp<uint64_t, CONST_16UL>(n);
                        }
                    } else {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = batch_idx * k * n + shuffle_k_next * k0 * n + tidx.n * n0;
                        } else {
                            offset_b_next =
                                batch_idx * RoundUp<uint64_t, CONST_16UL>(k) * RoundUp<uint64_t, CONST_16UL>(n) +
                                shuffle_k_next * k0 * BLOCK_SIZE_16 + tidx.n * n0 * RoundUp<uint64_t, CONST_16UL>(k);
                        }
                    }

                    uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;

                    LocalTensor l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
                    LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    // *** load matrix A to L1
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(l1_buf_a_next,        // dst
                                                                     gm_a[offset_a_next],  // src
                                                                     m_actual,             // nTileActual
                                                                     m_round,              // nTileCeil
                                                                     m,                    // nVal
                                                                     k_actual_next,        // kTileActual
                                                                     k_round_next,         // kTileCeil
                                                                     k);                   // dVal
                    } else {
                        if (TA) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,        // dst
                                                                         gm_a[offset_a_next],  // src
                                                                         k_actual_next,        // nTileActual
                                                                         k_round_next,         // nTileCeil
                                                                         k,                    // nVal
                                                                         m_actual,             // dTileActual
                                                                         m_round,              // dTileCeil
                                                                         m * batch_size);      // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,        // dst
                                                                         gm_a[offset_a_next],  // src
                                                                         m_actual,             // nTileActual
                                                                         m_round,              // nTileCeil
                                                                         m,                    // nVal
                                                                         k_actual_next,        // dTileActual
                                                                         k_round_next,         // dTileCeil
                                                                         k * batch_size);      // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // *** load matrix B to L1
                    wait_flag(PIPE_MTE1, PIPE_MTE2, event_id_next + 2);
                    if constexpr (FormatB != DataFormat::NZ) {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         n_actual,             // nTileActual
                                                                         n_round,              // nTileCeil
                                                                         n,                    // nVal
                                                                         k_actual_next,        // dTileActual
                                                                         k_round_next,         // dTileCeil
                                                                         k);                   // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         k_actual_next,        // nTileActual
                                                                         k_round_next,         // nTileCeil
                                                                         k,                    // nVal
                                                                         n_actual,             // dTileActual
                                                                         n_round,              // dTileCeil
                                                                         n);                   // dVal
                        }
                    } else {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         n_actual,             // nTileActual
                                                                         n_round,              // nTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(n),  // nVal
                                                                         k_actual_next,  // dTileActual
                                                                         k_round_next,   // dTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(k));  // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         k_actual_next,        // nTileActual
                                                                         k_round_next,         // nTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(k),  // nVal
                                                                         n_actual,  // dTileActual
                                                                         n_round,   // dTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(n));  // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);
                }

                if (tidx.k == tdim.k - 1 && loop_idx + num_core < core_loop) {
                    uint64_t b_idx_next = (loop_idx + num_core) / tdim.n / tdim.m;
                    MatCoord tidx{0};
                    GetBlockIdx(loop_idx + num_core, tidx);
                    uint64_t shuffle_k_next = en_shuffle_k ? (core_idx % tdim.k) : 0;
                    uint64_t m_actual_next = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
                    uint64_t n_actual_next = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
                    uint64_t m_round_next = (m_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t n_round_next = (n_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    if (TA) {
                        offset_a_next = shuffle_k_next * k0 * m * batch_size + b_idx_next * m + tidx.m * m0;
                    } else {
                        offset_a_next = tidx.m * m0 * batch_size * k + b_idx_next * k + shuffle_k_next * k0;
                    }

                    if (TB) {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = b_idx_next * k * n + tidx.n * n0 * k + shuffle_k_next * k0;
                        } else {
                            offset_b_next =
                                b_idx_next * RoundUp<uint64_t, CONST_16UL>(k) * RoundUp<uint64_t, CONST_16UL>(n) +
                                tidx.n * n0 * BLOCK_SIZE_16 + shuffle_k_next * k0 * RoundUp<uint64_t, CONST_16UL>(n);
                        }
                    } else {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = b_idx_next * k * n + shuffle_k_next * k0 * n + tidx.n * n0;
                        } else {
                            offset_b_next =
                                b_idx_next * RoundUp<uint64_t, CONST_16UL>(k) * RoundUp<uint64_t, CONST_16UL>(n) +
                                shuffle_k_next * k0 * BLOCK_SIZE_16 + tidx.n * n0 * RoundUp<uint64_t, CONST_16UL>(k);
                        }
                    }

                    LocalTensor l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_LEN];
                    LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_LEN];
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    // *** load matrix A to L1
                    if (m == 1 || m_actual_next == 1 && !TA) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(l1_buf_a_next,        // dst
                                                                     gm_a[offset_a_next],  // src
                                                                     m_actual_next,        // nTileActual
                                                                     m_round_next,         // nTileCeil
                                                                     m,                    // nVal
                                                                     k_actual_next,        // kTileActual
                                                                     k_round_next,         // kTileCeil
                                                                     k);                   // dVal
                    } else {
                        if (TA) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,        // dst
                                                                         gm_a[offset_a_next],  // src
                                                                         k_actual_next,        // nTileActual
                                                                         k_round_next,         // nTileCeil
                                                                         k,                    // nVal
                                                                         m_actual_next,        // dTileActual
                                                                         m_round_next,         // dTileCeil
                                                                         m * batch_size);      // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,        // dst
                                                                         gm_a[offset_a_next],  // src
                                                                         m_actual_next,        // nTileActual
                                                                         m_round_next,         // nTileCeil
                                                                         m,                    // nVal
                                                                         k_actual_next,        // dTileActual
                                                                         k_round_next,         // dTileCeil
                                                                         k * batch_size);      // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // *** load matrix B to L1
                    wait_flag(PIPE_MTE1, PIPE_MTE2, event_id_next + 2);
                    if constexpr (FormatB != DataFormat::NZ) {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         n_actual_next,        // nTileActual
                                                                         n_round_next,         // nTileCeil
                                                                         n,                    // nVal
                                                                         k_actual_next,        // dTileActual
                                                                         k_round_next,         // dTileCeil
                                                                         k);                   // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         k_actual_next,        // nTileActual
                                                                         k_round_next,         // nTileCeil
                                                                         k,                    // nVal
                                                                         n_actual_next,        // dTileActual
                                                                         n_round_next,         // dTileCeil
                                                                         n);                   // dVal
                        }
                    } else {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         n_actual_next,        // nTileActual
                                                                         n_round_next,         // nTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(n),  // nVal
                                                                         k_actual_next,  // dTileActual
                                                                         k_round_next,   // dTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(k));  // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,        // dst
                                                                         gm_b[offset_b_next],  // src
                                                                         k_actual_next,        // nTileActual
                                                                         k_round_next,         // nTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(k),  // nVal
                                                                         n_actual_next,  // dTileActual
                                                                         n_round_next,   // dTileCeil
                                                                         RoundUp<uint64_t, CONST_16UL>(n));  // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);
                }

                MatCoord fidx{0};
                for (fidx.k = 0; fidx.k < fdim.k; ++fidx.k) {
                    uint32_t k0_round = (fidx.k < fdim.k - 1) ? k_part_len : k_round - fidx.k * k_part_len;
                    uint32_t k0_actual = (fidx.k < fdim.k - 1) ? k_part_len : k_actual - fidx.k * k_part_len;

                    auto mte1_mad_ping_flag = 1 - fidx.k % 2;
                    auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                    auto l0a_buf = l0a_base[(fidx.k % 2) * L0_PINGPONG_BUFFER_LEN];
                    auto l0b_buf = l0b_base[(fidx.k % 2) * L0_PINGPONG_BUFFER_LEN];

                    // *** load matrix A from L1 to L0A
                    if (fidx.k == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id);
                    }
                    WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf,                        // dst
                            l1_buf_a[fidx.k * k_part_len],  // src
                            0,                              // mTileCeil
                            CeilDiv<CONST_256>(k0_round),   // kPartCeil
                            0,                              // mSrcStride
                            1,                              // kSrcStride
                            0,                              // mDstStride
                            0);                             // kDstStride
                    } else {
                        if (TA) {
                            LoadCbufToCa(l0a_buf,                                   // l0Tensor
                                         l1_buf_a[fidx.k * k_part_len * CONST_16],  // l1Tensor
                                         m_round,                                   // mTileCeil
                                         k0_round,                                  // kPartCeil
                                         k_round / CONST_16,                        // mSrcStride
                                         1,                                         // kSrcStride
                                         k0_round / CONST_16,                       // mDstStride
                                         1);                                        // kDstStride
                        } else {
                            LoadCbufToCa(l0a_buf,                                  // l0Tensor
                                         l1_buf_a[fidx.k * k_part_len * m_round],  // l1Tensor
                                         m_round,                                  // mTileCeil
                                         k0_round,                                 // kPartCeil
                                         1,                                        // mSrcStride
                                         m_round / CONST_16,                       // kSrcStride
                                         k0_round / CONST_16,                      // mDstStride
                                         1);                                       // kDstStride
                        }
                    }
                    if (fidx.k == fdim.k - 1) {
                        SET_FLAG(MTE1, MTE2, event_id);
                    }

                    // *** load matrix B from L1 to L0B
                    if (fidx.k == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id + 2);
                    }
                    if (TB) {
                        LoadCbufToCb(l0b_buf,                                  // l0Tensor
                                     l1_buf_b[fidx.k * k_part_len * n_round],  // l1Tensor
                                     n_round,                                  // nTileCeil
                                     k0_round,                                 // kPartCeil
                                     1,                                        // nSrcStride
                                     n_round / CONST_16,                       // kSrcStride
                                     1,                                        // nDstStride
                                     k0_round / CONST_16);                     // kDstStride
                    } else {
                        LoadCbufToCb(l0b_buf,                                   // l0Tensor
                                     l1_buf_b[fidx.k * k_part_len * CONST_16],  // l1Tensor
                                     n_round,                                   // nTileCeil
                                     k0_round,                                  // kPartCeil
                                     k_round / CONST_16,                        // nSrcStride
                                     1,                                         // kSrcStride
                                     1,                                         // nDstStride
                                     n_round / CONST_16);                       // kDstStride
                    }
                    if (fidx.k == fdim.k - 1) {
                        SET_FLAG(MTE1, MTE2, event_id + 2);
                    }

                    SET_FLAG(MTE1, M, mte1_mad_event_id);
                    WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                    bool init_c = (tidx.k == 0 && fidx.k == 0);
                    if (init_c) {
                        WAIT_FLAG(FIX, M, EVENT_ID0);
                    }

                    if (m != 1 && m_actual == 1 && TA) {
                        Mad(l0c_buf,    // c
                            l0a_buf,    // a
                            l0b_buf,    // b
                            CONST_16,   // mTileActual
                            n_actual,   // nTileActual
                            k0_actual,  // kTileActual
                            init_c);    // initC
                    } else {
                        Mad(l0c_buf,    // c
                            l0a_buf,    // a
                            l0b_buf,    // b
                            m_actual,   // mTileActual
                            n_actual,   // nTileActual
                            k0_actual,  // kTileActual
                            init_c);    // initC
                    }

                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, mte1_mad_event_id);
                }

                ping_flag = 1 - ping_flag;
            }

            SET_FLAG(M, FIX, EVENT_ID0);
            WAIT_FLAG(M, FIX, EVENT_ID0);

            // copy from L0C to gm
            CopyCcToGm(gm_c[offset_c],   // dst
                       l0c_buf,          // src
                       m_actual,         // mTileActual
                       n_actual,         // nTileActual
                       m_round,          // mTileCeil
                       n * batch_size);  // nActual
            SET_FLAG(FIX, M, EVENT_ID0);
        }

        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

private:
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<InDtype> gm_b;
    AscendC::GlobalTensor<OutDtype> gm_c;
    AscendC::LocalTensor<InDtype> l1_base_a;
    AscendC::LocalTensor<InDtype> l1_base_b;
    AscendC::LocalTensor<InDtype> l0a_base;
    AscendC::LocalTensor<InDtype> l0b_base;
    AscendC::LocalTensor<float> l0c_buf;

    uint32_t num_core{0};
    uint32_t batch_size{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    MatCoord tdim{0};
    MatCoord fdim{0};
    uint32_t core_loop{0};
    uint32_t swizzle_cnt{1};
    uint32_t core_idx{0};
    uint32_t en_shuffle_k{0};
    uint32_t ping_flag{0};
};

extern "C" __global__ __aicore__ void batch_matmul_transpose(GM_ADDR gm_a, GM_ADDR gm_b, GM_ADDR gm_c,
                                                             GM_ADDR gm_tiling_data)
{
    PpMatmulEinSum<0, false, false, half, half, DataFormat::ND>
        einsum_0_n_fp16_nd;  // swizzleDir[0] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, false, half, half, DataFormat::ND>
        einsum_1_n_fp16_nd;  // swizzleDir[1] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<0, false, true, half, half, DataFormat::ND>
        einsum_0_t_fp16_nd;  // swizzleDir[0] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, true, half, half, DataFormat::ND>
        einsum_1_t_fp16_nd;  // swizzleDir[1] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<0, false, false, __bf16, __bf16, DataFormat::ND>
        einsum_0_n_bf16_nd;  // swizzleDir[0] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, false, __bf16, __bf16, DataFormat::ND>
        einsum_1_n_bf16_nd;  // swizzleDir[1] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<0, false, true, __bf16, __bf16, DataFormat::ND>
        einsum_0_t_bf16_nd;  // swizzleDir[0] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]
    PpMatmulEinSum<1, false, true, __bf16, __bf16, DataFormat::ND>
        einsum_1_t_bf16_nd;  // swizzleDir[1] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[0]

    PpMatmulEinSum<0, false, false, half, half, DataFormat::NZ>
        einsum_0_n_fp16_nz;  // swizzleDir[0] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, false, half, half, DataFormat::NZ>
        einsum_1_n_fp16_nz;  // swizzleDir[1] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<0, false, true, half, half, DataFormat::NZ>
        einsum_0_t_fp16_nz;  // swizzleDir[0] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, true, half, half, DataFormat::NZ>
        einsum_1_t_fp16_nz;  // swizzleDir[1] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<0, false, false, __bf16, __bf16, DataFormat::NZ>
        einsum_0_n_bf16_nz;  // swizzleDir[0] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, false, __bf16, __bf16, DataFormat::NZ>
        einsum_1_n_bf16_nz;  // swizzleDir[1] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<0, false, true, __bf16, __bf16, DataFormat::NZ>
        einsum_0_t_bf16_nz;  // swizzleDir[0] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]
    PpMatmulEinSum<1, false, true, __bf16, __bf16, DataFormat::NZ>
        einsum_1_t_bf16_nz;  // swizzleDir[1] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0]
                             // DataFormatB[1]

    SetPadding<uint64_t>((uint64_t)0);
    SetNdpara(1, 0, 0);
    SetAtomicnone();

    // get tiling args
    auto tiling_data = reinterpret_cast<__gm__ pp_matmul::PpMatmulTilingData *>(gm_tiling_data);
    uint32_t masked_key = tiling_data->tilingKey >> 2;

    switch (masked_key) {
        case 0b00000100100100:
        case 0b01000100100100:
            einsum_0_n_fp16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_n_fp16_nd.Process();
            break;
        case 0b00100100100100:
        case 0b01100100100100:
            einsum_0_t_fp16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_t_fp16_nd.Process();
            break;
        case 0b10000100100100:
        case 0b11000100100100:
            einsum_1_n_fp16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_n_fp16_nd.Process();
            break;
        case 0b10100100100100:
        case 0b11100100100100:
            einsum_1_t_fp16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_t_fp16_nd.Process();
            break;
        case 0b00001001001000:
        case 0b01001001001000:
            einsum_0_n_bf16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_n_bf16_nd.Process();
            break;
        case 0b00101001001000:
        case 0b01101001001000:
            einsum_0_t_bf16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_t_bf16_nd.Process();
            break;
        case 0b10001001001000:
        case 0b11001001001000:
            einsum_1_n_bf16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_n_bf16_nd.Process();
            break;
        case 0b10101001001000:
        case 0b11101001001000:
            einsum_1_t_bf16_nd.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_t_bf16_nd.Process();
            break;

        case 0b00000100100101:
        case 0b01000100100101:
            einsum_0_n_fp16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_n_fp16_nz.Process();
            break;
        case 0b00100100100101:
        case 0b01100100100101:
            einsum_0_t_fp16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_t_fp16_nz.Process();
            break;
        case 0b10000100100101:
        case 0b11000100100101:
            einsum_1_n_fp16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_n_fp16_nz.Process();
            break;
        case 0b10100100100101:
        case 0b11100100100101:
            einsum_1_t_fp16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_t_fp16_nz.Process();
            break;
        case 0b00001001001001:
        case 0b01001001001001:
            einsum_0_n_bf16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_n_bf16_nz.Process();
            break;
        case 0b00101001001001:
        case 0b01101001001001:
            einsum_0_t_bf16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_0_t_bf16_nz.Process();
            break;
        case 0b10001001001001:
        case 0b11001001001001:
            einsum_1_n_bf16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_n_bf16_nz.Process();
            break;
        case 0b10101001001001:
        case 0b11101001001001:
            einsum_1_t_bf16_nz.Init(gm_a, gm_b, gm_c, gm_tiling_data);
            einsum_1_t_bf16_nz.Process();
            break;
        default:
            break;
    }
}


namespace vllm_ascend {

extern void batch_matmul_transpose_impl(
    void* stream,
    void* gm_a,
    void* gm_b,
    void* gm_c,
    void* gm_tiling_data,
    const uint32_t block_dim)
{
    batch_matmul_transpose<<<block_dim, nullptr, stream>>>(
        gm_a,
        gm_b,
        gm_c,
        gm_tiling_data);
}

}