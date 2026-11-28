// SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt.
//
// Derived from the Arch22 QSFA Cube and Vector stages in CANN 9.2.1
// ops-transformer (commit 8fbf1ffea0cf0c401407fdf202743a75e0a427b7).
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "runtime/rt_ffts.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/local_tensor_buffer.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

using A3 = Catlass::Arch::AtlasA2;
using AType = Catlass::Gemm::GemmType<bfloat16_t, Catlass::layout::RowMajor>;
using BType = Catlass::Gemm::GemmType<bfloat16_t, Catlass::layout::ColumnMajor>;
using CubeMmad = Catlass::Gemm::Tile::TileMmad<A3, AType, BType, void>;
constexpr int cores = 20;

template <class T>
__aicore__ inline AscendC::GlobalTensor<T> MakeGlobal(GM_ADDR address) {
  AscendC::GlobalTensor<T> tensor;
  tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
  return tensor;
}

extern "C" __global__ __aicore__ void qsfa_catlass_resident_kv_wide_384(
    GM_ADDR query_addr, GM_ADDR packed_addr, GM_ADDR indices_addr, GM_ADDR page_table_addr, GM_ADDR query_ends_addr,
    GM_ADDR kv_lengths_addr, GM_ADDR output_addr, GM_ADDR kv_nz_addr, GM_ADDR scores_addr, GM_ADDR probs_addr,
    GM_ADDR partial_addr, GM_ADDR numerator_addr, GM_ADDR valid_gm_addr, int rows, float softmax_scale,
    uint64_t ffts_addr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::SetSyncBaseAddr(ffts_addr);
  auto kv_nz = MakeGlobal<bfloat16_t>(kv_nz_addr);
  auto scores = MakeGlobal<float>(scores_addr);
  auto probs = MakeGlobal<bfloat16_t>(probs_addr);
  auto partial = MakeGlobal<float>(partial_addr);
  const int rows_per_core = (rows + cores - 1) / cores;

  if ASCEND_IS_AIC {
    const int core = AscendC::GetBlockIdx();
    const int row_begin = core * rows_per_core;
    if (row_begin >= rows) {
      return;
    }
    const int remaining = rows - row_begin;
    const int task_count = (remaining < rows_per_core ? remaining : rows_per_core) * 4;
    auto query = MakeGlobal<bfloat16_t>(query_addr);
    AscendC::TPipe pipe;
    Catlass::Arch::LocalTensorBuffer<A3, AscendC::TPosition::A1> cube_l1;
    Catlass::Arch::LocalTensorBuffer<A3, AscendC::TPosition::A2> cube_l0a;
    Catlass::Arch::LocalTensorBuffer<A3, AscendC::TPosition::B2> cube_l0b;
    Catlass::Arch::LocalTensorBuffer<A3, AscendC::TPosition::CO1> cube_l0c;
    pipe.Destroy();
    CubeMmad tile_mmad;
    constexpr uint32_t qsfa_heads = 64;
    constexpr uint32_t qsfa_s2 = 512;
    constexpr uint32_t qsfa_combined = 576;
    constexpr uint32_t qsfa_c0 = 16;
    constexpr uint32_t qsfa_n_panel = 128;
    constexpr uint32_t qsfa_bank_l1 = qsfa_heads * qsfa_combined;
    constexpr uint32_t qsfa_bank_ab = 32 * 1024 / sizeof(bfloat16_t);
    constexpr uint32_t qsfa_bank_c = 64 * 1024 / sizeof(float);
    constexpr uint32_t qsfa_ab_event_base = 3;
    constexpr uint32_t qsfa_qp_events[2] = {2, 3};
    constexpr uint32_t qsfa_kv_events[4] = {6, 7, 4, 5};
    uint32_t qsfa_kv_cursor = 0;
    uint32_t qsfa_ab_cursor = 0;
    uint32_t qsfa_c_cursor = 0;
    auto qsfa_l1 = cube_l1.GetBufferByByte<bfloat16_t>(0);
    auto qsfa_a = cube_l0a.GetBufferByByte<bfloat16_t>(0);
    auto qsfa_b = cube_l0b.GetBufferByByte<bfloat16_t>(0);
    auto qsfa_c = cube_l0c.GetBufferByByte<float>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base + 1);
    for (uint32_t qsfa_i = 0; qsfa_i < 2; ++qsfa_i) {
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_qp_events[qsfa_i]);
    }
    for (uint32_t qsfa_i = 0; qsfa_i < 4; ++qsfa_i) {
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_kv_events[qsfa_i]);
    }

    // L1: Q (72 KiB), P (64 KiB), two scratch banks (72 KiB each),
    // and one retained 384-channel QK panel per task parity (96 KiB each).
    constexpr uint32_t qsfa_query_l1_bank = 0;
    constexpr uint32_t qsfa_query_l1_event = 0;
    constexpr uint32_t qsfa_prob_offset = qsfa_bank_l1;
    constexpr uint32_t qsfa_prob_panel_elements = qsfa_heads * 256;
    constexpr uint32_t qsfa_work_kv_offset = qsfa_prob_offset + 2 * qsfa_prob_panel_elements;
    constexpr uint32_t qsfa_retained_kv_offset = qsfa_work_kv_offset + 2 * qsfa_bank_l1;
    constexpr uint32_t qsfa_retained_latent_channels = 384;
    constexpr uint32_t qsfa_reused_latent_channels = 384;
    constexpr uint32_t qsfa_retained_kv_elements = qsfa_n_panel * qsfa_retained_latent_channels;
    static_assert((qsfa_retained_kv_offset + 2 * qsfa_retained_kv_elements) * sizeof(bfloat16_t) <= 512 * 1024);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_query_l1_event);
    for (int credit = 0; credit < 4; ++credit) {
      AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(3);
    }
    for (int tick = 0; tick <= task_count; ++tick) {
      if (tick < task_count) {
        AscendC::CrossCoreWaitFlag(6);
        const int64_t qsfa_qk_query_offset = int64_t(row_begin + tick / 4) * 64 * 576;
        const int64_t qsfa_qk_kv_offset = (int64_t(core) * 4 + tick % 4) * 512 * 576;
        const int64_t qsfa_qk_result_offset = (int64_t(core) * 2 + tick % 2) * 128 * 512;
        constexpr uint32_t qsfa_qk_tokens = 512;
        [[maybe_unused]] const uint32_t qsfa_qk_panel = tick % 4;
        {
          constexpr uint32_t qk_l1_k = 288;
          constexpr uint32_t qk_l0_k = 96;
          const uint32_t qp_event = qsfa_query_l1_event;
          auto q_l1 = qsfa_l1[qsfa_query_l1_bank * qsfa_bank_l1];
          if (qsfa_qk_panel == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);
            AscendC::Nd2NzParams q_copy{};
            q_copy.ndNum = 1;
            q_copy.nValue = qsfa_heads;
            q_copy.dValue = q_copy.srcDValue = qsfa_combined;
            q_copy.dstNzC0Stride = qsfa_heads;
            q_copy.dstNzNStride = 1;
            AscendC::DataCopy(q_l1, query[qsfa_qk_query_offset], q_copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
          }
          for (uint32_t n0 = 0; n0 < qsfa_qk_tokens; n0 += qsfa_n_panel) {
            const uint32_t remaining = qsfa_qk_tokens - n0;
            const uint32_t n = remaining < qsfa_n_panel ? remaining : qsfa_n_panel;
            const uint32_t n_aligned = (n + 15) / 16 * 16;
            auto accumulator = qsfa_c[(qsfa_c_cursor++ % 2) * qsfa_bank_c];
            const bool keep_stripe = n0 + qsfa_n_panel == qsfa_qk_tokens;
            const uint32_t retained_slot = qsfa_qk_panel % 2;
            for (uint32_t k1 = 0; k1 < 2; ++k1) {
              const uint32_t first_panel_k = keep_stripe ? qsfa_retained_latent_channels : qk_l1_k;
              const uint32_t k_start = k1 == 0 ? 0 : first_panel_k;
              const uint32_t panel_k = k1 == 0 ? first_panel_k : qsfa_combined - first_panel_k;
              const uint32_t latent_k = k1 == 0 ? first_panel_k : qsfa_s2 - first_panel_k;
              const bool keep_latent = keep_stripe && k1 == 0;
              const uint32_t kv_bank = keep_latent ? 2 + retained_slot : qsfa_kv_cursor++ % 2;
              const uint32_t kv_event = qsfa_kv_events[kv_bank];
              const uint32_t kv_offset = keep_latent
                                             ? qsfa_retained_kv_offset + retained_slot * qsfa_retained_kv_elements
                                             : qsfa_work_kv_offset + kv_bank * qsfa_bank_l1;
              auto k_l1 = qsfa_l1[kv_offset];
              const uint64_t k_offset = qsfa_qk_kv_offset + n0 * qsfa_combined + k_start * qsfa_n_panel;
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
              AscendC::DataCopyParams k_copy{};
              k_copy.blockLen = n;
              k_copy.srcStride = qsfa_n_panel - n;
              k_copy.dstStride = n_aligned - n;
              // The native second L1 panel has a latent tail and a RoPE segment.
              k_copy.blockCount = latent_k / qsfa_c0;
              AscendC::DataCopy(k_l1, kv_nz[k_offset], k_copy);
              if (k1 == 1) {
                k_copy.blockCount = 64 / qsfa_c0;
                AscendC::DataCopy(k_l1[latent_k * n_aligned], kv_nz[k_offset + latent_k * qsfa_n_panel], k_copy);
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
              for (uint32_t k0 = 0; k0 < panel_k; k0 += qk_l0_k) {
                const uint32_t ab_bank = qsfa_ab_cursor++ % 2;
                const uint32_t ab_event = qsfa_ab_event_base + ab_bank;
                auto a_l0 = qsfa_a[ab_bank * qsfa_bank_ab];
                auto b_l0 = qsfa_b[ab_bank * qsfa_bank_ab];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ab_event);

                {
                  AscendC::LoadData3DParamsV2<bfloat16_t> load{};
                  load.l1H = (qsfa_heads) / qsfa_c0;
                  load.l1W = qsfa_c0;
                  load.padList[3] = 255;
                  load.mExtension = qsfa_heads;
                  load.kExtension = qk_l0_k;
                  load.strideH = load.strideW = 1;
                  load.filterH = load.filterW = 1;
                  load.dilationFilterH = load.dilationFilterW = 1;
                  load.enTranspose = 0;
                  load.channelSize = qk_l0_k;
                  // AscendC's default config resets both the fmatrix and padding.
                  AscendC::LoadData(a_l0, q_l1[(k_start + k0) * qsfa_heads], load);
                }

                AscendC::LoadData2DParams load_k{};
                load_k.repeatTimes = (n_aligned / qsfa_c0) * (qk_l0_k / qsfa_c0);
                load_k.srcStride = 1;
                AscendC::LoadData(b_l0, k_l1[k0 * n_aligned], load_k);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ab_event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ab_event);
                tile_mmad(accumulator, a_l0, b_l0, qsfa_heads, n_aligned, qk_l0_k, k1 == 0 && k0 == 0,
                          (k1 == 1 && k0 + qk_l0_k == panel_k) ? 0b11 : 0b10);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ab_event);
              }
              if (!keep_latent) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
              }
            }

            AscendC::FixpipeParamsV220 fix{};
            fix.mSize = qsfa_heads;
            fix.nSize = n_aligned;
            fix.srcStride = qsfa_heads;
            fix.dstStride = qsfa_s2;
            fix.ndNum = 1;
            fix.unitFlag = 0b11;
            AscendC::Fixpipe(scores[qsfa_qk_result_offset + n0], accumulator, fix);
          }
          if (qsfa_qk_panel == 3) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);
          }
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(7);
      }
      if (tick > 0) {
        AscendC::CrossCoreWaitFlag(8);
        const int task = tick - 1;
        const int64_t qsfa_pv_kv_offset = (int64_t(core) * 4 + task % 4) * 512 * 576;
        const int64_t qsfa_pv_result_offset = (int64_t(core) * 2 + task % 2) * 128 * 512;
        constexpr uint32_t qsfa_pv_tokens = 512;
        {
          constexpr uint32_t pv_l1_k = 256;
          constexpr uint32_t pv_l0_k = 128;
          // Score/partial GM result slots use the same task parity as the retained K stripes.
          const uint32_t retained_slot = (qsfa_pv_result_offset / (128 * qsfa_s2)) % 2;
          const uint32_t retained_kv_event = qsfa_kv_events[2 + retained_slot];
          const uint32_t retained_token_start = qsfa_s2 - qsfa_n_panel;
          auto retained_kv = qsfa_l1[qsfa_retained_kv_offset + retained_slot * qsfa_retained_kv_elements];
          for (uint32_t n0 = 0; n0 < qsfa_s2; n0 += qsfa_n_panel) {
            const bool reuse_stripe = n0 + qsfa_n_panel <= qsfa_reused_latent_channels;
            auto accumulator = qsfa_c[(qsfa_c_cursor++ % 2) * qsfa_bank_c];
            for (uint32_t k1 = 0; k1 < qsfa_pv_tokens; k1 += pv_l1_k) {
              const uint32_t remaining = qsfa_pv_tokens - k1;
              const uint32_t panel_k = remaining < pv_l1_k ? remaining : pv_l1_k;
              const uint32_t kv_bank = qsfa_kv_cursor++ % 2;
              const uint32_t kv_event = qsfa_kv_events[kv_bank];
              auto v_l1 = qsfa_l1[qsfa_work_kv_offset + kv_bank * qsfa_bank_l1];
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
              for (uint32_t k0 = 0; k0 < panel_k; k0 += pv_l0_k) {
                const uint32_t k_left = panel_k - k0;
                const uint32_t k = k_left < pv_l0_k ? k_left : pv_l0_k;
                const uint32_t k_aligned = (k + 15) / 16 * 16;
                if (!reuse_stripe || k1 + k0 != retained_token_start) {
                  AscendC::DataCopyParams v_copy{};
                  v_copy.blockCount = qsfa_n_panel / qsfa_c0;
                  v_copy.blockLen = k;
                  v_copy.srcStride = qsfa_n_panel - k;
                  v_copy.dstStride = k_aligned - k;
                  const uint64_t v_offset = qsfa_pv_kv_offset + (k1 + k0) * qsfa_combined + n0 * qsfa_n_panel;
                  AscendC::DataCopy(v_l1[k0 * qsfa_n_panel], kv_nz[v_offset], v_copy);
                }
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kv_event);
              const uint32_t qp_bank = k1 / pv_l1_k;
              const uint32_t qp_event = qsfa_qp_events[qp_bank];
              auto p_l1 = qsfa_l1[qsfa_prob_offset + qp_bank * qsfa_prob_panel_elements];
              if (n0 == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);
                AscendC::Nd2NzParams p_copy{};
                p_copy.ndNum = 1;
                p_copy.nValue = qsfa_heads;
                p_copy.dValue = panel_k;
                p_copy.srcDValue = qsfa_s2;
                p_copy.dstNzC0Stride = qsfa_heads;
                p_copy.dstNzNStride = 1;
                AscendC::DataCopy(p_l1, probs[qsfa_pv_result_offset + k1], p_copy);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(qp_event);
              }
              for (uint32_t k0 = 0; k0 < panel_k; k0 += pv_l0_k) {
                const uint32_t k_left = panel_k - k0;
                const uint32_t k = k_left < pv_l0_k ? k_left : pv_l0_k;
                const uint32_t k_aligned = (k + 15) / 16 * 16;

                const uint32_t ab_bank = qsfa_ab_cursor++ % 2;
                const uint32_t ab_event = qsfa_ab_event_base + ab_bank;
                auto a_l0 = qsfa_a[ab_bank * qsfa_bank_ab];
                auto b_l0 = qsfa_b[ab_bank * qsfa_bank_ab];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ab_event);

                {
                  AscendC::LoadData3DParamsV2<bfloat16_t> load{};
                  load.l1H = (k_aligned) / qsfa_c0;
                  load.l1W = qsfa_c0;
                  load.padList[3] = 255;
                  load.mExtension = k_aligned;
                  load.kExtension = qsfa_n_panel;
                  load.strideH = load.strideW = 1;
                  load.filterH = load.filterW = 1;
                  load.dilationFilterH = load.dilationFilterW = 1;
                  load.enTranspose = 1;
                  load.channelSize = qsfa_n_panel;
                  // AscendC's default config resets both the fmatrix and padding.
                  // QK and PV both address the compact 128-token NZ C0 groups.
                  auto value_l1 = reuse_stripe && k1 + k0 == retained_token_start ? retained_kv[n0 * qsfa_n_panel]
                                                                                  : v_l1[k0 * qsfa_n_panel];
                  AscendC::LoadData(b_l0, value_l1, load);
                }

                {
                  AscendC::LoadData3DParamsV2<bfloat16_t> load{};
                  load.l1H = (qsfa_heads) / qsfa_c0;
                  load.l1W = qsfa_c0;
                  load.padList[3] = 255;
                  load.mExtension = qsfa_heads;
                  load.kExtension = k_aligned;
                  load.strideH = load.strideW = 1;
                  load.filterH = load.filterW = 1;
                  load.dilationFilterH = load.dilationFilterW = 1;
                  load.enTranspose = 0;
                  load.channelSize = k_aligned;
                  // AscendC's default config resets both the fmatrix and padding.
                  AscendC::LoadData(a_l0, p_l1[k0 * qsfa_heads], load);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ab_event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ab_event);
                tile_mmad(accumulator, a_l0, b_l0, qsfa_heads, qsfa_n_panel, k, k1 == 0 && k0 == 0,
                          (k1 + k0 + k == qsfa_pv_tokens) ? 0b11 : 0b10);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ab_event);
              }
              if (n0 + qsfa_n_panel == qsfa_s2) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qp_event);
              }
              if (k1 + panel_k == qsfa_pv_tokens) {
                AscendC::FixpipeParamsV220 fix{};
                fix.mSize = qsfa_heads;
                fix.nSize = qsfa_n_panel;
                fix.srcStride = qsfa_heads;
                fix.dstStride = qsfa_s2;
                fix.ndNum = 1;
                fix.unitFlag = 0b11;
                AscendC::Fixpipe(partial[qsfa_pv_result_offset + n0], accumulator, fix);
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kv_event);
            }
            if (n0 + qsfa_n_panel == qsfa_retained_latent_channels) {
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(retained_kv_event);
            }
          }
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(9);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(3);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(qsfa_ab_event_base + 1);
    for (uint32_t qsfa_i = 0; qsfa_i < 2; ++qsfa_i) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_qp_events[qsfa_i]);
    }
    for (uint32_t qsfa_i = 0; qsfa_i < 4; ++qsfa_i) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_kv_events[qsfa_i]);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qsfa_query_l1_event);
  }

  if ASCEND_IS_AIV {
    const int block_index = AscendC::GetBlockIdx();
    const int core = block_index / 2;
    const int lane = block_index % 2;
    const int row_begin = core * rows_per_core;
    if (row_begin >= rows) {
      return;
    }
    const int remaining = rows - row_begin;
    const int task_count = (remaining < rows_per_core ? remaining : rows_per_core) * 4;
    auto packed = MakeGlobal<int8_t>(packed_addr);
    auto indices = MakeGlobal<int32_t>(indices_addr);
    auto page_table = MakeGlobal<int32_t>(page_table_addr);
    auto query_ends = MakeGlobal<int32_t>(query_ends_addr);
    auto kv_lengths = MakeGlobal<int32_t>(kv_lengths_addr);
    auto output = MakeGlobal<bfloat16_t>(output_addr);
    auto& decoded = kv_nz;
    auto valid_gm = MakeGlobal<int32_t>(valid_gm_addr);
    [[maybe_unused]] auto numerator = MakeGlobal<float>(numerator_addr);
    AscendC::TPipe pipe;
    Catlass::Arch::LocalTensorBuffer<A3, AscendC::TPosition::VECCALC> vec_ub;
    pipe.Destroy();
    auto vec_in1 = vec_ub.GetBufferByByte<float>(0);
    auto vec_in2 = vec_ub.GetBufferByByte<float>(65536);
    auto vec_out1 = vec_ub.GetBufferByByte<float>(98304);
    auto vec_out2 = vec_ub.GetBufferByByte<float>(131072);
    auto vec_tmp1 = vec_ub.GetBufferByByte<float>(135168);
    auto vec_tmp2 = vec_ub.GetBufferByByte<float>(167936);
    auto gather_valid = vec_ub.GetBufferByByte<int32_t>(176128);
    auto vec_max = vec_ub.GetBufferByByte<float>(178176);
    auto vec_sum = vec_ub.GetBufferByByte<float>(179200);
    auto vec_exp = vec_ub.GetBufferByByte<float>(180224);
    auto vec_max_default = vec_ub.GetBufferByByte<float>(181248);
    auto vec_sum_default = vec_ub.GetBufferByByte<float>(181760);
    int qsfa_next_vec1_task = 0;
    int qsfa_next_vec2_task = 0;
    int qsfa_vec_pingpong = 0;
    int qsfa_gather_last_request = -1;
    int qsfa_gather_causal_base = 0;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::Duplicate(vec_max_default, -2.0e38f, 128);
    AscendC::Duplicate(vec_sum_default, 0.0f, 128);
    const int qsfa_vec_row_begin = row_begin;
    for (int tick = 0; tick < task_count + 2; ++tick) {
      if (tick < task_count) {
        AscendC::CrossCoreWaitFlag(3);
        const int qsfa_gather_core = core;
        const int qsfa_gather_lane = lane;
        const int qsfa_gather_row = row_begin + tick / 4;
        const int qsfa_gather_tile = tick % 4;
        const int qsfa_gather_slot = qsfa_gather_tile;
        constexpr int qsfa_gather_tokens = 512;
        {
          using namespace AscendC;
          constexpr int topk = 2048;
          constexpr int tokens_per_page = 128;
          constexpr int pages_per_request = 1024;
          constexpr int queries_per_request = 6;
          constexpr int tokens_per_tile = 512;
          constexpr int rows_per_fetch = 32;
          constexpr int quant_channels = 512;
          constexpr int rope_channels = 64;
          constexpr int packed_bytes = 656;
          constexpr int packed_stride = 672;
          constexpr int nz_lanes = 16;
          constexpr int metadata_words = 128;
          constexpr int input_ping_pong_bytes = 32768;
          constexpr int output_nope_event = 4;
          constexpr int output_rope_event = 5;

          const int core_id = qsfa_gather_core;
          const int vector_id = qsfa_gather_lane;
          const int query_row = qsfa_gather_row;
          const int tile_index = qsfa_gather_tile;
          const int merge_slot = qsfa_gather_slot;
          const int tile_tokens = qsfa_gather_tokens;
          const int request = query_row / queries_per_request;
          if (request != qsfa_gather_last_request) {
            qsfa_gather_causal_base = kv_lengths.GetValue(request) - query_ends.GetValue(request);
            qsfa_gather_last_request = request;
          }
          const int causal_limit = qsfa_gather_causal_base + query_row + 1;
          const int pair_count = (tile_tokens + 1) / 2;
          const int first_half = (pair_count + 1) / 2 * 2;
          const int start = vector_id == 0 ? 0 : first_half;
          const int stop = vector_id == 0 && first_half < tile_tokens ? first_half : tile_tokens;
          const int64_t topk_base = static_cast<int64_t>(query_row) * topk + tile_index * tokens_per_tile;
          const int64_t page_base = request * pages_per_request;
          const int64_t decoded_base =
              static_cast<int64_t>(core_id * 4 + merge_slot) * tokens_per_tile * (quant_channels + rope_channels);

          auto raw = vec_in1.ReinterpretCast<int8_t>();
          auto fp16 = vec_tmp1.ReinterpretCast<half>();
          auto nd_bf16 = vec_tmp1.ReinterpretCast<bfloat16_t>();
          auto nope_nz = vec_out1.ReinterpretCast<bfloat16_t>();
          auto rope_nz = vec_out2.ReinterpretCast<bfloat16_t>();
          auto scales = vec_tmp2;
          auto work = vec_in2;
          DataCopyPadExtParams<int8_t> pad{true, 0, 16, 0};
          const DataCopyExtParams one_row{1, packed_bytes, 0, 0, 0};
          SetFlag<HardEvent::MTE3_MTE2>(0);
          SetFlag<HardEvent::MTE3_MTE2>(1);

          int fetch = 0;
          int loaded = 0;
          int written = 0;
          bool needs_input = true;
          for (int token = start; token < stop; token += 2) {
            const int buffer = fetch & 1;
            if (needs_input) {
              WaitFlag<HardEvent::MTE3_MTE2>(buffer);
              needs_input = false;
            }
            auto source = raw[buffer * input_ping_pong_bytes];
            const int first = indices.GetValue(topk_base + token);
            if (first >= 0) {
              const int second = token + 1 < stop ? indices.GetValue(topk_base + token + 1) : -1;
              int64_t first_row = -1;
              int64_t second_row = -1;
              if (first < causal_limit) {
                const int page = page_table.GetValue(page_base + first / tokens_per_page);
                first_row = static_cast<int64_t>(page) * tokens_per_page + first % tokens_per_page;
              }
              if (second >= 0 && second < causal_limit) {
                const int page = page_table.GetValue(page_base + second / tokens_per_page);
                second_row = static_cast<int64_t>(page) * tokens_per_page + second % tokens_per_page;
              }
              if (first_row >= 0 || second_row >= 0) {
                const int64_t low = first_row < second_row ? first_row : second_row;
                const int64_t high = first_row < second_row ? second_row : first_row;
                const int64_t stride = (high - low - 1) * packed_bytes;
                if (low >= 0 && stride >= 0 && stride < 2147483647LL && first + 1 < causal_limit &&
                    second + 1 < causal_limit) {
                  const DataCopyExtParams pair{2, packed_bytes, static_cast<uint32_t>(stride), 0, 0};
                  DataCopyPad(source[(loaded - written) * packed_stride], packed[low * packed_bytes], pair, pad);
                  loaded += 2;
                } else {
                  if (first_row >= 0) {
                    DataCopyPad(source[(loaded - written) * packed_stride], packed[first_row * packed_bytes], one_row,
                                pad);
                    ++loaded;
                  }
                  if (second_row >= 0) {
                    DataCopyPad(source[(loaded - written) * packed_stride], packed[second_row * packed_bytes], one_row,
                                pad);
                    ++loaded;
                  }
                }
              }
            }
            const int rows = loaded - written;
            if (first < 0 || rows + 2 > rows_per_fetch || token + 2 >= stop) {
              if (rows > 0) {
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                const auto no_round = RoundMode::CAST_NONE;
                if (rows == 1) {
                  Cast(fp16, source, no_round, static_cast<uint64_t>(128), 4, {1, 1, 8, 4});
                } else {
                  for (int channel = 0; channel < quant_channels; channel += 128) {
                    Cast(fp16[channel], source[channel], no_round, static_cast<uint64_t>(128),
                         static_cast<uint8_t>(rows), {1, 1, 32, 21});
                  }
                }
                PipeBarrier<PIPE_V>();
                auto scale_input = source[640].ReinterpretCast<float>();
                if (rows == 1) {
                  Brcb(scales, scale_input, 1, {1, 4});
                } else {
                  auto raw_scales = scales[1024];
                  const DataCopyParams scale_copy{static_cast<uint16_t>(rows), 1, 20, 0};
                  DataCopy(raw_scales, scale_input, scale_copy);
                  PipeBarrier<PIPE_V>();
                  Brcb(scales, raw_scales, rows, {1, 4});
                }
                PipeBarrier<PIPE_V>();
                for (int row_start = 0; row_start < rows; row_start += 16) {
                  const int chunk_rows = rows - row_start < 16 ? rows - row_start : 16;
                  const uint32_t elements = chunk_rows * quant_channels;
                  Cast(work, fp16[row_start * quant_channels], no_round, elements);
                  PipeBarrier<PIPE_V>();
                  for (int sub = 0; sub < 128; sub += 64) {
                    Mul(work[sub], work[sub], scales[row_start * 32], 64, 4 * chunk_rows, {1, 1, 0, 16, 16, 1});
                  }
                  PipeBarrier<PIPE_V>();
                  Cast(nd_bf16[row_start * quant_channels], work, RoundMode::CAST_RINT, elements);
                  PipeBarrier<PIPE_V>();
                }

                WaitFlag<HardEvent::MTE3_V>(output_nope_event);
                for (int row_start = 0; row_start < rows; row_start += 8) {
                  const int chunk_rows = rows - row_start < 8 ? rows - row_start : 8;
                  Copy(nope_nz[row_start * nz_lanes], nd_bf16[row_start * quant_channels],
                       static_cast<uint64_t>(chunk_rows * nz_lanes), 32, {1, 32, static_cast<uint16_t>(rows), 1});
                }
                SetFlag<HardEvent::V_MTE3>(output_nope_event);
                WaitFlag<HardEvent::V_MTE3>(output_nope_event);
                constexpr int panel_tokens = 128;
                constexpr int panel_elements = panel_tokens * (quant_channels + rope_channels);
                const int token = start + written;
                const int panel_row = token % panel_tokens;
                const int available = panel_tokens - panel_row;
                const int first_rows = rows < available ? rows : available;
                const int next_rows = rows - first_rows;
                const int64_t panel_base = decoded_base + (token / panel_tokens) * panel_elements;
                const int64_t first_base = panel_base + panel_row * nz_lanes;
                const int64_t next_base = panel_base + panel_elements;
                // UB groups retain their original `rows` pitch when the GM copy splits.
                DataCopyExtParams first_copy{32, static_cast<uint32_t>(first_rows * 32),
                                             static_cast<uint32_t>(next_rows),
                                             static_cast<uint32_t>((panel_tokens - first_rows) * 32), 0};
                DataCopyExtParams next_copy{32, static_cast<uint32_t>(next_rows * 32),
                                            static_cast<uint32_t>(first_rows),
                                            static_cast<uint32_t>((panel_tokens - next_rows) * 32), 0};
                DataCopyPad(kv_nz[first_base], nope_nz, first_copy);
                if (next_rows) {
                  DataCopyPad(kv_nz[next_base], nope_nz[first_rows * nz_lanes], next_copy);
                }
                SetFlag<HardEvent::MTE3_V>(output_nope_event);

                WaitFlag<HardEvent::MTE3_V>(output_rope_event);
                Copy(rope_nz, source[quant_channels].ReinterpretCast<bfloat16_t>(), rope_channels,
                     static_cast<uint8_t>(rows), {static_cast<uint16_t>(rows), 1, 1, 21});
                SetFlag<HardEvent::V_MTE3>(output_rope_event);
                WaitFlag<HardEvent::V_MTE3>(output_rope_event);
                constexpr int rope_offset = panel_tokens * quant_channels;
                first_copy.blockCount = rope_channels / nz_lanes;
                DataCopyPad(kv_nz[first_base + rope_offset], rope_nz, first_copy);
                if (next_rows) {
                  next_copy.blockCount = rope_channels / nz_lanes;
                  DataCopyPad(kv_nz[next_base + rope_offset], rope_nz[first_rows * nz_lanes], next_copy);
                }
                SetFlag<HardEvent::MTE3_V>(output_rope_event);
                written = loaded;
              }
              SetFlag<HardEvent::MTE3_MTE2>(buffer);
              ++fetch;
              needs_input = true;
              if (first < 0) {
                break;
              }
            }
          }
          if (start + loaded < stop) {
            const int buffer = fetch & 1;
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(buffer);
            auto zero = raw.ReinterpretCast<bfloat16_t>();
            Duplicate(zero, static_cast<bfloat16_t>(0.0f), quant_channels);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            constexpr int panel_tokens = 128;
            constexpr int panel_elements = panel_tokens * (quant_channels + rope_channels);
            const DataCopyExtParams latent_zero{32, 32, 0, (panel_tokens - 1) * 32, 0};
            const DataCopyExtParams rope_zero{rope_channels / nz_lanes, 32, 0, (panel_tokens - 1) * 32, 0};
            for (int token = start + loaded; token < stop; ++token) {
              const int64_t base =
                  decoded_base + (token / panel_tokens) * panel_elements + (token % panel_tokens) * nz_lanes;
              DataCopyPad(kv_nz[base], zero, latent_zero);
              DataCopyPad(kv_nz[base + panel_tokens * quant_channels], zero, rope_zero);
            }
            SetFlag<HardEvent::MTE3_MTE2>(buffer);
          }
          WaitFlag<HardEvent::MTE3_MTE2>(0);
          WaitFlag<HardEvent::MTE3_MTE2>(1);

          gather_valid.SetValue(merge_slot, loaded);
          SetFlag<HardEvent::S_MTE3>(1);
          WaitFlag<HardEvent::S_MTE3>(1);
          const DataCopyExtParams valid_copy{1, metadata_words * sizeof(int32_t), 0, 0, 0};
          const int64_t valid_base = (static_cast<int64_t>(core_id * 4 + merge_slot) * 2 + vector_id) * metadata_words;
          DataCopyPad(valid_gm[valid_base], gather_valid, valid_copy);
        }
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(6);
      }
      if (tick > 0 && tick < task_count + 1) {
        AscendC::CrossCoreWaitFlag(7);
        {
          const int task = qsfa_next_vec1_task++;
          const int core = AscendC::GetBlockIdx() / 2;
          const int lane = AscendC::GetBlockIdx() % 2;
          const int stats_out = (task % 2) * 128;
          const int stats_in = ((task + 1) % 2) * 128;
          const int gm_start = ((core * 2 + task % 2) * 128 + lane * 32) * 512;
          const AscendC::DataCopyExtParams valid_copy{1, 256 * sizeof(int32_t), 0, 0, 0};
          const AscendC::DataCopyPadExtParams<int32_t> valid_pad{false, 0, 0, 0};
          const int valid_start = (core * 4 + task % 4) * 256;
          AscendC::DataCopyPad(gather_valid[128], valid_gm[valid_start], valid_copy, valid_pad);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);

          for (int head = 0; head < 32; head += 16) {
            const int ping = qsfa_vec_pingpong;
            auto qk = vec_in1[ping * 8192];
            auto p = vec_out1.ReinterpretCast<bfloat16_t>();
            auto scratch = vec_tmp1.ReinterpretCast<uint8_t>();
            const int gm_offset = gm_start + head * 512;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + ping);
            AscendC::DataCopy(qk, scores[gm_offset], 8192);
            if (head == 0) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::Muls(qk, qk, softmax_scale, 8192);
            for (int half = 0; half < 2; ++half) {
              const int count = gather_valid.GetValue(128 + half * 128 + task % 4);
              if (count < 256) {
                const int start = half * 256 + count;
                const int stop = (half + 1) * 256;
                const int aligned = (start + 7) / 8 * 8;
                if (start < aligned) {
                  const uint64_t one_block = (0xffu << (start % 8)) & 0xffu;
                  uint64_t mask[1] = {one_block * 0x0101010101010101ULL};
                  for (int r = 0; r < 16; r += 8) {
                    AscendC::Duplicate(qk[r * 512 + start / 8 * 8], -2.0e38f, mask, 1, 64, 0);
                  }
                }
                if (aligned < stop) {
                  for (int r = 0; r < 16; ++r) {
                    AscendC::Duplicate(qk[r * 512 + aligned], -2.0e38f, stop - aligned);
                  }
                }
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            auto old_sum = task % 4 == 0 ? vec_sum_default : vec_sum[stats_in + head];
            auto old_max = task % 4 == 0 ? vec_max_default : vec_max[stats_in + head];
            const AscendC::SoftMaxShapeInfo shape{16, 512, 16, 512};
            const auto tiling =
                AscendC::SoftMaxFlashV2TilingFunc(shape, sizeof(float), sizeof(float), 32768, true, false);
            static constexpr AscendC::SoftmaxConfig cfg{false, 0, 0, AscendC::SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
            AscendC::SoftmaxFlashV2<float, true, true, false, false, cfg>(
                qk, vec_sum[stats_out + head], vec_max[stats_out + head], qk, vec_exp[stats_out + head], old_sum,
                old_max, scratch, tiling, shape);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
            AscendC::Cast(p, qk, AscendC::RoundMode::CAST_ROUND, 8192);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + ping);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
            AscendC::DataCopy(probs[gm_offset], p, 8192);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
            qsfa_vec_pingpong ^= 1;
          }
        }
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
      }
      if (tick > 1) {
        AscendC::CrossCoreWaitFlag(9);
        {
          const int task = qsfa_next_vec2_task++;
          const int core = AscendC::GetBlockIdx() / 2;
          const int lane = AscendC::GetBlockIdx() % 2;
          const int row = qsfa_vec_row_begin + task / 4;
          const int stats = (task % 2) * 128;
          const int gm_now = ((core * 2 + task % 2) * 128 + lane * 32) * 512;
          const int gm_prev = ((core * 2 + (task + 1) % 2) * 128 + lane * 32) * 512;
          const AscendC::BinaryRepeatParams by_head{1, 1, 0, 64, 64, 1};

          for (int head = 0; head < 32; head += 16) {
            const int ping = qsfa_vec_pingpong;
            const int offset = head * 512;
            auto pv = vec_in1[ping * 8192];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + ping);
            AscendC::DataCopy(pv, partial[gm_now + offset], 8192);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::DataCopy(vec_tmp1, pv, 8192);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + ping);

            if (task % 4 != 0) {
              AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
              AscendC::DataCopy(vec_in2, numerator[gm_prev + offset], 8192);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
              AscendC::Brcb(vec_tmp2, vec_exp[stats + head], 4, {1, 8});
              AscendC::PipeBarrier<PIPE_V>();
              for (int col = 0; col < 512; col += 64) {
                AscendC::Mul(vec_in2[col], vec_in2[col], vec_tmp2, 64, 16, by_head);
              }
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(vec_tmp1, vec_tmp1, vec_in2, 8192);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
            }

            if (task % 4 == 3) {
              AscendC::Brcb(vec_tmp2, vec_sum[stats + head], 4, {1, 8});
              AscendC::PipeBarrier<PIPE_V>();
              for (int col = 0; col < 512; col += 64) {
                AscendC::Div(vec_tmp1[col], vec_tmp1[col], vec_tmp2, 64, 16, by_head);
              }
              AscendC::PipeBarrier<PIPE_V>();
              auto result = vec_out1.ReinterpretCast<bfloat16_t>();
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
              AscendC::Cast(result, vec_tmp1, AscendC::RoundMode::CAST_RINT, 8192);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
              const int out_offset = (row * 64 + lane * 32 + head) * 512;
              const AscendC::DataCopyExtParams output_copy{16, 1024, 0, 0, 0};
              AscendC::DataCopyPad(output[out_offset], result, output_copy);
            } else {
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
              AscendC::DataCopy(vec_out1, vec_tmp1, 8192);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
              AscendC::DataCopy(numerator[gm_now + offset], vec_out1, 8192);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
            qsfa_vec_pingpong ^= 1;
          }
        }
      }
    }
    for (int credit = 0; credit < 4; ++credit) {
      AscendC::CrossCoreWaitFlag(3);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
  }
}

extern "C" int qsfa_catlass_launch(void* stream, GM_ADDR query, GM_ADDR packed, GM_ADDR indices, GM_ADDR page_table,
                                   GM_ADDR query_ends, GM_ADDR kv_lengths, GM_ADDR output, GM_ADDR kv_nz,
                                   GM_ADDR scores, GM_ADDR probs, GM_ADDR partial, GM_ADDR numerator, GM_ADDR valid_gm,
                                   int rows, float scale) {
  uint64_t ffts_addr = 0;
  uint32_t ffts_length = 0;
  const int status = rtGetC2cCtrlAddr(&ffts_addr, &ffts_length);
  if (status != 0) {
    return status;
  }
  qsfa_catlass_resident_kv_wide_384<<<cores, nullptr, stream>>>(query, packed, indices, page_table, query_ends,
                                                                kv_lengths, output, kv_nz, scores, probs, partial,
                                                                numerator, valid_gm, rows, scale, ffts_addr);
  return 0;
}
