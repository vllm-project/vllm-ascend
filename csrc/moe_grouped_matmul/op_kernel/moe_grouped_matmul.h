/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
using namespace AscendC;


constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                         0, 0, 0, 0, true};
struct GMMConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint64_t wBaseOffset = 0;
    uint64_t nAxisBaseOffset = 0;
    uint64_t mAxisBaseOffset = 0;
    uint64_t xBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wOutOffset = 0;
};


template <typename T, typename T2, CubeFormat formatWeight, bool transWeight>
class KernelMoeGMMNoQuant {

protected:
  using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
  using weightType = MatmulType<AscendC::TPosition::GM, formatWeight, T, transWeight>;
  using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
  using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
  using mmT = matmul::MatmulImpl<xType, weightType, yType, biasType, matmulCFGUnitFlag>;
  mmT mm;

  MoeGroupedMatmulTilingData tiling_;
  AscendC::TPipe *pipe_ = nullptr;

  GlobalTensor<T> x_gm_;
  GlobalTensor<T> weight_gm_;
  GlobalTensor<T> y_gm_;
  GlobalTensor<T2> group_list_gm_;
  ListTensorDesc x_list_;
  ListTensorDesc weight_list_;
  ListTensorDesc y_list_;

  uint32_t core_idx;
  uint32_t used_core_num;
  constexpr static bool transposeW = transWeight;
  constexpr static uint32_t UB_BLOCK_UNIT_SIZE = 32;

public:
  __aicore__ inline KernelMoeGMMNoQuant(AscendC::TPipe *pipe) {pipe_ = pipe;}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y, const MoeGroupedMatmulTilingData *tiling) {
      core_idx = GetBlockIdx();
      tiling_ = *tiling;
      used_core_num = GetBlockNum();
      group_list_gm_.SetGlobalBuffer((__gm__ T2*)group_list);
      x_list_.Init((__gm__ void*)x);
      weight_list_.Init((__gm__ void*)weight);
      y_list_.Init((__gm__ void*)y);
      GM_ADDR x_first_addr = (__gm__ uint8_t*)x_list_.GetDataPtr<__gm__ uint8_t>(0);
      GM_ADDR weight_first_addr = (__gm__ uint8_t*)weight_list_.GetDataPtr<__gm__ uint8_t>(0);
      GM_ADDR y_first_addr = (__gm__ uint8_t*)y_list_.GetDataPtr<__gm__ uint8_t>(0);
      x_gm_.SetGlobalBuffer((__gm__ T*)x_first_addr);
      weight_gm_.SetGlobalBuffer((__gm__ T*)weight_first_addr);
      y_gm_.SetGlobalBuffer((__gm__ T*)y_first_addr);

      mm.Init(&tiling_.mm_tiling, pipe_);
  }

  __aicore__ inline void Process() {

      uint32_t group_list_inner_shape = 2u;
      uint32_t group_list_shape_size = tiling_.group_num * group_list_inner_shape;
      GMMConfig mn_config;

      for (uint32_t loop = 0, count = 0; loop < group_list_shape_size; loop += group_list_inner_shape) {
        int32_t split_value = static_cast<int32_t>(group_list_gm_.GetValue(loop + 1));
        if (split_value <= 0) {
          break;
        }
        uint32_t group_idx = static_cast<int32_t>(group_list_gm_.GetValue(loop));
        mn_config.mAxisBaseOffset += mn_config.m;
        mn_config.xBaseOffset += mn_config.m * mn_config.k;
        mn_config.yBaseOffset += mn_config.m * mn_config.n;
        this->SetMNConfig(split_value, mn_config);
        mn_config.nAxisBaseOffset = group_idx * mn_config.n;
        if constexpr (formatWeight == CubeFormat::NZ) {
          mn_config.wBaseOffset = AlignUp(mn_config.k, 16) * AlignUp(mn_config.nAxisBaseOffset, 16);
        } else {
          mn_config.wBaseOffset = mn_config.k * mn_config.nAxisBaseOffset;
        }
        mn_config.blockDimM = Ceil(mn_config.m, mn_config.singleM);
        mn_config.blockDimN = Ceil(mn_config.n, mn_config.singleN);
        uint32_t cur_count = count + mn_config.blockDimM * mn_config.blockDimN;
        uint32_t cur_block = this->core_idx >= count ? this->core_idx : this->core_idx + used_core_num;
        while (cur_block < cur_count) {
            mn_config.mIdx = (cur_block - count) / mn_config.blockDimN;
            mn_config.nIdx = (cur_block - count) % mn_config.blockDimN;
            this->MMCompute(group_idx, mn_config, this->core_idx);
            cur_block += used_core_num;
        }
        count = cur_count % used_core_num;
      }
  }

protected:
  __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t base) {
      return (a + base - 1) / base * base;
  }

  __aicore__ inline uint32_t Ceil(uint32_t a, uint32_t base) {
      if (base == 0) {
        return a;
      }
      return (a + base - 1) / base;
  }

  __aicore__ inline void SetMNConfig(const int32_t split_value, GMMConfig & mn_config) {
      mn_config.m = split_value;
      mn_config.k = tiling_.k;
      mn_config.n = tiling_.n;
      mn_config.baseM = tiling_.single_m;
      mn_config.baseN = tiling_.single_n;
      mn_config.singleM = mn_config.baseM;
      mn_config.singleN = mn_config.baseN;
  }

  __aicore__ inline void MMCompute(uint32_t group_idx, GMMConfig& mn_config, uint32_t core_idx) {
      uint32_t tail_n = mn_config.nIdx * mn_config.singleN;
      uint32_t cur_single_n = mn_config.nIdx < mn_config.blockDimN - 1 ? mn_config.singleN : mn_config.n - tail_n;
      uint32_t cur_single_m = mn_config.mIdx < mn_config.blockDimM - 1 ? mn_config.singleM
                                                                    : mn_config.m - mn_config.mIdx * mn_config.singleM;
      uint64_t x_offset = mn_config.mIdx * mn_config.singleM * mn_config.k;
      uint64_t out_offset = mn_config.mIdx * mn_config.singleM * mn_config.n + tail_n;
      GlobalTensor<T> weight_gm_local = GetGlobalBufferW(group_idx, tail_n, mn_config);

      mm.SetOrgShape(mn_config.m, mn_config.n, mn_config.k);
      mm.SetSingleShape(cur_single_m, cur_single_n, mn_config.k);
      mm.SetTensorA(x_gm_[mn_config.xBaseOffset + x_offset], false);
      mm.SetTensorB(weight_gm_local, transposeW);
      mm.template IterateAll<false>(y_gm_[mn_config.yBaseOffset + out_offset], 0);
  }

  __aicore__ inline GlobalTensor<T> GetGlobalBufferW(uint32_t group_idx, uint32_t tail_n, GMMConfig& mn_config) {
      uint64_t w_offset = SetWOffset(tail_n, mn_config.k);
      GlobalTensor<T> weight_gm_local;
      weight_gm_local = weight_gm_[mn_config.wBaseOffset + w_offset];
      if (mn_config.blockDimM == 1) {
        weight_gm_local.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
      }
      return weight_gm_local;
  }

  __aicore__ inline uint64_t SetWOffset(uint32_t tail_n, uint32_t k) {
    uint64_t w_offset = 0;
    if constexpr (formatWeight == CubeFormat::NZ && transposeW) {
        w_offset = tail_n * (UB_BLOCK_UNIT_SIZE / sizeof(T));  // 32: quant is 32, float16 is 16
    } else if constexpr (formatWeight == CubeFormat::NZ) {
        w_offset = tail_n * AlignUp(k, 16);  // 16: nz format last two dim size
    } else if constexpr (transposeW) {
        w_offset = tail_n * k;
    } else {
        w_offset = tail_n;
    }
    return w_offset;
  }
};

