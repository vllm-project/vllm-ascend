/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

/*!
 * \file op_host_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_UTIL_H_

#include <memory>
#include <utility>
#include <type_traits>

namespace ops {

/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if <std::is_signed<T>::value, T>::type CeilDiv(T x, T y) {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
  }

  return x;
}

/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if <std::is_unsigned<T>::value, T>::type CeilDiv(T x, T y) {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0) ? (quotient + 1) : quotient;
  }

  return x;
}


/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type FloorDiv(T x, T y) {
  return y == 0 ? x : x / y;
}

/**
 * if align is 0, return 0
 */
template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type CeilAlign(T x, T align) {
  return CeilDiv(x, align) * align;
}

/**
 * if align is 0, return 0
 */
template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type FloorAlign(T x, T align) {
  return align == 0 ? 0 : x / align * align;
}

}  // namespace ops


namespace optiling {

enum CubeTilingType {
  CUBE_DYNAMIC_SHAPE_TILING,
  CUBE_DEFAULT_TILING,
  CUBE_BINARY_TILING,
};

constexpr uint64_t kInvalidTilingId = std::numeric_limits<uint64_t>::max();

class CubeCompileInfo {
 public:
  bool correct_range_flag = false;
  CubeTilingType tiling_type = CUBE_DYNAMIC_SHAPE_TILING;
  uint64_t default_tiling_id = kInvalidTilingId;
  std::vector<int64_t> default_range;
  std::vector<std::vector<int64_t>> repo_seeds;
  std::vector<std::vector<int64_t>> repo_range;
  std::vector<std::vector<int64_t>> cost_range;
  std::vector<std::vector<int64_t>> batch_range;  // for dynamic batch
  std::vector<uint64_t> repo_tiling_ids;
  std::vector<uint64_t> cost_tiling_ids;
  std::vector<uint64_t> batch_tiling_ids;  // for dynamic batch
  std::map<uint64_t, uint32_t> block_dim;
  std::string soc_version = "";

  uint32_t core_num = 0;
  uint64_t ub_size = 0;
  uint64_t l1_size = 0;
  uint64_t l2_size = 0;
  uint64_t l0a_size = 0;
  uint64_t l0b_size = 0;
  uint64_t l0c_size = 0;
  uint64_t bt_size = 0;
  int32_t cube_freq = 0;
  bool load3d_constraints = true;
  bool intrinsic_data_move_l12ub = true;
  bool intrinsic_matmul_ub_to_ub = false;
  bool intrinsic_conv_ub_to_ub = false;
  bool intrinsic_data_move_l0c2ub = true;
  bool intrinsic_fix_pipe_l0c2out = false;
  bool intrinsic_fix_pipe_l0c2ub = false;
  bool intrinsic_data_move_out2l1_nd2nz = false;
  bool intrinsic_data_move_l12bt_bf16 = false;
};

struct BatchmatmulCompileParas {
  bool binary_mode_flag = false;
  bool bias_flag = false;
  bool at_l1_flag = true;
  bool split_k_flag = false;
  bool pattern_flag = false;
  bool zero_flag = false;
  bool sparse_4to2_flag = false;
  bool binary_constant_flag = false;
  bool vector_pre_conv_mode = false;
  float fused_double_operand_num = 0;
  float aub_double_num = 0;
  float bub_double_num = 0;
  int64_t quant_scale = 0;
  int64_t eltwise_src = 0;
  int8_t enable_pad = 0;
  bool enable_nz_fusion = false;
  bool enable_rt_bank_cache = false;
};

struct Ub2UbBatchmatmulCompileParas {
  int64_t block_m0 = 1;
  int64_t block_n0 = 16;
  int64_t block_a_k0 = 16;
  int64_t block_b_k0 = 16;
  bool is_batch_matmul = false;
  bool bm_fusion_flag = false;

  std::string pre_conv = "None";
  std::string pre_activation = "None";
  std::string post_anti_quant = "None";
  std::string post_eltwise = "None";
  std::string post_activation = "None";
  std::string post_quant = "None";
  std::string post_transform = "None";
};

enum DynamicMode {
  DYNAMIC_MKN,
  DYNAMIC_MKNB,
  WEIGHT_QUANT_BMM
};

class HammingDistTopKCompileInfo : public CubeCompileInfo {
 public:
  // 构造函数：必须和类名 HammingDistTopKCompileInfo 一致
  HammingDistTopKCompileInfo() = default;
  // 析构函数：必须和类名一致，前面加 ~ 波浪号
  ~HammingDistTopKCompileInfo() = default;

  bool trans_a = false;
  bool trans_b = false;
  bool repo_seed_flag = false;
  bool repo_costmodel_flag = false;
  uint32_t workspace_num = 0;
  uint32_t ub_size = 0;
  BatchmatmulCompileParas params;
  Ub2UbBatchmatmulCompileParas ub2ub_params;
  DynamicMode dynamic_mode = DYNAMIC_MKN;
};

}

#endif  // CANN_OPS_BUILT_IN_OP_UTIL_H_
