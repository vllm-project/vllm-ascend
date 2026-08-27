/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_cache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_OP_CACHE_DEF_TILING_H
#define OPS_BUILT_IN_OP_TILING_OP_CACHE_DEF_TILING_H

#include <array>
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
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

struct BatchmatmulRunParas {
    bool nd_flag = false;
    bool use_pre_ub = false;
    bool trans_a_flag = false;
    bool trans_b_flag = false;
    bool format_a_nd = false;
    bool format_b_nd = false;
    bool format_out_nd = false;
    ge::Format format_a = ge::FORMAT_ND;
    ge::Format format_b = ge::FORMAT_ND;
    ge::Format format_out = ge::FORMAT_ND;
    bool reserved_bool = false;
    bool b_have_batch = false;         // dim num > 2
    bool is_batch_matmul_mode = false; // dynamic_mode == "dynamic_mknb"
    bool is_batch_matmul_op = false;   // BatchMatMulV2 or BatchMatMul
    bool used_aligned_pattern = false;
    bool non_factor_k = false;
    bool non_factor_bmn = false;
    bool bias_flag = false;
    bool pattern_flag = false;
    bool do_not_multi_batch = false;
    bool performance_flag = false;
    bool unaligned_flag = false;
    bool zero_flag = false;
    bool is_compress_quant = false;
    bool is_bmm_fixp = false;
    bool enable_nz_fusion = false;
    bool weight_nz_flag = false;
    int8_t enable_pad = 0;
    int8_t hf32_flag = 1;
    int8_t pad_flag = 0;
    int8_t nz_fusion_flag = 0;
    int32_t dtype_a = 0;
    int32_t dtype_b = 0;
    int32_t dtype_out = 0;
    int32_t dtype_bias = 0;
    int64_t m_mapped = 1;
    int64_t k_mapped = 1;
    int64_t n_mapped = 1;
    int64_t batch_mapped = 1;
    int64_t m = 1;
    int64_t k = 1;
    int64_t n = 1;
    int64_t batch = 1;
    int64_t ori_shape_m = 1;
    int64_t ori_shape_k = 1;
    int64_t ori_shape_n = 1;
    int64_t m_pad = 0;
    int64_t k_pad = 0;
    int64_t n_pad = 0;
    int64_t nl0 = 1;
    int64_t kl0 = 1;
    int64_t dim0_a = 0;
    int64_t dim1_a = 0;
    int64_t dim2_a = 0;
    int64_t dim0_b = 0;
    int64_t dim1_b = 0;
    int64_t dim2_b = 0;
    int64_t batch_a1 = 1;
    int64_t batch_a2 = 1;
    int64_t batch_a3 = 1;
    int64_t batch_a4 = 1;
    int64_t batch_b1 = 1;
    int64_t batch_b2 = 1;
    int64_t batch_b3 = 1;
    int64_t batch_b4 = 1;
    int64_t batch_c1 = 1;
    int64_t batch_c2 = 1;
    int64_t batch_c3 = 1;
    int64_t batch_c4 = 1;
    int32_t offset_x = 0;
    int32_t index_size = 0;
    bool m_quant_check = false;
    bool n_quant_check = false;
    bool is_weight_quant_bmm = false;
    bool vector_pre_conv_mode = false;
    bool is_quant_batch_matmul_v3 = false;
    bool is_weight_quant_batch_matmul_v2 = false;
    bool is_pertoken = false;
    // 3 is perm_a dim
    std::array<size_t, 3> perm_a = {0, 0, 0};
    // 3 is perm_b dim
    std::array<size_t, 3> perm_b = {0, 0, 0};
    ge::DataType bias_dtype = ge::DT_FLOAT16;
};

class CacheTilingData
{
public:
    uint64_t tiling_id;
    int64_t n_cub = 1;
    int64_t db_cub = 1;
    int64_t m_l0 = 1;
    int64_t k_l0 = 1;
    int64_t n_l0 = 1;
    int64_t batch_dim = 1;
    int64_t n_dim = 1;
    int64_t m_dim = 1;
    int64_t k_dim = 1;
    int64_t kal1_16 = 1;
    int64_t kbl1_16 = 1;
    int64_t kal1_factor = 1;
    int64_t kbl1_factor = 1;
    int64_t m_al1 = 1;
    int64_t n_bl1 = 1;
    int64_t db_al1 = 1;
    int64_t db_bl1 = 1;
    int64_t k_aub = 1;
    int64_t m_aub = 1;
    int64_t db_aub = 1;
    int64_t k_bub = 1;
    int64_t n_bub = 1;
    int64_t db_bub = 1;
    int64_t aub_dim = 1;
    int64_t bub_dim = 1;
    int64_t m1_aub = 1;
    int64_t n1_bub = 1;
    int64_t k1_aub = 1;
    int64_t k1_bub = 1;
    int64_t m_aub_dim = 1;
    int64_t n_bub_dim = 1;
    int64_t k_aub_dim = 1;
    int64_t k_bub_dim = 1;
    int64_t k_org_dim = 1;
    int64_t db_l0c = 1;
    int64_t batch_l0 = 1;
    int64_t batch_aub = 1;
    int64_t batch_bub = 1;
    int64_t batch_cub = 1;
    int32_t out_branch_flag = 1;
    int32_t bias_flag = 0;
    int32_t aub_multi_flag = 0;
    int32_t bub_multi_flag = 0;
    int64_t a_align_value = 1;
    int64_t b_align_value = 1;
    int64_t aub_align_bound = 0;
    int64_t bub_align_bound = 0;
    int64_t min_kl1_cmp_kl0 = 0;
    int32_t al1_attach_flag = 0;
    int32_t bl1_attach_flag = 0;
    int32_t abkl1_attach_flag = 0;
    int32_t l0c_multi_batch = 0;
    int64_t m_single_core = 1;
    int64_t n_single_core = 1;
    bool flag_cub_solving_bank_conflict = false;
    bool al1_full_load = false;
    bool bl1_full_load = false;
    int8_t hf32_flag = 1;
    int32_t zero_flag = 0;
    bool datatype_bf16 = false;
    uint64_t deq_scale_var = 0x3F800000;
    uint32_t l2_cache_flag = 0;
};
} // namespace optiling

#endif