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
 * \file tbe_tiling_api.h
 * \brief
 */
#ifndef TBE_TILING_API_H
#define TBE_TILING_API_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "graph/utils/type_utils.h"
#include "platform/platform_infos_def.h"

namespace optiling {
struct Conv3dBackpropV2TBETilingData {
    // L0 tiling parameters
    int32_t m_l0;          // Base M dimension at L0
    int32_t k_l0;          // Base K dimension at L0
    int32_t n_l0;          // Base N dimension at L0

    // L1 tiling parameters
    int32_t m_al1;         // Step M dimension at L1
    int32_t n_bl1;         // Step N dimension at L1
    int32_t k_al1;         // Step K dimension A at L1
    int32_t k_bl1;         // Step K dimension B at L1

    // Buffer parameters
    int32_t db_l0c;        // L0C buffer size
    int32_t db_al1;        // AL1 buffer size
    int32_t db_bl1;        // BL1 buffer size

    // Dimension parameters
    int32_t batch_dim;     // Batch dimension
    int32_t d_dim;         // Depth dimension
    int32_t group_dim;     // Group dimension
    int32_t m_dim;         // M dimension
    int32_t n_dim;         // N dimension
    int32_t k_dim;         // K dimension
};

struct Conv3dBpFilterV2RunInfo {
    int32_t batch;
    int32_t co;             // output channels
    int32_t ci;             // input channels
    int32_t cout1_g;        // output channels per group
    int32_t cin1_g;         // input channels per group
    int32_t dout;           // output depth
    int32_t wo;             // output width
    int32_t ho;             // output height
    int32_t wi;             // input width
    int32_t hi;             // input height
    int32_t di;             // input depth
    int32_t kw;             // kernel width
    int32_t kh;             // kernel height
    int32_t kd;             // kernel depth
    int32_t real_g;         // actual groups
    int32_t stride_w;
    int32_t stride_h;
    int32_t stride_d;
    int32_t pad_l;          // left padding
    int32_t pad_r;          // right padding
    int32_t pad_u;          // up padding
    int32_t pad_d;          // down padding
    int32_t pad_f;          // front padding
    int32_t pad_b;          // back padding
    int32_t dilation_w;
    int32_t dilation_h;
    int32_t dilation_d;
    int32_t ci1;            // another input channels parameter
    uint64_t bl1_bound;     // buffer limit 1 bound, tiling结果的衍生参数，不建议放在这里
    int32_t batch_dout_single_core;  // batch*dout per core, tiling结果的衍生参数，不建议放在这里

    // Tiling parameters
    uint32_t k0;
    uint32_t m0;
    uint32_t n0;
    uint32_t hf32Flag;

    ge::DataType a_dtype = ge::DT_FLOAT16;
    ge::DataType b_dtype = ge::DT_FLOAT16;
    ge::DataType c_dtype = ge::DT_FLOAT16;
    int32_t a_dtype_bytes = 2;
    int32_t b_dtype_bytes = 2;
    int32_t c_dtype_bytes = 2;
    uint32_t core_num;
};

struct Conv3dBpInputV2RunInfo {
    // Batch and group related
    int32_t batch_n;           // Batch size
    int32_t real_g;            // Number of groups

    // Input dimensions (dedx)
    int32_t dedx_d;            // Input depth
    int32_t dedx_cin;         // Input channels per group
    int32_t dedx_cin1;         // Input channels per group
    int32_t dedx_cin1_g;       // Input channels per group (grouped)
    int32_t dedx_h;            // Input height
    int32_t dedx_w;            // Input width

    // Output dimensions (dedy)
    int32_t dedy_d;            // Output depth
    int32_t dedy_cout;        // Output channels per group
    int32_t dedy_cout1;        // Output channels per group
    int32_t dedy_cout1_g;      // Output channels per group (grouped)
    int32_t dedy_h;            // Output height
    int32_t dedy_w;            // Output width

    // Kernel dimensions
    int32_t kernel_d;          // Kernel depth
    int32_t kernel_h;          // Kernel height
    int32_t kernel_w;          // Kernel width

    // Strides
    int32_t stride_d;          // Stride depth
    int32_t stride_h;          // Stride height
    int32_t stride_w;          // Stride width

    // Padding
    int32_t pad_h;             // Padding height
    int32_t pad_t;             // Padding top
    int32_t pad_u;             // Padding up
    int32_t pad_d;             // Padding down
    int32_t pad_l;             // Padding left
    int32_t pad_r;             // Padding right

    // Dilation
    int32_t dilation_d;        // Dilation depth
    int32_t dilation_h;        // Dilation height
    int32_t dilation_w;        // Dilation width

    // Backprop padding
    int32_t backprop_pad_h;    // Backprop padding height
    int32_t backprop_pad_t;    // Backprop padding top
    int32_t backprop_pad_u;    // Backprop padding up
    int32_t backprop_pad_d;    // Backprop padding down
    int32_t backprop_pad_l;    // Backprop padding left
    int32_t backprop_pad_r;    // Backprop padding right

    // Other flags
    int32_t hf32_flag;         // Flag for FP32 handling
    int32_t a_dtype_bytes = 2;
    int32_t b_dtype_bytes = 2;
    int32_t c_dtype_bytes = 2;
    int32_t initOutputFlag = 0;
};

struct Conv3DBackpropV2CompileInfo {
  std::string soc_version = "";
  platform_ascendc::SocVersion shortSocVersion = platform_ascendc::SocVersion::ASCEND910B;

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

enum OpTypeV2 : size_t {
  kConv3DBackpropFilterV2,
  kConv3DBackpropInputV2,
  kConv3DTransposeV2,
};

bool GetTbeTiling(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoForV2, Conv3dBackpropV2TBETilingData& tbeTilingForV2);
bool GetTbeTiling(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
    Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType);
bool GetTbeTiling(gert::TilingContext* context, Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType);
}
#endif  // TBE_TILING_API_H