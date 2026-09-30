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
 * \file mx_a8w4_scale_trans_id_constant.h
 * \brief
 */
#pragma once

namespace WeightQuantBatchMatmulV2::Arch35 {
// 8*16 transId lookup table for MxA8W4: value = i + j * 80
static constexpr volatile __gm__ uint16_t MX_TRANS_ID[128] = {
    0,    80,   160,  240,  320,  400,  480,  560,  640,  720,  800,  880,  960,  1040, 1120, 1200, 1,    81,   161,
    241,  321,  401,  481,  561,  641,  721,  801,  881,  961,  1041, 1121, 1201, 2,    82,   162,  242,  322,  402,
    482,  562,  642,  722,  802,  882,  962,  1042, 1122, 1202, 3,    83,   163,  243,  323,  403,  483,  563,  643,
    723,  803,  883,  963,  1043, 1123, 1203, 4,    84,   164,  244,  324,  404,  484,  564,  644,  724,  804,  884,
    964,  1044, 1124, 1204, 5,    85,   165,  245,  325,  405,  485,  565,  645,  725,  805,  885,  965,  1045, 1125,
    1205, 6,    86,   166,  246,  326,  406,  486,  566,  646,  726,  806,  886,  966,  1046, 1126, 1206, 7,    87,
    167,  247,  327,  407,  487,  567,  647,  727,  807,  887,  967,  1047, 1127, 1207};

} // namespace WeightQuantBatchMatmulV2::Arch35
