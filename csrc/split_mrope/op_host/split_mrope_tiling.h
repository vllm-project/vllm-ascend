/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SplitMropeTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, num_q_heads);
    TILING_DATA_FIELD_DEF(uint64_t, num_kv_heads);
    TILING_DATA_FIELD_DEF(uint64_t, head_size);
    TILING_DATA_FIELD_DEF(uint64_t, rotary_dim);
    TILING_DATA_FIELD_DEF(uint64_t, mrope_section_0);
    TILING_DATA_FIELD_DEF(uint64_t, mrope_section_1);
    TILING_DATA_FIELD_DEF(uint64_t, mrope_section_2);
    TILING_DATA_FIELD_DEF(uint64_t, is_neox_style);
    TILING_DATA_FIELD_DEF(uint64_t, num_tokens);
    TILING_DATA_FIELD_DEF(uint64_t, num_tokens_each_tail_core);
    TILING_DATA_FIELD_DEF(uint64_t, num_tokens_each_front_core);
    TILING_DATA_FIELD_DEF(uint64_t, front_core);
    TILING_DATA_FIELD_DEF(uint64_t, tail_core);
    TILING_DATA_FIELD_DEF(uint64_t, loop_time_each_front_core);
    TILING_DATA_FIELD_DEF(uint64_t, loop_time_each_tail_core);
    TILING_DATA_FIELD_DEF(uint64_t, num_tokens_front_core_each_loop);
    TILING_DATA_FIELD_DEF(uint64_t, num_tokens_tail_core_each_loop);
    TILING_DATA_FIELD_DEF(uint64_t, num_tokens_front_core_last_loop);
    TILING_DATA_FIELD_DEF(uint64_t, num_tokens_tail_core_last_loop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SplitMrope, SplitMropeTilingData)
}
