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
 * \file prompt_flash_attention_tiling_struct.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_TILING_STRUCT_H
#define PROMPT_FLASH_ATTENTION_TILING_STRUCT_H
namespace optiling {

enum class InputLayout {
    SH,
    BSH,
    BNSD,
    NSD,
    BSND,
    BNSD_BSND,
    TND,
    NTD,
    NTD_TND,
    NZ,
    BBH,
    BNBD,
    NONE,
};

enum class TilingMod {
    CVSAME = 0,
    CVDIFF,
    CVDIFF_BASE_API,
    CVDIFF_MLA,
};

enum class SplitCoreMode {
    SPLIT_NBS_VECTOR = 0,
    SPLIT_NBS_CUBE,
    SPLIT_ONEN_VECTOR,
    SPLIT_ONEN_CUBE,
    BALANCE_VECTOR,
    BALANCE_CUBE,
    SPLIT_S1OUT_CUBE,
};
} // namespace optiling

#endif // PROMPT_FLASH_ATTENTION_TILING_STRUCT_H