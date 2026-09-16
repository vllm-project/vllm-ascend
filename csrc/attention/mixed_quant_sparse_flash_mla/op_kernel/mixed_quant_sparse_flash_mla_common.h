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
 * \file mixed_quant_sparse_flash_mla_common.h
 * \brief
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_COMMON_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "mixed_quant_sparse_flash_mla_metadata.h"

using namespace AscendC;

enum class QSMLA_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

enum class QSMLATemplateMode {
    SWA_TEMPLATE_MODE = 0,
    HCA_TEMPLATE_MODE = 1,
    CSA_TEMPLATE_MODE = 2,
    ORI_SPARSE_TEMPLATE_MODE = 3,
    ORI_CMP_SPARSE_TEMPLATE_MODE = 4
};

enum class SCALE_CONTIGUOUS_MODE {
    CONTIGUOUS = 1,
    NONCONTIGUOUS = 2,
};
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_COMMON_H
