/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "../../incre_flash_attention/op_host/incre_flash_attention_tiling.h"
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "arch22/fused_infer_attention_score_tiling_v3.h"
#include "flash_attention_infer_tiling.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../incre_flash_attention/op_host/incre_flash_attention_tiling_impl.h"
#include "arch35/fused_infer_attention_score_tiling_v4.h"
#include "arch38/fused_infer_attention_score_tiling_arch38.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000210100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000211100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000201203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000100201203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010201203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000110201203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000210100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000211100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000000201203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5100000000010201203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200106, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010200106, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000201106, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010201106, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200206, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010200206, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000201206, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000010201206, FAInferTilingData)

// Decoding 场景 (pagedCacheFlag == true && qSeqlen == 1 && NO_MASK && !lseFlag)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5200000000010200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5200000000010200200, FAInferTilingData)

// Test purposes - using old key
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore, IncreFlashAttentionTilingDataV2)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_13, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_14, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_27, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_30, IncreFlashAttentionEmptyInputTilingData)

// full quant, org_dtype is bfloat16
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000000020222331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000000020322331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000001020222331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000001020322331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000001020222332, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000001020322332, IncreFlashAttentionTilingDataMla)
// Cv1:1
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000010020222331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000010020322331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000011020222331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000011020322331, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000011020222332, IncreFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_15000011020322332, IncreFlashAttentionTilingDataMla)

// PFA
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000001001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000001001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000020, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020210, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020211, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020215, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020216, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000210, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000211, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000215, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000216, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000010, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000011, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000015, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000016, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000300, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000400, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000110, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000111, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000115, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000116, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021112, PromptFlashAttentionTilingData)
// PA tilingkey
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001217, PromptFlashAttentionTilingData)
// prefix tilingkey
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000101001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000101001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001217, PromptFlashAttentionTilingData)

// msd tilingkey
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300021112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200021112, PromptFlashAttentionTilingData)

// msd tilingkey fp16
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200021612, PromptFlashAttentionTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000002004000012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000000004001012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000010004001012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000000004000012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000010004000012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000002004010112, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000000004010112, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_2000000010004010112, PromptFlashAttentionBaseApiTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_30000000000200302, IncreFlashAttentionTilingAtbDataV2)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_30000000000222322, IncreFlashAttentionTilingAtbDataV2)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000000000000, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000000000001, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000000000002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000000000003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000000100002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000000100003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000020000002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000020000003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000020100002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000020100003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000010000002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000010000003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000010100002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_4000000000010100003, MLAGeneralTilingData)

static void ConvertDataTypePFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    contextKeyParams.inputDataType = context.GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context.GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context.GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
        context.GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
        context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.quantScale2Type = (context.GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.blockTableType = (context.GetOptionalInputDesc(BLOCK_TABLE_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(BLOCK_TABLE_INDEX)->GetDataType() : ge::DT_INT32;
    contextKeyParams.outputDataType = context.GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
    contextKeyParams.KeyAntiquantScaleType = (context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.valueAntiquantScaleType = (context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.KeyAntiquantOffsetType = (context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.valueAntiquantOffsetType = (context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX)->GetDataType() : contextKeyParams.inputDataType;
}

static void ConvertShapePFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    contextKeyParams.queryInputShape = context.GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context.GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context.GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape = context.GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context.GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape = context.GetOptionalInputShape(DEQUANT_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context.GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context.GetOptionalInputShape(DEQUANT_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context.GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context.GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context.GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context.GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.blockTableShape = context.GetOptionalInputShape(BLOCK_TABLE_INDEX);
    contextKeyParams.queryRope = context.GetOptionalInputShape(QUERY_ROPE_INDEX);
    contextKeyParams.keyRope = context.GetOptionalInputShape(KEY_ROPE_INDEX);
    contextKeyParams.outputShape = context.GetOutputShape(ATTENTION_OUT_INDEX);
    contextKeyParams.lseoutputShape = context.GetOutputShape(SOFTMAX_LSE_INDEX);

    contextKeyParams.KeyAntiquantScaleShape = context.GetOptionalInputShape(KEY_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.valueAntiquantScaleShape = context.GetOptionalInputShape(VALUE_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.KeyAntiquantOffsetShape = context.GetOptionalInputShape(KEY_ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.valueAntiquantOffsetShape = context.GetOptionalInputShape(VALUE_ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.learnableSinkShape = context.GetOptionalInputShape(LEARNABLE_SINK_INDEX);
}

static ge::graphStatus ConvertAttrsPFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Attributes returned from GetAttrs() is a nullptr"),
        return ge::GRAPH_FAILED);
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    contextKeyParams.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.blockSize = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX);
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1U : 0U;
    contextKeyParams.softmaxLseFlag = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    contextKeyParams.isSoftMaxLseEnable =
        (contextKeyParams.softmaxLseFlag == nullptr) ? false : *contextKeyParams.softmaxLseFlag;
    contextKeyParams.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_KEY_ANTIQUANT_MODE_INDEX);
    contextKeyParams.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_VALUE_ANTIQUANT_MODE_INDEX);

    OP_CHECK_IF(context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX) != nullptr ||
        (attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX) != nullptr &&
        *attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX) != 0),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "PFA not support query dequant now"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(*contextKeyParams.sparseMode == 9U,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "PFA not support Tree Sparse(9) now"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetCumulativeKeyValueSInBSH(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    // DIM_2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
    auto standardH = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_2);
    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardH) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardH)) {
            OP_LOGE(context.GetNodeName(), "D is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE(context.GetNodeName(), "S from Key and Value are not equal!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) == 0) {
            contextKeyParams.emptyTensor = 1U;
        }
        cumulativeKeyS += contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        cumulativeValueS += contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            static_cast<uint32_t>(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetCumulativeKeyValueSInBNSD(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(1);
    auto standardD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_3);
    int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads :
                                                                    *contextKeyParams.headsNumber;
    if (tmpNKv != standardN) {
        OP_LOGE(context.GetNodeName(), "kvN from tensorlist does NOT EQUAL kvN from attribute!");
        return ge::GRAPH_FAILED;
    }

    for (int64_t idx = 0; idx < validBatchOfK; ++idx) {
        if ((contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(1) != standardN) ||
            (contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(1) != standardN)) {
            OP_LOGE(context.GetNodeName(), "N is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if ((contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_3) != standardD) ||
            (contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(DIM_3) != standardD)) {
            OP_LOGE(context.GetNodeName(), "D is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2) !=
            contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(DIM_2)) {
            OP_LOGE(context.GetNodeName(), "S from Key and Value does NOT equal but they should!");
            return ge::GRAPH_FAILED;
        }
        // DIM_2: Traverse the k list of the tiling key to check whether the second dimension of each tensor is 0.
        if (contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2) == 0) {
            contextKeyParams.emptyTensor = 1U;
        }
        cumulativeKeyS += contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2);
        cumulativeValueS += contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(DIM_2);
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            uint32_t(contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2)));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetCumulativeKeyValueSInBSND(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_2);
    auto standardD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_3);
    int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ?
        *contextKeyParams.numKeyValueHeads :*contextKeyParams.headsNumber;
    if (tmpNKv != standardN) {
        OP_LOGE(context.GetNodeName(), "kvN from tensorlist does NOT EQUAL kvN from attribute!");
        return ge::GRAPH_FAILED;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
         // DIM_2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardN) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardN)) {
            OP_LOGE(context.GetNodeName(), "N is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_3) != standardD) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_3) != standardD)) {
            OP_LOGE(context.GetNodeName(), "D is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE(context.GetNodeName(), "S from Key and Value does NOT equal but they should!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) == 0) {
            contextKeyParams.emptyTensor = 1U;
        }
        cumulativeKeyS += contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        cumulativeValueS += contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCumulativeKeyValue(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    const string layoutStr = string(contextKeyParams.layout);
    if (layoutStr == "BSH") {
        // check all H across batches and KVs are the same under BSH layout
        OP_CHECK_IF(GetCumulativeKeyValueSInBSH(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK)!= ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "get cumulativeKeyS and cumulativeValueS in BSH failed"), return ge::GRAPH_FAILED);
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") {
        // check N and D, respectively, are the same across batches and KVs under BNSD/BNSD_BSND
        OP_CHECK_IF(GetCumulativeKeyValueSInBNSD(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "get cumulativeKeyS and cumulativeValueS in BNSD/BNSD_BSND failed"), return ge::GRAPH_FAILED);
    } else {
        // check N and D, respectively, are the same across batches and KVs under BSND
        OP_CHECK_IF(GetCumulativeKeyValueSInBSND(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "get cumulativeKeyS and cumulativeValueS in BSND failed"), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((contextKeyParams.emptyTensor == 1) && (cumulativeKeyS != 0) && (cumulativeValueS != 0),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Got empty tensor in key and value which is not continuous.!!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckKvPFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    const string layoutStr = string(contextKeyParams.layout);
    auto batchOfQ = 1;
    auto batchOfK = 1;
    if (layoutStr != "NSD") {
        batchOfQ = contextKeyParams.queryInputShape->GetStorageShape().GetDim(0);
        batchOfK = contextKeyParams.keyInputShape->GetStorageShape().GetDim(0);
    }

    int64_t validBatchOfK = 0; // Obtain the actual number of K input elements and determine whether they belong to the tensorlist scene
    while (context.GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
        validBatchOfK++;
        if (validBatchOfK > 1) { // If there are more than 1, break. When the input is large, it saves time. The tensorlist scene also needs to verify separately whether it is 1
            break;
        }
    }
    if ((batchOfQ != batchOfK) && (validBatchOfK > 1) && (contextKeyParams.blockTable == nullptr)) {
        validBatchOfK = 0;
        int64_t validBatchOfV = 0;
        int64_t cumulativeKeyS = 0;
        int64_t cumulativeValueS = 0;
        contextKeyParams.kTensorList.resize(batchOfQ);
        contextKeyParams.vTensorList.resize(batchOfQ);
        while (context.GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
            contextKeyParams.kTensorList[validBatchOfK] = context.GetDynamicInputShape(KEY_INDEX, validBatchOfK);
            OP_CHECK_IF(contextKeyParams.kTensorList[validBatchOfK]->GetStorageShape().GetDim(0) != 1,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "Batch value of Key is NOT 1 but should be 1 under tensorlist mode!"),
                return ge::GRAPH_FAILED);
            validBatchOfK++;
        }

        while (context.GetDynamicInputShape(VALUE_INDEX, validBatchOfV) != nullptr) {
            contextKeyParams.vTensorList[validBatchOfV] = context.GetDynamicInputShape(VALUE_INDEX, validBatchOfV);
            OP_CHECK_IF(contextKeyParams.vTensorList[validBatchOfV]->GetStorageShape().GetDim(0) != 1,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "Batch value of Value is NOT 1 but should be 1 under tensorlist mode!"),
                return ge::GRAPH_FAILED);
            validBatchOfV++;
        }

        OP_CHECK_IF((batchOfQ != validBatchOfK) || (validBatchOfK != validBatchOfV),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "Batch of Query, Key and Value do NOT equal but should equal under tensorlist mode!"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(CheckCumulativeKeyValue(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "check cumulativeKeyS and cumulativeValueS failed"),
            return ge::GRAPH_FAILED);

        contextKeyParams.isKvContinuous = 0U;
    }
     return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParamsPFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    OP_CHECK_IF(CheckKvPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "check kv failed"), return ge::GRAPH_FAILED);

    const string layoutStr = string(contextKeyParams.layout);
    OP_CHECK_IF(((contextKeyParams.isKvContinuous == 0) && layoutStr == "TND"),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when layout is TND, tensorlist is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((contextKeyParams.queryPaddingSize != nullptr || contextKeyParams.kvPaddingSize != nullptr) && layoutStr == "TND"),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when layout is TND, left padding is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((contextKeyParams.isKvContinuous == 0) &&
         ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when tensorlist is used, left padding is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) &&
        (contextKeyParams.queryPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
        contextKeyParams.queryPaddingSize->GetStorageShape().GetDimNum() != 1)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query PaddingSize input is invalid!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) &&
        (contextKeyParams.kvPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
        contextKeyParams.kvPaddingSize->GetStorageShape().GetDimNum() != 1)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "KV PaddingSize input is invalid!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(((contextKeyParams.blockTable != nullptr) &&
        ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "when page attention is used, left padding is not supported!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthQ == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "if Query has leftpadding, the query's actual sequence lengths are required!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthKV == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "if KV has leftpadding, the key/value's actual sequence lengths are required!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsPFA(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams)
{
    constexpr uint32_t FROM_FUSED_FLAG = 71;

    contextKeyParams.opName = context.GetNodeName();
    bool inputOutputIsNullPtr =
        (context.GetInputDesc(QUERY_INDEX) == nullptr) || (context.GetInputDesc(KEY_INDEX) == nullptr) ||
        (context.GetInputDesc(VALUE_INDEX) == nullptr) || (context.GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
        (context.GetInputShape(QUERY_INDEX) == nullptr) || (context.GetInputShape(KEY_INDEX) == nullptr) ||
        (context.GetInputShape(VALUE_INDEX) == nullptr) || (context.GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OP_CHECK_IF(inputOutputIsNullPtr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "q, k, v or attenOut is nullptr!"),
        return ge::GRAPH_FAILED);

    contextKeyParams.isKvContinuous = 1U;
    contextKeyParams.emptyTensor = 0U;
    contextKeyParams.fromTilingSink = 0U;
    contextKeyParams.fromFused = FROM_FUSED_FLAG;
    contextKeyParams.maxKVs = 0U;
    contextKeyParams.pseShift = context.GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context.GetOptionalInputTensor(ATTEN_MASK_INDEX);
    OP_CHECK_IF((contextKeyParams.attentionMask != nullptr) &&
        (context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_BOOL) &&
        (context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_INT8) &&
        (context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_UINT8),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Invalid attention mask datatype! Only support BOOL, INT8 and UINT8"), return ge::GRAPH_FAILED);
    contextKeyParams.actualSequenceLengthQ = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSequenceLengthKV = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.queryPaddingSize = context.GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    contextKeyParams.kvPaddingSize = context.GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    contextKeyParams.blockTable = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    contextKeyParams.keySharedPrefix = context.GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    contextKeyParams.valueSharedPrefix = context.GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    contextKeyParams.actualSharedPrefixLen = context.GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    contextKeyParams.learnableSink = context.GetOptionalInputTensor(LEARNABLE_SINK_INDEX);
    contextKeyParams.hasKeyAntiquantScale =
        (context.GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX) == nullptr) ? false : true;
    contextKeyParams.hasValueAntiquantScale =
        (context.GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX) == nullptr) ? false : true;

    ConvertDataTypePFA(context, contextKeyParams);
    ConvertShapePFA(context, contextKeyParams);

    contextKeyParams.hasLearnableSink = ((contextKeyParams.learnableSink != nullptr) && (contextKeyParams.learnableSinkShape != nullptr) &&
                                        (contextKeyParams.learnableSinkShape->GetStorageShape().GetShapeSize() != 0) ) ? true : false;

    OP_CHECK_IF(ConvertAttrsPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "convert attrs failed"), return ge::GRAPH_FAILED);
    contextKeyParams.workspaceSize = context.GetWorkspaceSizes(1);

    OP_CHECK_IF(CheckParamsPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "check params failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void ConvertOptionalInputsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INDEX);
    ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INDEX);

    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INDEX);

    ifaContext.actualSeqLengthsQ.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX);
    ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    ifaContext.queryPaddingSize.tensor = context.GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    ifaContext.keyAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantOffset.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.keyAntiquantOffset.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantScale.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantScale.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantOffset.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantOffset.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.keySharedPrefix.tensor = context.GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    ifaContext.keySharedPrefix.desc = context.GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.tensor = context.GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.desc = context.GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.actualSharedPrefixLen.tensor = context.GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);

    ifaContext.queryRope.tensor = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    ifaContext.queryRope.desc = context.GetOptionalInputDesc(QUERY_ROPE_INDEX);
    ifaContext.keyRope.tensor = context.GetOptionalInputTensor(KEY_ROPE_INDEX);
    ifaContext.keyRope.desc = context.GetOptionalInputDesc(KEY_ROPE_INDEX);
    ifaContext.keyRopeAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyRopeAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    ifaContext.dequantScaleQuery.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    ifaContext.dequantScaleQuery.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX);
}

static ge::graphStatus ConvertAttrsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "attrs got from ge is nullptr"),
        return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    ifaContext.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX);
    ifaContext.antiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_ANTIQUANT_MODE_INDEX);
    ifaContext.softmaxLseFlag = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    ifaContext.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_KEY_ANTIQUANT_MODE_INDEX);
    ifaContext.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_VALUE_ANTIQUANT_MODE_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(ATTR_INNER_PRECISE_INDEX);
    ifaContext.sparseMode = attrs->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX);
    ifaContext.queryQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    ifaContext.windowSize = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    ifaContext.query.desc = context.GetInputDesc(QUERY_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INDEX);
    OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "shape of query of shape of key is null."),
        return ge::GRAPH_FAILED);
    auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(0);
    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, size));
            ifaContext.vCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(1);
        ifaContext.vCache.resize(1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, 0));
    }

    ifaContext.value.desc = context.GetInputDesc(VALUE_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(ATTENTION_OUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(ATTENTION_OUT_INDEX);

    ConvertOptionalInputsIFA(context, ifaContext);

    OP_CHECK_IF(ConvertAttrsIFA(context, ifaContext) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "convert attrs failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDequantParams(gert::TilingContext &context, const int64_t s)
{
    OP_CHECK_IF((context.GetAttrs()->GetAttrPointer<int64_t>(ATTR_ANTIQUANT_MODE_INDEX) != nullptr) &&
        (*context.GetAttrs()->GetAttrPointer<int64_t>(ATTR_ANTIQUANT_MODE_INDEX) != 0),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "antiquant_mode is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(s > NUM_16 && (context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX) != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when s(%ld) > 16, not support dequantScaleQuery exist", s),
        return ge::GRAPH_FAILED);

    auto qRope = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    OP_CHECK_IF(qRope == nullptr && (context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX) != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when qRope is null, not support dequantScaleQuery exist"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckLseShape(gert::TilingContext &context, bool lseFlag, const int64_t b,
    const int64_t s, const int64_t n)
{
    const string inputLayoutStr = string(context.GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    auto tempLse = context.GetOutputShape(SOFTMAX_LSE_INDEX);

    if (inputLayoutStr == "TND" || inputLayoutStr == "NTD_TND") {
        OP_CHECK_IF(((lseFlag != false) && (tempLse->GetStorageShape().GetDimNum() != 3)),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "Layout is %s SoftmaxLse shape dim should be 3, but got %zu!",
            inputLayoutStr.c_str(), tempLse->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

        auto tempQ = context.GetInputShape(QUERY_INDEX);
        OP_CHECK_IF(((lseFlag != false) &&
            ((tempLse->GetStorageShape().GetDim(DIM_0) != s) || (tempLse->GetStorageShape().GetDim(DIM_1) != n) ||
            (tempLse->GetStorageShape().GetDim(DIM_2) != 1))),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "Layout is %s query Shape is [%ld, %ld, %ld], expect SoftmaxLse shape TN1 [%ld, %ld, 1], but got "
                "SoftmaxLse shape [%ld, %ld, %ld]!",
                inputLayoutStr.c_str(),
                tempQ->GetStorageShape().GetDim(DIM_0), tempQ->GetStorageShape().GetDim(DIM_1),
                tempQ->GetStorageShape().GetDim(DIM_2), s, n,
                tempLse->GetStorageShape().GetDim(DIM_0), tempLse->GetStorageShape().GetDim(DIM_1),
                tempLse->GetStorageShape().GetDim(DIM_2)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(((lseFlag != false) && (tempLse->GetStorageShape().GetDimNum() != 4)),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "SoftmaxLse shape dim should be 4!"),
            return ge::GRAPH_FAILED);

        int64_t tempN = *context.GetAttrs()->GetAttrPointer<int64_t>(ATTR_N_INDEX);
        OP_CHECK_IF(((lseFlag != false) &&
            ((tempLse->GetStorageShape().GetDim(0) != b) || (tempLse->GetStorageShape().GetDim(1) != tempN) ||
            (tempLse->GetStorageShape().GetDim(DIM_2) != s) || (tempLse->GetStorageShape().GetDim(DIM_3) != 1))),
            OPS_REPORT_VECTOR_INNER_ERR(
                context.GetNodeName(),
                "SoftmaxLse shape size[%ld, %ld, %ld, %ld] does not match BNS1[%ld, %ld, %ld, 1]!",
                tempLse->GetStorageShape().GetDim(0), tempLse->GetStorageShape().GetDim(1),
                tempLse->GetStorageShape().GetDim(DIM_2), tempLse->GetStorageShape().GetDim(DIM_3), b, tempN, s),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetPlatformInfo(gert::TilingContext &context, PromptFlashAttentionCompileInfo &compileInfoPtr)
{
    auto platformInfoPtr = context.GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr.l0BSize);
    compileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();

    if (compileInfoPtr.socShortName == platform_ascendc::SocVersion::ASCEND310P) {
        // sys workspace size default value
        compileInfoPtr.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    } else {
        compileInfoPtr.defaultSysWorkspaceSize = 0U;

        OP_CHECK_IF((compileInfoPtr.aivNum != compileInfoPtr.aicNum) && (compileInfoPtr.aivNum != compileInfoPtr.aicNum * 2U),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "aicNum(%u):aivNum(%u) only support 1:1 or 1:2.",
                compileInfoPtr.aicNum, compileInfoPtr.aivNum), return GRAPH_FAILED);
        OP_CHECK_IF(compileInfoPtr.aivNum == compileInfoPtr.aicNum,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), 
                "when CV 1:1, only support MLA non-quantization(QKV type both are FP16 or BF16) "
                "and MLA fully quantization(QKV type both are int8)"), 
            return GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingProcess4PFA(gert::TilingContext *context, const uint32_t tempD, const int64_t b,
    const int64_t s, const int64_t n)
{
    constexpr uint64_t BENCHMARK_TILING_KEY = 1000000000000000000;
    constexpr int64_t D_ALIGN_32 = 32;
    constexpr int64_t D_ALIGN_16 = 16;

    PromptFlashAttentionTilingData pfaTilingData;
    PromptFlashAttentionTiling pfa_tiling(nullptr);
    ContextParamsForPFATiling contextParamsForPFATiling;
    PromptFlashAttentionCompileInfo tempCompileInfoPtr = {0, 0, 0, 0, 0, 0, 0, 0,
        platform_ascendc::SocVersion::ASCEND310P};

    auto ret = CheckDequantParams(*context, s);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = SetPlatformInfo(*context, tempCompileInfoPtr);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    contextParamsForPFATiling.compileInfoPtr = &tempCompileInfoPtr;
    ret = ConvertContextToParamsPFA(*context, contextParamsForPFATiling);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while convert tilingContext to PFA context");
        return ret;
    }

    bool lseFlag = *context->GetAttrs()->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    if (lseFlag != false) {
        if (pfa_tiling.CheckNonEmptyShapeExceptions(contextParamsForPFATiling,
                                                  contextParamsForPFATiling.lseoutputShape, "softmaxLse")) {
            return ge::GRAPH_FAILED;
        }
        ret = CheckLseShape(*context, lseFlag, b, s, n);
        if (ret != ge::GRAPH_SUCCESS) return ret;
    }

    const string inputLayout = string(contextParamsForPFATiling.layout);
    OP_CHECK_IF((((contextParamsForPFATiling.inputDataType == ge::DT_INT8) ||
        (contextParamsForPFATiling.kDataType == ge::DT_INT8) ||
        (contextParamsForPFATiling.outputDataType == ge::DT_INT8)) && (tempD % D_ALIGN_32 != 0)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "D should be 32 elements aligned when int8 is involved!!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempD % D_ALIGN_16 != 0), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "D should be 16 elements aligned when with FP16/BF16 dtype!"), return ge::GRAPH_FAILED);
    uint64_t tilingKey = 7U;
    uint32_t blockDimToBeSet;
    pfa_tiling.fromPFA_ = false;
    ret = pfa_tiling.RunBigKernelTilingWithParams(contextParamsForPFATiling, tilingKey, blockDimToBeSet, pfaTilingData);
    tilingKey += BENCHMARK_TILING_KEY;
    OP_LOGD(contextParamsForPFATiling.opName, "The final tiling key is: %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDimToBeSet);
    pfa_tiling.PromptFlashAttentionSetTilingData(context, pfaTilingData);

    return ret;
}

ge::graphStatus CheckFAISeqlenDataInTND(
    const gert::TilingContext *context, bool isPageAttention, int64_t actSeqLenDims, int64_t actSeqLenKVDims)
{
    auto actSeqLenData = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actSeqLenDataKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    auto queryShape = context->GetInputShape(QUERY_INDEX);
    auto keyShape = context->GetInputShape(KEY_INDEX);
    auto valueShape = context->GetInputShape(VALUE_INDEX);

    if (actSeqLenData->GetData<int64_t>() != nullptr && actSeqLenDataKV->GetData<int64_t>() != nullptr) {
        int64_t lastSeqLen = static_cast<int64_t>(actSeqLenData->GetData<int64_t>()[actSeqLenDims - 1]);
        int64_t lastSeqLenKV = static_cast<int64_t>(actSeqLenDataKV->GetData<int64_t>()[actSeqLenKVDims - 1]);
        int64_t queryT = queryShape->GetStorageShape().GetDim(DIM_0);
        int64_t keyT = keyShape->GetStorageShape().GetDim(DIM_0);
        int64_t valueT = valueShape->GetStorageShape().GetDim(DIM_0);

        OP_CHECK_IF(queryT != lastSeqLen,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "When layout is TND, queryT(%ld) must be equal to the last element of actualSequenceLengthQ(%ld)",
                queryT, lastSeqLen),
                return ge::GRAPH_FAILED);
        if (!isPageAttention) {
            OP_CHECK_IF((keyT != lastSeqLenKV) || (valueT != lastSeqLenKV),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "When layout is TND and PA not enabled, "
                    "keyT(%ld) and valueT(%ld) must be equal to the last element of actualSeqenceLengthKV(%ld)",
                    keyT, valueT, lastSeqLenKV),
                    return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSparseModeParams(const gert::TilingContext *context, int64_t actSeqLenDims)
{
    auto actSeqLenData = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actSeqLenDataKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    int64_t preToken = *(context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX));
    int64_t nextToken = *(context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX));
    if (actSeqLenData->GetData<int64_t>() != nullptr && actSeqLenDataKV->GetData<int64_t>() != nullptr) {
        int64_t minS = static_cast<int64_t>(actSeqLenData->GetData<int64_t>()[ATTR_N_INDEX]);
        int64_t minKV = static_cast<int64_t>(actSeqLenDataKV->GetData<int64_t>()[ATTR_N_INDEX]);
        for (uint32_t i = ATTR_SCALE_INDEX; i < actSeqLenDims; i++) {
            int64_t currS = static_cast<int64_t>(actSeqLenData->GetData<int64_t>()[i]) - static_cast<int64_t>(actSeqLenData->GetData<int64_t>()[i-1]);
            int64_t currKV = static_cast<int64_t>(actSeqLenDataKV->GetData<int64_t>()[i]) - static_cast<int64_t>(actSeqLenDataKV->GetData<int64_t>()[i-1]);
            minS = std::min(minS, currS);
            minKV = std::min(minKV, currKV);
        }
        
        OP_CHECK_IF((preToken < 0) && (nextToken < 0),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "preTokens and nextokens cannot neither be negative number, preTokens = %ld, nextTokens = %ld.",
                preToken, nextToken),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((nextToken * (-1)) > preToken, 
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "nexttoken line should be higher than pretoken line, preTokens = %ld, nextTokens = %ld.",
            preToken, nextToken),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckKVNzShape(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    int32_t blockSize = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX)));
    auto blockTable = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    auto tempK = context->GetInputShape(KEY_INDEX);
    auto tempV = context->GetInputShape(VALUE_INDEX);

    OP_CHECK_IF((tempK->GetStorageShape().GetDim(DIM_4) != 16) || (tempV->GetStorageShape().GetDim(DIM_4) != 16),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "The last dimension of kv should to be 16."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempK->GetStorageShape().GetDim(DIM_3) != blockSize) ||
                    (tempV->GetStorageShape().GetDim(DIM_3) != blockSize),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                            "The four dimension of kv should to be :%ld.", blockSize),
                                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAIIsTND(gert::TilingContext *context, bool isPageAttention)
{
    const gert::StorageShape* queryShape = context->GetInputShape(QUERY_INDEX);
    const gert::StorageShape* keyShape = context->GetInputShape(KEY_INDEX);
    const gert::StorageShape* valueShape = context->GetInputShape(VALUE_INDEX);

    auto qDimNum = queryShape->GetStorageShape().GetDimNum();
    auto kDimNum = keyShape->GetStorageShape().GetDimNum();
    auto vDimNum = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(qDimNum != 3U,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "When input layout is TND, Q must have three dims"),
            return ge::GRAPH_FAILED);
    if (!isPageAttention) {
        OP_CHECK_IF(kDimNum != 3U || vDimNum != 3U,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "When input layout is TND and paged cache is not used, K and V must have three dims"),
                return ge::GRAPH_FAILED);
    } else if (kDimNum == 3U || vDimNum == 3U) {
 	         OP_CHECK_IF(kDimNum != vDimNum,
 	             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
 	                 "When input layout is TND and paged cache and kvnd is used, the K and V must have three dims"),
 	                 return ge::GRAPH_FAILED);
    } else if (kDimNum == 5U || vDimNum == 5U) {
        OP_CHECK_IF(kDimNum != vDimNum,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "When input layout is TND and paged cache and kvnz is used, the K and V must have five dims"),
                return ge::GRAPH_FAILED);
        if (CheckKVNzShape(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Tensor* actSeqLenData = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    const gert::Tensor* actSeqLenDataKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    int64_t actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    int64_t actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;
    OP_CHECK_IF((actSeqLenData == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "When layout is TND, actualSequenceLengthQ is required, but now is nullptr!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((actSeqLenDataKV == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "When layout is TND, actualSequenceLengthKV is required, but now is nullptr!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((actSeqLenDims == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "When layout is TND, actualSequenceLengthQ is required, but the number of element in it is 0!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((actSeqLenKVDims == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "When layout is TND, actualSequenceLengthKV is required, but the number of element in it is 0!"),
        return ge::GRAPH_FAILED);
    if (CheckFAISeqlenDataInTND(context, isPageAttention, actSeqLenDims, actSeqLenKVDims) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int32_t sparseMode = static_cast<int32_t>(*(context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX)));
    if (sparseMode == 4 && CheckSparseModeParams(context, actSeqLenDims) != ge::GRAPH_SUCCESS) { // 4: sparseMode
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAIQKV(gert::TilingContext *context, bool isPageAttention)
{
    auto qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    auto vDataType = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF((qDataType != kDataType) || (qDataType != vDataType),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Input dtype of Q, K, and V must be consistent"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF((qDataType != ge::DT_FLOAT16) && (qDataType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Input dtype of Q, K, and V must be FP16 or BF16"),
            return ge::GRAPH_FAILED);

    int64_t validBatchOfK = 0;
    int64_t validBatchOfV = 0;
    while (context->GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
        validBatchOfK++;
        if (validBatchOfK > 1) {
            break;
        }
    }
    while (context->GetDynamicInputShape(VALUE_INDEX, validBatchOfV) != nullptr) {
        validBatchOfV++;
        if (validBatchOfV > 1) {
            break;
        }
    }
    OP_CHECK_IF((validBatchOfK > 1) || (validBatchOfV > 1),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Split fuse senario does not support incontinuous kv tensor list"),
            return ge::GRAPH_FAILED);

    const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "TND") {
        if (CheckFAIIsTND(context, isPageAttention) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAILearnableSink(const gert::TilingContext *context)
{
    auto qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto sinkDataType = context->GetOptionalInputDesc(LEARNABLE_SINK_INDEX)->GetDataType();
    auto queryShape = context->GetInputShape(QUERY_INDEX);
    auto learnableSinkShape = context->GetOptionalInputShape(LEARNABLE_SINK_INDEX);

    auto attrs = context->GetAttrs();
    int32_t tempInnerPrecise = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX)));
    int32_t sparseMode = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX)));

    OP_CHECK_IF(
        (sinkDataType != qDataType),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Input dtype of Q and learnable sink must be consistent"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((sinkDataType != ge::DT_FLOAT16) && (sinkDataType != ge::DT_BF16)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Input dtype of learnable sink must be FP16 or BF16"),
        return ge::GRAPH_FAILED);

    auto sinkDim = learnableSinkShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(sinkDim != 1U,
                OP_LOGE(context->GetNodeName(), "learnable_sink enable, sink shape dim(%u) must be 1!", sinkDim),
                return ge::GRAPH_FAILED);

    auto sinkDimValue = learnableSinkShape->GetStorageShape().GetDim(DIM_0);
    auto queryN = queryShape->GetStorageShape().GetDim(DIM_1);
    OP_CHECK_IF(sinkDimValue != queryN,
                OP_LOGE(context->GetNodeName(), "learnable_sink enable, sink shape(%u) must be same equal queryN(%u)!",
                        sinkDimValue, queryN),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((tempInnerPrecise == 1 || tempInnerPrecise == 2 || tempInnerPrecise == 3),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "When learnable sink is enabled, innerPrecise shall not be 1, 2 or 3"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAISinglePara(const gert::TilingContext *context, bool isPageAttention)
{
    auto attrs = context->GetAttrs();
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto tempK = context->GetInputShape(KEY_INDEX);
    auto tempV = context->GetInputShape(VALUE_INDEX);
    int64_t tempQD = tempQ->GetStorageShape().GetDim(DIM_2);
    int64_t tempKD = 0;
    int64_t tempVD = 0;
    constexpr int64_t BLOCK_SIZE_ALIGN_16 = 16;
    bool isLearnableSinkFlag = context->GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr;
    
    if (!isPageAttention) {
        tempKD = tempK->GetStorageShape().GetDim(DIM_2);
        tempVD = tempV->GetStorageShape().GetDim(DIM_2);
    } else {
        int64_t kvHeadNum = *(attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX));
        int64_t inputBlockSize = *(attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX));
        int64_t cacheBlockSize = 0;
        if (tempK->GetStorageShape().GetDimNum() == 3U) {
            tempKD = (tempK->GetStorageShape().GetDim(DIM_2)) / kvHeadNum;
            tempVD = (tempV->GetStorageShape().GetDim(DIM_2)) / kvHeadNum;
            cacheBlockSize = tempK->GetStorageShape().GetDim(DIM_1);
        } else if (tempK->GetStorageShape().GetDimNum() == 5U) {
            tempKD = (tempK->GetStorageShape().GetDim(DIM_2)) * 16;
            tempVD = (tempV->GetStorageShape().GetDim(DIM_2)) * 16;
            cacheBlockSize = tempK->GetStorageShape().GetDim(DIM_3);
        }
        OP_CHECK_IF(inputBlockSize != cacheBlockSize,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "When paged cache is used, the first dim of K and V must be consistent with input blockSize attr"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputBlockSize % BLOCK_SIZE_ALIGN_16 != 0),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "When input layout is TND and paged cache is used, the input blockSize must be a multiple of 16"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputBlockSize > MAX_BLOCK_SIZE),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "When input layout is TND and paged cache is used, the input blockSize must be less than 512"),
                return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((tempQD != tempKD) || (tempQD != tempVD),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "HeadDim of Q, K, and V must be consistent"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(tempQD > 256U,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "When input layout is TND, headDim shall not exceed 256"),
            return ge::GRAPH_FAILED);
    
    if (isLearnableSinkFlag) {
 	         return CheckFAILearnableSink(context);
 	}

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAIPseShift(gert::TilingContext *context)
{
    // check pse shift shape (B/1, N, S1, S2)
    auto pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    if (pseShiftShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t pseShiftDimNum = pseShiftShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(pseShiftDimNum != 4,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "The input shape of pse shift must have 4 dims, current dim num is %u.",
            pseShiftDimNum),
        return GRAPH_FAILED);
    uint32_t pseShiftBatch = pseShiftShape->GetStorageShape().GetDim(PSE_SHIFT_B);
    uint32_t pseShiftN = pseShiftShape->GetStorageShape().GetDim(PSE_SHIFT_N);
    uint32_t pseShiftS1 = pseShiftShape->GetStorageShape().GetDim(PSE_SHIFT_S0);
    uint32_t pseShiftS2 = pseShiftShape->GetStorageShape().GetDim(PSE_SHIFT_S1);
    uint32_t numHeads = static_cast<uint32_t>(*(context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_N_INDEX)));
    uint32_t batchSize = static_cast<uint32_t>((context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX))->GetShapeSize());
    
    int64_t maxQSeqlen = 0;
    int64_t maxKVSeqlen = 0;
    if (context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_N_INDEX) == nullptr ||
        context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX) == nullptr) {
        // No TND sequence information, skipping max sequence validation
        maxQSeqlen = static_cast<int64_t>(pseShiftS1);
        maxKVSeqlen = static_cast<int64_t>(pseShiftS2);
    } else {
        auto actualQSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
        auto actualKvSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
        const int64_t *actualSeqQTnd = actualQSeq->GetData<int64_t>();
        const int64_t *actualSeqKvTnd = actualKvSeq->GetData<int64_t>();
        if (actualSeqQTnd != nullptr && actualSeqKvTnd != nullptr) {
            for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
                int64_t qSeqlen = *(actualSeqQTnd + batchIdx);
                int64_t kvSeqlen = *(actualSeqKvTnd + batchIdx);
                int64_t prevQSeqlenSum = batchIdx == 0 ? 0 : *(actualSeqQTnd + batchIdx - 1);
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if (context->GetOptionalInputShape(BLOCK_TABLE_INDEX) == nullptr) {
                    int64_t prevKvSeqlenSum = batchIdx == 0 ? 0 : *(actualSeqKvTnd + batchIdx - 1);
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
                if (qSeqlen > maxQSeqlen) {
                    maxQSeqlen = qSeqlen;
                }
                if (kvSeqlen > maxKVSeqlen) {
                    maxKVSeqlen = kvSeqlen;
                }
            }
        }
    }
    
    OP_CHECK_IF(
        (pseShiftBatch != 1 && pseShiftBatch != batchSize) || (pseShiftN != numHeads) ||
        (pseShiftS1 < maxQSeqlen) || (pseShiftS2 < maxKVSeqlen),
        OP_LOGW(context->GetNodeName(),
            "The shape of pse shift is (%u, %u, %u, %u), which does not match (B, N, S1, S2) or (1, N, S1, S2).",
            pseShiftBatch, pseShiftN, pseShiftS1, pseShiftS2),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAIMaskShape(const gert::TilingContext *context)
{
    auto tempAttnMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    auto maskDimNum = tempAttnMaskShape->GetStorageShape().GetDimNum();
    constexpr int64_t OPT_ATTEN_MASK_LEN = 2048;
    constexpr int64_t EFFECTIVE_CAUSAL_DIMS = 2;
    int64_t dimCountDown = maskDimNum;
    while (dimCountDown > 0) {
        int64_t revOrderDim = tempAttnMaskShape->GetStorageShape().GetDim(dimCountDown - 1);
        if (maskDimNum - dimCountDown < EFFECTIVE_CAUSAL_DIMS) {
            OP_CHECK_IF(revOrderDim != OPT_ATTEN_MASK_LEN,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "In split fuse senario, when sparseMode is 3, "
                    "the input mask has %ld dims in total, "
                    "maskDim %ld shall be 2048", maskDimNum, (dimCountDown - 1)),
                    return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(revOrderDim != 1,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "In split fuse senario, when sparseMode is 3, "
                    "the input mask has %ld dims in total, "
                    "maskDim %ld shall be 1", maskDimNum, (dimCountDown - 1)),
                    return ge::GRAPH_FAILED);
        }
        dimCountDown--;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAIMask(gert::TilingContext *context)
{
    auto tempAttnMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    auto attrs = context->GetAttrs();
    int32_t sparseMode = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX)));
    OP_CHECK_IF((sparseMode != 0) && (sparseMode != 3) && (sparseMode != 4),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "In split fuse senario, sparseMode shall be 0 or 3 or 4"),
            return ge::GRAPH_FAILED);
    if (tempAttnMaskShape == nullptr) {
        OP_CHECK_IF(sparseMode != 0,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "When attnMask is not provided, sparseMode must be 0"),
                return ge::GRAPH_FAILED);
    } else {
        auto maskDimNum = tempAttnMaskShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(sparseMode != 3 && sparseMode != 4,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "When attnMask is provided, sparseMode must be 3 or 4"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(maskDimNum != 2 && maskDimNum != 3 && maskDimNum != 4,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "When attnMask is provided, it must have 2 or 3 or 4 dims"),
                return ge::GRAPH_FAILED);
        if (CheckFAIMaskShape(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckFAILseOutput(const gert::TilingContext *context)
{
    bool lseFlag = *(context->GetAttrs()->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX));
    auto lseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    auto queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(((lseFlag != false) && (lseShape->GetStorageShape().GetDimNum() != 3U)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "When layout is TND, SoftmaxLse shape dim must be 3"), return ge::GRAPH_FAILED);
    auto t = queryShape->GetStorageShape().GetDim(DIM_0);
    auto n = queryShape->GetStorageShape().GetDim(DIM_1);
    auto lseDim0 = lseShape->GetStorageShape().GetDim(DIM_0);
    auto lseDim1 = lseShape->GetStorageShape().GetDim(DIM_1);
    auto lseDim2 = lseShape->GetStorageShape().GetDim(DIM_2);
    OP_CHECK_IF((lseFlag != false) && ((lseDim0 != t) || (lseDim1 != n) || (lseDim2 != 1U)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "When layout is TND, SoftmaxLse shape must be [T, N, 1]."
        "In this case it is [%ld, %ld, 1]",
        t, n), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFAIAvailability(gert::TilingContext *context)
{
    bool isPageAttention = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    if (CheckFAIQKV(context, isPageAttention) != ge::GRAPH_SUCCESS ||
        CheckFAIMask(context) != ge::GRAPH_SUCCESS ||
        CheckFAISinglePara(context, isPageAttention) != ge::GRAPH_SUCCESS ||
        CheckFAILseOutput(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static int64_t ClampTokenValue(int64_t token)
{
    if (token > SPARSE_MODE_INT_MAX) {
        return SPARSE_MODE_INT_MAX;
    } else if (token < -(SPARSE_MODE_INT_MAX)) {
        return -(SPARSE_MODE_INT_MAX);
    }
    return token;
}

static void SetPagedCacheParamsForFAI(gert::TilingContext *context, FAInferContext& faInfo)
{
    auto blockTable = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    faInfo.pagedCacheFlag = blockTable != nullptr;
    if (!faInfo.pagedCacheFlag) {
        return;
    }
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto tempK = context->GetInputShape(KEY_INDEX);
    auto tempV = context->GetInputShape(VALUE_INDEX);
    constexpr int64_t HEAD_SIZE_ALIGN_16 = 16;
    bool isHeadSizeAligned = (tempQ->GetStorageShape().GetDim(DIM_2) % HEAD_SIZE_ALIGN_16 == 0);
    if (tempK->GetStorageShape().GetDimNum() == 5U && tempV->GetStorageShape().GetDimNum() == 5U &&
        isHeadSizeAligned) {
        faInfo.kvcacheNzFlag = true;
    }
    faInfo.maxNumBlocksPerBatch = blockTable->GetStorageShape().GetDim(DIM_1);
}

static void SetMaskTypeForFAI(gert::TilingContext *context, FAInferContext& faInfo,
    const gert::StorageShape *pseShift, int32_t sparseMode)
{
    auto pseCheck = CheckFAIPseShift(context);
    if (pseShift == nullptr) {
        faInfo.maskType = sparseMode == DIM_4 ? MaskType::SWA_MASK : static_cast<MaskType>(sparseMode == DIM_3);
    } else if (pseShift != nullptr && pseCheck != ge::GRAPH_SUCCESS) {
        OP_LOGW(context->GetNodeName(), "Warning, make sure pseshift must be (B/1, N, S1, S2)");
        faInfo.maskType = sparseMode == DIM_4 ? MaskType::SWA_MASK : static_cast<MaskType>(sparseMode == DIM_3);
    } else if (pseShift != nullptr && pseCheck == ge::GRAPH_SUCCESS) {
        faInfo.maskType = MaskType::FULL_MASK;
        faInfo.pseQ = pseShift->GetStorageShape().GetDim(DIM_2);
        faInfo.pseKv = pseShift->GetStorageShape().GetDim(DIM_3);
    }
}

static void ComputeSeqLenAndDecodeFlagsForFAI(FAInferContext& faInfo, uint32_t aicoreNum,
    const int64_t *actualSeqQTnd, const int64_t *actualSeqKvTnd, int32_t batch)
{
    constexpr int64_t KV_ACTUAL_SEQ_LEN_1024 = 1024;
    constexpr int64_t QUERY_ACTUAL_SEQ_LEN_16 = 16;
    constexpr int64_t QUERY_ACTUAL_SEQ_LEN_0 = 0;
    constexpr int64_t GROUP_SIZE_128 = 128;
    constexpr int32_t DECODING_PAGE_BLOCK_SIZE_128 = 128;

    faInfo.qSeqlenList = actualSeqQTnd;
    faInfo.kvSeqlenList = actualSeqKvTnd;
    faInfo.isTilingSink = false;
    int64_t maxQSeqlen = 0;
    int64_t minQSeqlen = INT64_MAX;
    int64_t maxKVSeqlen = 0;
    for (int32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
        int64_t qSeqlen = *(actualSeqQTnd + batchIdx);
        int64_t kvSeqlen = *(actualSeqKvTnd + batchIdx);
        if (faInfo.layout == "TND") {
            if (batchIdx > 0) {
                int64_t prevQSeqlenSum = *(actualSeqQTnd + batchIdx - 1);
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if (!faInfo.pagedCacheFlag) {
                    int64_t prevKvSeqlenSum = *(actualSeqKvTnd + batchIdx - 1);
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
        }
        if (qSeqlen > maxQSeqlen) {
            maxQSeqlen = qSeqlen;
        }
        if (qSeqlen < minQSeqlen) {
            minQSeqlen = qSeqlen;
        }
        if (kvSeqlen > maxKVSeqlen) {
            maxKVSeqlen = kvSeqlen;
        }
    }
    uint32_t numTasks = faInfo.batch * faInfo.kvHeads;
    bool isLongSeq = (numTasks <= 0.8 * aicoreNum) && (maxKVSeqlen >= aicoreNum * 512);
    bool isShortSeq = (numTasks <= 0.4 * aicoreNum) && (maxKVSeqlen >= KV_ACTUAL_SEQ_LEN_1024);
    if ((faInfo.pagedCacheFlag) && !(faInfo.maskType == MaskType::FULL_MASK) &&
        !(faInfo.maskType == MaskType::SWA_MASK) &&
        (!faInfo.learnableSinkFlag) &&
        !(faInfo.innerPrecise == 1) &&
        (maxQSeqlen * (faInfo.numHeads / faInfo.kvHeads) <= GROUP_SIZE_128) &&
        (maxQSeqlen <= QUERY_ACTUAL_SEQ_LEN_16) &&
        (maxKVSeqlen >= KV_ACTUAL_SEQ_LEN_1024) &&
        (minQSeqlen > QUERY_ACTUAL_SEQ_LEN_0) &&
        (isLongSeq || isShortSeq)) {
        faInfo.flashDecodeFlag = true;
    }
    if (faInfo.pagedCacheFlag && maxQSeqlen == 1 && minQSeqlen == 1 && faInfo.maskType == MaskType::NO_MASK &&
        (batch >= aicoreNum) && (faInfo.blockSize == DECODING_PAGE_BLOCK_SIZE_128) &&
        !faInfo.lseFlag && !faInfo.learnableSinkFlag && (faInfo.innerPrecise == 0) &&
        ((faInfo.numHeads / faInfo.kvHeads) <= GROUP_SIZE_128)) {
        faInfo.decodingFlag = true;
    }
}

static ge::graphStatus ConvertContextToParamsFAI(gert::TilingContext *context, FAInferContext& faInfo, uint32_t aicoreNum)
{
    auto qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto actualQSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actualKvSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    auto pseShift = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    auto attrs = context->GetAttrs();

    SetPagedCacheParamsForFAI(context, faInfo);

    faInfo.numHeads = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX)));
    int32_t tmpNKv = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX)));
    int32_t tmpBlkSize = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX)));
    int32_t sparseMode = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX)));
    float scaleValue = *(attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX));
    faInfo.sparseMode = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX)));
    int64_t preToken  = *(attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX));
    int64_t nextToken = *(attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX));
    faInfo.preToken = ClampTokenValue(preToken);
    faInfo.nextToken = ClampTokenValue(nextToken);

    string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    bool lseFlag = *(attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX));
    bool learnableSinkFlag = context->GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr ? true : false;
    int32_t innerPrecise = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX)));
    auto tempK = context->GetInputShape(KEY_INDEX);
    faInfo.numBlocks = tempK->GetStorageShape().GetDim(DIM_0);
    faInfo.blockSize = tmpBlkSize;
    faInfo.kvHeads = tmpNKv;
    faInfo.scaleValue = scaleValue;
    faInfo.layout = inputLayoutStr;
    faInfo.lseFlag = lseFlag;
    faInfo.learnableSinkFlag = learnableSinkFlag;
    faInfo.innerPrecise = innerPrecise;

    if (faInfo.layout == "TND") {
        faInfo.embeddingSize = tempQ->GetStorageShape().GetDim(DIM_2);
        faInfo.embeddingSizeV = faInfo.embeddingSize;
    }

    SetMaskTypeForFAI(context, faInfo, pseShift, sparseMode);

    faInfo.dataType = static_cast<DataType>(qDataType == ge::DT_BF16);
    int32_t batch = actualQSeq->GetShapeSize();
    faInfo.batch = batch;
    const int64_t *actualSeqQTnd = actualQSeq->GetData<int64_t>();
    const int64_t *actualSeqKvTnd = actualKvSeq->GetData<int64_t>();
    if (actualSeqQTnd != nullptr && actualSeqKvTnd != nullptr) {
        ComputeSeqLenAndDecodeFlagsForFAI(faInfo, aicoreNum, actualSeqQTnd, actualSeqKvTnd, batch);
    } else {
        faInfo.isTilingSink = true;
    }
    faInfo.workspaces = context->GetWorkspaceSizes(1);

    auto keyBlockStrides = context->GetDynamicInputStride(KEY_INDEX, 0);
    if (keyBlockStrides != nullptr && keyBlockStrides->GetDimNum() > 0) {
        faInfo.keyBnStride = keyBlockStrides->GetStride(0);
    }
    auto valueBlockStrides = context->GetDynamicInputStride(VALUE_INDEX, 0);
    if (valueBlockStrides != nullptr && valueBlockStrides->GetDimNum() > 0) {
        faInfo.valueBnStride = valueBlockStrides->GetStride(0);
    }

    return ge::GRAPH_SUCCESS;
}

static bool CheckFAIDSizeNoPA(int64_t tempD, int64_t tempKD, int64_t tempVD)
{
    return (tempD <= 256 && tempKD <= 256 && tempVD <= 256) && // 256: FAI max D
           (tempD == tempKD && tempD == tempVD);
}

static bool CheckFAIDSizePA3Dim(int64_t tempD, const gert::Shape *tempKShape,
    const gert::Shape *tempVShape, int64_t kvHeadNum)
{
    int64_t tempKD = tempKShape->GetDim(DIM_2) / kvHeadNum;
    int64_t tempVD = tempVShape->GetDim(DIM_2) / kvHeadNum;
    int64_t blockSize = tempKShape->GetDim(DIM_1);
    constexpr int64_t BLOCK_SIZE_ALIGN_16 = 16;
    bool isFAIDSize = (tempD <= 256 && tempKD <= 256 && tempVD <= 256) &&
            (tempD == tempKD && tempD == tempVD);
    bool blockSizeSupported = (blockSize % BLOCK_SIZE_ALIGN_16 == 0) &&
            (blockSize <= MAX_BLOCK_SIZE);
    return isFAIDSize && blockSizeSupported;
}

static bool CheckFAIDSizePA5Dim(int64_t tempD, const gert::Shape *tempKShape,
    const gert::Shape *tempVShape)
{
    int64_t tempKD = tempKShape->GetDim(DIM_2) * 16;
    int64_t tempVD = tempVShape->GetDim(DIM_2) * 16;
    int64_t blockSize = tempKShape->GetDim(DIM_3);
    constexpr int64_t BLOCK_SIZE_ALIGN_16 = 16;
    bool isFAIDSize = (tempD <= 256U && tempKD <= 256 && tempVD <= 256) &&
            (tempD == tempKD && tempD == tempVD) && (blockSize % BLOCK_SIZE_ALIGN_16 == 0);
    // D=64 and D=128 are excluded as they use other optimized paths
    isFAIDSize = isFAIDSize && !(tempD == 64 || tempD == 128);
    bool blockSizeSupported = (blockSize % BLOCK_SIZE_ALIGN_16 == 0) &&
            (blockSize <= MAX_BLOCK_SIZE);
    return isFAIDSize && blockSizeSupported;
}

static bool IsUsingFAI(gert::TilingContext &context, const string inputLayoutStr, const int64_t tempD)
{
    bool isPageAttention = context.GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    auto tempAttnMaskShape = context.GetOptionalInputShape(ATTEN_MASK_INDEX);
    auto qDataType = context.GetInputDesc(QUERY_INDEX)->GetDataType();
    auto tempK = context.GetInputShape(KEY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto kvDimNum = tempK->GetStorageShape().GetDimNum();
    auto attrs = context.GetAttrs();
    int64_t headNum = *(attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX));
    int64_t kvHeadNum = *(attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX));
    int32_t sparseMode = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX)));
    int32_t innerPrecise = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX)));
    bool isLearnableSink = context.GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr ? true : false;
    bool isLearnableSinkFlag = true;
    constexpr int64_t QUERY_HEAD_DIM_64 = 64;
    if (isLearnableSink && inputLayoutStr == "TND") {
        auto tempQ = context.GetInputShape(QUERY_INDEX);
        int64_t tempQD = tempQ->GetStorageShape().GetDim(DIM_2);
        auto sinkDataType = context.GetOptionalInputDesc(LEARNABLE_SINK_INDEX)->GetDataType();
        if (tempQD == QUERY_HEAD_DIM_64 && sinkDataType == ge::DT_BF16) {
            isLearnableSinkFlag = false;
        }
    }

    auto qRope = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    auto kRope = context.GetOptionalInputTensor(KEY_ROPE_INDEX);
    bool isRopeSplitMla = (qRope != nullptr) && (kRope != nullptr);
    bool sparseModeSupported = (sparseMode == 0) || (sparseMode == 3) || (sparseMode == 4);
    bool isMha = (kvHeadNum == 0) || (headNum == kvHeadNum);
    bool mhaConditions = isMha && !((qDataType == ge::DT_BF16) && (innerPrecise == 1)) &&
        !((sparseMode == 0) && (tempAttnMaskShape != nullptr));
    bool nonMhaConditions = !isMha && (innerPrecise == 0);

    if (!(inputLayoutStr == "TND" && isLearnableSinkFlag && !isRopeSplitMla &&
        sparseModeSupported && (nonMhaConditions || mhaConditions))) {
        return false;
    }

    if (!isPageAttention) {
        return CheckFAIDSizeNoPA(tempD, tempK->GetStorageShape().GetDim(DIM_2),
                                 tempV->GetStorageShape().GetDim(DIM_2));
    } else if (kvDimNum == 3U) {
        return CheckFAIDSizePA3Dim(tempD, &tempK->GetStorageShape(), &tempV->GetStorageShape(), kvHeadNum);
    } else if (kvDimNum == 5U) {
        return CheckFAIDSizePA5Dim(tempD, &tempK->GetStorageShape(), &tempV->GetStorageShape());
    }
    return false;
}

static ge::graphStatus TilingProcess4SplitFuse(gert::TilingContext *context)
{
    OP_CHECK_IF(CheckFAIAvailability(context) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Split fuse condition check failed"),
        return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "PlatformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t aicoreNum = ascendcPlatform.GetCoreNumAic();
    FAInferTilingData faiTilingData;
    FAInferContext faiContext;
    ConvertContextToParamsFAI(context, faiContext, aicoreNum);
    FAInferTiling fai_tiling(faiContext);
    fai_tiling.SetCoreNum(aicoreNum);
    auto ret = fai_tiling.DoTiling(faiTilingData);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Do fai tiling went wrong"),
        return ge::GRAPH_FAILED);
    faiTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faiTilingData.GetDataSize());
    faiContext.workspaces[0] = 16U * 1024U * 1024U +
        static_cast<uint64_t>(fai_tiling.GetCoreNum()) * WORKSPACE_BLOCK_SIZE_DB * 4U * 3U * 4U + static_cast<uint64_t>(faiTilingData.get_splitLseTotalSize()) + static_cast<uint64_t>(faiTilingData.get_splitOTotalSize());
    if (faiContext.flashDecodeFlag) {
        auto needCoreNum = faiTilingData.get_needCoreNum();
        if (needCoreNum == 0) {
            context->SetBlockDim(fai_tiling.GetCoreNum());
        } else {
            context->SetBlockDim(needCoreNum);
        }
    } else {
        context->SetBlockDim(fai_tiling.GetCoreNum());
    }
    context->SetTilingKey(fai_tiling.GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

bool IsGqaIfa(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryS, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr) {
        return false;
    }
    if ((inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND")) {
        if (queryD == 512) { // 512: qD need 512
            return true;
        }
    }
    else if ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")) {
        if (queryS == 1) {
            return true;
        }
    }
    return false;
}

int64_t GetTndQueryS(gert::TilingContext &context)
{
    auto queryShape = context.GetInputShape(QUERY_INDEX);
    auto actualSeqlenthsQ = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actualSeqlenthsKv = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    auto blockTable = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    int64_t batchSize = blockTable->GetStorageShape().GetDim(DIM_0);
    const int64_t *actualSeqQ = actualSeqlenthsQ->GetData<int64_t>();
    const int64_t *actualSeqKv = actualSeqlenthsKv->GetData<int64_t>();

    if (batchSize == 0) {
        return 0;
    }
    if (actualSeqQ == nullptr || actualSeqKv == nullptr) { // tiling下沉场景
        int64_t queryT4Tnd = queryShape->GetStorageShape().GetDim(DIM_0);
        return (queryT4Tnd + batchSize - 1) / batchSize;
    }
    int64_t qActualSeqMax = 0;
    for (int64_t i = 0; i < batchSize; i++) {
        int64_t tmpS1 = (i == 0) ? actualSeqQ[0] : (actualSeqQ[i] - actualSeqQ[i - 1U]);
        if (tmpS1 > qActualSeqMax) {
            qActualSeqMax = tmpS1;
        }
    }
    return qActualSeqMax;
}

bool IsGqaMtp(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryS, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr) {
        return false;
    }
    bool isIFALayout = (inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")
            || (inputLayoutStr == "TND");
    if (!isIFALayout) {
        return false;
    }
    auto tempK = context.GetInputShape(KEY_INDEX);
    bool isNz = (tempK->GetStorageShape().GetDimNum() == 5);
    if (!isNz) {
        return false;
    }

    int64_t actualQueryS = queryS;
    if (inputLayoutStr == "TND") {
        actualQueryS = GetTndQueryS(context);
    }
    if (!(actualQueryS >= 1 && actualQueryS <= 16)) { // 16: mtp
        return false;
    }

    if (inputLayoutStr == "TND") {
        if (queryD != 128) { // 128: queryD need 128 when gqa kv_nz
            return false;
        }
        if (context.GetInputDesc(QUERY_INDEX)->GetDataType() != ge::DT_BF16 ||
            context.GetInputDesc(KEY_INDEX)->GetDataType() != ge::DT_INT8 ||
            context.GetInputDesc(VALUE_INDEX)->GetDataType() != ge::DT_INT8) {
            return false;
        }
    }
    return true;
}

bool IsAtbIfa(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryD)
{
    bool isIsAtbIfaLayout = (inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND");
    if (!isIsAtbIfaLayout) {
        return false;
    }
    if (!((context.GetInputDesc(KEY_INDEX)->GetDataType() == ge::DT_INT8) &&
        (context.GetInputDesc(QUERY_INDEX)->GetDataType() != ge::DT_INT8))) { // KV antiquant
        return false;
    }
    if (context.GetOptionalInputShape(BLOCK_TABLE_INDEX) == nullptr) { // PA
        return false;
    }
    if (queryD == 512) { // 512: max D limix
        return false;
    }
    bool isAntiquantParamFloat = (context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX) != nullptr) &&
                                 (context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX) != nullptr) &&
                                 (context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX)->GetDataType() == ge::DT_FLOAT) &&
                                 (context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX)->GetDataType() == ge::DT_FLOAT);
    if (!isAntiquantParamFloat) {
        return false;
    }
    return true;
}

bool IsMlaIfaOrMtp(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryS, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) == nullptr) { // mla
        return false;
    }
    if ((inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND")) {
        if (queryD == 512) { // 512: qD need 512
            return true;
        }
    } else if ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")) {
        if (queryS == 1)  {
            return true;
        }
        if ((queryS > 1 && queryS <= 32) && (queryD == 512)) { // 16: mtp; 512: qD need 512
            return true;
        }
    }
    return false;
}

bool IsSlidingAttention(gert::TilingContext &context, const string inputLayoutStr, const int64_t queryD)
{
    if (context.GetOptionalInputTensor(QUERY_ROPE_INDEX) == nullptr) { // mla
        return false;
    }
    if ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD")) {
        if (queryD != 512) { // 512: qD need 512
            return false;
        }
        if (*context.GetAttrs()->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX) == 4) { // 4: sparseMode =4
            return true;
        }
    }
    return false;
}

static bool IsUsingIFA(gert::TilingContext &context, const string inputLayoutStr, const uint32_t queryD, 
    const int64_t queryS)
{
    if (IsGqaIfa(context, inputLayoutStr, queryS, queryD) || 
        IsGqaMtp(context, inputLayoutStr, queryS, queryD) || 
        IsAtbIfa(context, inputLayoutStr, queryD) || 
        IsMlaIfaOrMtp(context, inputLayoutStr, queryS, queryD) ||
        IsSlidingAttention(context, inputLayoutStr, queryD)) {
        return true;
    }
    return false;
}

static ge::graphStatus TilingProcess4IFA(gert::TilingContext *context)
{
    // IFA tiling path
    IncreFlashAttentionContext ifaContext {};
    auto ret = ConvertContextToParamsIFA(*context, ifaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occored while convert tilingContext to ifa context");
        return ret;
    }
    IFATiling ifaTiling(context);
    return ifaTiling.DoSubOpTiling(ifaContext);
}

static ge::graphStatus CheckQKV(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempK = context.GetInputShape(KEY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_IF((tempQ == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempK == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempOut == nullptr),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Attention_Out is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == 0) && (tempOut->GetStorageShape().GetShapeSize() != 0),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
               "Query head should not be 0, or when attentionOut is not empty tensor, query input shoud not be empty tensor!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input dims are invalid!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempK->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input dims are invalid!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input dims are invalid!"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInDim3(const gert::TilingContext *context, const string &outputLayoutStr, const gert::Shape outShape,
    const gert::Shape exceptOutShape)
{
    OP_CHECK_IF((outShape.GetDimNum() != DIM_NUM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "OutputLayout is %s, Attention out shape dim should be 3, but got %zu!",
            outputLayoutStr.c_str(), outShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF((exceptOutShape != outShape),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "Expect outputLayout is %s and Out shape size[%ld, %ld, %ld] does NOT match "
                                    "Attention Out shape size[%ld, %ld, %ld]!",
                                    outputLayoutStr.c_str(),
                                    exceptOutShape.GetDim(DIM_0),
                                    exceptOutShape.GetDim(DIM_1),
                                    exceptOutShape.GetDim(DIM_2),
                                    outShape.GetDim(DIM_0),
                                    outShape.GetDim(DIM_1),
                                    outShape.GetDim(DIM_2)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInDim4(const gert::TilingContext *context, const string &outputLayoutStr, const gert::Shape outShape,
    const gert::Shape exceptOutShape)
{
    OP_CHECK_IF((outShape.GetDimNum() != DIM_NUM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "OutputLayout is %s, Attention out shape dim should be 4, but got %zu!",
            outputLayoutStr.c_str(), outShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF((exceptOutShape != outShape),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                    "Expect outputLayout is %s and Out shape size[%ld, %ld, %ld, %ld] does NOT match "
                                    "Attention Out shape size[%ld, %ld, %ld, %ld]!",
                                    outputLayoutStr.c_str(),
                                    exceptOutShape.GetDim(DIM_0),
                                    exceptOutShape.GetDim(DIM_1),
                                    exceptOutShape.GetDim(DIM_2),
                                    exceptOutShape.GetDim(DIM_3),
                                    outShape.GetDim(DIM_0),
                                    outShape.GetDim(DIM_1),
                                    outShape.GetDim(DIM_2),
                                    outShape.GetDim(DIM_3)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShape(gert::TilingContext *context, const string &outputLayoutStr, const gert::Shape outShape, const gert::Shape qkvShapeInfo)
{
    int64_t b = qkvShapeInfo.GetDim(DIM_0);
    int64_t queryN = qkvShapeInfo.GetDim(DIM_1);
    int64_t queryS = qkvShapeInfo.GetDim(DIM_2);
    int64_t queryT = qkvShapeInfo.GetDim(DIM_3);
    int64_t valueD = qkvShapeInfo.GetDim(DIM_4);
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (outputLayoutStr == "NSD") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{queryN, queryS, valueD});
    } else if (outputLayoutStr == "BSH") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{b, queryS, valueD * queryN});
    }else if (outputLayoutStr == "NBSD") {
        ret = CheckOutShapeInDim4(context, outputLayoutStr, outShape, gert::Shape{queryN, b, queryS, valueD});
    } else if (outputLayoutStr == "BSND") {
        ret = CheckOutShapeInDim4(context, outputLayoutStr, outShape, gert::Shape{b, queryS, queryN, valueD});
    } else if (outputLayoutStr == "BNSD") {
        ret = CheckOutShapeInDim4(context, outputLayoutStr, outShape, gert::Shape{b, queryN, queryS, valueD});
    } else if (outputLayoutStr == "NTD") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{queryN, queryT, valueD});
    } else if (outputLayoutStr == "TND") {
        ret = CheckOutShapeInDim3(context, outputLayoutStr, outShape, gert::Shape{queryT, queryN, valueD});
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Not support outputLayout(%s)",
        outputLayoutStr.c_str());
        ret = ge::GRAPH_FAILED;
    }
    return ret;
}

static ge::graphStatus GetB(const gert::TilingContext *context, const string inputLayoutStr, int64_t &b)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD") {
        b = 1;
    } else {
        b = tempQ->GetStorageShape().GetDim(DIM_0);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryN(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryN)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "NTD_TND") {
        queryN = tempQ->GetStorageShape().GetDim(DIM_0);
    } else if (inputLayoutStr == "BSND_NBSD" || 
        inputLayoutStr == "BSND") {
        queryN = tempQ->GetStorageShape().GetDim(DIM_2);
    } else if (inputLayoutStr == "BSH" || 
        inputLayoutStr == "BSH_NBSD") {
        auto attrs = context->GetAttrs();
        int64_t numHeads = *attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
        queryN = numHeads;
    } else {
        queryN = tempQ->GetStorageShape().GetDim(DIM_1);
    }
    OP_CHECK_IF(queryN == 0, 
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Q numhead is 0!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryS(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryS)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "BSH" || 
        inputLayoutStr == "BSH_NBSD" || 
        inputLayoutStr == "BSND_NBSD" || 
        inputLayoutStr == "BSND") {
        queryS = tempQ->GetStorageShape().GetDim(DIM_1);
    } else if (inputLayoutStr == "BNSD_BSND" || 
        inputLayoutStr == "BNSD_NBSD" || 
        inputLayoutStr == "BNSD") {
        queryS = tempQ->GetStorageShape().GetDim(DIM_2);
    } else {
        int64_t queryT = 0;
        if (inputLayoutStr == "NTD_TND") {
            queryT = tempQ->GetStorageShape().GetDim(DIM_1);
        } else {
            queryT = tempQ->GetStorageShape().GetDim(DIM_0);
        }
        queryS = queryT;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryT(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryT)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD") {
        queryT = tempQ->GetStorageShape().GetDim(DIM_0);
    } else if (inputLayoutStr == "NTD_TND") {
        queryT = tempQ->GetStorageShape().GetDim(DIM_1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryD(const gert::TilingContext *context, const string inputLayoutStr, int64_t &queryD)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD" || 
        inputLayoutStr == "NTD_TND") {
        queryD = tempQ->GetStorageShape().GetDim(DIM_2);
    } else if (inputLayoutStr == "BNSD_BSND" || 
            inputLayoutStr == "BNSD_NBSD"  || 
            inputLayoutStr == "BNSD"  || 
            inputLayoutStr == "BSND_NBSD" || 
            inputLayoutStr == "BSND") {
        queryD = tempQ->GetStorageShape().GetDim(DIM_3);
    } else {
        int64_t queryH = tempQ->GetStorageShape().GetDim(DIM_2);
        auto attrs = context->GetAttrs();
        int64_t numHeads = *attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
        queryD = queryH / numHeads;
    }
    const int64_t maxDlimit = 512;
    OP_CHECK_IF((queryD > maxDlimit), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "D should be less than or equal to 512 of Q shape! but now D = %ld. "
        "When layout is BNSD, D is the last dimension of Q shape, and layout is BSH, D = h / n", queryD),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPAValueD(const gert::TilingContext *context, int64_t &valueD)
{
    auto tempV = context->GetInputShape(VALUE_INDEX);
    if (tempV->GetStorageShape().GetDimNum() == DIM_BSH) { // BnBsH
        auto attrs = context->GetAttrs();
        int64_t numKvHeads = *attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
        if (numKvHeads == 0) {
            numKvHeads = *attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
        }
        valueD = tempV->GetStorageShape().GetDim(DIM_2) / numKvHeads;
    } else if (tempV->GetStorageShape().GetDimNum() == DIM_BNSD_OR_BSND) { // BnNBsD
        valueD = tempV->GetStorageShape().GetDim(DIM_3);
    } else if (tempV->GetStorageShape().GetDimNum() == NUM5) { // NZ
        valueD = tempV->GetStorageShape().GetDim(DIM_2) * tempV->GetStorageShape().GetDim(DIM_4);
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),"PagedAttention not support Value DimNum is %zu.\n",
            tempV->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetValueD(gert::TilingContext *context, const string inputLayoutStr, int64_t &valueD, bool isPageAttention)
{
    if (isPageAttention) {
        return GetPAValueD(context, valueD);
    }
    auto tempV = context->GetInputShape(VALUE_INDEX);
    if (inputLayoutStr == "NSD" || 
        inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD" || 
        inputLayoutStr == "NTD_TND") {
        OP_CHECK_IF((tempV->GetStorageShape().GetDimNum() != DIM_NUM_3),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "When block_table is null and input_layout is %s, dim number of key/value should be 3, but it is %zu.\n",
                    inputLayoutStr.c_str(), tempV->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        valueD = tempV->GetStorageShape().GetDim(DIM_2);
    } else if (inputLayoutStr == "BNSD_BSND" || 
        inputLayoutStr == "BNSD_NBSD"  || 
        inputLayoutStr == "BNSD"  || 
        inputLayoutStr == "BSND_NBSD" || 
        inputLayoutStr == "BSND") {
        OP_CHECK_IF((tempV->GetStorageShape().GetDimNum() != DIM_NUM_4),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                    "When block_table is null and input_layout is %s, dim number of key/value should be 4, but it is %zu.\n",
                    inputLayoutStr.c_str(), tempV->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        valueD = tempV->GetStorageShape().GetDim(DIM_3);
    } else {
        int64_t valueH = tempV->GetStorageShape().GetDim(DIM_2);
        auto attrs = context->GetAttrs();
        int64_t numKvHeads = *attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
        if (numKvHeads == 0) {
            numKvHeads = *attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
        }
        valueD = valueH / numKvHeads;
    }
    return ge::GRAPH_SUCCESS;
}

string GetOutputLayoutStr(const string &inputLayoutStr)
{
    size_t underLinePos = inputLayoutStr.find_last_of('_');
    if (underLinePos == std::string::npos) {
        return inputLayoutStr;
    }
    return inputLayoutStr.substr(underLinePos + 1);
}

static ge::graphStatus IsMla(const gert::TilingContext *context)
{
    auto qRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    auto kRope = context->GetOptionalInputTensor(KEY_ROPE_INDEX);
    OP_CHECK_IF((qRope != nullptr && kRope == nullptr),
        OP_LOGE(context->GetNodeName(), "keyRope is null, but queryRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((qRope == nullptr && kRope != nullptr),
        OP_LOGE(context->GetNodeName(), "queryRope is null, but keyRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputLayoutMla(const gert::TilingContext *context, const string inputLayoutStr, const int64_t queryD, const bool isPageAttention)
{
    // MLA support layout
    bool isIfaMlaLayout = (inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") ||
        (inputLayoutStr == "BSND") || (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") ||
        (inputLayoutStr == "BSH_NBSD") || (inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND");
    OP_CHECK_IF((!isIfaMlaLayout && (inputLayoutStr != "NTD_TND")),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "inputlayout(%s) only support {BSH, BNSD, BSND, TND, BNSD_NBSD, "
                                                            "BSND_NBSD, BSH_NBSD, TND_NTD} in mla.", inputLayoutStr.c_str()),
        return ge::GRAPH_FAILED);

    if (inputLayoutStr == "TND") {
        bool ifaWithoutPA = ((queryD == 512)&& (!isPageAttention));
        OP_CHECK_IF(ifaWithoutPA,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Layout is %s, MLA enabled, PA must be enabled when query's D dimension is %ld.",
            inputLayoutStr.c_str(), queryD),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputLayout(gert::TilingContext *context, const string inputLayoutStr,const int64_t queryS, const int64_t queryD, const bool isPageAttention)
{
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto tempK = context->GetInputShape(KEY_INDEX);
    auto tempV = context->GetInputShape(VALUE_INDEX);
    // mla check
    auto qRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    if (IsMla(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qRope != nullptr) {
        if (CheckInputLayoutMla(context, inputLayoutStr, queryD, isPageAttention) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (inputLayoutStr == "BNSD_BSND") {
        OP_CHECK_IF((queryS == 1), 
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),"BNSD_BSND layout is not supported when S is 1!"),
            return ge::GRAPH_FAILED);
    }
    if (inputLayoutStr == "SH") {
        OP_LOGE(context->GetNodeName(), "SH layout is not supported!");
        return ge::GRAPH_FAILED;
    }
    // check Q DimNum
    if (inputLayoutStr == "BSH" || inputLayoutStr == "BSH_NBSD" || inputLayoutStr == "TND" || 
        inputLayoutStr == "TND_NTD" || inputLayoutStr == "NTD_TND" || inputLayoutStr == "NSD") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != DIM_NUM_3),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is %s, queryDims must be 3! but actual value is %zu.\n",
            inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    }
    else {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != DIM_NUM_4),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is %s, queryDims must be 4! but actual value is %zu.\n",
            inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    }
    // check KV DimNum when PA
    if (isPageAttention == false) {
        if (inputLayoutStr == "TND" || inputLayoutStr == "NTD_TND") {
            OP_CHECK_IF(((tempK->GetStorageShape().GetDimNum() != DIM_NUM_3) || (tempV->GetStorageShape().GetDimNum() != DIM_NUM_3)),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is %s, key and value dims must be %zu! but actual value is keydim(%zu) valuedim(%zu).\n",
                inputLayoutStr.c_str(), DIM_NUM_3, tempK->GetStorageShape().GetDimNum(), tempV->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (RouteToFia(context)) {
        return TilingFusedInferAttentionScoreV3(context);
    }
    
    OP_CHECK_IF(CheckQKV(*context) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check query/key/value failed"), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Attributes returned from GetAttrs() is a nullptr"), return ge::GRAPH_FAILED);
    const string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    // 获取关键轴信息
    int64_t b = 0;
    int64_t queryN = 0;
    int64_t queryS = 0;
    int64_t queryT = 0;
    int64_t queryD = 1;
    int64_t valueD = 0;
    bool isPageAttention = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    if ((GetB(context, inputLayoutStr, b) != ge::GRAPH_SUCCESS) ||
        (GetQueryN(context, inputLayoutStr, queryN) != ge::GRAPH_SUCCESS) ||
        (GetQueryS(context, inputLayoutStr, queryS) != ge::GRAPH_SUCCESS) ||
        (GetQueryT(context, inputLayoutStr, queryT) != ge::GRAPH_SUCCESS) ||
        (GetValueD(context, inputLayoutStr, valueD, isPageAttention) != ge::GRAPH_SUCCESS) ||
        (GetQueryD(context, inputLayoutStr, queryD) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    // 校验intput
    OP_CHECK_IF(CheckInputLayout(context, inputLayoutStr, queryS, queryD, isPageAttention) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check InputLayout failed"), return ge::GRAPH_FAILED);
    // 校验OutShape
    string outputLayoutStr = GetOutputLayoutStr(inputLayoutStr);
    auto outShape = context->GetOutputShape(ATTENTION_OUT_INDEX)->GetStorageShape();
    gert::Shape qkvShapeInfo{b, queryN, queryS, queryT, valueD};
    OP_CHECK_IF(CheckOutShape(context, outputLayoutStr, outShape, qkvShapeInfo) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check output shape failed"), return ge::GRAPH_FAILED);
    // 是否路由到IFA
    bool usingIFA = IsUsingIFA(*context, inputLayoutStr, queryD, queryS);
    bool usingFAI = IsUsingFAI(*context, inputLayoutStr, queryD);
    if (usingFAI) {
        OP_CHECK_IF(TilingProcess4SplitFuse(context) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process for split fuse failed"),
            return ge::GRAPH_FAILED);
    } else if (usingIFA) { // IFA tiling process
        OP_CHECK_IF(TilingProcess4IFA(context) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process for ifa failed"),
            return ge::GRAPH_FAILED);
    } else { // PFA tiling process
        OP_CHECK_IF(TilingProcess4PFA(context, static_cast<uint32_t>(queryD), b, queryS, queryN) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process for pfa failed"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("FusedInferAttentionScore", "Tiling context is null."),
        return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        return TilingFusedInferAttentionScoreV4(context);
    } else if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_5102) {
        return TilingFusedInferAttentionScoreArch38(context);
    } else {
        return TilingFusedInferAttentionScore(context);
    }
    return ge::GRAPH_SUCCESS;
}

extern "C" {
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context)
{
    return TilingIncreFlashAttention(context);
}
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(
    gert::TilingContext *context)
{
    return DoOpTilingFusedInferAttentionScore(context);
}
}
} // namespace optiling
