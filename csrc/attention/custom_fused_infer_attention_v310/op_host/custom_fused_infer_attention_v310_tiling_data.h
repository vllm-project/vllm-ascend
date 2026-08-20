/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file custom_fused_infer_attention_v310_tiling_data.h
 * \brief
 */
#ifndef ADN_FUSED_INFER_ATTENTION_TILING_DATA_H_
#define ADN_FUSED_INFER_ATTENTION_TILING_DATA_H_

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(IncreFlashAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, headSize)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, totalBlockNum)
TILING_DATA_FIELD_DEF(uint32_t, querySeqStep)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionBaseParamsOp, IncreFlashAttentionBaseParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, maskHeadStride)
TILING_DATA_FIELD_DEF(uint32_t, maskBatchStride)
TILING_DATA_FIELD_DEF(uint32_t, maskKvLen)
TILING_DATA_FIELD_DEF(uint32_t, qTokens)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, startTaskId);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, endTaskId);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, startBatch);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, endBatch);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSplitCoreParamsOp, IncreFlashAttentionSplitCoreParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataV2)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParams, tilingBase);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitCoreParams, tilingPerCore);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CustomFusedInferAttentionV310, IncreFlashAttentionTilingDataV2)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingAtbDataV2)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParams, tilingBase);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitCoreParams, tilingPerCore);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CustomFusedInferAttentionV310_30000000000200000, IncreFlashAttentionTilingAtbDataV2)
REGISTER_TILING_DATA_CLASS(CustomFusedInferAttentionV310_30000000000200001, IncreFlashAttentionTilingAtbDataV2)

} // namespace optiling
#endif // ADN_FUSED_INFER_ATTENTION_TILING_DATA_H_
