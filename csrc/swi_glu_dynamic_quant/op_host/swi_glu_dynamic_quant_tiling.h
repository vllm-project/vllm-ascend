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
 * \file swi_glu_dynamic_quant_tiling.h
 * \brief
 */

 #ifndef SWI_GLU_DYNAMIC_QUANT_TILING_H
 #define SWI_GLU_DYNAMIC_QUANT_TILING_H

 #include "register/tilingdata_base.h"
 #include "register/op_impl_registry.h"
 #include "util/math_util.h"
 #include "log/log.h"
 #include "tiling/platform/platform_ascendc.h"
 #include "platform/platform_infos_def.h"

 namespace optiling {
 BEGIN_TILING_DATA_DEF(SwiGluDynamicQuantTilingData)
     TILING_DATA_FIELD_DEF(uint32_t, groupLen);
     TILING_DATA_FIELD_DEF(uint32_t, rowLen);
     TILING_DATA_FIELD_DEF(uint32_t, colLen);
     TILING_DATA_FIELD_DEF(uint32_t, rowLenPerHeadCore);
     TILING_DATA_FIELD_DEF(uint32_t, rowLenPerTailCore);
     TILING_DATA_FIELD_DEF(uint32_t, basicRowLenHeadCore);
     TILING_DATA_FIELD_DEF(uint32_t, basicRowLenTailCore);
     TILING_DATA_FIELD_DEF(uint32_t, basicColLen);
     TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);
     TILING_DATA_FIELD_DEF(uint32_t, realCoreNum);
     TILING_DATA_FIELD_DEF(uint32_t, activateLeft);
     TILING_DATA_FIELD_DEF(int64_t, groupListType);
     TILING_DATA_FIELD_DEF(int64_t, dstType);
     TILING_DATA_FIELD_DEF(int64_t, hasGroup);
     TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
 END_TILING_DATA_DEF;

 REGISTER_TILING_DATA_CLASS(SwiGluDynamicQuant, SwiGluDynamicQuantTilingData)


 struct SwiGluDynamicQuantCoreCompileInfo {};

 struct SwiGluDynamicQuantCompileInfo {
     uint32_t totalCore = 1;
     uint32_t ubSize = 0;
     uint32_t inputDataByte = 2;
     uint32_t groupLength = 1;
     std::string curQuantMode = "dynamic";
     uint32_t activateLeft = 0;
     int64_t groupListType = 0;
     int64_t dstType = ge::DT_INT8;
     int64_t hasGroup = 0;

     uint32_t dataNumSingleUb = 1;
     uint32_t block_num = 1;
     uint32_t cacheLineLen = 1;
     bool isPerTensor = true;
 };

 struct SwiGluDynamicQuantTilingParam {
     uint32_t optBaseRowLenHeadCore = 1;
     uint32_t optBaseRowLenTailCore = 1;
     uint32_t optBaseColLen = 1;
     uint32_t rowLenPerHeadCore = 0;
     uint32_t rowLenPerTailCore = 0;
     uint32_t headCoreNum = 0;
     uint32_t coreNumUsed = 0;
 };

 enum class SwiGluDynamicQuantQuantMode : uint8_t {
    STATIC_PER_TENSOR = 0,
    STATIC_PER_CHANNEL,
    DYNAMIC
 };
 } // namespace optiling
 #endif // SWI_GLU_DYNAMIC_QUANT_TILING_H
