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
 * \file ffn_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FFNBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
TILING_DATA_FIELD_DEF(uint32_t, k1);
TILING_DATA_FIELD_DEF(uint32_t, n1);
TILING_DATA_FIELD_DEF(uint32_t, n2);
TILING_DATA_FIELD_DEF(uint32_t, expertNum);
TILING_DATA_FIELD_DEF(uint32_t, maxTokens);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint64_t, workspace1Size);
TILING_DATA_FIELD_DEF(uint64_t, workspace2Size);
TILING_DATA_FIELD_DEF(uint32_t, syncWorkspaceSize);
TILING_DATA_FIELD_DEF(uint32_t, dataTypeSize);
TILING_DATA_FIELD_DEF(uint32_t, scale1GroupNum);
TILING_DATA_FIELD_DEF(uint32_t, scale2GroupNum);
TILING_DATA_FIELD_DEF(uint32_t, tokensIndexFlag);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFNBaseParamsOp, FFNBaseParams)

BEGIN_TILING_DATA_DEF(FFNSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, baseM1);
TILING_DATA_FIELD_DEF(uint32_t, baseN1);
TILING_DATA_FIELD_DEF(uint32_t, baseN2);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(uint32_t, mm1ResUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFNSingleCoreParamsOp, FFNSingleCoreParams)

BEGIN_TILING_DATA_DEF(FFNTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FFNSingleCoreParams, ffnSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(FFNBaseParams, ffnBaseParams);
// ---- arch35 (ascend950) fused 路径字段：两个 matmul 的紧凑 tiling + hidden/workspace 信息 ----
TILING_DATA_FIELD_DEF(uint32_t, upUsedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, upM);
TILING_DATA_FIELD_DEF(uint32_t, upN);
TILING_DATA_FIELD_DEF(uint32_t, upK);
TILING_DATA_FIELD_DEF(uint32_t, upML1);
TILING_DATA_FIELD_DEF(uint32_t, upNL1);
TILING_DATA_FIELD_DEF(uint32_t, upKL1);
TILING_DATA_FIELD_DEF(uint32_t, upBaseM);
TILING_DATA_FIELD_DEF(uint32_t, upBaseN);
TILING_DATA_FIELD_DEF(uint32_t, upBaseK);
TILING_DATA_FIELD_DEF(uint32_t, upSkSingleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, upMTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, upNTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, upMBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, upNBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, upMTailMain);
TILING_DATA_FIELD_DEF(uint32_t, upNTailMain);
TILING_DATA_FIELD_DEF(uint8_t, upIsHf32);
TILING_DATA_FIELD_DEF(uint8_t, upL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, upL0cDB);
TILING_DATA_FIELD_DEF(uint8_t, upUbDB);
TILING_DATA_FIELD_DEF(uint32_t, downUsedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, downM);
TILING_DATA_FIELD_DEF(uint32_t, downN);
TILING_DATA_FIELD_DEF(uint32_t, downK);
TILING_DATA_FIELD_DEF(uint32_t, downML1);
TILING_DATA_FIELD_DEF(uint32_t, downNL1);
TILING_DATA_FIELD_DEF(uint32_t, downKL1);
TILING_DATA_FIELD_DEF(uint32_t, downBaseM);
TILING_DATA_FIELD_DEF(uint32_t, downBaseN);
TILING_DATA_FIELD_DEF(uint32_t, downBaseK);
TILING_DATA_FIELD_DEF(uint32_t, downSkSingleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, downMTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, downNTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, downMBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, downNBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, downMTailMain);
TILING_DATA_FIELD_DEF(uint32_t, downNTailMain);
TILING_DATA_FIELD_DEF(uint8_t, downIsHf32);
TILING_DATA_FIELD_DEF(uint8_t, downL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, downL0cDB);
TILING_DATA_FIELD_DEF(uint8_t, downUbDB);
TILING_DATA_FIELD_DEF(uint32_t, hiddenOffset);
TILING_DATA_FIELD_DEF(uint32_t, hiddenRows);
TILING_DATA_FIELD_DEF(uint32_t, hiddenCols);
TILING_DATA_FIELD_DEF(uint8_t, isFp16); // 0: x/weight bf16; 1: x/weight fp16（arch35 新增 fp16 支持）
TILING_DATA_FIELD_DEF(uint8_t, biasIsBf16); // 0: bias float32; 1: bias bf16（arch35 与 linear 对齐）
TILING_DATA_FIELD_DEF(uint8_t, biasIsFp16); // 0: 非 fp16 bias; 1: bias fp16（与 biasIsBf16 互斥）
TILING_DATA_FIELD_DEF(uint8_t, hasBias); // 0: 无 bias（swiglu 模型场景）；1: 有 bias
TILING_DATA_FIELD_DEF(uint8_t, transB); // 0: 权重 canonical [K,N]；1: 权重 linear [N,K]，kernel 内 transB
TILING_DATA_FIELD_DEF(uint8_t, swigluSingle); // swiglu 单 matmul（N=2H，B 装载交错）；0 回退旧三段
TILING_DATA_FIELD_DEF(uint8_t, upFullLoad); // up 的 L1 全载模式（0/1/2），host 官方资格判定后透传 kernel
TILING_DATA_FIELD_DEF(uint8_t, downFullLoad); // down 的 L1 全载模式（0/1/2）
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFN, FFNTilingData)
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_