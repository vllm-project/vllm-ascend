/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_tiling_arch38.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_ARCH38_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_ARCH38_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../common/op_host/fia_tiling_base.h"
#include "incre_flash_attention_tiling_context.h"

namespace optiling {
namespace arch38 {

class IFATilingArch38 : public FiaTilingBase {
 public:
  IFATilingArch38(gert::TilingContext *context) : FiaTilingBase(context) {}
  ~IFATilingArch38() override = default;
  void InitTilingInfo(TilingInfo *tilingInfo) override {}
  bool IsCapable() override {return true;}
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoSubOpTiling(IncreFlashAttentionContext& ifaContext);
  static ge::graphStatus ConvertContext(gert::TilingContext& context, IncreFlashAttentionContext& ifaContext);
};

}  // namespace arch38 
}  // namespace optiling 
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_