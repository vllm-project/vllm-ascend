/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FUSED_GDN_DECODE_TILING_H
#define FUSED_GDN_DECODE_TILING_H

#include "register/op_impl_registry.h"

namespace optiling {

struct FusedGdnDecodeCompileInfo {};

ge::graphStatus FusedGdnDecodeTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForFusedGdnDecode(gert::TilingParseContext *context);

} // namespace optiling

#endif // FUSED_GDN_DECODE_TILING_H
