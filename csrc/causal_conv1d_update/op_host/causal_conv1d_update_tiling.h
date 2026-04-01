/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update_tiling.h
 * \brief
 */

#ifndef CAUSAL_CONV1D_UPDATE_TILING_H
#define CAUSAL_CONV1D_UPDATE_TILING_H

#include "register/tilingdata_base.h"
//#include "tiling/tiling_base.h"
#include "../tiling_base/tiling_base.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

BEGIN_TILING_DATA_DEF(CausalConv1dUpdateTilingData)
    TILING_DATA_FIELD_DEF(int64_t, numCore);
    TILING_DATA_FIELD_DEF(int64_t, blockFactor);
    TILING_DATA_FIELD_DEF(int64_t, blockTailFactor);
    TILING_DATA_FIELD_DEF(int64_t, batch);
    TILING_DATA_FIELD_DEF(int64_t, seqLen);
    TILING_DATA_FIELD_DEF(int64_t, dim);
    TILING_DATA_FIELD_DEF(int64_t, width);
    TILING_DATA_FIELD_DEF(int64_t, stateLen);
    TILING_DATA_FIELD_DEF(int64_t, hasIndices);
    TILING_DATA_FIELD_DEF(int64_t, hasBias);
    TILING_DATA_FIELD_DEF(int64_t, hasNumAccept);
    TILING_DATA_FIELD_DEF(int64_t, hasQueryLoc);
    TILING_DATA_FIELD_DEF(int64_t, activationMode);
    TILING_DATA_FIELD_DEF(int64_t, padSlotId);
END_TILING_DATA_DEF;

struct CausalConv1dUpdateCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

namespace causalconv1dupdate {

class CausalConv1dUpdate {
public:
    explicit CausalConv1dUpdate(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoCausalConv1dUpdateTiling();

protected:
    ge::graphStatus GetPlatform();
    ge::graphStatus GetOpParam();
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckAttrs();

    int64_t GetCoreNum(int64_t factor, int64_t coreNum) const;
    void CalcTiling();
    void CalcTilingKey();
    void CalcBlockFactor(int64_t size);
    // void CalcUBFactor();
    void WriteTilingData();

private:
    gert::TilingContext* context_ = nullptr;

    int64_t coreNum_{0};
    uint64_t ubSize_{0};
    int64_t reserveUb_{2048};
    int64_t cacheLine_{256};

    gert::Shape xShape_;
    gert::Shape wShape_;
    gert::Shape convStateShape_;
    gert::Shape stateIndicesShape_;
    gert::Shape biasShape_;
    gert::Shape numShape_;
    gert::Shape locShape_;

    ge::DataType xDtype_{ge::DT_UNDEFINED};
    ge::DataType wDtype_{ge::DT_UNDEFINED};
    ge::DataType convStateDtype_{ge::DT_UNDEFINED};
    ge::DataType yDtype_{ge::DT_UNDEFINED};

    int64_t batch_{0};
    int64_t seqLen_{0};
    int64_t dim_{0};
    int64_t width_{0};
    int64_t stateLen_{0};
    int64_t hasIndices_{0};
    int64_t hasBias_{0};
    int64_t hasNumAccept_{0};
    int64_t hasQueryLoc_{0};

    int64_t activationMode_{0};
    int64_t padSlotId_{-1};

    int64_t actCoreNum_{0};
    int64_t blockFactor_{-1};
    int64_t blockTailFactor_{-1};
    // int64_t baseN_{1};
    uint64_t tilingKey_{0};

};
} // namespace causalconv1dupdate
} // namespace optiling
#endif