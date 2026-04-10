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
 * \file reshape_and_cache_common.h
 * \brief
 */

#ifndef ASCEND_OPS_RESHAPE_AND_CACHE_BY_GROUP_COMMON_H
#define ASCEND_OPS_RESHAPE_AND_CACHE_BY_GROUP_COMMON_H

#include <climits>
#include "register/tilingdata_base.h"
#include "tiling/tiling_base.h"
#include "error_log.h"
#include "../tiling_base/tiling_base.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr uint32_t DIM_5 = 5;
constexpr uint32_t DIM_6 = 6;

constexpr int32_t MAX_UB_USE_SIZE = 180 * 1024;


BEGIN_TILING_DATA_DEF(ReshapeAndCacheByGroupTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockTableSize);
    TILING_DATA_FIELD_DEF(uint32_t, typeByte);
    TILING_DATA_FIELD_DEF(uint32_t, tokenSize);
    TILING_DATA_FIELD_DEF(uint32_t, corePerNum);
    TILING_DATA_FIELD_DEF(uint32_t, coreTail);
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, numCache);
    TILING_DATA_FIELD_DEF(uint32_t, groupInfoLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ReshapeAndCacheByGroup,  ReshapeAndCacheByGroupTilingData)


struct  ReshapeAndCacheByGroupParams {
    uint32_t numTokens{0};
    uint32_t numCache{0};
    uint32_t numHeads{1};
    uint32_t headSize[5]{1,1,1,1,1};
    uint32_t blockTableSize{0};
    uint32_t typeByte{0};
    uint32_t tokenSize{1};
    uint32_t tilingKey{0};
    uint64_t workspaceSize{0};
    uint64_t groupInfoLen{0};
    uint32_t corepernum{0};
    uint32_t coretail{0};
    uint64_t sysWorkspaceSize{0};
    uint32_t coreNum{0};
};

class ReshapeAndCacheByGroupCommonTiling {
public:
    explicit ReshapeAndCacheByGroupCommonTiling(gert::TilingContext* context) : context_(context) {};
    virtual ~ReshapeAndCacheByGroupCommonTiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoCommonTiling();
    ge::graphStatus DoTiling();
    void SetTiling();
    void PrintTilingData();

protected:
    gert::TilingContext *context_ = nullptr;
    ReshapeAndCacheByGroupParams params;
    ReshapeAndCacheByGroupTilingData tilingData_;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

}
#endif // ASCEND_OPS_RESHAPE_AND_CACHE_COMMON_H
