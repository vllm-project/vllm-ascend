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
 * \file hc_post_tiling.h
 * \brief
 */
#ifndef HC_POST_TILING_H_
#define HC_POST_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HcPostTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, bParam);
TILING_DATA_FIELD_DEF(int64_t, sParam);
TILING_DATA_FIELD_DEF(int64_t, hcParam);
TILING_DATA_FIELD_DEF(int64_t, dParam);
TILING_DATA_FIELD_DEF(int64_t, batchOneCore);
TILING_DATA_FIELD_DEF(int64_t, batchOneCoreTail);
TILING_DATA_FIELD_DEF(int64_t, frontCore);
TILING_DATA_FIELD_DEF(int64_t, dSplitTime);
TILING_DATA_FIELD_DEF(int64_t, dOnceDealing);
TILING_DATA_FIELD_DEF(int64_t, dLastDealing);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HcPost, HcPostTilingData)

struct HcPostCompileInfo {
};

class HcPostTilingRegbase {
public:
    explicit HcPostTilingRegbase(gert::TilingContext* context) : context_(context)
    {}
    ~HcPostTilingRegbase()
    {}
    ge::graphStatus RunTilingRegbase();

protected:
    ge::graphStatus DoOpTilingRegbase();
    ge::graphStatus GetPlatformInfoRegbase();
    ge::graphStatus GetShapeInfoRegbase();
    ge::graphStatus GetInputShapeInfoRegbase();
    ge::graphStatus PostTilingRegbase();

private:
    ge::graphStatus GetInputDtypeInfoRegbase();

private:
    HcPostTilingData tilingRegbaseData_;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockSize_ = 0;

    int64_t bParam_ = 0;
    int64_t sParam_ = 0;
    int64_t dParam_ = 0;
    int64_t hcParam_ = 0;
    int64_t batchSize_ = 0;
    int64_t tilingKey_ = 0;

    gert::TilingContext *context_ = nullptr;
};

class HcPostTiling {
public:
    explicit HcPostTiling(gert::TilingContext* context) : context_(context)
    {}
    ~HcPostTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus GetInputShapeInfo();
    ge::graphStatus PostTiling();

private:
    ge::graphStatus GetInputDtypeInfo();

private:
    HcPostTilingData tilingData_;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockSize_ = 0;

    int64_t bParam_ = 0;
    int64_t sParam_ = 0;
    int64_t dParam_ = 0;
    int64_t hcParam_ = 0;

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling

#endif // GATHER_SELECTION_KV_CACHE_TILING_H_