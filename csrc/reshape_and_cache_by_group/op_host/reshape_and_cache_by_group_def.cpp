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
 * \file reshape_and_cache.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class ReshapeAndCacheByGroup : public OpDef {
 public:
  explicit ReshapeAndCacheByGroup(const char* name) : OpDef(name) {
    this->Input("keyIn")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("keyCacheIn")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("groupLen")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32 , ge::DT_INT32 , ge::DT_INT32 })
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("groupKeyIdx")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32 , ge::DT_INT32 , ge::DT_INT32 })
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("groupKeyCacheIdx")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32 , ge::DT_INT32 , ge::DT_INT32 })
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("blockSize").Int();
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(ReshapeAndCacheByGroup);
}  // namespace ops