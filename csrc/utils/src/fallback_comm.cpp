/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fallback_comm.cpp
 * \brief
 */

#include "fallback_comm.h"

#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "aclnn/aclnn_base.h"
#include "runtime/base.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace std;
using namespace gert;
using namespace ge;

aclDataType ToAclDataType(ge::DataType dtype) {
  static const std::vector<DataType> CANN_CONVERT_TO_ACL_DataType_LIST = {
      ge::DataType::DT_FLOAT,     ge::DataType::DT_FLOAT16,    ge::DataType::DT_INT8,   ge::DataType::DT_INT32,
      ge::DataType::DT_UINT8,     ge::DataType::DT_INT16,      ge::DataType::DT_UINT16, ge::DataType::DT_UINT32,
      ge::DataType::DT_INT64,     ge::DataType::DT_DOUBLE,     ge::DataType::DT_BOOL,   ge::DataType::DT_STRING,
      ge::DataType::DT_COMPLEX64, ge::DataType::DT_COMPLEX128, ge::DataType::DT_BF16,  ge::DataType::DT_UINT64,
      ge::DataType::DT_INT4};
  auto iter = std::find(CANN_CONVERT_TO_ACL_DataType_LIST.begin(), CANN_CONVERT_TO_ACL_DataType_LIST.end(), dtype);
  if (iter == CANN_CONVERT_TO_ACL_DataType_LIST.end()) {
    return aclDataType::ACL_DT_UNDEFINED;
  }
  return static_cast<aclDataType>(dtype);
}

}  // namespace fallback

#ifdef __cplusplus
}
#endif
