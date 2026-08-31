/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <string>

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/shape.h>
#include <tiling/tiling_api.h>

// The upstream Flash MLA checker uses these helpers only to format validation
// errors.  Pulling them from libopapi_nn makes the custom tiling library load a
// second op-api stack inside OPC and at process teardown.  Keep small hidden
// implementations in the custom opmaster instead.
namespace Ops {
namespace Base {

__attribute__((visibility("hidden"))) std::string ToString(ge::DataType dataType)
{
    return ge::TypeUtils::DataTypeToSerialString(dataType);
}

__attribute__((visibility("hidden"))) std::string ToString(ge::Format format)
{
    return ge::TypeUtils::FormatToSerialString(format);
}

__attribute__((visibility("hidden"))) std::string ToString(const gert::Shape &shape)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << shape.GetDim(index);
    }
    stream << "]";
    return stream.str();
}

}  // namespace Base
}  // namespace Ops
