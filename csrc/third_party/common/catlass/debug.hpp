/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DEBUG_HPP
#define CATLASS_DEBUG_HPP

#include <iostream>
#include <sstream>
#include <functional>

#include <acl/acl.h>

#define SINGLE_CORE_DUMPSIZE (1024 * 1024)
// 75 is from AscendC host stub
#define ALL_DUMPSIZE (75 * SINGLE_CORE_DUMPSIZE)

using LogFuncType = std::function<void(const char *)>;
/**
 * @brief Check acl api status code.
 * @param status The return code of acl api.
 * @param logFunc Log function, which receives a C-Style string.
 * @return
 */
inline void aclCheck(aclError status, LogFuncType logFunc = [](const char *logStrPtr) { std::cerr << logStrPtr; })
{
    if (status != ACL_SUCCESS) {
        std::stringstream ss;
        ss << "AclError: " << status;
        logFunc(ss.str().c_str());
    }
}
/**
 * @brief Check rt api status code.
 * @param status The return code of rt api.
 * @param logFunc Log function, which receives a C-Style string.
 * @return
 */
inline void rtCheck(int status, LogFuncType logFunc = [](const char *logStrPtr) { std::cerr << logStrPtr; })
{
    if (status != 0) {
        std::stringstream ss;
        ss << "RtError: " << status;
        logFunc(ss.str().c_str());
    }
}

namespace Adx {
void AdumpPrintWorkSpace(const void *dumpBufferAddr,
                         const size_t dumpBufferSize,
                         aclrtStream stream,
                         const char *opType);
} // namespace Adx

#endif // CATLASS_DEBUG_HPP
