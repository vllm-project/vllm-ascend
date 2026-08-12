/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_UTILS_DEVICE_UTILS_H
#define ARCH35_CATLASS_UTILS_DEVICE_UTILS_H

#define DEVICE __aicore__ inline

#if defined(__CCE_KT_TEST__)
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define X_LOG(format, ...)                                                                                             \
    do {                                                                                                               \
        std::string coreType = "";                                                                                     \
        std::string blockId = "Block_";                                                                                \
        if (g_coreType == AscendC::AIC_TYPE) {                                                                         \
            coreType = "AIC_";                                                                                         \
        } else if (g_coreType == AscendC::AIV_TYPE) {                                                                  \
            coreType = "AIV_";                                                                                         \
        } else {                                                                                                       \
            coreType = "MIX_";                                                                                         \
        }                                                                                                              \
        coreType += std::to_string(sub_block_idx);                                                                     \
        blockId += std::to_string(block_idx);                                                                          \
        printf("[%s][%s][%s:%d][%s][%ld] " format "\n", blockId.c_str(), coreType.c_str(), FILENAME, __LINE__,         \
               __FUNCTION__, (long)getpid(), ##__VA_ARGS__);                                                           \
    } while (0)

#else
#define X_LOG(format, ...)
#endif
#endif
