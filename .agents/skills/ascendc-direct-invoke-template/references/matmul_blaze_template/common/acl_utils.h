/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_UTILS_H
#define ACL_UTILS_H

#include <cstdint>

#include "acl/acl.h"

#include "common_utils.h"

class AclRtSession {
public:
    explicit AclRtSession(int32_t deviceId) : deviceId_(deviceId) {}

    void Init()
    {
        CHECK_COND(aclInit(nullptr) == ACL_SUCCESS, "Failed to initialize ACL runtime.");
        aclInitialized_ = true;

        uint32_t deviceCount = 0;
        CHECK_COND(aclrtGetDeviceCount(&deviceCount) == ACL_SUCCESS, "Failed to query ACL device count.");
        CHECK_COND(deviceCount > 0U, "No ACL devices are available.");
        CHECK_COND(deviceId_ >= 0 && static_cast<uint32_t>(deviceId_) < deviceCount, "ACL device id is out of range.");

        CHECK_COND(aclrtSetDevice(deviceId_) == ACL_SUCCESS, "Failed to set the ACL device.");
        deviceSet_ = true;
        CHECK_COND(aclrtCreateStream(&stream_) == ACL_SUCCESS, "Failed to create the ACL stream.");
    }

    aclrtStream GetStream() const
    {
        return stream_;
    }

    ~AclRtSession()
    {
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            (void)aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInitialized_) {
            (void)aclFinalize();
            aclInitialized_ = false;
        }
    }

    AclRtSession(const AclRtSession&) = delete;
    AclRtSession& operator=(const AclRtSession&) = delete;

private:
    int32_t deviceId_;
    aclrtStream stream_{nullptr};
    bool deviceSet_{false};
    bool aclInitialized_{false};
};

#endif
