/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lock.cpp
 * \brief
 */
#include "lock.h"

namespace Ops {
namespace Transformer {
namespace Optiling {
void RWLock::rdlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    _waiting_readers += 1;
    _read_cv.wait(lck, [&]() { return _waiting_writers == 0 && _status >= 0; });
    _waiting_readers -= 1;
    _status += 1;
}

void RWLock::wrlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    _waiting_writers += 1;
    _write_cv.wait(lck, [&]() { return _status == 0; });
    _waiting_writers -= 1;
    _status = -1;
}

void RWLock::unlock()
{
    std::unique_lock<std::mutex> lck(_mtx);
    if (_status == -1) {
        _status = 0;
    } else {
        _status -= 1;
    }
    if (_waiting_writers > 0) {
        if (_status == 0) {
            _write_cv.notify_one();
        }
    } else {
        _read_cv.notify_all();
    }
}
} // namespace Optiling
} // namespace Transformer
} // namespace Ops
