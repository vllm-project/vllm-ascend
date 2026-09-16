/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <map>

// #include "runtime/mem.h"
// #include "runtime/dev.h"
// #include "runtime/context.h"

#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

#define DEVICE_ID 0

// static string & DescribeError(rtError_t err) {
//   static string unknown = "[unknown error]";
//   static map<int32_t, string> desc = {
//     {0, "[success]"},
//     {107000, "[param invalid]"},
//     {107001, "[invalid device id]"},
//     {107002, "[current context null]"},
//     {107003, "[stream not in current context]"},
//     {507000, "[runtime internal error]"},
//     {507008, "[soc version error]"},
//     {507014, "[aicore timeout]"},
//     {507015, "[aicore exception]"},
//     {507016, "[aicore trap exception]"},
//     {507017, "[aicpu timeout]"},
//     {507018, "[aicpu exception]"},
//     {507021, "[profiling error]"},
//     {507034, "[vector core timeout]"},
//     {507035, "[vector core exception]"},
//     {507036, "[vector trap exception]"}
//   };
//   auto iter = desc.find(static_cast<int32_t>(err));
//   if (iter != desc.end()) {
//     return iter->second;
//   } else {
//     return unknown;
//   }
// }

void *MallocDeviceMemory(unsigned long size)
{
    return nullptr;
    //  if (size <= 0) {
    //    return nullptr;
    //  }
    //
    //  void * dev_mem = nullptr;
    //  rtError_t ret = rtMalloc(&dev_mem, size, RT_MEMORY_DEFAULT, TBE_MODULE_ID);
    //  if (ret != RT_ERROR_NONE) {
    ////    cout << "rtMalloc failed, return " << ret << DescribeError(ret) << endl;
    //  }
    //  return dev_mem;
}

void FreeDeviceMemory(void *device_mem_ptr)
{
    if (device_mem_ptr != nullptr) {
        //    (void)rtFree(device_mem_ptr);
    }
}

int MemcpyToDevice(const void *host_mem, void *dev_mem, unsigned long size)
{
    return 0;
    //  rtError_t ret = rtMemcpy(dev_mem, size, host_mem, size, RT_MEMCPY_HOST_TO_DEVICE);
    //  if (ret != RT_ERROR_NONE) {
    ////    cout << "rtMemcpy to device failed, return " << ret << DescribeError(ret) << endl;
    //  }
    //  return ret == RT_ERROR_NONE ? 0 : 1;
}

int MemcpyFromDevice(void *host_mem, const void *dev_mem, unsigned long size)
{
    return 0;
    //  rtError_t ret = rtMemcpy(host_mem, size, dev_mem, size, RT_MEMCPY_DEVICE_TO_HOST);
    //  if (ret != RT_ERROR_NONE) {
    ////    cout << "rtMemcpy to host failed, return " << ret << DescribeError(ret) << endl;
    //  }
    //  return ret == RT_ERROR_NONE ? 0 : 1;
}

bool IsFileExistsAccessable(string &name)
{
    return (access(name.c_str(), F_OK) != -1);
}

int RtsInit()
{
    // char cwd[256];
    // if (getcwd(cwd, sizeof(cwd)) == NULL) {
    //   cout << "Failed to getcwd" << cwd <<endl;
    //   return 1;
    // }
    // string dumpFile = string(cwd) + "/acl.json";
    // aclError aclRet = ACL_SUCCESS;

    // if (IsFileExistsAccessable(dumpFile)) {
    //   aclRet = aclInit(dumpFile.c_str());
    //   if (aclRet != ACL_SUCCESS) {
    //     cout << "aclInit with dump failed, return " << aclRet << DescribeError(aclRet) << endl;
    //     return 1;
    //   }
    // } else {
    //   aclRet = aclInit(nullptr);
    // }

    // if (aclRet != ACL_SUCCESS) {
    //   cout << "aclInit failed, return " << aclRet << DescribeError(aclRet) << endl;
    //   return 1;
    // }

    // aclRet = aclrtSetDevice(DEVICE_ID);
    // if (aclRet != ACL_SUCCESS) {
    //   cout << "aclrtSetDevice failed, return " << aclRet << DescribeError(aclRet) << endl;
    //   return 1;
    // }

    //  auto rtRet = rtSetDevice(DEVICE_ID);
    //  if (rtRet != RT_ERROR_NONE) {
    ////    cout << "rtSetDevice failed, return " << DescribeError(rtRet) << endl;
    //  }

    return 0;
}

void RtsUnInit()
{
    // aclrtResetDevice(DEVICE_ID);
    // aclFinalize();
    //  (void)rtDeviceReset(DEVICE_ID);
}

int SynchronizeStream()
{
    //  rtError_t ret = rtStreamSynchronize(nullptr);
    //  if (ret != RT_ERROR_NONE) {
    ////    cout << "rtStreamSynchronize failed, return " << ret << DescribeError(ret) << endl;
    //    return 1;
    //  }
    return 0;
}
