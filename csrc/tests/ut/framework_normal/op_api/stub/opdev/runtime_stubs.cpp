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
 * \file runtime_stubs.cpp
 * \brief
 */

#include <securec.h>
#include <cstdint>
#include <map>
#include <any>
#include <string>
#include <dlfcn.h>
#include <runtime/rt.h>
#include <runtime/base.h>

extern "C" {
rtError_t rtGetSocVersion(char *version, const uint32_t maxLen)
{
    std::string ver = "Ascend910B";
    (void)strncpy_s(version, maxLen, ver.c_str(), ver.length());
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceInfo(uint32_t device_id, int32_t module_type, int32_t info_type, int64_t *val)
{
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetDevice(int32_t *device)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchEx([[maybe_unused]] void *args, [[maybe_unused]] uint32_t argsSize,
                           [[maybe_unused]] uint32_t flags, [[maybe_unused]] rtStream_t stm)
{
    return RT_ERROR_NONE;
}

rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle)
{
    return RT_ERROR_NONE;
}

rtError_t rtDevBinaryUnRegister(void *handle)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                     rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                   rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtLaunchKernelByFuncHandleV2(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                       rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDevice(int32_t *device)
{
    *device = 0;
    return RT_ERROR_NONE;
}

rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *freeSize, size_t *totalSize)
{
    return RT_ERROR_NONE;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
{
    return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName, const void *kernelInfoExt,
                             uint32_t funcMode)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetFunctionByName(const char_t *stubName, void **stubFunc)
{
    return RT_ERROR_NONE;
}

rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aiCoreMemorySize)
{
    return RT_ERROR_NONE;
}

rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
                                        const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stream,
                                        uint32_t flags)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronize(rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stm, int32_t timeout)
{
    return RT_ERROR_NONE;
}
int64_t gDeterministic = 0;
rtFloatOverflowMode_t gOverflow = RT_OVERFLOW_MODE_SATURATION;

rtError_t rtCtxSetSysParamOpt(const rtSysParamOpt configOpt, const int64_t configVal)
{
    if (configOpt == SYS_OPT_DETERMINISTIC) {
        gDeterministic = configVal;
    }
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetSysParamOpt(const rtSysParamOpt configOpt, int64_t *const configVal)
{
    if (configOpt == SYS_OPT_DETERMINISTIC) {
        *configVal = gDeterministic;
    }
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t *floatOverflowMode)
{
    *floatOverflowMode = gOverflow;
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetOverflowAddr(void **overflowAddr)
{
    *overflowAddr = reinterpret_cast<void *>(0x005);
    return RT_ERROR_NONE;
}

rtError_t rtSetDeviceSatMode(rtFloatOverflowMode_t floatOverflowMode)
{
    gOverflow = floatOverflowMode;
    return RT_ERROR_NONE;
}

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);
    return RT_ERROR_NONE;
}

rtError_t rtFree(void *devptr)
{
    delete[] (uint8_t *)devptr;
    return RT_ERROR_NONE;
}

rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                        rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len)
{
    return RT_ERROR_NONE;
}

rtError_t rtCalcLaunchArgsSize(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum, size_t *launchArgsSize)
{
    *launchArgsSize = argsSize + hostInfoTotalSize;
    return RT_ERROR_NONE;
}

rtError_t rtCreateLaunchArgs(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum, void *argsData,
                             rtLaunchArgsHandle *argsHandle)
{
    static size_t hdlData = 0;
    static rtLaunchArgsHandle hdl = static_cast<void *>(&hdlData);
    *argsHandle = hdl;
    return RT_ERROR_NONE;
}


rtError_t rtDestroyLaunchArgs(rtLaunchArgsHandle argsHandle)
{
    return RT_ERROR_NONE;
}


rtError_t rtAppendLaunchAddrInfo(rtLaunchArgsHandle argsHandle, void *addrInfo)
{
    return RT_ERROR_NONE;
}


rtError_t rtAppendLaunchHostInfo(rtLaunchArgsHandle argsHandle, size_t hostInfoSize, void **hostInfo)
{
    return RT_ERROR_NONE;
}


rtError_t rtBinaryLoad(const rtDevBinary_t *bin, rtBinHandle *binHandle)
{
    return RT_ERROR_NONE;
}


rtError_t rtBinaryGetFunction(const rtBinHandle binHandle, const uint64_t tilingKey, rtFuncHandle *funcHandle)
{
    return RT_ERROR_NONE;
}


rtError_t rtLaunchKernelByFuncHandle(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                     rtStream_t stm)
{
    return RT_ERROR_NONE;
}


rtError_t rtSetExceptionExtInfo(const rtArgsSizeInfo_t *const sizeInfo)
{
    return RT_ERROR_NONE;
}


rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags)
{
    return RT_ERROR_NONE;
}


rtError_t rtStreamDestroy(rtStream_t stream)
{
    return RT_ERROR_NONE;
}


rtError_t rtEventCreateWithFlag(rtEvent_t *event, uint32_t flag)
{
    return RT_ERROR_NONE;
}


rtError_t rtEventDestroy(rtEvent_t event)
{
    return RT_ERROR_NONE;
}


rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream)
{
    return RT_ERROR_NONE;
}


rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event)
{
    return RT_ERROR_NONE;
}


rtError_t rtEventReset(rtEvent_t event, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

static uint64_t floatDebugStatus = 0;

rtError_t rtNpuGetFloatDebugStatus(void *outputAddrPtr, uint64_t outputSize, uint32_t checkMode, rtStream_t stm)
{
    uint64_t *status = static_cast<uint64_t *>(outputAddrPtr);
    floatDebugStatus = 1;
    *status = floatDebugStatus;
    return RT_ERROR_NONE;
}


rtError_t rtNpuClearFloatDebugStatus(uint32_t checkMode, rtStream_t stm)
{
    floatDebugStatus = 0;
    return RT_ERROR_NONE;
}


rtError_t rtCtxGetCurrent(rtContext_t *ctx)
{
    int64_t x = 1;
    *ctx = (void *)x;
    return RT_ERROR_NONE;
}

rtError_t rtBinaryLoadWithoutTilingKey(const void *data, const uint64_t length, rtBinHandle *binHandle)
{
    return RT_ERROR_NONE;
}

rtError_t rtBinaryGetFunctionByName(const rtBinHandle binHandle, const char *kernelName, rtFuncHandle *funcHandle)
{
    return RT_ERROR_NONE;
}

} // extern "C"
