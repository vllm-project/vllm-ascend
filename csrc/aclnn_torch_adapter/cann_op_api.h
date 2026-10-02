// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <dlfcn.h>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace vllm_ascend {

// Resolve a related set of ACLNN entry points from one official component.
// Never fall back to RTLD_DEFAULT or libcust_opapi.so: a custom operator may
// export the same name with a different GetWorkspaceSize ABI.
inline void* OpenCannOpApiLibrary(std::initializer_list<const char*> symbols)
{
    std::string errors;
    for (const char* library : {"libopapi_transformer.so", "libopapi.so"}) {
        void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char* error = dlerror();
            errors += std::string(library) + ": " + (error ? error : "cannot load") + "; ";
            continue;
        }
        bool complete = true;
        for (const char* symbol : symbols) {
            if (dlsym(handle, symbol) == nullptr) {
                errors += std::string(library) + ": missing " + symbol + "; ";
                complete = false;
            }
        }
        if (complete) {
            // Retain the successful handle for the lifetime of cached function
            // pointers and queued operators (including graph replay).
            return handle;
        }
        dlclose(handle);
    }
    throw std::runtime_error("No compatible official CANN operator library. " + errors);
}

inline void* GetCannQliOpApiFuncAddr(const char* symbol)
{
    // Both stages of QLI and its metadata producer must use one provider.
    static void* handle = OpenCannOpApiLibrary({
        "aclnnQuantLightningIndexerV2GetWorkspaceSize",
        "aclnnQuantLightningIndexerV2",
        "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize",
        "aclnnQuantLightningIndexerV2Metadata",
    });
    void* address = dlsym(handle, symbol);
    if (address == nullptr) {
        throw std::runtime_error(std::string("Missing official CANN symbol: ") + symbol);
    }
    return address;
}

}  // namespace vllm_ascend
