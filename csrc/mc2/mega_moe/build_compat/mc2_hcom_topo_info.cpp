/* Minimal host implementation needed by MegaMoe exception-dump handling. */
#include <dlfcn.h>

#include "mc2_hcom_topo_info.h"

namespace Mc2Hcom {
HcclResult MC2HcomTopology::CommGetHcclBufferByGroup(const char *group, void **buffer, uint64_t *size)
{
    if (group == nullptr || buffer == nullptr || size == nullptr) {
        return HCCL_E_PARA;
    }

    using GetComm = HcclResult (*)(const char *, HcclComm *);
    using GetBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);
    void *framework = dlopen("libhccl_fwk.so", RTLD_LAZY | RTLD_LOCAL);
    void *hccl = dlopen("libhccl.so", RTLD_LAZY | RTLD_LOCAL);
    if (framework == nullptr || hccl == nullptr) {
        return HCCL_E_INTERNAL;
    }
    auto getComm = reinterpret_cast<GetComm>(dlsym(framework, "HcclCommGetHandleWithName"));
    auto getBuffer = reinterpret_cast<GetBuffer>(dlsym(hccl, "HcclGetHcclBuffer"));
    if (getComm == nullptr || getBuffer == nullptr) {
        return HCCL_E_INTERNAL;
    }
    HcclComm communicator = nullptr;
    HcclResult result = getComm(group, &communicator);
    return result == HCCL_SUCCESS ? getBuffer(communicator, buffer, size) : result;
}
} // namespace Mc2Hcom
