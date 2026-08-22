/* Build-only subset of mc2_tiling_utils.h used by MegaMoe. */
#ifndef VLLM_ASCEND_MEGA_MOE_MC2_TILING_UTILS_COMPAT_H
#define VLLM_ASCEND_MEGA_MOE_MC2_TILING_UTILS_COMPAT_H

#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"

namespace ops {
template <typename T>
constexpr T CeilAlign(T value, T alignment)
{
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

template <typename T>
constexpr T CeilDiv(T value, T divisor)
{
    return divisor == 0 ? value : (value + divisor - 1) / divisor;
}
} // namespace ops

namespace mc2tiling {
inline std::string GetSocVersion(const gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    return socVersion;
}
} // namespace mc2tiling

#endif
