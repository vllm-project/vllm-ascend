// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef DSV41_MLA_PROLOG_TILING_DATA_H_
#define DSV41_MLA_PROLOG_TILING_DATA_H_

#include <cstdint>

namespace Dsv41 {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kQa = 1280;
constexpr uint32_t kKv = 512;
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kNopeDim = 448;
constexpr uint32_t kRopeDim = 64;
constexpr uint32_t kCacheRowsPerPage = 128;
constexpr uint32_t kMaxT = 4096;
constexpr bool SupportsHeads(uint32_t numHeads)
{
    return numHeads == 16 || numHeads == 32 || numHeads == 64;
}
constexpr uint32_t kWorkspaceStages = 2;
constexpr uint32_t kMaxProjectionSplits = 2;
constexpr uint32_t kSplitProjectionMaxT = 128;
constexpr bool kEnableRopeScatter = true;
constexpr float kProjectionUlpGuard = 1.000006F;
constexpr float kProjectionGeneralGuard = 0.99999994F;
constexpr float kRmsNormUlpGuard = 1.0000055F;

constexpr uint64_t Align512(uint64_t value)
{
    return (value + 511U) & ~uint64_t(511U);
}

struct WorkspaceLayout {
    uint64_t qaOffset;
    uint64_t qaBytes;
    uint64_t kvOffset;
    uint64_t kvBytes;
    uint64_t kvNormOffset;
    uint64_t kvNormBytes;
    uint64_t qInt8Offset;
    uint64_t qInt8Bytes;
    uint64_t qScaleOffset;
    uint64_t qScaleBytes;
    uint64_t wScaleFpOffset;
    uint64_t wScaleFpBytes;
    uint64_t qmmOffset;
    uint64_t qmmBytes;
    uint64_t totalBytes;
};

static inline WorkspaceLayout MakeWorkspaceLayout(uint32_t t, uint32_t aicCount, uint32_t numHeads)
{
    WorkspaceLayout layout{};
    layout.qaOffset = 0;
    layout.qaBytes = Align512(
        static_cast<uint64_t>(kMaxProjectionSplits) * t * kQa * sizeof(float));
    layout.kvOffset = layout.qaOffset + layout.qaBytes;
    layout.kvBytes = Align512(
        static_cast<uint64_t>(kMaxProjectionSplits) * t * kKv * sizeof(float));
    layout.kvNormOffset = layout.kvOffset + layout.kvBytes;
    layout.kvNormBytes = Align512(static_cast<uint64_t>(t) * kKv * sizeof(uint16_t));
    layout.qInt8Offset = layout.kvNormOffset + layout.kvNormBytes;
    layout.qInt8Bytes = Align512(static_cast<uint64_t>(t) * kQa * sizeof(int8_t));
    layout.qScaleOffset = layout.qInt8Offset + layout.qInt8Bytes;
    layout.qScaleBytes = Align512(static_cast<uint64_t>(t) * sizeof(float));
    layout.wScaleFpOffset = layout.qScaleOffset + layout.qScaleBytes;
    layout.wScaleFpBytes = Align512(static_cast<uint64_t>(numHeads) * kHeadDim * sizeof(float));
    layout.qmmOffset = layout.wScaleFpOffset + layout.wScaleFpBytes;
    layout.qmmBytes = Align512(
        static_cast<uint64_t>(aicCount) * kWorkspaceStages * 128U * 256U *
        sizeof(int32_t));
    layout.totalBytes = Align512(layout.qmmOffset + layout.qmmBytes);
    return layout;
}

}  // namespace Dsv41

#endif  // DSV41_MLA_PROLOG_TILING_DATA_H_
