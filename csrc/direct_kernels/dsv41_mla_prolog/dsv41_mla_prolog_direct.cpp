/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host-direct launch path for Dsv41MlaProlog.
 *
 * This translation unit is compiled by the AscendC toolchain and linked into
 * vllm_ascend_kernels; the kernel is launched with `<<<>>>` through the CATLASS
 * device adapter (the path validated by the standalone runner).
 */

#include "dsv41_mla_prolog_direct_api.h"

#include <acl/acl.h>

#include "../../attention/dsv41_mla_prolog/op_kernel/dsv41_mla_prolog_kernels.h"

namespace vllm_ascend {
namespace {

bool Valid(const Dsv41MlaPrologDirectArgs &args)
{
    return args.x != nullptr && args.wQa != nullptr && args.gammaQa != nullptr &&
        args.wQb != nullptr && args.wQbScale != nullptr && args.wKv != nullptr &&
        args.gammaKv != nullptr && args.ropeCos != nullptr &&
        args.ropeSin != nullptr && args.slots != nullptr &&
        args.kvCache != nullptr && args.qOutput != nullptr && args.qr != nullptr &&
        args.workspace != nullptr && args.t > 0 &&
        args.t <= static_cast<int32_t>(Dsv41::kMaxT) && args.epsilon > 0.0F;
}

uint32_t QueryAicCount()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return 0;
    }
    return platform->GetCoreNumAic();
}

}  // namespace

uint64_t dsv41_mla_prolog_workspace_size(int64_t t)
{
    if (t <= 0 || t > static_cast<int64_t>(Dsv41::kMaxT)) {
        return 0;
    }
    const uint32_t aicCount = QueryAicCount();
    if (aicCount == 0) {
        return 0;
    }
    return Dsv41::MakeWorkspaceLayout(
        static_cast<uint32_t>(t), aicCount).totalBytes;
}

int dsv41_mla_prolog_direct(
    const Dsv41MlaPrologDirectArgs &args, void *streamRaw)
{
    if (!Valid(args)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint32_t aicCount = QueryAicCount();
    if (aicCount == 0) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    const auto layout = Dsv41::MakeWorkspaceLayout(
        static_cast<uint32_t>(args.t), aicCount);
    if (args.workspaceSize < layout.totalBytes) {
        return ACL_ERROR_INVALID_PARAM;
    }

    // Defensive default: callers built before the strided-cache fields existed
    // leave them zeroed. Treat that as a contiguous 128x512 page layout.
    uint32_t cacheBlockSize = args.cacheBlockSize;
    uint64_t cacheStride0 = args.cacheStride0;
    uint64_t cacheStride1 = args.cacheStride1;
    if (cacheBlockSize == 0) {
        cacheBlockSize = 128;
        cacheStride1 = Dsv41::kHeadDim;
        cacheStride0 = static_cast<uint64_t>(cacheBlockSize) * cacheStride1;
    }

    auto *workspaceBase = static_cast<uint8_t *>(args.workspace);
    auto *qa = workspaceBase + layout.qaOffset;
    auto *kv = workspaceBase + layout.kvOffset;
    auto *kvNorm = workspaceBase + layout.kvNormOffset;
    auto *qInt8 = workspaceBase + layout.qInt8Offset;
    auto *qScale = workspaceBase + layout.qScaleOffset;
    auto *wScaleFp = workspaceBase + layout.wScaleFpOffset;
    auto *qmmWorkspace = workspaceBase + layout.qmmOffset;

    const uint32_t t = static_cast<uint32_t>(args.t);
    const uint32_t projectionSplits =
        t <= Dsv41::kSplitProjectionMaxT ? 2U : 1U;
    aclrtStream stream = reinterpret_cast<aclrtStream>(streamRaw);

    Dsv41::DualProjectionKernel::Arguments projectionArgs{
        GemmCoord(t, Dsv41::kQa, Dsv41::kHidden),
        GemmCoord(t, Dsv41::kKv, Dsv41::kHidden),
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.x)),
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.wQa)),
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.wKv)),
        qa,
        kv,
        projectionSplits};
    Dsv41::NormQuantKernel::Arguments normArgs{
        qa,
        kv,
        kvNorm,
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.gammaQa)),
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.gammaKv)),
        qInt8,
        qScale,
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.wQbScale)),
        wScaleFp,
        reinterpret_cast<uint8_t *>(args.qr),
        t,
        projectionSplits,
        args.epsilon};
    Dsv41::QuantMatmul::Arguments quantArgs{
        GemmCoord(t, Dsv41::kQbOut, Dsv41::kQa),
        aicCount,
        qInt8,
        reinterpret_cast<uint8_t *>(const_cast<int8_t *>(args.wQb)),
        wScaleFp,
        qScale,
        reinterpret_cast<uint8_t *>(args.qOutput)};
    Dsv41::RopeScatterKernel::Arguments ropeArgs{
        reinterpret_cast<uint8_t *>(args.qOutput),
        kvNorm,
        reinterpret_cast<uint8_t *>(const_cast<float *>(args.ropeCos)),
        reinterpret_cast<uint8_t *>(const_cast<float *>(args.ropeSin)),
        reinterpret_cast<uint8_t *>(const_cast<int32_t *>(args.slots)),
        reinterpret_cast<uint8_t *>(args.kvCache),
        cacheStride0,
        cacheStride1,
        cacheBlockSize,
        t};
    Dsv41::UnifiedMlaPrologKernel::Arguments unifiedArgs{
        projectionArgs,
        normArgs,
        quantArgs,
        ropeArgs};

    using UnifiedAdapter =
        Gemm::Device::DeviceGemm<Dsv41::UnifiedMlaPrologKernel>;
    if (UnifiedAdapter::CanImplement(unifiedArgs) != Status::kSuccess ||
        UnifiedAdapter::GetWorkspaceSize(unifiedArgs) > layout.qmmBytes) {
        return ACL_ERROR_INVALID_PARAM;
    }

    uint64_t hardwareSyncAddr = 0;
    if (aclrtGetHardwareSyncAddr(
            reinterpret_cast<void **>(&hardwareSyncAddr)) != ACL_SUCCESS) {
        hardwareSyncAddr = 0;
    }
    UnifiedAdapter unified;
    unified.Initialize(unifiedArgs, qmmWorkspace, stream);
    unified(stream, aicCount, hardwareSyncAddr);
    return ACL_SUCCESS;
}

}  // namespace vllm_ascend
