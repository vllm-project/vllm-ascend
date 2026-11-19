/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef DSV41_MLA_PROLOG_TORCH_ADPT_H_
#define DSV41_MLA_PROLOG_TORCH_ADPT_H_

#include <torch_npu/csrc/framework/OpCommand.h>

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <tuple>

#include "../../direct_kernels/dsv41_mla_prolog/dsv41_mla_prolog_direct_api.h"

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor> dsv41_mla_prolog_npu(
    const at::Tensor &x, const at::Tensor &wQa, const at::Tensor &gammaQa,
    const at::Tensor &wQb, const at::Tensor &wQbScale, const at::Tensor &wKv,
    const at::Tensor &gammaKv, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &slotMapping, at::Tensor &kvCache, double epsilon)
{
    constexpr int64_t kHidden = 5120;
    constexpr int64_t kQa = 1280;
    constexpr int64_t kKv = 512;
    constexpr int64_t kHeads = 64;
    constexpr int64_t kHeadDim = 512;
    constexpr int64_t kRopeDim = 64;
    constexpr int64_t kQbOut = kHeads * kHeadDim;
    TORCH_CHECK(
        x.device().is_privateuseone(),
        "DeepSeek-V4.1 MLA prolog tensors must be on NPU.");
    TORCH_CHECK(
        x.dim() == 2 && x.size(1) == kHidden && x.scalar_type() == at::kBFloat16,
        "x must be BF16 [T, 5120].");
    TORCH_CHECK(
        wQa.dim() == 2 && wQa.size(0) == kQa && wQa.size(1) == kHidden &&
            wQa.scalar_type() == at::kBFloat16,
        "wq_a must be BF16 [1280, 5120].");
    TORCH_CHECK(
        gammaQa.dim() == 1 && gammaQa.size(0) == kQa &&
            gammaQa.scalar_type() == at::kBFloat16,
        "gamma_q_a must be BF16 [1280].");
    TORCH_CHECK(
        wQb.dim() == 2 && wQb.size(0) == kQa && wQb.size(1) == kQbOut &&
            wQb.scalar_type() == at::kChar,
        "wq_b must be the NZ-cast INT8 [1280, 32768] weight.");
    TORCH_CHECK(
        wQbScale.numel() == kQbOut && wQbScale.scalar_type() == at::kBFloat16,
        "wq_b_scale must be BF16 with 32768 per-channel scales.");
    TORCH_CHECK(
        wKv.dim() == 2 && wKv.size(0) == kKv && wKv.size(1) == kHidden &&
            wKv.scalar_type() == at::kBFloat16,
        "wkv must be BF16 [512, 5120].");
    TORCH_CHECK(
        gammaKv.dim() == 1 && gammaKv.size(0) == kKv &&
            gammaKv.scalar_type() == at::kBFloat16,
        "gamma_kv must be BF16 [512].");
    TORCH_CHECK(
        cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat &&
            cos.numel() == x.size(0) * kRopeDim &&
            sin.numel() == x.size(0) * kRopeDim,
        "rope_cos/rope_sin must be FP32 with T * 64 values.");
    TORCH_CHECK(
        slotMapping.dim() == 1 && slotMapping.size(0) == x.size(0),
        "slot_mapping must be [T] flat indices.");
    const at::Tensor slots =
        slotMapping.scalar_type() == at::kLong
            ? slotMapping.to(at::kInt)
            : slotMapping;
    TORCH_CHECK(
        slots.scalar_type() == at::kInt,
        "slot_mapping must be INT32 or INT64.");
    TORCH_CHECK(
        kvCache.device() == x.device() &&
            (kvCache.dim() == 3 || (kvCache.dim() == 4 && kvCache.size(2) == 1)),
        "kv_cache must be [pages, 128, 512] or [pages, 128, 1, 512].");
    const bool blockDimOk =
        kvCache.dim() == 3
            ? (kvCache.size(1) == 128 && kvCache.size(2) == kHeadDim)
            : (kvCache.size(1) == 128 && kvCache.size(3) == kHeadDim);
    TORCH_CHECK(
        blockDimOk && kvCache.scalar_type() == at::kBFloat16,
        "kv_cache must be BF16 with 128 rows per page and head size 512.");
    TORCH_CHECK(
        kvCache.stride(-1) == 1 && kvCache.stride(1) >= kHeadDim,
        "kv_cache rows must be dense; page/block strides may contain gaps.");
    TORCH_CHECK(epsilon > 0.0, "epsilon must be positive.");

    const auto device = x.device();
    const at::Tensor tensors[] = {
        wQa, gammaQa, wQb, wQbScale, wKv, gammaKv, cos, sin, slots, kvCache};
    for (const auto &tensor : tensors) {
        TORCH_CHECK(
            tensor.device() == device,
            "all DeepSeek-V4.1 MLA prolog tensors must share one NPU.");
    }

    const int64_t numTokens = x.size(0);
    const uint64_t workspaceBytes =
        dsv41_mla_prolog_workspace_size(numTokens);
    TORCH_CHECK(
        workspaceBytes > 0,
        "Dsv41MlaProlog direct launcher is unavailable on this device.");
    at::Tensor workspace = at::empty(
        {static_cast<int64_t>(workspaceBytes)},
        x.options().dtype(at::kByte));
    at::Tensor qOutput =
        at::empty({numTokens, kHeads, kHeadDim}, x.options());
    at::Tensor qr = at::empty({numTokens, kQa}, x.options());

    Dsv41MlaPrologDirectArgs args{};
    args.x = reinterpret_cast<const uint16_t *>(x.data_ptr());
    args.wQa = reinterpret_cast<const uint16_t *>(wQa.data_ptr());
    args.gammaQa = reinterpret_cast<const uint16_t *>(gammaQa.data_ptr());
    args.wQb = reinterpret_cast<const int8_t *>(wQb.data_ptr());
    args.wQbScale = reinterpret_cast<const uint16_t *>(wQbScale.data_ptr());
    args.wKv = reinterpret_cast<const uint16_t *>(wKv.data_ptr());
    args.gammaKv = reinterpret_cast<const uint16_t *>(gammaKv.data_ptr());
    args.ropeCos = reinterpret_cast<const float *>(cos.data_ptr());
    args.ropeSin = reinterpret_cast<const float *>(sin.data_ptr());
    args.slots = reinterpret_cast<const int32_t *>(slots.data_ptr());
    args.kvCache = reinterpret_cast<uint16_t *>(kvCache.data_ptr());
    args.cacheStride0 = static_cast<uint64_t>(kvCache.stride(0));
    args.cacheStride1 = static_cast<uint64_t>(kvCache.stride(1));
    args.cacheBlockSize = static_cast<uint32_t>(kvCache.size(1));
    args.qOutput = reinterpret_cast<uint16_t *>(qOutput.data_ptr());
    args.qr = reinterpret_cast<uint16_t *>(qr.data_ptr());
    args.workspace = workspace.data_ptr();
    args.workspaceSize = workspaceBytes;
    args.t = static_cast<int32_t>(numTokens);
    args.epsilon = static_cast<float>(epsilon);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("dsv41_mla_prolog");
    cmd.SetCustomHandler([args, stream]() -> int {
        return dsv41_mla_prolog_direct(
            args, reinterpret_cast<void *>(stream));
    });
    cmd.Run();
    return std::tuple<at::Tensor, at::Tensor>(qOutput, qr);
}
}  // namespace vllm_ascend

#endif  // DSV41_MLA_PROLOG_TORCH_ADPT_H_
