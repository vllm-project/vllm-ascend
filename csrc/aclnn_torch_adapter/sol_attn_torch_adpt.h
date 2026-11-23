// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cmath>
#include <limits>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "op_api_common.h"

namespace vllm_ascend {
namespace sol_attn_detail {

constexpr int64_t kBlockSize = 64;
constexpr int64_t kGroupSize = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kQueryTile = 128;
constexpr int64_t kAlignment = 8;
constexpr int64_t kMaxBlocks = std::numeric_limits<uint16_t>::max() + 1;

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

struct Geometry {
    explicit Geometry(int64_t blocks)
        : blocks(blocks),
          groups(CeilDiv(blocks, kGroupSize)),
          paddedBlocks(groups * kGroupSize),
          queryTiles(CeilDiv(blocks, kQueryTile / kBlockSize)),
          paddedQueryTiles(CeilDiv(queryTiles, kAlignment) * kAlignment)
    {
        TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks, "Sol-Attn block count is out of range");
    }

    int64_t blocks;
    int64_t groups;
    int64_t paddedBlocks;
    int64_t queryTiles;
    int64_t paddedQueryTiles;
};

inline void CheckInput(const at::Tensor& tensor, int64_t rank, const char* name)
{
    TORCH_CHECK(tensor.device().is_privateuseone(), "Sol-Attn ", name, " must be on NPU");
    TORCH_CHECK(tensor.dim() == rank && tensor.numel() > 0, "Sol-Attn ", name, " has invalid dimensions");
    TORCH_CHECK(tensor.numel() <= std::numeric_limits<int32_t>::max(),
                "Sol-Attn ", name, " exceeds the kernel indexing range");
}

inline void CheckTensor(
    const at::Tensor& tensor, c10::Device device, at::IntArrayRef shape, at::ScalarType dtype, const char* name)
{
    TORCH_CHECK(tensor.device() == device, "Sol-Attn ", name, " must be on ", device);
    TORCH_CHECK(tensor.sizes() == shape && tensor.scalar_type() == dtype,
                "Sol-Attn ", name, " must have shape ", shape, " and dtype ", dtype);
    TORCH_CHECK(tensor.is_contiguous() && IsOpInputBaseFormat(tensor),
                "Sol-Attn ", name, " must be contiguous with base NPU format");
    TORCH_CHECK(tensor.numel() <= std::numeric_limits<int32_t>::max(),
                "Sol-Attn ", name, " exceeds the kernel indexing range");
}

inline void CheckScale(double scaleLog2)
{
    TORCH_CHECK(std::isfinite(static_cast<float>(scaleLog2)), "Sol-Attn scale_log2 must be finite");
}

}  // namespace sol_attn_detail

inline void sol_attn_prepare_qkv(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& kPad, const at::Tensor& vPad,
    const at::Tensor& kSummary, const at::Tensor& vSummary, const at::Tensor& kSummaryFp32,
    const at::Tensor& qSummary, const at::Tensor& tailLengths)
{
    using namespace sol_attn_detail;
    CheckInput(q, 4, "q");
    const auto device = q.device();
    const auto batch = q.size(0);
    const auto tokens = q.size(1);
    const auto heads = q.size(2);
    const Geometry geometry(CeilDiv(tokens, kBlockSize));
    const auto paddedTokens = geometry.blocks * kBlockSize;
    const auto tailLength = tokens - (geometry.blocks - 1) * kBlockSize;
    const double invTailLength = 1.0 / tailLength;
    CheckTensor(q, device, {batch, tokens, heads, kHeadDim}, at::kBFloat16, "q");
    CheckTensor(k, device, q.sizes(), at::kBFloat16, "k");
    CheckTensor(v, device, q.sizes(), at::kBFloat16, "v");
    CheckTensor(kPad, device, {batch, heads, paddedTokens, kHeadDim}, at::kBFloat16, "k_pad");
    CheckTensor(vPad, device, kPad.sizes(), at::kBFloat16, "v_pad");
    CheckTensor(kSummary, device, {batch, heads, geometry.paddedBlocks, kHeadDim}, at::kBFloat16, "k_summary");
    CheckTensor(vSummary, device, kSummary.sizes(), at::kBFloat16, "v_summary");
    CheckTensor(kSummaryFp32, device, kSummary.sizes(), at::kFloat, "k_summary_fp32");
    CheckTensor(qSummary, device, {batch, heads, geometry.blocks, kHeadDim}, at::kFloat, "q_summary");
    CheckTensor(tailLengths, device, {kGroupSize}, at::kFloat, "tail_lengths");

    const c10_npu::NPUGuard guard(device);
    EXEC_NPU_CMD(aclnnSolAttnPrepareQkvKernel, q, k, v, batch, tokens, heads, geometry.blocks, geometry.groups,
                 geometry.paddedBlocks, paddedTokens, invTailLength,
                 kPad, vPad, kSummary, vSummary, kSummaryFp32, qSummary, tailLengths);
}

inline void sol_attn_prepare_diag_thresh(
    const at::Tensor& qSummary, const at::Tensor& kSummaryFp32, double scaleLog2, double tau,
    const at::Tensor& threshold)
{
    using namespace sol_attn_detail;
    CheckInput(qSummary, 4, "q_summary");
    CheckScale(scaleLog2);
    TORCH_CHECK(std::isfinite(static_cast<float>(tau)), "Sol-Attn tau must be finite");
    const auto device = qSummary.device();
    const auto batch = qSummary.size(0);
    const auto heads = qSummary.size(1);
    const Geometry geometry(qSummary.size(2));
    const double invBlocks = 1.0 / geometry.blocks;
    CheckTensor(qSummary, device, {batch, heads, geometry.blocks, kHeadDim}, at::kFloat, "q_summary");
    CheckTensor(kSummaryFp32, device, {batch, heads, geometry.paddedBlocks, kHeadDim}, at::kFloat, "k_summary_fp32");
    CheckTensor(threshold, device, {batch, heads, geometry.blocks}, at::kFloat, "threshold");

    const c10_npu::NPUGuard guard(device);
    EXEC_NPU_CMD(aclnnSolAttnPrepareDiagThreshKernel, qSummary, kSummaryFp32, batch, heads, geometry.blocks,
                 geometry.groups, geometry.paddedBlocks, scaleLog2, tau, invBlocks, threshold);
}

inline void sol_attn_prepare_route_pack(
    const at::Tensor& rawScores, const at::Tensor& threshold, int64_t sinkStartBlock, int64_t sinkEndBlock,
    double scaleLog2, const at::Tensor& approxMask, const at::Tensor& exactCount, const at::Tensor& exactIndex)
{
    using namespace sol_attn_detail;
    CheckInput(rawScores, 3, "raw_scores");
    CheckScale(scaleLog2);
    const auto device = rawScores.device();
    const auto heads = rawScores.size(0);
    const Geometry geometry(rawScores.size(1));
    TORCH_CHECK(0 <= sinkStartBlock && sinkStartBlock <= sinkEndBlock && sinkEndBlock <= geometry.blocks,
                "Sol-Attn sink block range is out of bounds");
    CheckTensor(rawScores, device, {heads, geometry.blocks, geometry.paddedBlocks}, at::kFloat, "raw_scores");
    CheckTensor(threshold, device, {heads, geometry.blocks}, at::kFloat, "threshold");
    CheckTensor(approxMask, device, {heads, geometry.queryTiles, geometry.groups}, at::kUInt64, "approx_mask");
    CheckTensor(exactCount, device, {heads, geometry.paddedQueryTiles}, at::kInt, "exact_count");
    CheckTensor(exactIndex, device, {heads, geometry.queryTiles, geometry.paddedBlocks}, at::kUInt16, "exact_index");

    const c10_npu::NPUGuard guard(device);
    EXEC_NPU_CMD(aclnnSolAttnPrepareRoutePackKernel, rawScores, threshold, heads, geometry.blocks, geometry.queryTiles,
                 geometry.groups, geometry.paddedBlocks, geometry.paddedQueryTiles, kQueryTile, sinkStartBlock,
                 sinkEndBlock, scaleLog2, approxMask, exactCount, exactIndex);
}

inline void sol_attn_fwd_m128(
    const at::Tensor& q, const at::Tensor& kPad, const at::Tensor& vPad,
    const at::Tensor& kSummary, const at::Tensor& vSummary, const at::Tensor& exactCount,
    const at::Tensor& exactIndex, const at::Tensor& approxMask, const at::Tensor& tailLengths,
    double scaleLog2, const at::Tensor& output, const at::Tensor& lse)
{
    using namespace sol_attn_detail;
    CheckInput(q, 4, "q");
    CheckScale(scaleLog2);
    const auto device = q.device();
    const auto batch = q.size(0);
    const auto tokens = q.size(1);
    const auto heads = q.size(2);
    const Geometry geometry(CeilDiv(tokens, kBlockSize));
    const auto paddedTokens = geometry.blocks * kBlockSize;
    const auto paddedLse = CeilDiv(tokens, kAlignment) * kAlignment;
    CheckTensor(q, device, {batch, tokens, heads, kHeadDim}, at::kBFloat16, "q");
    CheckTensor(kPad, device, {batch, heads, paddedTokens, kHeadDim}, at::kBFloat16, "k_pad");
    CheckTensor(vPad, device, kPad.sizes(), at::kBFloat16, "v_pad");
    CheckTensor(kSummary, device, {batch, heads, geometry.paddedBlocks, kHeadDim}, at::kBFloat16, "k_summary");
    CheckTensor(vSummary, device, kSummary.sizes(), at::kBFloat16, "v_summary");
    CheckTensor(exactCount, device, {batch, heads, geometry.paddedQueryTiles}, at::kInt, "exact_count");
    CheckTensor(exactIndex, device, {batch, heads, geometry.queryTiles, geometry.paddedBlocks}, at::kUInt16, "exact_index");
    CheckTensor(approxMask, device, {batch, heads, geometry.queryTiles, geometry.groups}, at::kUInt64, "approx_mask");
    CheckTensor(tailLengths, device, {kGroupSize}, at::kFloat, "tail_lengths");
    CheckTensor(output, device, q.sizes(), at::kBFloat16, "output");
    CheckTensor(lse, device, {batch, heads, paddedLse}, at::kFloat, "lse");

    const c10_npu::NPUGuard guard(device);
    EXEC_NPU_CMD(aclnnSolAttnFwdM128Kernel, q, kPad, vPad, kSummary, vSummary, exactCount, exactIndex, approxMask,
                 tailLengths, batch, tokens, heads, geometry.blocks, geometry.queryTiles, geometry.groups,
                 geometry.paddedBlocks, paddedTokens, geometry.paddedQueryTiles, paddedLse, scaleLog2, output, lse);
}

}  // namespace vllm_ascend
