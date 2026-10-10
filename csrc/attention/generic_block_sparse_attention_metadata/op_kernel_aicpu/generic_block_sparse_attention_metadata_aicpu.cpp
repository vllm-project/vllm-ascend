/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "generic_block_sparse_attention_metadata_aicpu.h"

#include <limits>

namespace aicpu {
namespace {

constexpr uint32_t KERNEL_STATUS_OK = 0U;
constexpr uint32_t KERNEL_STATUS_PARAM_INVALID = 1U;
constexpr int32_t TND_SPARSE_BLOCK_IDX_DIM_NUM = 3;
constexpr int32_t TND_SPARSE_BLOCK_COUNT_DIM_NUM = 2;
constexpr int32_t BATCHED_SPARSE_BLOCK_IDX_DIM_NUM = 4;
constexpr int32_t BATCHED_SPARSE_BLOCK_COUNT_DIM_NUM = 3;
constexpr int32_t TND_BLOCK_CAPACITY_DIM_INDEX = 2;
constexpr int32_t BATCHED_Q_BLOCK_DIM_INDEX = 2;
constexpr int32_t BATCHED_BLOCK_CAPACITY_DIM_INDEX = 3;

bool HasTensorShape(const Tensor *tensor)
{
    return tensor != nullptr && tensor->GetTensorShape() != nullptr;
}

bool HasTensorData(const Tensor *tensor)
{
    return tensor != nullptr && tensor->GetData() != nullptr;
}

} // namespace

uint32_t GenericBlockSparseAttentionMetadataAicpu::Compute(CpuKernelContext &ctx)
{
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    std::vector<int64_t> qSeqLens;
    std::vector<int64_t> kvSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    std::vector<int64_t> validBlockNums;
    if (!ParseQSeqLens(qSeqLens, kvSeqLens, qStorageBlockStarts) ||
        !ParseValidBlockNums(qSeqLens, kvSeqLens, qStorageBlockStarts, validBlockNums) ||
        !GenerateMetadata(qSeqLens, validBlockNums)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

bool GenericBlockSparseAttentionMetadataAicpu::Prepare(CpuKernelContext &ctx)
{
    sparseBlockIdx_ = ctx.Input(static_cast<uint32_t>(ParamId::SPARSE_BLOCK_IDX));
    sparseBlockCount_ = ctx.Input(static_cast<uint32_t>(ParamId::SPARSE_BLOCK_COUNT));
    cuSeqLengths_ = ctx.Input(static_cast<uint32_t>(ParamId::CU_SEQ_LENGTHS));
    seqUsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::SEQ_USED_Q));
    cuSeqLengthsKv_ = ctx.Input(static_cast<uint32_t>(ParamId::CU_SEQ_LENGTHS_KV));
    seqUsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::SEQ_USED_KV));
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::METADATA));
    // An omitted optional input is represented by an empty Tensor in the AICPU context.
    if (!HasTensorData(cuSeqLengths_)) {
        cuSeqLengths_ = nullptr;
    }
    if (!HasTensorData(seqUsedQ_)) {
        seqUsedQ_ = nullptr;
    }

    if (!HasTensorData(cuSeqLengthsKv_)) {
        cuSeqLengthsKv_ = nullptr;
    }
    if (!HasTensorData(seqUsedKv_)) {
        seqUsedKv_ = nullptr;
    }

    const bool attrsValid =
        GetAttrValue(ctx, "max_q_seq_len", maxQSeqLen_) && GetAttrValue(ctx, "num_q_heads", numQHeads_) &&
        GetAttrValue(ctx, "num_kv_heads", numKvHeads_) && GetAttrValue(ctx, "head_dim", headDim_) &&
        GetAttrValue(ctx, "block_shape_x", blockShapeX_) && GetAttrValue(ctx, "block_shape_y", blockShapeY_) &&
        GetAttrValue(ctx, "is_packed_gqa", isPackedGQA_) && GetAttrValue(ctx, "q_input_layout", qInputLayout_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
        GetAttrValue(ctx, "residual_block_mode", residualBlockMode_) &&
        GetAttrValue(ctx, "is_consistent_topk", isConsistentTopk_);
    return attrsValid && CheckInputs() && CheckKvSeqLens();
}

bool GenericBlockSparseAttentionMetadataAicpu::CheckOptionalTensor(const Tensor *tensor, DataType dataType,
                                                                   int64_t elementNum, const char *tensorName) const
{
    if (tensor == nullptr) {
        return true;
    }
    if (!HasTensorShape(tensor) || !HasTensorData(tensor) || tensor->GetDataType() != dataType ||
        tensor->GetTensorShape()->GetDims() != 1 || tensor->GetTensorShape()->GetDimSize(0) != elementNum) {
        KERNEL_LOG_ERROR("%s has invalid dtype or shape.", tensorName);
        return false;
    }
    return true;
}

bool GenericBlockSparseAttentionMetadataAicpu::CheckInputs()
{
    using namespace optiling::generic_block_sparse_attention_metadata;
    // Keep memory-safety checks at the AICPU boundary because the kernel can be reused independently of ACLNN.
    if (!HasTensorShape(sparseBlockIdx_) || !HasTensorShape(sparseBlockCount_) || !HasTensorData(sparseBlockCount_) ||
        !HasTensorData(metadata_)) {
        KERNEL_LOG_ERROR("sparse block tensors and metadata must be valid.");
        return false;
    }
    if (sparseBlockIdx_->GetDataType() != DT_INT32 || sparseBlockCount_->GetDataType() != DT_INT32 ||
        metadata_->GetDataType() != DT_INT32 || metadata_->GetTensorShape() == nullptr ||
        metadata_->GetTensorShape()->GetDims() != 1 ||
        metadata_->GetTensorShape()->GetDimSize(0) < METADATA_TOTAL_SIZE) {
        KERNEL_LOG_ERROR("sparse block tensors and metadata must be INT32 with valid shapes.");
        return false;
    }
    if (qInputLayout_ != "TND" && qInputLayout_ != "BSND" && qInputLayout_ != "BNSD") {
        KERNEL_LOG_ERROR("q_input_layout only supports TND, BSND or BNSD.");
        return false;
    }
    if ((qInputLayout_ != "TND" && maxQSeqLen_ <= 0) || numQHeads_ <= 0 || numKvHeads_ <= 0 || headDim_ <= 0 ||
        blockShapeX_ != 1 || blockShapeY_ <= 0 || blockShapeY_ > 256 ||
        (isPackedGQA_ != 0 && isPackedGQA_ != 1) || aicCoreNum_ <= 0 ||
        aicCoreNum_ > MAX_AIC_CORE_NUM || (residualBlockMode_ != 0 && residualBlockMode_ != 1) ||
        (isConsistentTopk_ != 0 && isConsistentTopk_ != 1)) {
        KERNEL_LOG_ERROR("Invalid scheduling attrs for GenericBlockSparseAttentionMetadata.");
        return false;
    }

    sparseHeadNum_ = isPackedGQA_ == 1 ? numKvHeads_ : numQHeads_;
    const auto idxShape = sparseBlockIdx_->GetTensorShape();
    const auto countShape = sparseBlockCount_->GetTensorShape();
    if (qInputLayout_ == "TND") {
        if (idxShape->GetDims() != TND_SPARSE_BLOCK_IDX_DIM_NUM ||
            countShape->GetDims() != TND_SPARSE_BLOCK_COUNT_DIM_NUM || idxShape->GetDimSize(0) != sparseHeadNum_ ||
            countShape->GetDimSize(0) != sparseHeadNum_ || countShape->GetDimSize(1) != idxShape->GetDimSize(1)) {
            KERNEL_LOG_ERROR("TND sparse block tensor shapes do not match attrs.");
            return false;
        }
        qBlockStorageNum_ = idxShape->GetDimSize(1);
        blockIndexStride_ = idxShape->GetDimSize(TND_BLOCK_CAPACITY_DIM_INDEX);
        if (!HasTensorShape(cuSeqLengths_) || cuSeqLengths_->GetTensorShape()->GetDims() != 1) {
            KERNEL_LOG_ERROR("cu_seq_lengths is required for TND query.");
            return false;
        }
        batchSize_ = cuSeqLengths_->GetTensorShape()->GetDimSize(0) - 1;
    } else {
        if (idxShape->GetDims() != BATCHED_SPARSE_BLOCK_IDX_DIM_NUM ||
            countShape->GetDims() != BATCHED_SPARSE_BLOCK_COUNT_DIM_NUM) {
            KERNEL_LOG_ERROR("BSND/BNSD sparse block tensors must be 4D and 3D, but got %dD and %dD.",
                             idxShape->GetDims(), countShape->GetDims());
            return false;
        }
        if (idxShape->GetDimSize(0) != countShape->GetDimSize(0)) {
            KERNEL_LOG_ERROR("sparse block batch dims differ: idx=%lld, count=%lld.", idxShape->GetDimSize(0),
                             countShape->GetDimSize(0));
            return false;
        }
        if (idxShape->GetDimSize(1) != sparseHeadNum_ || countShape->GetDimSize(1) != sparseHeadNum_) {
            KERNEL_LOG_ERROR("sparse block head dims must be %lld, but got idx=%lld, count=%lld.", sparseHeadNum_,
                             idxShape->GetDimSize(1), countShape->GetDimSize(1));
            return false;
        }
        if (idxShape->GetDimSize(BATCHED_Q_BLOCK_DIM_INDEX) != countShape->GetDimSize(BATCHED_Q_BLOCK_DIM_INDEX)) {
            KERNEL_LOG_ERROR("sparse block Q dims differ: idx=%lld, count=%lld.",
                             idxShape->GetDimSize(BATCHED_Q_BLOCK_DIM_INDEX),
                             countShape->GetDimSize(BATCHED_Q_BLOCK_DIM_INDEX));
            return false;
        }
        if (idxShape->GetDimSize(BATCHED_Q_BLOCK_DIM_INDEX) != maxQSeqLen_) {
            KERNEL_LOG_ERROR("sparse block Q dim must equal max_q_seq_len: Q dim=%lld, max_q_seq_len=%lld.",
                             idxShape->GetDimSize(BATCHED_Q_BLOCK_DIM_INDEX), maxQSeqLen_);
            return false;
        }
        batchSize_ = idxShape->GetDimSize(0);
        qBlockStorageNum_ = idxShape->GetDimSize(BATCHED_Q_BLOCK_DIM_INDEX);
        blockIndexStride_ = idxShape->GetDimSize(BATCHED_BLOCK_CAPACITY_DIM_INDEX);
    }
    if (batchSize_ < 0 || qBlockStorageNum_ < 0 || blockIndexStride_ <= 0) {
        KERNEL_LOG_ERROR("Invalid batch, Q block or sparse block capacity.");
        return false;
    }
    if (!CheckOptionalTensor(cuSeqLengths_, DT_INT64, batchSize_ + 1, "cu_seq_lengths") ||
        !CheckOptionalTensor(seqUsedQ_, DT_INT32, batchSize_, "seq_used_q")) {
        return false;
    }
    return true;
}

bool GenericBlockSparseAttentionMetadataAicpu::CheckKvSeqLens() const
{
    if (!CheckOptionalTensor(cuSeqLengthsKv_, DT_INT64, batchSize_ + 1, "cu_seq_lengths_kv") ||
        !CheckOptionalTensor(seqUsedKv_, DT_INT32, batchSize_, "seq_used_kv")) {
        return false;
    }
    // The public API enforces the layout-specific presence rules. Validate the
    // prefix origin here; per-batch values are checked during the combined parse.
    const auto *cu = cuSeqLengthsKv_ == nullptr ? nullptr :
        static_cast<const int64_t *>(cuSeqLengthsKv_->GetData());
    if (cu != nullptr && cu[0] != 0) {
        KERNEL_LOG_ERROR("cu_seq_lengths_kv must start at zero.");
        return false;
    }
    // Per-batch values are validated while Q and KV lengths are parsed together.
    // The main kernel checks cu[B] against the actual KV tensor capacity.
    return true;
}

bool GenericBlockSparseAttentionMetadataAicpu::ParseQSeqLens(std::vector<int64_t> &qSeqLens,
                                                               std::vector<int64_t> &kvSeqLens,
                                                               std::vector<int64_t> &qStorageBlockStarts) const
{
    const auto *kvPrefix = cuSeqLengthsKv_ == nullptr ? nullptr :
        static_cast<const int64_t *>(cuSeqLengthsKv_->GetData());
    const auto *kvUsed = seqUsedKv_ == nullptr ? nullptr :
        static_cast<const int32_t *>(seqUsedKv_->GetData());
    if (kvPrefix == nullptr && kvUsed == nullptr) {
        KERNEL_LOG_ERROR("Either cu_seq_lengths_kv or seq_used_kv is required.");
        return false;
    }
    kvSeqLens.resize(static_cast<size_t>(batchSize_));
    const auto readKvSeqLen = [&](int64_t batchIdx, int64_t &kvSeqLen) -> bool {
        if (kvPrefix != nullptr && (kvPrefix[batchIdx] < 0 || kvPrefix[batchIdx + 1] < kvPrefix[batchIdx])) {
            KERNEL_LOG_ERROR("cu_seq_lengths_kv must be nonnegative and nondecreasing.");
            return false;
        }
        const int64_t storageKvSeqLen = kvPrefix == nullptr ? 0 : kvPrefix[batchIdx + 1] - kvPrefix[batchIdx];
        if (kvUsed != nullptr && (kvUsed[batchIdx] < 0 ||
                                  (kvPrefix != nullptr && kvUsed[batchIdx] > storageKvSeqLen))) {
            KERNEL_LOG_ERROR("seq_used_kv must fit the corresponding storage segment.");
            return false;
        }
        kvSeqLen = kvUsed == nullptr ? storageKvSeqLen : kvUsed[batchIdx];
        return true;
    };

    if (qInputLayout_ == "TND") {
        const auto *qPrefix = static_cast<const int64_t *>(cuSeqLengths_->GetData());
        if (qPrefix[0] != 0) {
            KERNEL_LOG_ERROR("cu_seq_lengths[0] must be 0 for TND Q.");
            return false;
        }
        const auto *qUsed = seqUsedQ_ == nullptr ? nullptr :
            static_cast<const int32_t *>(seqUsedQ_->GetData());
        qSeqLens.resize(static_cast<size_t>(batchSize_));
        qStorageBlockStarts.resize(static_cast<size_t>(batchSize_));
        int64_t storageBlockBase = 0;
        for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
            const int64_t storageQSeqLen = qPrefix[batchIdx + 1] - qPrefix[batchIdx];
            const int64_t actualQSeqLen = qUsed == nullptr ? storageQSeqLen : qUsed[batchIdx];
            if (qPrefix[batchIdx] < 0 || storageQSeqLen < 0 || actualQSeqLen < 0 ||
                actualQSeqLen > storageQSeqLen) {
                KERNEL_LOG_ERROR("Invalid TND Q sequence length at batch %lld.", batchIdx);
                return false;
            }
            qSeqLens[static_cast<size_t>(batchIdx)] = actualQSeqLen;
            qStorageBlockStarts[static_cast<size_t>(batchIdx)] = storageBlockBase;
            const int64_t storageBlockNum = storageQSeqLen == 0 ? 0 :
                (storageQSeqLen - 1) / blockShapeX_ + 1;
            if (storageBlockBase > std::numeric_limits<int64_t>::max() - storageBlockNum) {
                return false;
            }
            storageBlockBase += storageBlockNum;
            if (!readKvSeqLen(batchIdx, kvSeqLens[static_cast<size_t>(batchIdx)])) {
                return false;
            }
        }
        return storageBlockBase == qBlockStorageNum_;
    }

    qSeqLens.assign(static_cast<size_t>(batchSize_), maxQSeqLen_);
    qStorageBlockStarts.clear();
    if (seqUsedQ_ != nullptr) {
        const auto *qUsed = static_cast<const int32_t *>(seqUsedQ_->GetData());
        for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
            const int64_t qSeqLen = qUsed[batchIdx];
            if (qSeqLen < 0 || qSeqLen > maxQSeqLen_) {
                KERNEL_LOG_ERROR("seq_used_q[%lld]=%lld is outside [0, %lld].", batchIdx, qSeqLen, maxQSeqLen_);
                return false;
            }
            qSeqLens[static_cast<size_t>(batchIdx)] = qSeqLen;
            if (!readKvSeqLen(batchIdx, kvSeqLens[static_cast<size_t>(batchIdx)])) {
                return false;
            }
        }
        return true;
    }

    if (cuSeqLengths_ == nullptr) {
        for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
            if (!readKvSeqLen(batchIdx, kvSeqLens[static_cast<size_t>(batchIdx)])) {
                return false;
            }
        }
        return true;
    }
    const auto *qPrefix = static_cast<const int64_t *>(cuSeqLengths_->GetData());
    if (qPrefix[0] != 0) {
        KERNEL_LOG_ERROR("cu_seq_lengths[0] must be 0, but got %lld.", qPrefix[0]);
        return false;
    }
    for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
        if (qPrefix[batchIdx + 1] < qPrefix[batchIdx]) {
            KERNEL_LOG_ERROR("cu_seq_lengths must be nondecreasing at batch %lld.", batchIdx);
            return false;
        }
        const int64_t qSeqLen = qPrefix[batchIdx + 1] - qPrefix[batchIdx];
        if (qSeqLen > maxQSeqLen_) {
            KERNEL_LOG_ERROR("Q sequence length %lld exceeds max_q_seq_len %lld at batch %lld.", qSeqLen,
                             maxQSeqLen_, batchIdx);
            return false;
        }
        qSeqLens[static_cast<size_t>(batchIdx)] = qSeqLen;
        if (!readKvSeqLen(batchIdx, kvSeqLens[static_cast<size_t>(batchIdx)])) {
            return false;
        }
    }
    return true;
}

bool GenericBlockSparseAttentionMetadataAicpu::ParseValidBlockNums(const std::vector<int64_t> &qSeqLens,
                                                                   const std::vector<int64_t> &kvSeqLens,
                                                                   const std::vector<int64_t> &qStorageBlockStarts,
                                                                   std::vector<int64_t> &validBlockNums) const
{
    const auto *counts = static_cast<const int32_t *>(sparseBlockCount_->GetData());
    validBlockNums.clear();
    for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
        const int64_t qSeqLen = qSeqLens[static_cast<size_t>(batchIdx)];
        const int64_t kvSeqLen = kvSeqLens[static_cast<size_t>(batchIdx)];
        const int64_t qBlockNum = generic_block_sparse_attention_metadata::CeilDivQSeq(qSeqLen, blockShapeX_);
        for (int64_t qBlock = 0; qBlock < qBlockNum; ++qBlock) {
            const int64_t qToken = qBlock * blockShapeX_;
            const int64_t visibleKvEnd = kvSeqLen - qSeqLen + qToken + 1;
            const int64_t residual = residualBlockMode_ == 1 && visibleKvEnd > 0 &&
                visibleKvEnd % blockShapeY_ != 0 ? 1 : 0;
            for (int64_t sparseHead = 0; sparseHead < sparseHeadNum_; ++sparseHead) {
                const int64_t offset = qInputLayout_ == "TND" ?
                    sparseHead * qBlockStorageNum_ + qStorageBlockStarts[static_cast<size_t>(batchIdx)] + qBlock :
                    (batchIdx * sparseHeadNum_ + sparseHead) * qBlockStorageNum_ + qBlock;
                const int64_t count = counts[offset];
                if (count < 0 || count > blockIndexStride_) {
                    KERNEL_LOG_ERROR("sparse_block_count[%lld]=%lld exceeds capacity %lld.", offset, count,
                                     blockIndexStride_);
                    return false;
                }
                validBlockNums.push_back(count + residual);
            }
        }
    }
    return true;
}

bool GenericBlockSparseAttentionMetadataAicpu::GenerateMetadata(const std::vector<int64_t> &qSeqLens,
                                                                const std::vector<int64_t> &validBlockNums) const
{
    generic_block_sparse_attention_metadata::ScheduleInput input;
    input.batchSize = batchSize_;
    input.numQHeads = numQHeads_;
    input.numKvHeads = numKvHeads_;
    input.maxQSeqLen = maxQSeqLen_;
    input.headDim = headDim_;
    input.blockShapeX = blockShapeX_;
    input.blockShapeY = blockShapeY_;
    input.blockIndexStride = blockIndexStride_;
    input.qBlockStorageNum = qBlockStorageNum_;
    input.isPackedGQA = isPackedGQA_;
    input.residualBlockMode = residualBlockMode_;
    input.isConsistentTopk = isConsistentTopk_;
    input.aicCoreNum = aicCoreNum_;
    input.qSeqLens = qSeqLens;
    input.validBlockNums = validBlockNums;

    generic_block_sparse_attention_metadata::ScheduleResult result;
    const auto scheduleStatus = generic_block_sparse_attention_metadata::BuildSchedule(input, result);
    if (scheduleStatus != generic_block_sparse_attention_metadata::BSAScheduleStatus::BSA_SUCCESS) {
        KERNEL_LOG_ERROR("Failed to build metadata schedule, status=%u.", static_cast<uint32_t>(scheduleStatus));
        return false;
    }
    auto *metadata =
        static_cast<optiling::generic_block_sparse_attention_metadata::MetadataType *>(metadata_->GetData());
    const auto encodeStatus = generic_block_sparse_attention_metadata::EncodeMetadata(
        result, metadata, optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE);
    if (encodeStatus != generic_block_sparse_attention_metadata::BSAScheduleStatus::BSA_SUCCESS) {
        KERNEL_LOG_ERROR("Failed to encode metadata, status=%u.", static_cast<uint32_t>(encodeStatus));
        return false;
    }
    return true;
}

namespace {
const char *KERNEL_TYPE = "GenericBlockSparseAttentionMetadata";
REGISTER_CPU_KERNEL(KERNEL_TYPE, GenericBlockSparseAttentionMetadataAicpu);
} // namespace

} // namespace aicpu
